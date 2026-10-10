#!/usr/bin/env python3
"""Join exact390 x 8 objective frontend rows with frozen CAP/role outcomes.

This is the first clinical-outcome layer of W4.2. It is fail-closed: every
frontend cell must have exactly one valid GPT-5.4 adjudication for every frozen
recording. No missing judgment is imputed and no objective proxy is converted
into a clinical label.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


CELLS = (
    "largev3_pyannote", "largev3_nemo", "turbo_pyannote", "turbo_nemo",
    "small_pyannote", "small_nemo", "canary_pyannote", "canary_nemo",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_index(rows: list[dict[str, Any]], keys: tuple[str, ...], *, label: str) -> dict[tuple[str, ...], dict[str, Any]]:
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in keys)
        if any(not value for value in key):
            raise ValueError(f"{label}: blank key {key}")
        if key in result:
            raise ValueError(f"{label}: duplicate key {key}")
        result[key] = row
    return result


def clinical_fields(item: dict[str, Any]) -> dict[str, Any]:
    metrics = item.get("metrics")
    result = item.get("result")
    if not isinstance(metrics, dict) or not isinstance(result, dict):
        raise ValueError("CAP item lacks metrics/result")
    assessment = str(metrics.get("conversation_assessment") or "")
    if assessment not in {
        "no_major_error", "major_omission", "major_contradiction",
        "major_unsupported_addition", "uncertain",
    }:
        raise ValueError(f"invalid conversation assessment {assessment!r}")
    if str(result.get("conversation_assessment") or "") != assessment:
        raise ValueError("CAP result/metric conversation assessment mismatch")
    required_counts = (
        "num_caps", "preserved", "partially_preserved", "missing", "contradicted",
        "uncertain", "num_major_caps", "num_major_cap_errors",
        "num_unsupported_assertions", "num_major_unsupported_assertions",
        "role_change_count",
    )
    for field in required_counts:
        if not isinstance(metrics.get(field), int) or metrics[field] < 0:
            raise ValueError(f"invalid CAP metric {field}")
    return {
        "cap_count": metrics["num_caps"],
        "major_cap_count": metrics["num_major_caps"],
        "cap_strict_preservation_rate": metrics.get("strict_preservation_rate"),
        "cap_lenient_preservation_rate": metrics.get("lenient_preservation_rate"),
        "cap_missing_count": metrics["missing"],
        "cap_contradicted_count": metrics["contradicted"],
        "cap_uncertain_count": metrics["uncertain"],
        "major_cap_error_count": metrics["num_major_cap_errors"],
        "any_critical_major_cap_error": int(metrics["num_major_cap_errors"] > 0),
        "role_change_count": metrics["role_change_count"],
        "any_role_attribution_error": int(metrics["role_change_count"] > 0),
        "unsupported_target_assertion_count": metrics["num_unsupported_assertions"],
        "major_unsupported_target_assertion_count": metrics["num_major_unsupported_assertions"],
        "any_major_unsupported_target_assertion": int(metrics["num_major_unsupported_assertions"] > 0),
        "conversation_assessment": assessment,
        "conversation_no_major_error": int(assessment == "no_major_error"),
        "conversation_major_error": int(assessment in {
            "major_omission", "major_contradiction", "major_unsupported_addition",
        }),
        "conversation_uncertain": int(assessment == "uncertain"),
    }


def build(*, objective_path: Path, cap_root: Path, expected_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    objective_rows = read_jsonl(objective_path)
    objective = unique_index(objective_rows, ("cell", "audio_id"), label="objective")
    # Derive the frozen audio universe independently of row ordering.
    audio_ids = {str(row["audio_id"]) for row in objective_rows}
    if len(audio_ids) != expected_rows:
        raise ValueError(f"expected {expected_rows} audio IDs, got {len(audio_ids)}")
    expected_keys = {(cell, audio_id) for cell in CELLS for audio_id in audio_ids}
    if set(objective) != expected_keys:
        raise ValueError("objective registry is not exact expected_rows x 8")

    joined: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    per_cell: dict[str, Any] = {}
    for cell in CELLS:
        path = cap_root / cell / "cap_target_adjudications.jsonl"
        summary_path = cap_root / cell / "summary.json"
        if not path.is_file() or not summary_path.is_file():
            raise ValueError(f"{cell}: missing complete CAP artifacts")
        summary = json.loads(summary_path.read_text())
        if summary.get("num_completed") != expected_rows or summary.get("num_errors") != 0:
            raise ValueError(f"{cell}: CAP summary is incomplete")
        cap = unique_index(read_jsonl(path), ("audio_id",), label=f"CAP/{cell}")
        if set(key[0] for key in cap) != audio_ids:
            raise ValueError(f"{cell}: CAP population mismatch")
        counts: Counter[str] = Counter()
        for audio_id in sorted(audio_ids):
            fields = clinical_fields(cap[(audio_id,)])
            row = dict(objective[(cell, audio_id)])
            row.update(fields)
            joined.append(row)
            counts["major_cap_error"] += fields["any_critical_major_cap_error"]
            counts["role_error"] += fields["any_role_attribution_error"]
            counts["conversation_major_error"] += fields["conversation_major_error"]
            counts["conversation_uncertain"] += fields["conversation_uncertain"]
        hashes[cell] = sha256(path)
        per_cell[cell] = dict(counts)

    if len(joined) != expected_rows * len(CELLS):
        raise ValueError("joined registry row count mismatch")
    report = {
        "schema_version": "aegisvox.frontend_cap_role_registry.v1",
        "status": "complete",
        "num_cells": len(CELLS),
        "num_rows_per_cell": expected_rows,
        "num_rows": len(joined),
        "primary_denominator": "intention_to_transform recording population",
        "objective_registry_sha256": sha256(objective_path),
        "cap_adjudication_sha256_by_cell": hashes,
        "event_counts_by_cell": per_cell,
        "deterministic_clinical_proxy_used": False,
        "claim_boundary": (
            "Source-CAP-anchored GPT-5.4 full-context adjudication joined to measured frontend fields; "
            "not human ground truth and not yet the complete W4.2 outcome registry."
        ),
    }
    return joined, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective-registry", type=Path, required=True)
    parser.add_argument("--cap-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=390)
    args = parser.parse_args()
    rows, summary = build(
        objective_path=args.objective_registry, cap_root=args.cap_root,
        expected_rows=args.expected_rows,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["output_jsonl_sha256"] = sha256(args.output_jsonl)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "num_rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
