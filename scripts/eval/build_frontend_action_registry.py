#!/usr/bin/env python3
"""Join exact390 x 8 frontend rows with source-anchored action outcomes.

The output is an intention-to-transform, recording-level registry.  Empty
source-action inventories remain in the denominator and are represented as
not-applicable rather than being silently dropped.
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
PLAN = {
    "invariant", "minor_change", "major_change", "potentially_harmful_change",
    "insufficient_evidence", "not_applicable",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index(rows: list[dict[str, Any]], keys: tuple[str, ...], label: str) -> dict[tuple[str, ...], dict[str, Any]]:
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in keys)
        if any(not value for value in key) or key in out:
            raise ValueError(f"{label}: blank or duplicate key {key}")
        out[key] = row
    return out


def outcomes(item: dict[str, Any]) -> dict[str, Any]:
    result = item.get("result")
    if not isinstance(result, dict):
        raise ValueError("action row lacks result")
    actions = result.get("adjudications")
    unsupported = result.get("unsupported_target_actions")
    assessment = str(result.get("conversation_plan_assessment") or "")
    if not isinstance(actions, list) or not isinstance(unsupported, list) or assessment not in PLAN:
        raise ValueError("invalid action adjudication result")
    statuses = [str(row.get("status") or "") for row in actions]
    state = [str(row.get("state_status") or "") for row in actions]
    role = [str(row.get("role_status") or "") for row in actions]
    unsafe = [str(row.get("unsafe_change") or "") for row in actions]
    impacts = [str(row.get("clinical_impact") or "") for row in actions]
    return {
        "source_action_count": len(actions),
        "strict_action_preservation_rate": (
            sum(value == "preserved" for value in statuses) / len(statuses) if statuses else None
        ),
        "any_action_not_strictly_preserved": int(any(value != "preserved" for value in statuses)),
        "any_action_state_error": int(any(value in {"changed", "missing"} for value in state)),
        "any_action_role_error": int(any(value == "changed" for value in role)),
        "any_unsafe_action_change": int(any(value not in {"none", "uncertain"} for value in unsafe)),
        "any_potentially_harmful_action_change": int(any(value == "potentially_harmful" for value in impacts)),
        "unsupported_target_action_count": len(unsupported),
        "any_unsupported_target_action": int(bool(unsupported)),
        "conversation_plan_assessment": assessment,
        "conversation_major_or_harmful_change": int(assessment in {"major_change", "potentially_harmful_change"}),
        "conversation_plan_uncertain": int(assessment == "insufficient_evidence"),
        "conversation_plan_not_applicable": int(assessment == "not_applicable"),
    }


def build(objective_path: Path, action_root: Path, expected_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    objective_rows = read_jsonl(objective_path)
    objective = index(objective_rows, ("cell", "audio_id"), "objective")
    audio_ids = {str(row.get("audio_id") or "") for row in objective_rows}
    expected = {(cell, audio_id) for cell in CELLS for audio_id in audio_ids}
    if len(audio_ids) != expected_rows or set(objective) != expected:
        raise ValueError("objective registry is not exact expected_rows x 8")

    joined: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for cell in CELLS:
        rows_path = action_root / cell / "target_actions.jsonl"
        summary_path = action_root / cell / "summary.json"
        if not rows_path.is_file() or not summary_path.is_file():
            raise ValueError(f"{cell}: missing action artifacts")
        summary = json.loads(summary_path.read_text())
        if summary.get("schema_version") != "aegisvox.target_clinical_actions.v2":
            raise ValueError(f"{cell}: unexpected action schema")
        if summary.get("num_completed") != expected_rows or summary.get("num_errors") != 0:
            raise ValueError(f"{cell}: incomplete action summary")
        action = index(read_jsonl(rows_path), ("audio_id",), f"action/{cell}")
        if {key[0] for key in action} != audio_ids:
            raise ValueError(f"{cell}: action population mismatch")
        event_counts: Counter[str] = Counter()
        for audio_id in sorted(audio_ids):
            fields = outcomes(action[(audio_id,)])
            row = dict(objective[(cell, audio_id)])
            row.update(fields)
            joined.append(row)
            for field, value in fields.items():
                if field.startswith("any_") or field in {"conversation_major_or_harmful_change", "conversation_plan_uncertain"}:
                    event_counts[field] += int(value)
        hashes[cell] = sha256(rows_path)
        counts[cell] = dict(sorted(event_counts.items()))

    report = {
        "schema_version": "aegisvox.frontend_action_registry.v1",
        "status": "complete",
        "num_cells": len(CELLS),
        "num_rows_per_cell": expected_rows,
        "num_rows": len(joined),
        "primary_denominator": "intention-to-transform recording population, including no-action conversations",
        "objective_registry_sha256": sha256(objective_path),
        "action_rows_sha256_by_cell": hashes,
        "event_counts_by_cell": counts,
        "deterministic_clinical_proxy_used": False,
        "claim_boundary": "Source-action-anchored GPT-5.4 adjudication; not human ground truth or clinical efficacy.",
    }
    return joined, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective-registry", type=Path, required=True)
    ap.add_argument("--action-root", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--expected-rows", type=int, default=390)
    args = ap.parse_args()
    rows, summary = build(args.objective_registry, args.action_root, args.expected_rows)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["output_jsonl_sha256"] = sha256(args.output_jsonl)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "num_rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
