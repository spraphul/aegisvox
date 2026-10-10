#!/usr/bin/env python3
"""Bind one condition's fresh target-ASR evidence to the frozen 601-edit plan.

Condition-level evaluator CSVs may contain additional mask-only detections that
were never part of the frozen synthetic-relexicalization population. They are
retained in an explicit exclusion artifact, never silently promoted into or
used to redefine the 601-item denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical-rows", type=Path, required=True)
    ap.add_argument("--paired-transcripts", type=Path, required=True)
    ap.add_argument("--phi-edits-csv", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--excluded-output-jsonl", type=Path)
    ap.add_argument("--expected-edits", type=int, default=601)
    args = ap.parse_args()

    canonical = read_jsonl(args.canonical_rows)
    refs = {str(row["item_id"]): row for row in canonical}
    if len(refs) != len(canonical):
        raise SystemExit("duplicate item_id in frozen canonical rows")
    pairs: dict[str, dict[str, dict]] = {}
    for row in read_jsonl(args.paired_transcripts):
        pairs.setdefault(str(row["run_id"]), {})[str(row["source"])] = row
    with args.phi_edits_csv.open(newline="") as handle:
        audits = list(csv.DictReader(handle))
    audit_by_id = {f"{row['run_id']}::{row['edit_id']}": row for row in audits}
    if len(audit_by_id) != len(audits):
        raise SystemExit("duplicate run_id/edit_id in condition PHI audit")

    expected_ids = set(refs)
    if len(canonical) != args.expected_edits or not expected_ids.issubset(audit_by_id):
        missing = sorted(expected_ids - set(audit_by_id))[:5]
        extra = sorted(set(audit_by_id) - expected_ids)[:5]
        raise SystemExit(
            f"frozen edit mismatch canonical={len(canonical)} audit={len(audit_by_id)} "
            f"expected={args.expected_edits} missing={missing} extra={extra}"
        )
    extra_ids = sorted(set(audit_by_id) - expected_ids)
    excluded_rows = [
        {
            "item_id": item_id,
            "condition": args.condition,
            "reason": "outside_frozen_601_synthetic_relexicalization_population",
            "audit_row": audit_by_id[item_id],
        }
        for item_id in extra_ids
    ]

    output = []
    for source in canonical:
        row = dict(source)
        item_id = str(row["item_id"])
        audit = audit_by_id[item_id]
        pair = pairs.get(str(row["run_id"]), {})
        reference, hypothesis = pair.get("reference"), pair.get("hypothesis")
        if reference is None or hypothesis is None:
            raise SystemExit(f"missing matched ASR pair for {row['run_id']}")
        row["system"] = args.condition
        row["score_status"] = "ok"
        row["asr"] = [
            {
                "model": hypothesis.get("hypothesis_backend_version") or "faster-whisper-large-v3/final-local-window",
                "language": hypothesis.get("source_language") or row.get("language"),
                "text": audit.get("local_asr_text") or "",
            },
            {
                "model": hypothesis.get("hypothesis_backend_version") or "faster-whisper-large-v3/final-full-conversation",
                "language": hypothesis.get("source_language") or row.get("language"),
                "text": hypothesis.get("text") or "",
            },
        ]
        row["evidence_policy"] = "fresh_condition_target_asr_bound_to_frozen_edit_plan_no_deterministic_matching"
        output.append(row)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output))
    excluded_path = args.excluded_output_jsonl or args.output_jsonl.with_name(args.output_jsonl.stem + ".excluded.jsonl")
    excluded_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in excluded_rows))
    summary = {
        "schema_version": "aegisvox.condition_relex_adjudication_rows.v1",
        "status": "complete",
        "condition": args.condition,
        "num_rows": len(output),
        "frozen_expected_rows": args.expected_edits,
        "num_condition_audit_rows_total": len(audits),
        "num_condition_audit_rows_outside_frozen_population": len(excluded_rows),
        "excluded_rows_jsonl": str(excluded_path),
        "excluded_rows_sha256": digest(excluded_path),
        "num_runs": len({row["run_id"] for row in output}),
        "canonical_rows_sha256": digest(args.canonical_rows),
        "paired_transcripts_sha256": digest(args.paired_transcripts),
        "phi_edits_csv_sha256": digest(args.phi_edits_csv),
        "deterministic_matching_in_prompt": False,
        "population_policy": "exact_frozen_601_canonical_ids_with_fail_visible_outside_population_rows",
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
