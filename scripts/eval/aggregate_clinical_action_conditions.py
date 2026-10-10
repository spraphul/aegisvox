#!/usr/bin/env python3
"""Fail-closed aggregation of all five source-anchored action studies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIELDS = (
    "num_rows", "num_completed", "total_actions", "status_counts",
    "strict_action_recall", "lenient_action_recall", "state_status_counts",
    "role_status_counts", "action_kind_results", "unsafe_change_counts",
    "potentially_harmful_action_change_rate", "unsupported_target_action_count",
    "unsupported_target_action_impacts", "conversation_plan_assessment_counts",
    "conversation_plan_invariance_rate", "conversation_major_or_harmful_change_rate",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True, help="slug=summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=390)
    parser.add_argument("--expected-conditions", type=int, default=5)
    parser.add_argument("--primary-system", default="aegisvox_f5")
    parser.add_argument("--primary-baseline", default="stock_voicecraft")
    parser.add_argument(
        "--no-primary-comparison", action="store_true",
        help="Omit method-level primary comparison for factorial frontend reports.",
    )
    args = parser.parse_args()
    if len(args.condition) != args.expected_conditions:
        raise ValueError(f"expected {args.expected_conditions} conditions")
    conditions = {}
    for value in args.condition:
        slug, raw_path = value.split("=", 1)
        path = Path(raw_path)
        data = json.loads(path.read_text())
        if data.get("schema_version") != "aegisvox.target_clinical_actions.v2":
            raise ValueError(f"unexpected action schema for {slug}")
        if data.get("num_completed") != args.expected_rows or data.get("num_errors") != 0:
            raise ValueError(f"incomplete action condition {slug}")
        conditions[slug] = {field: data.get(field) for field in FIELDS}
        conditions[slug]["summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    totals = {row["total_actions"] for row in conditions.values()}
    if len(totals) != 1:
        raise ValueError("condition source-action denominators differ")
    report = {
        "schema_version": "aegisvox.clinical_action_conditions.v1",
        "status": "complete",
        "num_conditions": len(conditions),
        "expected_rows_per_condition": args.expected_rows,
        "shared_source_action_count": next(iter(totals)),
        "primary_comparison": None if args.no_primary_comparison else {
            "system": args.primary_system, "baseline": args.primary_baseline,
        },
        "conditions": conditions,
        "claim_boundary": "source-anchored GPT-5.4 full-context model adjudication; not diagnostic accuracy or human ground truth",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
