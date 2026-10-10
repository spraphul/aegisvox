#!/usr/bin/env python3
"""Aggregate CAP preservation, modifier, and speaker-role endpoints by condition.

This is deterministic aggregation of evidence-linked GPT-5.4 adjudications;
it does not create new clinical labels or replace missing judgments with a
heuristic proxy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total <= 0:
        return [None, None]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def condition_report(
    target_path: Path,
    source_caps: dict[str, dict[str, Any]],
    *,
    expected_rows: int,
) -> dict[str, Any]:
    items = read_jsonl(target_path)
    if len(items) != expected_rows or len({row["audio_id"] for row in items}) != expected_rows:
        raise ValueError(f"{target_path}: expected {expected_rows} unique rows, got {len(items)}")
    status = Counter()
    modifier_errors = Counter()
    modifier_denominators = Counter()
    role_status = Counter()
    role_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    role_confusions = Counter()
    critical_role_changed = 0
    critical_role_total = 0
    conversation_assessment = Counter()
    unsupported = Counter()
    total_caps = 0
    for item in items:
        audio_id = str(item["audio_id"])
        caps = {str(cap["cap_id"]): cap for cap in source_caps[audio_id]["caps"]}
        result = item["result"]
        conversation_assessment[str(result["conversation_assessment"])] += 1
        for assertion in result.get("unsupported_target_assertions") or []:
            unsupported[str(assertion["clinical_impact"])] += 1
        for judgment in result["adjudications"]:
            cap = caps[str(judgment["cap_id"])]
            total_caps += 1
            status[str(judgment["status"])] += 1
            for name in ("negation", "temporality", "dose_value", "severity", "verification", "action_type", "final_state"):
                # Every CAP is eligible for error detection; applicability is
                # retained in the frozen source attributes for stratification.
                modifier_denominators[name] += 1
            modifier_errors.update(judgment.get("modifier_errors") or [])
            source_role = str(cap.get("source_role") or "unknown")
            target_role = str(judgment.get("target_role") or "unknown")
            role = str(judgment.get("role_status") or "uncertain")
            if source_role != "unknown":
                role_status[role] += 1
                role_by_source[source_role][role] += 1
                if role == "changed":
                    role_confusions[f"{source_role}_to_{target_role}"] += 1
                if cap.get("clinical_importance") in {"critical", "major"}:
                    critical_role_total += 1
                    critical_role_changed += int(role == "changed")
    preserved = status["preserved"]
    lenient = preserved + status["partially_preserved"]
    role_total = sum(role_status.values())
    no_major = conversation_assessment["no_major_error"]
    return {
        "num_conversations": len(items),
        "num_caps": total_caps,
        "cap_status_counts": dict(status),
        "strict_cap_preservation_rate": preserved / total_caps if total_caps else None,
        "strict_cap_preservation_wilson95": wilson(preserved, total_caps),
        "lenient_cap_preservation_rate": lenient / total_caps if total_caps else None,
        "lenient_cap_preservation_wilson95": wilson(lenient, total_caps),
        "modifier_error_counts": dict(modifier_errors),
        "modifier_error_rates": {
            key: modifier_errors[key] / modifier_denominators[key]
            for key in sorted(modifier_denominators)
        },
        "role_eligible_caps": role_total,
        "role_status_counts": dict(role_status),
        "role_preservation_rate": role_status["preserved"] / role_total if role_total else None,
        "role_preservation_wilson95": wilson(role_status["preserved"], role_total),
        "critical_major_role_change_rate": critical_role_changed / critical_role_total if critical_role_total else None,
        "critical_major_role_changed": critical_role_changed,
        "critical_major_role_total": critical_role_total,
        "role_status_by_source_role": {key: dict(value) for key, value in sorted(role_by_source.items())},
        "role_confusion_counts": dict(role_confusions),
        "conversation_assessment_counts": dict(conversation_assessment),
        "conversation_no_major_error_rate": no_major / len(items),
        "conversation_no_major_error_wilson95": wilson(no_major, len(items)),
        "unsupported_target_assertions_by_impact": dict(unsupported),
        "input_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-caps", type=Path, required=True)
    parser.add_argument("--condition", action="append", required=True, help="slug=cap_target_adjudications.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=390)
    args = parser.parse_args()
    source_rows = read_jsonl(args.source_caps)
    source_caps = {str(row["audio_id"]): row for row in source_rows}
    if len(source_caps) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} unique source CAP rows, got {len(source_caps)}")
    conditions = {}
    for raw in args.condition:
        slug, path = raw.split("=", 1)
        conditions[slug] = condition_report(Path(path), source_caps, expected_rows=args.expected_rows)
    report = {
        "schema_version": "aegisvox.cap_condition_studies.v1",
        "status": "complete",
        "num_conditions": len(conditions),
        "expected_rows_per_condition": args.expected_rows,
        "conditions": conditions,
        "claim_boundary": "evidence-linked GPT-5.4 adjudication; not human ground truth",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "conditions": sorted(conditions), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
