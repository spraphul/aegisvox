#!/usr/bin/env python3
"""Fail-closed registry for Study 4 plan safety and condition realization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-safety", type=Path, required=True)
    ap.add_argument("--condition", action="append", default=[], help="slug=summary.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-edits", type=int, default=601)
    ap.add_argument("--expected-conditions", type=int, default=5)
    args = ap.parse_args()
    plan = json.loads(args.plan_safety.read_text())
    if plan.get("status") != "complete" or plan.get("num_edits") != args.expected_edits:
        raise SystemExit(f"incomplete plan-safety evidence: {plan}")
    conditions = {}
    evidence = {"plan_safety": {"path": str(args.plan_safety), "sha256": digest(args.plan_safety)}}
    for spec in args.condition:
        slug, raw = spec.split("=", 1)
        path = Path(raw)
        row = json.loads(path.read_text())
        if row.get("status") != "complete" or row.get("num_requested_rows") != args.expected_edits:
            raise SystemExit(f"incomplete realization evidence for {slug}: {row}")
        conditions[slug] = row
        evidence[slug] = {"path": str(path), "sha256": digest(path)}
    if len(conditions) != args.expected_conditions:
        raise SystemExit(f"expected {args.expected_conditions} conditions, found {len(conditions)}")
    output = {
        "schema_version": "aegisvox.relex_condition_studies.v1",
        "status": "complete",
        "primary_denominator": "intention_to_transform",
        "num_frozen_edits": args.expected_edits,
        "plan_clinical_safety": plan,
        "condition_realization": conditions,
        "evidence_registry": evidence,
        "deterministic_proxy_used": False,
        "claim_boundary": "Plan safety and ASR-evidenced audible realization are model-adjudicated endpoints; human listening is reported separately.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "num_conditions": len(conditions), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
