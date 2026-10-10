#!/usr/bin/env python3
"""Recording-paired contrasts for source-CAP/role frontend outcomes.

Binary contrasts are paired risk differences. This report is a preregistered
descriptive/inferential layer and does not replace the final clustered ITT
mixed models or human evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compute_frontend_objective_contrasts import build_report, read_jsonl


FIELDS = (
    "any_critical_major_cap_error",
    "any_role_attribution_error",
    "any_major_unsupported_target_assertion",
    "conversation_major_error",
    "conversation_uncertain",
    "conversation_no_major_error",
    "cap_strict_preservation_rate",
    "cap_lenient_preservation_rate",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--permutation-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2702)
    args = parser.parse_args()
    report = build_report(
        read_jsonl(args.registry_jsonl), fields=list(FIELDS),
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations, seed=args.seed,
    )
    report["schema_version"] = "aegisvox.frontend_cap_role_contrasts.v1"
    report["multiplicity"]["family"] = "all frozen CAP/role frontend contrasts in this report"
    report["input"] = {
        "path": str(args.registry_jsonl),
        "sha256": hashlib.sha256(args.registry_jsonl.read_bytes()).hexdigest(),
    }
    report["effect_scale"] = {
        "binary_fields": "paired absolute risk difference",
        "rate_fields": "paired mean rate difference",
    }
    report["claim_boundary"] = (
        "Source-CAP-anchored GPT-5.4 full-context adjudication with recording-paired "
        "contrasts; not human ground truth and not the final W4.2 mixed-model report."
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "num_fields": len(FIELDS), "num_tests": report["multiplicity"]["num_tests"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
