#!/usr/bin/env python3
"""Report frontend action-risk outcomes across frozen error/quality strata."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("frontend_equity",HERE/"compute_frontend_cap_role_equity.py")
assert spec and spec.loader
M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
ENDPOINTS=(
    "any_action_not_strictly_preserved","any_action_state_error","any_action_role_error",
    "any_unsafe_action_change","any_potentially_harmful_action_change",
    "any_unsupported_target_action","conversation_major_or_harmful_change","conversation_plan_uncertain",
)


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--registry-jsonl",type=Path,required=True)
    ap.add_argument("--natural-difficulty-registry",type=Path,required=True); ap.add_argument("--output-json",type=Path,required=True)
    ap.add_argument("--expected-rows",type=int,default=390); args=ap.parse_args()
    M.ENDPOINTS=ENDPOINTS
    report=M.build(M.read_jsonl(args.registry_jsonl),expected_rows=args.expected_rows,
                   natural_rows=M.read_jsonl(args.natural_difficulty_registry))
    report["schema_version"]="aegisvox.frontend_action_equity.v1"
    report["claim_boundary"]="Source-action-anchored GPT-5.4 strata; not human ground truth."
    report["input_sha256"]=hashlib.sha256(args.registry_jsonl.read_bytes()).hexdigest()
    report["natural_difficulty_input_sha256"]=hashlib.sha256(args.natural_difficulty_registry.read_bytes()).hexdigest()
    args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"complete","num_cells":report["num_cells"]}))
    return 0


if __name__=="__main__": raise SystemExit(main())
