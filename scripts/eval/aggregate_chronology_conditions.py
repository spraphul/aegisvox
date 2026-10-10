#!/usr/bin/env python3
"""Combine complete per-condition chronology adjudication reports."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--condition",action="append",required=True,help="slug=summary.json"); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--expected-rows",type=int,default=390); ap.add_argument("--expected-conditions",type=int,default=5); args=ap.parse_args()
    if len(args.condition) != args.expected_conditions:
        raise ValueError(f"expected {args.expected_conditions} chronology conditions, received {len(args.condition)}")
    conditions={}
    for raw in args.condition:
        slug,path_raw=raw.split("=",1); path=Path(path_raw); data=json.loads(path.read_text())
        if data.get("num_completed")!=args.expected_rows or data.get("num_errors")!=0: raise ValueError(f"incomplete chronology condition {slug}")
        if data.get("schema_version") != "aegisvox.target_chronology.v2": raise ValueError(f"unexpected chronology schema for {slug}")
        conditions[slug]={key:data.get(key) for key in ("num_rows","num_completed","total_relations","chronology_preserved_rate","final_state_preserved_rate","revision_relation_count","revision_loss_count","revision_loss_rate","clinically_meaningful_event_order_error_count","clinically_meaningful_event_order_error_rate","relation_status_counts","relation_type_counts","relation_error_counts","final_state_status_counts","conversation_assessment_counts","long_range_strata")}; conditions[slug]["summary_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    relation_counts={value["total_relations"] for value in conditions.values()}
    if len(relation_counts) != 1: raise ValueError("condition relation denominators differ")
    report={"schema_version":"aegisvox.chronology_conditions.v2","status":"complete","num_conditions":len(conditions),"expected_rows_per_condition":args.expected_rows,"shared_source_relation_count":next(iter(relation_counts)),"conditions":conditions,"predeclared_primary_comparison":{"system":"aegisvox_f5","baseline":"stock_voicecraft"},"claim_boundary":"source-anchored GPT-5.4 full-context model adjudication with exact evidence; not human ground truth"}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps({"status":"complete","conditions":sorted(conditions),"output":str(args.output)})); return 0
if __name__=="__main__": raise SystemExit(main())
