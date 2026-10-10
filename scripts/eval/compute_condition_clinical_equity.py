#!/usr/bin/env python3
"""Compute condition-specific Study-9 endpoint rates across frozen strata."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "language", "dataset_id", "duration_quartile", "speaker_count_bin",
    "edit_count_bin", "max_edit_duration_bin",
)


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(k: int, n: int) -> list[float]:
    z = 1.959963984540054; p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0., c-h), min(1., c+h)]


def summarize(rows: list[dict[str, Any]], endpoints: list[str]) -> dict[str, Any]:
    result = {}
    for endpoint in endpoints:
        values = [row.get(endpoint) for row in rows]
        if any(value not in {0, 1} for value in values): raise ValueError(f"{endpoint}: incomplete")
        events = sum(values); n = len(values)
        result[endpoint] = {"n": n, "events": events, "rate": events/n, "wilson95": wilson(events,n)}
    return result


def build(rows: list[dict[str, Any]], expected_rows: int) -> dict[str, Any]:
    families = {str(row.get("outcome_family") or "") for row in rows}
    conditions = {str(row.get("condition") or "") for row in rows}
    if len(families) != 1 or len(conditions) != 5 or len(rows) != 5*expected_rows:
        raise ValueError("expected one exact five-condition family")
    family = next(iter(families))
    endpoint_candidates = [key for key in rows[0] if key.startswith("any_") or key.startswith("conversation_")]
    endpoints = [key for key in endpoint_candidates if all(row.get(key) in {0,1} for row in rows)]
    if not endpoints: raise ValueError("no complete binary endpoints")
    report = {}
    for condition in sorted(conditions):
        cr = [row for row in rows if row["condition"] == condition]
        if len(cr) != expected_rows: raise ValueError(f"{condition}: wrong denominator")
        dimensions = {}
        for dimension in DIMENSIONS:
            groups: dict[str,list[dict[str,Any]]] = defaultdict(list)
            for row in cr: groups[str(row.get(dimension) if row.get(dimension) is not None else "missing")].append(row)
            dimensions[dimension] = {group: summarize(items,endpoints) for group,items in sorted(groups.items())}
        phi: dict[str,list[dict[str,Any]]] = defaultdict(list)
        for row in cr:
            active=[str(k) for k,v in (row.get("phi_type_counts") or {}).items() if int(v)>0]
            if not active: phi["none"].append(row)
            else:
                phi["any_phi"].append(row)
                for key in active: phi[key].append(row)
        dimensions["phi_entity_type_multilabel"]={group:summarize(items,endpoints) for group,items in sorted(phi.items())}
        report[condition]={"overall":summarize(cr,endpoints),"strata":dimensions}
    return {
        "schema_version":"aegisvox.condition_clinical_equity.v1","status":"complete",
        "outcome_family":family,"num_conditions":5,"num_rows_per_condition":expected_rows,
        "endpoints":endpoints,"dimensions":list(DIMENSIONS)+["phi_entity_type_multilabel"],
        "conditions":report,
        "language_claim_boundary":"English and French are reported separately; no broader multilingual claim.",
        "claim_boundary":"Model-adjudicated outcomes with explicit denominators and Wilson intervals; not human ground truth.",
    }


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry-jsonl",type=Path,required=True); ap.add_argument("--output-json",type=Path,required=True)
    ap.add_argument("--expected-rows",type=int,default=390); args=ap.parse_args()
    report=build(read(args.registry_jsonl),args.expected_rows)
    report["input_sha256"]=hashlib.sha256(args.registry_jsonl.read_bytes()).hexdigest()
    args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"complete","family":report["outcome_family"]}))
    return 0


if __name__=="__main__": raise SystemExit(main())
