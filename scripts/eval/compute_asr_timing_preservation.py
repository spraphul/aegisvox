#!/usr/bin/env python3
"""Compute same-ASR source/target speech-rate and pause preservation.

This is deterministic timing evidence derived from the same frozen ASR backend
on source and target. It is not a transcript-content or clinical judgment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any

import numpy as np


WORD=re.compile(r"[^\W_]+(?:['’-][^\W_]+)*",re.UNICODE)
ENDPOINTS=(
    "speech_rate_words_per_minute","speech_activity_fraction","pause_rate_per_minute",
    "median_pause_sec","p90_pause_sec","longest_pause_sec",
)


def read_jsonl(path:Path)->list[dict[str,Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values:list[float],q:float)->float:
    ordered=sorted(values)
    if not ordered: return 0.0
    position=(len(ordered)-1)*q; lower=int(position); upper=min(lower+1,len(ordered)-1); weight=position-lower
    return ordered[lower]*(1-weight)+ordered[upper]*weight


def merged_intervals(segments:list[dict[str,Any]])->list[tuple[float,float]]:
    intervals=sorted((max(0.0,float(row["start_sec"])),max(0.0,float(row["end_sec"]))) for row in segments if float(row["end_sec"])>float(row["start_sec"]))
    merged=[]
    for start,end in intervals:
        if merged and start<=merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],end))
        else: merged.append((start,end))
    return merged


def summarize_asr(path:Path)->dict[str,Any]:
    payload=json.loads(path.read_text()); segments=payload.get("segments")
    if not isinstance(segments,list) or not segments: raise ValueError(f"missing ASR segments: {path}")
    intervals=merged_intervals(segments)
    if not intervals: raise ValueError(f"no positive ASR intervals: {path}")
    speech_seconds=sum(end-start for start,end in intervals); duration=max(end for _,end in intervals)
    pauses=[max(0.0,intervals[index+1][0]-intervals[index][1]) for index in range(len(intervals)-1)]
    pauses=[value for value in pauses if value>=0.5]
    words=sum(len(WORD.findall(str(row.get("text") or ""))) for row in segments)
    return {
        "backend":payload.get("backend"),"backend_version":payload.get("backend_version"),"num_segments":len(segments),
        "num_words":words,"asr_timeline_duration_sec":duration,"speech_seconds":speech_seconds,
        "speech_rate_words_per_minute":60.0*words/speech_seconds,
        "speech_activity_fraction":speech_seconds/duration if duration else 0.0,
        "num_pauses_ge_0p5sec":len(pauses),"num_pauses_ge_1sec":sum(value>=1.0 for value in pauses),"num_pauses_ge_2sec":sum(value>=2.0 for value in pauses),
        "pause_rate_per_minute":60.0*len(pauses)/duration if duration else 0.0,
        "median_pause_sec":statistics.median(pauses) if pauses else 0.0,
        "p90_pause_sec":percentile(pauses,0.9),"longest_pause_sec":max(pauses,default=0.0),
    }


def bootstrap_ci(values:list[float],iterations:int,seed:int)->list[float]:
    array=np.asarray(values,dtype=np.float64); rng=np.random.default_rng(seed); means=[]
    for start in range(0,iterations,1000):
        count=min(1000,iterations-start)
        indices=rng.integers(0,len(array),size=(count,len(array)))
        means.extend(array[indices].mean(axis=1).tolist())
    return [percentile(means,0.025),percentile(means,0.975)]


def pair_manifest(path:Path,expected_rows:int)->list[dict[str,Any]]:
    rows=[]
    for row in read_jsonl(path):
        if row.get("source")!="reference": continue
        rows.append({"audio_id":str(row["audio_id"]),"source_asr":Path(row["reference_asr_path"]),"target_asr":Path(row["hypothesis_asr_path"]),"backend":row.get("reference_backend_version")})
    if len(rows)!=expected_rows or len({row["audio_id"] for row in rows})!=expected_rows: raise ValueError(f"paired population mismatch: {path}")
    return sorted(rows,key=lambda row:row["audio_id"])


def compute(condition_paths:dict[str,Path],expected_rows:int,bootstrap_iterations:int,seed:int)->tuple[dict[str,Any],list[dict[str,Any]]]:
    output=[]; populations=[]; input_hashes=[]
    for condition,path in condition_paths.items():
        pairs=pair_manifest(path,expected_rows); populations.append({row["audio_id"] for row in pairs}); input_hashes.append({"condition":condition,"path":str(path),"sha256":sha256(path)})
        for pair in pairs:
            source=summarize_asr(pair["source_asr"]); target=summarize_asr(pair["target_asr"])
            if source["backend_version"]!=target["backend_version"]: raise ValueError(f"same-ASR contract failed: {condition}/{pair['audio_id']}")
            row={"audio_id":pair["audio_id"],"condition":condition,"backend_version":source["backend_version"],"source_asr":str(pair["source_asr"]),"target_asr":str(pair["target_asr"])}
            for endpoint in ENDPOINTS:
                row[f"source_{endpoint}"]=source[endpoint]; row[f"target_{endpoint}"]=target[endpoint]
                row[f"delta_{endpoint}"]=target[endpoint]-source[endpoint]; row[f"absolute_delta_{endpoint}"]=abs(row[f"delta_{endpoint}"])
            row["source_num_words"]=source["num_words"]; row["target_num_words"]=target["num_words"]
            output.append(row)
    if len({frozenset(population) for population in populations})!=1: raise ValueError("condition populations differ")
    summaries={}
    for condition in condition_paths:
        rows=[row for row in output if row["condition"]==condition]; endpoint_summary={}
        for index,endpoint in enumerate(ENDPOINTS):
            signed=[row[f"delta_{endpoint}"] for row in rows]; absolute=[row[f"absolute_delta_{endpoint}"] for row in rows]
            endpoint_summary[endpoint]={"mean_source":statistics.fmean(row[f"source_{endpoint}"] for row in rows),"mean_target":statistics.fmean(row[f"target_{endpoint}"] for row in rows),"mean_signed_delta":statistics.fmean(signed),"mean_absolute_delta":statistics.fmean(absolute),"recording_bootstrap_signed_delta_ci95":bootstrap_ci(signed,bootstrap_iterations,seed+100*list(condition_paths).index(condition)+index)}
        summaries[condition]={"n":len(rows),"endpoints":endpoint_summary}
    report={"schema_version":"aegisvox.same_asr_timing_preservation.v1","status":"complete","analysis_unit":"recording","expected_rows_per_condition":expected_rows,"num_conditions":len(condition_paths),"num_rows":len(output),"pause_definition":"gap >=0.5 s between merged frozen-ASR segments","conditions":summaries,"inputs":input_hashes,"bootstrap_iterations":bootstrap_iterations,"seed":seed,"claim_boundary":"Same-ASR speech-rate/activity/pause timing preservation; not non-speech-event recognition, clinical ground truth, or human perception."}
    return report,output


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--condition",action="append",required=True); ap.add_argument("--expected-rows",type=int,default=390); ap.add_argument("--bootstrap-iterations",type=int,default=10000); ap.add_argument("--seed",type=int,default=20260808); ap.add_argument("--output-json",type=Path,required=True); ap.add_argument("--rows-csv",type=Path,required=True); args=ap.parse_args()
    conditions={}
    for specification in args.condition:
        name,separator,path=specification.partition("=")
        if not separator or not name or name in conditions: raise SystemExit(f"invalid/duplicate --condition {specification}")
        conditions[name]=Path(path)
    report,rows=compute(conditions,args.expected_rows,args.bootstrap_iterations,args.seed)
    args.output_json.parent.mkdir(parents=True,exist_ok=True); args.rows_csv.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    with args.rows_csv.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status":"complete","num_rows":len(rows),"output":str(args.output_json)})); return 0


if __name__=="__main__": raise SystemExit(main())
