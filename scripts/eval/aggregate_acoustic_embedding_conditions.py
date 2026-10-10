#!/usr/bin/env python3
"""Aggregate exact-population paired CLAP/HeAR condition evidence.

The recording is the analysis/cluster unit.  Every condition must expose the
same exact audio IDs; missing or duplicate rows fail closed.  The primary
contrast is predeclared separately from BH-adjusted secondary contrasts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any


def sha256(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values:list[float],q:float)->float:
    if not values:
        raise ValueError("empty percentile")
    ordered=sorted(values); position=(len(ordered)-1)*q
    lower=int(position); upper=min(lower+1,len(ordered)-1); fraction=position-lower
    return ordered[lower]*(1-fraction)+ordered[upper]*fraction


def bootstrap_mean_ci(values:list[float],iterations:int,seed:int)->list[float]:
    rng=random.Random(seed); size=len(values)
    means=[sum(values[rng.randrange(size)] for _ in range(size))/size for _ in range(iterations)]
    return [percentile(means,0.025),percentile(means,0.975)]


def sign_flip_pvalue(differences:list[float],iterations:int,seed:int)->float:
    observed=abs(statistics.fmean(differences)); rng=random.Random(seed); exceed=0
    for _ in range(iterations):
        value=abs(sum(value if rng.random()<0.5 else -value for value in differences)/len(differences))
        exceed += value>=observed
    return (exceed+1)/(iterations+1)


def benjamini_hochberg(values:list[float])->list[float]:
    size=len(values); order=sorted(range(size),key=lambda index:values[index]); adjusted=[1.0]*size; running=1.0
    for rank_index in range(size-1,-1,-1):
        original_index=order[rank_index]; rank=rank_index+1
        running=min(running,values[original_index]*size/rank); adjusted[original_index]=min(1.0,running)
    return adjusted


def load_condition(path:Path,metric:str,expected_rows:int)->dict[str,float]:
    payload=json.loads(path.read_text()); rows=payload.get("rows")
    if payload.get("num_completed")!=expected_rows or payload.get("num_errors")!=0 or not isinstance(rows,list) or len(rows)!=expected_rows:
        raise ValueError(f"incomplete condition summary: {path}")
    result={}
    for row in rows:
        audio_id=str(row.get("audio_id") or "")
        if not audio_id or audio_id in result or row.get(metric) is None:
            raise ValueError(f"blank/duplicate/nonfinite row in {path}: {audio_id!r}")
        value=float(row[metric])
        if not (-1.0<=value<=1.0):
            raise ValueError(f"out-of-range cosine in {path}: {audio_id}={value}")
        result[audio_id]=value
    return result


def aggregate(
    conditions:dict[str,Path],*,metric:str,expected_rows:int,primary:str,
    headline_comparator:str,bootstrap_iterations:int,permutation_iterations:int,seed:int,
)->tuple[dict[str,Any],list[dict[str,Any]],list[dict[str,Any]]]:
    if primary not in conditions or headline_comparator not in conditions:
        raise ValueError("primary/headline comparator absent")
    values={name:load_condition(path,metric,expected_rows) for name,path in conditions.items()}
    populations={frozenset(rows) for rows in values.values()}
    if len(populations)!=1:
        raise ValueError("condition audio populations differ")
    audio_ids=sorted(next(iter(populations)))
    long_rows=[{"audio_id":audio_id,"condition":name,metric:values[name][audio_id]} for audio_id in audio_ids for name in conditions]
    summaries={}
    for index,(name,rows) in enumerate(values.items()):
        observed=[rows[audio_id] for audio_id in audio_ids]
        summaries[name]={
            "n":len(observed),"mean":statistics.fmean(observed),"sd":statistics.stdev(observed),
            "median":statistics.median(observed),"q1":percentile(observed,0.25),"q3":percentile(observed,0.75),
            "recording_clustered_bootstrap_mean_ci95":bootstrap_mean_ci(observed,bootstrap_iterations,seed+index),
        }
    contrasts=[]
    for index,comparator in enumerate(name for name in conditions if name!=primary):
        differences=[values[primary][audio_id]-values[comparator][audio_id] for audio_id in audio_ids]
        contrasts.append({
            "contrast":f"{primary}_minus_{comparator}","primary_condition":primary,"comparator":comparator,
            "is_predeclared_headline":comparator==headline_comparator,"n":len(differences),
            "mean_paired_difference":statistics.fmean(differences),"median_paired_difference":statistics.median(differences),
            "recording_clustered_bootstrap_mean_difference_ci95":bootstrap_mean_ci(differences,bootstrap_iterations,seed+100+index),
            "paired_sign_flip_pvalue":sign_flip_pvalue(differences,permutation_iterations,seed+200+index),
        })
    secondary=[row for row in contrasts if not row["is_predeclared_headline"]]
    adjusted=benjamini_hochberg([row["paired_sign_flip_pvalue"] for row in secondary])
    for row,value in zip(secondary,adjusted): row["bh_adjusted_secondary_pvalue"]=value
    for row in contrasts:
        if row["is_predeclared_headline"]: row["bh_adjusted_secondary_pvalue"]=None
    report={
        "schema_version":"aegisvox.acoustic_embedding_condition_statistics.v1","status":"complete",
        "metric":metric,"analysis_unit":"recording","expected_rows":expected_rows,"num_audio_ids":len(audio_ids),
        "primary_condition":primary,"headline_comparator":headline_comparator,
        "bootstrap_iterations":bootstrap_iterations,"permutation_iterations":permutation_iterations,"seed":seed,
        "conditions":summaries,"contrasts":contrasts,
        "inputs":[{"condition":name,"path":str(path),"sha256":sha256(path)} for name,path in conditions.items()],
        "missingness_policy":"Exact shared population required; no complete-case substitution.",
        "multiplicity_policy":"Predeclared headline contrast unadjusted; remaining primary-versus-comparator contrasts BH-adjusted as secondary.",
        "claim_boundary":"Paired acoustic-representation preservation only; not clinical ground truth, diagnosis, or human perception.",
    }
    return report,long_rows,contrasts


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition",action="append",required=True,help="NAME=summary.json")
    ap.add_argument("--metric",default="mean_dtw_cosine"); ap.add_argument("--expected-rows",type=int,default=390)
    ap.add_argument("--primary",default="aegisvox_f5"); ap.add_argument("--headline-comparator",default="stock_voicecraft")
    ap.add_argument("--bootstrap-iterations",type=int,default=10000); ap.add_argument("--permutation-iterations",type=int,default=10000); ap.add_argument("--seed",type=int,default=20260808)
    ap.add_argument("--output-json",type=Path,required=True); ap.add_argument("--rows-csv",type=Path,required=True); ap.add_argument("--contrasts-csv",type=Path,required=True); args=ap.parse_args()
    conditions={}
    for specification in args.condition:
        name,separator,path=specification.partition("=")
        if not separator or not name or name in conditions: raise SystemExit(f"invalid/duplicate --condition: {specification}")
        conditions[name]=Path(path)
    report,rows,contrasts=aggregate(conditions,metric=args.metric,expected_rows=args.expected_rows,primary=args.primary,headline_comparator=args.headline_comparator,bootstrap_iterations=args.bootstrap_iterations,permutation_iterations=args.permutation_iterations,seed=args.seed)
    for path in (args.output_json,args.rows_csv,args.contrasts_csv): path.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    with args.rows_csv.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with args.contrasts_csv.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(contrasts[0])); writer.writeheader(); writer.writerows(contrasts)
    print(json.dumps({"status":"complete","num_audio_ids":report["num_audio_ids"],"output":str(args.output_json)})); return 0


if __name__=="__main__": raise SystemExit(main())
