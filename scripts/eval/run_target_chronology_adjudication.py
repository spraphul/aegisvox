#!/usr/bin/env python3
"""Adjudicate frozen source chronology relations against full target transcripts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from run_gpt_concept_preservation import call_openai_compat_bearer_chat, call_chat_completions, extract_json_text_chat, group_pairs, read_jsonl as read_paired_jsonl, safe_parse_model_json  # noqa: E402
from run_clinical_cap_extraction import literalize_quote  # noqa: E402

SYSTEM_PROMPT="""You are a conservative source-anchored clinical chronology adjudicator.

The chronology relations were frozen from the complete source before you saw
the target. Judge each supplied relation using the complete source and target.
Preserve event order, past/current/future state, duration/trend, conditions,
planned-versus-completed status, and the final state after a correction. A safe
identity replacement is not a chronology error. Every non-empty target quote
must be copied literally and contiguously from the target transcript. Return
strict JSON only and do not invent missing evidence.
"""

USER_TEMPLATE="""audio_id: {audio_id}
dataset_id: {dataset_id}
language: {language}

FROZEN SOURCE CHRONOLOGY RELATIONS:
{relations}

COMPLETE SOURCE TRANSCRIPT:
{source}

COMPLETE TARGET TRANSCRIPT (same evaluator ASR):
{target}

Return:
{{
  "audio_id":"{audio_id}",
  "adjudications":[
    {{
      "relation_id":"R1",
      "status":"preserved|changed|missing|uncertain",
      "target_event_a_quote":"literal target quote or empty",
      "target_event_b_quote":"literal target quote or empty",
      "relation_error":"none|event_order|past_current_future|duration_or_trend|planned_vs_completed|conditionality|revision_or_correction|other|uncertain",
      "final_state_status":"preserved|changed|missing|uncertain|not_applicable",
      "clinical_impact":"none|minor|major|potentially_harmful|uncertain",
      "rationale":"brief evidence-grounded reason"
    }}
  ],
  "conversation_assessment":"chronology_preserved|meaningful_chronology_error|insufficient_evidence|not_applicable"
}}
Emit exactly one adjudication for every supplied relation_id and no others.
"""

STATUSES={"preserved","changed","missing","uncertain"}; ERRORS={"none","event_order","past_current_future","duration_or_trend","planned_vs_completed","conditionality","revision_or_correction","other","uncertain"}; FINAL={"preserved","changed","missing","uncertain","not_applicable"}; IMPACT={"none","minor","major","potentially_harmful","uncertain"}; ASSESS={"chronology_preserved","meaningful_chronology_error","insufficient_evidence","not_applicable"}


def literalize_target_evidence(result: dict[str, Any], target: str) -> dict[str, Any]:
    """Align only unique token-identical model quotes to literal target text.

    This changes no clinical judgment. It repairs case, punctuation,
    apostrophe, or whitespace normalization only when the quoted token sequence
    has exactly one match in the complete target transcript. Ambiguous quotes,
    paraphrases, omissions, and typos remain fail-closed for model repair.
    """
    for row in result.get("adjudications") or []:
        if not isinstance(row, dict):
            continue
        for field in ("target_event_a_quote", "target_event_b_quote"):
            quote = str(row.get(field) or "")
            if not quote or quote in target:
                continue
            aligned = literalize_quote(target, quote)
            if aligned is None:
                continue
            row[field] = aligned[0]
            row[f"{field}_alignment_method"] = aligned[1]
    return result

def load_env(path:Path|None)->None:
    if not path or not path.is_file(): return
    for raw in path.read_text().splitlines():
        line=raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key,value=line.split("=",1); os.environ.setdefault(key.strip(),value.strip().strip("'").strip('"'))

def read_raw_jsonl(path:Path)->list[dict[str,Any]]:
    """Read internal result JSONL without applying the paired-ASR row contract."""
    rows=[]
    with path.open("r",encoding="utf-8") as handle:
        for line_no,line in enumerate(handle,1):
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row,dict): raise SystemExit(f"invalid object at {path}:{line_no}")
            rows.append(row)
    return rows

def validate(
    result:dict[str,Any],*,audio_id:str,source_relations:dict[str,dict[str,Any]],target:str
)->None:
    if result.get("audio_id")!=audio_id: raise ValueError("audio_id mismatch")
    rows=result.get("adjudications")
    if not isinstance(rows,list): raise ValueError("adjudications must be a list")
    ids=[str(row.get("relation_id") or "") for row in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(source_relations): raise ValueError("relation bijection failed")
    for row in rows:
        rid=str(row["relation_id"])
        if row.get("status") not in STATUSES: raise ValueError(f"{rid}: invalid status")
        for field in ("target_event_a_quote","target_event_b_quote"):
            quote=str(row.get(field) or "")
            if quote and quote not in target: raise ValueError(f"{rid}: {field} is not literal target evidence")
        if row["status"] == "preserved" and not (row.get("target_event_a_quote") and row.get("target_event_b_quote")):
            raise ValueError(f"{rid}: preserved status requires target evidence for both events")
        if row["status"] == "changed" and not (row.get("target_event_a_quote") or row.get("target_event_b_quote")):
            raise ValueError(f"{rid}: changed status requires target evidence")
        if row.get("relation_error") not in ERRORS: raise ValueError(f"{rid}: invalid relation_error")
        if row["status"] == "preserved" and row.get("relation_error") != "none":
            raise ValueError(f"{rid}: preserved status requires relation_error=none")
        if row["status"] == "changed" and row.get("relation_error") in {"none", "uncertain"}:
            raise ValueError(f"{rid}: changed status requires a concrete relation_error")
        if row["status"] == "uncertain" and row.get("relation_error") != "uncertain":
            raise ValueError(f"{rid}: uncertain status requires relation_error=uncertain")
        if row.get("final_state_status") not in FINAL: raise ValueError(f"{rid}: invalid final_state_status")
        source_final = source_relations[rid].get("final_state")
        if source_final == "not_applicable" and row.get("final_state_status") != "not_applicable":
            raise ValueError(f"{rid}: source final state is not_applicable")
        if source_final != "not_applicable" and row.get("final_state_status") == "not_applicable":
            raise ValueError(f"{rid}: source final state requires adjudication")
        if row.get("clinical_impact") not in IMPACT: raise ValueError(f"{rid}: invalid clinical_impact")
        if not str(row.get("rationale") or "").strip(): raise ValueError(f"{rid}: empty rationale")
    if result.get("conversation_assessment") not in ASSESS: raise ValueError("invalid conversation_assessment")
    assessment = result["conversation_assessment"]
    if rows and all(row["status"] == "preserved" for row in rows) and assessment != "chronology_preserved":
        raise ValueError("all-preserved relations require chronology_preserved assessment")
    if assessment == "chronology_preserved" and any(row["status"] != "preserved" for row in rows):
        raise ValueError("chronology_preserved assessment conflicts with relation judgments")

def call(args:argparse.Namespace,messages:list[dict[str,str]])->dict[str,Any]:
    if args.backend=="openai_compat-bearer":
        token=os.environ.get("OPENAI_API_KEY","")
        if not token: raise RuntimeError("missing OPENAI_API_KEY")
        response=call_openai_compat_bearer_chat(base_url=args.base_url,bearer_token=token,model=args.model,messages=messages,temperature=0.0,max_tokens=args.max_tokens,timeout=args.timeout,verify_ssl=not args.no_verify_ssl,api_version=None,omit_api_version=False,service_name=None,completion_token_param="max_completion_tokens")
    else: response=call_chat_completions(base_url=args.base_url,api_key=args.api_key or "not-required",model=args.model,messages=messages,temperature=0.0,timeout=args.timeout,max_retries=2,verify_ssl=not args.no_verify_ssl,max_tokens=args.max_tokens)
    return safe_parse_model_json(extract_json_text_chat(response))

def evaluate(pair:dict[str,Any],source_item:dict[str,Any],args:argparse.Namespace)->dict[str,Any]:
    relations=source_item["result"]["relations"]
    if not relations: return {"audio_id":pair["audio_id"],"adjudications":[],"conversation_assessment":"not_applicable"}
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":USER_TEMPLATE.format(audio_id=pair["audio_id"],dataset_id=pair.get("source_dataset_id") or "unknown",language=pair.get("source_language") or "unknown",relations=json.dumps(relations,ensure_ascii=False),source=pair["reference"],target=pair["hypothesis"])}]
    result=literalize_target_evidence(call(args,messages),pair["hypothesis"]); source_relations={str(row["relation_id"]):row for row in relations}
    for attempt in range(args.schema_retries+1):
        try: validate(result,audio_id=pair["audio_id"],source_relations=source_relations,target=pair["hypothesis"]); return result
        except Exception as exc:
            if attempt>=args.schema_retries: raise
            messages += [{"role":"assistant","content":json.dumps(result,ensure_ascii=False)},{"role":"user","content":f"Repair the complete JSON after this validation error: {exc}. Keep valid judgments and copy target evidence literally from the full target transcript."}]; result=literalize_target_evidence(call(args,messages),pair["hypothesis"])
    raise AssertionError("unreachable")

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--paired-transcripts",type=Path,required=True); ap.add_argument("--source-chronology",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--backend",choices=["openai_compat-bearer","openai-compatible"],required=True); ap.add_argument("--model",required=True); ap.add_argument("--base-url",required=True); ap.add_argument("--env-file",type=Path); ap.add_argument("--api-key",default=""); ap.add_argument("--max-tokens",type=int,default=20000); ap.add_argument("--timeout",type=float,default=300); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--schema-retries",type=int,default=8); ap.add_argument("--limit",type=int); ap.add_argument("--resume",action="store_true"); ap.add_argument("--no-verify-ssl",action="store_true"); args=ap.parse_args(); load_env(args.env_file)
    pairs=group_pairs(read_paired_jsonl(args.paired_transcripts)); pairs=pairs[:args.limit] if args.limit is not None else pairs
    sources={str(row["audio_id"]):row for row in read_raw_jsonl(args.source_chronology)}; pairs=[p for p in pairs if p["audio_id"] in sources]
    args.output_dir.mkdir(parents=True,exist_ok=True); lock=(args.output_dir/".target_chronology.lock").open("w")
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc: raise RuntimeError("another target chronology process owns this output") from exc
    output=args.output_dir/"target_chronology.jsonl"; error_path=args.output_dir/"errors.jsonl"; completed={}
    if args.resume and output.exists():
        for row in read_raw_jsonl(output):
            pair=next((p for p in pairs if p["audio_id"]==row.get("audio_id")),None)
            if pair:
                try: validate(row["result"],audio_id=pair["audio_id"],source_relations={str(x["relation_id"]):x for x in sources[pair["audio_id"]]["result"]["relations"]},target=pair["hypothesis"]); completed[pair["audio_id"]]=row
                except ValueError: pass
    pending=[p for p in pairs if p["audio_id"] not in completed]; errors=[]
    with output.open("a") as out,error_path.open("a") as err,ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures={pool.submit(evaluate,p,sources[p["audio_id"]],args):p for p in pending}
        for future in as_completed(futures):
            pair=futures[future]
            try:
                result=future.result(); item={"audio_id":pair["audio_id"],"dataset_id":pair.get("source_dataset_id"),"language":pair.get("source_language"),"result":result}; completed[pair["audio_id"]]=item; out.write(json.dumps(item,ensure_ascii=False)+"\n"); out.flush()
            except Exception as exc:
                item={"audio_id":pair["audio_id"],"error_type":type(exc).__name__,"error":str(exc)[:4000]}; errors.append(item); err.write(json.dumps(item,ensure_ascii=False)+"\n"); err.flush()
            print(json.dumps({"completed":len(completed),"errors":len(errors),"total":len(pairs)}),flush=True)
    with output.open("w") as out:
        for pair in pairs:
            if pair["audio_id"] in completed: out.write(json.dumps(completed[pair["audio_id"]],ensure_ascii=False)+"\n")
    status=Counter(); error_types=Counter(); final=Counter(); assessments=Counter(); relation_types=Counter()
    revision_total=0; revision_loss=0; meaningful_event_order_errors=0
    strata: dict[str, dict[str, Counter[str]]] = {
        "turn_distance": {}, "time_distance": {}, "conversation_duration": {}
    }
    for item in completed.values():
        assessments[item["result"]["conversation_assessment"]]+=1
        source_by_id={str(row["relation_id"]):row for row in sources[item["audio_id"]]["result"]["relations"]}
        for row in item["result"]["adjudications"]:
            status[row["status"]]+=1; error_types[row["relation_error"]]+=1; final[row["final_state_status"]]+=1
            source_relation=source_by_id[str(row["relation_id"])]
            relation_types[str(source_relation["relation"])]+=1
            if source_relation["relation"] in {"revises", "corrects"}:
                revision_total += 1
                if row["status"] != "preserved" or row["final_state_status"] != "preserved":
                    revision_loss += 1
            if row["relation_error"] == "event_order" and row["clinical_impact"] in {"major", "potentially_harmful"}:
                meaningful_event_order_errors += 1
            timeline=source_relation.get("source_timeline") or {}
            bucket_values={
                "turn_distance": timeline.get("turn_distance_bucket"),
                "time_distance": timeline.get("time_distance_bucket"),
                "conversation_duration": timeline.get("conversation_duration_bucket"),
            }
            for dimension,bucket in bucket_values.items():
                if not bucket: continue
                counts=strata[dimension].setdefault(str(bucket),Counter())
                counts["total"] += 1; counts[row["status"]] += 1
                if row["clinical_impact"] in {"major", "potentially_harmful"} and row["status"] != "preserved": counts["major_or_harmful_error"] += 1
                if row["relation_error"] == "event_order": counts["event_order_error"] += 1
                if source_relation["final_state"] != "not_applicable":
                    counts["final_state_applicable"] += 1
                    if row["final_state_status"] == "preserved": counts["final_state_preserved"] += 1
    total=sum(status.values()); applicable_final=sum(value for key,value in final.items() if key!="not_applicable")
    strata_report={}
    for dimension,buckets in strata.items():
        strata_report[dimension]={}
        for bucket,counts in sorted(buckets.items()):
            n=counts["total"]
            strata_report[dimension][bucket]={
                **dict(counts),
                "chronology_preserved_rate": counts["preserved"]/n if n else None,
                "major_or_harmful_error_rate": counts["major_or_harmful_error"]/n if n else None,
                "event_order_error_rate": counts["event_order_error"]/n if n else None,
                "final_state_preserved_rate": counts["final_state_preserved"]/counts["final_state_applicable"] if counts["final_state_applicable"] else None,
            }
    summary={"schema_version":"aegisvox.target_chronology.v2","model":args.model,"num_rows":len(pairs),"num_completed":len(completed),"num_errors":len(errors),"total_relations":total,"relation_status_counts":dict(status),"relation_type_counts":dict(relation_types),"chronology_preserved_rate":status["preserved"]/total if total else None,"relation_error_counts":dict(error_types),"clinically_meaningful_event_order_error_count":meaningful_event_order_errors,"clinically_meaningful_event_order_error_rate":meaningful_event_order_errors/total if total else None,"final_state_status_counts":dict(final),"final_state_preserved_rate":final["preserved"]/applicable_final if applicable_final else None,"revision_relation_count":revision_total,"revision_loss_count":revision_loss,"revision_loss_rate":revision_loss/revision_total if revision_total else None,"conversation_assessment_counts":dict(assessments),"long_range_strata":strata_report,"input_sha256":hashlib.sha256(args.paired_transcripts.read_bytes()).hexdigest(),"source_chronology_sha256":hashlib.sha256(args.source_chronology.read_bytes()).hexdigest(),"prompt_sha256":hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}; (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); print(json.dumps(summary,indent=2)); return 0 if len(completed)==len(pairs) and not errors else 2

if __name__=="__main__": raise SystemExit(main())
