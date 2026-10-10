#!/usr/bin/env python3
"""Extract evidence-linked source chronology/revision relations from full transcripts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from run_gpt_concept_preservation import (  # noqa: E402
    call_openai_compat_bearer_chat, call_chat_completions, extract_json_text_chat,
    group_pairs, read_jsonl, safe_parse_model_json,
)

SYSTEM_PROMPT = """You are a conservative clinical chronology extractor.

Read the complete source conversation only. Extract clinically meaningful
relations between two explicitly stated events, states, plans, or revisions.
Include onset/progression, past-current-future relations, planned-versus-done,
conditional escalation, and later corrections of earlier statements. Do not
manufacture an ordering merely because sentences appear in sequence. Every
event requires a literal contiguous source quote that is sufficiently specific
to locate it. Return strict JSON only.
"""

USER_TEMPLATE = """audio_id: {audio_id}
dataset_id: {dataset_id}
language: {language}

COMPLETE SOURCE TRANSCRIPT:
{source}

Return:
{{
  "audio_id": "{audio_id}",
  "relations": [
    {{
      "relation_id": "R1",
      "event_a": "atomic event/state/plan",
      "event_a_quote": "exact source quote",
      "event_b": "atomic event/state/plan",
      "event_b_quote": "exact source quote",
      "relation": "before|after|same_time|overlaps|conditional_on|revises|corrects|causes_or_triggers",
      "final_state": "event_a|event_b|both|not_applicable",
      "clinical_importance": "critical|major|minor",
      "rationale": "brief source-grounded explanation"
    }}
  ]
}}

An empty relations list is allowed only when no explicit clinically meaningful
chronology/revision relation exists in the complete source.
"""

RELATIONS = {"before", "after", "same_time", "overlaps", "conditional_on", "revises", "corrects", "causes_or_triggers"}
FINAL_STATES = {"event_a", "event_b", "both", "not_applicable"}
IMPORTANCE = {"critical", "major", "minor"}


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-paired timeline or chronology result JSONL."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_timeline_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_raw_jsonl(path)
    timelines = {str(row.get("audio_id") or ""): row for row in rows}
    if "" in timelines or len(timelines) != len(rows):
        raise ValueError("timeline metadata has empty or duplicate audio_id values")
    return timelines


def _segment_character_ranges(
    transcript: str, segments: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Locate each ASR segment exactly in the top-level ASR transcript."""
    cursor = 0
    ranges: list[tuple[int, int]] = []
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "").strip()
        if not text:
            raise ValueError(f"ASR segment {index} has empty text")
        start = transcript.find(text, cursor)
        if start < 0:
            raise ValueError(f"ASR segment {index} is not an exact ordered transcript substring")
        end = start + len(text)
        ranges.append((start, end))
        cursor = end
    return ranges


def _speaker_turns(segments: list[dict[str, Any]], max_same_speaker_gap_sec: float = 1.5) -> list[dict[str, Any]]:
    """Coalesce consecutive diarization fragments into deterministic turns."""
    turns: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda row: (float(row["start_sec"]), float(row["end_sec"]))):
        speaker = str(segment.get("speaker_id") or "").strip()
        if not speaker:
            continue
        start, end = float(segment["start_sec"]), float(segment["end_sec"])
        if (
            turns
            and turns[-1]["speaker_id"] == speaker
            and start - float(turns[-1]["end_sec"]) <= max_same_speaker_gap_sec
        ):
            turns[-1]["end_sec"] = max(float(turns[-1]["end_sec"]), end)
        else:
            turns.append({"speaker_id": speaker, "start_sec": start, "end_sec": end})
    if not turns:
        raise ValueError("timeline has no usable diarization turns")
    return turns


def _anchor_quote(
    *,
    quote: str,
    transcript: str,
    asr_segments: list[dict[str, Any]],
    segment_ranges: list[tuple[int, int]],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    start = transcript.find(quote)
    if start < 0 or transcript.find(quote, start + 1) >= 0:
        raise ValueError("chronology quote must occur exactly once in source transcript")
    end = start + len(quote)
    midpoint = (start + end) / 2.0
    segment_index = min(
        range(len(segment_ranges)),
        key=lambda idx: 0.0
        if segment_ranges[idx][0] <= midpoint <= segment_ranges[idx][1]
        else min(abs(midpoint - segment_ranges[idx][0]), abs(midpoint - segment_ranges[idx][1])),
    )
    char_start, char_end = segment_ranges[segment_index]
    segment = asr_segments[segment_index]
    denominator = max(1.0, float(char_end - char_start))
    fraction = min(1.0, max(0.0, (midpoint - char_start) / denominator))
    time_sec = float(segment["start_sec"]) + fraction * (
        float(segment["end_sec"]) - float(segment["start_sec"])
    )
    turn_index = min(
        range(len(turns)),
        key=lambda idx: 0.0
        if float(turns[idx]["start_sec"]) <= time_sec <= float(turns[idx]["end_sec"])
        else min(
            abs(time_sec - float(turns[idx]["start_sec"])),
            abs(time_sec - float(turns[idx]["end_sec"])),
        ),
    )
    return {
        "char_start": start,
        "char_end": end,
        "asr_segment_index": segment_index,
        "time_sec": round(time_sec, 3),
        "speaker_turn_index": turn_index,
        "speaker_id": turns[turn_index]["speaker_id"],
    }


def _turn_distance_bucket(distance: int) -> str:
    if distance <= 1:
        return "same_or_adjacent_0_1"
    if distance <= 4:
        return "medium_2_4"
    return "long_5_plus"


def _time_distance_bucket(distance_sec: float) -> str:
    if distance_sec <= 30.0:
        return "short_le_30s"
    if distance_sec <= 120.0:
        return "medium_30_120s"
    return "long_gt_120s"


def _duration_bucket(duration_sec: float) -> str:
    if duration_sec < 180.0:
        return "short_lt_3m"
    if duration_sec <= 600.0:
        return "medium_3_10m"
    return "long_gt_10m"


def enrich_with_timeline(
    result: dict[str, Any], *, transcript: str, timeline: dict[str, Any]
) -> dict[str, Any]:
    expected_sha = hashlib.sha256(transcript.encode()).hexdigest()
    if timeline.get("transcript_sha256") != expected_sha:
        raise ValueError("timeline transcript hash does not match source transcript")
    asr_segments = list(timeline.get("asr_segments") or [])
    segment_ranges = _segment_character_ranges(transcript, asr_segments)
    turns = _speaker_turns(list(timeline.get("diarization_segments") or []))
    duration_sec = float(timeline.get("duration_sec") or 0.0)
    if duration_sec <= 0:
        raise ValueError("timeline duration must be positive")
    for relation in result["relations"]:
        event_a = _anchor_quote(
            quote=str(relation["event_a_quote"]), transcript=transcript,
            asr_segments=asr_segments, segment_ranges=segment_ranges, turns=turns,
        )
        event_b = _anchor_quote(
            quote=str(relation["event_b_quote"]), transcript=transcript,
            asr_segments=asr_segments, segment_ranges=segment_ranges, turns=turns,
        )
        turn_distance = abs(int(event_a["speaker_turn_index"]) - int(event_b["speaker_turn_index"]))
        time_distance = abs(float(event_a["time_sec"]) - float(event_b["time_sec"]))
        relation["source_timeline"] = {
            "event_a": event_a,
            "event_b": event_b,
            "turn_distance": turn_distance,
            "time_distance_sec": round(time_distance, 3),
            "relative_time_distance": round(time_distance / duration_sec, 6),
            "conversation_duration_sec": round(duration_sec, 3),
            "turn_distance_bucket": _turn_distance_bucket(turn_distance),
            "time_distance_bucket": _time_distance_bucket(time_distance),
            "conversation_duration_bucket": _duration_bucket(duration_sec),
            "turn_definition": "adjacent ASR spans assigned to coalesced same-speaker diarization fragments",
        }
    return result


def load_env(path: Path | None) -> None:
    if not path or not path.is_file(): return
    for raw in path.read_text().splitlines():
        line=raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key,value=line.split("=",1)
            os.environ.setdefault(key.strip(),value.strip().strip("'").strip('"'))


def validate(result: dict[str, Any], *, audio_id: str, source: str) -> None:
    if result.get("audio_id") != audio_id: raise ValueError("audio_id mismatch")
    rows=result.get("relations")
    if not isinstance(rows,list): raise ValueError("relations must be a list")
    ids=[]
    for row in rows:
        if not isinstance(row,dict): raise ValueError("relation must be an object")
        rid=str(row.get("relation_id") or ""); ids.append(rid)
        if not rid: raise ValueError("empty relation_id")
        for field in ("event_a","event_b","rationale"):
            if not str(row.get(field) or "").strip(): raise ValueError(f"{rid}: empty {field}")
        for field in ("event_a_quote","event_b_quote"):
            quote=str(row.get(field) or "")
            if len(quote)<6 or quote not in source: raise ValueError(f"{rid}: {field} must be a literal source substring")
            if source.count(quote) != 1: raise ValueError(f"{rid}: {field} must uniquely locate one source span")
        if row.get("relation") not in RELATIONS: raise ValueError(f"{rid}: invalid relation")
        if row.get("final_state") not in FINAL_STATES: raise ValueError(f"{rid}: invalid final_state")
        if row.get("relation") in {"revises", "corrects"} and row.get("final_state") == "not_applicable":
            raise ValueError(f"{rid}: revision/correction requires an explicit final_state")
        if row.get("clinical_importance") not in IMPORTANCE: raise ValueError(f"{rid}: invalid clinical_importance")
    if len(ids)!=len(set(ids)): raise ValueError("duplicate relation_id")


def call(args: argparse.Namespace, messages: list[dict[str,str]]) -> dict[str,Any]:
    if args.backend=="openai_compat-bearer":
        token=os.environ.get("OPENAI_API_KEY","")
        if not token: raise RuntimeError("missing OPENAI_API_KEY")
        response=call_openai_compat_bearer_chat(base_url=args.base_url,bearer_token=token,model=args.model,messages=messages,temperature=0.0,max_tokens=args.max_tokens,timeout=args.timeout,verify_ssl=not args.no_verify_ssl,api_version=None,omit_api_version=False,service_name=None,completion_token_param="max_completion_tokens")
    else:
        response=call_chat_completions(base_url=args.base_url,api_key=args.api_key or "not-required",model=args.model,messages=messages,temperature=0.0,timeout=args.timeout,max_retries=2,verify_ssl=not args.no_verify_ssl,max_tokens=args.max_tokens)
    return safe_parse_model_json(extract_json_text_chat(response))


def evaluate(pair: dict[str,Any], timeline: dict[str, Any], args: argparse.Namespace) -> dict[str,Any]:
    messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":USER_TEMPLATE.format(audio_id=pair["audio_id"],dataset_id=pair.get("source_dataset_id") or "unknown",language=pair.get("source_language") or "unknown",source=pair["reference"])}]
    result=call(args,messages)
    for attempt in range(args.schema_retries+1):
        try:
            validate(result,audio_id=pair["audio_id"],source=pair["reference"])
            return enrich_with_timeline(result, transcript=pair["reference"], timeline=timeline)
        except Exception as exc:
            if attempt>=args.schema_retries: raise
            messages += [{"role":"assistant","content":json.dumps(result,ensure_ascii=False)},{"role":"user","content":f"Repair the complete JSON after this validation error: {exc}. Keep valid relations. All evidence quotes must be copied literally from the complete source transcript."}]
            result=call(args,messages)
    raise AssertionError("unreachable")


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired-transcripts",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--source-timeline",type=Path,required=True)
    ap.add_argument("--backend",choices=["openai_compat-bearer","openai-compatible"],required=True); ap.add_argument("--model",required=True); ap.add_argument("--base-url",required=True)
    ap.add_argument("--env-file",type=Path); ap.add_argument("--api-key",default=""); ap.add_argument("--max-tokens",type=int,default=20000); ap.add_argument("--timeout",type=float,default=300); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--schema-retries",type=int,default=8); ap.add_argument("--limit",type=int); ap.add_argument("--resume",action="store_true"); ap.add_argument("--no-verify-ssl",action="store_true")
    args=ap.parse_args(); load_env(args.env_file)
    pairs=group_pairs(read_jsonl(args.paired_transcripts)); pairs=pairs[:args.limit] if args.limit is not None else pairs
    timelines=read_timeline_metadata(args.source_timeline)
    missing_timeline=sorted({pair["audio_id"] for pair in pairs}-set(timelines))
    if missing_timeline: raise ValueError(f"missing source timeline metadata for {len(missing_timeline)} rows")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    lock=(args.output_dir/".source_chronology.lock").open("w")
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc: raise RuntimeError("another source chronology process owns this output") from exc
    output=args.output_dir/"source_chronology.jsonl"; errors_path=args.output_dir/"errors.jsonl"
    completed={}
    if args.resume and output.exists():
        for row in read_raw_jsonl(output):
            pair=next((p for p in pairs if p["audio_id"]==row.get("audio_id")),None)
            if pair:
                try:
                    validate(row["result"],audio_id=pair["audio_id"],source=pair["reference"])
                    # Old/incomplete results are deliberately recomputed.
                    if any("source_timeline" not in rel for rel in row["result"]["relations"]):
                        raise ValueError("missing source_timeline enrichment")
                    completed[pair["audio_id"]]=row
                except ValueError: pass
    pending=[p for p in pairs if p["audio_id"] not in completed]; errors=[]
    with output.open("a") as out, errors_path.open("a") as err, ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures={pool.submit(evaluate,p,timelines[p["audio_id"]],args):p for p in pending}
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
    rows=list(completed.values()); relation_count=sum(len(x["result"]["relations"]) for x in rows)
    summary={"schema_version":"aegisvox.source_chronology.v2","model":args.model,"num_rows":len(pairs),"num_completed":len(rows),"num_errors":len(errors),"total_relations":relation_count,"num_empty_relation_conversations":sum(not x["result"]["relations"] for x in rows),"input_sha256":hashlib.sha256(args.paired_transcripts.read_bytes()).hexdigest(),"source_timeline_sha256":hashlib.sha256(args.source_timeline.read_bytes()).hexdigest(),"prompt_sha256":hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),"predeclared_strata":{"turn_distance":["same_or_adjacent_0_1","medium_2_4","long_5_plus"],"time_distance":["short_le_30s","medium_30_120s","long_gt_120s"],"conversation_duration":["short_lt_3m","medium_3_10m","long_gt_10m"]}}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); print(json.dumps(summary,indent=2)); return 0 if len(rows)==len(pairs) and not errors else 2


if __name__=="__main__": raise SystemExit(main())
