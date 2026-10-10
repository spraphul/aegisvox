#!/usr/bin/env python3
"""Judge CAP-anchored source/target clinical notes with exact note evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_gpt_concept_preservation import (  # noqa: E402
    call_openai_compat_bearer_chat, call_chat_completions, extract_json_text_chat,
    group_pairs, read_jsonl as read_paired_jsonl, safe_parse_model_json,
)
from run_clinical_note_generation import SECTIONS, read_jsonl  # noqa: E402

SECTION_ENUM = set(SECTIONS) | {"not_present", "uncertain"}
TARGET_STATUS = {"preserved", "partially_preserved", "missing", "contradicted", "uncertain"}
SOURCE_STATUS = {"included", "omitted", "uncertain"}
ROLE_STATUS = {"preserved", "changed", "not_applicable", "uncertain"}
MODIFIERS = {"negation", "temporality", "dose_value", "severity", "verification", "action_type", "final_state"}
IMPACTS = {"none", "minor", "major", "potentially_harmful", "uncertain"}
EXACTNESS = {"preserved", "changed", "missing", "uncertain", "not_applicable"}
ASSESSMENTS = {"no_major_error", "major_omission", "major_contradiction", "major_unsupported_addition", "uncertain"}

SYSTEM_PROMPT = """You are a conservative cross-note clinical documentation adjudicator.

The source CAP inventory was frozen before either note was generated. Compare
the structured source and target notes against the complete source and target
transcripts. For every CAP, judge whether it appears in the source note and is
preserved in the target note, including modifiers, speaker role, final state,
and chronology. Detect unsupported target-note assertions and section leakage.
Use only supplied evidence, never medical world knowledge. Every non-empty note
item must be copied exactly from a supplied note item. Return strict JSON."""

USER_TEMPLATE = """audio_id: {audio_id}
dataset_id: {dataset_id}
language: {language}

FROZEN SOURCE CAPS:
{caps}

COMPLETE SOURCE TRANSCRIPT:
{source_transcript}

COMPLETE TARGET TRANSCRIPT:
{target_transcript}

SOURCE NOTE:
{source_note}

TARGET NOTE:
{target_note}

Return:
{{
  "audio_id":"{audio_id}",
  "cap_adjudications":[
    {{
      "cap_id":"C1",
      "source_note_status":"included|omitted|uncertain",
      "source_section":"chief_concern|history_of_present_illness|medications|allergies|pertinent_history_and_review_of_systems|assessment_and_problem_list|plan_and_follow_up|not_present|uncertain",
      "source_note_item":"exact source-note item text or empty",
      "target_note_status":"preserved|partially_preserved|missing|contradicted|uncertain",
      "target_section":"same section enum",
      "target_note_item":"exact target-note item text or empty",
      "modifier_errors":["negation|temporality|dose_value|severity|verification|action_type|final_state"],
      "role_status":"preserved|changed|not_applicable|uncertain",
      "clinical_impact":"none|minor|major|potentially_harmful|uncertain",
      "rationale":"brief evidence-grounded reason"
    }}
  ],
  "unsupported_target_note_assertions":[
    {{"target_section":"section enum","target_note_item":"exact target-note item text","clinical_impact":"minor|major|potentially_harmful","rationale":"why unsupported"}}
  ],
  "section_leakage":[
    {{"cap_id":"C1","source_section":"section enum","target_section":"section enum","clinical_impact":"minor|major|potentially_harmful"}}
  ],
  "medication_exactness":"preserved|changed|missing|uncertain|not_applicable",
  "allergy_exactness":"preserved|changed|missing|uncertain|not_applicable",
  "plan_exactness":"preserved|changed|missing|uncertain|not_applicable",
  "final_state_status":"preserved|changed|missing|uncertain|not_applicable",
  "chronology_status":"preserved|changed|missing|uncertain|not_applicable",
  "conversation_assessment":"no_major_error|major_omission|major_contradiction|major_unsupported_addition|uncertain"
}}
Emit exactly one CAP adjudication for every frozen cap_id and no others.
"""

# Used only after a complete model judgment fails the unchanged semantic
# validator. The model must resolve the inconsistency from evidence rather than
# being forced toward a favorable outcome. Reusing only the latest invalid JSON
# avoids accumulating multiple ~85k-character failed responses in context.
SCHEMA_REPAIR_POLICY_V2 = "evidence_preserving_consistency_repair_v2"
SCHEMA_REPAIR_PROMPT_V2 = """Repair and return the complete JSON after this validation error:
{error}

Keep every judgment that is already valid and keep every note-evidence string
exact. For each CAP implicated by the error, resolve the inconsistency from the
supplied transcripts, notes, and CAP evidence—not merely to satisfy validation:

- target_note_status=\"preserved\" is valid only when modifier_errors=[] and
  clinical_impact=\"none\";
- if the evidence truly supports a modifier error or non-none impact, retain
  that finding and choose the evidence-supported non-preserved status
  (partially_preserved, contradicted, missing, or uncertain);
- if the evidence supports full preservation, keep preserved and remove only
  the unsupported error/impact;
- never erase a genuine error, invent evidence, or change an unrelated CAP.

Emit the entire strict JSON object, not a patch or explanation."""

# V3 retains the complete direction-neutral v2 contract and additionally makes
# the already-frozen role/impact enums explicit. This addresses invalid enum
# spellings without choosing or normalizing the scientific judgment.
SCHEMA_REPAIR_POLICY = "evidence_preserving_consistency_repair_v3"
SCHEMA_REPAIR_PROMPT = SCHEMA_REPAIR_PROMPT_V2.replace(
    "- target_note_status=\"preserved\" is valid only when modifier_errors=[] and",
    """- role_status must be exactly one of: preserved, changed, not_applicable,
  uncertain; clinical_impact must be exactly one of: none, minor, major,
  potentially_harmful, uncertain. Choose only the evidence-supported enum;
- target_note_status=\"preserved\" is valid only when modifier_errors=[] and""",
)


def repair_messages(
    base_messages: list[dict[str, str]], result: dict[str, Any], error: Exception,
) -> list[dict[str, str]]:
    """Build one bounded evidence-preserving schema-repair request."""
    return [
        *base_messages,
        {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
        {"role": "user", "content": SCHEMA_REPAIR_PROMPT.format(error=error)},
    ]


def load_env(path: Path | None) -> None:
    if not path or not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def note_items(note: dict[str, Any]) -> dict[str, set[str]]:
    return {
        section: {str(item["text"]) for item in note.get(section) or []}
        for section in SECTIONS
    }


def validate(
    result: dict[str, Any], *, audio_id: str, caps: dict[str, dict[str, Any]],
    source_note: dict[str, Any], target_note: dict[str, Any],
) -> None:
    if result.get("audio_id") != audio_id:
        raise ValueError("audio_id mismatch")
    rows = result.get("cap_adjudications")
    if not isinstance(rows, list):
        raise ValueError("cap_adjudications must be a list")
    ids = [str(row.get("cap_id") or "") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(caps):
        raise ValueError("CAP bijection failed")
    source_items, target_items = note_items(source_note), note_items(target_note)
    for row in rows:
        cid = str(row["cap_id"])
        if row.get("source_note_status") not in SOURCE_STATUS or row.get("target_note_status") not in TARGET_STATUS:
            raise ValueError(f"{cid}: invalid source/target note status")
        source_section, target_section = row.get("source_section"), row.get("target_section")
        if source_section not in SECTION_ENUM or target_section not in SECTION_ENUM:
            raise ValueError(f"{cid}: invalid section")
        source_item, target_item = str(row.get("source_note_item") or ""), str(row.get("target_note_item") or "")
        if source_item and (source_section not in source_items or source_item not in source_items[source_section]):
            raise ValueError(f"{cid}: source note item is not exact section evidence")
        if target_item and (target_section not in target_items or target_item not in target_items[target_section]):
            raise ValueError(f"{cid}: target note item is not exact section evidence")
        if row["source_note_status"] == "included" and not source_item:
            raise ValueError(f"{cid}: included source status requires note evidence")
        if row["target_note_status"] in {"preserved", "partially_preserved", "contradicted"} and not target_item:
            raise ValueError(f"{cid}: target status requires note evidence")
        modifiers = row.get("modifier_errors")
        if not isinstance(modifiers, list) or not set(modifiers).issubset(MODIFIERS):
            raise ValueError(f"{cid}: invalid modifier errors")
        if row.get("role_status") not in ROLE_STATUS or row.get("clinical_impact") not in IMPACTS:
            raise ValueError(f"{cid}: invalid role/impact")
        if row["target_note_status"] == "preserved" and (modifiers or row["clinical_impact"] != "none"):
            raise ValueError(f"{cid}: preserved target status conflicts with errors")
        if not str(row.get("rationale") or "").strip():
            raise ValueError(f"{cid}: empty rationale")
    unsupported = result.get("unsupported_target_note_assertions")
    if not isinstance(unsupported, list):
        raise ValueError("unsupported_target_note_assertions must be a list")
    for item in unsupported:
        section = item.get("target_section"); text = str(item.get("target_note_item") or "")
        if section not in target_items or text not in target_items[section]:
            raise ValueError("unsupported assertion lacks exact target-note evidence")
        if item.get("clinical_impact") not in {"minor", "major", "potentially_harmful"} or not str(item.get("rationale") or "").strip():
            raise ValueError("invalid unsupported target-note assertion")
    leakage = result.get("section_leakage")
    if not isinstance(leakage, list):
        raise ValueError("section_leakage must be a list")
    for item in leakage:
        if str(item.get("cap_id") or "") not in caps or item.get("source_section") not in SECTION_ENUM or item.get("target_section") not in SECTION_ENUM:
            raise ValueError("invalid section leakage row")
        if item.get("clinical_impact") not in {"minor", "major", "potentially_harmful"}:
            raise ValueError("invalid section leakage impact")
    for field in ("medication_exactness", "allergy_exactness", "plan_exactness", "final_state_status", "chronology_status"):
        if result.get(field) not in EXACTNESS:
            raise ValueError(f"invalid {field}")
    if result.get("conversation_assessment") not in ASSESSMENTS:
        raise ValueError("invalid conversation assessment")


def call(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    if args.backend == "openai_compat-bearer":
        token = os.environ.get("OPENAI_API_KEY", "")
        if not token:
            raise RuntimeError("missing OPENAI_API_KEY")
        response = call_openai_compat_bearer_chat(
            base_url=args.base_url, bearer_token=token, model=args.model, messages=messages,
            temperature=0.0, max_tokens=args.max_tokens, timeout=args.timeout,
            verify_ssl=not args.no_verify_ssl, api_version=None, omit_api_version=False,
            service_name=None, completion_token_param="max_completion_tokens",
        )
    else:
        response = call_chat_completions(
            base_url=args.base_url, api_key=args.api_key or "not-required", model=args.model,
            messages=messages, temperature=0.0, timeout=args.timeout, max_retries=2,
            verify_ssl=not args.no_verify_ssl, max_tokens=args.max_tokens,
        )
    return safe_parse_model_json(extract_json_text_chat(response))


def evaluate(pair: dict[str, Any], caps: list[dict[str, Any]], source_note: dict[str, Any], target_note: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            audio_id=pair["audio_id"], dataset_id=pair.get("source_dataset_id") or "unknown",
            language=pair.get("source_language") or "unknown", caps=json.dumps(caps, ensure_ascii=False),
            source_transcript=pair["reference"], target_transcript=pair["hypothesis"],
            source_note=json.dumps(source_note, ensure_ascii=False), target_note=json.dumps(target_note, ensure_ascii=False),
        )},
    ]
    caps_by_id = {str(row["cap_id"]): row for row in caps}
    result = call(args, base_messages)
    for attempt in range(args.schema_retries + 1):
        try:
            validate(result, audio_id=pair["audio_id"], caps=caps_by_id, source_note=source_note, target_note=target_note)
            return result
        except Exception as exc:
            if attempt >= args.schema_retries:
                raise
            result = call(args, repair_messages(base_messages, result, exc))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-transcripts", type=Path, required=True)
    parser.add_argument("--source-caps", type=Path, required=True)
    parser.add_argument("--source-notes", type=Path, required=True)
    parser.add_argument("--target-notes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["openai_compat-bearer", "openai-compatible"], required=True)
    parser.add_argument("--model", required=True); parser.add_argument("--base-url", required=True)
    parser.add_argument("--env-file", type=Path); parser.add_argument("--api-key", default="")
    # The largest frozen exact-CAP rows emit roughly 85k characters of JSON.
    # A 20k-token cap truncated those otherwise valid responses. Keep enough
    # output headroom while preserving the same prompt, evidence, and strict
    # validator.
    parser.add_argument("--max-tokens", type=int, default=40000); parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--schema-retries", type=int, default=8)
    parser.add_argument("--limit", type=int); parser.add_argument("--resume", action="store_true"); parser.add_argument("--no-verify-ssl", action="store_true")
    args = parser.parse_args(); load_env(args.env_file)
    # Paired transcripts have a strict two-row schema (reference/hypothesis).
    # CAP inventories, generated notes, resumable judgments, and errors are
    # generic JSONL artifacts and must not be passed through that schema.
    pairs = group_pairs(read_paired_jsonl(args.paired_transcripts)); pairs = pairs[:args.limit] if args.limit is not None else pairs
    caps_by_audio = {str(row["audio_id"]): row["caps"] for row in read_jsonl(args.source_caps)}
    source_notes = {str(row["audio_id"]): row["result"]["note"] for row in read_jsonl(args.source_notes)}
    target_notes = {str(row["audio_id"]): row["result"]["note"] for row in read_jsonl(args.target_notes)}
    population = {pair["audio_id"] for pair in pairs}
    if population != set(caps_by_audio) or population != set(source_notes) or population != set(target_notes):
        raise ValueError("CAP/note/transcript populations differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".note_pair_adjudication.lock").open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another note adjudicator owns this output") from exc
    output = args.output_dir / "note_pair_adjudications.jsonl"; errors_path = args.output_dir / "errors.jsonl"; completed = {}
    if args.resume and output.exists():
        for row in read_jsonl(output):
            aid = str(row.get("audio_id") or "")
            if aid in population:
                try:
                    validate(row["result"], audio_id=aid, caps={str(x["cap_id"]): x for x in caps_by_audio[aid]}, source_note=source_notes[aid], target_note=target_notes[aid])
                    completed[aid] = row
                except ValueError:
                    pass
    pending = [pair for pair in pairs if pair["audio_id"] not in completed]; errors = []
    with output.open("a") as out, errors_path.open("a") as err, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluate, pair, caps_by_audio[pair["audio_id"]], source_notes[pair["audio_id"]], target_notes[pair["audio_id"]], args): pair for pair in pending}
        for future in as_completed(futures):
            pair = futures[future]
            try:
                result = future.result(); item = {"audio_id": pair["audio_id"], "dataset_id": pair.get("source_dataset_id"), "language": pair.get("source_language"), "result": result}; completed[pair["audio_id"]] = item; out.write(json.dumps(item, ensure_ascii=False) + "\n"); out.flush()
            except Exception as exc:
                item = {"audio_id": pair["audio_id"], "error_type": type(exc).__name__, "error": str(exc)[:4000]}; errors.append(item); err.write(json.dumps(item, ensure_ascii=False) + "\n"); err.flush()
            print(json.dumps({"completed": len(completed), "errors": len(errors), "total": len(pairs)}), flush=True)
    with output.open("w") as out:
        for pair in pairs:
            if pair["audio_id"] in completed:
                out.write(json.dumps(completed[pair["audio_id"]], ensure_ascii=False) + "\n")

    section_counts: dict[str, Counter[str]] = defaultdict(Counter); modifier_errors = Counter(); unsupported_impacts = Counter(); assessments = Counter(); exactness = {key: Counter() for key in ("medication", "allergy", "plan", "final_state", "chronology")}; leakage_count = 0
    structural_jaccards = []; exact_structure = 0; source_words = 0; target_words = 0
    for aid, item in completed.items():
        result = item["result"]; assessments[result["conversation_assessment"]] += 1; leakage_count += len(result["section_leakage"])
        unsupported_impacts.update(row["clinical_impact"] for row in result["unsupported_target_note_assertions"])
        for field, key in (("medication_exactness", "medication"), ("allergy_exactness", "allergy"), ("plan_exactness", "plan"), ("final_state_status", "final_state"), ("chronology_status", "chronology")):
            exactness[key][result[field]] += 1
        for row in result["cap_adjudications"]:
            section = str(row["source_section"])
            section_counts[section][row["target_note_status"]] += 1
            modifier_errors.update(row["modifier_errors"])
        source_occupied={section for section,items in source_notes[aid].items() if items}; target_occupied={section for section,items in target_notes[aid].items() if items}
        union=source_occupied|target_occupied; structural_jaccards.append(len(source_occupied&target_occupied)/len(union) if union else 1.0); exact_structure += int(source_occupied==target_occupied)
        source_words += sum(len(str(row["text"]).split()) for items in source_notes[aid].values() for row in items)
        target_words += sum(len(str(row["text"]).split()) for items in target_notes[aid].values() for row in items)
    section_report = {}
    for section, counts in sorted(section_counts.items()):
        n = sum(counts.values()); section_report[section] = {"total": n, "status_counts": dict(counts), "strict_cap_recall": counts["preserved"] / n if n else None, "lenient_cap_recall": (counts["preserved"] + counts["partially_preserved"]) / n if n else None}
    summary = {
        "schema_version": "aegisvox.clinical_note_pair_adjudication.v1", "judge_model": args.model, "judge_backend": args.backend,
        "num_rows": len(pairs), "num_completed": len(completed), "num_errors": len(errors), "section_results": section_report,
        "unsupported_note_assertion_count": sum(unsupported_impacts.values()), "unsupported_note_assertion_impacts": dict(unsupported_impacts),
        "modifier_error_counts": dict(modifier_errors), "section_leakage_count": leakage_count,
        "exactness_counts": {key: dict(value) for key, value in exactness.items()}, "conversation_assessment_counts": dict(assessments),
        "note_no_major_error_rate": assessments["no_major_error"] / len(completed) if completed else None,
        "mean_structural_section_jaccard": sum(structural_jaccards) / len(structural_jaccards) if structural_jaccards else None,
        "exact_structural_section_agreement_rate": exact_structure / len(completed) if completed else None,
        "source_note_word_count": source_words, "target_note_word_count": target_words,
        "target_source_note_compression_ratio": target_words / source_words if source_words else None,
        "input_sha256": hashlib.sha256(args.paired_transcripts.read_bytes()).hexdigest(), "source_caps_sha256": hashlib.sha256(args.source_caps.read_bytes()).hexdigest(),
        "source_notes_sha256": hashlib.sha256(args.source_notes.read_bytes()).hexdigest(), "target_notes_sha256": hashlib.sha256(args.target_notes.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "schema_repair_policy": SCHEMA_REPAIR_POLICY,
        "schema_repair_prompt_sha256": hashlib.sha256(SCHEMA_REPAIR_PROMPT.encode()).hexdigest(),
        "schema_repair_policy_history": [
            "generic_complete_json_repair_v1",
            SCHEMA_REPAIR_POLICY_V2,
            SCHEMA_REPAIR_POLICY,
        ],
        "schema_repair_scope": "invalid model outputs only; every accepted row satisfies the unchanged strict validator",
        "full_transcript_used": True, "deterministic_proxy_used": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2)); return 0 if len(completed) == len(pairs) and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
