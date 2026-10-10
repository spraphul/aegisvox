#!/usr/bin/env python3
"""Adjudicate frozen source actions against complete transformed transcripts."""
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_gpt_concept_preservation import (  # noqa: E402
    call_openai_compat_bearer_chat, call_chat_completions, extract_json_text_chat,
    group_pairs, read_jsonl as read_paired_jsonl, safe_parse_model_json,
)

SYSTEM_PROMPT = """You are a conservative source-anchored clinical action adjudicator.

The source action inventory was frozen before target inspection. Compare only
the supplied complete source and target transcripts. Judge whether each action,
state, actor/target role, condition, timing and final decision is preserved.
Identify clinically meaningful unsupported target actions. Do not recommend the
medically correct action from world knowledge. Every non-empty target quote and
unsupported-action quote must be an exact contiguous target substring. Return
strict JSON only. Safe identifier replacement is not an action error.
"""

USER_TEMPLATE = """audio_id: {audio_id}
dataset_id: {dataset_id}
language: {language}

FROZEN SOURCE ACTIONS:
{actions}

COMPLETE SOURCE TRANSCRIPT:
{source}

COMPLETE TARGET TRANSCRIPT (same evaluator ASR):
{target}

Return:
{{
  "audio_id": "{audio_id}",
  "adjudications": [
    {{
      "action_id": "A1",
      "status": "preserved|partially_preserved|missing|contradicted|uncertain",
      "target_quote": "exact target quote or empty",
      "state_status": "preserved|changed|missing|uncertain",
      "role_status": "preserved|changed|not_applicable|uncertain",
      "unsafe_change": "none|wrong_action|wrong_medication_or_dose|wrong_timing_or_followup|missed_return_precaution|reversed_start_stop_continue|other|uncertain",
      "clinical_impact": "none|minor|major|potentially_harmful|uncertain",
      "rationale": "brief source-target rationale"
    }}
  ],
  "unsupported_target_actions": [
    {{
      "target_quote": "exact target quote",
      "action_text": "unsupported target action",
      "clinical_impact": "minor|major|potentially_harmful",
      "rationale": "why not supported by source"
    }}
  ],
  "conversation_plan_assessment": "invariant|minor_change|major_change|potentially_harmful_change|insufficient_evidence|not_applicable"
}}
Emit exactly one adjudication per supplied action_id and no others.
"""

STATUSES = {"preserved", "partially_preserved", "missing", "contradicted", "uncertain"}
STATE = {"preserved", "changed", "missing", "uncertain"}
ROLE = {"preserved", "changed", "not_applicable", "uncertain"}
UNSAFE = {"none", "wrong_action", "wrong_medication_or_dose", "wrong_timing_or_followup", "missed_return_precaution", "reversed_start_stop_continue", "other", "uncertain"}
IMPACT = {"none", "minor", "major", "potentially_harmful", "uncertain"}
ASSESS = {"invariant", "minor_change", "major_change", "potentially_harmful_change", "insufficient_evidence", "not_applicable"}


def load_env(path: Path | None) -> None:
    if not path or not path.is_file(): return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1); os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read source/result rows without applying the paired-transcript schema."""
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"invalid JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def validate_source_population(
    pair_ids: set[str], source_ids: set[str], *, allow_source_superset: bool
) -> set[str]:
    """Return extra frozen anchors only when the caller explicitly permits them."""
    missing_source_ids = pair_ids - source_ids
    extra_source_ids = source_ids - pair_ids
    if missing_source_ids:
        raise ValueError("target pairs lack frozen source actions")
    if extra_source_ids and not allow_source_superset:
        raise ValueError("source action population does not match target pairs")
    return extra_source_ids


def validate(result: dict[str, Any], *, audio_id: str, source_actions: dict[str, dict], target: str) -> None:
    if result.get("audio_id") != audio_id: raise ValueError("audio_id mismatch")
    rows = result.get("adjudications")
    if not isinstance(rows, list): raise ValueError("adjudications must be a list")
    ids = [str(row.get("action_id") or "") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(source_actions): raise ValueError("action bijection failed")
    for row in rows:
        aid = str(row["action_id"]); quote = str(row.get("target_quote") or "")
        if quote and quote not in target:
            raise ValueError(f"{aid}: target_quote is not exact evidence: {quote!r}")
        if row.get("status") not in STATUSES: raise ValueError(f"{aid}: invalid status")
        if row["status"] in {"preserved", "partially_preserved", "contradicted"} and not quote:
            raise ValueError(f"{aid}: status requires target evidence")
        if row.get("state_status") not in STATE or row.get("role_status") not in ROLE:
            raise ValueError(f"{aid}: invalid state/role status")
        if row.get("unsafe_change") not in UNSAFE or row.get("clinical_impact") not in IMPACT:
            raise ValueError(f"{aid}: invalid unsafe/impact")
        if row["status"] == "preserved" and (row["unsafe_change"] != "none" or row["clinical_impact"] != "none"):
            raise ValueError(f"{aid}: preserved conflicts with unsafe/impact")
        if row["status"] == "preserved" and row["state_status"] != "preserved":
            raise ValueError(f"{aid}: preserved action requires preserved state")
        if row["status"] == "missing" and quote:
            raise ValueError(f"{aid}: missing action cannot cite target evidence")
        if not str(row.get("rationale") or "").strip(): raise ValueError(f"{aid}: empty rationale")
    unsupported = result.get("unsupported_target_actions")
    if not isinstance(unsupported, list): raise ValueError("unsupported_target_actions must be a list")
    for item in unsupported:
        quote = str(item.get("target_quote") or "")
        if not quote or quote not in target:
            raise ValueError(f"unsupported action lacks exact target evidence: {quote!r}")
        if item.get("clinical_impact") not in {"minor", "major", "potentially_harmful"}:
            raise ValueError("invalid unsupported action impact")
        if not str(item.get("action_text") or "").strip() or not str(item.get("rationale") or "").strip():
            raise ValueError("unsupported action text/rationale missing")
    if result.get("conversation_plan_assessment") not in ASSESS: raise ValueError("invalid plan assessment")
    assessment = result["conversation_plan_assessment"]
    if not rows and not unsupported and assessment != "not_applicable":
        raise ValueError("empty action case must be not_applicable")
    if rows and all(row["status"] == "preserved" for row in rows) and not unsupported and assessment != "invariant":
        raise ValueError("all-preserved action case must be invariant")
    if assessment == "invariant" and (unsupported or any(row["status"] != "preserved" for row in rows)):
        raise ValueError("invariant assessment conflicts with action judgments")


def call(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    if args.backend == "openai_compat-bearer":
        token = os.environ.get("OPENAI_API_KEY", "")
        if not token: raise RuntimeError("missing OPENAI_API_KEY")
        response = call_openai_compat_bearer_chat(base_url=args.base_url, bearer_token=token, model=args.model, messages=messages, temperature=0.0, max_tokens=args.max_tokens, timeout=args.timeout, verify_ssl=not args.no_verify_ssl, api_version=None, omit_api_version=False, service_name=None, completion_token_param="max_completion_tokens")
    else:
        response = call_chat_completions(base_url=args.base_url, api_key=args.api_key or "not-required", model=args.model, messages=messages, temperature=0.0, timeout=args.timeout, max_retries=2, verify_ssl=not args.no_verify_ssl, max_tokens=args.max_tokens)
    return safe_parse_model_json(extract_json_text_chat(response))


def evaluate(pair: dict[str, Any], source_item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    actions = source_item["result"]["actions"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_TEMPLATE.format(audio_id=pair["audio_id"], dataset_id=pair.get("source_dataset_id") or "unknown", language=pair.get("source_language") or "unknown", actions=json.dumps(actions, ensure_ascii=False), source=pair["reference"], target=pair["hypothesis"])}]
    result = call(args, messages); source_by_id = {str(row["action_id"]): row for row in actions}
    for attempt in range(args.schema_retries + 1):
        try:
            validate(result, audio_id=pair["audio_id"], source_actions=source_by_id, target=pair["hypothesis"]); return result
        except Exception as exc:
            if attempt >= args.schema_retries: raise
            messages += [{"role": "assistant", "content": json.dumps(result, ensure_ascii=False)}, {"role": "user", "content": f"Repair the complete JSON after this validation error: {exc}. Preserve valid judgments and exact target evidence."}]
            result = call(args, messages)
    raise AssertionError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired-transcripts", type=Path, required=True); ap.add_argument("--source-actions", type=Path, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--backend", choices=["openai_compat-bearer", "openai-compatible"], required=True); ap.add_argument("--model", required=True); ap.add_argument("--base-url", required=True); ap.add_argument("--env-file", type=Path); ap.add_argument("--api-key", default="")
    ap.add_argument("--max-tokens", type=int, default=18000); ap.add_argument("--timeout", type=float, default=300); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--schema-retries", type=int, default=8); ap.add_argument("--limit", type=int); ap.add_argument("--resume", action="store_true"); ap.add_argument("--no-verify-ssl", action="store_true"); ap.add_argument("--allow-source-superset", action="store_true")
    args = ap.parse_args(); load_env(args.env_file)
    pairs = group_pairs(read_paired_jsonl(args.paired_transcripts)); pairs = pairs[:args.limit] if args.limit is not None else pairs
    sources = {str(row["audio_id"]): row for row in read_raw_jsonl(args.source_actions)}
    pair_ids = {p["audio_id"] for p in pairs}; source_ids = set(sources)
    try:
        extra_source_ids = validate_source_population(
            pair_ids, source_ids, allow_source_superset=args.allow_source_superset
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output_dir.mkdir(parents=True, exist_ok=True); lock = (args.output_dir / ".target_actions.lock").open("w")
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc: raise RuntimeError("another target action process owns this output") from exc
    output = args.output_dir / "target_actions.jsonl"; errors_path = args.output_dir / "errors.jsonl"; completed = {}
    if args.resume and output.exists():
        for row in read_raw_jsonl(output):
            pair = next((p for p in pairs if p["audio_id"] == row.get("audio_id")), None)
            if pair:
                try:
                    validate(row["result"], audio_id=pair["audio_id"], source_actions={str(x["action_id"]): x for x in sources[pair["audio_id"]]["result"]["actions"]}, target=pair["hypothesis"]); completed[pair["audio_id"]] = row
                except ValueError: pass
    pending = [p for p in pairs if p["audio_id"] not in completed]; errors = []
    with output.open("a") as out, errors_path.open("a") as err, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluate, p, sources[p["audio_id"]], args): p for p in pending}
        for future in as_completed(futures):
            pair = futures[future]
            try:
                result = future.result(); item = {"audio_id": pair["audio_id"], "dataset_id": pair.get("source_dataset_id"), "language": pair.get("source_language"), "result": result}; completed[pair["audio_id"]] = item; out.write(json.dumps(item, ensure_ascii=False) + "\n"); out.flush()
            except Exception as exc:
                item = {"audio_id": pair["audio_id"], "error_type": type(exc).__name__, "error": str(exc)[:4000]}; errors.append(item); err.write(json.dumps(item, ensure_ascii=False) + "\n"); err.flush()
            print(json.dumps({"completed": len(completed), "errors": len(errors), "total": len(pairs)}), flush=True)
    with output.open("w") as out:
        for pair in pairs:
            if pair["audio_id"] in completed: out.write(json.dumps(completed[pair["audio_id"]], ensure_ascii=False) + "\n")
    statuses = Counter(); unsafe = Counter(); impacts = Counter(); assessments = Counter(); unsupported_impacts = Counter(); state_statuses = Counter(); role_statuses = Counter(); action_kinds: dict[str, Counter[str]] = {}
    for item in completed.values():
        assessments[item["result"]["conversation_plan_assessment"]] += 1
        unsupported_impacts.update(row["clinical_impact"] for row in item["result"]["unsupported_target_actions"])
        source_by_id={str(row["action_id"]):row for row in sources[item["audio_id"]]["result"]["actions"]}
        for row in item["result"]["adjudications"]:
            statuses[row["status"]] += 1; unsafe[row["unsafe_change"]] += 1; impacts[row["clinical_impact"]] += 1; state_statuses[row["state_status"]] += 1; role_statuses[row["role_status"]] += 1
            kind=str(source_by_id[str(row["action_id"])]["action_kind"]); action_kinds.setdefault(kind,Counter())[row["status"]] += 1
    total = sum(statuses.values()); strict = statuses["preserved"]; lenient = strict + statuses["partially_preserved"]
    kind_report={}
    for kind,counts in sorted(action_kinds.items()):
        n=sum(counts.values()); kind_report[kind]={"total":n,"status_counts":dict(counts),"strict_action_recall":counts["preserved"]/n if n else None,"lenient_action_recall":(counts["preserved"]+counts["partially_preserved"])/n if n else None}
    summary = {"schema_version": "aegisvox.target_clinical_actions.v2", "model": args.model, "num_rows": len(pairs), "num_completed": len(completed), "num_errors": len(errors), "num_frozen_source_rows": len(source_ids), "num_source_rows_excluded_without_target_pair": len(extra_source_ids), "source_superset_allowed": bool(args.allow_source_superset), "total_actions": total, "status_counts": dict(statuses), "strict_action_recall": strict / total if total else None, "lenient_action_recall": lenient / total if total else None, "state_status_counts":dict(state_statuses),"role_status_counts":dict(role_statuses),"action_kind_results":kind_report,"unsafe_change_counts": dict(unsafe), "potentially_harmful_action_change_rate": impacts["potentially_harmful"] / total if total else None, "unsupported_target_action_count": sum(unsupported_impacts.values()),"unsupported_target_action_impacts":dict(unsupported_impacts), "conversation_plan_assessment_counts": dict(assessments),"conversation_plan_invariance_rate":assessments["invariant"]/len(completed) if completed else None,"conversation_major_or_harmful_change_rate":(assessments["major_change"]+assessments["potentially_harmful_change"])/len(completed) if completed else None, "input_sha256": hashlib.sha256(args.paired_transcripts.read_bytes()).hexdigest(), "source_actions_sha256": hashlib.sha256(args.source_actions.read_bytes()).hexdigest(), "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), "deterministic_proxy_used": False}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); print(json.dumps(summary, indent=2)); return 0 if len(completed) == len(pairs) and not errors else 2


if __name__ == "__main__": raise SystemExit(main())
