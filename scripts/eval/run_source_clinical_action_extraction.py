#!/usr/bin/env python3
"""Extract source-supported clinical actions and decisions with exact evidence."""
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

SYSTEM_PROMPT = """You are a conservative source-only clinical action extractor.

Read the complete source conversation and extract only explicitly supported
clinical actions or decisions: tests ordered/considered, medications started,
stopped, continued or changed, procedures/referrals, follow-up intervals,
return precautions/escalation triggers, and explicit assessment/differential
statements. Include conditional and declined actions. Do not convert medication
history, prior procedures, symptoms, or world knowledge into a current action.
Every action needs one unique exact contiguous source quote. Return strict JSON.
"""

USER_TEMPLATE = """audio_id: {audio_id}
dataset_id: {dataset_id}
language: {language}

COMPLETE SOURCE TRANSCRIPT:
{source}

Return:
{{
  "audio_id": "{audio_id}",
  "actions": [
    {{
      "action_id": "A1",
      "action_kind": "test|medication|procedure|referral|followup|return_precaution|escalation|assessment",
      "action_state": "ordered|considered|recommended|started|stopped|continued|changed|completed|conditional|declined|reported_existing",
      "action_text": "one atomic source-supported action or decision",
      "source_quote": "unique exact contiguous source quote",
      "actor_role": "patient|clinician|dispatcher|caller|caregiver|third_party|unknown",
      "target_role": "patient|clinician|caller|caregiver|third_party|not_applicable|unknown",
      "clinical_importance": "critical|major|minor"
    }}
  ]
}}

An empty list is allowed only when the source contains no explicit action or
decision. Atomize independent actions and preserve conditionality and final
start/stop/continue state.
"""

ENUMS = {
    "action_kind": {"test", "medication", "procedure", "referral", "followup", "return_precaution", "escalation", "assessment"},
    "action_state": {"ordered", "considered", "recommended", "started", "stopped", "continued", "changed", "completed", "conditional", "declined", "reported_existing"},
    "actor_role": {"patient", "clinician", "dispatcher", "caller", "caregiver", "third_party", "unknown"},
    "target_role": {"patient", "clinician", "caller", "caregiver", "third_party", "not_applicable", "unknown"},
    "clinical_importance": {"critical", "major", "minor"},
}


def load_env(path: Path | None) -> None:
    if not path or not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read this runner's own result rows without applying paired-ASR validation."""
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


def validate(result: dict[str, Any], *, audio_id: str, source: str) -> None:
    if result.get("audio_id") != audio_id:
        raise ValueError("audio_id mismatch")
    actions = result.get("actions")
    if not isinstance(actions, list):
        raise ValueError("actions must be a list")
    ids = []
    for action in actions:
        action_id = str(action.get("action_id") or "")
        ids.append(action_id)
        if not action_id or not str(action.get("action_text") or "").strip():
            raise ValueError("empty action_id/action_text")
        quote = str(action.get("source_quote") or "")
        if len(quote) < 6 or source.count(quote) != 1:
            raise ValueError(f"{action_id}: source_quote must uniquely locate exact evidence")
        for field, allowed in ENUMS.items():
            if action.get(field) not in allowed:
                raise ValueError(f"{action_id}: invalid {field}={action.get(field)!r}")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate action_id")


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


def evaluate(pair: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            audio_id=pair["audio_id"], dataset_id=pair.get("source_dataset_id") or "unknown",
            language=pair.get("source_language") or "unknown", source=pair["reference"],
        )},
    ]
    result = call(args, messages)
    for attempt in range(args.schema_retries + 1):
        try:
            validate(result, audio_id=pair["audio_id"], source=pair["reference"])
            return result
        except Exception as exc:
            if attempt >= args.schema_retries:
                raise
            messages += [
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                {"role": "user", "content": f"Repair the complete JSON after this validation error: {exc}. Keep valid actions and copy exact source evidence."},
            ]
            result = call(args, messages)
    raise AssertionError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired-transcripts", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--backend", choices=["openai_compat-bearer", "openai-compatible"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--env-file", type=Path)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--schema-retries", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-verify-ssl", action="store_true")
    args = ap.parse_args(); load_env(args.env_file)
    pairs = group_pairs(read_paired_jsonl(args.paired_transcripts))
    if args.limit is not None:
        pairs = pairs[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".source_actions.lock").open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another source action process owns this output") from exc
    output = args.output_dir / "source_actions.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    completed: dict[str, dict] = {}
    if args.resume and output.exists():
        for row in read_raw_jsonl(output):
            pair = next((p for p in pairs if p["audio_id"] == row.get("audio_id")), None)
            if pair:
                try:
                    validate(row["result"], audio_id=pair["audio_id"], source=pair["reference"])
                    completed[pair["audio_id"]] = row
                except ValueError:
                    pass
    pending = [pair for pair in pairs if pair["audio_id"] not in completed]
    errors = []
    with output.open("a") as out, errors_path.open("a") as err, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluate, pair, args): pair for pair in pending}
        for future in as_completed(futures):
            pair = futures[future]
            try:
                result = future.result()
                item = {"audio_id": pair["audio_id"], "dataset_id": pair.get("source_dataset_id"), "language": pair.get("source_language"), "result": result}
                completed[pair["audio_id"]] = item
                out.write(json.dumps(item, ensure_ascii=False) + "\n"); out.flush()
            except Exception as exc:
                item = {"audio_id": pair["audio_id"], "error_type": type(exc).__name__, "error": str(exc)[:4000]}
                errors.append(item); err.write(json.dumps(item, ensure_ascii=False) + "\n"); err.flush()
            print(json.dumps({"completed": len(completed), "errors": len(errors), "total": len(pairs)}), flush=True)
    with output.open("w") as out:
        for pair in pairs:
            if pair["audio_id"] in completed:
                out.write(json.dumps(completed[pair["audio_id"]], ensure_ascii=False) + "\n")
    actions = [a for row in completed.values() for a in row["result"]["actions"]]
    summary = {
        "schema_version": "aegisvox.source_clinical_actions.v1", "model": args.model,
        "num_rows": len(pairs), "num_completed": len(completed), "num_errors": len(errors),
        "total_actions": len(actions),
        "action_kind_counts": dict(Counter(a["action_kind"] for a in actions)),
        "action_state_counts": dict(Counter(a["action_state"] for a in actions)),
        "input_sha256": hashlib.sha256(args.paired_transcripts.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "deterministic_proxy_used": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2)); return 0 if len(completed) == len(pairs) and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
