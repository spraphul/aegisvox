#!/usr/bin/env python3
"""Adjudicate frozen source CAPs against complete transformed-audio transcripts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gpt_concept_preservation import (  # noqa: E402
    call_openai_compat_bearer_chat,
    call_chat_completions,
    extract_json_text_chat,
    group_pairs,
    read_jsonl as read_paired_jsonl,
    safe_parse_model_json,
)
from run_clinical_cap_extraction import literalize_quote  # noqa: E402


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-paired JSONL such as the frozen source-CAP inventory.

    The imported paired reader deliberately validates the paired-transcript
    contract and must not be reused for CAP rows.
    """
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


SYSTEM_PROMPT = """You are a conservative clinical source-to-target integrity adjudicator.

The source clinical atomic propositions (CAPs) were frozen before you saw the
target. For every CAP, compare the complete source and target transcripts and
judge preservation without inventing medical facts. Preserve exact modifiers:
negation, uncertainty, temporality, dose/value, severity, verification state,
action type, final corrected state, and speaker role. A safe synthetic identity
replacement is not a clinical error unless it changes meaning, chronology, or
role. Detect material unsupported target assertions separately. Every evidence
quote must be a literal contiguous target-transcript substring. Return strict
JSON only.
"""


USER_TEMPLATE = """audio_id: {audio_id}
dataset_id: {dataset_id}
language: {language}

FROZEN SOURCE CAPS:
{caps}

COMPLETE SOURCE TRANSCRIPT:
{source}

COMPLETE TARGET TRANSCRIPT (same frozen evaluator ASR):
{target}

Return:
{{
  "audio_id": "{audio_id}",
  "adjudications": [
    {{
      "cap_id": "C1",
      "status": "preserved|partially_preserved|missing|contradicted|uncertain",
      "target_quote": "exact target quote or empty only for missing/uncertain",
      "role_status": "preserved|changed|uncertain|not_applicable",
      "target_role": "patient|clinician|dispatcher|caller|caregiver|third_party|unknown",
      "modifier_errors": ["negation|temporality|dose_value|severity|verification|action_type|final_state"],
      "clinical_impact": "none|minor|major|potentially_harmful|uncertain",
      "rationale": "brief evidence-grounded reason"
    }}
  ],
  "unsupported_target_assertions": [
    {{
      "assertion": "one material unsupported target fact",
      "target_quote": "exact target quote",
      "clinical_impact": "minor|major|potentially_harmful|uncertain",
      "source_search_rationale": "why no source CAP/transcript evidence supports it"
    }}
  ],
  "conversation_assessment": "no_major_error|major_omission|major_contradiction|major_unsupported_addition|uncertain"
}}

Emit exactly one adjudication for every supplied cap_id and no others.
"""


STATUSES = {"preserved", "partially_preserved", "missing", "contradicted", "uncertain"}
ROLE_STATUSES = {"preserved", "changed", "uncertain", "not_applicable"}
TARGET_ROLES = {"patient", "clinician", "dispatcher", "caller", "caregiver", "third_party", "unknown"}
MODIFIERS = {"negation", "temporality", "dose_value", "severity", "verification", "action_type", "final_state"}
IMPACTS = {"none", "minor", "major", "potentially_harmful", "uncertain"}
ASSESSMENTS = {"no_major_error", "major_omission", "major_contradiction", "major_unsupported_addition", "uncertain"}


def load_env(path: Path | None) -> None:
    if not path or not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip("'").strip('"')


def exact_or_empty(transcript: str, quote: Any) -> bool:
    text = str(quote or "")
    return not text or text in transcript


def literalize_target_evidence(result: dict[str, Any], target: str) -> dict[str, Any]:
    """Copy unique token-identical model evidence from the literal target.

    This is evidence alignment only. Clinical statuses, roles, modifiers, and
    impacts are untouched. Ambiguous text, paraphrases, omissions, and typos
    remain invalid and are returned to the model's fail-closed repair path.
    """
    containers = (
        result.get("adjudications") or [],
        result.get("unsupported_target_assertions") or [],
    )
    for rows in containers:
        for row in rows:
            if not isinstance(row, dict):
                continue
            quote = str(row.get("target_quote") or "")
            if not quote or quote in target:
                continue
            aligned = literalize_quote(target, quote)
            if aligned is None:
                continue
            row["target_quote"] = aligned[0]
            row["target_quote_alignment_method"] = aligned[1]
    return result


def build_target_evidence_units(target: str) -> dict[str, str]:
    """Expose lossless target substrings for fail-closed evidence repair.

    The clinical judgment remains model-produced.  These units only let the
    model select a literal span identifier when it repeatedly paraphrases or
    normalizes punctuation while copying evidence.  The complete transcript is
    still supplied; no content is truncated or summarized.
    """
    pieces: list[str] = []
    for line in target.splitlines() or [target]:
        line = line.strip()
        if not line:
            continue
        sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", line) if piece.strip()]
        pieces.extend(sentences or [line])
    # An ASR transcript can be one punctuation-free line.  Split only for
    # addressability, at whitespace boundaries, while preserving every unit as
    # a literal contiguous target substring.
    units: list[str] = []
    for piece in pieces:
        if len(piece) <= 600:
            units.append(piece)
            continue
        cursor = 0
        while cursor < len(piece):
            end = min(len(piece), cursor + 600)
            if end < len(piece):
                boundary = piece.rfind(" ", cursor, end)
                if boundary > cursor:
                    end = boundary
            unit = piece[cursor:end].strip()
            if unit:
                units.append(unit)
            cursor = end
            while cursor < len(piece) and piece[cursor].isspace():
                cursor += 1
    return {f"T{index:04d}": unit for index, unit in enumerate(units, start=1)}


def resolve_target_evidence_ids(result: dict[str, Any], units: dict[str, str]) -> dict[str, Any]:
    for rows in (result.get("adjudications") or [], result.get("unsupported_target_assertions") or []):
        for row in rows:
            if not isinstance(row, dict):
                continue
            evidence_id = str(row.get("target_evidence_id") or "")
            if evidence_id in units:
                row["target_quote"] = units[evidence_id]
                row["target_quote_alignment_method"] = "model_selected_literal_unit"
    return result


def validate_result(result: dict[str, Any], *, audio_id: str, cap_ids: set[str], target: str) -> None:
    if result.get("audio_id") != audio_id:
        raise ValueError("audio_id mismatch")
    rows = result.get("adjudications")
    if not isinstance(rows, list):
        raise ValueError("adjudications must be a list")
    returned = [str(row.get("cap_id") or "") for row in rows if isinstance(row, dict)]
    if len(returned) != len(set(returned)) or set(returned) != cap_ids:
        raise ValueError(f"CAP bijection failed: missing={sorted(cap_ids-set(returned))} extra={sorted(set(returned)-cap_ids)}")
    for index, row in enumerate(rows):
        cap_id = str(row.get("cap_id"))
        status = row.get("status")
        if status not in STATUSES:
            raise ValueError(f"{cap_id}: invalid status")
        quote = str(row.get("target_quote") or "")
        if status in {"preserved", "partially_preserved", "contradicted"} and not quote:
            raise ValueError(f"{cap_id}: {status} requires target_quote")
        if not exact_or_empty(target, quote):
            raise ValueError(f"{cap_id}: target_quote is not a literal target substring")
        if row.get("role_status") not in ROLE_STATUSES:
            raise ValueError(f"{cap_id}: invalid role_status")
        if row.get("target_role") not in TARGET_ROLES:
            raise ValueError(f"{cap_id}: invalid target_role")
        modifiers = row.get("modifier_errors")
        if not isinstance(modifiers, list) or not set(modifiers).issubset(MODIFIERS):
            raise ValueError(f"{cap_id}: invalid modifier_errors")
        if row.get("clinical_impact") not in IMPACTS:
            raise ValueError(f"{cap_id}: invalid clinical_impact")
        if not str(row.get("rationale") or "").strip():
            raise ValueError(f"{cap_id}: empty rationale")
    additions = result.get("unsupported_target_assertions")
    if not isinstance(additions, list):
        raise ValueError("unsupported_target_assertions must be a list")
    for index, row in enumerate(additions):
        if not isinstance(row, dict) or not str(row.get("assertion") or "").strip():
            raise ValueError(f"unsupported_target_assertions[{index}] invalid")
        quote = str(row.get("target_quote") or "")
        if not quote or quote not in target:
            raise ValueError(f"unsupported_target_assertions[{index}] requires exact target_quote")
        if row.get("clinical_impact") not in IMPACTS - {"none"}:
            raise ValueError(f"unsupported_target_assertions[{index}] invalid clinical_impact")
        if not str(row.get("source_search_rationale") or "").strip():
            raise ValueError(f"unsupported_target_assertions[{index}] empty rationale")
    if result.get("conversation_assessment") not in ASSESSMENTS:
        raise ValueError("invalid conversation_assessment")


def call_model(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    if args.backend == "openai_compat-bearer":
        token = os.environ.get("OPENAI_API_KEY", "")
        if not token:
            raise RuntimeError("missing OPENAI_API_KEY")
        response = call_openai_compat_bearer_chat(
            base_url=args.base_url, bearer_token=token, model=args.model,
            messages=messages, temperature=0.0, max_tokens=args.max_tokens,
            timeout=args.timeout, verify_ssl=not args.no_verify_ssl,
            api_version=args.api_version, omit_api_version=args.omit_api_version,
            service_name=None, completion_token_param="max_completion_tokens",
        )
    else:
        response = call_chat_completions(
            base_url=args.base_url, api_key=args.api_key or "not-required", model=args.model,
            messages=messages, temperature=0.0, timeout=args.timeout, max_retries=2,
            verify_ssl=not args.no_verify_ssl, max_tokens=args.max_tokens,
        )
    return safe_parse_model_json(extract_json_text_chat(response))


def evaluate(pair: dict[str, Any], caps_item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    caps = caps_item["caps"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            audio_id=pair["audio_id"], dataset_id=pair.get("source_dataset_id") or "unknown",
            language=pair.get("source_language") or caps_item.get("language") or "unknown",
            caps=json.dumps(caps, ensure_ascii=False), source=pair["reference"], target=pair["hypothesis"],
        )},
    ]
    result = literalize_target_evidence(call_model(args, messages), pair["hypothesis"])
    cap_ids = {str(cap["cap_id"]) for cap in caps}
    evidence_units = build_target_evidence_units(pair["hypothesis"])
    for attempt in range(args.schema_retries + 1):
        try:
            validate_result(result, audio_id=pair["audio_id"], cap_ids=cap_ids, target=pair["hypothesis"])
            return result
        except Exception as exc:  # noqa: BLE001
            if attempt >= args.schema_retries:
                raise
            messages = [
                *messages,
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                {"role": "user", "content": (
                    f"The preceding object failed validation: {type(exc).__name__}: {exc}. "
                    "Repair only invalid fields and return the corrected complete JSON. Preserve all valid CAP judgments. "
                    "Evidence must be copied literally and contiguously from the complete target transcript. "
                    "For every object requiring nonempty target evidence, add target_evidence_id using exactly one ID "
                    "from TARGET_EVIDENCE_UNITS below; the program will copy that exact unit into target_quote. "
                    "Use an empty target_quote and omit target_evidence_id only where the schema permits missing/uncertain.\n\n"
                    "TARGET_EVIDENCE_UNITS:\n" + json.dumps(evidence_units, ensure_ascii=False)
                )},
            ]
            result = call_model(args, messages)
            result = resolve_target_evidence_ids(result, evidence_units)
            result = literalize_target_evidence(result, pair["hypothesis"])
    raise AssertionError("unreachable")


def item_metrics(result: dict[str, Any], caps_item: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(row["status"] for row in result["adjudications"])
    total = len(result["adjudications"])
    cap_by_id = {str(cap["cap_id"]): cap for cap in caps_item["caps"]}
    major_rows = [row for row in result["adjudications"] if cap_by_id[row["cap_id"]].get("clinical_importance") in {"critical", "major"}]
    major_errors = [row for row in result["adjudications"] if row["clinical_impact"] in {"major", "potentially_harmful"} and row["status"] not in {"preserved"}]
    major_additions = [row for row in result["unsupported_target_assertions"] if row["clinical_impact"] in {"major", "potentially_harmful"}]
    modifier_counts = Counter(value for row in result["adjudications"] for value in row["modifier_errors"])
    return {
        "num_caps": total,
        "preserved": counts["preserved"],
        "partially_preserved": counts["partially_preserved"],
        "missing": counts["missing"],
        "contradicted": counts["contradicted"],
        "uncertain": counts["uncertain"],
        "strict_preservation_rate": counts["preserved"] / total if total else None,
        "lenient_preservation_rate": (counts["preserved"] + counts["partially_preserved"]) / total if total else None,
        "num_major_caps": len(major_rows),
        "num_major_cap_errors": len(major_errors),
        "num_unsupported_assertions": len(result["unsupported_target_assertions"]),
        "num_major_unsupported_assertions": len(major_additions),
        "role_change_count": sum(row["role_status"] == "changed" for row in result["adjudications"]),
        "modifier_error_counts": dict(modifier_counts),
        "conversation_assessment": result["conversation_assessment"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-transcripts", type=Path, required=True)
    parser.add_argument("--source-caps", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["openai_compat-bearer", "openai-compatible"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--omit-api-version", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--schema-retries", type=int, default=4)
    parser.add_argument("--no-verify-ssl", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_env(args.env_file)
    pairs = group_pairs(read_paired_jsonl(args.paired_transcripts))
    caps = {row["audio_id"]: row for row in read_raw_jsonl(args.source_caps)}
    pairs = [pair for pair in pairs if pair["audio_id"] in caps]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output_dir / ".cap_target.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f"another CAP target adjudication process already owns {args.output_dir}"
        ) from exc
    output_path = args.output_dir / "cap_target_adjudications.jsonl"
    error_path = args.output_dir / "errors.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        # Resume artifacts are CAP-adjudication result rows, not paired
        # transcript rows.  Reusing the paired reader here made every retry
        # after a partially successful first pass fail before reaching the two
        # remaining conversations.
        for row in read_raw_jsonl(output_path):
            aid = row.get("audio_id")
            pair = next((p for p in pairs if p["audio_id"] == aid), None)
            if pair is None:
                continue
            try:
                validate_result(row["result"], audio_id=aid, cap_ids={str(x["cap_id"]) for x in caps[aid]["caps"]}, target=pair["hypothesis"])
                completed[aid] = row
            except ValueError:
                continue
    pending = [pair for pair in pairs if pair["audio_id"] not in completed]
    errors: list[dict[str, Any]] = []
    with output_path.open("a", encoding="utf-8") as out, error_path.open("a", encoding="utf-8") as err:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(evaluate, pair, caps[pair["audio_id"]], args): pair for pair in pending}
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    item = {"audio_id": pair["audio_id"], "dataset_id": pair.get("source_dataset_id"), "language": pair.get("source_language"), "result": result, "metrics": item_metrics(result, caps[pair["audio_id"]])}
                    completed[pair["audio_id"]] = item
                    out.write(json.dumps(item, ensure_ascii=False) + "\n"); out.flush()
                except Exception as exc:  # noqa: BLE001
                    row = {"audio_id": pair["audio_id"], "error_type": type(exc).__name__, "error": str(exc)[:4000]}
                    errors.append(row); err.write(json.dumps(row, ensure_ascii=False) + "\n"); err.flush()
                print(json.dumps({"completed": len(completed), "errors": len(errors), "total": len(pairs)}), flush=True)
    with output_path.open("w", encoding="utf-8") as out:
        for pair in pairs:
            if pair["audio_id"] in completed:
                out.write(json.dumps(completed[pair["audio_id"]], ensure_ascii=False) + "\n")
    metric_rows = [item["metrics"] for item in completed.values()]
    total_caps = sum(row["num_caps"] for row in metric_rows)
    summary = {
        "schema_version": "aegisvox.cap_target_adjudication.v1",
        "model": args.model,
        "backend": args.backend,
        "num_rows": len(pairs), "num_completed": len(completed), "num_errors": len(errors),
        "total_caps": total_caps,
        "micro_strict_preservation_rate": sum(row["preserved"] for row in metric_rows) / total_caps if total_caps else None,
        "micro_lenient_preservation_rate": sum(row["preserved"] + row["partially_preserved"] for row in metric_rows) / total_caps if total_caps else None,
        "total_missing": sum(row["missing"] for row in metric_rows),
        "total_contradicted": sum(row["contradicted"] for row in metric_rows),
        "total_major_cap_errors": sum(row["num_major_cap_errors"] for row in metric_rows),
        "total_unsupported_assertions": sum(row["num_unsupported_assertions"] for row in metric_rows),
        "total_major_unsupported_assertions": sum(row["num_major_unsupported_assertions"] for row in metric_rows),
        "conversation_assessment_counts": dict(Counter(row["conversation_assessment"] for row in metric_rows)),
        "input_sha256": hashlib.sha256(args.paired_transcripts.read_bytes()).hexdigest(),
        "source_caps_sha256": hashlib.sha256(args.source_caps.read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors and len(completed) == len(pairs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
