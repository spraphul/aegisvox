#!/usr/bin/env python3
"""GPT-5.4 item-by-item transcript adjudication for final390 relex edits."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_gpt_concept_preservation import (  # noqa: E402
    call_openai_compat_bearer_chat,
    extract_json_text_chat,
    safe_parse_model_json,
)


SYSTEM = """You are an exacting research co-author adjudicating localized speech relexicalization from transcript evidence.

You must judge every item independently. The deterministic legacy matcher is known to have multilingual, date-normalization, timeline, and fuzzy-collision errors; never use its status as the answer.

Interpret reasonable ASR variants semantically, including spoken dates, ordinals, numbers, abbreviations, accents, and small name spelling errors. Prefer aligned and timestamp-local context over global candidates. Do not accept unrelated whole-conversation fuzzy matches. A literal bracketed HIPAA placeholder is not a natural replacement. Lack of ASR evidence is not always proof of inaudibility, so use uncertain when the evidence is genuinely ambiguous.

Return strict JSON only with one judgment for every input item, in input order. Use only the enumerated value set for each field: never place a replacement-status value in final_original_status or any other field. Repeat the adjudication_id exactly. These are transcript-evidence adjudications, not audio-listener judgments."""


ENUMS = {
    "source_entity_status": {"present", "absent", "uncertain"},
    "final_original_status": {"absent", "present", "uncertain", "not_applicable"},
    "replacement_status": {"realized", "acceptable_variant", "not_realized", "uncertain", "not_applicable"},
    "overall_outcome": {
        "successful_relexicalization", "removal_only", "residual_original_only",
        "mixed_original_and_replacement", "invalid_source_edit",
        "invalid_replacement_placeholder", "indeterminate_asr",
    },
    "confidence": {"high", "medium", "low"},
}


def prompt(items: list[dict]) -> list[dict[str, str]]:
    compact = []
    keep = [
        "adjudication_id", "dataset", "language", "phi_type", "start_sec", "end_sec",
        "original_text", "replacement_text", "planning_asr_aligned_context",
        "original_input_asr_best_context", "original_input_asr_best_candidate",
        "original_input_asr_best_score",
        "final_aligned_context", "final_timestamp_local_context",
        "replacement_best_global_context", "replacement_best_global_candidate",
        "replacement_best_global_score", "original_best_final_context",
        "original_best_final_candidate", "original_best_final_score",
    ]
    for item in items:
        compact.append({k: item.get(k) for k in keep})
    user = {
        "label_contract": {
            "source_entity_status": sorted(ENUMS["source_entity_status"]),
            "final_original_status": sorted(ENUMS["final_original_status"]),
            "replacement_status": sorted(ENUMS["replacement_status"]),
            "overall_outcome": sorted(ENUMS["overall_outcome"]),
            "confidence": sorted(ENUMS["confidence"]),
            "observed_final_phrase": "verbatim final-ASR evidence or null",
            "observed_source_phrase": "verbatim planning-ASR evidence or null",
            "reason": "concise item-specific rationale",
        },
        "required_top_level": {"judgments": "array"},
        "items": compact,
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def validate(payload: dict, items: list[dict]) -> list[dict]:
    rows = payload.get("judgments")
    if not isinstance(rows, list) or len(rows) != len(items):
        raise ValueError(f"expected {len(items)} judgments, got {len(rows) if isinstance(rows, list) else type(rows)}")
    expected = [x["adjudication_id"] for x in items]
    observed = [str(x.get("adjudication_id")) for x in rows]
    if all(value == "None" for value in observed):
        # Some GPT-5.4 responses honor array order but omit the repeated ID.
        # The input/output cardinality and order are already locked, so restore
        # those IDs positionally rather than discarding otherwise valid manual
        # judgments. Mixed/misaligned IDs still fail closed below.
        for row, expected_id in zip(rows, expected):
            row["adjudication_id"] = expected_id
        observed = expected
    if observed != expected:
        raise ValueError(f"judgment IDs/order mismatch: expected={expected}, observed={observed}")
    for row in rows:
        for key, allowed in ENUMS.items():
            if row.get(key) not in allowed:
                raise ValueError(f"{row['adjudication_id']}: invalid {key}={row.get(key)!r}")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError(f"{row['adjudication_id']}: missing reason")
        if "observed_source_phrase" not in row:
            raise ValueError(f"{row['adjudication_id']}: missing observed_source_phrase")
        if row.get("source_entity_status") == "present" and not row.get("observed_source_phrase"):
            raise ValueError(f"{row['adjudication_id']}: source marked present without verbatim source evidence")
        row["manual_reviewer"] = "gpt-5.4 item-level transcript adjudication"
        row["adjudication_modality"] = "original/final ASR transcript evidence"
    return rows


def run_batch(index: int, items: list[dict], args, token: str) -> tuple[int, list[dict]]:
    raw_path = args.output_dir / "raw" / f"batch_{index:03d}.json"
    if args.resume and raw_path.exists():
        return index, validate(json.loads(raw_path.read_text()), items)
    last = None
    for attempt in range(args.retries + 1):
        try:
            response = call_openai_compat_bearer_chat(
                base_url=args.base_url,
                bearer_token=token,
                model=args.model,
                messages=prompt(items),
                temperature=0.0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                verify_ssl=not args.no_verify_ssl,
                api_version=args.api_version or None,
                omit_api_version=not bool(args.api_version),
                service_name=None,
                completion_token_param="max_completion_tokens",
            )
            payload = safe_parse_model_json(extract_json_text_chat(response))
            judgments = validate(payload, items)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps({"judgments": judgments}, ensure_ascii=False, indent=2) + "\n")
            return index, judgments
        except Exception as exc:  # preserve exact error only in local batch log
            last = exc
            if attempt < args.retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"batch {index} failed: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-version", default="")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-verify-ssl", action="store_true")
    args = ap.parse_args()
    if args.model != "gpt-5.4":
        raise SystemExit("paper adjudication is locked to gpt-5.4")
    token = os.environ.get("OPENAI_API_KEY")
    if not token:
        raise SystemExit("OPENAI_API_KEY is missing")
    rows = [json.loads(x) for x in args.packet.read_text().splitlines() if x.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_batch, i, batch, args, token) for i, batch in enumerate(batches)]
        for future in as_completed(futures):
            index, judgments = future.result()
            completed[index] = judgments
            print(json.dumps({"batch": index, "items": len(judgments), "completed_batches": len(completed), "total_batches": len(batches)}), flush=True)
    merged = [row for i in range(len(batches)) for row in completed[i]]
    out = args.output_dir / "item_judgments.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in merged))
    (args.output_dir / "run_summary.json").write_text(json.dumps({
        "schema_version": "aegisvox.final390.item_relex_adjudication.v1",
        "model": args.model,
        "num_items": len(merged),
        "num_batches": len(batches),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "adjudication_modality": "item-level original/final ASR transcript evidence",
        "human_listener_study": False,
    }, indent=2) + "\n")
    print(json.dumps({"status": "complete", "num_items": len(merged), "output": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
