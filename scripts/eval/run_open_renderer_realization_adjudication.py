#!/usr/bin/env python3
"""Independent open-model adjudication of renderer outputs from frozen ASR evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Any


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("aegisvox_renderer_realization_contract", HERE / "run_llm_renderer_realization_adjudication.py")
CLIENT = load_module("aegisvox_open_chat_client", HERE / "run_gpt_concept_preservation.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_batch(index: int, items: list[dict[str, Any]], args: argparse.Namespace) -> tuple[int, list[dict[str, Any]]]:
    raw_path = args.output_dir / "raw" / f"batch_{index:04d}.json"
    if args.resume and raw_path.exists():
        payload = json.loads(raw_path.read_text())
        judgments = BASE.validate(payload, items)
        raw_path.write_text(json.dumps({"judgments": judgments}, ensure_ascii=False, indent=2) + "\n")
        return index, judgments
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            response = CLIENT.call_chat_completions(
                base_url=args.base_url,
                api_key="not-required",
                model=args.model,
                messages=BASE.build_messages(items),
                temperature=0.0,
                timeout=args.timeout,
                max_retries=1,
                verify_ssl=True,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
            )
            parsed = CLIENT.safe_parse_model_json(CLIENT.extract_json_text_chat(response))
            judgments = BASE.validate(parsed, items)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps({"judgments": judgments}, ensure_ascii=False, indent=2) + "\n")
            return index, judgments
        except Exception as exc:  # bounded, fail-visible schema retry
            last_error = exc
            if attempt < args.retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"batch {index} failed: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-32b-independent-cap-judge")
    parser.add_argument("--reported-model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8908/v1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    expected_hashes = {
        "final390_turn_corrected_voicecraft_stock330m_20260808": "48af28c70ace75b528ca093150df66af257da70b8cab784a13047f231fea320d",
        "final390_turn_corrected_f5tts_20260808": "b59dbc46ca463bbcae2f7b8aaf149dfd13ed5307e77eeef3aec739a892c722b6",
    }
    parent = args.rows.parent.name
    if parent not in expected_hashes or sha256(args.rows) != expected_hashes[parent]:
        raise SystemExit("rows file is not one of the two frozen 601-item renderer evidence files")
    all_rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    if len(all_rows) != 601 or len({str(row["item_id"]) for row in all_rows}) != 601:
        raise SystemExit("frozen renderer evidence must contain 601 unique item IDs")
    failed = [row for row in all_rows if row.get("score_status") != "ok"]
    source_rows = [row for row in all_rows if row.get("score_status") == "ok"]
    batches = [source_rows[index:index + args.batch_size] for index in range(0, len(source_rows), args.batch_size)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir / "evaluation_lock.json"
    if not lock_path.exists():
        lock_path.write_text(json.dumps({
            "schema_version": "aegisvox.open_renderer_realization_lock.v1",
            "status": "frozen_before_first_request",
            "model": args.reported_model,
            "rows_sha256": sha256(args.rows),
            "num_items": 601,
            "contract_system_prompt_sha256": hashlib.sha256(BASE.SYSTEM.encode()).hexdigest(),
            "temperature": 0.0,
            "batch_size": args.batch_size,
            "max_output_tokens": args.max_tokens,
            "claim_boundary": "independent open-model adjudication of frozen multi-ASR transcript evidence",
        }, indent=2) + "\n")

    completed: dict[int, list[dict[str, Any]]] = {}
    errors: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_batch, index, batch, args): index for index, batch in enumerate(batches)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                batch_index, judgments = future.result()
                completed[batch_index] = judgments
            except Exception as exc:
                errors[index] = str(exc)
            print(json.dumps({
                "completed_batches": len(completed), "failed_batches": len(errors), "total_batches": len(batches),
            }), flush=True)
    if errors:
        (args.output_dir / "batch_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        raise SystemExit(f"{len(errors)} batches remain invalid; rerun with --resume")

    merged = [row for index in range(len(batches)) for row in completed[index]]
    for row in failed:
        merged.append({
            "adjudication_id": row["item_id"],
            "replacement_status": "uncertain",
            "original_status": "uncertain",
            "content_status": "uncertain",
            "overall_outcome": "indeterminate_asr",
            "confidence": "high",
            "replacement_evidence": None,
            "original_evidence": None,
            "reason": "Renderer generation failed; no generated audio was available for ASR adjudication.",
            "operational_status": "generation_failed",
        })
    # Preserve the frozen item order, including fail-closed generation failures.
    by_id = {str(row["adjudication_id"]): row for row in merged}
    merged = [by_id[str(row["item_id"])] for row in all_rows]
    judgments_path = args.output_dir / "item_judgments.jsonl"
    judgments_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged))
    summary = BASE.summarize(merged, all_rows, args.reported_model)
    summary.update({
        "schema_version": "aegisvox.renderer_realization.open_model_sensitivity.v1",
        "adjudication_modality": "Qwen3-32B semantic adjudication of two frozen independent ASR transcripts",
        "judge_role": "independent open-weight sensitivity judge; not listening ground truth",
        "num_generation_failed": len(failed),
        "num_requested_rows": len(all_rows),
        "rows_sha256": sha256(args.rows),
        "item_judgments_sha256": sha256(judgments_path),
    })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "complete", "num_items": len(merged), "summary": str(args.output_dir / 'summary.json')}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
