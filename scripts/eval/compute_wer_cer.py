#!/usr/bin/env python3
"""Compute transcript utility metrics for AegisVox eval runs.

This script is deterministic and dependency-light. It reports WER/CER against
public reference transcripts and labels metrics as protocol-level unless PHI
normalization/gold labels are added by a later paper-grade pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import audio_id_for_summary_row, chars, config_input_for_summary_row, config_inputs, exact_levenshtein, final_audio_path, input_audio_path, load_asr_text, load_summary_rows, mean, read_reference_transcript, row_completed, run_dir_from_summary_row, tokens, wav_info, write_csv, write_json


def bounded_levenshtein(ref: list[str], hyp: list[str], *, max_cells: int) -> tuple[int, bool, str | None, str]:
    distance, backend = exact_levenshtein(ref, hyp, dp_cell_threshold=max_cells)
    return distance, True, None, backend


def text_metric_row(row: dict[str, Any], config_by_audio: dict[str, dict[str, Any]], *, max_edit_cells: int = 2_000_000) -> dict[str, Any]:
    audio_id = audio_id_for_summary_row(row, config_by_audio)
    run_dir = run_dir_from_summary_row(row)
    metadata = (config_input_for_summary_row(row, config_by_audio).get("metadata") or {})
    ref_path = Path(metadata["transcript_path"]) if metadata.get("transcript_path") else None
    ref_text = read_reference_transcript(ref_path)
    hyp_text = load_asr_text(run_dir)
    ref_words = tokens(ref_text)
    hyp_words = tokens(hyp_text)
    ref_chars = chars(ref_text)
    hyp_chars = chars(hyp_text)
    wer_edit, wer_exact, wer_skip_reason, wer_backend = bounded_levenshtein(ref_words, hyp_words, max_cells=max_edit_cells) if ref_words else (len(hyp_words), True, None, "empty_reference")
    cer_edit, cer_exact, cer_skip_reason, cer_backend = bounded_levenshtein(ref_chars, hyp_chars, max_cells=max_edit_cells) if ref_chars else (len(hyp_chars), True, None, "empty_reference")
    ref_set = set(ref_words)
    hyp_set = set(hyp_words)
    inter = len(ref_set & hyp_set)
    union = len(ref_set | hyp_set)
    return {
        "audio_id": audio_id,
        "run_id": row.get("run_id"),
        "source_dataset_id": metadata.get("source_dataset_id"),
        "source_sample_id": metadata.get("source_sample_id"),
        "completed": row_completed(row),
        "reference_transcript_path": str(ref_path) if ref_path else None,
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words),
        "reference_char_count": len(ref_chars),
        "hypothesis_char_count": len(hyp_chars),
        "wer": (wer_edit / len(ref_words)) if ref_words and wer_edit is not None else None,
        "cer": (cer_edit / len(ref_chars)) if ref_chars and cer_edit is not None else None,
        "wer_exact": wer_exact,
        "cer_exact": cer_exact,
        "wer_skip_reason": wer_skip_reason,
        "cer_skip_reason": cer_skip_reason,
        "wer_distance_backend": wer_backend,
        "cer_distance_backend": cer_backend,
        "word_edit_distance": wer_edit,
        "char_edit_distance": cer_edit,
        "token_recall_set": (inter / len(ref_set)) if ref_set else None,
        "token_precision_set": (inter / len(hyp_set)) if hyp_set else None,
        "token_jaccard_set": (inter / union) if union else None,
        "input_duration_sec": wav_info(input_audio_path(run_dir))["duration_sec"],
        "final_duration_sec": wav_info(final_audio_path(run_dir))["duration_sec"],
        "claim_boundary": "Exact reference-transcript utility under public transcript conventions; primary transformation utility is the matched original-ASR versus final-ASR analysis.",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_completed": sum(1 for row in rows if row.get("completed")),
        "mean_wer": mean([row.get("wer") for row in rows]),
        "mean_cer": mean([row.get("cer") for row in rows]),
        "num_exact_wer_rows": sum(1 for row in rows if row.get("wer_exact")),
        "num_exact_cer_rows": sum(1 for row in rows if row.get("cer_exact")),
        "num_skipped_wer_rows": sum(1 for row in rows if row.get("wer_skip_reason")),
        "num_skipped_cer_rows": sum(1 for row in rows if row.get("cer_skip_reason")),
        "mean_token_recall_set": mean([row.get("token_recall_set") for row in rows]),
        "mean_token_precision_set": mean([row.get("token_precision_set") for row in rows]),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# WER/CER Utility Metrics",
        "",
        "**Claim boundary:** exact public-reference utility under dataset transcript conventions; primary transformation utility is reported with matched original/final ASR.",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | dataset | WER | CER | token recall | words ref/hyp |", "|---|---|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        lines.append(
            f"| {row.get('audio_id')} | {row.get('source_dataset_id')} | {_fmt(row.get('wer'))} | {_fmt(row.get('cer'))} | {_fmt(row.get('token_recall_set'))} | {row.get('reference_word_count')}/{row.get('hypothesis_word_count')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-edit-cells", type=int, default=2_000_000, help="Use bit-parallel exact distance above this dynamic-programming cell threshold; rows are not skipped.")
    args = parser.parse_args()
    cfg = config_inputs(args.config)
    rows = [text_metric_row(row, cfg, max_edit_cells=args.max_edit_cells) for row in load_summary_rows(args.run_summary)]
    payload = {
        "schema_version": "aegisvox.eval.wer_cer.v1",
        "run_summary": str(args.run_summary),
        "config": str(args.config),
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
