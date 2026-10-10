#!/usr/bin/env python3
"""Compare public-reference WER/CER before and after AegisVox transformation.

The original and final transcripts must come from the matched-ASR packet, so
both sides use the same recognizer and settings.  Public references are used as
the fixed anchor.  This separates ASR error already present on the source audio
from additional transcript error after transformation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from common import chars, config_inputs, exact_levenshtein, percentile, read_reference_transcript, tokens  # noqa: E402


def mean(values: list[Any]) -> float | None:
    vals = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(vals) / len(vals) if vals else None


def read_pairs(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        audio_id = str(payload.get("audio_id") or "")
        source = str(payload.get("source") or "")
        if not audio_id or source not in {"reference", "hypothesis"}:
            raise SystemExit(f"invalid paired-ASR row {line_no}: {payload}")
        item = grouped.setdefault(audio_id, {"audio_id": audio_id, "original_asr": None, "final_asr": None, "metadata": {}})
        target = "original_asr" if source == "reference" else "final_asr"
        if item[target] is not None:
            raise SystemExit(f"duplicate {source} transcript for {audio_id}")
        item[target] = str(payload.get("text") or "")
        item["metadata"] = {**item["metadata"], **payload}
    pairs = []
    for audio_id, item in sorted(grouped.items()):
        if not str(item.get("original_asr") or "").strip() or not str(item.get("final_asr") or "").strip():
            raise SystemExit(f"missing or empty original/final ASR transcript for {audio_id}")
        pairs.append(item)
    return pairs


def distance_metrics(reference: str, hypothesis: str, *, max_edit_cells: int) -> dict[str, Any]:
    ref_words, hyp_words = tokens(reference), tokens(hypothesis)
    ref_chars, hyp_chars = chars(reference), chars(hypothesis)
    word_distance, word_backend = exact_levenshtein(ref_words, hyp_words, dp_cell_threshold=max_edit_cells)
    char_distance, char_backend = exact_levenshtein(ref_chars, hyp_chars, dp_cell_threshold=max_edit_cells)
    return {
        "wer": word_distance / len(ref_words) if ref_words else None,
        "cer": char_distance / len(ref_chars) if ref_chars else None,
        "word_edit_distance": word_distance,
        "char_edit_distance": char_distance,
        "word_distance_backend": word_backend,
        "char_distance_backend": char_backend,
        "exact": True,
    }


def build_rows(*, paired_transcripts: Path, config: Path, max_edit_cells: int) -> list[dict[str, Any]]:
    cfg = config_inputs(config)
    rows: list[dict[str, Any]] = []
    for pair in read_pairs(paired_transcripts):
        audio_id = pair["audio_id"]
        config_row = cfg.get(audio_id)
        if not isinstance(config_row, dict):
            raise SystemExit(f"audio_id missing from frozen config: {audio_id}")
        metadata = config_row.get("metadata") or {}
        transcript_path = Path(str(metadata.get("transcript_path") or ""))
        if not transcript_path.is_file():
            raise SystemExit(f"missing public reference transcript for {audio_id}: {transcript_path}")
        public_reference = read_reference_transcript(transcript_path)
        if not public_reference.strip():
            raise SystemExit(f"empty public reference transcript for {audio_id}: {transcript_path}")
        before = distance_metrics(public_reference, pair["original_asr"], max_edit_cells=max_edit_cells)
        after = distance_metrics(public_reference, pair["final_asr"], max_edit_cells=max_edit_cells)
        pair_meta = pair.get("metadata") or {}
        rows.append({
            "audio_id": audio_id,
            "run_id": pair_meta.get("run_id"),
            "source_dataset_id": metadata.get("source_dataset_id") or pair_meta.get("source_dataset_id"),
            "source_language": metadata.get("language") or metadata.get("source_language") or pair_meta.get("source_language"),
            "reference_transcript_path": str(transcript_path),
            "reference_word_count": len(tokens(public_reference)),
            "original_asr_word_count": len(tokens(pair["original_asr"])),
            "final_asr_word_count": len(tokens(pair["final_asr"])),
            "original_wer": before["wer"],
            "final_wer": after["wer"],
            "delta_wer_final_minus_original": (after["wer"] - before["wer"]) if before["wer"] is not None and after["wer"] is not None else None,
            "original_cer": before["cer"],
            "final_cer": after["cer"],
            "delta_cer_final_minus_original": (after["cer"] - before["cer"]) if before["cer"] is not None and after["cer"] is not None else None,
            "original_word_edit_distance": before["word_edit_distance"],
            "final_word_edit_distance": after["word_edit_distance"],
            "original_char_edit_distance": before["char_edit_distance"],
            "final_char_edit_distance": after["char_edit_distance"],
            "original_wer_distance_backend": before["word_distance_backend"],
            "final_wer_distance_backend": after["word_distance_backend"],
            "original_cer_distance_backend": before["char_distance_backend"],
            "final_cer_distance_backend": after["char_distance_backend"],
            "original_exact": before["exact"],
            "final_exact": after["exact"],
            "reference_backend": pair_meta.get("reference_backend"),
            "reference_backend_version": pair_meta.get("reference_backend_version"),
            "hypothesis_backend": pair_meta.get("hypothesis_backend"),
            "hypothesis_backend_version": pair_meta.get("hypothesis_backend_version"),
            "claim_boundary": "Public-reference before/after transcript utility using same-backend ASR on original and final audio; delta is final minus original.",
        })
    return rows


def aggregate_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_exact_original_rows": sum(row.get("original_exact") is True for row in rows),
        "num_exact_final_rows": sum(row.get("final_exact") is True for row in rows),
        "mean_original_wer": mean([row.get("original_wer") for row in rows]),
        "median_original_wer": percentile([row.get("original_wer") for row in rows], 0.5),
        "mean_final_wer": mean([row.get("final_wer") for row in rows]),
        "median_final_wer": percentile([row.get("final_wer") for row in rows], 0.5),
        "mean_delta_wer_final_minus_original": mean([row.get("delta_wer_final_minus_original") for row in rows]),
        "mean_original_cer": mean([row.get("original_cer") for row in rows]),
        "median_original_cer": percentile([row.get("original_cer") for row in rows], 0.5),
        "mean_final_cer": mean([row.get("final_cer") for row in rows]),
        "median_final_cer": percentile([row.get("final_cer") for row in rows], 0.5),
        "mean_delta_cer_final_minus_original": mean([row.get("delta_cer_final_minus_original") for row in rows]),
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    values = sorted({str(row.get(key) or "unknown") for row in rows})
    return {value: aggregate_subset([row for row in rows if str(row.get(key) or "unknown") == value]) for value in values}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {**aggregate_subset(rows), "by_source_dataset": grouped(rows, "source_dataset_id"), "by_source_language": grouped(rows, "source_language")}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Public-reference ASR utility before and after transformation",
        "",
        "Original and final audio are transcribed by the same ASR backend. Both are scored against the fixed public transcript; paired deltas are final minus original.",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | dataset | original WER | final WER | delta WER | original CER | final CER |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        lines.append(f"| {row['audio_id']} | {row.get('source_dataset_id')} | {fmt(row.get('original_wer'))} | {fmt(row.get('final_wer'))} | {fmt(row.get('delta_wer_final_minus_original'))} | {fmt(row.get('original_cer'))} | {fmt(row.get('final_cer'))} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired-transcripts-jsonl", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--max-edit-cells", type=int, default=2_000_000)
    args = ap.parse_args()
    rows = build_rows(paired_transcripts=args.paired_transcripts_jsonl, config=args.config, max_edit_cells=args.max_edit_cells)
    payload = {
        "schema_version": "aegisvox.eval.reference_asr_before_after.v1",
        "paired_transcripts_jsonl": str(args.paired_transcripts_jsonl),
        "config": str(args.config),
        "aggregate": aggregate(rows),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, payload)
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
