#!/usr/bin/env python3
"""Compute WER/CER for matched original-ASR vs final-ASR transcript pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from common import exact_levenshtein, percentile  # noqa: E402

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def norm_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(TOKEN_RE.findall(text))


def tokens(text: str) -> list[str]:
    normalized = norm_text(text)
    return normalized.split() if normalized else []


def chars(text: str) -> list[str]:
    return list(norm_text(text).replace(" ", ""))


def levenshtein(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (0 if r == h else 1)))
        prev = cur
    return prev[-1]


def mean(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return sum(nums) / len(nums) if nums else None


def read_pairs(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        audio_id = str(row.get("audio_id") or "")
        source = str(row.get("source") or "")
        if not audio_id or source not in {"reference", "hypothesis"}:
            raise SystemExit(f"invalid row {line_no}: {row}")
        item = grouped.setdefault(audio_id, {"audio_id": audio_id, "metadata": row, "reference": "", "hypothesis": ""})
        item[source] = str(row.get("text") or "")
        item["metadata"] = row
    pairs = []
    for audio_id, item in sorted(grouped.items()):
        if not item["reference"] or not item["hypothesis"]:
            raise SystemExit(f"missing reference/hypothesis for {audio_id}")
        pairs.append(item)
    return pairs


def bounded_levenshtein(ref: list[str], hyp: list[str], *, max_cells: int) -> tuple[int, bool, str | None, str]:
    distance, backend = exact_levenshtein(ref, hyp, dp_cell_threshold=max_cells)
    return distance, True, None, backend


def row_metrics(pair: dict[str, Any], *, max_edit_cells: int) -> dict[str, Any]:
    ref_text = pair["reference"]
    hyp_text = pair["hypothesis"]
    ref_words = tokens(ref_text)
    hyp_words = tokens(hyp_text)
    ref_chars = chars(ref_text)
    hyp_chars = chars(hyp_text)
    wer_edit, wer_exact, wer_skip_reason, wer_backend = bounded_levenshtein(ref_words, hyp_words, max_cells=max_edit_cells)
    cer_edit, cer_exact, cer_skip_reason, cer_backend = bounded_levenshtein(ref_chars, hyp_chars, max_cells=max_edit_cells)
    ref_set = set(ref_words)
    hyp_set = set(hyp_words)
    inter = len(ref_set & hyp_set)
    union = len(ref_set | hyp_set)
    meta = pair.get("metadata") or {}
    return {
        "audio_id": pair["audio_id"],
        "run_id": meta.get("run_id"),
        "source_dataset_id": meta.get("source_dataset_id"),
        "source_language": meta.get("source_language"),
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
        "reference_backend": meta.get("reference_backend"),
        "reference_backend_version": meta.get("reference_backend_version"),
        "hypothesis_backend": meta.get("hypothesis_backend"),
        "hypothesis_backend_version": meta.get("hypothesis_backend_version"),
        "claim_boundary": "Matched-ASR utility: original-audio ASR compared to final-audio ASR with the same recognizer/settings.",
    }


def aggregate_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "mean_wer": mean([row.get("wer") for row in rows]),
        "median_wer": percentile([row.get("wer") for row in rows], 0.5),
        "mean_cer": mean([row.get("cer") for row in rows]),
        "median_cer": percentile([row.get("cer") for row in rows], 0.5),
        "num_exact_wer_rows": sum(1 for row in rows if row.get("wer_exact")),
        "num_exact_cer_rows": sum(1 for row in rows if row.get("cer_exact")),
        "num_skipped_wer_rows": sum(1 for row in rows if row.get("wer_skip_reason")),
        "num_skipped_cer_rows": sum(1 for row in rows if row.get("cer_skip_reason")),
        "mean_token_recall_set": mean([row.get("token_recall_set") for row in rows]),
        "mean_token_precision_set": mean([row.get("token_precision_set") for row in rows]),
        "mean_token_jaccard_set": mean([row.get("token_jaccard_set") for row in rows]),
    }


def grouped_aggregate(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    values = sorted({str(row.get(field) or "unknown") for row in rows})
    return {value: aggregate_subset([row for row in rows if str(row.get(field) or "unknown") == value]) for value in values}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **aggregate_subset(rows),
        "by_source_dataset": grouped_aggregate(rows, "source_dataset_id"),
        "by_source_language": grouped_aggregate(rows, "source_language"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    a = payload["aggregate"]
    lines = [
        "# Matched-ASR WER/CER Utility Metrics",
        "",
        "**Protocol:** compare ASR(original audio) against ASR(final transformed audio) using the same recognizer/settings.",
        "",
        "**Claim boundary:** exact transcript utility under matched ASR; long rows use the exact bit-parallel distance backend rather than being skipped.",
        "",
    ]
    for key, value in a.items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | dataset | WER | CER | token recall | words ref/hyp |", "|---|---|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        lines.append(
            f"| {row.get('audio_id')} | {row.get('source_dataset_id')} | {fmt(row.get('wer'))} | {fmt(row.get('cer'))} | {fmt(row.get('token_recall_set'))} | {row.get('reference_word_count')}/{row.get('hypothesis_word_count')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired-transcripts-jsonl", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--max-edit-cells", type=int, default=2_000_000)
    args = ap.parse_args()

    rows = [row_metrics(pair, max_edit_cells=args.max_edit_cells) for pair in read_pairs(args.paired_transcripts_jsonl)]
    payload = {
        "schema_version": "aegisvox.eval.matched_asr_wer_cer.v2",
        "paired_transcripts_jsonl": str(args.paired_transcripts_jsonl),
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
