#!/usr/bin/env python3
"""Build a cross-run comparison table for M8 ASR/diarization full390 ablations.

The script is intentionally tolerant of partially-complete metric packets: it emits one
row per requested ablation/run and marks missing artifacts instead of failing the whole
comparison. This lets us monitor the 8x full390 grid as results arrive.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


from typing import NamedTuple


class MetricField(NamedTuple):
    output_name: str
    metric_name: str
    aggregate_key: str


DEFAULT_FIELDS: tuple[MetricField, ...] = (
    MetricField("num_rows", "duration_completeness", "num_rows"),
    MetricField("num_completed", "duration_completeness", "num_completed"),
    MetricField("duration_ratio_mean", "duration_completeness", "mean_duration_ratio"),
    MetricField("duration_pass", "duration_completeness", "num_duration_pass"),
    MetricField("matched_asr_wer", "matched_asr_wer_cer", "mean_wer"),
    MetricField("matched_asr_cer", "matched_asr_wer_cer", "mean_cer"),
    MetricField("matched_token_recall", "matched_asr_wer_cer", "mean_token_recall_set"),
    MetricField("matched_token_precision", "matched_asr_wer_cer", "mean_token_precision_set"),
    MetricField("speech_activity_f1", "diarization_metrics", "mean_speech_activity_f1"),
    MetricField("speaker_count_abs_diff", "diarization_metrics", "mean_speaker_count_abs_diff"),
    MetricField("asv_eer", "asv_privacy", "asv_proxy_eer"),
    MetricField("asv_linkability_far1", "asv_privacy", "linkability_proxy_at_far1"),
    MetricField("source_target_similarity", "asv_privacy", "mean_max_source_converted_similarity"),
    MetricField("target_target_similarity", "asv_privacy", "mean_min_target_converted_similarity"),
    MetricField("phi_local_residual_rate", "phi_residual_asr", "residual_phi_local_rate"),
    MetricField("phi_global_residual_rate", "phi_residual_asr", "residual_phi_global_rate"),
    MetricField("replacement_local_recall", "phi_residual_asr", "replacement_local_recall"),
    MetricField("replacement_global_or_local_recall", "phi_residual_asr", "replacement_global_or_local_recall"),
    MetricField("concept_f1", "clinical_concept_utility", "mean_concept_f1"),
    MetricField("concept_recall", "clinical_concept_utility", "mean_concept_recall"),
    MetricField("concept_precision", "clinical_concept_utility", "mean_concept_precision"),
    MetricField("final_rms_dbfs", "audio_quality_proxy", "mean_final_rms_dbfs"),
    MetricField("clipped_fraction", "audio_quality_proxy", "mean_final_clipped_sample_fraction"),
    MetricField("repetition_fraction", "audio_quality_proxy", "mean_final_repeated_adjacent_chunk_fraction"),
    MetricField("runtime_rtf", "runtime_system_metrics", "mean_total_wall_clock_rtf_proxy"),
    MetricField("runtime_sec", "runtime_system_metrics", "mean_total_wall_clock_sec_proxy"),
)

COMBO_RE = re.compile(
    r"(?P<asr>small|largev3|large_v3|large-v3|turbo|canary|canary_1b_v2|canary-1b-v2).*?(?P<dia>pyannote|nemo|sortformer)",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_asr(value: str) -> str:
    v = value.lower().replace("-", "_")
    if v in {"largev3", "large_v3"}:
        return "faster_whisper_large_v3"
    if v == "small":
        return "faster_whisper_small"
    if v == "turbo":
        return "faster_whisper_turbo"
    if v in {"canary", "canary_1b_v2"}:
        return "canary_1b_v2"
    return v


def normalize_diarization(value: str) -> str:
    v = value.lower()
    if "nemo" in v or "sortformer" in v:
        return "nemo_sortformer"
    if "pyannote" in v:
        return "pyannote31"
    return v


def infer_combo(path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    text = " ".join([path.name, path.parent.name, str(payload.get("run_id") or ""), str(payload.get("batch_id") or "")])
    m = COMBO_RE.search(text)
    if not m:
        return "unknown_asr", "unknown_diarization"
    return normalize_asr(m.group("asr")), normalize_diarization(m.group("dia"))


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    # M8 packets transitioned from the original names to matched-reference names.
    # Keep both so the comparison table does not show false blanks.
    "matched_asr_wer_cer": ("matched_asr_wer_cer", "wer_cer"),
    "runtime_system_metrics": ("runtime_system_metrics", "runtime_system"),
}


def metric_aggregate(metrics: dict[str, Any], metric_name: str) -> dict[str, Any]:
    names = METRIC_ALIASES.get(metric_name, (metric_name,))
    for name in names:
        item = metrics.get(name) or {}
        aggregate = item.get("aggregate") or {}
        if aggregate:
            return aggregate
    return {}


def metric_value(metrics: dict[str, Any], metric_name: str, aggregate_key: str) -> Any:
    names = METRIC_ALIASES.get(metric_name, (metric_name,))
    for name in names:
        item = metrics.get(name) or {}
        aggregate = item.get("aggregate") or {}
        if aggregate.get(aggregate_key) is not None:
            return aggregate.get(aggregate_key)
    return None


def extract_row(metrics_dir: Path, *, run_label: str | None = None) -> dict[str, Any]:
    agg_path = metrics_dir / "eval_aggregate.json"
    payload: dict[str, Any] = {"metrics": {}}
    if agg_path.exists():
        payload = load_json(agg_path)
    metrics: dict[str, Any] = dict(payload.get("metrics") or {})

    # Optional newer metric packets may live beside eval_aggregate and not be included in it yet.
    optional_files = {
        "matched_asr_wer_cer": metrics_dir / "matched_asr_wer_cer.json",
        "m7_relex_render_metrics": metrics_dir / "m7_relex_render_metrics.json",
    }
    for name, path in optional_files.items():
        if path.exists() and name not in metrics:
            metrics[name] = {"path": str(path), "aggregate": load_json(path).get("aggregate", {})}

    asr, diarization = infer_combo(metrics_dir, payload)
    row: dict[str, Any] = {
        "run_label": run_label or metrics_dir.name,
        "metrics_dir": str(metrics_dir),
        "asr_backend": asr,
        "diarization_backend": diarization,
        "has_eval_aggregate": agg_path.exists(),
        "metric_groups_present": ",".join(sorted(metrics)),
    }
    missing_fields: list[str] = []
    for field in DEFAULT_FIELDS:
        value = metric_value(metrics, field.metric_name, field.aggregate_key)
        row[field.output_name] = value
        if value is None:
            missing_fields.append(f"{field.metric_name}.{field.aggregate_key}")
    row["missing_metric_fields"] = ";".join(missing_fields)
    return row


def read_manifest(path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            label, p = line.split("\t", 1)
        elif "," in line:
            label, p = line.split(",", 1)
        else:
            label, p = Path(line.strip()).name, line.strip()
        rows.append((label.strip(), Path(p.strip())))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_label",
        "asr_backend",
        "diarization_backend",
        "num_rows",
        "num_completed",
        "duration_ratio_mean",
        "matched_asr_wer",
        "matched_token_recall",
        "speech_activity_f1",
        "speaker_count_abs_diff",
        "asv_eer",
        "asv_linkability_far1",
        "source_target_similarity",
        "target_target_similarity",
        "phi_local_residual_rate",
        "replacement_global_or_local_recall",
        "concept_f1",
        "final_rms_dbfs",
        "runtime_rtf",
        "has_eval_aggregate",
        "missing_metric_fields",
        "metrics_dir",
        "metric_groups_present",
    ]
    extras = sorted({k for row in rows for k in row} - set(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields + extras)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# M8 ASR × Diarization Ablation Comparison",
        "",
        "Scope: one row per full390 ablation metric packet. Missing cells mean that metric has not been produced/pulled yet.",
        "",
        "| run | ASR | diarization | rows | matched token recall | speech F1 | ASV EER | linkability@FAR1 | PHI local residual | replacement recall | concept F1 | RTF | missing |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        missing_count = len([x for x in str(row.get("missing_metric_fields") or "").split(";") if x])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("run_label") or ""),
                    str(row.get("asr_backend") or ""),
                    str(row.get("diarization_backend") or ""),
                    fmt(row.get("num_rows")),
                    fmt(row.get("matched_token_recall")),
                    fmt(row.get("speech_activity_f1")),
                    fmt(row.get("asv_eer")),
                    fmt(row.get("asv_linkability_far1")),
                    fmt(row.get("phi_local_residual_rate")),
                    fmt(row.get("replacement_global_or_local_recall")),
                    fmt(row.get("concept_f1")),
                    fmt(row.get("runtime_rtf")),
                    str(missing_count),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-dir", type=Path, action="append", default=[], help="Directory containing eval_aggregate.json and related metric JSON files.")
    ap.add_argument("--manifest", type=Path, help="Optional label/path manifest. TSV or CSV: label<tab>metrics_dir.")
    ap.add_argument("--discover-root", type=Path, action="append", default=[], help="Discover */eval_aggregate.json below this root.")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()

    requested: list[tuple[str | None, Path]] = [(None, p) for p in args.metrics_dir]
    if args.manifest:
        requested.extend(read_manifest(args.manifest))
    for root in args.discover_root:
        for path in sorted(root.glob("**/eval_aggregate.json")):
            requested.append((path.parent.name, path.parent))

    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for label, metrics_dir in requested:
        metrics_dir = metrics_dir.expanduser().resolve()
        if metrics_dir in seen:
            continue
        seen.add(metrics_dir)
        rows.append(extract_row(metrics_dir, run_label=label))

    rows.sort(key=lambda r: (str(r.get("asr_backend")), str(r.get("diarization_backend")), str(r.get("run_label"))))
    payload = {
        "schema_version": "aegisvox.eval.m8_ablation_comparison.v1",
        "num_runs": len(rows),
        "rows": rows,
        "claim_boundary": "Comparison table only. Each metric keeps its own protocol boundary; missing cells are not evidence.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(json.dumps({"num_runs": len(rows), "output_json": str(args.output_json)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
