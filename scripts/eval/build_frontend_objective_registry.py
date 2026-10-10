#!/usr/bin/env python3
"""Build the exact 8 x 390 objective frontend-analysis registry for W4.2.

The registry joins only measured frontend/audio/privacy fields and frozen
metadata strata.  It intentionally does not manufacture clinical outcomes;
CAP, role, relex, action, and note outcomes are joined only after their frozen
condition adjudications exist.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CELL_FACTORS = {
    "largev3_pyannote": ("faster-whisper-large-v3", "pyannote-3.1"),
    "largev3_nemo": ("faster-whisper-large-v3", "nemo-sortformer"),
    "turbo_pyannote": ("faster-whisper-turbo", "pyannote-3.1"),
    "turbo_nemo": ("faster-whisper-turbo", "nemo-sortformer"),
    "small_pyannote": ("faster-whisper-small", "pyannote-3.1"),
    "small_nemo": ("faster-whisper-small", "nemo-sortformer"),
    "canary_pyannote": ("nvidia-canary-1b-v2", "pyannote-3.1"),
    "canary_nemo": ("nvidia-canary-1b-v2", "nemo-sortformer"),
}

METRIC_FIELDS = {
    "matched_asr_wer_cer.json": {
        "wer": "matched_asr_wer",
        "cer": "matched_asr_cer",
        "token_recall_set": "matched_asr_token_recall",
        "token_precision_set": "matched_asr_token_precision",
    },
    "wer_cer.json": {
        "wer": "public_reference_wer",
        "cer": "public_reference_cer",
    },
    "diarization_metrics.json": {
        "official_der": "official_der",
        "official_jer": "official_jer",
        "speech_activity_f1": "speech_activity_f1",
        "source_final_speaker_count_abs_diff": "source_final_speaker_count_abs_diff",
        "source_final_speaker_count_exact_match": "source_final_speaker_count_exact_match",
        "source_final_turn_count_ratio": "source_final_turn_count_ratio",
    },
    "phi_residual_asr.json": {
        "num_phi_edits": "num_phi_edits_measured",
        "num_fail_residual_phi_local": "num_fail_residual_phi_local",
        "num_fail_residual_phi_global": "num_fail_residual_phi_global",
        "num_warn_replacement_missing": "num_warn_replacement_missing",
        "replacement_local_recall": "replacement_local_recall",
        "residual_phi_local_rate": "residual_phi_local_rate",
        "residual_phi_global_rate": "residual_phi_global_rate",
    },
    "audio_quality.json": {
        "input_rms_dbfs": "input_rms_dbfs",
        "input_silence_fraction_chunks": "input_silence_fraction_chunks",
        "final_rms_dbfs": "final_rms_dbfs",
        "final_silence_fraction_chunks": "final_silence_fraction_chunks",
        "final_clipped_sample_fraction": "final_clipped_sample_fraction",
        "final_repeated_adjacent_chunk_fraction": "final_repeated_adjacent_chunk_fraction",
        "rms_delta_db_final_minus_input": "rms_delta_db_final_minus_input",
    },
    "duration_completeness.json": {
        "duration_ratio_final_over_input": "duration_ratio_final_over_input",
        "absolute_duration_delta_sec": "absolute_duration_delta_sec",
    },
    "runtime_system_metrics.json": {
        "measured_active_pipeline_rtf": "measured_active_pipeline_rtf",
        "measured_active_pipeline_runtime_sec": "measured_active_pipeline_runtime_sec",
        "stage_timing_coverage": "stage_timing_coverage",
        "num_stage_failure_events": "num_stage_failure_events",
        "num_stage_retry_attempts": "num_stage_retry_attempts",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(row)
    return rows


def index_rows(rows: list[dict[str, Any]], *, path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        audio_id = str(row.get("audio_id") or "")
        if not audio_id:
            raise ValueError(f"{path}: row missing audio_id")
        if audio_id in indexed:
            raise ValueError(f"{path}: duplicate audio_id={audio_id}")
        indexed[audio_id] = row
    return indexed


def load_metric(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing rows[]")
    return index_rows(rows, path=path)


def finite_mean(values: list[Any]) -> float | None:
    numeric = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
    return sum(numeric) / len(numeric) if numeric else None


def build_registry(
    *, matrix_root: Path, strata_path: Path, expected_rows: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata_rows = read_jsonl(strata_path)
    strata = index_rows(strata_rows, path=strata_path)
    if len(strata) != expected_rows:
        raise ValueError(f"expected {expected_rows} strata rows, got {len(strata)}")

    output: list[dict[str, Any]] = []
    input_hashes: dict[str, Any] = {"strata_registry": sha256(strata_path), "cells": {}}
    per_cell_missingness: dict[str, dict[str, int]] = {}
    per_cell_means: dict[str, dict[str, float | None]] = {}

    for cell, (asr_backend, diarization_backend) in CELL_FACTORS.items():
        metric_dir = matrix_root / cell / "evaluation" / "metrics"
        completion = metric_dir / "condition_complete.json"
        if not completion.is_file():
            raise ValueError(f"{cell}: missing completion artifact {completion}")
        completion_payload = json.loads(completion.read_text(encoding="utf-8"))
        if completion_payload.get("status") != "complete" or int(completion_payload.get("num_rows") or 0) != expected_rows:
            raise ValueError(f"{cell}: invalid completion artifact")

        metrics: dict[str, dict[str, dict[str, Any]]] = {}
        input_hashes["cells"][cell] = {"condition_complete": sha256(completion), "metrics": {}}
        expected_ids = set(strata)
        for filename in METRIC_FIELDS:
            path = metric_dir / filename
            if not path.is_file():
                raise ValueError(f"{cell}: missing required metric {path}")
            indexed = load_metric(path)
            if set(indexed) != expected_ids:
                raise ValueError(
                    f"{cell}/{filename}: non-bijective IDs; "
                    f"missing={sorted(expected_ids-set(indexed))[:5]} extra={sorted(set(indexed)-expected_ids)[:5]}"
                )
            metrics[filename] = indexed
            input_hashes["cells"][cell]["metrics"][filename] = sha256(path)

        cell_rows: list[dict[str, Any]] = []
        for audio_id in sorted(expected_ids):
            s = strata[audio_id]
            row: dict[str, Any] = {
                "audio_id": audio_id,
                "cell": cell,
                "asr_backend": asr_backend,
                "diarization_backend": diarization_backend,
                "dataset_id": s["dataset_id"],
                "language": s["language"],
                "duration_sec": s["duration_sec"],
                "duration_quartile": s["duration_quartile"],
                "speaker_count": s["speaker_count"],
                "speaker_count_bin": s["speaker_count_bin"],
                "edit_count": s["edit_count"],
                "edit_count_bin": s["edit_count_bin"],
                "total_edit_duration_sec": s["total_edit_duration_sec"],
                "max_edit_duration_sec": s["max_edit_duration_sec"],
                "phi_type_counts": s["phi_type_counts"],
            }
            for filename, mapping in METRIC_FIELDS.items():
                source_row = metrics[filename][audio_id]
                for source_field, output_field in mapping.items():
                    row[output_field] = source_row.get(source_field)
            cell_rows.append(row)
        output.extend(cell_rows)

        missing = Counter()
        measured_fields = [field for mapping in METRIC_FIELDS.values() for field in mapping.values()]
        for row in cell_rows:
            for field in measured_fields:
                if row.get(field) is None:
                    missing[field] += 1
        per_cell_missingness[cell] = dict(sorted(missing.items()))
        per_cell_means[cell] = {
            field: finite_mean([row.get(field) for row in cell_rows])
            for field in measured_fields
        }

    expected_total = len(CELL_FACTORS) * expected_rows
    if len(output) != expected_total or len({(r["cell"], r["audio_id"]) for r in output}) != expected_total:
        raise ValueError("frontend registry failed exact 8 x expected-row bijection")
    summary = {
        "schema_version": "aegisvox.frontend_objective_registry.v1",
        "status": "complete",
        "num_cells": len(CELL_FACTORS),
        "num_rows_per_cell": expected_rows,
        "num_rows": len(output),
        "cell_factors": {
            cell: {"asr_backend": factors[0], "diarization_backend": factors[1]}
            for cell, factors in CELL_FACTORS.items()
        },
        "dataset_counts_per_cell": dict(sorted(Counter(r["dataset_id"] for r in strata.values()).items())),
        "language_counts_per_cell": dict(sorted(Counter(r["language"] for r in strata.values()).items())),
        "missingness_by_cell": per_cell_missingness,
        "unadjusted_means_by_cell": per_cell_means,
        "inputs": input_hashes,
        "claim_boundary": (
            "Exact measured frontend/audio/privacy registry only. No CAP, role, action, note, "
            "or human clinical outcome is inferred here; missing values are retained."
        ),
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--strata-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=390)
    args = parser.parse_args()
    rows, summary = build_registry(
        matrix_root=args.matrix_root,
        strata_path=args.strata_jsonl,
        expected_rows=args.expected_rows,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary["output_jsonl"] = str(args.output_jsonl)
    summary["output_sha256"] = sha256(args.output_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "num_cells": len(CELL_FACTORS), "num_rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
