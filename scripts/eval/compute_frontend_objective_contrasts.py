#!/usr/bin/env python3
"""Compute recording-paired objective contrasts for the frozen 4x2 frontend matrix.

This is the objective-measure component of W4.2.  It preserves the recording as
the resampling unit, reports balanced denominators and missingness, and applies
Benjamini-Hochberg correction across the declared secondary contrast family.
It does not create or substitute clinical outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = (
    "matched_asr_wer",
    "matched_asr_cer",
    "public_reference_wer",
    "public_reference_cer",
    "official_der",
    "official_jer",
    "speech_activity_f1",
    "source_final_speaker_count_abs_diff",
    "replacement_local_recall",
    "residual_phi_local_rate",
    "duration_ratio_final_over_input",
    "final_clipped_sample_fraction",
    "final_repeated_adjacent_chunk_fraction",
    "measured_active_pipeline_rtf",
)

ASRS = (
    "faster-whisper-large-v3",
    "faster-whisper-turbo",
    "faster-whisper-small",
    "nvidia-canary-1b-v2",
)
DIARIZERS = ("pyannote-3.1", "nemo-sortformer")
REFERENCE_ASR = "faster-whisper-large-v3"
REFERENCE_DIARIZER = "pyannote-3.1"


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


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def paired_inference(
    differences: list[float], *, bootstrap_iterations: int, permutation_iterations: int, seed: int
) -> dict[str, Any]:
    n = len(differences)
    if not n:
        return {
            "n": 0, "mean_difference": None, "ci95_low": None, "ci95_high": None,
            "raw_p_two_sided": None,
        }
    values = np.asarray(differences, dtype=np.float64)
    mean_diff = float(values.mean())
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, n, size=(bootstrap_iterations, n))
    bootstrap = values[bootstrap_indices].mean(axis=1)
    observed = abs(mean_diff)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutation_iterations, n))
    permuted = (signs * values).mean(axis=1)
    extreme = int(np.count_nonzero(np.abs(permuted) >= observed - 1e-15))
    return {
        "n": n,
        "mean_difference": mean_diff,
        "median_difference": percentile(differences, 0.5),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "raw_p_two_sided": (extreme + 1) / (permutation_iterations + 1),
        "bootstrap_iterations": bootstrap_iterations,
        "permutation_iterations": permutation_iterations,
        "resampling_unit": "recording",
    }


def benjamini_hochberg(records: list[dict[str, Any]]) -> None:
    valid = [(index, float(row["raw_p_two_sided"])) for index, row in enumerate(records) if row.get("raw_p_two_sided") is not None]
    valid.sort(key=lambda item: item[1])
    m = len(valid)
    adjusted: dict[int, float] = {}
    running = 1.0
    for rank_from_end, (index, p_value) in enumerate(reversed(valid), 1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[index] = min(1.0, running)
    for index, row in enumerate(records):
        row["fdr_bh_q"] = adjusted.get(index)


def build_report(
    rows: list[dict[str, Any]], *, fields: list[str], bootstrap_iterations: int,
    permutation_iterations: int, seed: int,
) -> dict[str, Any]:
    by_audio: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    duplicate_keys = []
    for row in rows:
        audio_id = str(row.get("audio_id") or "")
        key = (str(row.get("asr_backend") or ""), str(row.get("diarization_backend") or ""))
        if not audio_id or key[0] not in ASRS or key[1] not in DIARIZERS:
            raise ValueError("invalid audio_id or frontend factor")
        if key in by_audio[audio_id]:
            duplicate_keys.append((audio_id, key))
        by_audio[audio_id][key] = row
    if duplicate_keys:
        raise ValueError(f"duplicate frontend rows: {duplicate_keys[:5]}")
    expected_keys = {(a, d) for a in ASRS for d in DIARIZERS}
    incomplete = {audio_id: sorted(expected_keys - set(cells)) for audio_id, cells in by_audio.items() if set(cells) != expected_keys}
    if incomplete:
        raise ValueError(f"frontend matrix is incomplete: {list(incomplete.items())[:3]}")

    field_reports: dict[str, Any] = {}
    all_contrasts: list[dict[str, Any]] = []
    for field_index, field in enumerate(fields):
        missing_by_cell: Counter[str] = Counter()
        complete_ids = []
        for audio_id, cells in by_audio.items():
            valid = True
            for (asr, diarizer), row in cells.items():
                if numeric(row.get(field)) is None:
                    missing_by_cell[f"{asr}|{diarizer}"] += 1
                    valid = False
            if valid:
                complete_ids.append(audio_id)

        cell_means = {}
        for asr in ASRS:
            for diarizer in DIARIZERS:
                values = [numeric(by_audio[audio_id][(asr, diarizer)].get(field)) for audio_id in by_audio]
                finite = [value for value in values if value is not None]
                cell_means[f"{asr}|{diarizer}"] = {
                    "n": len(finite), "mean": sum(finite) / len(finite) if finite else None,
                }

        contrasts = []
        # ASR main effects, recording-paired after averaging the two diarizers.
        for asr in ASRS:
            if asr == REFERENCE_ASR:
                continue
            differences = []
            for audio_id in complete_ids:
                cells = by_audio[audio_id]
                candidate = sum(numeric(cells[(asr, d)][field]) for d in DIARIZERS) / len(DIARIZERS)  # type: ignore[arg-type]
                reference = sum(numeric(cells[(REFERENCE_ASR, d)][field]) for d in DIARIZERS) / len(DIARIZERS)  # type: ignore[arg-type]
                differences.append(candidate - reference)
            result = {
                "field": field,
                "contrast_family": "asr_main_effect",
                "contrast": f"{asr} minus {REFERENCE_ASR}",
                **paired_inference(
                    differences, bootstrap_iterations=bootstrap_iterations,
                    permutation_iterations=permutation_iterations,
                    seed=seed + field_index * 10007 + len(contrasts) * 101,
                ),
            }
            contrasts.append(result); all_contrasts.append(result)

        # Diarizer main effect, averaged over all four ASRs.
        differences = []
        for audio_id in complete_ids:
            cells = by_audio[audio_id]
            nemo = sum(numeric(cells[(a, "nemo-sortformer")][field]) for a in ASRS) / len(ASRS)  # type: ignore[arg-type]
            pyannote = sum(numeric(cells[(a, "pyannote-3.1")][field]) for a in ASRS) / len(ASRS)  # type: ignore[arg-type]
            differences.append(nemo - pyannote)
        result = {
            "field": field,
            "contrast_family": "diarizer_main_effect",
            "contrast": "nemo-sortformer minus pyannote-3.1",
            **paired_inference(
                differences, bootstrap_iterations=bootstrap_iterations,
                permutation_iterations=permutation_iterations,
                seed=seed + field_index * 10007 + 701,
            ),
        }
        contrasts.append(result); all_contrasts.append(result)

        # Every full cell against the canonical large-v3/pyannote frontend.
        for asr in ASRS:
            for diarizer in DIARIZERS:
                if (asr, diarizer) == (REFERENCE_ASR, REFERENCE_DIARIZER):
                    continue
                differences = [
                    numeric(by_audio[audio_id][(asr, diarizer)][field])
                    - numeric(by_audio[audio_id][(REFERENCE_ASR, REFERENCE_DIARIZER)][field])  # type: ignore[operator]
                    for audio_id in complete_ids
                ]
                result = {
                    "field": field,
                    "contrast_family": "cell_vs_canonical",
                    "contrast": f"{asr}|{diarizer} minus {REFERENCE_ASR}|{REFERENCE_DIARIZER}",
                    **paired_inference(
                        differences, bootstrap_iterations=bootstrap_iterations,
                        permutation_iterations=permutation_iterations,
                        seed=seed + field_index * 10007 + 1001 + len(contrasts) * 101,
                    ),
                }
                contrasts.append(result); all_contrasts.append(result)
        field_reports[field] = {
            "num_recordings_total": len(by_audio),
            "num_recordings_balanced_complete": len(complete_ids),
            "missing_by_cell": dict(sorted(missing_by_cell.items())),
            "cell_means_available_case": cell_means,
            "contrasts_balanced_complete_case": contrasts,
        }

    benjamini_hochberg(all_contrasts)
    return {
        "schema_version": "aegisvox.frontend_objective_contrasts.v1",
        "status": "complete",
        "num_recordings": len(by_audio),
        "num_frontend_cells": len(expected_keys),
        "fields": field_reports,
        "multiplicity": {
            "method": "Benjamini-Hochberg",
            "family": "all objective frontend contrasts in this report",
            "num_tests": len([row for row in all_contrasts if row.get("raw_p_two_sided") is not None]),
        },
        "claim_boundary": (
            "Recording-paired objective frontend contrasts only. Complete-case denominators are "
            "reported per field; this report does not substitute for ITT clinical-outcome models."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fields", nargs="*", default=list(DEFAULT_FIELDS))
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--permutation-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2701)
    args = parser.parse_args()
    rows = read_jsonl(args.registry_jsonl)
    report = build_report(
        rows,
        fields=list(args.fields),
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed,
    )
    report["input"] = {
        "path": str(args.registry_jsonl),
        "sha256": hashlib.sha256(args.registry_jsonl.read_bytes()).hexdigest(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "num_fields": len(report["fields"]), "num_tests": report["multiplicity"]["num_tests"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
