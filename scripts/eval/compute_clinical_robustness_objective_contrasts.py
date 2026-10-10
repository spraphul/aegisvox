#!/usr/bin/env python3
"""Compute exact recording-paired objective contrasts for Study-11 robustness.

The unperturbed AegisVox-F5 output is the within-recording reference.  Each
acoustic stress is compared independently against that reference.  The report
keeps the full intention-to-transform denominator visible, uses recording-level
paired bootstrap confidence intervals, sign-flip randomization tests, and one
Benjamini-Hochberg family over the predeclared objective contrasts.

This module intentionally does *not* treat ASR string matches or embeddings as
clinical outcomes.  Clinical CAP, chronology, action, and relexicalization are
handled by their separately qualified adjudication protocols.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_CONDITION = "aegisvox_f5_unperturbed"
STRESS_CONDITIONS = (
    "room_noise_snr12",
    "telephony_mulaw8k",
    "mic_bandpass_gain_minus6db",
)
EXPECTED_CONDITIONS = (REFERENCE_CONDITION, *STRESS_CONDITIONS)

# Direction is expressed from the standpoint of degradation: +1 means a raw
# increase is worse; -1 means a raw decrease is worse.  The report preserves
# both raw candidate-minus-reference differences and direction-normalized
# degradation differences so readers never have to infer sign conventions.
DEFAULT_ENDPOINTS: dict[str, dict[str, Any]] = {
    "matched_asr_wer": {"higher_is_better": False, "family": "transcript"},
    "matched_asr_cer": {"higher_is_better": False, "family": "transcript"},
    "matched_asr_token_recall": {"higher_is_better": True, "family": "transcript"},
    "matched_asr_token_precision": {"higher_is_better": True, "family": "transcript"},
    "official_der": {"higher_is_better": False, "family": "diarization"},
    "official_jer": {"higher_is_better": False, "family": "diarization"},
    "speech_activity_f1": {"higher_is_better": True, "family": "diarization"},
    "speaker_count_abs_diff": {"higher_is_better": False, "family": "diarization"},
    "speaker_count_exact": {"higher_is_better": True, "family": "diarization"},
    "turn_count_ratio_abs_error": {"higher_is_better": False, "family": "diarization"},
    "duration_ratio_abs_error": {"higher_is_better": False, "family": "completeness"},
    "duration_delta_sec": {"higher_is_better": False, "family": "completeness"},
    "final_silence_fraction": {"higher_is_better": None, "family": "audio_quality"},
    "final_clipped_sample_fraction": {"higher_is_better": False, "family": "audio_quality"},
    "final_repetition_fraction": {"higher_is_better": False, "family": "audio_quality"},
    "rms_delta_abs_db": {"higher_is_better": False, "family": "audio_quality"},
    "active_pipeline_rtf": {"higher_is_better": False, "family": "runtime"},
    "stage_failure_events": {"higher_is_better": False, "family": "runtime"},
    "stage_retry_attempts": {"higher_is_better": False, "family": "runtime"},
    "retrieval_constraints_pass": {"higher_is_better": True, "family": "retrieval"},
    "source_target_threshold_failures": {"higher_is_better": False, "family": "retrieval"},
    "target_target_threshold_failures": {"higher_is_better": False, "family": "retrieval"},
    "duplicate_target_assignment_rate": {"higher_is_better": False, "family": "retrieval"},
    "ecapa_max_source_converted_similarity": {"higher_is_better": False, "family": "privacy"},
    "ecapa_min_target_converted_similarity": {"higher_is_better": True, "family": "privacy"},
    "ecapa_source_linkability_flags": {"higher_is_better": False, "family": "privacy"},
    "xvector_max_source_converted_similarity": {"higher_is_better": False, "family": "privacy"},
}


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


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def paired_inference(
    differences: list[float], *, bootstrap_iterations: int,
    permutation_iterations: int, seed: int,
) -> dict[str, Any]:
    n = len(differences)
    if n == 0:
        return {
            "n_paired": 0,
            "mean_difference": None,
            "median_difference": None,
            "ci95_low": None,
            "ci95_high": None,
            "raw_p_two_sided": None,
        }
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_iterations, dtype=np.float64)
    # Chunk to avoid allocating iterations x 390 for larger future cohorts.
    chunk = 1000
    for start in range(0, bootstrap_iterations, chunk):
        stop = min(start + chunk, bootstrap_iterations)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    observed = abs(float(values.mean()))
    extreme = 0
    for start in range(0, permutation_iterations, chunk):
        stop = min(start + chunk, permutation_iterations)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(stop - start, n))
        permuted = (signs * values).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted) >= observed - 1e-15))
    return {
        "n_paired": n,
        "mean_difference": float(values.mean()),
        "median_difference": percentile(values, 0.5),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
        "raw_p_two_sided": (extreme + 1) / (permutation_iterations + 1),
        "bootstrap_iterations": bootstrap_iterations,
        "permutation_iterations": permutation_iterations,
        "resampling_unit": "source_recording",
    }


def benjamini_hochberg(rows: list[dict[str, Any]]) -> None:
    valid = [
        (i, float(row["raw_p_two_sided"]))
        for i, row in enumerate(rows)
        if row.get("raw_p_two_sided") is not None
    ]
    valid.sort(key=lambda pair: pair[1])
    m = len(valid)
    adjusted: dict[int, float] = {}
    running = 1.0
    for reverse_index, (index, p_value) in enumerate(reversed(valid), 1):
        rank = m - reverse_index + 1
        running = min(running, p_value * m / rank)
        adjusted[index] = min(1.0, running)
    for i, row in enumerate(rows):
        row["fdr_bh_q"] = adjusted.get(i)


def build_report(
    rows: list[dict[str, Any]], *, endpoints: dict[str, dict[str, Any]],
    expected_rows: int, bootstrap_iterations: int,
    permutation_iterations: int, seed: int,
    reference_condition: str = REFERENCE_CONDITION,
    stress_conditions: tuple[str, ...] = STRESS_CONDITIONS,
) -> dict[str, Any]:
    expected_conditions = (reference_condition, *stress_conditions)
    if not reference_condition or not stress_conditions or len(set(expected_conditions)) != len(expected_conditions):
        raise ValueError("reference/stress conditions must be non-empty and unique")
    matrix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        audio_id = str(row.get("audio_id") or "")
        condition = str(row.get("condition") or "")
        if not audio_id or condition not in expected_conditions:
            raise ValueError("invalid audio_id or condition")
        if condition in matrix[audio_id]:
            raise ValueError(f"duplicate row: {audio_id}/{condition}")
        matrix[audio_id][condition] = row
    if len(matrix) != expected_rows:
        raise ValueError(f"expected {expected_rows} source recordings, got {len(matrix)}")
    expected = set(expected_conditions)
    incomplete = {audio_id: sorted(expected - set(cells)) for audio_id, cells in matrix.items() if set(cells) != expected}
    if incomplete:
        raise ValueError(f"condition matrix incomplete: {list(incomplete.items())[:5]}")

    all_contrasts: list[dict[str, Any]] = []
    endpoint_reports: dict[str, Any] = {}
    for endpoint_index, (field, metadata) in enumerate(endpoints.items()):
        available_means: dict[str, dict[str, Any]] = {}
        missing_counts: dict[str, int] = {}
        for condition in expected_conditions:
            values = [numeric(cells[condition].get(field)) for cells in matrix.values()]
            finite = [value for value in values if value is not None]
            available_means[condition] = {
                "n_available": len(finite),
                "n_itt": expected_rows,
                "mean": float(np.mean(finite)) if finite else None,
            }
            missing_counts[condition] = expected_rows - len(finite)

        contrasts: list[dict[str, Any]] = []
        for condition_index, condition in enumerate(stress_conditions):
            raw_differences: list[float] = []
            pair_ids: list[str] = []
            for audio_id, cells in matrix.items():
                reference = numeric(cells[reference_condition].get(field))
                candidate = numeric(cells[condition].get(field))
                if reference is None or candidate is None:
                    continue
                raw_differences.append(candidate - reference)
                pair_ids.append(audio_id)
            inference = paired_inference(
                raw_differences,
                bootstrap_iterations=bootstrap_iterations,
                permutation_iterations=permutation_iterations,
                seed=seed + endpoint_index * 10007 + condition_index * 1009,
            )
            higher_is_better = metadata.get("higher_is_better")
            degradation_multiplier = None if higher_is_better is None else (-1.0 if higher_is_better else 1.0)
            record = {
                "endpoint": field,
                "endpoint_family": metadata["family"],
                "reference_condition": reference_condition,
                "candidate_condition": condition,
                "contrast": f"{condition} minus {reference_condition}",
                "n_itt": expected_rows,
                "n_pair_missing": expected_rows - len(pair_ids),
                "higher_is_better": higher_is_better,
                **inference,
            }
            if degradation_multiplier is None or inference["mean_difference"] is None:
                record["mean_degradation"] = None
                record["degradation_ci95_low"] = None
                record["degradation_ci95_high"] = None
            else:
                record["mean_degradation"] = degradation_multiplier * inference["mean_difference"]
                bounds = sorted([
                    degradation_multiplier * inference["ci95_low"],
                    degradation_multiplier * inference["ci95_high"],
                ])
                record["degradation_ci95_low"], record["degradation_ci95_high"] = bounds
            contrasts.append(record)
            all_contrasts.append(record)
        endpoint_reports[field] = {
            "endpoint_family": metadata["family"],
            "higher_is_better": metadata.get("higher_is_better"),
            "available_case_condition_summaries": available_means,
            "missing_by_condition": missing_counts,
            "paired_contrasts": contrasts,
        }

    benjamini_hochberg(all_contrasts)
    return {
        "schema_version": "aegisvox.clinical_robustness_objective_contrasts.v1",
        "status": "complete",
        "reference_condition": reference_condition,
        "stress_conditions": list(stress_conditions),
        "num_recordings_itt": expected_rows,
        "num_registry_rows": len(rows),
        "endpoints": endpoint_reports,
        "multiplicity": {
            "method": "Benjamini-Hochberg",
            "family": "all predeclared objective acoustic-stress contrasts in this report",
            "num_tests": len([row for row in all_contrasts if row.get("raw_p_two_sided") is not None]),
        },
        "inference": {
            "comparison": f"within-source declared stress minus {reference_condition}",
            "bootstrap": "recording-paired nonparametric bootstrap percentile 95% CI",
            "test": "recording-paired two-sided sign-flip randomization test",
            "seed": seed,
        },
        "claim_boundary": (
            "Objective/privacy acoustic-robustness contrasts only. The full ITT denominator and "
            "pairwise missingness are explicit. ASR string evidence and acoustic embeddings are not "
            "clinical endpoints; qualified CAP/chronology/action/relex adjudication is reported separately."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Study 11 objective acoustic-robustness contrasts", "",
        f"ITT population: **{report['num_recordings_itt']} recordings**. Values are candidate minus unperturbed; positive normalized degradation is worse.",
        "",
        "| Endpoint | Stress | n paired / ITT | Raw difference (95% CI) | Normalized degradation | BH q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for field, endpoint in report["endpoints"].items():
        for row in endpoint["paired_contrasts"]:
            def fmt(value: Any) -> str:
                return "NA" if value is None else f"{float(value):.4f}"
            raw = f"{fmt(row['mean_difference'])} [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]"
            deg = fmt(row.get("mean_degradation"))
            lines.append(
                f"| `{field}` | `{row['candidate_condition']}` | {row['n_paired']} / {row['n_itt']} | {raw} | {deg} | {fmt(row.get('fdr_bh_q'))} |"
            )
    lines.extend(["", f"> {report['claim_boundary']}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-jsonl", type=Path, required=True)
    parser.add_argument("--reference-condition", default=REFERENCE_CONDITION)
    parser.add_argument("--stress-condition", action="append", dest="stress_conditions")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=390)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--permutation-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=11027)
    args = parser.parse_args()
    report = build_report(
        read_jsonl(args.registry_jsonl), endpoints=DEFAULT_ENDPOINTS,
        expected_rows=args.expected_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        seed=args.seed, reference_condition=args.reference_condition,
        stress_conditions=tuple(args.stress_conditions or STRESS_CONDITIONS),
    )
    report["input"] = {
        "path": str(args.registry_jsonl),
        "sha256": hashlib.sha256(args.registry_jsonl.read_bytes()).hexdigest(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(report))
    print(json.dumps({"status": "complete", "num_tests": report["multiplicity"]["num_tests"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
