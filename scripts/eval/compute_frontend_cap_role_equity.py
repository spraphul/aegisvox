#!/usr/bin/env python3
"""Report CAP/role clinical outcomes across frozen population/difficulty strata."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ENDPOINTS = (
    "any_critical_major_cap_error",
    "any_role_attribution_error",
    "any_major_unsupported_target_assertion",
    "conversation_major_error",
    "conversation_uncertain",
)
BASE_DIMENSIONS = (
    "dataset_id", "language", "duration_quartile", "speaker_count_bin",
    "edit_count_bin", "max_edit_duration_bin",
    "speech_rate_quartile", "backchannel_density_quartile",
    "nonspeech_fraction_quartile", "maximum_nonspeech_gap_quartile",
    "asr_error_bin", "diarization_error_bin",
    "input_rms_quartile", "input_silence_fraction_quartile",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0: return None
    p = k / n; denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return [max(0.0, center-half), min(1.0, center+half)]


def _duration_bin(value: Any) -> str:
    if value is None: return "missing"
    number = float(value)
    if not math.isfinite(number) or number < 0: return "missing"
    if number == 0: return "no_edit"
    if number < 1: return "lt_1s"
    if number < 2: return "1_to_lt_2s"
    if number < 5: return "2_to_lt_5s"
    if number < 10: return "5_to_lt_10s"
    return "ge_10s"


def _unique_numeric(rows: list[dict[str, Any]], field: str, keys: tuple[str, ...]) -> list[float]:
    observed: dict[tuple[str, ...], float] = {}
    for row in rows:
        value = row.get(field)
        if value is None: continue
        number = float(value)
        if not math.isfinite(number): continue
        key = tuple(str(row.get(name) or "") for name in keys)
        if key in observed and not math.isclose(observed[key], number, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{field}: inconsistent duplicate value for {key}")
        observed[key] = number
    return list(observed.values())


def _quartile_thresholds(values: list[float]) -> tuple[float, float, float] | None:
    if not values: return None
    ordered = sorted(values)
    def pick(fraction: float) -> float:
        # Nearest-rank, deterministic and dependency-free. Strata are source-only.
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]
    return pick(.25), pick(.50), pick(.75)


def _quartile(value: Any, thresholds: tuple[float, float, float] | None) -> str:
    if value is None or thresholds is None: return "missing"
    number = float(value)
    if not math.isfinite(number): return "missing"
    if number <= thresholds[0]: return "Q1"
    if number <= thresholds[1]: return "Q2"
    if number <= thresholds[2]: return "Q3"
    return "Q4"


def _binary_median_bin(value: Any, threshold: float | None) -> str:
    if value is None or threshold is None: return "missing"
    number = float(value)
    if not math.isfinite(number): return "missing"
    return "low_le_median" if number <= threshold else "high_gt_median"


def _endpoint_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = {}
    for endpoint in ENDPOINTS:
        values = [item.get(endpoint) for item in items]
        if any(value not in {0, 1} for value in values):
            raise ValueError(f"{endpoint}: incomplete binary endpoint")
        events = sum(values)
        endpoints[endpoint] = {
            "n": len(values), "events": events,
            "rate": events / len(values), "wilson95": wilson(events, len(values)),
        }
    return endpoints


def _natural_index(natural_rows: list[dict[str, Any]] | None, expected_rows: int) -> dict[str, dict[str, Any]]:
    if natural_rows is None: return {}
    result: dict[str, dict[str, Any]] = {}
    for row in natural_rows:
        audio_id = str(row.get("audio_id") or "")
        if not audio_id or audio_id in result: raise ValueError("natural-difficulty registry has blank/duplicate audio_id")
        result[audio_id] = row
    if len(result) != expected_rows: raise ValueError("natural-difficulty registry population mismatch")
    return result


def build(
    rows: list[dict[str, Any]], expected_rows: int = 390,
    natural_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    keys = {(str(row.get("cell") or ""), str(row.get("audio_id") or "")) for row in rows}
    if len(keys) != 8 * expected_rows:
        raise ValueError("expected exact eight-cell population")
    natural = _natural_index(natural_rows, expected_rows)
    enriched: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        audio_id = str(row.get("audio_id") or "")
        if natural:
            if audio_id not in natural: raise ValueError(f"natural-difficulty registry missing {audio_id}")
            for field in (
                "speech_rate_quartile", "backchannel_density_quartile",
                "nonspeech_fraction_quartile", "maximum_nonspeech_gap_quartile",
            ):
                row[field] = natural[audio_id].get(field)
        row["max_edit_duration_bin"] = _duration_bin(row.get("max_edit_duration_sec"))
        enriched.append(row)

    asr_values: dict[str, list[float]] = defaultdict(list)
    diar_values: dict[str, list[float]] = defaultdict(list)
    # Frontend errors are measured after the complete ASR x diarizer pipeline,
    # so thresholds are frozen within each of the eight paired cells rather than
    # pretending that the same ASR has identical errors under both diarizers.
    for cell in sorted({str(row.get("cell") or "missing") for row in enriched}):
        asr_values[cell] = _unique_numeric(
            [row for row in enriched if str(row.get("cell") or "missing") == cell],
            "public_reference_wer", ("audio_id",),
        )
        diar_values[cell] = _unique_numeric(
            [row for row in enriched if str(row.get("cell") or "missing") == cell],
            "official_der", ("audio_id",),
        )
    asr_thresholds = {key: (median(values) if values else None) for key, values in asr_values.items()}
    diar_thresholds = {key: (median(values) if values else None) for key, values in diar_values.items()}
    rms_q = _quartile_thresholds(_unique_numeric(enriched, "input_rms_dbfs", ("audio_id",)))
    silence_q = _quartile_thresholds(_unique_numeric(enriched, "input_silence_fraction_chunks", ("audio_id",)))
    for row in enriched:
        cell = str(row.get("cell") or "missing")
        row["asr_error_bin"] = _binary_median_bin(row.get("public_reference_wer"), asr_thresholds[cell])
        row["diarization_error_bin"] = _binary_median_bin(row.get("official_der"), diar_thresholds[cell])
        row["input_rms_quartile"] = _quartile(row.get("input_rms_dbfs"), rms_q)
        row["input_silence_fraction_quartile"] = _quartile(row.get("input_silence_fraction_chunks"), silence_q)

    cells = sorted({str(row["cell"]) for row in rows})
    output: dict[str, Any] = {}
    for cell in cells:
        cell_rows = [row for row in enriched if row["cell"] == cell]
        cell_report: dict[str, Any] = {}
        for dimension in BASE_DIMENSIONS:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in cell_rows:
                value = row.get(dimension)
                groups[str(value) if value is not None else "missing"].append(row)
            dimension_rows = {}
            for group, items in sorted(groups.items()):
                dimension_rows[group] = {"n": len(items), "endpoints": _endpoint_summary(items)}
            cell_report[dimension] = dimension_rows
        # PHI type is multi-label: a recording may appear in more than one type.
        phi_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cell_rows:
            counts = row.get("phi_type_counts") or {}
            if not isinstance(counts, dict): raise ValueError("phi_type_counts must be an object")
            active = sorted(str(key) for key, count in counts.items() if int(count) > 0)
            if not active:
                phi_groups["none"].append(row)
            else:
                phi_groups["any_phi"].append(row)
                for phi_type in active: phi_groups[phi_type].append(row)
        cell_report["phi_entity_type_multilabel"] = {
            group: {"n": len(items), "endpoints": _endpoint_summary(items)}
            for group, items in sorted(phi_groups.items())
        }
        output[cell] = cell_report
    return {
        "schema_version": "aegisvox.frontend_cap_role_equity.v1",
        "status": "complete",
        "num_cells": len(cells), "num_rows_per_cell": expected_rows,
        "dimensions": list(BASE_DIMENSIONS) + ["phi_entity_type_multilabel"],
        "endpoints": list(ENDPOINTS),
        "cells": output,
        "stratum_definitions": {
            "max_edit_duration_bin": "maximum planned edit duration per recording; no-edit retained separately",
            "asr_error_bin": "public-reference WER split at the median within each frozen ASR x diarizer cell; missing reference retained",
            "diarization_error_bin": "official DER split at the median within each frozen ASR x diarizer cell; unavailable reference retained as missing",
            "input_rms_quartile": "source/input RMS dBFS nearest-rank quartiles over unique recordings",
            "input_silence_fraction_quartile": "source/input silence-fraction nearest-rank quartiles over unique recordings",
            "phi_entity_type_multilabel": "recording-level presence; recordings with multiple entity types contribute to each applicable type",
        },
        "thresholds": {
            "asr_public_reference_wer_median_by_cell": asr_thresholds,
            "diarization_official_der_median_by_cell": diar_thresholds,
            "input_rms_dbfs_quartiles": rms_q,
            "input_silence_fraction_quartiles": silence_q,
        },
        "language_claim_boundary": "English and French are reported separately; no broader multilingual claim.",
        "claim_boundary": "Model-adjudicated CAP/role strata with explicit denominators and Wilson intervals; not human ground truth.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=390)
    parser.add_argument("--natural-difficulty-registry", type=Path)
    args = parser.parse_args()
    natural_rows = read_jsonl(args.natural_difficulty_registry) if args.natural_difficulty_registry else None
    report = build(read_jsonl(args.registry_jsonl), expected_rows=args.expected_rows, natural_rows=natural_rows)
    report["input_sha256"] = hashlib.sha256(args.registry_jsonl.read_bytes()).hexdigest()
    if args.natural_difficulty_registry:
        report["natural_difficulty_input_sha256"] = hashlib.sha256(args.natural_difficulty_registry.read_bytes()).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status":"complete","num_cells":report["num_cells"]}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
