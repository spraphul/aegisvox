#!/usr/bin/env python3
"""Build the exact 4 x 390 objective registry for Study-11 acoustic robustness."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REFERENCE_CONDITION = "aegisvox_f5_unperturbed"
STRESS_CONDITIONS = (
    "room_noise_snr12",
    "telephony_mulaw8k",
    "mic_bandpass_gain_minus6db",
)
EXPECTED_CONDITIONS = (REFERENCE_CONDITION, *STRESS_CONDITIONS)

METRIC_FIELDS = {
    "matched_asr_wer_cer.json": {
        "wer": "matched_asr_wer", "cer": "matched_asr_cer",
        "token_recall_set": "matched_asr_token_recall", "token_precision_set": "matched_asr_token_precision",
    },
    "diarization_metrics.json": {
        "official_der": "official_der", "official_jer": "official_jer", "speech_activity_f1": "speech_activity_f1",
        "source_final_speaker_count_abs_diff": "speaker_count_abs_diff",
        "source_final_speaker_count_exact_match": "speaker_count_exact",
        "source_final_turn_count_ratio": "turn_count_ratio",
    },
    "phi_residual_asr.json": {
        "num_phi_edits": "num_phi_edits", "replacement_local_recall": "replacement_local_recall_auto",
        "residual_phi_local_rate": "residual_phi_local_rate_auto", "residual_phi_global_rate": "residual_phi_global_rate_auto",
        "num_warn_replacement_missing": "num_warn_replacement_missing_auto",
    },
    "audio_quality.json": {
        "final_rms_dbfs": "final_rms_dbfs", "final_silence_fraction_chunks": "final_silence_fraction",
        "final_clipped_sample_fraction": "final_clipped_sample_fraction",
        "final_repeated_adjacent_chunk_fraction": "final_repetition_fraction",
        "rms_delta_db_final_minus_input": "rms_delta_db",
    },
    "duration_completeness.json": {
        "duration_ratio_final_over_input": "duration_ratio", "absolute_duration_delta_sec": "duration_delta_sec",
    },
    "runtime_system_metrics.json": {
        "measured_active_pipeline_rtf": "active_pipeline_rtf", "measured_active_pipeline_runtime_sec": "active_pipeline_runtime_sec",
        "stage_timing_coverage": "stage_timing_coverage", "num_stage_failure_events": "stage_failure_events",
        "num_stage_retry_attempts": "stage_retry_attempts",
    },
    "retrieval_assignment.json": {
        "all_assignment_constraints_pass": "retrieval_constraints_pass",
        "num_source_target_threshold_failures": "source_target_threshold_failures",
        "num_target_target_threshold_failures": "target_target_threshold_failures",
        "duplicate_target_assignment_rate": "duplicate_target_assignment_rate",
    },
    "asv_privacy.json": {
        "max_source_converted_similarity": "ecapa_max_source_converted_similarity",
        "min_target_converted_similarity": "ecapa_min_target_converted_similarity",
        "num_source_linkability_flags": "ecapa_source_linkability_flags",
    },
    "asv_privacy_independent.json": {
        "max_source_converted_similarity": "xvector_max_source_converted_similarity",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        audio_id = str(row.get("audio_id") or "")
        if not audio_id or audio_id in result:
            raise ValueError(f"{path}: missing or duplicate audio_id={audio_id!r}")
        result[audio_id] = row
    return result


def load_metric(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing rows[]")
    return index(rows, path)


def finite(value: Any) -> float | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def build(
    *, conditions: dict[str, Path], strata_path: Path, expected_rows: int,
    condition_audits: dict[str, Path] | None = None,
    reference_condition: str = REFERENCE_CONDITION,
    stress_conditions: tuple[str, ...] = STRESS_CONDITIONS,
) -> tuple[list[dict], dict]:
    expected_conditions = (reference_condition, *stress_conditions)
    if not reference_condition or not stress_conditions or len(set(expected_conditions)) != len(expected_conditions):
        raise ValueError("reference/stress conditions must be non-empty and unique")
    if tuple(conditions) != expected_conditions:
        raise ValueError(f"conditions must be ordered exactly as {expected_conditions}, got {tuple(conditions)}")
    strata = index(read_jsonl(strata_path), strata_path)
    if len(strata) != expected_rows:
        raise ValueError(f"expected {expected_rows} strata rows, got {len(strata)}")
    expected_ids = set(strata)
    condition_audits = condition_audits or {}
    if condition_audits and set(condition_audits) != set(stress_conditions):
        raise ValueError(
            f"condition audits must cover exactly {tuple(stress_conditions)}, "
            f"got {tuple(condition_audits)}"
        )
    rows_out: list[dict] = []
    hashes: dict[str, Any] = {"strata": sha256(strata_path), "conditions": {}}
    missingness: dict[str, dict[str, int]] = {}
    pipeline_failures: dict[str, dict[str, Any]] = {}
    fields = [target for mapping in METRIC_FIELDS.values() for target in mapping.values()]

    for condition, metric_dir in conditions.items():
        completion = metric_dir / "condition_complete.json"
        if not completion.is_file():
            raise ValueError(f"{condition}: missing completion marker")
        completed = json.loads(completion.read_text())
        observed_rows = int(completed.get("num_rows", completed.get("expected_rows")) or 0)
        audit_path = condition_audits.get(condition)
        explicit_failures = 0
        audit_payload: dict[str, Any] | None = None
        if audit_path is not None:
            if not audit_path.is_file():
                raise ValueError(f"{condition}: missing condition audit {audit_path}")
            audit_payload = json.loads(audit_path.read_text())
            renderable = int(audit_payload.get("num_renderable_rows", -1))
            explicit_failures = int(audit_payload.get("num_explicit_pipeline_failures", -1))
            if not (
                audit_payload.get("schema_version") == "aegisvox.clinical_robustness_condition_complete.v1"
                and audit_payload.get("status") == "complete"
                and audit_payload.get("expected_rows") == expected_rows
                and audit_payload.get("num_failures") == 0
                and audit_payload.get("protocol_proxies_included") is False
                and renderable + explicit_failures == expected_rows
                and observed_rows == renderable
            ):
                raise ValueError(f"{condition}: invalid ITT condition audit")
            declared_metric_dir = Path(str(audit_payload.get("metric_dir") or ""))
            if declared_metric_dir.resolve() != metric_dir.resolve():
                raise ValueError(f"{condition}: metric directory differs from condition audit")
            itt_audit = Path(str(audit_payload.get("itt_population_audit") or ""))
            if not itt_audit.is_file() or sha256(itt_audit) != audit_payload.get("itt_population_audit_sha256"):
                raise ValueError(f"{condition}: missing or changed ITT population audit")
            itt = json.loads(itt_audit.read_text())
            if not (
                itt.get("schema_version") == "aegisvox.stagea.itt_population_audit.v1"
                and itt.get("status") == "complete"
                and itt.get("expected_rows") == expected_rows
                and itt.get("num_renderable") == renderable
                and itt.get("num_explicit_failures") == explicit_failures
                and not itt.get("failures")
            ):
                raise ValueError(f"{condition}: invalid ITT population accounting")
        elif condition != reference_condition:
            raise ValueError(f"{condition}: missing required ITT condition audit")
        if completed.get("status") != "complete" or observed_rows != expected_rows - explicit_failures:
            raise ValueError(f"{condition}: incomplete condition marker")
        metrics: dict[str, dict[str, dict[str, Any]]] = {}
        hashes["conditions"][condition] = {"condition_complete": sha256(completion), "metrics": {}}
        if audit_path is not None:
            hashes["conditions"][condition]["condition_audit"] = sha256(audit_path)
            hashes["conditions"][condition]["itt_population_audit"] = str(
                audit_payload["itt_population_audit_sha256"]
            )
        observed_ids: set[str] | None = None
        for filename in METRIC_FIELDS:
            path = metric_dir / filename
            if not path.is_file():
                raise ValueError(f"{condition}: missing {filename}")
            metric = load_metric(path)
            metric_ids = set(metric)
            if not metric_ids.issubset(expected_ids):
                raise ValueError(f"{condition}/{filename}: metric IDs outside frozen strata")
            if len(metric_ids) != observed_rows:
                raise ValueError(
                    f"{condition}/{filename}: observed IDs={len(metric_ids)} expected={observed_rows}"
                )
            if observed_ids is None:
                observed_ids = metric_ids
            elif metric_ids != observed_ids:
                raise ValueError(f"{condition}/{filename}: metric populations disagree")
            if audit_payload is not None:
                declared_hash = (audit_payload.get("required_metric_hashes") or {}).get(filename)
                if declared_hash != sha256(path):
                    raise ValueError(f"{condition}/{filename}: condition-audit hash mismatch")
            metrics[filename] = metric
            hashes["conditions"][condition]["metrics"][filename] = sha256(path)

        observed_ids = observed_ids or set()
        failure_ids = expected_ids - observed_ids
        if len(failure_ids) != explicit_failures:
            raise ValueError(
                f"{condition}: derived failure IDs={len(failure_ids)} declared={explicit_failures}"
            )
        pipeline_failures[condition] = {
            "num_itt": expected_rows,
            "num_renderable": len(observed_ids),
            "num_explicit_pipeline_failures": len(failure_ids),
            "pipeline_failure_rate_itt": len(failure_ids) / expected_rows,
            "failure_audio_ids": sorted(failure_ids),
        }

        condition_rows = []
        for audio_id in sorted(expected_ids):
            s = strata[audio_id]
            row: dict[str, Any] = {
                "audio_id": audio_id, "condition": condition,
                "dataset_id": s["dataset_id"], "language": s["language"],
                "duration_quartile": s.get("duration_quartile"), "speaker_count_bin": s.get("speaker_count_bin"),
                "edit_count_bin": s.get("edit_count_bin"), "edit_duration_bin": s.get("edit_duration_bin"),
            }
            row["pipeline_failure"] = audio_id in failure_ids
            row["outcome_observed"] = audio_id not in failure_ids
            for filename, mapping in METRIC_FIELDS.items():
                source = metrics[filename].get(audio_id, {})
                for source_field, target_field in mapping.items():
                    row[target_field] = finite(source.get(source_field))
            # The inferential report treats deviations from the ideal ratio or
            # zero loudness shift as errors rather than assigning an arbitrary
            # direction to the raw signed quantity.
            turn_ratio = row.get("turn_count_ratio")
            duration_ratio = row.get("duration_ratio")
            rms_delta = row.get("rms_delta_db")
            row["turn_count_ratio_abs_error"] = abs(turn_ratio - 1.0) if turn_ratio is not None else None
            row["duration_ratio_abs_error"] = abs(duration_ratio - 1.0) if duration_ratio is not None else None
            row["rms_delta_abs_db"] = abs(rms_delta) if rms_delta is not None else None
            condition_rows.append(row)
        rows_out.extend(condition_rows)
        missingness[condition] = dict(sorted(Counter(field for row in condition_rows for field in fields if row.get(field) is None).items()))

    expected_total = len(expected_conditions) * expected_rows
    if len(rows_out) != expected_total or len({(r["condition"], r["audio_id"]) for r in rows_out}) != expected_total:
        raise ValueError("registry is not an exact condition x audio_id bijection")
    summary = {
        "schema_version": "aegisvox.clinical_robustness_objective_registry.v1", "status": "complete",
        "conditions": list(expected_conditions), "reference_condition": reference_condition,
        "stress_conditions": list(stress_conditions),
        "num_conditions": len(expected_conditions), "num_rows_per_condition": expected_rows, "num_rows": len(rows_out),
        "metric_fields": fields + ["turn_count_ratio_abs_error", "duration_ratio_abs_error", "rms_delta_abs_db"],
        "missingness_by_condition": missingness, "pipeline_failures_by_condition": pipeline_failures,
        "input_hashes": hashes,
        "claim_boundary": "Exact measured objective/privacy registry. Automatic residual-PHI string metrics are retained only as audit signals; clinical and relex outcomes come from qualified adjudication.",
    }
    return rows_out, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", action="append", required=True, help="slug=metric_dir; ordered reference then three frozen stresses")
    ap.add_argument(
        "--condition-audit", action="append", default=[],
        help="stress_slug=clinical_robustness_condition_complete.json; required for every stress",
    )
    ap.add_argument("--reference-condition", default=REFERENCE_CONDITION)
    ap.add_argument("--stress-condition", action="append", dest="stress_conditions")
    ap.add_argument("--strata-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--expected-rows", type=int, default=390)
    args = ap.parse_args()
    conditions: dict[str, Path] = {}
    for raw in args.condition:
        slug, value = raw.split("=", 1); conditions[slug] = Path(value)
    condition_audits: dict[str, Path] = {}
    for raw in args.condition_audit:
        slug, value = raw.split("=", 1)
        if slug in condition_audits:
            raise ValueError(f"duplicate condition audit: {slug}")
        condition_audits[slug] = Path(value)
    rows, summary = build(
        conditions=conditions, strata_path=args.strata_jsonl, expected_rows=args.expected_rows,
        condition_audits=condition_audits,
        reference_condition=args.reference_condition,
        stress_conditions=tuple(args.stress_conditions or STRESS_CONDITIONS),
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["output_jsonl"] = str(args.output_jsonl); summary["output_sha256"] = sha256(args.output_jsonl)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "num_rows": len(rows), "output": str(args.output_jsonl)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
