#!/usr/bin/env python3
"""Compute measured runtime/latency/system metrics for AegisVox eval runs.

Paper-facing runtime is the sum of requested-to-completed service intervals
recorded in each run manifest, with the intentionally local LLM bridge duration
replaced by its own measured monotonic latency. File-mtime lifecycle spans are
not reported as runtime because the two-stage experiment intentionally waits
between ORD and local execution.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from common import (
    input_audio_path,
    load_json,
    load_summary_rows,
    mean,
    percentile,
    row_completed,
    run_dir_from_summary_row,
    source_dataset_id,
    source_language,
    wav_info,
    write_csv,
    write_json,
)


OUTPUT_PIPELINE_EVENT_STAGES = {
    "diarization": "diarization",
    "segmentation": "segmentation",
    "speaker_profiling": "speaker_profile",
    "assignment": "pseudo_assignment",
    "voice_conversion": "voice_conversion",
    "stitching": "stitching",
    "asr_alignment": "asr",
    "text_deid": "text_deid",
    # Workflow event names are speech_edit.requested/completed even though the
    # worker service is called speech_editing. Using the service name here
    # silently made every full run appear to miss its final required timing.
    "speech_editing": "speech_edit",
}


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def collect_manifest_stage_durations(run_dir: Path) -> dict[str, float]:
    manifest = load_json(run_dir / "manifest.json", {})
    events = manifest.get("events") if isinstance(manifest, dict) else []
    requested: dict[str, float] = {}
    durations: dict[str, float] = {}
    for event in events or []:
        event_type = str(event.get("event_type") or "")
        ts = parse_time(event.get("emitted_at"))
        if ts is None or "." not in event_type:
            continue
        stage, suffix = event_type.rsplit(".", 1)
        if suffix == "requested":
            requested[stage] = ts
        elif suffix == "completed" and stage in requested:
            durations[f"manifest_{stage}_runtime_sec"] = max(0.0, ts - requested[stage])
    return durations


def collect_manifest_execution_audit(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json", {})
    events = manifest.get("events") if isinstance(manifest, dict) else []
    requested: dict[str, int] = {}
    completed: dict[str, int] = {}
    failed: dict[str, int] = {}
    model_versions: dict[str, set[str]] = {}
    service_versions: dict[str, set[str]] = {}
    for event in events or []:
        event_type = str(event.get("event_type") or "")
        if "." in event_type:
            stage, suffix = event_type.rsplit(".", 1)
            target = requested if suffix == "requested" else completed if suffix == "completed" else failed if suffix == "failed" else None
            if target is not None:
                target[stage] = target.get(stage, 0) + 1
        payload = event.get("payload") or {}
        for key, out in (("model_versions", model_versions), ("service_versions", service_versions)):
            values = payload.get(key) or {}
            if isinstance(values, dict):
                for name, value in values.items():
                    out.setdefault(str(name), set()).add(str(value))
    retries = {stage: max(0, count - 1) for stage, count in requested.items()}
    return {
        "manifest_status": manifest.get("status") if isinstance(manifest, dict) else None,
        "manifest_failure_reason": manifest.get("failure_reason") if isinstance(manifest, dict) else None,
        "stage_attempt_counts": requested,
        "stage_completion_counts": completed,
        "stage_failure_counts": failed,
        "num_stage_retry_attempts": sum(retries.values()),
        "num_stage_failure_events": sum(failed.values()),
        "model_versions": {key: sorted(values) for key, values in model_versions.items()},
        "service_versions": {key: sorted(values) for key, values in service_versions.items()},
    }


def artifact_counts(path: Path, keys: tuple[str, ...]) -> dict[str, int]:
    data = load_json(path, {})
    if not isinstance(data, dict):
        return {key: 0 for key in keys}
    return {key: int(data.get(key) or 0) for key in keys}


def sum_segment_latencies(path: Path) -> tuple[float | None, int]:
    data = load_json(path, {})
    vals: list[float] = []
    if isinstance(data, dict):
        for seg in data.get("segments") or []:
            latency = (((seg.get("backend_metadata") or {}).get("service_response") or {}).get("latency_sec"))
            if isinstance(latency, (int, float)):
                vals.append(float(latency))
    return (sum(vals), len(vals)) if vals else (None, 0)


def sum_voicecraft_job_latencies(path: Path) -> tuple[float | None, int]:
    data = load_json(path, {})
    vals: list[float] = []
    if isinstance(data, dict):
        for job in data.get("jobs") or []:
            latency = (((job.get("backend_metadata") or {}).get("service_response") or {}).get("latency_sec"))
            if isinstance(latency, (int, float)):
                vals.append(float(latency))
    return (sum(vals), len(vals)) if vals else (None, 0)


def sum_fallback_vc_latencies(path: Path) -> tuple[float | None, int]:
    data = load_json(path, {})
    vals: list[float] = []
    if isinstance(data, dict):
        for row in data.get("rows") or []:
            latency = (((row.get("voice_conversion_metadata") or {}).get("service_response") or {}).get("latency_sec"))
            if isinstance(latency, (int, float)):
                vals.append(float(latency))
    return (sum(vals), len(vals)) if vals else (None, 0)


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    run_dir = run_dir_from_summary_row(row)
    input_duration = wav_info(input_audio_path(run_dir))["duration_sec"]
    manifest = load_json(run_dir / "manifest.json", {})
    deid = load_json(run_dir / "deid_text.json", {})

    diar_raw = load_json(run_dir / "diarization.raw.json", {})
    validation = load_json(run_dir / "voicecraft_latest_final_validation_fasterwhisper.json", {})
    direct_validation = load_json(run_dir / "voicecraft_direct_input_latest_validation_fasterwhisper.json", {})
    fallback = load_json(run_dir / "lexical_fallback_after_voicecraft_validation_latest_report.json", {})
    conversion_path = run_dir / "segments.converted_direct_input_v1.json"
    conversion_latency, num_converted = sum_segment_latencies(conversion_path)
    if conversion_latency is None:
        conversion_path = run_dir / "segments.converted.direct_input_v1.json"
        conversion_latency, num_converted = sum_segment_latencies(conversion_path)
    if conversion_latency is None:
        conversion_path = run_dir / "segments.converted.json"
        conversion_latency, num_converted = sum_segment_latencies(conversion_path)
    conversion_counts = artifact_counts(conversion_path, ("num_failed", "num_resumed_existing"))
    voicecraft_path = run_dir / "voicecraft_edit_jobs_direct_input_v1_utterance_window_v4.json"
    voicecraft_counts = artifact_counts(voicecraft_path, ("num_failed_jobs", "num_resumed_existing"))
    voicecraft_latency, num_voicecraft_jobs = sum_voicecraft_job_latencies(run_dir / "voicecraft_edit_jobs_direct_input_v1_utterance_window_v4.json")
    fallback_vc_latency, num_fallback_vc = sum_fallback_vc_latencies(run_dir / "lexical_fallback_after_voicecraft_validation_latest_report.json")
    stage_durations = collect_manifest_stage_durations(run_dir)
    execution_audit = collect_manifest_execution_audit(run_dir)
    local_llm_latency = (
        (deid.get("local_bridge_provenance") or {}).get("latency_sec")
        if isinstance(deid, dict)
        else None
    )
    measured_stage_durations: dict[str, float] = {}
    for stage, event_stage in OUTPUT_PIPELINE_EVENT_STAGES.items():
        value = stage_durations.get(f"manifest_{event_stage}_runtime_sec")
        if stage == "text_deid" and isinstance(local_llm_latency, (int, float)):
            value = float(local_llm_latency)
        if isinstance(value, (int, float)):
            measured_stage_durations[stage] = float(value)
    measured_active_runtime = sum(measured_stage_durations.values()) if measured_stage_durations else None
    missing_stage_timings = [stage for stage in OUTPUT_PIPELINE_EVENT_STAGES if stage not in measured_stage_durations]

    structured_latency = sum(v for v in [
        ((diar_raw.get("metadata") or {}).get("latency_sec") if isinstance(diar_raw, dict) else None),
        conversion_latency,
        voicecraft_latency,
        (direct_validation.get("latency_sec") if isinstance(direct_validation, dict) else None),
        fallback_vc_latency,
        (validation.get("latency_sec") if isinstance(validation, dict) else None),
    ] if isinstance(v, (int, float)))

    return {
        "audio_id": row.get("audio_id"),
        "run_id": row.get("run_id"),
        "source_dataset_id": source_dataset_id(row),
        "source_language": source_language(row),
        "completed": row_completed(row),
        "batch_returncode": row.get("returncode"),
        "stage_a_gpu_group": row.get("gpu_group"),
        "input_duration_sec": input_duration,
        "measured_active_pipeline_runtime_sec": measured_active_runtime,
        "measured_active_pipeline_rtf": (measured_active_runtime / input_duration) if isinstance(measured_active_runtime, (int, float)) and input_duration else None,
        "num_required_stage_timings": len(OUTPUT_PIPELINE_EVENT_STAGES),
        "num_present_stage_timings": len(measured_stage_durations),
        "stage_timing_coverage": len(measured_stage_durations) / len(OUTPUT_PIPELINE_EVENT_STAGES),
        "missing_stage_timings": missing_stage_timings,
        "measured_stage_durations": measured_stage_durations,
        "local_llm_latency_sec": local_llm_latency,
        "known_structured_latency_sec_sum": structured_latency,
        "known_structured_latency_rtf_sum": (structured_latency / input_duration) if input_duration else None,
        "diarization_latency_sec": ((diar_raw.get("metadata") or {}).get("latency_sec") if isinstance(diar_raw, dict) else None),
        "voice_conversion_latency_sec_sum": conversion_latency,
        "num_converted_segments": num_converted,
        "voicecraft_latency_sec_sum": voicecraft_latency,
        "num_voicecraft_jobs": num_voicecraft_jobs,
        "direct_validation_latency_sec": direct_validation.get("latency_sec") if isinstance(direct_validation, dict) else None,
        "final_validation_latency_sec": validation.get("latency_sec") if isinstance(validation, dict) else None,
        "fallback_voice_conversion_latency_sec_sum": fallback_vc_latency,
        "num_fallback_voice_conversions": num_fallback_vc,
        "num_fallback_insertions": fallback.get("num_fallback_insertions") if isinstance(fallback, dict) else row.get("fallback_insertions"),
        "num_validation_fail_like": validation.get("num_fail_like") if isinstance(validation, dict) else row.get("num_fail_like"),
        "num_voice_conversion_failures": conversion_counts["num_failed"],
        "num_voice_conversion_resumed_existing": conversion_counts["num_resumed_existing"],
        "num_voicecraft_failed_jobs": voicecraft_counts["num_failed_jobs"],
        "num_voicecraft_resumed_existing": voicecraft_counts["num_resumed_existing"],
        **execution_audit,
        "manifest_stage_durations": stage_durations,
        "claim_boundary": "Measured active service time excludes intentional inter-stage waiting and batch queue delay. Report stage-timing coverage and hardware/concurrency alongside RTF.",
    }


def aggregate_core(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_values: dict[str, list[float]] = {stage: [] for stage in OUTPUT_PIPELINE_EVENT_STAGES}
    for row in rows:
        for stage, value in (row.get("measured_stage_durations") or {}).items():
            if stage in stage_values and isinstance(value, (int, float)):
                stage_values[stage].append(float(value))
    return {
        "num_rows": len(rows),
        "num_completed": sum(1 for row in rows if row.get("completed")),
        "num_rows_with_complete_stage_timing": sum(1 for row in rows if row.get("stage_timing_coverage") == 1.0),
        "mean_measured_active_pipeline_runtime_sec": mean([row.get("measured_active_pipeline_runtime_sec") for row in rows]),
        "p50_measured_active_pipeline_runtime_sec": percentile([row.get("measured_active_pipeline_runtime_sec") for row in rows], 0.50),
        "p90_measured_active_pipeline_runtime_sec": percentile([row.get("measured_active_pipeline_runtime_sec") for row in rows], 0.90),
        "p95_measured_active_pipeline_runtime_sec": percentile([row.get("measured_active_pipeline_runtime_sec") for row in rows], 0.95),
        "mean_measured_active_pipeline_rtf": mean([row.get("measured_active_pipeline_rtf") for row in rows]),
        "p50_measured_active_pipeline_rtf": percentile([row.get("measured_active_pipeline_rtf") for row in rows], 0.50),
        "p90_measured_active_pipeline_rtf": percentile([row.get("measured_active_pipeline_rtf") for row in rows], 0.90),
        "p95_measured_active_pipeline_rtf": percentile([row.get("measured_active_pipeline_rtf") for row in rows], 0.95),
        "mean_stage_timing_coverage": mean([row.get("stage_timing_coverage") for row in rows]),
        "mean_known_structured_latency_sec_sum": mean([row.get("known_structured_latency_sec_sum") for row in rows]),
        "mean_known_structured_latency_rtf_sum": mean([row.get("known_structured_latency_rtf_sum") for row in rows]),
        "total_fallback_insertions": sum(int(row.get("num_fallback_insertions") or 0) for row in rows),
        "total_validation_fail_like": sum(int(row.get("num_validation_fail_like") or 0) for row in rows),
        "total_stage_retry_attempts": sum(int(row.get("num_stage_retry_attempts") or 0) for row in rows),
        "total_stage_failure_events": sum(int(row.get("num_stage_failure_events") or 0) for row in rows),
        "num_rows_with_stage_retries": sum(int(row.get("num_stage_retry_attempts") or 0) > 0 for row in rows),
        "num_rows_with_stage_failure_events": sum(int(row.get("num_stage_failure_events") or 0) > 0 for row in rows),
        "num_rows_with_nonzero_batch_returncode": sum(
            isinstance(row.get("batch_returncode"), int) and row.get("batch_returncode") != 0 for row in rows
        ),
        "total_voice_conversion_failures": sum(int(row.get("num_voice_conversion_failures") or 0) for row in rows),
        "total_voicecraft_failed_jobs": sum(int(row.get("num_voicecraft_failed_jobs") or 0) for row in rows),
        "total_resumed_existing_outputs": sum(
            int(row.get("num_voice_conversion_resumed_existing") or 0)
            + int(row.get("num_voicecraft_resumed_existing") or 0)
            for row in rows
        ),
        "stage_latency_seconds": {
            stage: {
                "n": len(values),
                "mean": mean(values),
                "median": percentile(values, 0.50),
                "p90": percentile(values, 0.90),
                "p95": percentile(values, 0.95),
            }
            for stage, values in stage_values.items()
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = aggregate_core(rows)
    out["by_source_dataset"] = {
        group: aggregate_core([row for row in rows if row.get("source_dataset_id") == group])
        for group in sorted({str(row.get("source_dataset_id")) for row in rows})
    }
    out["by_source_language"] = {
        group: aggregate_core([row for row in rows if row.get("source_language") == group])
        for group in sorted({str(row.get("source_language")) for row in rows})
    }
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Measured Runtime / Latency Metrics", "", "**Claim boundary:** active requested-to-completed service time; intentional stage waiting and batch queue delay are excluded. Stage coverage is reported explicitly.", "", "## Aggregate", ""]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | input sec | active sec | active RTF | stage coverage | structured sec sum | fallbacks |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        lines.append(f"| {row.get('audio_id')} | {_fmt(row.get('input_duration_sec'))} | {_fmt(row.get('measured_active_pipeline_runtime_sec'))} | {_fmt(row.get('measured_active_pipeline_rtf'))} | {_fmt(row.get('stage_timing_coverage'))} | {_fmt(row.get('known_structured_latency_sec_sum'))} | {row.get('num_fallback_insertions')} |")
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--execution-provenance-json", type=Path)
    args = parser.parse_args()
    rows = [row_metrics(row) for row in load_summary_rows(args.run_summary)]
    provenance = load_json(args.execution_provenance_json, {}) if args.execution_provenance_json else {}
    payload = {
        "schema_version": "aegisvox.eval.runtime_system.v3",
        "run_summary": str(args.run_summary),
        "aggregate": aggregate(rows),
        "rows": rows,
        "required_output_stage_event_map": OUTPUT_PIPELINE_EVENT_STAGES,
        "execution_provenance": provenance,
        "execution_provenance_json": str(args.execution_provenance_json) if args.execution_provenance_json else None,
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
