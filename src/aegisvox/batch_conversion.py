"""Batch voice conversion over cleaned segments."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import redact_text
from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore
from .voice_conversion import VoiceConversionBackend, VoiceConversionService, _duration_ratio, segments_converted_payload


class BatchConversionError(RuntimeError):
    """Raised when batch conversion cannot proceed."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise AegisVoxContractError(f"expected JSON object at {path}")
    return payload


def _segment_source_audio_uri(row: dict[str, Any]) -> str:
    uri = row.get("segment_audio_uri") or row.get("source_audio_uri")
    if not uri:
        raise AegisVoxContractError(f"segment {row.get('segment_id')} missing segment_audio_uri/source_audio_uri")
    validate_artifact_uri(uri)
    return uri


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _cuda_devices(options: dict[str, Any]) -> list[str]:
    raw = options.get("cuda_devices")
    if raw is None:
        raw = options.get("cuda_visible_devices")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


@dataclass
class BatchVoiceConversionService:
    artifact_store: LocalArtifactStore
    backend: VoiceConversionBackend

    def convert_batch_to_artifact(
        self,
        *,
        segments_uri: str,
        assignments_uri: str,
        output_segments_uri: str,
        output_prefix_uri: str,
        overlap_policy: str = "skip_and_flag",
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validate_artifact_uri(segments_uri)
        validate_artifact_uri(assignments_uri)
        validate_artifact_uri(output_segments_uri)
        validate_artifact_uri(output_prefix_uri.rstrip("/") + "/")
        supported_overlap_policies = {"skip_and_flag", "convert_and_flag"}
        if overlap_policy not in supported_overlap_policies:
            raise BatchConversionError(
                f"unsupported overlap_policy {overlap_policy!r}; supported policies: {sorted(supported_overlap_policies)}"
            )
        segments_payload = _load_json(self.artifact_store.local_path(segments_uri))
        assignment_payload = _load_json(self.artifact_store.local_path(assignments_uri))
        rows = segments_payload.get("segments")
        assignments = assignment_payload.get("assignments")
        if not isinstance(rows, list):
            raise AegisVoxContractError("segments.cleaned.json must contain segments list")
        if not isinstance(assignments, dict):
            raise AegisVoxContractError("speaker_target_map.json must contain assignments object")
        options = dict(options or {})
        num_workers = min(_positive_int(options.get("num_workers") or options.get("workers"), default=1), max(1, len(rows)))
        cuda_devices = _cuda_devices(options)

        def convert_one(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            conversion_service = VoiceConversionService(self.artifact_store, self.backend)
            segment_id = str(row.get("segment_id") or "")
            speaker_id = str(row.get("speaker_id") or "")
            if not segment_id or not speaker_id:
                raise AegisVoxContractError("segment row missing segment_id/speaker_id")
            source_uri = _segment_source_audio_uri(row)
            converted_uri = f"{output_prefix_uri.rstrip('/')}/{segment_id}.converted.wav"
            is_overlap = bool(row.get("overlap"))
            if is_overlap and overlap_policy == "skip_and_flag":
                skipped = {
                    "segment_id": segment_id,
                    "speaker_id": speaker_id,
                    "source_audio_uri": source_uri,
                    "converted_audio_uri": converted_uri,
                    "target_reference_audio_uri": None,
                    "duration_ratio": None,
                    "conversion_status": "skipped_overlap",
                    "failure_reason": "overlap_policy=skip_and_flag",
                    "backend": self.backend.backend_name,
                    "start_sec": row.get("start_sec"),
                    "end_sec": row.get("end_sec"),
                    "padded_start_sec": row.get("padded_start_sec"),
                    "padded_end_sec": row.get("padded_end_sec"),
                    "overlap": True,
                    "overlap_policy": overlap_policy,
                }
                return index, skipped
            assignment = assignments.get(speaker_id)
            if not assignment:
                raise BatchConversionError(f"missing pseudo-speaker assignment for {speaker_id}")
            target_uri = assignment.get("target_reference_audio_uri")
            if not target_uri:
                raise BatchConversionError(f"assignment for {speaker_id} missing target_reference_audio_uri")
            segment_options = dict(options)
            if cuda_devices:
                segment_options["cuda_visible_devices"] = cuda_devices[index % len(cuda_devices)]
            resume_existing = str(segment_options.get("resume_existing", "true")).strip().lower() not in {"0", "false", "no"}
            output_path = self.artifact_store.local_path(converted_uri)
            if resume_existing and output_path.exists() and output_path.stat().st_size > 0:
                converted = {
                    "segment_id": segment_id,
                    "speaker_id": speaker_id,
                    "source_audio_uri": source_uri,
                    "target_reference_audio_uri": target_uri,
                    "converted_audio_uri": converted_uri,
                    "duration_ratio": _duration_ratio(self.artifact_store.local_path(source_uri), output_path),
                    "conversion_status": "converted",
                    "failure_reason": None,
                    "backend": self.backend.backend_name,
                    "backend_metadata": {"resumed_from_existing_output": True},
                    "resumed_from_existing_output": True,
                }
                converted.update(
                    {
                        "start_sec": row.get("start_sec"),
                        "end_sec": row.get("end_sec"),
                        "padded_start_sec": row.get("padded_start_sec"),
                        "padded_end_sec": row.get("padded_end_sec"),
                        "overlap": is_overlap,
                        "overlap_policy": overlap_policy if is_overlap else None,
                        "overlap_converted": bool(is_overlap and overlap_policy == "convert_and_flag"),
                        "overlap_claim_boundary": (
                            "Converted overlapped diarization turn for engineering review; "
                            "not validated speaker-speaker separation."
                            if is_overlap and overlap_policy == "convert_and_flag"
                            else None
                        ),
                    }
                )
                if segment_options.get("cuda_visible_devices") not in {None, ""}:
                    converted["cuda_visible_devices"] = str(segment_options["cuda_visible_devices"])
                return index, converted
            try:
                converted = conversion_service.convert_segment_to_artifact(
                    segment_id=segment_id,
                    speaker_id=speaker_id,
                    source_audio_uri=source_uri,
                    target_reference_audio_uri=target_uri,
                    converted_audio_uri=converted_uri,
                    options=segment_options,
                )
                converted.update(
                    {
                        "start_sec": row.get("start_sec"),
                        "end_sec": row.get("end_sec"),
                        "padded_start_sec": row.get("padded_start_sec"),
                        "padded_end_sec": row.get("padded_end_sec"),
                        "overlap": is_overlap,
                        "overlap_policy": overlap_policy if is_overlap else None,
                        "overlap_converted": bool(is_overlap and overlap_policy == "convert_and_flag"),
                        "overlap_claim_boundary": (
                            "Converted overlapped diarization turn for engineering review; "
                            "not validated speaker-speaker separation."
                            if is_overlap and overlap_policy == "convert_and_flag"
                            else None
                        ),
                    }
                )
                if segment_options.get("cuda_visible_devices") not in {None, ""}:
                    converted["cuda_visible_devices"] = str(segment_options["cuda_visible_devices"])
                return index, converted
            except Exception as exc:
                failed = (
                    {
                        "segment_id": segment_id,
                        "speaker_id": speaker_id,
                        "source_audio_uri": source_uri,
                        "converted_audio_uri": converted_uri,
                        "target_reference_audio_uri": target_uri,
                        "duration_ratio": None,
                        "conversion_status": "failed",
                        "failure_reason": redact_text(str(exc), max_length=1000),
                        "backend": self.backend.backend_name,
                        "start_sec": row.get("start_sec"),
                        "end_sec": row.get("end_sec"),
                        "padded_start_sec": row.get("padded_start_sec"),
                        "padded_end_sec": row.get("padded_end_sec"),
                        "overlap": is_overlap,
                        "overlap_policy": overlap_policy if is_overlap else None,
                    }
                )
                if segment_options.get("cuda_visible_devices") not in {None, ""}:
                    failed["cuda_visible_devices"] = str(segment_options["cuda_visible_devices"])
                return index, failed

        converted_rows: list[dict[str, Any] | None] = [None] * len(rows)
        if num_workers <= 1:
            for index, row in enumerate(rows):
                result_index, converted = convert_one(index, row)
                converted_rows[result_index] = converted
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(convert_one, index, row) for index, row in enumerate(rows)]
                for future in as_completed(futures):
                    result_index, converted = future.result()
                    converted_rows[result_index] = converted
        final_rows = [row for row in converted_rows if row is not None]
        payload = segments_converted_payload(final_rows, backend=self.backend)
        payload["overlap_policy"] = overlap_policy
        payload["conversion_parallelism"] = {
            "num_workers": num_workers,
            "cuda_devices": cuda_devices,
            "strategy": "segment_threadpool_preserve_order" if num_workers > 1 else "serial",
            "resume_existing": True,
        }
        payload["num_converted"] = sum(1 for row in final_rows if row.get("conversion_status") == "converted")
        payload["num_resumed_existing"] = sum(1 for row in final_rows if row.get("resumed_from_existing_output"))
        payload["num_failed"] = sum(1 for row in final_rows if row.get("conversion_status") == "failed")
        payload["num_skipped_overlap"] = sum(1 for row in final_rows if row.get("conversion_status") == "skipped_overlap")
        payload["num_converted_overlap"] = sum(
            1 for row in final_rows if row.get("conversion_status") == "converted" and row.get("overlap_converted")
        )
        self.artifact_store.write_json(output_segments_uri, payload)
        return payload
