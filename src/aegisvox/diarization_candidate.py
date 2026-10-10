"""Side-by-side diarization candidate reruns.

This module builds an engineering-review candidate from an existing diarization
artifact without mutating the canonical run. It deliberately reuses the real
AegisVox service boundaries (segmentation, speaker profiling, assignment,
voice conversion, stitching) so a candidate can be listened to next to the
canonical output before promoting a diarization backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .assignment import AssignmentService
from .batch_conversion import BatchVoiceConversionService
from .schema import AegisVoxContractError, validate_artifact_uri
from .segmentation import wav_duration_sec
from .services.segmentation import SegmentationService
from .speaker_profile import SpeakerEmbeddingBackend, SpeakerProfilingService
from .speaker_profile_traits import SpeakerProfileTraitEnrichmentPolicy, enrich_speaker_profiles_with_audio_traits
from .stitching import StitchingService
from .storage import LocalArtifactStore
from .voice_conversion import VoiceConversionBackend
from .voice_pool import VoicePoolIndex


class DiarizationCandidateError(RuntimeError):
    """Raised when a candidate rerun cannot be built."""


@dataclass(frozen=True)
class DiarizationCandidateConfig:
    source_run_id: str
    candidate_id: str
    diarization_uri: str | None = None
    diarization_path: Path | None = None
    source_audio_uri: str | None = None
    artifact_bucket: str = "aegisvox"
    output_prefix_uri: str | None = None
    conversion_options: dict[str, Any] = field(default_factory=dict)
    overlap_policy: str = "skip_and_flag"
    extract_source_audio_traits: bool = False
    audio_trait_model: str = "gpt-audio-1.5"

    def __post_init__(self) -> None:
        if not self.source_run_id:
            raise DiarizationCandidateError("source_run_id is required")
        if not self.candidate_id:
            raise DiarizationCandidateError("candidate_id is required")
        if bool(self.diarization_uri) == bool(self.diarization_path):
            raise DiarizationCandidateError("provide exactly one of diarization_uri or diarization_path")
        if self.diarization_uri is not None:
            validate_artifact_uri(self.diarization_uri)
        if self.source_audio_uri is not None:
            validate_artifact_uri(self.source_audio_uri)
        if self.output_prefix_uri is not None:
            validate_artifact_uri(self.output_prefix_uri.rstrip("/") + "/")
        if self.overlap_policy not in {"skip_and_flag", "convert_and_flag"}:
            raise DiarizationCandidateError(
                "overlap_policy must be one of: skip_and_flag, convert_and_flag"
            )

    @property
    def prefix_uri(self) -> str:
        if self.output_prefix_uri:
            return self.output_prefix_uri.rstrip("/")
        return f"s3://{self.artifact_bucket}/runs/{self.source_run_id}/candidates/{self.candidate_id}"


def _load_json_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DiarizationCandidateError(f"diarization_path not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise DiarizationCandidateError("diarization artifact must be a JSON object")
    return payload


def _copy_or_register_diarization(
    store: LocalArtifactStore,
    config: DiarizationCandidateConfig,
    *,
    candidate_diarization_uri: str,
) -> tuple[dict[str, Any], str | None]:
    if config.diarization_uri is not None:
        payload = store.read_json(config.diarization_uri)
        if not isinstance(payload, dict):
            raise DiarizationCandidateError("diarization artifact must be a JSON object")
        source_ref = config.diarization_uri
    else:
        assert config.diarization_path is not None
        payload = _load_json_path(config.diarization_path)
        source_ref = str(config.diarization_path)
    payload = dict(payload)
    payload.setdefault("candidate_source", {})
    if isinstance(payload["candidate_source"], dict):
        payload["candidate_source"].update(
            {
                "source_run_id": config.source_run_id,
                "candidate_id": config.candidate_id,
                "input_diarization_ref": source_ref,
            }
        )
    store.write_json(candidate_diarization_uri, payload)
    return payload, source_ref


def _source_audio_uri_from_payload(payload: dict[str, Any], override: str | None) -> str:
    source_audio_uri = override or payload.get("source_audio_uri")
    if not source_audio_uri:
        raise DiarizationCandidateError(
            "source audio URI is missing; provide --source-audio-uri or use a diarization artifact with source_audio_uri"
        )
    validate_artifact_uri(str(source_audio_uri))
    return str(source_audio_uri)


def _count_by_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("conversion_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def build_diarization_candidate(
    *,
    artifact_store: LocalArtifactStore,
    config: DiarizationCandidateConfig,
    embedding_backend: SpeakerEmbeddingBackend,
    voice_pool_index: VoicePoolIndex,
    conversion_backend: VoiceConversionBackend,
) -> dict[str, Any]:
    """Build a candidate changed-identity audio from a selected diarization.

    The canonical run path is never written. All outputs are under
    `.../runs/<source_run_id>/candidates/<candidate_id>/` unless an explicit
    `output_prefix_uri` is supplied.
    """

    prefix = config.prefix_uri
    candidate_diarization_uri = f"{prefix}/diarization.raw.json"
    segments_uri = f"{prefix}/segments.cleaned.json"
    profiles_uri = f"{prefix}/speaker_profiles.json"
    source_trait_profiles_uri = f"{prefix}/speaker_profiles.audio_traits.json"
    source_trait_report_uri = f"{prefix}/speaker_profiles.audio_traits_report.json"
    assignments_uri = f"{prefix}/speaker_target_map.json"
    assignment_report_uri = f"{prefix}/assignment_report.md"
    converted_segments_uri = f"{prefix}/segments.converted.json"
    changed_identity_uri = f"{prefix}/changed_identity.wav"
    stitch_report_uri = f"{prefix}/stitch_report.json"
    summary_uri = f"{prefix}/candidate_summary.json"

    diarization_payload, input_diarization_ref = _copy_or_register_diarization(
        artifact_store,
        config,
        candidate_diarization_uri=candidate_diarization_uri,
    )
    source_audio_uri = _source_audio_uri_from_payload(diarization_payload, config.source_audio_uri)
    source_audio_path = artifact_store.local_path(source_audio_uri)
    if not source_audio_path.exists():
        raise DiarizationCandidateError(f"source audio not found for candidate rerun: {source_audio_path}")
    audio_duration = wav_duration_sec(source_audio_path)

    segmentation_payload = SegmentationService(artifact_store).clean_to_artifact(
        candidate_diarization_uri,
        segments_uri,
        source_audio_uri=source_audio_uri,
        audio_duration_sec=audio_duration,
        output_segment_prefix_uri=f"{prefix}/segments",
    )
    profile_payload = SpeakerProfilingService(
        artifact_store,
        embedding_backend=embedding_backend,
    ).profile_to_artifact(segments_uri, profiles_uri, output_prefix_uri=prefix)
    assignment_profiles_uri = profiles_uri
    source_trait_report: dict[str, Any] | None = None
    if config.extract_source_audio_traits:
        source_trait_report = enrich_speaker_profiles_with_audio_traits(
            artifact_store=artifact_store,
            speaker_profiles_uri=profiles_uri,
            output_profiles_uri=source_trait_profiles_uri,
            report_uri=source_trait_report_uri,
            policy=SpeakerProfileTraitEnrichmentPolicy(model=config.audio_trait_model),
        )
        assignment_profiles_uri = source_trait_profiles_uri
    assignment_payload = AssignmentService(
        artifact_store,
        voice_pool_index=voice_pool_index,
    ).assign_to_artifact(assignment_profiles_uri, assignments_uri, output_report_uri=assignment_report_uri)
    conversion_payload = BatchVoiceConversionService(
        artifact_store,
        backend=conversion_backend,
    ).convert_batch_to_artifact(
        segments_uri=segments_uri,
        assignments_uri=assignments_uri,
        output_segments_uri=converted_segments_uri,
        output_prefix_uri=f"{prefix}/converted_segments",
        options=config.conversion_options,
        overlap_policy=config.overlap_policy,
    )
    stitch_payload = StitchingService(artifact_store).stitch_to_artifact(
        converted_segments_uri=converted_segments_uri,
        output_audio_uri=changed_identity_uri,
        stitch_report_uri=stitch_report_uri,
        audio_duration_sec=audio_duration,
    )

    segments = segmentation_payload.get("segments") if isinstance(segmentation_payload.get("segments"), list) else []
    converted_rows = conversion_payload.get("segments") if isinstance(conversion_payload.get("segments"), list) else []
    summary = {
        "status": "completed",
        "source_run_id": config.source_run_id,
        "candidate_id": config.candidate_id,
        "candidate_prefix_uri": prefix,
        "input_diarization_ref": input_diarization_ref,
        "source_audio_uri": source_audio_uri,
        "audio_duration_sec": round(audio_duration, 4),
        "backend": diarization_payload.get("backend") or diarization_payload.get("model_id") or "unknown",
        "model_id": diarization_payload.get("model_id"),
        "num_diarization_segments": diarization_payload.get("num_segments") or len(diarization_payload.get("segments") or []),
        "num_clean_segments": segmentation_payload.get("num_segments"),
        "num_clean_speakers": len({str(row.get("speaker_id")) for row in segments if isinstance(row, dict) and row.get("speaker_id")}),
        "clean_speaker_ids": sorted({str(row.get("speaker_id")) for row in segments if isinstance(row, dict) and row.get("speaker_id")}),
        "num_profiled_speakers": profile_payload.get("num_speakers"),
        "profile_reliability_by_speaker": {
            speaker_id: (profile.get("metadata") or {}).get("profile_reliability")
            for speaker_id, profile in sorted((profile_payload.get("speakers") or {}).items())
            if isinstance(profile, dict)
        },
        "num_assignments": len(assignment_payload.get("assignments") or {}),
        "source_audio_trait_enrichment": (
            {
                "enabled": True,
                "model": config.audio_trait_model,
                "num_enriched": source_trait_report.get("num_enriched") if source_trait_report else None,
                "num_errors": source_trait_report.get("num_errors") if source_trait_report else None,
            }
            if config.extract_source_audio_traits
            else {"enabled": False}
        ),
        "overlap_policy": config.overlap_policy,
        "conversion_status_counts": _count_by_status(converted_rows),
        "num_converted_overlap": conversion_payload.get("num_converted_overlap"),
        "num_skipped_overlap": conversion_payload.get("num_skipped_overlap"),
        "num_stitched": stitch_payload.get("num_stitched"),
        "num_not_stitched": stitch_payload.get("num_not_stitched"),
        "artifacts": {
            "diarization_uri": candidate_diarization_uri,
            "segments_uri": segments_uri,
            "speaker_profiles_uri": profiles_uri,
            "speaker_profiles_audio_traits_uri": source_trait_profiles_uri if config.extract_source_audio_traits else None,
            "speaker_profiles_audio_traits_report_uri": source_trait_report_uri if config.extract_source_audio_traits else None,
            "assignments_uri": assignments_uri,
            "assignment_report_uri": assignment_report_uri,
            "converted_segments_uri": converted_segments_uri,
            "changed_identity_audio_uri": changed_identity_uri,
            "stitch_report_uri": stitch_report_uri,
            "candidate_summary_uri": summary_uri,
        },
        "claim_boundary": (
            "Engineering review candidate only. It does not replace the canonical run; "
            "listen side-by-side and inspect diarization warnings before promotion."
        ),
    }
    artifact_store.write_json(summary_uri, summary)
    return summary
