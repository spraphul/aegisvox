"""Run provenance manifest generation for ad-hoc AegisVox smokes.

The production orchestrator records service versions as completion events
arrive. Public runtime smokes often run service CLIs directly, so this module
creates a compact manifest from their existing artifacts. It is intentionally
artifact-driven and does not infer success beyond what those artifacts record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import __version__
from .asr_alignment import create_asr_backend
from .diarization import DEFAULT_PYANNOTE_MODEL_ID
from .schema import validate_artifact_uri
from .service_job import (
    ASR_ALIGNMENT_SERVICE_VERSION,
    ASSIGNMENT_SERVICE_VERSION,
    DIARIZATION_SERVICE_VERSION,
    SEGMENTATION_SERVICE_VERSION,
    SPEAKER_PROFILING_SERVICE_VERSION,
    SPEECH_EDITING_SERVICE_VERSION,
    STITCHING_SERVICE_VERSION,
    TEXT_DEID_SERVICE_VERSION,
    VOICE_CONVERSION_SERVICE_VERSION,
)
from .speaker_profile import SpeechBrainECAPAEmbeddingBackend
from .storage import LocalArtifactStore
from .voice_conversion import SeedVCCommandBackend


def _load_optional_json(store: LocalArtifactStore, uri: str | None) -> dict[str, Any] | None:
    if not uri:
        return None
    validate_artifact_uri(uri)
    if not store.exists(uri):
        return None
    payload = store.read_json(uri)
    return payload if isinstance(payload, dict) else None


def _non_empty(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


@dataclass(frozen=True)
class SmokeProvenanceInputs:
    run_id: str
    diarization_uri: str | None = None
    segments_uri: str | None = None
    speaker_profiles_uri: str | None = None
    assignments_uri: str | None = None
    converted_segments_uri: str | None = None
    stitch_report_uri: str | None = None
    asr_uri: str | None = None
    deid_text_uri: str | None = None
    voicecraft_edit_jobs_uri: str | None = None
    speaker_privacy_metrics_uri: str | None = None
    evaluation_report_uri: str | None = None
    pass_fail_uri: str | None = None
    voice_pool_id: str = "openswara_smoke"
    voice_pool_version: str = "v1"
    openai_asr_model: str | None = None
    claim_boundary: str = "public/non-sensitive runtime integration provenance; not clinical evidence"
    extra_artifacts: dict[str, str] = field(default_factory=dict)

    def artifact_uris(self) -> dict[str, str]:
        rows = {
            "diarization.raw.json": self.diarization_uri,
            "segments.cleaned.json": self.segments_uri,
            "speaker_profiles.json": self.speaker_profiles_uri,
            "speaker_target_map.json": self.assignments_uri,
            "segments.converted.json": self.converted_segments_uri,
            "stitch_report.json": self.stitch_report_uri,
            "asr.changed_identity.json": self.asr_uri,
            "deid_text.json": self.deid_text_uri,
            "voicecraft_edit_jobs.json": self.voicecraft_edit_jobs_uri,
            "speaker_privacy_metrics.json": self.speaker_privacy_metrics_uri,
            "evaluation_report.json": self.evaluation_report_uri,
            "pass_fail.json": self.pass_fail_uri,
        }
        rows.update(self.extra_artifacts)
        return {key: str(value) for key, value in rows.items() if value}


def build_smoke_provenance_manifest(store: LocalArtifactStore, inputs: SmokeProvenanceInputs) -> dict[str, Any]:
    """Build a compact run manifest with evaluator-required service versions."""

    artifacts = inputs.artifact_uris()
    for uri in artifacts.values():
        validate_artifact_uri(uri)
    diarization = _load_optional_json(store, inputs.diarization_uri)
    profiles = _load_optional_json(store, inputs.speaker_profiles_uri)
    assignments = _load_optional_json(store, inputs.assignments_uri)
    converted = _load_optional_json(store, inputs.converted_segments_uri)
    asr = _load_optional_json(store, inputs.asr_uri)
    deid = _load_optional_json(store, inputs.deid_text_uri)
    edit_jobs = _load_optional_json(store, inputs.voicecraft_edit_jobs_uri)
    privacy = _load_optional_json(store, inputs.speaker_privacy_metrics_uri)

    speechbrain_default = SpeechBrainECAPAEmbeddingBackend().backend_version
    seedvc_default = SeedVCCommandBackend.backend_version
    asr_backend_default = create_asr_backend()
    asr_model_default = getattr(asr_backend_default, "model", None) or getattr(asr_backend_default, "model_size", None) or getattr(
        asr_backend_default, "model_id", "unknown"
    )
    speech_edit_backend = "no_edits_passthrough"
    if edit_jobs:
        backend = edit_jobs.get("backend")
        backend_version = edit_jobs.get("backend_version")
        if backend or backend_version:
            speech_edit_backend = ":".join(str(x) for x in (backend, backend_version) if x)

    service_versions = {
        "orchestrator": __version__,
        "diarization_service": DIARIZATION_SERVICE_VERSION,
        "pyannote": _non_empty(
            (diarization or {}).get("model_id") or (diarization or {}).get("backend_version"),
            DEFAULT_PYANNOTE_MODEL_ID,
        ),
        "segmentation": SEGMENTATION_SERVICE_VERSION,
        "speaker_profiling": SPEAKER_PROFILING_SERVICE_VERSION,
        "speechbrain-ecapa": _non_empty(
            (privacy or {}).get("backend_version")
            or (profiles or {}).get("backend_version")
            or (profiles or {}).get("embedding_backend"),
            speechbrain_default,
        ),
        "assignment": ASSIGNMENT_SERVICE_VERSION,
        "voice_pool": f"{inputs.voice_pool_id}:{inputs.voice_pool_version}",
        "voice_conversion": VOICE_CONVERSION_SERVICE_VERSION,
        "seed-vc": _non_empty((converted or {}).get("backend_version") or (converted or {}).get("backend"), seedvc_default),
        "stitching": STITCHING_SERVICE_VERSION,
        "asr_alignment": ASR_ALIGNMENT_SERVICE_VERSION,
        "openai-asr": _non_empty(
            inputs.openai_asr_model or (asr or {}).get("model") or (asr or {}).get("backend_version"),
            asr_model_default,
        ),
        "text_deid": _non_empty((deid or {}).get("service_version") or (deid or {}).get("policy_mode"), TEXT_DEID_SERVICE_VERSION),
        "speech_editing": SPEECH_EDITING_SERVICE_VERSION,
        "speech_editing_backend": speech_edit_backend,
    }
    artifact_status = {
        name: {"uri": uri, "exists": store.exists(uri)}
        for name, uri in artifacts.items()
    }
    return {
        "run_id": inputs.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_kind": "aegisvox_smoke_provenance",
        "service_versions": service_versions,
        "artifacts": artifacts,
        "artifact_status": artifact_status,
        "claim_boundary": inputs.claim_boundary,
        "source": "artifact-driven provenance manifest for direct CLI/runtime smokes",
        "assignment_summary": {
            "privacy_mode": (assignments or {}).get("privacy_mode"),
            "num_assignments": len((assignments or {}).get("assignments") or {}),
        },
    }
