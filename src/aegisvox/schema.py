"""Shared AegisVox data contracts.

These contracts intentionally avoid model inference. Model-backed services should
fail loudly when their real backend is unavailable instead of writing fake
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


class AegisVoxContractError(ValueError):
    """Raised when a service contract or artifact manifest is invalid."""


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class PrivacyMode(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OverlapMode(str, Enum):
    SKIP_AND_FLAG = "skip_and_flag"
    SEPARATE_CONVERT_REMIX = "separate_convert_remix"


class BackgroundPreservationMode(str, Enum):
    DISABLED = "disabled"
    SEPARATE_AND_REMIX = "separate_and_remix"


EVENT_SEQUENCE = [
    "run.created",
    "source_separation.requested",
    "source_separation.completed",
    "background_leakage_check.requested",
    "background_leakage_check.completed",
    "diarization.requested",
    "diarization.completed",
    "segmentation.requested",
    "segmentation.completed",
    "speaker_profile.requested",
    "speaker_profile.completed",
    "pseudo_assignment.requested",
    "pseudo_assignment.completed",
    "voice_conversion.requested",
    "voice_conversion.completed",
    "stitching.requested",
    "stitching.completed",
    "asr.requested",
    "asr.completed",
    "text_deid.requested",
    "text_deid.completed",
    "speech_edit.requested",
    "speech_edit.completed",
    "final_background_remix.requested",
    "final_background_remix.completed",
    "speaker_privacy_metrics.requested",
    "speaker_privacy_metrics.completed",
    "overlap_remix.requested",
    "overlap_remix.completed",
    "evaluation.requested",
    "evaluation.completed",
    "run.completed",
    "run.canceled",
    "run.failed",
]

REQUIRED_ARTIFACT_NAMES = [
    "diarization.raw.json",
    "source_separation_report.json",
    "foreground_vocals.wav",
    "preserved_background.wav",
    "background_leakage_report.json",
    "background_leakage_asr.json",
    "segments.cleaned.json",
    "speaker_profiles.json",
    "speaker_target_map.json",
    "segments.converted.json",
    "changed_identity.wav",
    "asr.changed_identity.json",
    "deid_text.json",
    "speaker_privacy_metrics.json",
    "voicecraft_edit_jobs.json",
    "final_sanitized.wav",
    "final_sanitized_with_background.wav",
    "final_background_remix_report.json",
    "overlap_remix_report.json",
    "overlap_remix_apply_report.json",
    "overlap_remixed_identity.wav",
    "diarization_backend_matrix.json",
    "diarization_comparison_report.json",
    "evaluation_report.json",
    "evaluation_report.md",
    "output_policy_report.json",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"


def validate_artifact_uri(uri: str) -> str:
    if not isinstance(uri, str) or not uri.strip():
        raise AegisVoxContractError("artifact URI must be a non-empty string")
    if uri.startswith(("s3://", "file://")):
        return uri
    raise AegisVoxContractError(f"unsupported artifact URI scheme for {uri!r}; expected s3:// or file://")


@dataclass(frozen=True)
class RunRequest:
    input_audio_uri: str
    privacy_mode: PrivacyMode = PrivacyMode.MEDIUM
    voice_pool_id: str = "openswara_clinical_v1"
    language_hint: str | None = "en"
    enable_lexical_deid: bool = True
    enable_speech_editing: bool = True
    overlap_mode: OverlapMode = OverlapMode.SKIP_AND_FLAG
    background_preservation_mode: BackgroundPreservationMode = BackgroundPreservationMode.DISABLED
    experiment_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_artifact_uri(self.input_audio_uri)
        if isinstance(self.privacy_mode, str):
            object.__setattr__(self, "privacy_mode", PrivacyMode(self.privacy_mode))
        if isinstance(self.overlap_mode, str):
            object.__setattr__(self, "overlap_mode", OverlapMode(self.overlap_mode))
        if isinstance(self.background_preservation_mode, str):
            object.__setattr__(self, "background_preservation_mode", BackgroundPreservationMode(self.background_preservation_mode))
        if not self.voice_pool_id:
            raise AegisVoxContractError("voice_pool_id is required")
        if self.experiment_config is None:
            object.__setattr__(self, "experiment_config", {})
        if not isinstance(self.experiment_config, Mapping):
            raise AegisVoxContractError("experiment_config must be an object when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_audio_uri": self.input_audio_uri,
            "privacy_mode": self.privacy_mode.value,
            "voice_pool_id": self.voice_pool_id,
            "language_hint": self.language_hint,
            "enable_lexical_deid": self.enable_lexical_deid,
            "enable_speech_editing": self.enable_speech_editing,
            "overlap_mode": self.overlap_mode.value,
            "background_preservation_mode": self.background_preservation_mode.value,
            "experiment_config": dict(self.experiment_config),
        }


@dataclass
class ArtifactRecord:
    name: str
    uri: str
    artifact_type: str
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise AegisVoxContractError("artifact name is required")
        if not self.artifact_type:
            raise AegisVoxContractError("artifact_type is required")
        validate_artifact_uri(self.uri)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "artifact_type": self.artifact_type,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class EventEnvelope:
    event_type: str
    run_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"evt_{uuid4().hex}")
    emitted_at: str = field(default_factory=utc_now_iso)
    service: str = "unknown"

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_SEQUENCE:
            raise AegisVoxContractError(f"unsupported event_type {self.event_type!r}")
        if not self.run_id:
            raise AegisVoxContractError("run_id is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "emitted_at": self.emitted_at,
            "service": self.service,
            "payload": dict(self.payload),
        }


@dataclass
class RunManifest:
    run_id: str
    request: RunRequest
    status: RunStatus = RunStatus.CREATED
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)
    events: list[EventEnvelope] = field(default_factory=list)
    service_versions: dict[str, str] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise AegisVoxContractError("run_id is required")
        if isinstance(self.status, str):
            self.status = RunStatus(self.status)

    def add_event(self, event: EventEnvelope) -> None:
        if event.run_id != self.run_id:
            raise AegisVoxContractError("event run_id does not match manifest run_id")
        self.events.append(event)
        self.updated_at = utc_now_iso()

    def add_artifact(self, artifact: ArtifactRecord) -> None:
        self.artifacts[artifact.name] = artifact
        self.updated_at = utc_now_iso()

    def set_status(self, status: RunStatus | str, failure_reason: str | None = None) -> None:
        self.status = RunStatus(status)
        self.failure_reason = failure_reason
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request": self.request.to_dict(),
            "artifacts": {name: rec.to_dict() for name, rec in self.artifacts.items()},
            "events": [event.to_dict() for event in self.events],
            "service_versions": dict(self.service_versions),
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        request = RunRequest(**dict(payload["request"]))
        manifest = cls(
            run_id=str(payload["run_id"]),
            request=request,
            status=RunStatus(payload.get("status", RunStatus.CREATED)),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            service_versions=dict(payload.get("service_versions", {})),
            failure_reason=payload.get("failure_reason"),
        )
        for name, rec in dict(payload.get("artifacts", {})).items():
            manifest.artifacts[name] = ArtifactRecord(**rec)
        for event in payload.get("events", []):
            manifest.events.append(EventEnvelope(**event))
        return manifest
