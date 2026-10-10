"""Experiment YAML contracts for AegisVox batch/e2e runs.

The YAML file is an operator-controlled experiment plan, not a secrets store.
It resolves to per-audio run requests plus execution options so long clinical
conversation batches can be reproduced without reconstructing shell flags.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .audit import sanitize_for_audit
from .schema import AegisVoxContractError, BackgroundPreservationMode, OverlapMode, PrivacyMode, RunRequest, validate_artifact_uri

SCHEMA_VERSION = "aegisvox_experiment.v1"
_SECRET_KEY_RE = re.compile(r"(secret|token|password|passwd|api[_-]?key|credential|authorization)", re.IGNORECASE)
_ALLOWED_BACKEND_ENV_KEYS = {
    "AEGISVOX_REQUIRE_PERSISTENT_MODEL_SERVICES",
    "AEGISVOX_SEGMENTATION_EXCLUSIVE_SPEAKER_TIMELINE",
    "AEGISVOX_DISABLE_OVERLAP_HANDLING",
    "AEGISVOX_SOURCE_SEPARATION_BACKEND",
    "AEGISVOX_SEPARATOR_SERVICE_URL",
    "BS_ROFORMER_SERVICE_URL",
    "AEGISVOX_DIARIZATION_SERVICE_URL",
    "PYANNOTE_SERVICE_URL",
    "AEGISVOX_SPEAKER_EMBEDDING_BACKEND",
    "AEGISVOX_SPEECHBRAIN_SERVICE_URL",
    "SPEECHBRAIN_SERVICE_URL",
    "AEGISVOX_VOICE_CONVERSION_BACKEND",
    "AEGISVOX_ALLOW_IDENTITY_VOICE_CONVERSION_BASELINE",
    "AEGISVOX_SEEDVC_SERVICE_URL",
    "SEEDVC_SERVICE_URL",
    "AEGISVOX_SPEECH_EDITING_BACKEND",
    "AEGISVOX_VOICECRAFT_SERVICE_URL",
    "VOICECRAFT_SERVICE_URL",
    "AEGISVOX_DIARIZATION_BACKEND",
    "AEGISVOX_DIARIZATION_MODEL_ID",
    "AEGISVOX_DIARIZATION_DEVICE",
    "AEGISVOX_DIARIZATION_NUM_SPEAKERS",
    "AEGISVOX_DIARIZATION_MIN_SPEAKERS",
    "AEGISVOX_DIARIZATION_MAX_SPEAKERS",
    "AEGISVOX_PYANNOTE_MODEL_ID",
    "AEGISVOX_PYANNOTE_DEVICE",
    "AEGISVOX_PYANNOTE_NUM_SPEAKERS",
    "AEGISVOX_PYANNOTE_MIN_SPEAKERS",
    "AEGISVOX_PYANNOTE_MAX_SPEAKERS",
    "AEGISVOX_NEMO_SORTFORMER_COMMAND",
    "AEGISVOX_NEMO_SORTFORMER_MODEL_PATH",
    "AEGISVOX_NEMO_SORTFORMER_THRESHOLD",
    "AEGISVOX_NEMO_SORTFORMER_FRAME_SEC",
    "AEGISVOX_NEMO_SORTFORMER_MIN_ACTIVITY_SEC",
    "AEGISVOX_DIARIZATION_COMMAND",
    "AEGISVOX_SPEECHBRAIN_DEVICE",
    "AEGISVOX_SPEAKER_PROFILE_MAX_EMBED_SECONDS",
    "AEGISVOX_ASSIGNMENT_EXTRACT_SOURCE_AUDIO_TRAITS",
    "AEGISVOX_ASSIGNMENT_SOURCE_AUDIO_TRAITS_CONTINUE_ON_ERROR",
    "AEGISVOX_ASSIGNMENT_MIN_SOURCE_TARGET_DISTANCE",
    "AEGISVOX_ASSIGNMENT_MIN_TARGET_TARGET_DISTANCE",
    "AEGISVOX_ASSIGNMENT_MIN_REFERENCE_DURATION_SEC",
    "AEGISVOX_ASSIGNMENT_MIN_REFERENCE_SAMPLE_RATE_HZ",
    "AEGISVOX_ASSIGNMENT_CANDIDATE_TOP_K",
    "AEGISVOX_ASSIGNMENT_VECTOR_WEIGHT",
    "AEGISVOX_ASSIGNMENT_TRAIT_WEIGHT",
    "AEGISVOX_ASSIGNMENT_PRESERVE_LANGUAGE",
    "AEGISVOX_ASSIGNMENT_PRESERVE_GENDER_PRESENTATION",
    "AEGISVOX_ASSIGNMENT_PRESERVE_ACCENT_GROUP",
    "AEGISVOX_ASSIGNMENT_PRESERVE_PITCH_BAND",
    "AEGISVOX_ASSIGNMENT_PRESERVE_SPEAKING_RATE",
    "AEGISVOX_ASSIGNMENT_PRESERVE_ENERGY",
    "AEGISVOX_ASSIGNMENT_PRESERVE_GEOGRAPHY_REGION",
    "AEGISVOX_ASSIGNMENT_PRESERVE_TONE",
    "AEGISVOX_ASSIGNMENT_PRESERVE_VOCAL_WEIGHT",
    "AEGISVOX_ASSIGNMENT_PRESERVE_RESONANCE",
    "AEGISVOX_ASSIGNMENT_PRESERVE_VOICE_MATURITY",
    "AEGISVOX_ASSIGNMENT_ACCENT_STYLE_CONSTRAINED",
    "AEGISVOX_ASSIGNMENT_REQUIRE_NATIVE_ENGLISH_TARGETS",
    "AEGISVOX_ASSIGNMENT_REQUIRE_NATIVE_ENGLISH_FOR_NORTH_AMERICA",
    "AEGISVOX_ASSIGNMENT_USE_HYBRID_AUDIO_TRAITS",
    "AEGISVOX_AUDIO_TRAIT_MODEL",
    "AEGISVOX_AUDIO_TRAIT_MAX_SECONDS",
    "AEGISVOX_TEXT_DEID_MODE",
    "AEGISVOX_TEXT_DEID_LLM_REPLACEMENTS",
    "AEGISVOX_TEXT_DEID_MODEL",
    "OPENAI_TEXT_DEID_MODEL",
    "AEGISVOX_OPENAI_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_ASR_MODEL",
    "OPENAI_ASR_RESPONSE_FORMAT",
    "OPENAI_ASR_TIMESTAMP_GRANULARITIES",
    "AEGISVOX_ASR_BACKEND",
    "AEGISVOX_ASR_MODEL",
    "AEGISVOX_ASR_DEVICE",
    "AEGISVOX_ASR_COMPUTE_TYPE",
    "AEGISVOX_ASR_VAD_FILTER",
    "AEGISVOX_FAST_WHISPER_MODEL",
    "AEGISVOX_FAST_WHISPER_DEVICE",
    "AEGISVOX_FAST_WHISPER_COMPUTE_TYPE",
    "AEGISVOX_FAST_WHISPER_BEAM_SIZE",
    "AEGISVOX_FAST_WHISPER_VAD_FILTER",
    "AEGISVOX_CANARY_MODEL",
    "AEGISVOX_CANARY_DEVICE",
    "SEEDVC_COMMAND",
    "AEGISVOX_VOICE_CONVERSION_WORKERS",
    "AEGISVOX_VOICE_CONVERSION_CUDA_DEVICES",
    "AEGISVOX_VOICE_CONVERSION_TIMEOUT_SEC",
    "VOICECRAFT_COMMAND",
    "VOICECRAFT_MODEL_DIR",
    "AEGISVOX_SPEECH_EDITING_TIMEOUT_SEC",
    "AEGISVOX_SPEECH_EDITING_WORKERS",
    "AEGISVOX_SPEECH_EDITING_CUDA_DEVICES",
    "AEGISVOX_ENABLE_SHORT_SPAN_CONTEXT_RENDERING",
    "AEGISVOX_SHORT_EDIT_SPAN_THRESHOLD_SEC",
    "AEGISVOX_SHORT_EDIT_WORD_THRESHOLD",
    "AEGISVOX_STITCHING_ALIGN_DURATION",
    "AEGISVOX_STITCHING_LOUDNESS_NORMALIZE",
    "AEGISVOX_STITCHING_FADE_MS",
    "AEGISVOX_STITCHING_FADE_CURVE",
    "AEGISVOX_STITCHING_OVERLAP_ADD",
    "AEGISVOX_STITCHING_BOUNDARY_ALIGNMENT",
    "AEGISVOX_STITCHING_BOUNDARY_ALIGNMENT_SEARCH_MS",
    "AEGISVOX_STITCHING_BOUNDARY_ALIGNMENT_MIN_OVERLAP_MS",
    "AEGISVOX_STITCHING_BOUNDARY_ALIGNMENT_MIN_SCORE",
    "AEGISVOX_STITCHING_SUPPRESS_INTERNAL_BOUNDARY_FADES",
    "AEGISVOX_STITCHING_INTERNAL_BOUNDARY_GAP_MS",
    "AEGISVOX_STITCHING_MAX_DURATION_DRIFT_SEC",
    "AEGISVOX_STITCHING_MAX_LOUDNESS_GAIN",
    "AEGISVOX_STITCHING_MIN_LOUDNESS_GAIN",
    "OVERLAP_SEPARATION_COMMAND",
    "AEGISVOX_AUDIO_SEPARATOR_ENDPOINT",
    "AEGISVOX_TSE_BACKEND",
    "AEGISVOX_TSE_SERVICE_URL",
    "TSE_SERVICE_URL",
    "AEGISVOX_OVERLAP_REMIX_ALLOW_MULTI_SPEAKER_SKIP",
    "AEGISVOX_OVERLAP_REMIX_DISABLE_MULTI_SPEAKER_ASSIGNMENT_GATE",
    "AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_GAIN",
    "AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_BACKGROUND_GAIN",
    "AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_TARGET_RMS_RATIO",
    "AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_MIN_RMS_GAIN",
    "AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_MAX_RMS_GAIN",
    "AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_MIN_ASSIGNMENT_MARGIN",
    "AEGISVOX_OVERLAP_REMIX_SINGLE_SPEAKER_TARGET_RMS_RATIO",
    "AEGISVOX_OVERLAP_REMIX_SINGLE_SPEAKER_MIN_RMS_GAIN",
    "AEGISVOX_OVERLAP_REMIX_SINGLE_SPEAKER_MAX_RMS_GAIN",
    "AEGISVOX_OVERLAP_REMIX_FADE_MS",
    "AEGISVOX_OVERLAP_REMIX_CONTEXT_EDGE_GAIN",
    "AEGISVOX_PRECONVERSION_AUDIT_ENABLED",
    "AEGISVOX_PRECONVERSION_MIN_EXPECTED_SPEAKERS",
    "AEGISVOX_PRECONVERSION_MIN_CLEANED_SPEAKER_SWITCHES",
    "AEGISVOX_PRECONVERSION_MIN_MINOR_SPEAKER_DURATION_SEC",
}


@dataclass(frozen=True)
class AudioInput:
    audio_id: str
    path: str | None = None
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, index: int) -> "AudioInput":
        if not isinstance(payload, Mapping):
            raise AegisVoxContractError("each audio entry must be an object")
        audio_id = str(payload.get("audio_id") or payload.get("id") or f"audio_{index:04d}")
        path = payload.get("path")
        uri = payload.get("uri") or payload.get("input_audio_uri")
        if not path and not uri:
            raise AegisVoxContractError(f"audio entry {audio_id!r} requires path or uri")
        if path and uri:
            raise AegisVoxContractError(f"audio entry {audio_id!r} must use either path or uri, not both")
        if uri:
            validate_artifact_uri(str(uri))
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise AegisVoxContractError(f"audio entry {audio_id!r} metadata must be an object")
        _reject_secret_keys(metadata, context=f"audio {audio_id} metadata")
        return cls(audio_id=audio_id, path=str(path) if path else None, uri=str(uri) if uri else None, metadata=dict(metadata))


@dataclass(frozen=True)
class RuntimeOptions:
    normalize_audio: bool = True
    sample_rate_hz: int = 16000
    channels: int = 1
    max_seconds: float | None = None
    concurrency: int = 1
    smoke_first_n: int = 2
    heartbeat_interval_seconds: int = 5
    job_timeout_seconds: int = 7200
    continue_on_run_failure: bool = True

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "RuntimeOptions":
        raw = dict(payload or {})
        unknown = set(raw) - {
            "normalize_audio",
            "sample_rate_hz",
            "channels",
            "max_seconds",
            "concurrency",
            "smoke_first_n",
            "heartbeat_interval_seconds",
            "job_timeout_seconds",
            "continue_on_run_failure",
        }
        if unknown:
            raise AegisVoxContractError(f"unknown runtime option(s): {sorted(unknown)}")
        options = cls(**raw)
        if options.max_seconds is not None and float(options.max_seconds) <= 0:
            raise AegisVoxContractError("runtime.max_seconds must be positive or null for full audio")
        if int(options.concurrency) < 1:
            raise AegisVoxContractError("runtime.concurrency must be >= 1")
        if int(options.smoke_first_n) < 0:
            raise AegisVoxContractError("runtime.smoke_first_n must be >= 0")
        if int(options.sample_rate_hz) <= 0:
            raise AegisVoxContractError("runtime.sample_rate_hz must be positive")
        if int(options.channels) not in {1, 2}:
            raise AegisVoxContractError("runtime.channels must be 1 or 2")
        return options

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalize_audio": self.normalize_audio,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "max_seconds": self.max_seconds,
            "concurrency": self.concurrency,
            "smoke_first_n": self.smoke_first_n,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "job_timeout_seconds": self.job_timeout_seconds,
            "continue_on_run_failure": self.continue_on_run_failure,
        }


@dataclass(frozen=True)
class ExperimentDefaults:
    privacy_mode: PrivacyMode = PrivacyMode.MEDIUM
    voice_pool_id: str = "openswara_clinical_v1"
    language_hint: str | None = "en"
    enable_lexical_deid: bool = True
    enable_speech_editing: bool = True
    overlap_mode: OverlapMode = OverlapMode.SKIP_AND_FLAG
    background_preservation_mode: BackgroundPreservationMode = BackgroundPreservationMode.SEPARATE_AND_REMIX
    services: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ExperimentDefaults":
        raw = dict(payload or {})
        unknown = set(raw) - {
            "privacy_mode",
            "voice_pool_id",
            "language_hint",
            "enable_lexical_deid",
            "enable_speech_editing",
            "overlap_mode",
            "background_preservation_mode",
            "services",
        }
        if unknown:
            raise AegisVoxContractError(f"unknown default option(s): {sorted(unknown)}")
        services = raw.pop("services", []) or []
        if not isinstance(services, list) or not all(isinstance(item, str) for item in services):
            raise AegisVoxContractError("defaults.services must be a list of service names")
        return cls(
            privacy_mode=PrivacyMode(raw.pop("privacy_mode", PrivacyMode.MEDIUM)),
            voice_pool_id=str(raw.pop("voice_pool_id", "openswara_clinical_v1")),
            language_hint=raw.pop("language_hint", "en"),
            enable_lexical_deid=bool(raw.pop("enable_lexical_deid", True)),
            enable_speech_editing=bool(raw.pop("enable_speech_editing", True)),
            overlap_mode=OverlapMode(raw.pop("overlap_mode", OverlapMode.SKIP_AND_FLAG)),
            background_preservation_mode=BackgroundPreservationMode(
                raw.pop("background_preservation_mode", BackgroundPreservationMode.SEPARATE_AND_REMIX)
            ),
            services=list(services),
        )

    def to_request(self, *, input_audio_uri: str, experiment_config: dict[str, Any]) -> RunRequest:
        return RunRequest(
            input_audio_uri=input_audio_uri,
            privacy_mode=self.privacy_mode,
            voice_pool_id=self.voice_pool_id,
            language_hint=self.language_hint,
            enable_lexical_deid=self.enable_lexical_deid,
            enable_speech_editing=self.enable_speech_editing,
            overlap_mode=self.overlap_mode,
            background_preservation_mode=self.background_preservation_mode,
            experiment_config=experiment_config,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "privacy_mode": self.privacy_mode.value,
            "voice_pool_id": self.voice_pool_id,
            "language_hint": self.language_hint,
            "enable_lexical_deid": self.enable_lexical_deid,
            "enable_speech_editing": self.enable_speech_editing,
            "overlap_mode": self.overlap_mode.value,
            "background_preservation_mode": self.background_preservation_mode.value,
            "services": list(self.services),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    schema_version: str = SCHEMA_VERSION
    description: str | None = None
    claim_boundary: str | None = None
    defaults: ExperimentDefaults = field(default_factory=ExperimentDefaults)
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)
    backend_env: dict[str, str] = field(default_factory=dict)
    audio_inputs: list[AudioInput] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(payload, Mapping):
            raise AegisVoxContractError("experiment config must be an object")
        _reject_secret_keys(payload, context="experiment config")
        schema_version = str(payload.get("schema_version") or "")
        if schema_version != SCHEMA_VERSION:
            raise AegisVoxContractError(f"schema_version must be {SCHEMA_VERSION!r}")
        experiment_id = str(payload.get("experiment_id") or "").strip()
        if not experiment_id:
            raise AegisVoxContractError("experiment_id is required")
        audio_payload = payload.get("audio_inputs") or payload.get("audios") or []
        if not isinstance(audio_payload, list) or not audio_payload:
            raise AegisVoxContractError("audio_inputs must be a non-empty list")
        backend_env = _validate_backend_env(payload.get("backend_env") or {})
        tags = payload.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise AegisVoxContractError("tags must be a list of strings")
        return cls(
            schema_version=schema_version,
            experiment_id=experiment_id,
            description=str(payload.get("description")) if payload.get("description") is not None else None,
            claim_boundary=str(payload.get("claim_boundary")) if payload.get("claim_boundary") is not None else None,
            defaults=ExperimentDefaults.from_payload(payload.get("defaults") or {}),
            runtime=RuntimeOptions.from_payload(payload.get("runtime") or {}),
            backend_env=backend_env,
            audio_inputs=[AudioInput.from_payload(row, index=idx) for idx, row in enumerate(audio_payload)],
            tags=list(tags),
        )

    def to_dict(self) -> dict[str, Any]:
        return sanitize_for_audit(
            {
                "schema_version": self.schema_version,
                "experiment_id": self.experiment_id,
                "description": self.description,
                "claim_boundary": self.claim_boundary,
                "defaults": self.defaults.to_dict(),
                "runtime": self.runtime.to_dict(),
                "backend_env": dict(self.backend_env),
                "audio_inputs": [row.__dict__ for row in self.audio_inputs],
                "tags": list(self.tags),
            }
        )

    def per_run_experiment_config(self, audio: AudioInput) -> dict[str, Any]:
        return sanitize_for_audit(
            {
                "schema_version": self.schema_version,
                "experiment_id": self.experiment_id,
                "audio_id": audio.audio_id,
                "description": self.description,
                "claim_boundary": self.claim_boundary,
                "runtime": self.runtime.to_dict(),
                "defaults": self.defaults.to_dict(),
                "backend_env_keys": sorted(self.backend_env),
                "tags": list(self.tags),
                "audio_metadata": dict(audio.metadata),
            }
        )

    def resolved_run_requests(self, *, bucket: str = "aegisvox") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, audio in enumerate(self.audio_inputs):
            run_id = _safe_run_id(f"{self.experiment_id}_{idx:04d}_{audio.audio_id}")
            input_uri = audio.uri or f"s3://{bucket}/runs/{run_id}/input.wav"
            defaults = self.defaults
            language_hint = audio.metadata.get("language_hint")
            if language_hint:
                defaults = ExperimentDefaults(
                    privacy_mode=self.defaults.privacy_mode,
                    voice_pool_id=self.defaults.voice_pool_id,
                    language_hint=str(language_hint),
                    enable_lexical_deid=self.defaults.enable_lexical_deid,
                    enable_speech_editing=self.defaults.enable_speech_editing,
                    overlap_mode=self.defaults.overlap_mode,
                    background_preservation_mode=self.defaults.background_preservation_mode,
                    services=list(self.defaults.services),
                )
            request = defaults.to_request(input_audio_uri=input_uri, experiment_config=self.per_run_experiment_config(audio))
            rows.append(
                {
                    "run_id": run_id,
                    "audio_id": audio.audio_id,
                    "source_path": audio.path,
                    "input_audio_uri": input_uri,
                    "request": request.to_dict(),
                    "runtime": self.runtime.to_dict(),
                    "backend_env": dict(self.backend_env),
                }
            )
        return sanitize_for_audit(rows)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser()
    try:
        import yaml  # type: ignore
    except Exception as exc:  # noqa: BLE001 - setup error should be actionable
        raise AegisVoxContractError("PyYAML is required to load experiment YAML files") from exc
    payload = yaml.safe_load(path.read_text())
    return ExperimentConfig.from_payload(payload)


def write_resolved_run_manifest(config: ExperimentConfig, path: str | Path, *, bucket: str = "aegisvox") -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = config.resolved_run_requests(bucket=bucket)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def _validate_backend_env(payload: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise AegisVoxContractError("backend_env must be an object")
    env: dict[str, str] = {}
    for key, value in payload.items():
        key = str(key)
        if _SECRET_KEY_RE.search(key):
            raise AegisVoxContractError(f"backend_env must not contain secret-like key {key!r}")
        if key not in _ALLOWED_BACKEND_ENV_KEYS:
            raise AegisVoxContractError(f"backend_env key {key!r} is not in the allow-list")
        if value is None or isinstance(value, (dict, list)):
            raise AegisVoxContractError(f"backend_env value for {key!r} must be scalar")
        env[key] = str(value)
    return env


def _reject_secret_keys(payload: Mapping[str, Any], *, context: str) -> None:
    for key, value in payload.items():
        key_str = str(key)
        if _SECRET_KEY_RE.search(key_str):
            raise AegisVoxContractError(f"{context} contains secret-like key {key_str!r}; keep secrets in env files only")
        if isinstance(value, Mapping):
            _reject_secret_keys(value, context=f"{context}.{key_str}")
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, Mapping):
                    _reject_secret_keys(item, context=f"{context}.{key_str}[{idx}]")


def _safe_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:180]
