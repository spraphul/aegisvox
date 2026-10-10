"""Command-job adapters for AegisVox service workers.

This module is the bridge between the queue/control plane and the service
implementations. It reads a claimed job payload, runs exactly one real service,
writes the canonical artifacts, and returns the JSON object expected by
``CommandWorker``. It does not fabricate model outputs: model-backed services
use their fail-loud default backends and will error when credentials/checkpoints
are missing.
"""

from __future__ import annotations

import json
import os
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .asr_alignment import ASRAlignmentService, create_asr_backend
from .assignment import AssignmentPolicy, AssignmentService
from .background_leakage import BackgroundLeakageService
from .background_preservation import FinalBackgroundRemixService, SourceSeparationService
from .batch_conversion import BatchVoiceConversionService
from .diarization import CommandDiarizationConfig, CommandDiarizationService, PyannoteDiarizationConfig, PyannoteDiarizationService, PyannoteHTTPDiarizationService
from .evaluation import EvaluationInputs, EvaluationPolicy, EvaluationService
from .manifest import artifact_uri
from .overlap_handling import OverlapHandlingService
from .output_policy import OutputPolicyInputs, build_output_policy
from .privacy_metrics import SpeakerPrivacyMetricsService
from .schema import AegisVoxContractError, OverlapMode, PrivacyMode, validate_artifact_uri
from .segmentation import wav_duration_sec
from .services.segmentation import SegmentationService
from .speaker_profile import SpeakerProfilingService, SpeechBrainECAPAEmbeddingBackend, SpeechBrainHTTPEmbeddingBackend
from .speaker_profile_traits import SpeakerProfileTraitEnrichmentPolicy, enrich_speaker_profiles_with_audio_traits
from .speech_editing import FasterWhisperEditedSpanValidator, SpeechEditingService, VoiceCraftCommandBackend, VoiceCraftHTTPBackend
from .stitching import StitchingPolicy, StitchingService
from .storage import LocalArtifactStore
from .text_deid import DeIDPolicy, OpenAILLMTextDeIDBackend, TextDeIDService
from .model_residency import ModelResidency, ProcessMode, assert_persistent_allowed
from .voice_conversion import IdentityVoiceConversionBackend, SeedVCCommandBackend, SeedVCHTTPBackend, VoiceConversionBackend
from .voice_pool import VoicePoolIndex


SERVICE_COMPLETION_EVENTS: dict[str, str] = {
    "source_separation": "source_separation.completed",
    "background_leakage_check": "background_leakage_check.completed",
    "diarization": "diarization.completed",
    "segmentation": "segmentation.completed",
    "speaker_profiling": "speaker_profile.completed",
    "assignment": "pseudo_assignment.completed",
    "voice_conversion": "voice_conversion.completed",
    "stitching": "stitching.completed",
    "asr_alignment": "asr.completed",
    "text_deid": "text_deid.completed",
    "speech_editing": "speech_edit.completed",
    "final_background_remix": "final_background_remix.completed",
    "speaker_privacy_metrics": "speaker_privacy_metrics.completed",
    "overlap_remix": "overlap_remix.completed",
    "evaluation": "evaluation.completed",
}


@dataclass(frozen=True)
class ServiceJobResult:
    completion_event_type: str
    artifacts: list[dict[str, Any]]
    metrics: dict[str, Any]
    service_versions: dict[str, str] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)

    def to_worker_payload(self) -> dict[str, Any]:
        completion_payload = {
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }
        if self.service_versions:
            completion_payload["service_versions"] = dict(self.service_versions)
        if self.model_versions:
            completion_payload["model_versions"] = dict(self.model_versions)
        return {
            "completion_event_type": self.completion_event_type,
            "completion_payload": completion_payload,
        }


SERVICE_JOB_ADAPTER_VERSION = "service-job-adapter-v1"
DIARIZATION_SERVICE_VERSION = "diarization-wrapper-v2"
SOURCE_SEPARATION_SERVICE_VERSION = "source-separation-bs-roformer-v1"
BACKGROUND_LEAKAGE_SERVICE_VERSION = "background-leakage-check-v2-llm-hipaa"
SEGMENTATION_SERVICE_VERSION = "deterministic-cleanup-speaker-fragment-merge-v2"
SPEAKER_PROFILING_SERVICE_VERSION = "speaker-profiling-v1"
ASSIGNMENT_SERVICE_VERSION = "privacy-assignment-policy-v1"
VOICE_CONVERSION_SERVICE_VERSION = "seedvc-batch-adapter-v1"
STITCHING_SERVICE_VERSION = "timeline-dsp-v1"
ASR_ALIGNMENT_SERVICE_VERSION = "asr-speaker-attribution-v1"
TEXT_DEID_SERVICE_VERSION = "llm-hipaa-safe-harbor-v1"
SPEECH_EDITING_SERVICE_VERSION = "voicecraft-editing-v1"
FINAL_BACKGROUND_REMIX_SERVICE_VERSION = "final-background-remix-v1"
SPEAKER_PRIVACY_METRICS_SERVICE_VERSION = "speaker-privacy-metrics-v1"
OVERLAP_REMIX_SERVICE_VERSION = "overlap-remix-service-v1"
EVALUATION_SERVICE_VERSION = "phase9-gates-v1"


def load_job_json(path: str | Path | None = None) -> dict[str, Any]:
    raw_path = path or os.environ.get("AEGISVOX_JOB_JSON")
    if not raw_path:
        raise AegisVoxContractError("AEGISVOX_JOB_JSON is required for service job execution")
    job_path = Path(raw_path)
    payload = json.loads(job_path.read_text())
    if not isinstance(payload, dict):
        raise AegisVoxContractError("job JSON must be an object")
    return payload


def _input_artifacts(job: dict[str, Any]) -> list[str]:
    raw = job.get("input_artifacts")
    if raw is None:
        raw = job.get("input_artifacts_json") or "[]"
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        raise AegisVoxContractError("job input_artifacts must be a list")
    uris = [str(uri) for uri in raw]
    for uri in uris:
        validate_artifact_uri(uri)
    return uris


def _basename(uri: str) -> str:
    parsed = urlparse(uri)
    return PurePosixPath(parsed.path).name


def _find_input(job: dict[str, Any], basename: str, *, default_first: bool = False) -> str:
    uris = _input_artifacts(job)
    for uri in uris:
        if _basename(uri) == basename:
            return uri
    if default_first and uris:
        return uris[0]
    raise AegisVoxContractError(f"job for service {job.get('service')!r} missing input artifact {basename!r}")


def _run_id(job: dict[str, Any]) -> str:
    run_id = str(job.get("run_id") or "")
    if not run_id:
        raise AegisVoxContractError("job missing run_id")
    return run_id


def _service(job: dict[str, Any]) -> str:
    service = str(job.get("service") or "")
    if service not in SERVICE_COMPLETION_EVENTS:
        raise AegisVoxContractError(f"unsupported service job {service!r}")
    return service


def _artifact(run_id: str, name: str, bucket: str) -> str:
    return artifact_uri(run_id, name, bucket=bucket)


def _record(name: str, uri: str, artifact_type: str | None = None, **metadata: Any) -> dict[str, Any]:
    return {
        "name": name,
        "uri": uri,
        "artifact_type": artifact_type or name,
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _source_audio_from_diarization(store: LocalArtifactStore, diarization_uri: str) -> str | None:
    payload = store.read_json(diarization_uri)
    if isinstance(payload, dict) and payload.get("source_audio_uri"):
        return str(payload["source_audio_uri"])
    return None


def _expected_speaker_count_from_manifest(
    job: dict[str, Any],
    artifact_store: LocalArtifactStore,
    bucket: str,
) -> int | None:
    """Return an auditable expected-speaker bound when the manifest provides it.

    This is intentionally conservative. We only infer a bound from explicit
    experiment metadata, e.g. constructed PriMock role-track mixtures where the
    source sidecar names the role tracks. Unknown-count recordings still use the
    automatic over-speaker heuristic rather than an invented speaker count.
    """

    manifest_uri = f"s3://{bucket}/runs/{_run_id(job)}/manifest.json"
    if not artifact_store.exists(manifest_uri):
        return None
    try:
        manifest = artifact_store.read_json(manifest_uri)
    except Exception:  # noqa: BLE001 - segmentation should not fail solely because optional metadata is unreadable
        return None
    request = manifest.get("request") if isinstance(manifest, dict) else None
    experiment_config = request.get("experiment_config") if isinstance(request, dict) else None
    if not isinstance(experiment_config, dict):
        return None
    audio_metadata = experiment_config.get("audio_metadata")
    if not isinstance(audio_metadata, dict):
        return None
    explicit = audio_metadata.get("expected_speaker_count") or audio_metadata.get("num_expected_speakers")
    if explicit is not None:
        try:
            value = int(explicit)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    role_tracks = audio_metadata.get("source_role_tracks")
    if isinstance(role_tracks, dict):
        count = len([value for value in role_tracks.values() if value])
        return count if count > 0 else None
    if isinstance(role_tracks, list):
        count = len([value for value in role_tracks if value])
        return count if count > 0 else None
    return None


def _voice_pool_index_from_env() -> VoicePoolIndex:
    manifest_path = os.environ.get("AEGISVOX_VOICE_POOL_MANIFEST")
    index_path = os.environ.get("AEGISVOX_VOICE_POOL_INDEX")
    if not manifest_path or not index_path:
        raise AegisVoxContractError(
            "assignment worker requires AEGISVOX_VOICE_POOL_MANIFEST and AEGISVOX_VOICE_POOL_INDEX"
        )
    return VoicePoolIndex.from_paths(manifest_path, index_path)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise AegisVoxContractError(f"{name} must be an integer when set") from exc



def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise AegisVoxContractError(f"{name} must be a float when set") from exc


def _stitching_policy_from_env() -> StitchingPolicy:
    kwargs: dict[str, Any] = {}
    bool_fields = {
        "align_duration": "AEGISVOX_STITCHING_ALIGN_DURATION",
        "loudness_normalize": "AEGISVOX_STITCHING_LOUDNESS_NORMALIZE",
        "overlap_add": "AEGISVOX_STITCHING_OVERLAP_ADD",
        "suppress_internal_boundary_fades": "AEGISVOX_STITCHING_SUPPRESS_INTERNAL_BOUNDARY_FADES",
    }
    for field_name, env_name in bool_fields.items():
        if os.environ.get(env_name) not in (None, ""):
            kwargs[field_name] = _env_bool(env_name)
    float_fields = {
        "fade_ms": "AEGISVOX_STITCHING_FADE_MS",
        "boundary_alignment_search_ms": "AEGISVOX_STITCHING_BOUNDARY_ALIGNMENT_SEARCH_MS",
        "boundary_alignment_min_overlap_ms": "AEGISVOX_STITCHING_BOUNDARY_ALIGNMENT_MIN_OVERLAP_MS",
        "boundary_alignment_min_score": "AEGISVOX_STITCHING_BOUNDARY_ALIGNMENT_MIN_SCORE",
        "internal_boundary_gap_ms": "AEGISVOX_STITCHING_INTERNAL_BOUNDARY_GAP_MS",
        "max_duration_drift_sec": "AEGISVOX_STITCHING_MAX_DURATION_DRIFT_SEC",
        "max_loudness_gain": "AEGISVOX_STITCHING_MAX_LOUDNESS_GAIN",
        "min_loudness_gain": "AEGISVOX_STITCHING_MIN_LOUDNESS_GAIN",
    }
    for field_name, env_name in float_fields.items():
        value = _env_float(env_name)
        if value is not None:
            kwargs[field_name] = value
    if os.environ.get("AEGISVOX_STITCHING_FADE_CURVE"):
        kwargs["fade_curve"] = os.environ["AEGISVOX_STITCHING_FADE_CURVE"]
    if os.environ.get("AEGISVOX_STITCHING_BOUNDARY_ALIGNMENT"):
        kwargs["boundary_alignment"] = os.environ["AEGISVOX_STITCHING_BOUNDARY_ALIGNMENT"]
    return StitchingPolicy(**kwargs)

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}






def _speaker_embedding_backend_from_env():
    backend = os.environ.get("AEGISVOX_SPEAKER_EMBEDDING_BACKEND", "speechbrain_local").strip().lower()
    if backend in {"http", "speechbrain_http", "persistent", "persistent_http"}:
        selected = SpeechBrainHTTPEmbeddingBackend()
        assert_persistent_allowed(
            ModelResidency(
                service="speaker_profiling",
                backend=selected.backend_name,
                process_mode=ProcessMode.PERSISTENT_WORKER,
                singleton=True,
                model_load_scope="per persistent SpeechBrain service process",
                endpoint=selected.base_url,
            )
        )
        return selected
    if backend in {"speechbrain_local", "local", "speechbrain"}:
        selected = SpeechBrainECAPAEmbeddingBackend()
        assert_persistent_allowed(
            ModelResidency(
                service="speaker_profiling",
                backend=selected.backend_name,
                process_mode=ProcessMode.IN_PROCESS_CACHED,
                singleton=True,
                model_load_scope="per speaker profiling job process",
                notes="acceptable for small debug jobs; use AEGISVOX_SPEAKER_EMBEDDING_BACKEND=http for batch/full-quality runs",
            )
        )
        return selected
    raise AegisVoxContractError("AEGISVOX_SPEAKER_EMBEDDING_BACKEND must be speechbrain_local or http")


def _voice_conversion_backend_from_env() -> VoiceConversionBackend:
    backend = os.environ.get("AEGISVOX_VOICE_CONVERSION_BACKEND", "seedvc_command").strip().lower()
    if backend in {"identity", "identity_baseline", "no_speaker_conversion", "none"}:
        selected = IdentityVoiceConversionBackend()
        assert_persistent_allowed(
            ModelResidency(
                service="voice_conversion",
                backend=selected.backend_name,
                process_mode=ProcessMode.DETERMINISTIC,
                singleton=True,
                model_load_scope="no model loaded; explicit lexical-only baseline identity copy",
                notes=(
                    "B3 lexical-only/no-speaker-conversion baseline. Requires "
                    "AEGISVOX_ALLOW_IDENTITY_VOICE_CONVERSION_BASELINE=true and is not speaker-privacy-preserving."
                ),
            )
        )
        return selected
    if backend in {"http", "seedvc_http", "persistent", "persistent_http"}:
        selected = SeedVCHTTPBackend()
        assert_persistent_allowed(
            ModelResidency(
                service="voice_conversion",
                backend=selected.backend_name,
                process_mode=ProcessMode.PERSISTENT_WORKER,
                singleton=True,
                model_load_scope="per persistent Seed-VC service process",
                endpoint=selected.base_url,
            )
        )
        return selected
    if backend in {"seedvc_command", "command", "seed-vc"}:
        selected = SeedVCCommandBackend()
        assert_persistent_allowed(
            ModelResidency(
                service="voice_conversion",
                backend=selected.backend_name,
                process_mode=ProcessMode.SUBPROCESS_PER_ITEM,
                singleton=False,
                model_load_scope="per converted segment subprocess",
                notes="debug only; use AEGISVOX_VOICE_CONVERSION_BACKEND=http for production/full-quality runs",
            )
        )
        return selected
    raise AegisVoxContractError("AEGISVOX_VOICE_CONVERSION_BACKEND must be seedvc_command, http, or explicit identity baseline")


def _speech_editing_backend_from_env():
    backend = os.environ.get("AEGISVOX_SPEECH_EDITING_BACKEND", "voicecraft_command").strip().lower()
    if backend in {"http", "voicecraft_http", "persistent", "persistent_http"}:
        selected = VoiceCraftHTTPBackend()
        assert_persistent_allowed(
            ModelResidency(
                service="speech_editing",
                backend=selected.backend_name,
                process_mode=ProcessMode.PERSISTENT_WORKER,
                singleton=True,
                model_load_scope="per persistent VoiceCraft service process",
                endpoint=selected.base_url,
            )
        )
        return selected
    if backend in {"voicecraft_command", "command", "voicecraft"}:
        selected = VoiceCraftCommandBackend()
        assert_persistent_allowed(
            ModelResidency(
                service="speech_editing",
                backend=selected.backend_name,
                process_mode=ProcessMode.SUBPROCESS_PER_ITEM,
                singleton=False,
                model_load_scope="per edited PHI span subprocess",
                notes="debug only; use AEGISVOX_SPEECH_EDITING_BACKEND=http for production/full-quality runs",
            )
        )
        return selected
    raise AegisVoxContractError("AEGISVOX_SPEECH_EDITING_BACKEND must be voicecraft_command or http")


def _assert_source_separation_residency() -> tuple[str, str | None]:
    backend = os.environ.get("AEGISVOX_SOURCE_SEPARATION_BACKEND", "command").strip().lower()
    endpoint = os.environ.get("AEGISVOX_SEPARATOR_SERVICE_URL") or os.environ.get("BS_ROFORMER_SERVICE_URL")
    if backend in {"http", "audio_separator_http", "persistent", "persistent_http", "service"} or endpoint:
        assert_persistent_allowed(
            ModelResidency(
                service="source_separation",
                backend="audio-separator-http",
                process_mode=ProcessMode.PERSISTENT_WORKER,
                singleton=True,
                model_load_scope="per persistent BS-RoFormer/audio-separator service process",
                endpoint=endpoint,
            )
        )
        if backend == "command" and endpoint:
            os.environ["AEGISVOX_SOURCE_SEPARATION_BACKEND"] = "http"
        return "audio-separator-http", endpoint
    assert_persistent_allowed(
        ModelResidency(
            service="source_separation",
            backend="command-separation",
            process_mode=ProcessMode.SUBPROCESS_PER_AUDIO,
            singleton=False,
            model_load_scope="per source-separation command job",
            notes="debug only; use AEGISVOX_SOURCE_SEPARATION_BACKEND=http and AEGISVOX_SEPARATOR_SERVICE_URL for production/full-quality runs",
        )
    )
    return "command-separation", None


def _diarization_service_from_env(artifact_store: LocalArtifactStore):
    backend = os.environ.get("AEGISVOX_DIARIZATION_BACKEND", "pyannote").strip().lower()
    if backend in {"http", "pyannote_http", "persistent", "persistent_http"}:
        selected = PyannoteHTTPDiarizationService(
            artifact_store,
            timeout_sec=_env_int("AEGISVOX_DIARIZATION_TIMEOUT_SEC") or 1800,
        )
        assert_persistent_allowed(
            ModelResidency(
                service="diarization",
                backend=selected.backend_name,
                process_mode=ProcessMode.PERSISTENT_WORKER,
                singleton=True,
                model_load_scope="per persistent pyannote service process",
                endpoint=selected.base_url,
            )
        )
        return selected, {"pyannote_http": selected.backend_version}
    if backend == "pyannote":
        selected = PyannoteDiarizationService(
            artifact_store,
            config=PyannoteDiarizationConfig(
                model_id=os.environ.get("AEGISVOX_PYANNOTE_MODEL_ID", "pyannote/speaker-diarization-3.1"),
                device=os.environ.get("AEGISVOX_PYANNOTE_DEVICE"),
                num_speakers=_env_int("AEGISVOX_PYANNOTE_NUM_SPEAKERS"),
                min_speakers=_env_int("AEGISVOX_PYANNOTE_MIN_SPEAKERS"),
                max_speakers=_env_int("AEGISVOX_PYANNOTE_MAX_SPEAKERS"),
            ),
        )
        assert_persistent_allowed(
            ModelResidency(
                service="diarization",
                backend="pyannote",
                process_mode=ProcessMode.SUBPROCESS_PER_AUDIO,
                singleton=False,
                model_load_scope="per diarization job process",
                notes="debug only; use AEGISVOX_DIARIZATION_BACKEND=http for production/full-quality runs",
            )
        )
        return selected, {"pyannote": selected.config.model_id}
    if backend in {"command", "nemo_sortformer", "nemo-msdd", "nemo_msdd"}:
        env_name = "AEGISVOX_NEMO_SORTFORMER_COMMAND" if backend == "nemo_sortformer" else "AEGISVOX_DIARIZATION_COMMAND"
        normalized_backend = "nemo_msdd" if backend == "nemo-msdd" else backend
        selected = CommandDiarizationService(
            artifact_store,
            config=CommandDiarizationConfig(
                command_env=env_name,
                backend=normalized_backend,
                model_id=os.environ.get("AEGISVOX_DIARIZATION_MODEL_ID", normalized_backend),
                device=os.environ.get("AEGISVOX_DIARIZATION_DEVICE") or os.environ.get("AEGISVOX_PYANNOTE_DEVICE"),
                num_speakers=_env_int("AEGISVOX_DIARIZATION_NUM_SPEAKERS"),
                min_speakers=_env_int("AEGISVOX_DIARIZATION_MIN_SPEAKERS"),
                max_speakers=_env_int("AEGISVOX_DIARIZATION_MAX_SPEAKERS"),
                timeout_seconds=_env_int("AEGISVOX_DIARIZATION_TIMEOUT_SEC"),
            ),
        )
        assert_persistent_allowed(
            ModelResidency(
                service="diarization",
                backend=normalized_backend,
                process_mode=ProcessMode.SUBPROCESS_PER_AUDIO,
                singleton=False,
                model_load_scope="per diarization command job",
                notes="debug only; use AEGISVOX_DIARIZATION_BACKEND=http for production/full-quality runs",
            )
        )
        return selected, {normalized_backend: selected.config.model_id}
    raise AegisVoxContractError("AEGISVOX_DIARIZATION_BACKEND must be pyannote, http, command, nemo_sortformer, or nemo_msdd")


def _assignment_policy_from_env(mode: PrivacyMode) -> AssignmentPolicy:
    policy = AssignmentPolicy.for_privacy_mode(mode)
    overrides: dict[str, Any] = {}
    float_fields = {
        "min_source_target_distance": "AEGISVOX_ASSIGNMENT_MIN_SOURCE_TARGET_DISTANCE",
        "min_target_target_distance": "AEGISVOX_ASSIGNMENT_MIN_TARGET_TARGET_DISTANCE",
        "vector_weight": "AEGISVOX_ASSIGNMENT_VECTOR_WEIGHT",
        "trait_weight": "AEGISVOX_ASSIGNMENT_TRAIT_WEIGHT",
        "min_reference_duration_sec": "AEGISVOX_ASSIGNMENT_MIN_REFERENCE_DURATION_SEC",
    }
    for field_name, env_name in float_fields.items():
        value = _env_float(env_name)
        if value is not None:
            overrides[field_name] = value
    candidate_top_k = _env_int("AEGISVOX_ASSIGNMENT_CANDIDATE_TOP_K")
    if candidate_top_k is not None:
        overrides["candidate_top_k"] = candidate_top_k
    min_reference_sample_rate_hz = _env_int("AEGISVOX_ASSIGNMENT_MIN_REFERENCE_SAMPLE_RATE_HZ")
    if min_reference_sample_rate_hz is not None:
        overrides["min_reference_sample_rate_hz"] = min_reference_sample_rate_hz
    bool_fields = {
        "preserve_language": "AEGISVOX_ASSIGNMENT_PRESERVE_LANGUAGE",
        "preserve_gender_presentation": "AEGISVOX_ASSIGNMENT_PRESERVE_GENDER_PRESENTATION",
        "preserve_accent_group": "AEGISVOX_ASSIGNMENT_PRESERVE_ACCENT_GROUP",
        "preserve_pitch_band": "AEGISVOX_ASSIGNMENT_PRESERVE_PITCH_BAND",
        "preserve_speaking_rate": "AEGISVOX_ASSIGNMENT_PRESERVE_SPEAKING_RATE",
        "preserve_energy": "AEGISVOX_ASSIGNMENT_PRESERVE_ENERGY",
        "preserve_geography_region": "AEGISVOX_ASSIGNMENT_PRESERVE_GEOGRAPHY_REGION",
        "preserve_tone": "AEGISVOX_ASSIGNMENT_PRESERVE_TONE",
        "use_hybrid_audio_traits": "AEGISVOX_ASSIGNMENT_USE_HYBRID_AUDIO_TRAITS",
        "accent_style_constrained": "AEGISVOX_ASSIGNMENT_ACCENT_STYLE_CONSTRAINED",
        "require_native_english_targets": "AEGISVOX_ASSIGNMENT_REQUIRE_NATIVE_ENGLISH_TARGETS",
        "require_native_english_for_north_america": "AEGISVOX_ASSIGNMENT_REQUIRE_NATIVE_ENGLISH_FOR_NORTH_AMERICA",
        "preserve_vocal_weight": "AEGISVOX_ASSIGNMENT_PRESERVE_VOCAL_WEIGHT",
        "preserve_resonance": "AEGISVOX_ASSIGNMENT_PRESERVE_RESONANCE",
        "preserve_voice_maturity": "AEGISVOX_ASSIGNMENT_PRESERVE_VOICE_MATURITY",
    }
    for field_name, env_name in bool_fields.items():
        if os.environ.get(env_name) not in (None, ""):
            overrides[field_name] = _env_bool(env_name)
    if not overrides:
        return policy
    from dataclasses import replace

    return replace(policy, **overrides)


def _asr_backend_version(backend: Any) -> str:
    """Return JSON-safe ASR model/backend provenance.

    The OpenAI backend historically exposed `_model()` as a string. Local
    faster-whisper uses `_model()` to load and return the model object. Never
    call `_model()` for provenance because it can load a large model and can
    make worker payloads non-serializable.
    """

    version = getattr(backend, "backend_version", None)
    if version:
        return str(version)
    for attr in ("model", "model_size", "model_id"):
        value = getattr(backend, attr, None)
        if value:
            return str(value)
    model_getter = getattr(backend, "_model_id", None) or getattr(backend, "_model_name", None)
    if callable(model_getter):
        try:
            return str(model_getter())
        except Exception:
            return str(getattr(backend, "backend_name", "unknown"))
    return str(getattr(backend, "backend_name", "unknown"))


def _run_overlap_stem_remix(
    *,
    artifact_root: Path,
    run_id: str,
    base_audio_uri: str,
    overlap_report_uri: str,
    assignments_uri: str,
    output_audio_uri: str,
    output_report_uri: str,
) -> dict[str, Any]:
    """Load the real overlap stem-remix implementation used by the ORD scripts.

    The runtime implementation lives in `scripts/run_overlap_stem_remix.py`
    because it is also useful as a post-hoc operator tool. The queued service
    adapter loads the same function so the canonical DAG path and post-hoc path
    do not diverge.
    """

    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "run_overlap_stem_remix.py"
    spec = importlib.util.spec_from_file_location("aegisvox_overlap_stem_remix_runtime", script_path)
    if spec is None or spec.loader is None:
        raise AegisVoxContractError(f"cannot load overlap stem remix runtime from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        return float(raw)

    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}

    return module.remix_run(
        artifact_root=artifact_root,
        run_id=run_id,
        base_audio_uri=base_audio_uri,
        overlap_report_uri=overlap_report_uri,
        assignments_uri=assignments_uri,
        output_audio_uri=output_audio_uri,
        output_report_uri=output_report_uri,
        seedvc_command=None
        if os.environ.get("AEGISVOX_VOICE_CONVERSION_BACKEND", "").strip().lower()
        in {"http", "seedvc_http", "persistent", "persistent_http"}
        else os.environ.get("SEEDVC_COMMAND"),
        # Default to protected passthrough for unresolved multi-speaker
        # overlaps. A hard failure drops the whole long-audio row and blocks
        # metric accounting; preserving the already changed-identity timeline
        # keeps speech/content intact while the evaluation/report records the
        # protected overlap regions. Operators can still opt into fail-fast by
        # setting AEGISVOX_OVERLAP_REMIX_ALLOW_MULTI_SPEAKER_SKIP=false.
        fail_on_multi_speaker_region=not _env_bool("AEGISVOX_OVERLAP_REMIX_ALLOW_MULTI_SPEAKER_SKIP", True),
        fade_ms=_env_float("AEGISVOX_OVERLAP_REMIX_FADE_MS", 5.0),
        context_edge_gain=_env_bool("AEGISVOX_OVERLAP_REMIX_CONTEXT_EDGE_GAIN", False),
        single_speaker_target_rms_ratio=_env_float("AEGISVOX_OVERLAP_REMIX_SINGLE_SPEAKER_TARGET_RMS_RATIO", 0.90),
        single_speaker_min_rms_gain=_env_float("AEGISVOX_OVERLAP_REMIX_SINGLE_SPEAKER_MIN_RMS_GAIN", 0.05),
        single_speaker_max_rms_gain=_env_float("AEGISVOX_OVERLAP_REMIX_SINGLE_SPEAKER_MAX_RMS_GAIN", 1.0),
        multi_speaker_gain=_env_float("AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_GAIN", 0.45),
        multi_speaker_background_gain=_env_float("AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_BACKGROUND_GAIN", 0.0),
        multi_speaker_target_rms_ratio=_env_float("AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_TARGET_RMS_RATIO", 0.85),
        multi_speaker_min_rms_gain=_env_float("AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_MIN_RMS_GAIN", 0.05),
        multi_speaker_max_rms_gain=_env_float("AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_MAX_RMS_GAIN", 1.0),
        multi_speaker_min_assignment_margin=_env_float("AEGISVOX_OVERLAP_REMIX_MULTI_SPEAKER_MIN_ASSIGNMENT_MARGIN", 0.03),
        disable_multi_speaker_assignment_gate=_env_bool("AEGISVOX_OVERLAP_REMIX_DISABLE_MULTI_SPEAKER_ASSIGNMENT_GATE", False),
    )


def _evaluation_inputs_from_job(job: dict[str, Any], *, bucket: str = "aegisvox") -> EvaluationInputs:
    by_name = {_basename(uri): uri for uri in _input_artifacts(job)}
    run_id = _run_id(job)
    manifest_uri = next((uri for uri in _input_artifacts(job) if _basename(uri) == "manifest.json"), None)
    if manifest_uri is None:
        manifest_uri = f"s3://{bucket}/runs/{run_id}/manifest.json"
    return EvaluationInputs(
        speaker_profiles_uri=by_name.get("speaker_profiles.json"),
        assignments_uri=by_name.get("speaker_target_map.json"),
        converted_segments_uri=by_name.get("segments.converted.json"),
        stitch_report_uri=by_name.get("stitch_report.json"),
        asr_uri=by_name.get("asr.changed_identity.json"),
        deid_text_uri=by_name.get("deid_text.json"),
        voicecraft_edit_jobs_uri=by_name.get("voicecraft_edit_jobs.json"),
        final_sanitized_audio_uri=by_name.get("final_sanitized.wav"),
        speaker_privacy_metrics_uri=by_name.get("speaker_privacy_metrics.json"),
        changed_identity_audio_uri=by_name.get("changed_identity.wav"),
        run_manifest_uri=manifest_uri,
        overlap_remix_apply_report_uri=by_name.get("overlap_remix_apply_report.json"),
        overlap_remixed_audio_uri=by_name.get("overlap_remixed_identity.wav"),
        background_leakage_report_uri=by_name.get("background_leakage_report.json"),
        final_sanitized_with_background_audio_uri=by_name.get("final_sanitized_with_background.wav"),
        final_background_remix_report_uri=by_name.get("final_background_remix_report.json"),
        diarization_comparison_report_uri=by_name.get("diarization_backend_matrix.json")
        or by_name.get("diarization_comparison_report.json"),
    )



def _evaluation_policy_from_manifest(job: dict[str, Any], artifact_store: LocalArtifactStore, bucket: str) -> EvaluationPolicy:
    manifest_uri = f"s3://{bucket}/runs/{_run_id(job)}/manifest.json"
    require_lexical = True
    require_speech_editing = True
    if artifact_store.exists(manifest_uri):
        manifest = artifact_store.read_json(manifest_uri)
        request = manifest.get("request") if isinstance(manifest, dict) else None
        if isinstance(request, dict):
            require_lexical = bool(request.get("enable_lexical_deid", True))
            require_speech_editing = bool(request.get("enable_speech_editing", True))
    return EvaluationPolicy(
        require_lexical_deid_artifacts=require_lexical,
        require_speech_editing_artifacts=require_speech_editing,
    )

def run_service_job(job: dict[str, Any], *, artifact_store: LocalArtifactStore, bucket: str = "aegisvox") -> ServiceJobResult:
    service = _service(job)
    run_id = _run_id(job)
    completion_event = SERVICE_COMPLETION_EVENTS[service]
    prefix_uri = _artifact(run_id, "", bucket).rstrip("/")

    if service == "source_separation":
        source_audio_uri = _find_input(job, "input.wav", default_first=True)
        vocal_uri = _artifact(run_id, "foreground_vocals.wav", bucket)
        background_uri = _artifact(run_id, "preserved_background.wav", bucket)
        report_uri = _artifact(run_id, "source_separation_report.json", bucket)
        source_sep_backend, source_sep_endpoint = _assert_source_separation_residency()
        payload = SourceSeparationService(artifact_store).separate_to_artifacts(
            input_audio_uri=source_audio_uri,
            output_vocal_uri=vocal_uri,
            output_background_uri=background_uri,
            output_report_uri=report_uri,
            output_prefix_uri=f"{prefix_uri}/source_separation",
        )
        return ServiceJobResult(
            completion_event,
            [
                _record("source_separation_report.json", report_uri, "source_separation_report", status=payload.get("status")),
                _record("foreground_vocals.wav", vocal_uri, "audio", role="foreground_vocals"),
                _record("preserved_background.wav", background_uri, "audio", role="preserved_background"),
            ],
            {"status": payload.get("status")},
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "source_separation": SOURCE_SEPARATION_SERVICE_VERSION,
            },
            model_versions={"background_separator": str(payload.get("backend_version") or source_sep_backend), "source_separation_endpoint": source_sep_endpoint or ""},
        )

    if service == "diarization":
        source_audio_uri = _find_input(job, "input.wav", default_first=True)
        output_uri = _artifact(run_id, "diarization.raw.json", bucket)
        diarization_service, model_versions = _diarization_service_from_env(artifact_store)
        payload = diarization_service.diarize_to_artifact(source_audio_uri, output_uri)
        return ServiceJobResult(
            completion_event,
            [_record("diarization.raw.json", output_uri, "diarization")],
            {
                "num_segments": payload.get("num_segments"),
                "backend": payload.get("backend"),
                "num_speakers": len({row.get("speaker_id") for row in payload.get("segments", []) if isinstance(row, dict)}),
            },
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "diarization_service": DIARIZATION_SERVICE_VERSION,
            },
            model_versions=model_versions,
        )

    if service == "background_leakage_check":
        background_uri = _find_input(job, "preserved_background.wav", default_first=True)
        report_uri = _artifact(run_id, "background_leakage_report.json", bucket)
        asr_uri = _artifact(run_id, "background_leakage_asr.json", bucket)
        asr_backend = create_asr_backend()
        payload = BackgroundLeakageService(
            artifact_store,
            backend=asr_backend,
            llm_backend=OpenAILLMTextDeIDBackend(),
        ).check_to_artifact(
            background_audio_uri=background_uri,
            output_report_uri=report_uri,
        )
        return ServiceJobResult(
            completion_event,
            [
                _record("background_leakage_report.json", report_uri, "background_leakage_report", status=payload.get("status")),
                _record("background_leakage_asr.json", asr_uri, "asr_alignment", purpose="background_leakage_check"),
            ],
            {
                "status": payload.get("status"),
                "text_chars": payload.get("text_chars"),
                "num_high_confidence_phi": payload.get("num_high_confidence_phi"),
            },
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "background_leakage_check": BACKGROUND_LEAKAGE_SERVICE_VERSION,
            },
            model_versions={asr_backend.backend_name: _asr_backend_version(asr_backend)},
        )

    if service == "segmentation":
        diarization_uri = _find_input(job, "diarization.raw.json", default_first=True)
        output_uri = _artifact(run_id, "segments.cleaned.json", bucket)
        source_audio_uri = _source_audio_from_diarization(artifact_store, diarization_uri)
        duration = wav_duration_sec(artifact_store.local_path(source_audio_uri)) if source_audio_uri else None
        payload = SegmentationService(artifact_store).clean_to_artifact(
            diarization_uri,
            output_uri,
            source_audio_uri=source_audio_uri,
            audio_duration_sec=duration,
            output_segment_prefix_uri=f"{prefix_uri}/segments",
            expected_speaker_count=_env_int("AEGISVOX_EXPECTED_SPEAKER_COUNT")
            or _env_int("AEGISVOX_SEGMENTATION_EXPECTED_SPEAKER_COUNT")
            or _env_int("AEGISVOX_DIARIZATION_NUM_SPEAKERS")
            or _expected_speaker_count_from_manifest(job, artifact_store, bucket),
            exclusive_speaker_timeline=_env_bool("AEGISVOX_SEGMENTATION_EXCLUSIVE_SPEAKER_TIMELINE", False),
        )
        return ServiceJobResult(
            completion_event,
            [_record("segments.cleaned.json", output_uri, "segmentation", extracted_audio=True)],
            {"num_segments": payload.get("num_segments")},
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "segmentation": SEGMENTATION_SERVICE_VERSION,
            },
        )

    if service == "speaker_profiling":
        segments_uri = _find_input(job, "segments.cleaned.json", default_first=True)
        output_uri = _artifact(run_id, "speaker_profiles.json", bucket)
        embedding_backend = _speaker_embedding_backend_from_env()
        payload = SpeakerProfilingService(
            artifact_store,
            embedding_backend=embedding_backend,
        ).profile_to_artifact(segments_uri, output_uri, output_prefix_uri=prefix_uri)
        return ServiceJobResult(
            completion_event,
            [_record("speaker_profiles.json", output_uri, "speaker_profiles")],
            {"num_speakers": payload.get("num_speakers")},
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "speaker_profiling": SPEAKER_PROFILING_SERVICE_VERSION,
            },
            model_versions={embedding_backend.backend_name: embedding_backend.backend_version},
        )

    if service == "assignment":
        profiles_uri = _find_input(job, "speaker_profiles.json", default_first=True)
        output_uri = _artifact(run_id, "speaker_target_map.json", bucket)
        report_uri = _artifact(run_id, "assignment_report.md", bucket)
        mode = PrivacyMode(os.environ.get("AEGISVOX_PRIVACY_MODE", "medium"))
        voice_pool_index = _voice_pool_index_from_env()
        source_trait_profiles_uri = None
        source_trait_report_uri = None
        if _env_bool("AEGISVOX_ASSIGNMENT_EXTRACT_SOURCE_AUDIO_TRAITS", False):
            source_trait_profiles_uri = _artifact(run_id, "speaker_profiles.audio_traits.json", bucket)
            source_trait_report_uri = _artifact(run_id, "speaker_profiles.audio_traits_report.json", bucket)
            enrich_speaker_profiles_with_audio_traits(
                artifact_store=artifact_store,
                speaker_profiles_uri=profiles_uri,
                output_profiles_uri=source_trait_profiles_uri,
                report_uri=source_trait_report_uri,
                policy=SpeakerProfileTraitEnrichmentPolicy(
                    model=os.environ.get("AEGISVOX_AUDIO_TRAIT_MODEL", "gpt-audio-1.5"),
                    continue_on_error=_env_bool("AEGISVOX_ASSIGNMENT_SOURCE_AUDIO_TRAITS_CONTINUE_ON_ERROR", False),
                ),
            )
            profiles_uri = source_trait_profiles_uri
        payload = AssignmentService(
            artifact_store,
            voice_pool_index=voice_pool_index,
            policy=_assignment_policy_from_env(mode),
        ).assign_to_artifact(profiles_uri, output_uri, output_report_uri=report_uri)
        return ServiceJobResult(
            completion_event,
            [
                _record("speaker_target_map.json", output_uri, "speaker_assignment"),
                _record("assignment_report.md", report_uri, "assignment_report"),
            ]
            + (
                [
                    _record("speaker_profiles.audio_traits.json", source_trait_profiles_uri, "speaker_profiles", enrichment="audio_traits"),
                    _record(
                        "speaker_profiles.audio_traits_report.json",
                        source_trait_report_uri,
                        "speaker_profile_audio_trait_report",
                    ),
                ]
                if source_trait_profiles_uri and source_trait_report_uri
                else []
            ),
            {
                "num_assignments": len(payload.get("assignments") or {}),
                "source_audio_trait_enrichment": bool(source_trait_profiles_uri),
            },
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "assignment": ASSIGNMENT_SERVICE_VERSION,
            },
            model_versions={
                "voice_pool": f"{voice_pool_index.manifest.voice_pool_id}:{voice_pool_index.manifest.version}",
            },
        )

    if service == "voice_conversion":
        segments_uri = _find_input(job, "segments.cleaned.json")
        assignments_uri = _find_input(job, "speaker_target_map.json")
        output_uri = _artifact(run_id, "segments.converted.json", bucket)
        conversion_backend = _voice_conversion_backend_from_env()
        payload = BatchVoiceConversionService(
            artifact_store,
            backend=conversion_backend,
        ).convert_batch_to_artifact(
            segments_uri=segments_uri,
            assignments_uri=assignments_uri,
            output_segments_uri=output_uri,
            output_prefix_uri=f"{prefix_uri}/converted_segments",
            options={
                "num_workers": _env_int("AEGISVOX_VOICE_CONVERSION_WORKERS") or 1,
                "cuda_devices": os.environ.get("AEGISVOX_VOICE_CONVERSION_CUDA_DEVICES", ""),
                "timeout_sec": _env_int("AEGISVOX_VOICE_CONVERSION_TIMEOUT_SEC") or 600,
            },
        )
        return ServiceJobResult(
            completion_event,
            [_record("segments.converted.json", output_uri, "voice_conversion")],
            {
                "num_converted": payload.get("num_converted"),
                "num_failed": payload.get("num_failed"),
                "conversion_parallelism": payload.get("conversion_parallelism"),
            },
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "voice_conversion": VOICE_CONVERSION_SERVICE_VERSION,
            },
            model_versions={conversion_backend.backend_name: conversion_backend.backend_version},
        )

    if service == "stitching":
        converted_uri = _find_input(job, "segments.converted.json", default_first=True)
        audio_uri = _artifact(run_id, "changed_identity.wav", bucket)
        report_uri = _artifact(run_id, "stitch_report.json", bucket)
        payload = StitchingService(artifact_store, policy=_stitching_policy_from_env()).stitch_to_artifact(
            converted_segments_uri=converted_uri,
            output_audio_uri=audio_uri,
            stitch_report_uri=report_uri,
        )
        return ServiceJobResult(
            completion_event,
            [
                _record("changed_identity.wav", audio_uri, "audio", duration_sec=payload.get("duration_sec")),
                _record("stitch_report.json", report_uri, "stitch_report"),
            ],
            {"num_stitched": payload.get("num_stitched"), "num_not_stitched": payload.get("num_not_stitched")},
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "stitching": STITCHING_SERVICE_VERSION,
            },
        )

    if service == "asr_alignment":
        audio_uri = _find_input(job, "changed_identity.wav", default_first=True)
        diarization_uri = next((uri for uri in _input_artifacts(job) if _basename(uri) == "diarization.raw.json"), None)
        output_uri = _artifact(run_id, "asr.changed_identity.json", bucket)
        asr_backend = create_asr_backend()
        payload = ASRAlignmentService(artifact_store, backend=asr_backend).transcribe_to_artifact(
            audio_uri=audio_uri,
            output_asr_uri=output_uri,
            diarization_uri=diarization_uri,
        )
        return ServiceJobResult(
            completion_event,
            [_record("asr.changed_identity.json", output_uri, "asr_alignment")],
            {"num_segments": len(payload.get("segments") or []), "num_words": len(payload.get("words") or [])},
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "asr_alignment": ASR_ALIGNMENT_SERVICE_VERSION,
            },
            model_versions={asr_backend.backend_name: _asr_backend_version(asr_backend)},
        )

    if service == "text_deid":
        asr_uri = _find_input(job, "asr.changed_identity.json", default_first=True)
        output_uri = _artifact(run_id, "deid_text.json", bucket)
        policy_mode = os.environ.get("AEGISVOX_TEXT_DEID_MODE", "hybrid")
        use_llm_replacements = os.environ.get("AEGISVOX_TEXT_DEID_LLM_REPLACEMENTS", "").lower() in {"1", "true", "yes"}
        payload = TextDeIDService(
            artifact_store,
            policy=DeIDPolicy(mode=policy_mode),
            llm_backend=OpenAILLMTextDeIDBackend(),
        ).deidentify_to_artifact(
            asr_uri=asr_uri,
            output_deid_uri=output_uri,
            use_llm_replacements=use_llm_replacements,
        )
        return ServiceJobResult(
            completion_event,
            [_record("deid_text.json", output_uri, "text_deid")],
            {"num_edits": len(payload.get("edits") or [])},
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "text_deid": TEXT_DEID_SERVICE_VERSION,
            },
        )

    if service == "speech_editing":
        changed_identity_uri = _find_input(job, "changed_identity.wav")
        deid_uri = _find_input(job, "deid_text.json")
        jobs_uri = _artifact(run_id, "voicecraft_edit_jobs.json", bucket)
        final_uri = _artifact(run_id, "final_sanitized.wav", bucket)
        editing_timeout = _env_int("AEGISVOX_SPEECH_EDITING_TIMEOUT_SEC") or 1200
        validate_spans = os.environ.get("AEGISVOX_VALIDATE_EDITED_SPANS", "").strip().lower() in {"1", "true", "yes"}
        require_validated = os.environ.get("AEGISVOX_REQUIRE_VALIDATED_REPLACEMENTS", "").strip().lower() in {"1", "true", "yes"}
        span_validator = FasterWhisperEditedSpanValidator() if (validate_spans or require_validated) else None
        payload = SpeechEditingService(artifact_store, backend=_speech_editing_backend_from_env(), edited_span_validator=span_validator).edit_to_artifacts(
            changed_identity_audio_uri=changed_identity_uri,
            deid_text_uri=deid_uri,
            output_jobs_uri=jobs_uri,
            output_span_prefix_uri=f"{prefix_uri}/edited_spans",
            final_audio_uri=final_uri,
            options={
                "timeout_sec": editing_timeout,
                "num_workers": _env_int("AEGISVOX_SPEECH_EDITING_WORKERS") or _env_int("AEGISVOX_VOICE_CONVERSION_WORKERS") or 1,
                "cuda_devices": os.environ.get("AEGISVOX_SPEECH_EDITING_CUDA_DEVICES")
                or os.environ.get("AEGISVOX_VOICE_CONVERSION_CUDA_DEVICES"),
                "require_validated_replacements": require_validated,
                "allow_partial_failures": os.environ.get("AEGISVOX_SPEECH_EDITING_ALLOW_PARTIAL_FAILURES", "").strip().lower() in {"1", "true", "yes"},
            },
        )
        return ServiceJobResult(
            completion_event,
            [
                _record("voicecraft_edit_jobs.json", jobs_uri, "speech_editing_jobs"),
                _record("final_sanitized.wav", final_uri, "audio", status=payload.get("status")),
            ],
            {"num_jobs": payload.get("num_jobs"), "num_unlocalized_edits": payload.get("num_unlocalized_edits")},
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "speech_editing": SPEECH_EDITING_SERVICE_VERSION,
            },
            model_versions={
                "speech_editing_backend": str(payload.get("backend_version") or payload.get("status") or "unknown"),
            },
        )

    if service == "speaker_privacy_metrics":
        profiles_uri = _find_input(job, "speaker_profiles.json")
        assignments_uri = _find_input(job, "speaker_target_map.json")
        converted_uri = _find_input(job, "segments.converted.json")
        output_uri = _artifact(run_id, "speaker_privacy_metrics.json", bucket)
        embedding_backend = _speaker_embedding_backend_from_env()
        payload = SpeakerPrivacyMetricsService(
            artifact_store,
            embedding_backend=embedding_backend,
        ).compute_to_artifact(
            speaker_profiles_uri=profiles_uri,
            assignments_uri=assignments_uri,
            converted_segments_uri=converted_uri,
            output_metrics_uri=output_uri,
            output_prefix_uri=f"{prefix_uri}/privacy_metrics",
        )
        return ServiceJobResult(
            completion_event,
            [_record("speaker_privacy_metrics.json", output_uri, "speaker_privacy_metrics")],
            dict(payload.get("summary") or {}),
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "speaker_privacy_metrics": SPEAKER_PRIVACY_METRICS_SERVICE_VERSION,
            },
            model_versions={embedding_backend.backend_name: embedding_backend.backend_version},
        )

    if service == "final_background_remix":
        foreground_uri = _find_input(job, "final_sanitized.wav")
        background_uri = _find_input(job, "preserved_background.wav")
        try:
            leakage_report_uri = _find_input(job, "background_leakage_report.json")
        except AegisVoxContractError:
            leakage_report_uri = None
        output_uri = _artifact(run_id, "final_sanitized_with_background.wav", bucket)
        report_uri = _artifact(run_id, "final_background_remix_report.json", bucket)
        payload = FinalBackgroundRemixService(artifact_store).remix_to_artifacts(
            foreground_audio_uri=foreground_uri,
            background_audio_uri=background_uri,
            background_leakage_report_uri=leakage_report_uri,
            output_audio_uri=output_uri,
            output_report_uri=report_uri,
        )
        return ServiceJobResult(
            completion_event,
            [
                _record("final_sanitized_with_background.wav", output_uri, "audio", status=payload.get("status")),
                _record("final_background_remix_report.json", report_uri, "final_background_remix_report"),
            ],
            {
                "status": payload.get("status"),
                "duration_sec": payload.get("duration_sec"),
                "background_policy_action": payload.get("background_policy_action"),
                "effective_background_gain": payload.get("effective_background_gain"),
            },
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "final_background_remix": FINAL_BACKGROUND_REMIX_SERVICE_VERSION,
            },
        )

    if service == "overlap_remix":
        segments_uri = _find_input(job, "segments.cleaned.json")
        assignments_uri = _find_input(job, "speaker_target_map.json")
        base_audio_uri = _find_input(job, "changed_identity.wav")
        overlap_report_uri = _artifact(run_id, "overlap_remix_report.json", bucket)
        remixed_audio_uri = _artifact(run_id, "overlap_remixed_identity.wav", bucket)
        apply_report_uri = _artifact(run_id, "overlap_remix_apply_report.json", bucket)
        overlap_payload = OverlapHandlingService(artifact_store).handle_to_artifact(
            segments_uri=segments_uri,
            output_overlap_uri=overlap_report_uri,
            mode=OverlapMode.SEPARATE_CONVERT_REMIX,
            output_prefix_uri=f"{prefix_uri}/overlap_handling",
        )
        artifacts = [_record("overlap_remix_report.json", overlap_report_uri, "overlap_remix_report")]
        if int(overlap_payload.get("num_overlap_regions") or 0) == 0:
            apply_payload = {
                "service": "overlap_stem_remix",
                "status": "no_overlaps",
                "run_id": run_id,
                "base_audio_uri": base_audio_uri,
                "output_audio_uri": None,
                "overlap_report_uri": overlap_report_uri,
                "assignments_uri": assignments_uri,
                "num_regions": 0,
                "num_converted_vocals": 0,
                "num_remixed_regions": 0,
                "claim_boundary": "no overlap regions were detected; no overlap-remixed audio candidate was produced",
            }
            artifact_store.write_json(apply_report_uri, apply_payload)
        else:
            apply_payload = _run_overlap_stem_remix(
                artifact_root=artifact_store.root,
                run_id=run_id,
                base_audio_uri=base_audio_uri,
                overlap_report_uri=overlap_report_uri,
                assignments_uri=assignments_uri,
                output_audio_uri=remixed_audio_uri,
                output_report_uri=apply_report_uri,
            )
            artifacts.append(_record("overlap_remixed_identity.wav", remixed_audio_uri, "audio", status=apply_payload.get("status")))
        artifacts.append(_record("overlap_remix_apply_report.json", apply_report_uri, "overlap_remix_apply_report"))
        return ServiceJobResult(
            completion_event,
            artifacts,
            {
                "num_overlap_regions": overlap_payload.get("num_overlap_regions"),
                "num_remixed_regions": apply_payload.get("num_remixed_regions"),
                "status": apply_payload.get("status"),
            },
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "overlap_remix": OVERLAP_REMIX_SERVICE_VERSION,
            },
            model_versions={
                "overlap_separation": "audio-separator-http-or-command",
                "overlap_voice_conversion": str(apply_payload.get("voice_conversion_backend_version") or apply_payload.get("status") or "unknown"),
            },
        )

    if service == "evaluation":
        report_uri = _artifact(run_id, "evaluation_report.json", bucket)
        markdown_uri = _artifact(run_id, "evaluation_report.md", bucket)
        pass_fail_uri = _artifact(run_id, "pass_fail.json", bucket)
        output_policy_uri = _artifact(run_id, "output_policy_report.json", bucket)
        eval_inputs = _evaluation_inputs_from_job(job, bucket=bucket)
        payload = EvaluationService(artifact_store, policy=_evaluation_policy_from_manifest(job, artifact_store, bucket)).evaluate_to_artifacts(
            inputs=eval_inputs,
            output_report_uri=report_uri,
            output_markdown_uri=markdown_uri,
            output_pass_fail_uri=pass_fail_uri,
        )
        pass_fail_payload = artifact_store.read_json(pass_fail_uri)
        output_policy = build_output_policy(
            OutputPolicyInputs(
                canonical_audio_uri=eval_inputs.final_sanitized_audio_uri,
                background_candidate_audio_uri=eval_inputs.final_sanitized_with_background_audio_uri,
                evaluation_report=payload,
                pass_fail=pass_fail_payload,
                background_leakage_report=artifact_store.read_json(eval_inputs.background_leakage_report_uri)
                if eval_inputs.background_leakage_report_uri and artifact_store.exists(eval_inputs.background_leakage_report_uri)
                else None,
                final_background_remix_report=artifact_store.read_json(eval_inputs.final_background_remix_report_uri)
                if eval_inputs.final_background_remix_report_uri and artifact_store.exists(eval_inputs.final_background_remix_report_uri)
                else None,
            )
        )
        artifact_store.write_json(output_policy_uri, output_policy)
        return ServiceJobResult(
            completion_event,
            [
                _record("evaluation_report.json", report_uri, "evaluation_report", overall_status=payload.get("overall_status")),
                _record("evaluation_report.md", markdown_uri, "evaluation_report_markdown"),
                _record("pass_fail.json", pass_fail_uri, "evaluation_pass_fail"),
                _record(
                    "output_policy_report.json",
                    output_policy_uri,
                    "output_policy_report",
                    background_candidate_status=(output_policy.get("background_candidate") or {}).get("status"),
                    recommended_for_download=output_policy.get("recommended_for_download"),
                ),
            ],
            {
                "overall_status": payload.get("overall_status"),
                "overall_pass": payload.get("overall_pass"),
                "background_candidate_status": (output_policy.get("background_candidate") or {}).get("status"),
                "recommended_for_download": output_policy.get("recommended_for_download"),
            },
            service_versions={
                "service_job": SERVICE_JOB_ADAPTER_VERSION,
                "evaluation": EVALUATION_SERVICE_VERSION,
            },
        )

    raise AegisVoxContractError(f"unsupported service job {service!r}")
