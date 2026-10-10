"""Central runtime preflight and provenance registry for AegisVox.

The registry gives operators one place to answer two questions before launching
real ORD workers:

1. Which service/model version keys should appear in the run manifest?
2. Are the configured runtime backends available, or will they fail loudly?

This module does not create mock outputs and does not call external inference
APIs. Checks validate packages, tokens/env, command templates, model/cache paths,
and configured voice-pool artifacts. Model-backed services still perform their
own fail-loud preflight at execution time.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .asr_alignment import ASRAlignmentService, ASRSetupError, create_asr_backend
from .auth import APIAuthConfig
from .audit import sanitize_for_audit
from .diarization import DEFAULT_PYANNOTE_MODEL_ID, TOKEN_ENV_NAMES, import_pyannote_pipeline
from .schema import AegisVoxContractError
from .service_job import (
    ASR_ALIGNMENT_SERVICE_VERSION,
    ASSIGNMENT_SERVICE_VERSION,
    BACKGROUND_LEAKAGE_SERVICE_VERSION,
    DIARIZATION_SERVICE_VERSION,
    EVALUATION_SERVICE_VERSION,
    FINAL_BACKGROUND_REMIX_SERVICE_VERSION,
    OVERLAP_REMIX_SERVICE_VERSION,
    SEGMENTATION_SERVICE_VERSION,
    SERVICE_JOB_ADAPTER_VERSION,
    SOURCE_SEPARATION_SERVICE_VERSION,
    SPEAKER_PROFILING_SERVICE_VERSION,
    SPEAKER_PRIVACY_METRICS_SERVICE_VERSION,
    SPEECH_EDITING_SERVICE_VERSION,
    STITCHING_SERVICE_VERSION,
    TEXT_DEID_SERVICE_VERSION,
    VOICE_CONVERSION_SERVICE_VERSION,
)
from .speaker_profile import SpeakerProfilingSetupError, SpeechBrainECAPAEmbeddingBackend
from .speech_editing import SpeechEditingService, SpeechEditingSetupError
from .storage import LocalArtifactStore
from .voice_conversion import SeedVCCommandBackend, VoiceConversionSetupError
from .voice_pool import VoicePoolIndex

CheckFn = Callable[["PreflightContext", "PreflightDefinition"], dict[str, Any]]


@dataclass(frozen=True)
class PreflightContext:
    artifact_root: Path = Path("/tmp/aegisvox_preflight")
    deep: bool = False


@dataclass(frozen=True)
class PreflightDefinition:
    service: str
    description: str
    required: bool
    service_versions: dict[str, str] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    required_env_any: tuple[str, ...] = ()
    required_env_all: tuple[str, ...] = ()
    optional_env: tuple[str, ...] = ()
    check: CheckFn | None = None


def _env_presence(names: tuple[str, ...]) -> dict[str, bool]:
    return {name: bool(os.environ.get(name)) for name in names}


def _env_requirements(definition: PreflightDefinition) -> dict[str, Any]:
    return {
        "required_any": list(definition.required_env_any),
        "required_all": list(definition.required_env_all),
        "optional": list(definition.optional_env),
        "present": _env_presence(definition.required_env_any + definition.required_env_all + definition.optional_env),
    }


def _check_env_requirements(definition: PreflightDefinition) -> None:
    if definition.required_env_any and not any(os.environ.get(name) for name in definition.required_env_any):
        raise RuntimeError("one of these runtime env vars is required: " + ", ".join(definition.required_env_any))
    missing_all = [name for name in definition.required_env_all if not os.environ.get(name)]
    if missing_all:
        raise RuntimeError("missing required runtime env vars: " + ", ".join(missing_all))


def _check_static(_: PreflightContext, __: PreflightDefinition) -> dict[str, Any]:
    return {"ok": True, "backend": "deterministic"}


def _check_orchestrator(_: PreflightContext, __: PreflightDefinition) -> dict[str, Any]:
    auth = APIAuthConfig.from_env()
    return {"ok": True, "service": "orchestrator", "version": __version__, "api_auth": auth.status()}


def _check_pyannote(_: PreflightContext, definition: PreflightDefinition) -> dict[str, Any]:
    _check_env_requirements(definition)
    import_pyannote_pipeline()
    return {"ok": True, "backend": "pyannote", "model_id": DEFAULT_PYANNOTE_MODEL_ID, "token_present": True}


def _check_speechbrain(context: PreflightContext, _: PreflightDefinition) -> dict[str, Any]:
    backend = SpeechBrainECAPAEmbeddingBackend()
    if context.deep:
        backend._load()
        return {"ok": True, "backend": backend.backend_name, "model_id": backend.model_id, "depth": "model_load"}
    has_speechbrain = importlib.util.find_spec("speechbrain") is not None
    if not has_speechbrain:
        raise SpeakerProfilingSetupError("speechbrain package is not installed; install voicepool/speaker profiling dependencies")
    return {"ok": True, "backend": backend.backend_name, "model_id": backend.model_id, "depth": "package_only"}


def _check_voice_pool(_: PreflightContext, definition: PreflightDefinition) -> dict[str, Any]:
    _check_env_requirements(definition)
    manifest_path = Path(os.environ["AEGISVOX_VOICE_POOL_MANIFEST"])
    index_path = Path(os.environ["AEGISVOX_VOICE_POOL_INDEX"])
    if not manifest_path.exists():
        raise AegisVoxContractError(f"voice-pool manifest not found: {manifest_path}")
    if not index_path.exists():
        raise AegisVoxContractError(f"voice-pool index not found: {index_path}")
    index = VoicePoolIndex.from_paths(manifest_path, index_path)
    return {
        "ok": True,
        "backend": "voice-pool-index",
        "voice_pool_id": index.manifest.voice_pool_id,
        "version": index.manifest.version,
        "num_samples": len(index.samples),
    }


def _check_seedvc(_: PreflightContext, __: PreflightDefinition) -> dict[str, Any]:
    return SeedVCCommandBackend().preflight()


def _check_asr(context: PreflightContext, __: PreflightDefinition) -> dict[str, Any]:
    return ASRAlignmentService(LocalArtifactStore(context.artifact_root), backend=create_asr_backend()).preflight()


def _check_voicecraft(context: PreflightContext, __: PreflightDefinition) -> dict[str, Any]:
    return SpeechEditingService(LocalArtifactStore(context.artifact_root)).preflight()


def _check_overlap_remix(_: PreflightContext, definition: PreflightDefinition) -> dict[str, Any]:
    _check_env_requirements(definition)
    seedvc = SeedVCCommandBackend().preflight()
    return {
        "ok": True,
        "overlap_separation_backend": "command-separation",
        "seedvc": seedvc,
    }


def _check_source_separation(_: PreflightContext, definition: PreflightDefinition) -> dict[str, Any]:
    _check_env_requirements(definition)
    return {"ok": True, "backend": "bs-roformer-command-separation"}


PREFLIGHT_DEFINITIONS: dict[str, PreflightDefinition] = {
    "orchestrator": PreflightDefinition(
        service="orchestrator",
        description="FastAPI/control-plane orchestrator version",
        required=True,
        service_versions={"orchestrator": __version__},
        check=_check_orchestrator,
    ),
    "source_separation": PreflightDefinition(
        service="source_separation",
        description="BS-RoFormer vocal/background preprocessing for background-preserving mode",
        required=False,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "source_separation": SOURCE_SEPARATION_SERVICE_VERSION},
        model_versions={"background_separator": "bs-roformer:command-separation"},
        required_env_all=("OVERLAP_SEPARATION_COMMAND",),
        check=_check_source_separation,
    ),
    "background_leakage_check": PreflightDefinition(
        service="background_leakage_check",
        description="ASR/LLM-HIPAA-PHI leakage screen for preserved background stem",
        required=False,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "background_leakage_check": BACKGROUND_LEAKAGE_SERVICE_VERSION},
        model_versions={"openai-asr": os.environ.get("OPENAI_ASR_MODEL") or os.environ.get("AEGISVOX_OPENAI_ASR_MODEL") or "gpt-4o-transcribe"},
        required_env_any=("OPENAI_API_KEY", "OPENAI_API_KEY"),
        optional_env=("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_ASR_MODEL", "AEGISVOX_OPENAI_ASR_MODEL"),
        check=_check_asr,
    ),
    "diarization": PreflightDefinition(
        service="diarization",
        description="pyannote speaker diarization backend",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "diarization_service": DIARIZATION_SERVICE_VERSION},
        model_versions={"pyannote": DEFAULT_PYANNOTE_MODEL_ID},
        required_env_any=TOKEN_ENV_NAMES,
        check=_check_pyannote,
    ),
    "segmentation": PreflightDefinition(
        service="segmentation",
        description="deterministic segment cleanup/extraction",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "segmentation": SEGMENTATION_SERVICE_VERSION},
        check=_check_static,
    ),
    "speaker_profiling": PreflightDefinition(
        service="speaker_profiling",
        description="SpeechBrain ECAPA speaker embedding backend",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "speaker_profiling": SPEAKER_PROFILING_SERVICE_VERSION},
        model_versions={"speechbrain-ecapa": "speechbrain/spkrec-ecapa-voxceleb"},
        check=_check_speechbrain,
    ),
    "assignment": PreflightDefinition(
        service="assignment",
        description="voice-pool-backed pseudo-speaker assignment",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "assignment": ASSIGNMENT_SERVICE_VERSION},
        model_versions={"voice_pool": "configured-manifest-version"},
        required_env_all=("AEGISVOX_VOICE_POOL_MANIFEST", "AEGISVOX_VOICE_POOL_INDEX"),
        check=_check_voice_pool,
    ),
    "voice_conversion": PreflightDefinition(
        service="voice_conversion",
        description="Seed-VC command backend",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "voice_conversion": VOICE_CONVERSION_SERVICE_VERSION},
        model_versions={"seed-vc": "command-adapter-v1"},
        required_env_all=("SEEDVC_COMMAND",),
        optional_env=("SEEDVC_MODEL_DIR",),
        check=_check_seedvc,
    ),
    "stitching": PreflightDefinition(
        service="stitching",
        description="deterministic timeline DSP stitching",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "stitching": STITCHING_SERVICE_VERSION},
        check=_check_static,
    ),
    "asr_alignment": PreflightDefinition(
        service="asr_alignment",
        description="OpenAI ASR and speaker attribution backend",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "asr_alignment": ASR_ALIGNMENT_SERVICE_VERSION},
        model_versions={"openai-asr": os.environ.get("OPENAI_ASR_MODEL") or os.environ.get("AEGISVOX_OPENAI_ASR_MODEL") or "gpt-4o-transcribe"},
        required_env_any=("OPENAI_API_KEY", "OPENAI_API_KEY"),
        optional_env=("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_ASR_MODEL", "AEGISVOX_OPENAI_ASR_MODEL"),
        check=_check_asr,
    ),
    "text_deid": PreflightDefinition(
        service="text_deid",
        description="deterministic text de-identification validators",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "text_deid": TEXT_DEID_SERVICE_VERSION},
        check=_check_static,
    ),
    "speech_editing": PreflightDefinition(
        service="speech_editing",
        description="VoiceCraft command backend for edited-span synthesis",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "speech_editing": SPEECH_EDITING_SERVICE_VERSION},
        model_versions={"speech_editing_backend": "voicecraft:command-adapter-v1"},
        required_env_all=("VOICECRAFT_COMMAND",),
        optional_env=("VOICECRAFT_MODEL_DIR",),
        check=_check_voicecraft,
    ),
    "final_background_remix": PreflightDefinition(
        service="final_background_remix",
        description="mix processed foreground back with preserved background candidate",
        required=False,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "final_background_remix": FINAL_BACKGROUND_REMIX_SERVICE_VERSION},
        check=_check_static,
    ),
    "speaker_privacy_metrics": PreflightDefinition(
        service="speaker_privacy_metrics",
        description="SpeechBrain ECAPA speaker privacy/linkability metrics",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "speaker_privacy_metrics": SPEAKER_PRIVACY_METRICS_SERVICE_VERSION},
        model_versions={"speechbrain-ecapa": "speechbrain/spkrec-ecapa-voxceleb"},
        check=_check_speechbrain,
    ),
    "overlap_remix": PreflightDefinition(
        service="overlap_remix",
        description="overlap separation plus vocal conversion/remix candidate artifact",
        required=False,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "overlap_remix": OVERLAP_REMIX_SERVICE_VERSION},
        model_versions={"overlap_separation": "command-separation:adapter-v1", "seed-vc": "command-adapter-v1"},
        required_env_all=("OVERLAP_SEPARATION_COMMAND", "SEEDVC_COMMAND"),
        optional_env=("AEGISVOX_OVERLAP_REMIX_ALLOW_MULTI_SPEAKER_SKIP",),
        check=_check_overlap_remix,
    ),
    "evaluation": PreflightDefinition(
        service="evaluation",
        description="phase-9 privacy/utility/safety/audit evaluation gates",
        required=True,
        service_versions={"service_job": SERVICE_JOB_ADAPTER_VERSION, "evaluation": EVALUATION_SERVICE_VERSION},
        check=_check_static,
    ),
}


def expected_version_registry() -> dict[str, Any]:
    """Return the expected manifest provenance keys by service."""

    return {
        "aegisvox_version": __version__,
        "service_job_adapter_version": SERVICE_JOB_ADAPTER_VERSION,
        "services": {
            name: {
                "required": definition.required,
                "description": definition.description,
                "service_versions": dict(definition.service_versions),
                "model_versions": dict(definition.model_versions),
            }
            for name, definition in PREFLIGHT_DEFINITIONS.items()
        },
    }


def run_preflight_checks(
    *,
    services: list[str] | None = None,
    artifact_root: str | Path = "/tmp/aegisvox_preflight",
    deep: bool = False,
) -> dict[str, Any]:
    """Run configured runtime checks and return a sanitized report."""

    selected = services or list(PREFLIGHT_DEFINITIONS)
    unknown = [service for service in selected if service not in PREFLIGHT_DEFINITIONS]
    if unknown:
        raise AegisVoxContractError(f"unknown preflight service(s): {', '.join(unknown)}")
    context = PreflightContext(artifact_root=Path(artifact_root), deep=deep)
    service_reports: list[dict[str, Any]] = []
    for service in selected:
        definition = PREFLIGHT_DEFINITIONS[service]
        detail: dict[str, Any]
        try:
            _check_env_requirements(definition)
            detail = definition.check(context, definition) if definition.check else {"ok": True}
            ok = bool(detail.get("ok", True))
            status = "pass" if ok else "fail"
            error = None
        except Exception as exc:  # noqa: BLE001 - preflight report should aggregate all service setup errors
            ok = False
            status = "fail" if definition.required else "warn"
            detail = {}
            error = str(exc)
        service_reports.append(
            sanitize_for_audit(
                {
                    "service": service,
                    "required": definition.required,
                    "status": status,
                    "ok": ok,
                    "description": definition.description,
                    "service_versions": dict(definition.service_versions),
                    "model_versions": dict(definition.model_versions),
                    "env": _env_requirements(definition),
                    "detail": detail,
                    "error": error,
                }
            )
        )
    failed_required = [row["service"] for row in service_reports if row["required"] and row["status"] == "fail"]
    warned_optional = [row["service"] for row in service_reports if not row["required"] and row["status"] in {"fail", "warn"}]
    report = {
        "ok": not failed_required,
        "status": "pass" if not failed_required else "fail",
        "depth": "deep" if deep else "configuration",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "artifact_root": str(Path(artifact_root)),
        },
        "summary": {
            "num_services": len(service_reports),
            "num_failed_required": len(failed_required),
            "num_warned_optional": len(warned_optional),
            "failed_required": failed_required,
            "warned_optional": warned_optional,
        },
        "version_registry": expected_version_registry(),
        "services": service_reports,
    }
    return sanitize_for_audit(report)


def preflight_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown preflight report."""

    lines = [
        "# AegisVox preflight report",
        "",
        f"- Status: **{report.get('status')}**",
        f"- Depth: `{report.get('depth')}`",
    ]
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines.extend(
        [
            f"- Services checked: {summary.get('num_services')}",
            f"- Failed required services: {summary.get('num_failed_required')}",
            f"- Warned optional services: {summary.get('num_warned_optional')}",
            "",
            "## Services",
            "",
            "| Service | Required | Status | Detail |",
            "|---|---:|---|---|",
        ]
    )
    for row in report.get("services") or []:
        if not isinstance(row, dict):
            continue
        detail = row.get("error") or (row.get("detail") if row.get("detail") else row.get("description"))
        detail_text = json.dumps(detail, sort_keys=True) if isinstance(detail, (dict, list)) else str(detail)
        detail_text = detail_text.replace("|", "\\|")[:300]
        lines.append(f"| `{row.get('service')}` | {row.get('required')} | `{row.get('status')}` | {detail_text} |")
    lines.extend(["", "## Claim boundary", "", "A passing preflight proves configuration availability only. It does not prove privacy, utility, clinical safety, or successful end-to-end execution."])
    return "\n".join(lines) + "\n"
