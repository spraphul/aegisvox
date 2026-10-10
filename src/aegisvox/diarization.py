"""pyannote-backed diarization service boundary.

This module does not provide mock diarization. It either runs a real pyannote
pipeline or fails loudly with an actionable setup error.
"""

from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .audit import redact_text
from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore

DEFAULT_PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-3.1"
TOKEN_ENV_NAMES = ("PYANNOTE_AUTH_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN")
DEFAULT_COMMAND_DIARIZATION_MODEL_ID = "external-diarization-command"


class DiarizationSetupError(RuntimeError):
    """Raised when pyannote/model/token setup is incomplete."""


class _PyannotePipelineLike(Protocol):
    def __call__(self, audio: str | dict[str, Any], **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class PyannoteDiarizationConfig:
    model_id: str = DEFAULT_PYANNOTE_MODEL_ID
    auth_token_env: tuple[str, ...] = TOKEN_ENV_NAMES
    device: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    num_speakers: int | None = None

    def resolve_token(self) -> str:
        for name in self.auth_token_env:
            value = os.environ.get(name)
            if value:
                return value
        names = ", ".join(self.auth_token_env)
        raise DiarizationSetupError(
            f"pyannote diarization requires a Hugging Face token in one of: {names}. "
            "Inject it at runtime; do not store it in repo files."
        )


@dataclass(frozen=True)
class DiarizationSegment:
    speaker_id: str
    start_sec: float
    end_sec: float
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.speaker_id:
            raise AegisVoxContractError("speaker_id is required")
        if self.start_sec < 0 or self.end_sec <= self.start_sec:
            raise AegisVoxContractError("invalid diarization segment timing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


def _post_json(url: str, payload: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - operator-configured local/internal endpoint
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise DiarizationSetupError(f"diarization persistent service request failed: {exc}") from exc
    try:
        decoded = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise DiarizationSetupError(f"diarization persistent service returned non-JSON response: {raw[:500]}") from exc
    if not isinstance(decoded, dict):
        raise DiarizationSetupError("diarization persistent service response must be a JSON object")
    return decoded


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def import_pyannote_pipeline() -> type:
    try:
        module = importlib.import_module("pyannote.audio")
    except ImportError as exc:
        raise DiarizationSetupError(
            "pyannote.audio is not installed. Install the diarization service dependencies "
            "before running this service."
        ) from exc
    pipeline = getattr(module, "Pipeline", None)
    if pipeline is None:
        raise DiarizationSetupError("pyannote.audio.Pipeline is unavailable; check pyannote.audio installation")
    return pipeline


def load_pyannote_pipeline(config: PyannoteDiarizationConfig) -> _PyannotePipelineLike:
    token = config.resolve_token()
    pipeline_cls = import_pyannote_pipeline()
    try:
        pipeline = pipeline_cls.from_pretrained(config.model_id, use_auth_token=token)
    except TypeError:
        # Some releases accept `token=` instead of `use_auth_token=`.
        pipeline = pipeline_cls.from_pretrained(config.model_id, token=token)
    except Exception as exc:  # pragma: no cover - depends on external model service
        detail = redact_text(f"{type(exc).__name__}: {exc}", max_length=1000)
        raise DiarizationSetupError(
            f"failed to load pyannote model {config.model_id!r}. Confirm token access, model terms, and cache paths. "
            f"Sanitized cause: {detail}"
        ) from exc
    if config.device and hasattr(pipeline, "to"):
        try:
            import torch  # type: ignore

            pipeline.to(torch.device(config.device))
        except Exception as exc:  # pragma: no cover - optional runtime path
            raise DiarizationSetupError(f"failed to move pyannote pipeline to device {config.device!r}") from exc
    return pipeline


def _annotation_to_segments(annotation: Any) -> list[DiarizationSegment]:
    if not hasattr(annotation, "itertracks"):
        raise AegisVoxContractError("pyannote pipeline output does not expose itertracks(yield_label=True)")
    segments: list[DiarizationSegment] = []
    for turn, track, speaker in annotation.itertracks(yield_label=True):
        start = float(getattr(turn, "start"))
        end = float(getattr(turn, "end"))
        segments.append(
            DiarizationSegment(
                speaker_id=str(speaker),
                start_sec=start,
                end_sec=end,
                metadata={"track": str(track)},
            )
        )
    segments.sort(key=lambda row: (row.start_sec, row.end_sec, row.speaker_id))
    return segments


def diarization_payload(
    *,
    source_audio_uri: str,
    model_id: str,
    segments: list[DiarizationSegment],
) -> dict[str, Any]:
    validate_artifact_uri(source_audio_uri)
    return {
        "source_audio_uri": source_audio_uri,
        "backend": "pyannote",
        "model_id": model_id,
        "num_segments": len(segments),
        "segments": [segment.to_dict() for segment in segments],
    }


def _normalize_external_diarization_payload(
    raw: dict[str, Any],
    *,
    source_audio_uri: str,
    backend: str,
    model_id: str,
) -> dict[str, Any]:
    rows = raw.get("segments") or raw.get("turns")
    if not isinstance(rows, list):
        raise AegisVoxContractError("external diarization payload must contain a segments or turns list")
    segments: list[DiarizationSegment] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AegisVoxContractError(f"external diarization segment {index} is not an object")
        speaker_id = row.get("speaker_id") or row.get("speaker") or row.get("label")
        start = row.get("start_sec", row.get("start", row.get("begin")))
        end = row.get("end_sec", row.get("end", row.get("stop")))
        if speaker_id is None or start is None or end is None:
            raise AegisVoxContractError(f"external diarization segment {index} missing speaker/start/end")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        segments.append(
            DiarizationSegment(
                speaker_id=str(speaker_id),
                start_sec=float(start),
                end_sec=float(end),
                confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
                metadata=dict(metadata),
            )
        )
    payload = diarization_payload(source_audio_uri=source_audio_uri, model_id=str(raw.get("model_id") or model_id), segments=segments)
    payload["backend"] = str(raw.get("backend") or backend)
    if raw.get("metadata") and isinstance(raw["metadata"], dict):
        payload["metadata"] = dict(raw["metadata"])
    return payload


@dataclass(frozen=True)
class CommandDiarizationConfig:
    """Configuration for an external real diarization backend command.

    This is intentionally generic so ORD can run NVIDIA NeMo Sortformer, NeMo
    MSDD/clustering, or another backend without adding fake in-process
    fallbacks. The command must either write the requested output JSON artifact
    or print a JSON object containing `segments`/`turns` to stdout.
    """

    command: str | None = None
    command_env: str = "AEGISVOX_DIARIZATION_COMMAND"
    backend: str = "command"
    model_id: str = DEFAULT_COMMAND_DIARIZATION_MODEL_ID
    device: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    num_speakers: int | None = None
    timeout_seconds: int | None = None

    def resolve_command(self) -> str:
        command = self.command or os.environ.get(self.command_env)
        if not command:
            raise DiarizationSetupError(
                f"{self.backend} diarization requires a real command via {self.command_env} or explicit config.command. "
                "No mock diarization fallback is allowed."
            )
        return command


@dataclass
class CommandDiarizationService:
    artifact_store: LocalArtifactStore
    config: CommandDiarizationConfig = field(default_factory=CommandDiarizationConfig)

    def preflight(self) -> dict[str, Any]:
        command = self.config.resolve_command()
        return {
            "ok": True,
            "backend": self.config.backend,
            "model_id": self.config.model_id,
            "command_present": bool(command),
            "device": self.config.device,
        }

    def _command_env(self, source_audio_uri: str, output_diarization_uri: str) -> dict[str, str]:
        source_path = self.artifact_store.local_path(source_audio_uri)
        output_path = self.artifact_store.local_path(output_diarization_uri)
        env = os.environ.copy()
        env.update(
            {
                "AEGISVOX_ARTIFACT_ROOT": str(self.artifact_store.root),
                "AEGISVOX_DIARIZATION_SOURCE_URI": source_audio_uri,
                "AEGISVOX_DIARIZATION_SOURCE_PATH": str(source_path),
                "AEGISVOX_DIARIZATION_OUTPUT_URI": output_diarization_uri,
                "AEGISVOX_DIARIZATION_OUTPUT_PATH": str(output_path),
                "AEGISVOX_DIARIZATION_BACKEND_NAME": self.config.backend,
                "AEGISVOX_DIARIZATION_MODEL_ID": self.config.model_id,
            }
        )
        if self.config.device is not None:
            env["AEGISVOX_DIARIZATION_DEVICE"] = str(self.config.device)
        for name in ("num_speakers", "min_speakers", "max_speakers"):
            value = getattr(self.config, name)
            if value is not None:
                env[f"AEGISVOX_DIARIZATION_{name.upper()}"] = str(value)
        return env

    @staticmethod
    def _last_stdout_json(stdout: str) -> dict[str, Any]:
        for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for offset, char in enumerate(stdout):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(stdout[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                candidates.append(payload)
        if not candidates:
            raise AegisVoxContractError("external diarization command did not write output artifact or print JSON to stdout")
        return candidates[-1]

    def diarize_to_artifact(self, source_audio_uri: str, output_diarization_uri: str) -> dict[str, Any]:
        validate_artifact_uri(source_audio_uri)
        validate_artifact_uri(output_diarization_uri)
        source_path = self.artifact_store.local_path(source_audio_uri)
        if not source_path.exists():
            raise AegisVoxContractError(f"source audio artifact not found: {source_path}")
        output_path = self.artifact_store.local_path(output_diarization_uri)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = self.config.resolve_command()
        completed = subprocess.run(
            shlex.split(command),
            env=self._command_env(source_audio_uri, output_diarization_uri),
            cwd=str(Path.cwd()),
            text=True,
            capture_output=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            detail = redact_text((completed.stderr or completed.stdout or "").strip(), max_length=1200)
            raise DiarizationSetupError(
                f"{self.config.backend} diarization command failed with exit code {completed.returncode}. "
                f"Sanitized output: {detail}"
            )
        if output_path.exists():
            raw = json.loads(output_path.read_text())
        else:
            raw = self._last_stdout_json(completed.stdout or "")
        if not isinstance(raw, dict):
            raise AegisVoxContractError("external diarization output must be a JSON object")
        payload = _normalize_external_diarization_payload(
            raw,
            source_audio_uri=source_audio_uri,
            backend=self.config.backend,
            model_id=self.config.model_id,
        )
        self.artifact_store.write_json(output_diarization_uri, payload)
        return payload


@dataclass
class PyannoteDiarizationService:
    artifact_store: LocalArtifactStore
    config: PyannoteDiarizationConfig = field(default_factory=PyannoteDiarizationConfig)
    pipeline: _PyannotePipelineLike | None = None

    def preflight(self) -> dict[str, Any]:
        token = self.config.resolve_token()
        if self.pipeline is None:
            import_pyannote_pipeline()
        return {
            "ok": True,
            "backend": "pyannote",
            "model_id": self.config.model_id,
            "token_present": bool(token),
            "device": self.config.device,
        }

    def _get_pipeline(self) -> _PyannotePipelineLike:
        if self.pipeline is not None:
            return self.pipeline
        self.pipeline = load_pyannote_pipeline(self.config)
        return self.pipeline

    def diarize_to_artifact(
        self,
        source_audio_uri: str,
        output_diarization_uri: str,
    ) -> dict[str, Any]:
        validate_artifact_uri(source_audio_uri)
        validate_artifact_uri(output_diarization_uri)
        audio_path = self.artifact_store.local_path(source_audio_uri)
        if not audio_path.exists():
            raise AegisVoxContractError(f"source audio artifact not found: {audio_path}")
        kwargs: dict[str, Any] = {}
        for name in ("num_speakers", "min_speakers", "max_speakers"):
            value = getattr(self.config, name)
            if value is not None:
                kwargs[name] = value
        annotation = self._get_pipeline()(str(audio_path), **kwargs)
        segments = _annotation_to_segments(annotation)
        payload = diarization_payload(source_audio_uri=source_audio_uri, model_id=self.config.model_id, segments=segments)
        self.artifact_store.write_json(output_diarization_uri, payload)
        return payload


@dataclass
class PyannoteHTTPDiarizationService:
    """Persistent pyannote diarization service client.

    The service keeps the pyannote pipeline loaded and accepts shared-SDS paths
    at ``POST /diarize``. This adapter validates/writes the canonical artifact;
    it never spawns pyannote in the job process.
    """

    artifact_store: LocalArtifactStore
    base_url: str | None = None
    model_id: str = "pyannote-http"
    timeout_sec: int = 1800

    @property
    def backend_name(self) -> str:
        return "pyannote-http"

    @property
    def backend_version(self) -> str:
        return f"persistent-http-v1;endpoint={self.base_url or 'unset'}"

    def __post_init__(self) -> None:
        if self.base_url is None:
            self.base_url = os.environ.get("AEGISVOX_DIARIZATION_SERVICE_URL") or os.environ.get("PYANNOTE_SERVICE_URL")

    def preflight(self) -> dict[str, Any]:
        if not self.base_url:
            raise DiarizationSetupError("pyannote persistent service is not configured. Set AEGISVOX_DIARIZATION_SERVICE_URL.")
        try:
            with urllib.request.urlopen(_join_url(self.base_url, "/health"), timeout=10) as response:  # noqa: S310
                health_raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise DiarizationSetupError(f"pyannote persistent service health check failed: {exc}") from exc
        return {"ok": True, "backend": self.backend_name, "backend_version": self.backend_version, "base_url": self.base_url, "health": health_raw[:500]}

    def diarize_to_artifact(self, source_audio_uri: str, output_diarization_uri: str) -> dict[str, Any]:
        validate_artifact_uri(source_audio_uri)
        validate_artifact_uri(output_diarization_uri)
        audio_path = self.artifact_store.local_path(source_audio_uri)
        if not audio_path.exists():
            raise AegisVoxContractError(f"source audio artifact not found: {audio_path}")
        self.preflight()
        request = {
            "source": str(audio_path),
            "source_audio_uri": source_audio_uri,
            "num_speakers": int(os.environ["AEGISVOX_PYANNOTE_NUM_SPEAKERS"]) if os.environ.get("AEGISVOX_PYANNOTE_NUM_SPEAKERS") else None,
            "min_speakers": int(os.environ["AEGISVOX_PYANNOTE_MIN_SPEAKERS"]) if os.environ.get("AEGISVOX_PYANNOTE_MIN_SPEAKERS") else None,
            "max_speakers": int(os.environ["AEGISVOX_PYANNOTE_MAX_SPEAKERS"]) if os.environ.get("AEGISVOX_PYANNOTE_MAX_SPEAKERS") else None,
        }
        raw = _post_json(_join_url(str(self.base_url), "/diarize"), request, timeout_sec=self.timeout_sec)
        payload = _normalize_external_diarization_payload(
            raw,
            source_audio_uri=source_audio_uri,
            backend=str(raw.get("backend") or self.backend_name),
            model_id=str(raw.get("model_id") or self.model_id),
        )
        self.artifact_store.write_json(output_diarization_uri, payload)
        return payload


def write_diarization_payload(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
