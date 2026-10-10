"""Voice conversion service boundary.

Runtime conversion requires a real backend (initially Seed-VC). Test backends may
be injected in unit tests, but the default service never fabricates converted
audio or silently copies input audio.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore


class VoiceConversionSetupError(RuntimeError):
    """Raised when Seed-VC or another real backend is unavailable."""


class VoiceConversionBackend(Protocol):
    backend_name: str
    backend_version: str

    def preflight(self) -> dict[str, Any]: ...

    def convert(self, *, source_wav: Path, target_wav: Path, output_wav: Path, options: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ConversionSegment:
    segment_id: str
    speaker_id: str
    source_audio_uri: str
    target_reference_audio_uri: str
    converted_audio_uri: str
    duration_ratio: float | None
    conversion_status: str
    failure_reason: str | None = None
    backend: str = "unknown"

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise AegisVoxContractError("segment_id is required")
        validate_artifact_uri(self.source_audio_uri)
        validate_artifact_uri(self.target_reference_audio_uri)
        validate_artifact_uri(self.converted_audio_uri)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "source_audio_uri": self.source_audio_uri,
            "target_reference_audio_uri": self.target_reference_audio_uri,
            "converted_audio_uri": self.converted_audio_uri,
            "duration_ratio": self.duration_ratio,
            "conversion_status": self.conversion_status,
            "failure_reason": self.failure_reason,
            "backend": self.backend,
        }


def _post_json(url: str, payload: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - operator-configured local/internal endpoint
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise VoiceConversionSetupError(f"Seed-VC service request failed: {exc}") from exc
    try:
        decoded = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise VoiceConversionSetupError(f"Seed-VC service returned non-JSON response: {raw[:500]}") from exc
    if not isinstance(decoded, dict):
        raise VoiceConversionSetupError("Seed-VC service response must be a JSON object")
    return decoded


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


@dataclass
class SeedVCCommandBackend:
    """Seed-VC command adapter.

    Configure with either constructor args or environment:

    - `SEEDVC_COMMAND`: command template containing `{source}`, `{target}`, `{output}`.
    - `SEEDVC_MODEL_DIR`: optional model/checkpoint directory that must exist.

    Example command template:

    ```text
    python /opt/Seed-VC/inference.py --source {source} --target {target} --output {output}
    ```
    """

    command_template: str | None = None
    model_dir: str | Path | None = None
    backend_name: str = "seed-vc"
    backend_version: str = "command-adapter-v1"

    def __post_init__(self) -> None:
        if self.command_template is None:
            self.command_template = os.environ.get("SEEDVC_COMMAND")
        if self.model_dir is None:
            self.model_dir = os.environ.get("SEEDVC_MODEL_DIR")

    def preflight(self) -> dict[str, Any]:
        if not self.command_template:
            raise VoiceConversionSetupError(
                "Seed-VC command is not configured. Set SEEDVC_COMMAND to a real conversion command template."
            )
        if "{source}" not in self.command_template or "{target}" not in self.command_template or "{output}" not in self.command_template:
            raise VoiceConversionSetupError("SEEDVC_COMMAND must include {source}, {target}, and {output} placeholders")
        if self.model_dir is not None and not Path(self.model_dir).expanduser().exists():
            raise VoiceConversionSetupError(f"SEEDVC_MODEL_DIR does not exist: {self.model_dir}")
        return {
            "ok": True,
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "model_dir": str(self.model_dir) if self.model_dir else None,
        }

    def convert(self, *, source_wav: Path, target_wav: Path, output_wav: Path, options: dict[str, Any]) -> dict[str, Any]:
        self.preflight()
        if not source_wav.exists():
            raise AegisVoxContractError(f"source segment WAV not found: {source_wav}")
        if not target_wav.exists():
            raise AegisVoxContractError(f"target reference WAV not found: {target_wav}")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        command = self.command_template.format(
            source=shlex.quote(str(source_wav)),
            target=shlex.quote(str(target_wav)),
            output=shlex.quote(str(output_wav)),
            model_dir=shlex.quote(str(self.model_dir)) if self.model_dir else "",
        )
        env = os.environ.copy()
        cuda_visible_devices = options.get("cuda_visible_devices")
        if cuda_visible_devices not in {None, ""}:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=int(options.get("timeout_sec", 600)),
            env=env,
        )
        if completed.returncode != 0:
            raise VoiceConversionSetupError(
                f"Seed-VC command failed with exit {completed.returncode}: {completed.stderr[-1000:]}"
            )
        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise VoiceConversionSetupError("Seed-VC command completed but did not create a non-empty output WAV")
        return {"command_stdout_tail": completed.stdout[-1000:], "command_stderr_tail": completed.stderr[-1000:]}


@dataclass
class SeedVCHTTPBackend:
    """Persistent Seed-VC service client.

    The target service must keep the model loaded and accept JSON requests at
    ``POST /convert`` with source/target/output paths on the shared ORD SDS
    filesystem. This adapter only validates the produced artifact; it does not
    spawn the model process.
    """

    base_url: str | None = None
    backend_name: str = "seed-vc-http"
    backend_version: str = "persistent-http-v1"

    def __post_init__(self) -> None:
        if self.base_url is None:
            self.base_url = os.environ.get("AEGISVOX_SEEDVC_SERVICE_URL") or os.environ.get("SEEDVC_SERVICE_URL")

    def preflight(self) -> dict[str, Any]:
        if not self.base_url:
            raise VoiceConversionSetupError("Seed-VC persistent service is not configured. Set AEGISVOX_SEEDVC_SERVICE_URL.")
        try:
            with urllib.request.urlopen(_join_url(self.base_url, "/health"), timeout=10) as response:  # noqa: S310
                health_raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise VoiceConversionSetupError(f"Seed-VC persistent service health check failed: {exc}") from exc
        return {"ok": True, "backend": self.backend_name, "backend_version": self.backend_version, "base_url": self.base_url, "health": health_raw[:500]}

    def convert(self, *, source_wav: Path, target_wav: Path, output_wav: Path, options: dict[str, Any]) -> dict[str, Any]:
        self.preflight()
        if not source_wav.exists():
            raise AegisVoxContractError(f"source segment WAV not found: {source_wav}")
        if not target_wav.exists():
            raise AegisVoxContractError(f"target reference WAV not found: {target_wav}")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(source_wav),
            "target": str(target_wav),
            "output": str(output_wav),
            "options": dict(options),
        }
        response = _post_json(_join_url(str(self.base_url), "/convert"), payload, timeout_sec=int(options.get("timeout_sec", 600)))
        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise VoiceConversionSetupError("Seed-VC persistent service completed but did not create a non-empty output WAV")
        return {"service_response": response}


@dataclass
class IdentityVoiceConversionBackend:
    """Explicit no-speaker-conversion backend for lexical-only baselines.

    This backend deliberately copies the source segment to the converted segment
    path. It is not a fallback and is not available by accident: callers must
    opt in with ``AEGISVOX_VOICE_CONVERSION_BACKEND=identity`` and
    ``AEGISVOX_ALLOW_IDENTITY_VOICE_CONVERSION_BASELINE=true``. This gives B3 a
    real, auditable pipeline branch while preserving the project rule that the
    default conversion path never silently fabricates model output.
    """

    backend_name: str = "identity-no-speaker-conversion"
    backend_version: str = "baseline-v1"

    def preflight(self) -> dict[str, Any]:
        allowed = os.environ.get("AEGISVOX_ALLOW_IDENTITY_VOICE_CONVERSION_BASELINE", "").strip().lower()
        if allowed not in {"1", "true", "yes"}:
            raise VoiceConversionSetupError(
                "Identity voice conversion is baseline-only. Set "
                "AEGISVOX_ALLOW_IDENTITY_VOICE_CONVERSION_BASELINE=true with "
                "AEGISVOX_VOICE_CONVERSION_BACKEND=identity."
            )
        return {
            "ok": True,
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "claim_boundary": "B3 lexical-only/no-speaker-conversion baseline; not privacy-preserving by speaker identity.",
        }

    def convert(self, *, source_wav: Path, target_wav: Path, output_wav: Path, options: dict[str, Any]) -> dict[str, Any]:
        self.preflight()
        if not source_wav.exists():
            raise AegisVoxContractError(f"source segment WAV not found: {source_wav}")
        # target_wav is intentionally ignored for this baseline, but the
        # assignment artifact is still produced upstream for paired metric
        # compatibility and auditability.
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_wav, output_wav)
        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise VoiceConversionSetupError("identity baseline failed to create a non-empty output WAV")
        return {
            "identity_baseline": True,
            "target_reference_ignored": str(target_wav),
            "claim_boundary": "No speaker conversion was performed.",
        }


def _duration_ratio(source_path: Path, output_path: Path) -> float | None:
    try:
        from .segmentation import wav_duration_sec

        source = wav_duration_sec(source_path)
        output = wav_duration_sec(output_path)
        if source <= 0:
            return None
        return round(output / source, 4)
    except Exception:
        return None


@dataclass
class VoiceConversionService:
    artifact_store: LocalArtifactStore
    backend: VoiceConversionBackend = field(default_factory=SeedVCCommandBackend)

    def preflight(self) -> dict[str, Any]:
        return self.backend.preflight()

    def convert_segment_to_artifact(
        self,
        *,
        segment_id: str,
        speaker_id: str,
        source_audio_uri: str,
        target_reference_audio_uri: str,
        converted_audio_uri: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validate_artifact_uri(source_audio_uri)
        validate_artifact_uri(target_reference_audio_uri)
        validate_artifact_uri(converted_audio_uri)
        options = options or {}
        source_path = self.artifact_store.local_path(source_audio_uri)
        target_path = self.artifact_store.local_path(target_reference_audio_uri)
        output_path = self.artifact_store.local_path(converted_audio_uri)
        metadata = self.backend.convert(source_wav=source_path, target_wav=target_path, output_wav=output_path, options=options)
        record = ConversionSegment(
            segment_id=segment_id,
            speaker_id=speaker_id,
            source_audio_uri=source_audio_uri,
            target_reference_audio_uri=target_reference_audio_uri,
            converted_audio_uri=converted_audio_uri,
            duration_ratio=_duration_ratio(source_path, output_path),
            conversion_status="converted",
            backend=self.backend.backend_name,
        )
        payload = record.to_dict()
        payload["backend_metadata"] = metadata
        return payload


def segments_converted_payload(segments: list[dict[str, Any]], *, backend: VoiceConversionBackend) -> dict[str, Any]:
    return {
        "backend": backend.backend_name,
        "backend_version": backend.backend_version,
        "num_segments": len(segments),
        "segments": segments,
    }
