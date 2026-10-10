"""Target-speaker extraction service boundary for overlapped speech.

This module defines the AegisVox contract for speaker-conditioned overlap
repair. It supports ClearerVoice-Studio and WeSep/WebSep as first-class backend
families behind one service API. Runtime deployments should keep the chosen
models resident inside a singleton service process; command templates are strict
adapters for installation smoke tests or wrappers around model-resident servers.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore


class TargetSpeakerExtractionError(RuntimeError):
    """Raised when target-speaker extraction cannot produce usable stems."""


class TargetSpeakerExtractionBackend(Protocol):
    backend_name: str
    backend_version: str

    def preflight(self) -> dict[str, Any]: ...

    def extract(
        self,
        *,
        mixture_wav: Path,
        enrollment_wav: Path,
        output_wav: Path,
        speaker_id: str,
        options: dict[str, Any],
    ) -> dict[str, Any]: ...


def _post_json(url: str, payload: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - operator-configured local endpoint
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise TargetSpeakerExtractionError(f"TSE service request failed: {exc}") from exc
    try:
        decoded = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise TargetSpeakerExtractionError(f"TSE service returned non-JSON response: {raw[:500]}") from exc
    if not isinstance(decoded, dict):
        raise TargetSpeakerExtractionError("TSE service response must be a JSON object")
    return decoded


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


@dataclass
class CommandTargetSpeakerExtractionBackend:
    """Strict command adapter for a target-speaker extraction backend.

    The command must include `{mixture}`, `{enrollment}`, and `{output}`. It may
    also use `{speaker_id}` and `{output_dir}`. The command must create a
    non-empty WAV at `{output}`.
    """

    command_template: str | None
    backend_name: str
    backend_version: str

    def preflight(self) -> dict[str, Any]:
        if not self.command_template:
            raise TargetSpeakerExtractionError(f"{self.backend_name} TSE command is not configured")
        required = ["{mixture}", "{enrollment}", "{output}"]
        missing = [token for token in required if token not in self.command_template]
        if missing:
            raise TargetSpeakerExtractionError(f"{self.backend_name} TSE command missing placeholders: {missing}")
        return {"ok": True, "backend": self.backend_name, "backend_version": self.backend_version}

    def extract_many(
        self,
        *,
        mixture_wav: Path,
        enrollment_wavs: dict[str, Path],
        output_wavs: dict[str, Path],
        options: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Extract stems for multiple speakers with a command backend.

        Command adapters such as WeSep/WebSep wrap an operator-supplied CLI or
        checkpoint-specific script; they do not have a resident Python model or
        assignment helper. The joint/resident implementation lives in
        ClearerVoiceStudioSeparateAssignBackend. For command backends, the safe
        behavior is to invoke the validated single-speaker command once per
        enrollment and return per-speaker metadata.
        """

        if not mixture_wav.exists() or mixture_wav.stat().st_size <= 44:
            raise AegisVoxContractError(f"mixture WAV missing or empty: {mixture_wav}")
        if not enrollment_wavs:
            raise AegisVoxContractError("at least one enrollment WAV is required")
        metadata_by_speaker: dict[str, dict[str, Any]] = {}
        for speaker_id, enrollment_wav in sorted(enrollment_wavs.items()):
            output_wav = output_wavs[speaker_id]
            metadata_by_speaker[speaker_id] = self.extract(
                mixture_wav=mixture_wav,
                enrollment_wav=enrollment_wav,
                output_wav=output_wav,
                speaker_id=speaker_id,
                options=options,
            )
        return metadata_by_speaker

    def extract(
        self,
        *,
        mixture_wav: Path,
        enrollment_wav: Path,
        output_wav: Path,
        speaker_id: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        self.preflight()
        if not mixture_wav.exists() or mixture_wav.stat().st_size == 0:
            raise AegisVoxContractError(f"mixture WAV missing or empty: {mixture_wav}")
        if not enrollment_wav.exists() or enrollment_wav.stat().st_size == 0:
            raise AegisVoxContractError(f"enrollment WAV missing or empty: {enrollment_wav}")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        command = self.command_template.format(
            mixture=shlex.quote(str(mixture_wav)),
            enrollment=shlex.quote(str(enrollment_wav)),
            output=shlex.quote(str(output_wav)),
            speaker_id=shlex.quote(str(speaker_id)),
            output_dir=shlex.quote(str(output_wav.parent)),
        )
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=int(options.get("timeout_sec", 900)),
            env=os.environ.copy(),
        )
        if completed.returncode != 0:
            raise TargetSpeakerExtractionError(
                f"{self.backend_name} TSE command failed with exit {completed.returncode}: {completed.stderr[-1000:]}"
            )
        if not output_wav.exists() or output_wav.stat().st_size <= 44:
            raise TargetSpeakerExtractionError(f"{self.backend_name} TSE command did not create non-empty output WAV")
        return {
            "command_stdout_tail": completed.stdout[-1000:],
            "command_stderr_tail": completed.stderr[-1000:],
        }


class ClearerVoiceTargetSpeakerExtractionBackend(CommandTargetSpeakerExtractionBackend):
    def __init__(self, command_template: str | None = None):
        super().__init__(
            command_template=command_template
            or os.environ.get("AEGISVOX_CLEARERVOICE_TSE_COMMAND")
            or os.environ.get("CLEARERVOICE_TSE_COMMAND"),
            backend_name="clearervoice-tse",
            backend_version="clearervoice-studio-command-adapter-v1",
        )


class WeSepTargetSpeakerExtractionBackend(CommandTargetSpeakerExtractionBackend):
    def __init__(self, command_template: str | None = None):
        super().__init__(
            command_template=command_template
            or os.environ.get("AEGISVOX_WESEP_TSE_COMMAND")
            or os.environ.get("WESEP_TSE_COMMAND")
            or os.environ.get("WEBSEP_TSE_COMMAND"),
            backend_name="wesep-tse",
            backend_version="wesep-command-adapter-v1",
        )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return -1.0
    return dot / (na * nb)


def _collect_wav_stems(work_dir: Path, mixture_wav: Path) -> list[Path]:
    stems: list[Path] = []
    for path in sorted(work_dir.rglob("*.wav")):
        if path.resolve() == mixture_wav.resolve():
            continue
        if path.stat().st_size > 44:
            stems.append(path)
    return stems


def _best_unique_assignment(score_matrix: dict[str, dict[str, float]]) -> tuple[dict[str, str], float]:
    """Return max-score one-to-one speaker->stem assignment.

    The number of overlap speakers is small, so brute force is clearer and avoids
    adding scipy as a runtime dependency. Scores are speaker embedding cosine
    similarities against anonymous separated stems.
    """

    import itertools

    speakers = sorted(score_matrix)
    if not speakers:
        return {}, 0.0
    stem_ids = sorted({stem_id for row in score_matrix.values() for stem_id in row})
    if len(stem_ids) < len(speakers):
        raise TargetSpeakerExtractionError(
            f"not enough separated stems for unique assignment: speakers={len(speakers)} stems={len(stem_ids)}"
        )
    best_assignment: dict[str, str] | None = None
    best_score = -float("inf")
    for candidate_stems in itertools.permutations(stem_ids, len(speakers)):
        total = 0.0
        valid = True
        for speaker_id, stem_id in zip(speakers, candidate_stems):
            if stem_id not in score_matrix[speaker_id]:
                valid = False
                break
            total += float(score_matrix[speaker_id][stem_id])
        if valid and total > best_score:
            best_score = total
            best_assignment = dict(zip(speakers, candidate_stems))
    if best_assignment is None:
        raise TargetSpeakerExtractionError("could not compute unique speaker/stem assignment")
    return best_assignment, best_score


@dataclass
class ClearerVoiceStudioSeparateAssignBackend:
    """Resident ClearerVoice-Studio separation + SpeechBrain assignment backend.

    ClearerVoice-Studio's packaged audio-only pretrained path exposes speech
    separation. We keep the ClearVoice model resident inside the TSE singleton,
    separate the overlap mixture, then assign the best separated stem to each
    enrollment using the persistent SpeechBrain speaker-embedding service.
    """

    clearervoice_root: str | None = None
    speechbrain_url: str | None = None
    backend_name: str = "clearervoice-studio-resident-separate-assign"
    backend_version: str = "resident-clearvoice-mossformer2-ss16k-v1"

    def __post_init__(self) -> None:
        if self.clearervoice_root is None:
            self.clearervoice_root = os.environ.get("CLEARERVOICE_STUDIO_ROOT") or "./third_party/ClearerVoice-Studio"
        if self.speechbrain_url is None:
            self.speechbrain_url = os.environ.get("AEGISVOX_SPEECHBRAIN_SERVICE_URL") or os.environ.get("SPEECHBRAIN_SERVICE_URL")
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        clearvoice_pkg = Path(str(self.clearervoice_root)).expanduser().resolve() / "clearvoice"
        if not clearvoice_pkg.exists():
            raise TargetSpeakerExtractionError(f"ClearerVoice-Studio clearvoice package missing: {clearvoice_pkg}")
        import sys

        if str(clearvoice_pkg) not in sys.path:
            sys.path.insert(0, str(clearvoice_pkg))
        try:
            from clearvoice import ClearVoice  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise TargetSpeakerExtractionError(f"failed to import ClearerVoice-Studio ClearVoice: {exc}") from exc
        if os.environ.get("AEGISVOX_CLEARERVOICE_FORCE_VISIBLE_DEVICE_ZERO", "1").strip().lower() in {"1", "true", "yes"}:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if visible and "," not in visible:
                try:
                    from clearvoice.networks import SpeechModel  # type: ignore

                    SpeechModel.get_free_gpu = lambda self: 0  # noqa: ARG005
                except Exception as exc:  # noqa: BLE001
                    raise TargetSpeakerExtractionError(f"failed to patch ClearerVoice GPU selection: {exc}") from exc
        self._model = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
        return self._model

    def _embed(self, wav_path: Path) -> list[float]:
        if not self.speechbrain_url:
            raise TargetSpeakerExtractionError("SpeechBrain singleton URL is required for ClearerVoice stem assignment")
        response = _post_json(
            _join_url(str(self.speechbrain_url), "/embed"),
            {"audio": str(wav_path)},
            timeout_sec=120,
        )
        embedding = response.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise TargetSpeakerExtractionError(f"SpeechBrain singleton returned invalid embedding for {wav_path}")
        return [float(x) for x in embedding]

    def preflight(self) -> dict[str, Any]:
        self._load_model()
        if not self.speechbrain_url:
            raise TargetSpeakerExtractionError("AEGISVOX_SPEECHBRAIN_SERVICE_URL is not configured")
        return {
            "ok": True,
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "clearervoice_root": str(self.clearervoice_root),
            "speechbrain_url": self.speechbrain_url,
        }

    def extract(
        self,
        *,
        mixture_wav: Path,
        enrollment_wav: Path,
        output_wav: Path,
        speaker_id: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        model = self._load_model()
        if not mixture_wav.exists() or mixture_wav.stat().st_size <= 44:
            raise AegisVoxContractError(f"mixture WAV missing or empty: {mixture_wav}")
        if not enrollment_wav.exists() or enrollment_wav.stat().st_size <= 44:
            raise AegisVoxContractError(f"enrollment WAV missing or empty: {enrollment_wav}")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=options.get("tmp_dir")) as tmp:
            work_dir = Path(tmp)
            stem_base = work_dir / "clearervoice_stem.wav"
            separated = model(input_path=str(mixture_wav), online_write=False)
            model.write(separated, output_path=str(stem_base))
            stems = _collect_wav_stems(work_dir, mixture_wav)
            if not stems:
                raise TargetSpeakerExtractionError("ClearerVoice-Studio did not produce separated WAV stems")
            enrollment_embedding = self._embed(enrollment_wav)
            scored = []
            for stem in stems:
                scored.append((stem, _cosine(enrollment_embedding, self._embed(stem))))
            scored.sort(key=lambda item: item[1], reverse=True)
            best_stem, best_score = scored[0]
            second_score = scored[1][1] if len(scored) > 1 else None
            margin = best_score - second_score if second_score is not None else None
            shutil.copy2(best_stem, output_wav)
        return {
            "selected_backend": self.backend_name,
            "selected_backend_version": self.backend_version,
            "selected_cosine": best_score,
            "second_best_cosine": second_score,
            "selected_margin": margin,
            "selected_stem_name": best_stem.name,
            "selected_stem_path": str(best_stem),
            "candidate_scores": [
                {"stem_name": stem.name, "stem_path": str(stem), "cosine": score}
                for stem, score in scored
            ],
            "num_candidate_stems": len(scored),
        }


@dataclass
class FallbackTargetSpeakerExtractionBackend:
    """Try ClearerVoice first, then WeSep/WebSep for each speaker stem."""

    backends: list[TargetSpeakerExtractionBackend] = field(default_factory=list)
    backend_name: str = "clearervoice-wesep-fallback-tse"
    backend_version: str = "fallback-adapter-v1"

    def __post_init__(self) -> None:
        if not self.backends:
            order = [part.strip().lower() for part in os.environ.get("AEGISVOX_TSE_BACKENDS", "clearervoice,wesep").split(",")]
            for name in order:
                if name in {"clearervoice", "clearervoice-studio", "cv"}:
                    if os.environ.get("AEGISVOX_CLEARERVOICE_RESIDENT", "").strip().lower() in {"1", "true", "yes"}:
                        self.backends.append(ClearerVoiceStudioSeparateAssignBackend())
                    else:
                        self.backends.append(ClearerVoiceTargetSpeakerExtractionBackend())
                elif name in {"wesep", "websep"}:
                    self.backends.append(WeSepTargetSpeakerExtractionBackend())

    def preflight(self) -> dict[str, Any]:
        available = []
        errors = []
        for backend in self.backends:
            try:
                available.append(backend.preflight())
            except Exception as exc:  # noqa: BLE001 - collect backend availability
                errors.append({"backend": getattr(backend, "backend_name", "unknown"), "error": str(exc)})
        if not available:
            raise TargetSpeakerExtractionError(f"no configured TSE backends are available: {errors}")
        return {"ok": True, "backend": self.backend_name, "available_backends": available, "unavailable_backends": errors}

    def extract_many(
        self,
        *,
        mixture_wav: Path,
        enrollment_wavs: dict[str, Path],
        output_wavs: dict[str, Path],
        options: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        errors = []
        for backend in self.backends:
            try:
                many = getattr(backend, "extract_many", None)
                if callable(many):
                    metadata = many(
                        mixture_wav=mixture_wav,
                        enrollment_wavs=enrollment_wavs,
                        output_wavs=output_wavs,
                        options=options,
                    )
                    return {
                        speaker_id: {
                            "selected_backend": backend.backend_name,
                            "selected_backend_version": backend.backend_version,
                            **row,
                        }
                        for speaker_id, row in metadata.items()
                    }
                out: dict[str, dict[str, Any]] = {}
                for speaker_id, enrollment_wav in sorted(enrollment_wavs.items()):
                    out[speaker_id] = backend.extract(
                        mixture_wav=mixture_wav,
                        enrollment_wav=enrollment_wav,
                        output_wav=output_wavs[speaker_id],
                        speaker_id=speaker_id,
                        options=options,
                    )
                return {
                    speaker_id: {
                        "selected_backend": backend.backend_name,
                        "selected_backend_version": backend.backend_version,
                        **row,
                    }
                    for speaker_id, row in out.items()
                }
            except Exception as exc:  # noqa: BLE001 - fallback to next backend
                errors.append({"backend": getattr(backend, "backend_name", "unknown"), "error": str(exc)})
        raise TargetSpeakerExtractionError(f"all TSE backends failed for joint extraction: {errors}")

    def extract(
        self,
        *,
        mixture_wav: Path,
        enrollment_wav: Path,
        output_wav: Path,
        speaker_id: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        errors = []
        for backend in self.backends:
            try:
                metadata = backend.extract(
                    mixture_wav=mixture_wav,
                    enrollment_wav=enrollment_wav,
                    output_wav=output_wav,
                    speaker_id=speaker_id,
                    options=options,
                )
                return {"selected_backend": backend.backend_name, "selected_backend_version": backend.backend_version, **metadata}
            except Exception as exc:  # noqa: BLE001 - fallback to next backend
                errors.append({"backend": getattr(backend, "backend_name", "unknown"), "error": str(exc)})
        raise TargetSpeakerExtractionError(f"all TSE backends failed for {speaker_id}: {errors}")


@dataclass
class TargetSpeakerExtractionHTTPBackend:
    """Client for a persistent AegisVox TSE singleton service."""

    base_url: str | None = None
    backend_name: str = "tse-http"
    backend_version: str = "persistent-http-v1"

    def __post_init__(self) -> None:
        if self.base_url is None:
            self.base_url = os.environ.get("AEGISVOX_TSE_SERVICE_URL") or os.environ.get("TSE_SERVICE_URL")

    def preflight(self) -> dict[str, Any]:
        if not self.base_url:
            raise TargetSpeakerExtractionError("TSE persistent service is not configured. Set AEGISVOX_TSE_SERVICE_URL.")
        try:
            with urllib.request.urlopen(_join_url(self.base_url, "/health"), timeout=10) as response:  # noqa: S310
                health_raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise TargetSpeakerExtractionError(f"TSE persistent service health check failed: {exc}") from exc
        return {"ok": True, "backend": self.backend_name, "backend_version": self.backend_version, "base_url": self.base_url, "health": health_raw[:500]}

    def extract_to_paths(
        self,
        *,
        mixture_wav: Path,
        enrollment_wavs: dict[str, Path],
        output_dir: Path,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.preflight()
        payload = {
            "mixture": str(mixture_wav),
            "enrollments": {speaker_id: str(path) for speaker_id, path in enrollment_wavs.items()},
            "output_dir": str(output_dir),
            "options": dict(options or {}),
        }
        response = _post_json(_join_url(str(self.base_url), "/extract"), payload, timeout_sec=int((options or {}).get("timeout_sec", 1800)))
        stems = response.get("stems") if isinstance(response.get("stems"), dict) else {}
        for speaker_id, row in stems.items():
            path = Path(str(row.get("path") if isinstance(row, dict) else row))
            if not path.exists() or path.stat().st_size <= 44:
                raise TargetSpeakerExtractionError(f"TSE service did not create usable stem for {speaker_id}: {path}")
        return response


@dataclass
class TargetSpeakerExtractionService:
    artifact_store: LocalArtifactStore
    backend: TargetSpeakerExtractionBackend = field(default_factory=FallbackTargetSpeakerExtractionBackend)

    def extract_to_artifact(
        self,
        *,
        mixture_audio_uri: str,
        enrollment_audio_uris: dict[str, str],
        output_prefix_uri: str,
        output_report_uri: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validate_artifact_uri(mixture_audio_uri)
        validate_artifact_uri(output_prefix_uri.rstrip("/") + "/")
        validate_artifact_uri(output_report_uri)
        if not enrollment_audio_uris:
            raise AegisVoxContractError("TSE requires at least one speaker enrollment")
        options = dict(options or {})
        mixture_path = self.artifact_store.local_path(mixture_audio_uri)
        output_prefix = output_prefix_uri.rstrip("/")
        self.backend.preflight()
        stems: dict[str, dict[str, Any]] = {}
        failures: dict[str, str] = {}
        for speaker_id, enrollment_uri in sorted(enrollment_audio_uris.items()):
            validate_artifact_uri(enrollment_uri)
            stem_uri = f"{output_prefix}/{speaker_id}.stem.wav"
            try:
                metadata = self.backend.extract(
                    mixture_wav=mixture_path,
                    enrollment_wav=self.artifact_store.local_path(enrollment_uri),
                    output_wav=self.artifact_store.local_path(stem_uri),
                    speaker_id=speaker_id,
                    options=options,
                )
                stems[speaker_id] = {
                    "speaker_id": speaker_id,
                    "stem_audio_uri": stem_uri,
                    "enrollment_audio_uri": enrollment_uri,
                    "backend_metadata": metadata,
                }
            except Exception as exc:  # noqa: BLE001 - preserve per-speaker failure
                failures[speaker_id] = str(exc)
        status = "completed" if stems and not failures else ("partial" if stems else "failed")
        report = {
            "service": "target_speaker_extraction",
            "status": status,
            "mixture_audio_uri": mixture_audio_uri,
            "output_prefix_uri": output_prefix_uri,
            "backend": self.backend.backend_name,
            "backend_version": self.backend.backend_version,
            "num_requested_speakers": len(enrollment_audio_uris),
            "num_extracted_stems": len(stems),
            "stems": stems,
            "failures": failures,
            "claim_boundary": (
                "ClearerVoice-Studio and WeSep/WebSep target-speaker extraction outputs must pass stem assignment, "
                "residual leakage, speaker-collapse, and manual listening gates before product promotion."
            ),
        }
        self.artifact_store.write_json(output_report_uri, report)
        return report
