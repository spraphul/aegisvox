"""Speech editing / re-synthesis boundary for AegisVox.

Phase 8 edits lexical PHI spans after speaker identity has already been changed.
Runtime synthesis requires a real backend (VoiceCraft first). Tests may inject a
small backend for mechanics, but the default runtime never fabricates edited
speech when timestamped edits exist.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import urllib.error
import urllib.request
import tempfile
import time
import unicodedata
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .audit import redact_text
from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore


class SpeechEditingSetupError(RuntimeError):
    """Raised when VoiceCraft or another real speech editing backend is unavailable."""


class SpeechEditingError(RuntimeError):
    """Raised when edit jobs cannot be planned or stitched safely."""


class SpeechEditingBackend(Protocol):
    backend_name: str
    backend_version: str

    def preflight(self) -> dict[str, Any]: ...

    def synthesize_edit(
        self,
        *,
        source_wav: Path,
        output_wav: Path,
        job: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]: ...


class EditedSpanValidator(Protocol):
    validator_name: str
    validator_version: str

    def preflight(self) -> dict[str, Any]: ...

    def validate_edit(
        self,
        *,
        source_wav: Path,
        edited_wav: Path,
        job: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]: ...


def _canonical_text(text: Any) -> str:
    raw = str(text or "").casefold()
    raw = "".join(ch for ch in unicodedata.normalize("NFKD", raw) if not unicodedata.combining(ch))
    raw = re.sub(r"[^\w\s]", " ", raw, flags=re.UNICODE)
    return re.sub(r"\s+", " ", raw).strip()


def _contains_phrase(text: Any, phrase: Any) -> bool:
    needle = _canonical_text(phrase)
    if not needle:
        return False
    return needle in _canonical_text(text)


@dataclass
class FasterWhisperEditedSpanValidator:
    """ASR validation for generated edit spans before final stitching.

    This validator is intentionally optional. It is used as a hard gate when
    ``require_validated_replacements`` is enabled: synthesized edited-span WAVs
    must contain ASR-recoverable replacement text before they are inserted into
    the final timeline. Failed spans remain available for a fallback renderer.
    """

    model: str | None = None
    device: str | None = None
    compute_type: str | None = None
    download_root: str | Path | None = None
    validator_name: str = "faster-whisper-edited-span"
    validator_version: str = "replacement-presence-v1"
    _model_obj: Any = field(default=None, init=False, repr=False)

    def _model_id(self) -> str:
        return self.model or os.environ.get("AEGISVOX_EDITED_SPAN_VALIDATION_MODEL") or "small"

    def _device(self) -> str:
        return self.device or os.environ.get("AEGISVOX_EDITED_SPAN_VALIDATION_DEVICE") or "cuda"

    def _compute_type(self) -> str:
        return self.compute_type or os.environ.get("AEGISVOX_EDITED_SPAN_VALIDATION_COMPUTE_TYPE") or "float16"

    def _download_root(self) -> str | None:
        value = self.download_root or os.environ.get("AEGISVOX_EDITED_SPAN_VALIDATION_DOWNLOAD_ROOT")
        return str(value) if value else None

    def preflight(self) -> dict[str, Any]:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError as exc:  # pragma: no cover - runtime dependency gate
            raise SpeechEditingSetupError("faster-whisper is required for edited-span ASR validation") from exc
        return {
            "ok": True,
            "validator": self.validator_name,
            "validator_version": self.validator_version,
            "model": self._model_id(),
            "device": self._device(),
            "compute_type": self._compute_type(),
            "download_root": self._download_root(),
        }

    def _model(self) -> Any:
        if self._model_obj is None:
            from faster_whisper import WhisperModel

            kwargs: dict[str, Any] = {"device": self._device(), "compute_type": self._compute_type()}
            download_root = self._download_root()
            if download_root:
                kwargs["download_root"] = download_root
            self._model_obj = WhisperModel(self._model_id(), **kwargs)
        return self._model_obj

    def validate_edit(
        self,
        *,
        source_wav: Path,
        edited_wav: Path,
        job: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        self.preflight()
        if not edited_wav.exists() or edited_wav.stat().st_size == 0:
            return {
                "status": "fail_missing_edited_span",
                "replacement_found": False,
                "asr_text": "",
                "validator": self.validator_name,
            }
        segments_iter, info = self._model().transcribe(
            str(edited_wav),
            beam_size=int(options.get("edited_span_validation_beam_size", 1)),
            vad_filter=str(options.get("edited_span_validation_vad_filter", "false")).strip().lower() in {"1", "true", "yes", "on"},
            word_timestamps=False,
        )
        segments = [str(getattr(segment, "text", "") or "").strip() for segment in segments_iter]
        text = " ".join(part for part in segments if part).strip()
        replacement = str(job.get("replacement_text") or "")
        found = _contains_phrase(text, replacement)
        return {
            "status": "pass_replacement_found" if found else "fail_no_replacement_evidence",
            "replacement_found": found,
            "asr_text": text,
            "replacement_text": replacement,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "validator": self.validator_name,
            "validator_version": self.validator_version,
            "model": self._model_id(),
        }


def _post_json(url: str, payload: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - operator-configured local/internal endpoint
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive HTTPError body handling
            body = ""
        detail = f"HTTP {exc.code} {exc.reason}"
        if body:
            detail = f"{detail}: {body[:1000]}"
        raise SpeechEditingSetupError(f"VoiceCraft service request failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SpeechEditingSetupError(f"VoiceCraft service request failed: {exc}") from exc
    try:
        decoded = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise SpeechEditingSetupError(f"VoiceCraft service returned non-JSON response: {raw[:500]}") from exc
    if not isinstance(decoded, dict):
        raise SpeechEditingSetupError("VoiceCraft service response must be a JSON object")
    return decoded


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


@dataclass
class VoiceCraftCommandBackend:
    """VoiceCraft command adapter.

    Configure with `VOICECRAFT_COMMAND`, a shell command template containing at
    least `{source}`, `{output}`, `{start}`, `{end}`, and `{text}`. Optional
    placeholders include `{speaker}`, `{job_json}`, `{job_json_path}`, and `{model_dir}`.

    Prefer `{job_json_path}` over `{job_json}` for real clinical runs so the
    full replacement plan / target transcript is passed through a temporary file
    rather than exposed in process listings or shell history.

    Example template:

    ```text
    python /opt/VoiceCraft/infer.py --input {source} --output {output} \
      --start {start} --end {end} --text {text}
    ```
    """

    command_template: str | None = None
    model_dir: str | Path | None = None
    backend_name: str = "voicecraft"
    backend_version: str = "command-adapter-v1"

    def __post_init__(self) -> None:
        if self.command_template is None:
            self.command_template = os.environ.get("VOICECRAFT_COMMAND")
        if self.model_dir is None:
            self.model_dir = os.environ.get("VOICECRAFT_MODEL_DIR")

    def preflight(self) -> dict[str, Any]:
        if not self.command_template:
            raise SpeechEditingSetupError(
                "VoiceCraft command is not configured. Set VOICECRAFT_COMMAND to a real speech editing command template."
            )
        required = ["{source}", "{output}", "{start}", "{end}", "{text}"]
        missing = [placeholder for placeholder in required if placeholder not in self.command_template]
        if missing:
            raise SpeechEditingSetupError(f"VOICECRAFT_COMMAND missing required placeholders: {', '.join(missing)}")
        if self.model_dir is not None and not Path(self.model_dir).expanduser().exists():
            raise SpeechEditingSetupError(f"VOICECRAFT_MODEL_DIR does not exist: {self.model_dir}")
        return {
            "ok": True,
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "model_dir": str(self.model_dir) if self.model_dir else None,
        }

    def synthesize_edit(
        self,
        *,
        source_wav: Path,
        output_wav: Path,
        job: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        self.preflight()
        if not source_wav.exists():
            raise AegisVoxContractError(f"changed-identity source WAV not found: {source_wav}")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        job_json = json.dumps(job, sort_keys=True)
        tmp_path: Path | None = None
        try:
            if "{job_json_path}" in self.command_template:
                tmp = tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    prefix="aegisvox_voicecraft_job_",
                    suffix=".json",
                    delete=False,
                )
                with tmp:
                    tmp.write(job_json)
                tmp_path = Path(tmp.name)
            command = self.command_template.format(
                source=shlex.quote(str(source_wav)),
                output=shlex.quote(str(output_wav)),
                start=shlex.quote(str(job["start_sec"])),
                end=shlex.quote(str(job["end_sec"])),
                text=shlex.quote(str(job["replacement_text"])),
                # Avoid using {target_text} in production command templates; it is
                # retained only for legacy public smokes. Prefer {job_json_path}.
                target_text=shlex.quote(str(job.get("target_text") or job.get("deidentified_text") or job.get("replacement_text") or "")),
                original_text=shlex.quote(str(job.get("original_text") or "")),
                speaker=shlex.quote(str(job.get("speaker_id") or "")),
                job_json=shlex.quote(job_json),
                job_json_path=shlex.quote(str(tmp_path)) if tmp_path else "",
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
                timeout=int(options.get("timeout_sec", 900)),
                env=env,
            )
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
        if completed.returncode != 0:
            raise SpeechEditingSetupError(
                f"VoiceCraft command failed with exit {completed.returncode}: {completed.stderr[-1000:]}"
            )
        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise SpeechEditingSetupError("VoiceCraft command completed but did not create a non-empty output WAV")
        return {"command_stdout_tail": completed.stdout[-1000:], "command_stderr_tail": completed.stderr[-1000:]}


@dataclass
class VoiceCraftHTTPBackend:
    """Persistent VoiceCraft service client.

    The target service must keep VoiceCraft and codec models loaded and accept
    ``POST /edit`` requests. Source/output paths are shared SDS paths; the
    service writes the requested span WAV directly to ``output``.
    """

    base_url: str | None = None
    backend_name: str = "voicecraft-http"
    backend_version: str = "persistent-http-v1"

    def __post_init__(self) -> None:
        if self.base_url is None:
            self.base_url = os.environ.get("AEGISVOX_VOICECRAFT_SERVICE_URL") or os.environ.get("VOICECRAFT_SERVICE_URL")

    def preflight(self) -> dict[str, Any]:
        if not self.base_url:
            raise SpeechEditingSetupError("VoiceCraft persistent service is not configured. Set AEGISVOX_VOICECRAFT_SERVICE_URL.")
        try:
            with urllib.request.urlopen(_join_url(self.base_url, "/health"), timeout=10) as response:  # noqa: S310
                health_raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise SpeechEditingSetupError(f"VoiceCraft persistent service health check failed: {exc}") from exc
        return {"ok": True, "backend": self.backend_name, "backend_version": self.backend_version, "base_url": self.base_url, "health": health_raw[:500]}

    def synthesize_edit(
        self,
        *,
        source_wav: Path,
        output_wav: Path,
        job: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        self.preflight()
        if not source_wav.exists():
            raise AegisVoxContractError(f"changed-identity source WAV not found: {source_wav}")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(source_wav),
            "output": str(output_wav),
            "job": dict(job),
            "options": dict(options),
        }
        max_retries = max(0, int(options.get("max_retries", os.environ.get("AEGISVOX_VOICECRAFT_MAX_RETRIES", "2"))))
        retry_backoff_sec = max(0.0, float(options.get("retry_backoff_sec", os.environ.get("AEGISVOX_VOICECRAFT_RETRY_BACKOFF_SEC", "1.0"))))
        last_error: SpeechEditingSetupError | None = None
        for attempt in range(1, max_retries + 2):
            try:
                if output_wav.exists():
                    output_wav.unlink()
                response = _post_json(_join_url(str(self.base_url), "/edit"), payload, timeout_sec=int(options.get("timeout_sec", 900)))
                if not output_wav.exists() or output_wav.stat().st_size == 0:
                    raise SpeechEditingSetupError("VoiceCraft persistent service completed but did not create a non-empty output WAV")
                return {"service_response": response, "attempt": attempt, "max_retries": max_retries}
            except SpeechEditingSetupError as exc:
                last_error = exc
                if attempt > max_retries:
                    break
                if retry_backoff_sec > 0:
                    time.sleep(retry_backoff_sec * attempt)
        assert last_error is not None
        raise SpeechEditingSetupError(f"VoiceCraft persistent service failed after {max_retries + 1} attempt(s): {last_error}") from last_error


@dataclass(frozen=True)
class WavFormat:
    nchannels: int
    sampwidth: int
    framerate: int



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

def _read_wav(path: Path) -> tuple[WavFormat, bytes, int]:
    with wave.open(str(path), "rb") as wav:
        fmt = WavFormat(wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        nframes = wav.getnframes()
        return fmt, wav.readframes(nframes), nframes


def _write_wav(path: Path, fmt: WavFormat, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(fmt.nchannels)
        wav.setsampwidth(fmt.sampwidth)
        wav.setframerate(fmt.framerate)
        wav.writeframes(frames)


def _float_audio_to_pcm_bytes(data: "Any", sampwidth: int) -> bytes:
    import numpy as np

    clipped = np.clip(data, -1.0, 1.0)
    if sampwidth == 1:
        return ((clipped + 1.0) * 127.5).astype(np.uint8).tobytes()
    if sampwidth == 2:
        return (clipped * 32767.0).astype("<i2").tobytes()
    if sampwidth == 4:
        return (clipped * 2147483647.0).astype("<i4").tobytes()
    raise SpeechEditingError(f"unsupported WAV sample width for normalization: {sampwidth}")


def _resample_float_audio(data: "Any", source_rate: int, target_rate: int) -> "Any":
    if source_rate == target_rate:
        return data
    try:
        from scipy.signal import resample_poly
        from math import gcd

        divisor = gcd(int(source_rate), int(target_rate))
        up = int(target_rate) // divisor
        down = int(source_rate) // divisor
        return resample_poly(data, up, down, axis=0)
    except Exception:  # pragma: no cover - fallback for minimal envs
        import numpy as np

        if len(data) == 0:
            return data
        old_x = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        new_len = max(1, int(round(len(data) * float(target_rate) / float(source_rate))))
        new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        channels = []
        for channel_idx in range(data.shape[1]):
            channels.append(np.interp(new_x, old_x, data[:, channel_idx]))
        return np.stack(channels, axis=1)


def _normalize_wav_to_format(path: Path, target_fmt: WavFormat) -> None:
    """Rewrite a backend-produced WAV into the source timeline format.

    VoiceCraft and other real speech-editing backends may emit IEEE-float WAVs
    or a different sample rate. Python's stdlib ``wave`` reader only accepts PCM
    and the timeline stitcher requires the source format. This conversion keeps
    the backend output real while making the artifact stitchable.
    """

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - runtime dependency gate
        raise SpeechEditingError(
            "edited span WAV is not source-format PCM and soundfile/numpy are required for normalization"
        ) from exc

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if data.size == 0:
        raise SpeechEditingError(f"edited span WAV is empty and cannot be normalized: {path}")
    if target_fmt.nchannels == 1 and data.shape[1] != 1:
        data = data.mean(axis=1, keepdims=True)
    elif target_fmt.nchannels > data.shape[1]:
        repeats = int(np.ceil(target_fmt.nchannels / data.shape[1]))
        data = np.tile(data, (1, repeats))[:, : target_fmt.nchannels]
    elif target_fmt.nchannels < data.shape[1]:
        data = data[:, : target_fmt.nchannels]
    data = _resample_float_audio(data, int(sample_rate), target_fmt.framerate)
    frames = _float_audio_to_pcm_bytes(data, target_fmt.sampwidth)
    _write_wav(path, target_fmt, frames)


def _read_wav_matching_format(path: Path, target_fmt: WavFormat) -> tuple[WavFormat, bytes, int]:
    try:
        span_fmt, span_frames, span_nframes = _read_wav(path)
    except wave.Error:
        _normalize_wav_to_format(path, target_fmt)
        return _read_wav(path)
    if span_fmt != target_fmt:
        _normalize_wav_to_format(path, target_fmt)
        return _read_wav(path)
    return span_fmt, span_frames, span_nframes


def _silence_bytes(nframes: int, fmt: WavFormat) -> bytes:
    return b"\x00" * nframes * fmt.nchannels * fmt.sampwidth


def _stitch_intervals_for_job(job: dict[str, Any], options: dict[str, Any]) -> list[tuple[float, float]]:
    """Return absolute timeline intervals to replace for a synthesized job.

    VoiceCraft can use a wide source utterance/window and generate the full
    window while only masking PHI words. Replacing the entire generated window
    is unsafe: overlapping windows can overwrite earlier PHI edits and generated
    non-masked context can drift/repeat. For native-style utterance-window jobs,
    stitch only the mask interval(s) back into the timeline.
    """

    stitch_policy = str(options.get("stitch_policy") or options.get("replacement_policy") or "window").strip().lower()
    if stitch_policy not in {"mask", "masks", "masked", "mask_intervals", "phi_only"}:
        return [(float(job["start_sec"]), float(job["end_sec"]))]
    intervals: list[tuple[float, float]] = []
    raw_intervals = job.get("mask_intervals")
    if isinstance(raw_intervals, list):
        for row in raw_intervals:
            if not isinstance(row, dict):
                continue
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                intervals.append((start, end))
    if not intervals and job.get("mask_start_sec") is not None and job.get("mask_end_sec") is not None:
        start = float(job["mask_start_sec"])
        end = float(job["mask_end_sec"])
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return [(float(job["start_sec"]), float(job["end_sec"]))]
    job_start = float(job["start_sec"])
    job_end = float(job["end_sec"])
    stitch_context_sec = max(0.0, float(options.get("stitch_context_sec", 0.0) or 0.0))
    bounded: list[tuple[float, float]] = []
    for start, end in intervals:
        bounded_start = max(job_start, start - stitch_context_sec)
        bounded_end = min(job_end, end + stitch_context_sec)
        if bounded_end > bounded_start:
            bounded.append((bounded_start, bounded_end))
    return bounded or [(job_start, job_end)]


def _as_float(value: Any, field_name: str) -> float:
    if value is None:
        raise SpeechEditingError(f"edit missing {field_name}; timestamped speech editing requires localized PHI spans")
    return float(value)


def _word_count(text: Any) -> int:
    return len([part for part in str(text or "").strip().split() if part])


def _language_for_edit(edit: dict[str, Any]) -> str:
    raw = str(edit.get("replacement_language") or edit.get("source_language") or edit.get("language") or "").strip().casefold()
    if raw.startswith("fr") or raw in {"french", "fra", "fre"}:
        return "fr"
    if raw.startswith("en") or raw in {"english", "eng"}:
        return "en"
    return raw or "unknown"


def _short_span_filler(language: str) -> str:
    if language == "fr":
        return "euh"
    return "uh"


def _target_text_for_short_span(edit: dict[str, Any], replacement_text: str) -> tuple[str | None, dict[str, Any]]:
    """Build acoustically safer text for very short lexical edits.

    VoiceCraft/VC backends tend to sound robotic on naked one-token replacements.
    The preferred path is to render the containing phrase/utterance with the PHI
    token replaced. If no phrase context is available, add one neutral filler
    token solely to provide phonetic/prosodic context.
    """

    original_text = str(edit.get("original_text") or "").strip()
    source_segment_text = str(edit.get("source_segment_text") or "").strip()
    language = _language_for_edit(edit)
    if source_segment_text and original_text and original_text in source_segment_text:
        return source_segment_text.replace(original_text, replacement_text), {
            "strategy": "context_expanded_phrase",
            "source": "source_segment_text",
            "language": language,
        }
    filler = _short_span_filler(language)
    return f"{filler} {replacement_text}".strip(), {
        "strategy": "neutral_filler_prefix",
        "filler": filler,
        "language": language,
        "semantic_constraint": "neutral_disfluency_only_no_clinical_fact_added",
    }


def _route_edit_target_text(
    *,
    edit: dict[str, Any],
    start_sec: float,
    end_sec: float,
    default_target_text: Any,
) -> tuple[str | None, dict[str, Any]]:
    duration = max(0.0, end_sec - start_sec)
    replacement_text = str(edit.get("replacement_text") or "")
    threshold_sec = float(os.environ.get("AEGISVOX_SHORT_EDIT_SPAN_THRESHOLD_SEC", "1.2"))
    threshold_words = int(os.environ.get("AEGISVOX_SHORT_EDIT_WORD_THRESHOLD", "2"))
    enable = os.environ.get("AEGISVOX_ENABLE_SHORT_SPAN_CONTEXT_RENDERING", "true").strip().lower() not in {"0", "false", "no"}
    base = {
        "enabled": enable,
        "span_duration_sec": round(duration, 3),
        "replacement_word_count": _word_count(replacement_text),
        "threshold_sec": threshold_sec,
        "threshold_words": threshold_words,
    }
    if not enable:
        return str(default_target_text) if default_target_text is not None else None, {**base, "strategy": "disabled"}
    is_short = duration < threshold_sec or _word_count(replacement_text) <= threshold_words
    if not is_short:
        return str(default_target_text) if default_target_text is not None else None, {**base, "strategy": "direct_span"}
    current = str(default_target_text or "").strip()
    if current and current != replacement_text and _word_count(current) > _word_count(replacement_text):
        return current, {**base, "strategy": "already_contextualized"}
    expanded, policy = _target_text_for_short_span(edit, replacement_text)
    return expanded, {**base, **policy}


def _edits_from_deid_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    edits = payload.get("edits")
    if not isinstance(edits, list):
        raise AegisVoxContractError("deid_text.json must contain an edits list")
    return [dict(edit) for edit in edits]


def build_edit_jobs_payload(
    *,
    changed_identity_audio_uri: str,
    deid_text_uri: str,
    output_span_prefix_uri: str,
    edits: list[dict[str, Any]],
    deidentified_text: str | None = None,
) -> dict[str, Any]:
    validate_artifact_uri(changed_identity_audio_uri)
    validate_artifact_uri(deid_text_uri)
    validate_artifact_uri(output_span_prefix_uri.rstrip("/") + "/")
    jobs = []
    unlocalized = []
    for idx, edit in enumerate(edits, start=1):
        edit_id = str(edit.get("edit_id") or f"edit_{idx:04d}")
        try:
            start_sec = _as_float(edit.get("start_sec"), "start_sec")
            end_sec = _as_float(edit.get("end_sec"), "end_sec")
        except SpeechEditingError as exc:
            unlocalized.append({"edit_id": edit_id, "reason": str(exc), "phi_type": edit.get("phi_type")})
            continue
        if end_sec <= start_sec:
            unlocalized.append({"edit_id": edit_id, "reason": "end_sec must be greater than start_sec", "phi_type": edit.get("phi_type")})
            continue
        target_text_scope = os.environ.get("AEGISVOX_SPEECH_EDITING_TARGET_TEXT_SCOPE", "span").strip().lower()
        if target_text_scope in {"full", "transcript"}:
            target_text = deidentified_text or edit.get("target_text") or edit.get("replacement_text")
        else:
            # VoiceCraft edits a short cropped audio context. Passing an entire
            # long clinical transcript per tiny PHI span makes inference much
            # slower and can dominate runtime. Default to the local replacement
            # text unless operators explicitly request full-transcript context.
            target_text = edit.get("target_text") or edit.get("replacement_text")
        target_text, short_span_policy = _route_edit_target_text(
            edit=edit,
            start_sec=start_sec,
            end_sec=end_sec,
            default_target_text=target_text,
        )
        jobs.append(
            {
                "job_id": f"voicecraft_job_{len(jobs)+1:04d}",
                "edit_id": edit_id,
                "phi_type": edit.get("phi_type"),
                "speaker_id": edit.get("speaker_id"),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "mask_start_sec": edit.get("mask_start_sec"),
                "mask_end_sec": edit.get("mask_end_sec"),
                "source_edit_ids": edit.get("source_edit_ids"),
                "source_segment_index": edit.get("source_segment_index"),
                "source_segment_text": edit.get("source_segment_text"),
                "mask_interval_evidence": edit.get("mask_interval_evidence"),
                "mask_intervals": edit.get("mask_intervals"),
                "replacement_application": edit.get("replacement_application"),
                "original_text": edit.get("original_text"),
                "replacement_text": edit.get("replacement_text"),
                "target_text": target_text,
                "target_text_scope": target_text_scope or "span",
                "short_span_render_policy": short_span_policy,
                "source_audio_uri": changed_identity_audio_uri,
                "edited_span_audio_uri": f"{output_span_prefix_uri.rstrip('/')}/{edit_id}.edited.wav",
                "status": "planned",
            }
        )
    return {
        "changed_identity_audio_uri": changed_identity_audio_uri,
        "deid_text_uri": deid_text_uri,
        "num_edits": len(edits),
        "num_jobs": len(jobs),
        "num_unlocalized_edits": len(unlocalized),
        "jobs": jobs,
        "unlocalized_edits": unlocalized,
    }


@dataclass
class SpeechEditingService:
    artifact_store: LocalArtifactStore
    backend: SpeechEditingBackend = field(default_factory=VoiceCraftCommandBackend)
    edited_span_validator: EditedSpanValidator | None = None

    def preflight(self) -> dict[str, Any]:
        payload = self.backend.preflight()
        if self.edited_span_validator is not None:
            payload = {**payload, "edited_span_validator": self.edited_span_validator.preflight()}
        return payload

    def plan_jobs_to_artifact(
        self,
        *,
        changed_identity_audio_uri: str,
        deid_text_uri: str,
        output_jobs_uri: str,
        output_span_prefix_uri: str,
    ) -> dict[str, Any]:
        validate_artifact_uri(output_jobs_uri)
        deid_payload = self.artifact_store.read_json(deid_text_uri)
        if not isinstance(deid_payload, dict):
            raise AegisVoxContractError("deid_text artifact must be a JSON object")
        payload = build_edit_jobs_payload(
            changed_identity_audio_uri=changed_identity_audio_uri,
            deid_text_uri=deid_text_uri,
            output_span_prefix_uri=output_span_prefix_uri,
            edits=_edits_from_deid_payload(deid_payload),
            deidentified_text=str(deid_payload.get("deidentified_text") or "") or None,
        )
        self.artifact_store.write_json(output_jobs_uri, payload)
        return payload

    def edit_to_artifacts(
        self,
        *,
        changed_identity_audio_uri: str,
        deid_text_uri: str,
        output_jobs_uri: str,
        output_span_prefix_uri: str,
        final_audio_uri: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validate_artifact_uri(final_audio_uri)
        options = options or {}
        jobs_payload = self.plan_jobs_to_artifact(
            changed_identity_audio_uri=changed_identity_audio_uri,
            deid_text_uri=deid_text_uri,
            output_jobs_uri=output_jobs_uri,
            output_span_prefix_uri=output_span_prefix_uri,
        )
        source_path = self.artifact_store.local_path(changed_identity_audio_uri)
        final_path = self.artifact_store.local_path(final_audio_uri)
        if not source_path.exists():
            raise AegisVoxContractError(f"changed-identity audio not found: {source_path}")
        if jobs_payload["num_unlocalized_edits"]:
            self.artifact_store.write_json(output_jobs_uri, {**jobs_payload, "status": "failed_unlocalized_edits"})
            raise SpeechEditingError(
                f"cannot synthesize {jobs_payload['num_unlocalized_edits']} unlocalized edits; rerun ASR/alignment with timestamps"
            )
        if not jobs_payload["jobs"]:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, final_path)
            payload = {
                **jobs_payload,
                "status": "no_edits_passthrough",
                "final_sanitized_audio_uri": final_audio_uri,
                "backend": None,
                "backend_version": None,
            }
            self.artifact_store.write_json(output_jobs_uri, payload)
            return payload
        self.backend.preflight()
        source_fmt, source_frames, total_frames = _read_wav(source_path)
        timeline = bytearray(source_frames)
        require_validated_replacements = str(options.get("require_validated_replacements", "false")).strip().lower() in {"1", "true", "yes"}
        edited_span_validator = self.edited_span_validator
        if require_validated_replacements and edited_span_validator is None:
            edited_span_validator = FasterWhisperEditedSpanValidator()
        if edited_span_validator is not None:
            edited_span_validator.preflight()
        num_workers = min(_positive_int(options.get("num_workers") or options.get("workers"), default=1), max(1, len(jobs_payload["jobs"])))
        cuda_devices = _cuda_devices(options)
        resume_existing = str(options.get("resume_existing", "true")).strip().lower() not in {"0", "false", "no"}

        def synthesize_one(index: int, job: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            output_path = self.artifact_store.local_path(job["edited_span_audio_uri"])
            job_options = dict(options)
            if cuda_devices:
                job_options["cuda_visible_devices"] = cuda_devices[index % len(cuda_devices)]
            try:
                if resume_existing and output_path.exists() and output_path.stat().st_size > 0:
                    metadata = {"resumed_from_existing_output": True}
                    updated = {**job, "status": "synthesized", "backend_metadata": metadata, "resumed_from_existing_output": True}
                else:
                    metadata = self.backend.synthesize_edit(source_wav=source_path, output_wav=output_path, job=job, options=job_options)
                    updated = {**job, "status": "synthesized", "backend_metadata": metadata}
                if edited_span_validator is not None:
                    try:
                        _read_wav_matching_format(output_path, source_fmt)
                        validation = edited_span_validator.validate_edit(
                            source_wav=source_path,
                            edited_wav=output_path,
                            job=updated,
                            options=job_options,
                        )
                    except Exception as exc:  # noqa: BLE001 - preserve validation failure evidence
                        validation = {
                            "status": "fail_validation_error",
                            "replacement_found": False,
                            "failure_reason": redact_text(str(exc), max_length=1000),
                            "validator": getattr(edited_span_validator, "validator_name", "unknown"),
                        }
                    updated = {**updated, "edited_span_validation": validation}
                    if require_validated_replacements and not bool(validation.get("replacement_found")):
                        updated = {
                            **updated,
                            "status": "failed_validation",
                            "failure_reason": redact_text(str(validation.get("status") or "edited span lacks replacement evidence"), max_length=1000),
                        }
                if job_options.get("cuda_visible_devices") not in {None, ""}:
                    updated["cuda_visible_devices"] = str(job_options["cuda_visible_devices"])
                return index, updated
            except Exception as exc:  # noqa: BLE001 - preserve per-job failure evidence
                failed = {**job, "status": "failed", "failure_reason": redact_text(str(exc), max_length=1000)}
                if job_options.get("cuda_visible_devices") not in {None, ""}:
                    failed["cuda_visible_devices"] = str(job_options["cuda_visible_devices"])
                return index, failed

        updated_jobs_by_index: list[dict[str, Any] | None] = [None] * len(jobs_payload["jobs"])
        if num_workers <= 1:
            for index, job in enumerate(jobs_payload["jobs"]):
                result_index, updated = synthesize_one(index, job)
                updated_jobs_by_index[result_index] = updated
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(synthesize_one, index, job) for index, job in enumerate(jobs_payload["jobs"])]
                for future in as_completed(futures):
                    result_index, updated = future.result()
                    updated_jobs_by_index[result_index] = updated
        updated_jobs = [job for job in updated_jobs_by_index if job is not None]

        stitch_operations: list[dict[str, Any]] = []
        for job in updated_jobs:
            if job.get("status") != "synthesized":
                continue
            output_path = self.artifact_store.local_path(job["edited_span_audio_uri"])
            span_fmt, span_frames, span_nframes = _read_wav_matching_format(output_path, source_fmt)
            if span_fmt != source_fmt:
                raise SpeechEditingError(f"edited span format normalization failed for {output_path}")
            job_start_frame = max(0, int(round(float(job["start_sec"]) * source_fmt.framerate)))
            for interval_start_sec, interval_end_sec in _stitch_intervals_for_job(job, options):
                start_frame = max(0, int(round(interval_start_sec * source_fmt.framerate)))
                end_frame = min(total_frames, int(round(interval_end_sec * source_fmt.framerate)))
                target_nframes = max(0, end_frame - start_frame)
                target_bytes = target_nframes * source_fmt.nchannels * source_fmt.sampwidth
                if target_bytes <= 0:
                    continue
                span_offset_frame = max(0, start_frame - job_start_frame)
                span_offset_byte = span_offset_frame * source_fmt.nchannels * source_fmt.sampwidth
                replacement = span_frames[span_offset_byte : span_offset_byte + target_bytes]
                if len(replacement) < target_bytes:
                    replacement += _silence_bytes(target_nframes, source_fmt)[: target_bytes - len(replacement)]
                start_byte = start_frame * source_fmt.nchannels * source_fmt.sampwidth
                timeline[start_byte : start_byte + target_bytes] = replacement[:target_bytes]
                stitch_operations.append(
                    {
                        "job_id": job.get("job_id"),
                        "edit_id": job.get("edit_id"),
                        "start_sec": round(interval_start_sec, 3),
                        "end_sec": round(interval_end_sec, 3),
                        "source_window_start_sec": job.get("start_sec"),
                        "source_window_end_sec": job.get("end_sec"),
                    }
                )
        failed = [job for job in updated_jobs if str(job.get("status", "")).startswith("failed")]
        allow_partial_failures = str(options.get("allow_partial_failures", "false")).strip().lower() in {"1", "true", "yes"}
        if failed and not allow_partial_failures:
            payload = {
                **jobs_payload,
                "status": "failed",
                "backend": self.backend.backend_name,
                "backend_version": self.backend.backend_version,
                "jobs": updated_jobs,
                "num_failed_jobs": len(failed),
                "edit_parallelism": {
                    "num_workers": num_workers,
                    "cuda_devices": cuda_devices,
                    "strategy": "job_threadpool_preserve_order" if num_workers > 1 else "serial",
                    "resume_existing": resume_existing,
                },
                "num_resumed_existing": sum(1 for job in updated_jobs if job.get("resumed_from_existing_output")),
            }
            self.artifact_store.write_json(output_jobs_uri, payload)
            raise SpeechEditingError(f"speech editing failed for {len(failed)} jobs; see voicecraft_edit_jobs.json")
        _write_wav(final_path, source_fmt, bytes(timeline))
        payload = {
            **jobs_payload,
            "status": "completed_with_failed_jobs" if failed else "completed",
            "backend": self.backend.backend_name,
            "backend_version": self.backend.backend_version,
            "jobs": updated_jobs,
            "num_failed_jobs": len(failed),
            "failed_job_ids": [str(job.get("job_id")) for job in failed],
            "partial_failure_policy": "failed or validation-rejected spans preserved from source pending fallback" if failed else None,
            "num_resumed_existing": sum(1 for job in updated_jobs if job.get("resumed_from_existing_output")),
            "edited_span_validation": {
                "enabled": edited_span_validator is not None,
                "required_for_stitching": require_validated_replacements,
                "validator": getattr(edited_span_validator, "validator_name", None) if edited_span_validator is not None else None,
                "num_validated": sum(1 for job in updated_jobs if isinstance(job.get("edited_span_validation"), dict)),
                "num_validation_failed": sum(1 for job in updated_jobs if str(job.get("status", "")).startswith("failed_validation")),
            },
            "edit_parallelism": {
                "num_workers": num_workers,
                "cuda_devices": cuda_devices,
                "strategy": "job_threadpool_preserve_order" if num_workers > 1 else "serial",
                "resume_existing": resume_existing,
            },
            "stitch_policy": str(options.get("stitch_policy") or options.get("replacement_policy") or "window"),
            "stitch_operations": stitch_operations,
            "final_sanitized_audio_uri": final_audio_uri,
        }
        self.artifact_store.write_json(output_jobs_uri, payload)
        return payload
