"""ASR + speaker-attribution service boundary."""

from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .schema import AegisVoxContractError, validate_artifact_uri
from .segmentation import load_diarization_turns
from .storage import LocalArtifactStore


class ASRSetupError(RuntimeError):
    """Raised when the ASR backend is not configured."""


class ASRBackend(Protocol):
    backend_name: str
    backend_version: str

    def preflight(self) -> dict[str, Any]: ...

    def transcribe(self, audio_path: Path) -> dict[str, Any]: ...


@dataclass
class OpenAIASRBackend:
    """OpenAI ASR backend wrapper.

    Requires `OPENAI_API_KEY` or `OPENAI_API_KEY` at runtime and the `openai`
    package installed. The API call is intentionally isolated for testability.
    """

    model: str = "gpt-4o-transcribe"
    backend_name: str = "openai-asr"
    backend_version: str = "openai-python"

    def _api_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ASRSetupError("OpenAI ASR requires OPENAI_API_KEY or OPENAI_API_KEY at runtime; do not store it in repo files")
        return key

    def _base_url(self) -> str | None:
        return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")

    def _model(self) -> str:
        return os.environ.get("OPENAI_ASR_MODEL") or os.environ.get("AEGISVOX_OPENAI_ASR_MODEL") or self.model

    def _response_format(self) -> str:
        explicit = os.environ.get("OPENAI_ASR_RESPONSE_FORMAT") or os.environ.get("AEGISVOX_ASR_RESPONSE_FORMAT")
        if explicit:
            return explicit
        model = self._model()
        if model.startswith("gpt-4o"):
            return "json"
        return "verbose_json"

    def _timestamp_granularities(self) -> list[str] | None:
        configured = os.environ.get("OPENAI_ASR_TIMESTAMP_GRANULARITIES") or os.environ.get(
            "AEGISVOX_ASR_TIMESTAMP_GRANULARITIES"
        )
        if configured:
            values = [value.strip() for value in configured.split(",") if value.strip()]
            return values or None
        if self._response_format() == "verbose_json":
            return ["word", "segment"]
        return None

    def _max_upload_bytes(self) -> int:
        return int(os.environ.get("AEGISVOX_ASR_MAX_UPLOAD_BYTES", "24000000"))

    def _chunk_seconds(self) -> float:
        return float(os.environ.get("AEGISVOX_ASR_CHUNK_SECONDS", "300"))

    def preflight(self) -> dict[str, Any]:
        self._api_key()
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ASRSetupError("openai package is not installed; install ASR service dependencies first") from exc
        return {
            "ok": True,
            "backend": self.backend_name,
            "model": self._model(),
            "response_format": self._response_format(),
            "base_url_configured": self._base_url() is not None,
            "max_upload_bytes": self._max_upload_bytes(),
            "chunk_seconds": self._chunk_seconds(),
        }

    def _client(self) -> Any:
        from openai import OpenAI

        client_kwargs: dict[str, str] = {"api_key": self._api_key()}
        base_url = self._base_url()
        if base_url:
            client_kwargs["base_url"] = base_url
        return OpenAI(**client_kwargs)

    def _request_kwargs(self) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self._model(),
            "response_format": self._response_format(),
        }
        timestamp_granularities = self._timestamp_granularities()
        if timestamp_granularities:
            request_kwargs["timestamp_granularities"] = timestamp_granularities
        return request_kwargs

    def _transcribe_single(self, client: Any, audio_path: Path) -> dict[str, Any]:
        request_kwargs = self._request_kwargs()
        with audio_path.open("rb") as audio_file:
            request_kwargs["file"] = audio_file
            response = client.audio.transcriptions.create(**request_kwargs)
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, dict):
            return response
        return dict(response)

    def _should_chunk(self, audio_path: Path) -> bool:
        if audio_path.stat().st_size <= self._max_upload_bytes():
            return False
        try:
            with wave.open(str(audio_path), "rb") as wav:
                return wav.getnframes() > 0
        except Exception:
            return False

    def _write_chunk(self, reader: wave.Wave_read, path: Path, nframes: int) -> int:
        frames = reader.readframes(nframes)
        actual_frames = len(frames) // max(1, reader.getnchannels() * reader.getsampwidth())
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(reader.getnchannels())
            writer.setsampwidth(reader.getsampwidth())
            writer.setframerate(reader.getframerate())
            writer.writeframes(frames)
        return actual_frames

    def _max_frames_per_upload(self, reader: wave.Wave_read) -> int:
        """Return a conservative frame count that should fit the upload limit.

        The OpenAI endpoint enforces the request body size, not just the WAV
        payload. Keep an overhead/safety margin so full-bandwidth stereo or
        float WAV stems from source-separation backends do not create an
        over-limit first chunk even when `AEGISVOX_ASR_CHUNK_SECONDS` is large.
        """

        bytes_per_frame = max(1, reader.getnchannels() * reader.getsampwidth())
        wav_header_margin = 4096
        multipart_safety_factor = float(os.environ.get("AEGISVOX_ASR_UPLOAD_SAFETY_FACTOR", "0.80"))
        multipart_safety_factor = min(max(multipart_safety_factor, 0.1), 0.95)
        safe_payload_bytes = max(1, int((self._max_upload_bytes() - wav_header_margin) * multipart_safety_factor))
        return max(1, safe_payload_bytes // bytes_per_frame)

    @staticmethod
    def _offset_rows(rows: Any, offset_sec: float) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if not isinstance(rows, list):
            return output
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            for key in ("start", "end", "start_sec", "end_sec"):
                if row.get(key) is not None:
                    row[key] = round(float(row[key]) + offset_sec, 3)
            output.append(row)
        return output

    def _transcribe_chunked_wav(self, client: Any, audio_path: Path) -> dict[str, Any]:
        chunk_seconds = self._chunk_seconds()
        if chunk_seconds <= 0:
            raise ASRSetupError("AEGISVOX_ASR_CHUNK_SECONDS must be positive")
        chunks: list[dict[str, Any]] = []
        all_segments: list[dict[str, Any]] = []
        all_words: list[dict[str, Any]] = []
        texts: list[str] = []
        with tempfile.TemporaryDirectory(prefix="aegisvox_asr_chunks_") as tmpdir:
            tmp = Path(tmpdir)
            with wave.open(str(audio_path), "rb") as reader:
                rate = reader.getframerate()
                frames_per_time_chunk = max(1, int(round(chunk_seconds * rate)))
                frames_per_chunk = min(frames_per_time_chunk, self._max_frames_per_upload(reader))
                total_frames = reader.getnframes()
                chunk_index = 0
                start_frame = 0
                while start_frame < total_frames:
                    chunk_path = tmp / f"chunk_{chunk_index:04d}.wav"
                    actual_frames = self._write_chunk(reader, chunk_path, min(frames_per_chunk, total_frames - start_frame))
                    if actual_frames <= 0:
                        break
                    offset_sec = start_frame / float(rate)
                    try:
                        raw = self._transcribe_single(client, chunk_path)
                    except Exception as exc:  # pragma: no cover - external API path
                        raise ASRSetupError(
                            f"OpenAI ASR transcription failed for chunk {chunk_index} of {audio_path}: {exc}"
                        ) from exc
                    text = str(raw.get("text") or "").strip()
                    if text:
                        texts.append(text)
                    all_segments.extend(self._offset_rows(raw.get("segments"), offset_sec))
                    all_words.extend(self._offset_rows(raw.get("words"), offset_sec))
                    chunks.append(
                        {
                            "chunk_index": chunk_index,
                            "offset_sec": round(offset_sec, 3),
                            "duration_sec": round(actual_frames / float(rate), 3),
                            "bytes": chunk_path.stat().st_size,
                            "text_chars": len(text),
                            "num_segments": len(raw.get("segments") or []),
                            "num_words": len(raw.get("words") or []),
                        }
                    )
                    start_frame += actual_frames
                    chunk_index += 1
        return {
            "text": " ".join(texts).strip(),
            "segments": all_segments,
            "words": all_words,
            "chunked_asr": True,
            "chunks": chunks,
            "source_audio_bytes": audio_path.stat().st_size,
            "chunk_seconds": chunk_seconds,
        }

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        self.preflight()
        if not audio_path.exists():
            raise AegisVoxContractError(f"audio path not found: {audio_path}")
        try:
            client = self._client()
            if self._should_chunk(audio_path):
                return self._transcribe_chunked_wav(client, audio_path)
            return self._transcribe_single(client, audio_path)
        except ASRSetupError:
            raise
        except Exception as exc:  # pragma: no cover - external API path
            raise ASRSetupError(f"OpenAI ASR transcription failed for {audio_path}: {exc}") from exc


@dataclass
class LocalFasterWhisperASRBackend:
    """Local faster-whisper/CTranslate2 ASR backend for ORD.

    This is the default fast path for long-audio engineering stages such as
    background leakage checks. It avoids remote upload limits/costs and can run
    fully on ORD GPUs.
    """

    model_size: str = "small.en"
    backend_name: str = "local-faster-whisper"
    backend_version: str = "faster-whisper"
    _model_obj: Any = field(default=None, init=False, repr=False)

    def _model_id(self) -> str:
        return (
            os.environ.get("AEGISVOX_FAST_WHISPER_MODEL")
            or os.environ.get("AEGISVOX_LOCAL_ASR_MODEL")
            or os.environ.get("AEGISVOX_ASR_MODEL")
            or self.model_size
        )

    def _device(self) -> str:
        return os.environ.get("AEGISVOX_FAST_WHISPER_DEVICE") or os.environ.get("AEGISVOX_ASR_DEVICE") or "cuda"

    def _compute_type(self) -> str:
        return os.environ.get("AEGISVOX_FAST_WHISPER_COMPUTE_TYPE") or os.environ.get("AEGISVOX_ASR_COMPUTE_TYPE") or "float16"

    def _vad_filter(self) -> bool:
        value = os.environ.get("AEGISVOX_FAST_WHISPER_VAD_FILTER") or os.environ.get("AEGISVOX_ASR_VAD_FILTER") or "true"
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    def _beam_size(self) -> int:
        return int(os.environ.get("AEGISVOX_FAST_WHISPER_BEAM_SIZE", "1"))

    def preflight(self) -> dict[str, Any]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError as exc:
            raise ASRSetupError(
                "faster-whisper is not installed; install AegisVox local ASR dependencies first"
            ) from exc
        return {
            "ok": True,
            "backend": self.backend_name,
            "model": self._model_id(),
            "device": self._device(),
            "compute_type": self._compute_type(),
            "vad_filter": self._vad_filter(),
            "beam_size": self._beam_size(),
        }

    def _model(self) -> Any:
        if self._model_obj is not None:
            return self._model_obj
        try:
            from faster_whisper import WhisperModel

            self._model_obj = WhisperModel(self._model_id(), device=self._device(), compute_type=self._compute_type())
        except Exception as exc:  # pragma: no cover - external model path
            raise ASRSetupError(
                f"failed to load faster-whisper model {self._model_id()!r} on {self._device()!r}"
            ) from exc
        self.backend_version = f"faster-whisper;model={self._model_id()};device={self._device()};compute={self._compute_type()}"
        return self._model_obj

    @staticmethod
    def _segment_to_dict(segment: Any) -> dict[str, Any]:
        words = []
        for word in getattr(segment, "words", None) or []:
            words.append(
                {
                    "word": getattr(word, "word", ""),
                    "start": getattr(word, "start", None),
                    "end": getattr(word, "end", None),
                    "probability": getattr(word, "probability", None),
                }
            )
        row = {
            "id": getattr(segment, "id", None),
            "text": str(getattr(segment, "text", "") or "").strip(),
            "start": getattr(segment, "start", None),
            "end": getattr(segment, "end", None),
        }
        if words:
            row["words"] = words
        return row

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        self.preflight()
        if not audio_path.exists():
            raise AegisVoxContractError(f"audio path not found: {audio_path}")
        word_timestamp_fallback = None
        try:
            segments_iter, info = self._model().transcribe(
                str(audio_path),
                beam_size=self._beam_size(),
                vad_filter=self._vad_filter(),
                word_timestamps=True,
            )
            segments = [self._segment_to_dict(segment) for segment in segments_iter]
        except Exception as exc:  # pragma: no cover - external model path
            # CTranslate2 word alignment can fail on an otherwise decodable
            # utterance with this exact CUDA error. Matched-ASR utility consumes
            # transcript text, not word alignment, so retry only that narrow
            # failure with the same model/beam/VAD settings and timestamps off.
            if "parallel_for failed: cudaErrorInvalidDevice: invalid device ordinal" not in str(exc):
                raise ASRSetupError(f"faster-whisper transcription failed for {audio_path}: {exc}") from exc
            try:
                segments_iter, info = self._model().transcribe(
                    str(audio_path),
                    beam_size=self._beam_size(),
                    vad_filter=self._vad_filter(),
                    word_timestamps=False,
                )
                segments = [self._segment_to_dict(segment) for segment in segments_iter]
                word_timestamp_fallback = "cuda_alignment_invalid_device"
            except Exception as fallback_exc:  # pragma: no cover - external model path
                raise ASRSetupError(
                    f"faster-whisper transcription failed for {audio_path}: {fallback_exc}"
                ) from fallback_exc
        words: list[dict[str, Any]] = []
        for segment in segments:
            words.extend(segment.get("words") or [])
        payload = {
            "text": " ".join(segment.get("text", "") for segment in segments).strip(),
            "segments": segments,
            "words": words,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "backend": self.backend_name,
            "model": self._model_id(),
        }
        if word_timestamp_fallback is not None:
            payload["word_timestamps_available"] = False
            payload["word_timestamp_fallback"] = word_timestamp_fallback
        return payload


@dataclass
class NVIDIACanaryASRBackend:
    """NVIDIA Canary local ASR backend.

    This backend is heavier than faster-whisper and currently returns transcript
    text without reliable word timestamps. Keep it as a quality/benchmark path,
    while faster-whisper remains the fast default for the e2e pipeline.
    """

    model_id: str = "nvidia/canary-1b-v2"
    backend_name: str = "nvidia-canary"
    backend_version: str = "nemo-canary"
    _model_obj: Any = field(default=None, init=False, repr=False)

    def _model_name(self) -> str:
        return os.environ.get("AEGISVOX_CANARY_MODEL") or os.environ.get("AEGISVOX_ASR_MODEL") or self.model_id

    def _device(self) -> str:
        return os.environ.get("AEGISVOX_CANARY_DEVICE") or os.environ.get("AEGISVOX_ASR_DEVICE") or "cuda"

    def preflight(self) -> dict[str, Any]:
        try:
            import nemo.collections.asr as nemo_asr  # noqa: F401
        except ImportError as exc:
            raise ASRSetupError("NVIDIA Canary ASR requires NeMo ASR dependencies in the runtime env") from exc
        return {"ok": True, "backend": self.backend_name, "model": self._model_name(), "device": self._device()}

    def _model(self) -> Any:
        if self._model_obj is not None:
            return self._model_obj
        try:
            import nemo.collections.asr as nemo_asr

            model_cls = getattr(nemo_asr.models, "EncDecMultiTaskModel", None) or getattr(nemo_asr.models, "ASRModel")
            model = model_cls.from_pretrained(self._model_name())
            device = self._device()
            if hasattr(model, "to") and device:
                model = model.to(device)
            if hasattr(model, "eval"):
                model.eval()
            self._model_obj = model
        except Exception as exc:  # pragma: no cover - external model path
            raise ASRSetupError(f"failed to load NVIDIA Canary ASR model {self._model_name()!r}") from exc
        self.backend_version = f"nemo-canary;model={self._model_name()};device={self._device()}"
        return self._model_obj

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        self.preflight()
        if not audio_path.exists():
            raise AegisVoxContractError(f"audio path not found: {audio_path}")
        try:
            result = self._model().transcribe([str(audio_path)], batch_size=1)
        except TypeError:
            result = self._model().transcribe(paths2audio_files=[str(audio_path)], batch_size=1)
        except Exception as exc:  # pragma: no cover - external model path
            raise ASRSetupError(f"NVIDIA Canary transcription failed for {audio_path}: {exc}") from exc
        text = result[0] if isinstance(result, list) and result else result
        if hasattr(text, "text"):
            text = text.text
        return {
            "text": str(text or "").strip(),
            "segments": [],
            "words": [],
            "backend": self.backend_name,
            "model": self._model_name(),
        }


def create_asr_backend(kind: str | None = None) -> ASRBackend:
    backend = (kind or os.environ.get("AEGISVOX_ASR_BACKEND") or "openai").strip().lower()
    if backend in {"openai", "openai_asr", "openai-asr"}:
        return OpenAIASRBackend()
    if backend in {"local_faster_whisper", "faster_whisper", "faster-whisper", "whisper"}:
        return LocalFasterWhisperASRBackend()
    if backend in {"nvidia_canary", "canary", "nemo_canary", "nvidia-canary"}:
        return NVIDIACanaryASRBackend()
    raise ASRSetupError(
        f"unknown AegisVox ASR backend {backend!r}; expected openai, local_faster_whisper, or nvidia_canary"
    )


@dataclass(frozen=True)
class TranscriptWord:
    word: str
    start_sec: float | None = None
    end_sec: float | None = None
    speaker_id: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "speaker_id": self.speaker_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_sec: float | None = None
    end_sec: float | None = None
    speaker_id: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "speaker_id": self.speaker_id,
            "confidence": self.confidence,
        }


def _speaker_at_time(seconds: float | None, diarization_rows: list[dict[str, Any]]) -> str | None:
    if seconds is None:
        return None
    for row in diarization_rows:
        start = row.get("start_sec", row.get("start"))
        end = row.get("end_sec", row.get("end"))
        if start is None or end is None:
            continue
        if float(start) <= seconds <= float(end):
            return str(row.get("speaker_id") or row.get("speaker") or row.get("label"))
    return None


def _midpoint(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return start if start is not None else end
    return (start + end) / 2.0


def _norm_time(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def normalize_asr_output(raw: dict[str, Any], diarization_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    diarization_rows = diarization_rows or []
    raw_segments = raw.get("segments") or []
    raw_words = raw.get("words") or []
    segments: list[TranscriptSegment] = []
    words: list[TranscriptWord] = []
    for row in raw_segments:
        start = _norm_time(row.get("start") if "start" in row else row.get("start_sec"))
        end = _norm_time(row.get("end") if "end" in row else row.get("end_sec"))
        speaker = _speaker_at_time(_midpoint(start, end), diarization_rows)
        segments.append(
            TranscriptSegment(
                text=str(row.get("text") or "").strip(),
                start_sec=start,
                end_sec=end,
                speaker_id=speaker,
                confidence=row.get("confidence"),
            )
        )
    for row in raw_words:
        start = _norm_time(row.get("start") if "start" in row else row.get("start_sec"))
        end = _norm_time(row.get("end") if "end" in row else row.get("end_sec"))
        speaker = _speaker_at_time(_midpoint(start, end), diarization_rows)
        words.append(
            TranscriptWord(
                word=str(row.get("word") or row.get("text") or ""),
                start_sec=start,
                end_sec=end,
                speaker_id=speaker,
                confidence=row.get("confidence"),
            )
        )
    return {
        "text": raw.get("text") or " ".join(segment.text for segment in segments).strip(),
        "segments": [segment.to_dict() for segment in segments],
        "words": [word.to_dict() for word in words],
    }


@dataclass
class ASRAlignmentService:
    artifact_store: LocalArtifactStore
    backend: ASRBackend = field(default_factory=create_asr_backend)

    def preflight(self) -> dict[str, Any]:
        return self.backend.preflight()

    def transcribe_to_artifact(
        self,
        *,
        audio_uri: str,
        output_asr_uri: str,
        diarization_uri: str | None = None,
    ) -> dict[str, Any]:
        validate_artifact_uri(audio_uri)
        validate_artifact_uri(output_asr_uri)
        audio_path = self.artifact_store.local_path(audio_uri)
        if not audio_path.exists():
            raise AegisVoxContractError(f"audio artifact not found: {audio_path}")
        diarization_rows: list[dict[str, Any]] = []
        if diarization_uri:
            validate_artifact_uri(diarization_uri)
            turns = load_diarization_turns(self.artifact_store.local_path(diarization_uri))
            diarization_rows = [turn.__dict__ for turn in turns]
        backend_metadata = self.backend.preflight()
        raw = self.backend.transcribe(audio_path)
        normalized = normalize_asr_output(raw, diarization_rows)
        payload = {
            "audio_uri": audio_uri,
            "backend": self.backend.backend_name,
            "backend_version": self.backend.backend_version,
            "model": backend_metadata.get("model") or raw.get("model"),
            "response_format": backend_metadata.get("response_format") or raw.get("response_format"),
            **normalized,
        }
        self.artifact_store.write_json(output_asr_uri, payload)
        return payload
