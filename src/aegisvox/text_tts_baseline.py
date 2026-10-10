"""Text-only redaction + full TTS regeneration baseline.

This module implements the B1 paper baseline: consume the de-identified /
relexicalized text artifact, synthesize the complete transcript with a TTS
backend, and emit a regenerated full-audio WAV plus an auditable report. It is
not the main AegisVox pipeline and does not preserve source speaker identity.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore
from .text_deid import _openai_compatible_client_kwargs, _openai_compatible_api_key_from_env


class TextTTSBaselineError(RuntimeError):
    """Raised when the B1 text-only TTS baseline cannot be produced."""


class TextTTSBackend(Protocol):
    backend_name: str
    backend_version: str

    def synthesize(self, *, text: str, output_wav: Path, options: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class OpenAITextTTSBackend:
    """OpenAI TTS backend for full-transcript baseline regeneration."""

    model: str | None = None
    voice: str | None = None
    response_format: str = "wav"
    instructions: str | None = None
    backend_name: str = "openai-tts"
    backend_version: str = "openai-python-audio-speech"

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = os.environ.get("AEGISVOX_OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        if self.voice is None:
            self.voice = os.environ.get("AEGISVOX_OPENAI_TTS_VOICE", "marin")
        if self.instructions is None:
            self.instructions = os.environ.get(
                "AEGISVOX_OPENAI_TTS_INSTRUCTIONS",
                "Read as a clear, neutral clinical conversation transcript. Preserve punctuation and speak naturally.",
            )

    def _api_key(self) -> str:
        try:
            return _openai_compatible_api_key_from_env()
        except Exception as exc:
            raise TextTTSBaselineError(str(exc)) from exc

    def synthesize(self, *, text: str, output_wav: Path, options: dict[str, Any]) -> dict[str, Any]:
        self._api_key()
        if not text.strip():
            raise TextTTSBaselineError("cannot synthesize empty text")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - requires optional dep
            raise TextTTSBaselineError("openai package is not installed for B1 TTS baseline") from exc
        client = OpenAI(**_openai_compatible_client_kwargs(str(self.model)))
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        request_kwargs: dict[str, Any] = {
            "model": str(self.model),
            "voice": str(self.voice),
            "input": text,
            "response_format": self.response_format,
        }
        if self.instructions:
            request_kwargs["instructions"] = self.instructions
        try:
            with client.audio.speech.with_streaming_response.create(**request_kwargs) as response:
                response.stream_to_file(output_wav)
        except TypeError:
            # Some provider-compatible endpoints may not support instructions.
            request_kwargs.pop("instructions", None)
            with client.audio.speech.with_streaming_response.create(**request_kwargs) as response:
                response.stream_to_file(output_wav)
        except Exception as exc:  # pragma: no cover - requires live API
            raise TextTTSBaselineError(f"OpenAI TTS request failed: {exc}") from exc
        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise TextTTSBaselineError("OpenAI TTS completed without producing a non-empty WAV")
        return {
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "model": self.model,
            "voice": self.voice,
            "response_format": self.response_format,
            "num_chars": len(text),
        }


@dataclass
class ChatterboxTextTTSBackend:
    """Open-source Chatterbox/Chatterbox-Turbo backend for B1 baseline regeneration.

    B1 is a text-only baseline, so it intentionally uses one fixed reference
    voice for the regenerated transcript rather than preserving source speakers.
    The reference must be explicit to keep the baseline auditable.
    """

    reference_wav: str | Path | None = None
    variant: str = "turbo"
    device: str = "auto"
    exaggeration: float = 0.35
    cfg_weight: float | None = None
    backend_name: str = "chatterbox-turbo-tts"
    backend_version: str = "chatterbox-tts"
    _model: Any | None = field(default=None, init=False, repr=False)
    _model_device: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.reference_wav is None:
            self.reference_wav = os.environ.get("AEGISVOX_B1_CHATTERBOX_REFERENCE_WAV")
        if not self.reference_wav:
            raise TextTTSBaselineError(
                "Chatterbox B1 TTS requires --chatterbox-reference-wav or AEGISVOX_B1_CHATTERBOX_REFERENCE_WAV"
            )
        self.reference_wav = Path(self.reference_wav)
        if not self.reference_wav.exists() or self.reference_wav.stat().st_size == 0:
            raise TextTTSBaselineError(f"Chatterbox reference WAV is missing or empty: {self.reference_wav}")
        normalized_variant = str(self.variant or "turbo").strip().lower().replace("-", "_")
        if normalized_variant not in {"turbo", "standard", "chatterbox", "chatterbox_turbo"}:
            raise TextTTSBaselineError(f"unsupported Chatterbox variant: {self.variant!r}")
        self.variant = "turbo" if normalized_variant in {"turbo", "chatterbox_turbo"} else "standard"
        self.backend_name = "chatterbox-turbo-tts" if self.variant == "turbo" else "chatterbox-tts"

    def _load_model(self, resolved_device: str) -> Any:
        if self._model is not None and self._model_device == resolved_device:
            return self._model
        try:
            if self.variant == "turbo":
                try:
                    from chatterbox.tts_turbo import ChatterboxTurboTTS as ChatterboxModel
                except ImportError as exc:  # pragma: no cover - depends on installed chatterbox release
                    raise TextTTSBaselineError("chatterbox-tts with Turbo support is not installed") from exc
            else:
                from chatterbox.tts import ChatterboxTTS as ChatterboxModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise TextTTSBaselineError("chatterbox-tts is required for B1 Chatterbox TTS") from exc
        self._model = ChatterboxModel.from_pretrained(device=resolved_device)
        self._model_device = resolved_device
        return self._model

    def synthesize(self, *, text: str, output_wav: Path, options: dict[str, Any]) -> dict[str, Any]:
        if not text.strip():
            raise TextTTSBaselineError("cannot synthesize empty text")
        try:
            import numpy as np
            import soundfile as sf
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise TextTTSBaselineError("torch, numpy, and soundfile are required for B1 Chatterbox TTS") from exc

        resolved_device = str(options.get("device") or self.device)
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        try:
            model = self._load_model(resolved_device)
            kwargs: dict[str, Any] = {
                "text": text,
                "audio_prompt_path": str(self.reference_wav),
                "exaggeration": float(options.get("exaggeration", self.exaggeration)),
            }
            cfg_weight = options.get("cfg_weight", self.cfg_weight)
            if cfg_weight is not None:
                kwargs["cfg_weight"] = float(cfg_weight)
            wav = model.generate(**kwargs)
            wav_np = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
            wav_np = np.asarray(wav_np, dtype="float32").squeeze()
            if wav_np.ndim > 1:
                wav_np = wav_np.mean(axis=0)
            if wav_np.size == 0:
                raise TextTTSBaselineError("Chatterbox generated empty audio")
            sf.write(str(output_wav), wav_np, int(model.sr), subtype="PCM_16")
        except Exception as exc:  # noqa: BLE001 - external model path
            if isinstance(exc, TextTTSBaselineError):
                raise
            raise TextTTSBaselineError(f"Chatterbox TTS request failed: {exc}") from exc
        if not output_wav.exists() or output_wav.stat().st_size == 0:
            raise TextTTSBaselineError("Chatterbox TTS completed without producing a non-empty WAV")
        return {
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "variant": self.variant,
            "device": resolved_device,
            "reference_wav": str(self.reference_wav),
            "num_chars": len(text),
            "sample_rate_hz": int(model.sr),
            "claim_boundary": "B1 uses one fixed reference voice and intentionally does not preserve source speaker identities.",
        }


def _deid_segments(deid: dict[str, Any]) -> list[dict[str, Any]]:
    rows = deid.get("deidentified_segments")
    if isinstance(rows, list) and rows:
        return [row for row in rows if isinstance(row, dict) and str(row.get("text") or "").strip()]
    text = str(deid.get("deidentified_text") or "").strip()
    if text:
        return [{"text": text, "start_sec": None, "end_sec": None, "speaker_id": None}]
    raise TextTTSBaselineError("deid_text artifact has no deidentified_segments or deidentified_text")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def transcript_chunks_from_deid(deid: dict[str, Any], *, max_chars: int = 3500) -> list[dict[str, Any]]:
    """Build TTS chunks from deidentified transcript, preserving segment order."""

    if max_chars < 200:
        raise TextTTSBaselineError("max_chars must be >= 200")
    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_segment_indices: list[int] = []
    current_start: Any = None
    current_end: Any = None

    def flush() -> None:
        nonlocal current, current_segment_indices, current_start, current_end
        text = _clean_text(" ".join(current))
        if text:
            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "text": text,
                    "num_chars": len(text),
                    "source_segment_indices": list(current_segment_indices),
                    "source_start_sec": current_start,
                    "source_end_sec": current_end,
                }
            )
        current = []
        current_segment_indices = []
        current_start = None
        current_end = None

    for idx, row in enumerate(_deid_segments(deid)):
        text = _clean_text(str(row.get("text") or ""))
        if not text:
            continue
        if current and len(_clean_text(" ".join(current + [text]))) > max_chars:
            flush()
        if not current:
            current_start = row.get("start_sec")
        current.append(text)
        current_segment_indices.append(idx)
        current_end = row.get("end_sec")
        # A single pathological segment can exceed max_chars; split on sentence-ish boundaries.
        if len(_clean_text(" ".join(current))) > max_chars:
            long_text = _clean_text(" ".join(current))
            current = []
            current_segment_indices = []
            parts = re.split(r"(?<=[.!?])\s+", long_text)
            for part in parts:
                if not part:
                    continue
                if current and len(_clean_text(" ".join(current + [part]))) > max_chars:
                    flush()
                    current_start = row.get("start_sec")
                current.append(part)
                current_segment_indices.append(idx)
                current_end = row.get("end_sec")
    flush()
    if not chunks:
        raise TextTTSBaselineError("no non-empty TTS chunks were generated")
    return chunks


def _read_wav_bytes(path: Path) -> tuple[tuple[int, int, int], bytes, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        return (channels, sample_width, frame_rate), frames, wav.getnframes()


def concatenate_wavs(chunk_paths: list[Path], output_wav: Path, *, gap_ms: int = 250) -> dict[str, Any]:
    if not chunk_paths:
        raise TextTTSBaselineError("no chunk WAVs to concatenate")
    first_params, first_frames, first_num_frames = _read_wav_bytes(chunk_paths[0])
    channels, sample_width, frame_rate = first_params
    gap_frames = max(0, int(frame_rate * gap_ms / 1000.0))
    silence = b"\x00" * gap_frames * channels * sample_width
    total_frames = first_num_frames
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(sample_width)
        out.setframerate(frame_rate)
        out.writeframes(first_frames)
        for path in chunk_paths[1:]:
            params, frames, nframes = _read_wav_bytes(path)
            if params != first_params:
                raise TextTTSBaselineError(
                    f"TTS chunks have inconsistent WAV params: {path} has {params}, expected {first_params}"
                )
            if gap_frames:
                out.writeframes(silence)
                total_frames += gap_frames
            out.writeframes(frames)
            total_frames += nframes
    duration_sec = total_frames / float(frame_rate) if frame_rate else None
    return {
        "sample_rate_hz": frame_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "num_frames": total_frames,
        "duration_sec": duration_sec,
        "gap_ms": gap_ms,
    }


def build_b1_text_tts_baseline_to_artifacts(
    *,
    artifact_store: LocalArtifactStore,
    deid_text_uri: str,
    output_audio_uri: str,
    output_report_uri: str,
    backend: TextTTSBackend | None = None,
    chunk_prefix_uri: str | None = None,
    max_chars_per_chunk: int = 3500,
    gap_ms: int = 250,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_artifact_uri(deid_text_uri)
    validate_artifact_uri(output_audio_uri)
    validate_artifact_uri(output_report_uri)
    backend = backend or OpenAITextTTSBackend()
    options = dict(options or {})
    deid = artifact_store.read_json(deid_text_uri)
    if not isinstance(deid, dict):
        raise TextTTSBaselineError("deid_text artifact must be a JSON object")
    chunks = transcript_chunks_from_deid(deid, max_chars=max_chars_per_chunk)
    if chunk_prefix_uri is None:
        parsed_output = output_audio_uri.rstrip("/")
        chunk_prefix_uri = parsed_output.rsplit("/", 1)[0] + "/b1_tts_chunks"
    validate_artifact_uri(chunk_prefix_uri.rstrip("/") + "/chunk_000000.wav")

    chunk_records: list[dict[str, Any]] = []
    chunk_paths: list[Path] = []
    for chunk in chunks:
        idx = int(chunk["chunk_index"])
        chunk_uri = f"{chunk_prefix_uri.rstrip('/')}/chunk_{idx:06d}.wav"
        chunk_path = artifact_store.local_path(chunk_uri)
        meta = backend.synthesize(text=str(chunk["text"]), output_wav=chunk_path, options=options)
        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            raise TextTTSBaselineError(f"TTS backend did not create chunk {idx}")
        chunk_paths.append(chunk_path)
        chunk_records.append({**chunk, "audio_uri": chunk_uri, "tts_metadata": meta})

    output_path = artifact_store.local_path(output_audio_uri)
    audio_meta = concatenate_wavs(chunk_paths, output_path, gap_ms=gap_ms)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise TextTTSBaselineError("B1 baseline did not create a non-empty final WAV")
    edits = deid.get("edits") if isinstance(deid.get("edits"), list) else []
    report = {
        "schema_version": "aegisvox.b1_text_tts_baseline.v1",
        "baseline_id": "B1_text_only_redaction_full_tts",
        "deid_text_uri": deid_text_uri,
        "output_audio_uri": output_audio_uri,
        "output_report_uri": output_report_uri,
        "policy_mode": deid.get("policy_mode"),
        "num_deid_edits": len(edits),
        "num_tts_chunks": len(chunk_records),
        "total_text_chars": sum(int(row["num_chars"]) for row in chunk_records),
        "audio": audio_meta,
        "chunks": chunk_records,
        "backend": getattr(backend, "backend_name", "unknown"),
        "backend_version": getattr(backend, "backend_version", "unknown"),
        "claim_boundary": (
            "B1 baseline only: text PHI de-id/relexicalization followed by full TTS regeneration. "
            "It does not preserve speaker identity, turn acoustics, overlap, or natural conversation timing."
        ),
    }
    artifact_store.write_json(output_report_uri, report)
    return report
