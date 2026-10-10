"""Public non-sensitive smoke audio dataset helpers for AegisVox.

The default source is a tiny public LibriSpeech-derived ASR dummy dataset. It is
public read speech, not clinical, and is used only for engineering smoke tests.
The helpers write local PCM WAV files and a small two-speaker stitched sample so
runtime services can be exercised without PHI.
"""

from __future__ import annotations

import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ID = "omar47/dummy_en-asr"
DEFAULT_SPLIT = "validation"
DEFAULT_OUTPUT_DIR = "data/public_smoke_audio/dummy_en_asr"
DEFAULT_NUM_SAMPLES = 8


class PublicSmokeAudioError(RuntimeError):
    """Raised when public smoke audio cannot be downloaded or written."""


@dataclass(frozen=True)
class PublicAudioRow:
    sample_id: str
    speaker_id: str
    path: str
    sample_rate_hz: int
    duration_seconds: float
    text: str
    source_dataset: str
    license: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "speaker_id": self.speaker_id,
            "path": self.path,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_seconds": self.duration_seconds,
            "text": self.text,
            "source_dataset": self.source_dataset,
            "license": self.license,
            "provenance": dict(self.provenance),
        }


def _require_imports() -> tuple[Any, Any, Any]:
    try:
        import datasets  # type: ignore
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as exc:  # noqa: BLE001 - fail loudly with install hint
        raise PublicSmokeAudioError(
            "public smoke audio download requires datasets, numpy, and soundfile; "
            "install them in the runtime environment before downloading public samples"
        ) from exc
    return datasets, np, sf


def _as_mono_float32(audio: dict[str, Any], np: Any, sf: Any) -> tuple[Any, int]:
    array = audio.get("array")
    sampling_rate = int(audio.get("sampling_rate") or 0)
    if array is None:
        if audio.get("bytes"):
            array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        elif audio.get("path"):
            array, sampling_rate = sf.read(str(audio["path"]), dtype="float32", always_2d=False)
    if array is None or int(sampling_rate) <= 0:
        raise PublicSmokeAudioError("dataset audio row is missing decoded array/path/bytes or sampling_rate")
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1) if arr.shape[0] >= arr.shape[1] else arr.mean(axis=0)
    if arr.ndim != 1:
        raise PublicSmokeAudioError(f"expected mono/2D audio array, got shape {arr.shape}")
    return arr, sampling_rate


def _write_wav(path: str | Path, samples: Any, sample_rate: int, sf: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def _wav_duration(path: str | Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        rate = reader.getframerate()
    return rate, (float(frames) / float(rate)) if rate else 0.0


def _speaker_id(row: dict[str, Any], fallback: str) -> str:
    for key in ("speaker_id", "speaker", "speaker_name", "chapter_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    file_value = row.get("file") or row.get("id") or fallback
    stem = Path(str(file_value)).stem
    return stem.split("-")[0] if "-" in stem else stem


def _sample_id(row: dict[str, Any], index: int) -> str:
    for key in ("id", "file"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return Path(str(value)).stem.replace("/", "_")
    return f"sample_{index:04d}"


def _text(row: dict[str, Any]) -> str:
    for key in ("text", "sentence", "transcription"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def build_two_speaker_stitched_wav(*, rows: list[PublicAudioRow], output_path: str | Path, silence_seconds: float = 0.35) -> dict[str, Any]:
    """Create a simple alternating two-speaker public smoke WAV from existing rows."""

    if len(rows) < 2:
        raise PublicSmokeAudioError("need at least two public samples to build a stitched smoke conversation")
    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise PublicSmokeAudioError("stitching public smoke audio requires numpy and soundfile") from exc
    selected: list[PublicAudioRow] = []
    seen_speakers: set[str] = set()
    for row in rows:
        if row.speaker_id not in seen_speakers:
            selected.append(row)
            seen_speakers.add(row.speaker_id)
        if len(selected) == 2:
            break
    if len(selected) < 2:
        selected = rows[:2]
    audio_parts = []
    transcript_turns = []
    sample_rate: int | None = None
    cursor = 0.0
    for idx, row in enumerate(selected):
        samples, rate = sf.read(row.path, dtype="float32", always_2d=False)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if sample_rate is None:
            sample_rate = int(rate)
        if int(rate) != sample_rate:
            raise PublicSmokeAudioError("selected public smoke samples have mixed sample rates; resampling is not implemented")
        audio_parts.append(samples)
        duration = float(len(samples)) / float(sample_rate)
        transcript_turns.append(
            {
                "speaker": f"speaker_{idx + 1}",
                "source_speaker_id": row.speaker_id,
                "sample_id": row.sample_id,
                "start_seconds": round(cursor, 6),
                "end_seconds": round(cursor + duration, 6),
                "text": row.text,
            }
        )
        cursor += duration
        if idx < len(selected) - 1:
            silence = np.zeros(int(sample_rate * silence_seconds), dtype=np.float32)
            audio_parts.append(silence)
            cursor += silence_seconds
    assert sample_rate is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
    sf.write(str(output_path), stitched, sample_rate, subtype="PCM_16")
    return {
        "path": str(output_path),
        "sample_rate_hz": sample_rate,
        "duration_seconds": round(float(len(stitched)) / float(sample_rate), 6),
        "turns": transcript_turns,
        "source": "stitched_from_public_dummy_en_asr",
    }


def download_public_smoke_audio(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = DEFAULT_SPLIT,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    license: str = "LibriSpeech/OpenSLR public corpus; engineering smoke only",
) -> dict[str, Any]:
    """Download a tiny public non-sensitive audio set and write a manifest."""

    if num_samples <= 0:
        raise PublicSmokeAudioError("num_samples must be positive")
    datasets, np, sf = _require_imports()
    output_dir = Path(output_dir).expanduser().resolve()
    wav_dir = output_dir / "wavs"
    try:
        dataset = datasets.load_dataset(dataset_id, split=split)
    except Exception as exc:  # noqa: BLE001
        raise PublicSmokeAudioError(f"failed to load public smoke dataset {dataset_id!r} split {split!r}: {exc}") from exc
    candidates: list[tuple[int, dict[str, Any], Any, int, str, str]] = []
    max_scan = max(num_samples * 5, num_samples)
    for idx, row in enumerate(dataset):
        if idx >= max_scan and len(candidates) >= num_samples:
            break
        audio = row.get("audio")
        if not isinstance(audio, dict):
            continue
        samples, sample_rate = _as_mono_float32(audio, np, sf)
        sample_id = _sample_id(row, idx)
        speaker = _speaker_id(row, sample_id)
        candidates.append((idx, row, samples, sample_rate, sample_id, speaker))
        if idx + 1 >= max_scan:
            break
    if not candidates:
        raise PublicSmokeAudioError(f"no decodable public samples found in {dataset_id}")

    selected: list[tuple[int, dict[str, Any], Any, int, str, str]] = []
    seen_speakers: set[str] = set()
    for candidate in candidates:
        speaker = candidate[5]
        if speaker not in seen_speakers:
            selected.append(candidate)
            seen_speakers.add(speaker)
        if len(selected) >= num_samples:
            break
    for candidate in candidates:
        if len(selected) >= num_samples:
            break
        if candidate not in selected:
            selected.append(candidate)

    rows: list[PublicAudioRow] = []
    for output_idx, (idx, row, samples, sample_rate, sample_id, speaker) in enumerate(selected[:num_samples]):
        wav_path = wav_dir / f"{output_idx:03d}_{sample_id}.wav"
        _write_wav(wav_path, samples, sample_rate, sf)
        rate, duration = _wav_duration(wav_path)
        rows.append(
            PublicAudioRow(
                sample_id=sample_id,
                speaker_id=speaker,
                path=str(wav_path),
                sample_rate_hz=rate,
                duration_seconds=round(duration, 6),
                text=_text(row),
                source_dataset=dataset_id,
                license=license,
                provenance={"dataset_id": dataset_id, "split": split, "row_index": idx, "purpose": "non_sensitive_runtime_smoke"},
            )
        )
    if len(rows) < min(num_samples, 2):
        raise PublicSmokeAudioError(f"only exported {len(rows)} public samples from {dataset_id}; expected at least {min(num_samples, 2)}")
    manifest_path = output_dir / "public_smoke_manifest.jsonl"
    _write_jsonl(manifest_path, [row.to_dict() for row in rows])
    stitched = build_two_speaker_stitched_wav(rows=rows, output_path=output_dir / "two_speaker_public_smoke.wav")
    stitched_manifest_path = output_dir / "two_speaker_public_smoke.json"
    stitched_manifest_path.write_text(json.dumps(stitched, indent=2, sort_keys=True) + "\n")
    return {
        "ok": True,
        "dataset_id": dataset_id,
        "split": split,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "stitched_audio": stitched["path"],
        "stitched_manifest": str(stitched_manifest_path),
        "num_samples": len(rows),
        "speakers": sorted({row.speaker_id for row in rows}),
        "total_duration_seconds": round(sum(row.duration_seconds for row in rows), 6),
        "claim_boundary": "Public smoke audio is non-sensitive engineering data; it is not clinical evidence.",
    }
