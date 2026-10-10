#!/usr/bin/env python3
"""Shared helpers for AegisVox public-eval metric scripts.

The helpers intentionally avoid heavy dependencies so protocol metrics can run
locally or on ORD without model downloads. Model-backed paper metrics should be
implemented as opt-in scripts that fail loudly when dependencies are missing.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import wave
from typing import Any

import yaml

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
SRT_BLOCK_RE = re.compile(r"(?ms)^\s*\d+\s*\n\s*[^\n]+\s*-->\s*[^\n]+\s*\n(.*?)(?=\n\s*\n|\Z)")


def load_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten(row) for row in rows]
    keys = sorted({k for row in flat for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)
        else:
            out[key] = value
    return out


def mean(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return sum(nums) / len(nums) if nums else None


def percentile(values: list[Any], p: float) -> float | None:
    nums = sorted(float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v)))
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    rank = (len(nums) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return nums[lo]
    frac = rank - lo
    return nums[lo] * (1.0 - frac) + nums[hi] * frac


def stdev(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return statistics.stdev(nums) if len(nums) >= 2 else None


def distribution_summary(values: list[Any]) -> dict[str, Any]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]
    return {
        "n": len(nums),
        "mean": mean(nums),
        "median": percentile(nums, 0.50),
        "p05": percentile(nums, 0.05),
        "p25": percentile(nums, 0.25),
        "p75": percentile(nums, 0.75),
        "p95": percentile(nums, 0.95),
        "min": min(nums) if nums else None,
        "max": max(nums) if nums else None,
    }


def source_dataset_id(row: dict[str, Any]) -> str:
    """Return the locked public390 dataset bucket without relying on filenames.

    Most final metric rows already carry ``source_dataset_id``.  Older metric
    producers only persisted ``audio_id``/``run_id``, so the final statistics
    layer uses the frozen public390 naming contract as a fail-transparent
    fallback rather than silently dropping dataset stratification.
    """
    explicit = row.get("source_dataset_id")
    if explicit:
        return str(explicit)
    identifier = str(row.get("audio_id") or row.get("run_id") or "").lower()
    if "primock57" in identifier:
        return "primock57_mixed"
    if "fareez" in identifier:
        return "fareez_respiratory_interviews"
    if "simsamu" in identifier:
        return "simsamu"
    return "unknown"


def source_language(row: dict[str, Any]) -> str:
    explicit = str(row.get("source_language") or row.get("language") or "").strip().lower().replace("_", "-")
    if explicit:
        if explicit in {"english", "eng"} or explicit.startswith("en"):
            return "en"
        if explicit in {"french", "fra", "fre"} or explicit.startswith("fr"):
            return "fr"
        return explicit.split("-", 1)[0]
    dataset = source_dataset_id(row)
    if dataset == "simsamu":
        return "fr"
    if dataset in {"primock57_mixed", "fareez_respiratory_interviews"}:
        return "en"
    return "unknown"


def wav_info(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or path.suffix.lower() != ".wav":
        return {"duration_sec": None, "sample_rate": None, "num_channels": None, "num_frames": None, "sample_width_bytes": None}
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            sr = w.getframerate()
            return {
                "duration_sec": frames / float(sr) if sr else None,
                "sample_rate": sr,
                "num_channels": w.getnchannels(),
                "num_frames": frames,
                "sample_width_bytes": w.getsampwidth(),
            }
    except Exception:
        # Python's stdlib wave reader rejects valid IEEE-float WAV (format 3),
        # which is what the corrected F5 renderer writes.  SoundFile reads the
        # header without decoding the full long-form conversation.
        try:
            import soundfile as sf

            info = sf.info(str(path))
            subtype_width = {
                "PCM_U8": 1,
                "PCM_16": 2,
                "PCM_24": 3,
                "PCM_32": 4,
                "FLOAT": 4,
                "DOUBLE": 8,
            }.get(str(info.subtype).upper())
            return {
                "duration_sec": info.frames / float(info.samplerate) if info.samplerate else None,
                "sample_rate": info.samplerate,
                "num_channels": info.channels,
                "num_frames": info.frames,
                "sample_width_bytes": subtype_width,
            }
        except Exception:
            return {"duration_sec": None, "sample_rate": None, "num_channels": None, "num_frames": None, "sample_width_bytes": None}


def config_inputs(config_path: Path | str) -> dict[str, dict[str, Any]]:
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(cfg.get("audio_inputs", [])):
        if "audio_id" in row:
            out[str(row["audio_id"])] = row
        out[f"__ordinal__:{idx}"] = row
    return out


def config_input_for_summary_row(row: dict[str, Any], config_by_audio: dict[str, dict[str, Any]]) -> dict[str, Any]:
    audio_id = str(row.get("audio_id") or "")
    if audio_id and audio_id in config_by_audio:
        return config_by_audio[audio_id]
    ordinal = row.get("ordinal")
    if ordinal is not None:
        cfg = config_by_audio.get(f"__ordinal__:{ordinal}")
        if cfg:
            return cfg
    return {}


def audio_id_for_summary_row(row: dict[str, Any], config_by_audio: dict[str, dict[str, Any]] | None = None) -> str:
    audio_id = str(row.get("audio_id") or "")
    if audio_id:
        return audio_id
    if config_by_audio is not None:
        cfg = config_input_for_summary_row(row, config_by_audio)
        if cfg.get("audio_id"):
            return str(cfg["audio_id"])
    run_id = str(row.get("run_id") or "")
    if run_id:
        parts = run_id.split("_")
        if len(parts) >= 3 and parts[-2].isdigit():
            return "_".join(parts[-3:])
    return ""


def row_completed(row: dict[str, Any]) -> bool:
    if row.get("completed") is not None:
        return bool(row.get("completed"))
    if row.get("run_status") == "completed":
        return True
    if row.get("status") in {"completed", "failed"} and row.get("returncode") is not None:
        return True
    return False


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path, {})
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"summary missing rows: {path}")
    return rows


def run_dir_from_summary_row(row: dict[str, Any]) -> Path:
    if row.get("final_audio"):
        return Path(row["final_audio"]).parent
    if row.get("run_root"):
        return Path(row["run_root"])
    run_id = row.get("run_id")
    if run_id:
        artifact_root = Path(
            os.environ.get("AEGISVOX_ARTIFACT_ROOT")
            or os.environ.get("ARTIFACT_ROOT")
            or "${AEGISVOX_DATA_ROOT}/artifacts"
        )
        candidate = artifact_root / "aegisvox" / "runs" / str(run_id)
        if candidate.exists():
            return candidate
    raise ValueError(f"row lacks final_audio/run_root: {row}")


def final_audio_path(run_dir: Path) -> Path:
    for name in ["final_full_pipeline.wav", "final_sanitized.wav", "changed_identity.wav"]:
        path = run_dir / name
        if path.exists():
            return path
    return run_dir / "final_full_pipeline.wav"


def input_audio_path(run_dir: Path) -> Path:
    for name in ["input.wav", "original.wav", "source.wav"]:
        path = run_dir / name
        if path.exists():
            return path
    return run_dir / "input.wav"


def norm_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(TOKEN_RE.findall(text))


def tokens(text: str) -> list[str]:
    normalized = norm_text(text)
    return normalized.split() if normalized else []


def chars(text: str) -> list[str]:
    return list(norm_text(text).replace(" ", ""))


def levenshtein(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (0 if r == h else 1)))
        prev = cur
    return prev[-1]


def bitparallel_levenshtein(ref: list[Any], hyp: list[Any]) -> int:
    """Exact Levenshtein distance using Myers' arbitrary-length bit vectors.

    Python integers provide the bit vector, so this remains dependency-free but
    avoids the quadratic Python-loop bottleneck for long conversation
    transcripts. Elements only need to be hashable; it therefore works for
    word-token and character sequences.
    """
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    # The bit-vector width is the pattern length; keep the shorter sequence as
    # the pattern because edit distance is symmetric.
    if len(ref) > len(hyp):
        ref, hyp = hyp, ref
    width = len(ref)
    mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    equality_masks: dict[Any, int] = {}
    for idx, item in enumerate(ref):
        equality_masks[item] = equality_masks.get(item, 0) | (1 << idx)
    positive = mask
    negative = 0
    score = width
    for item in hyp:
        equal = equality_masks.get(item, 0)
        xv = equal | negative
        xh = ((((equal & positive) + positive) ^ positive) | equal) & mask
        ph = (negative | ~(xh | positive)) & mask
        mh = positive & xh
        if ph & high_bit:
            score += 1
        if mh & high_bit:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        positive = (mh | ~(xv | ph)) & mask
        negative = ph & xv
    return score


def exact_levenshtein(
    ref: list[Any],
    hyp: list[Any],
    *,
    dp_cell_threshold: int = 2_000_000,
) -> tuple[int, str]:
    """Return exact distance and the implementation used.

    Small pairs use the simple dynamic program; larger pairs switch to the
    exact bit-parallel implementation rather than skipping paper rows.
    """
    cells = (len(ref) + 1) * (len(hyp) + 1)
    if cells <= dp_cell_threshold:
        return levenshtein(ref, hyp), "dynamic_programming"
    return bitparallel_levenshtein(ref, hyp), "myers_bitparallel"


def read_reference_transcript(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = load_json(path, {})
        if isinstance(data, dict) and isinstance(data.get("utterances"), list):
            return " ".join(str(u.get("text") or "") for u in data["utterances"])
        if isinstance(data, dict):
            for key in ["text", "transcript", "transcription"]:
                if isinstance(data.get(key), str):
                    return data[key]
            return ""
    if suffix == ".srt":
        raw = path.read_text(errors="replace")
        chunks = []
        for match in SRT_BLOCK_RE.finditer(raw):
            chunks.append(re.sub(r"<[^>]+>", " ", match.group(1)).replace("\n", " "))
        return " ".join(chunks) if chunks else raw
    return path.read_text(errors="replace")


def load_asr_text(run_dir: Path) -> str:
    candidates = sorted(run_dir.glob("asr*.json")) + sorted(run_dir.glob("*asr*.json"))
    preferred = [
        run_dir / "asr.input_direct_input_v1.json",
        run_dir / "asr.changed_identity.json",
        run_dir / "asr.final.json",
    ]
    for path in preferred + candidates:
        data = load_json(path, None)
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            return data["text"]
    return ""


def parse_rttm(path: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path is None or not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0].upper() != "SPEAKER":
            continue
        start = float(parts[3])
        dur = float(parts[4])
        rows.append({"start_sec": start, "end_sec": start + dur, "speaker_id": parts[7]})
    return rows


def predicted_diarization_segments(run_dir: Path) -> list[dict[str, Any]]:
    # Current paper-scale direct-input/no-overlap runs write cleaned segment
    # artifacts rather than the older diarization.raw.json name.  Prefer the
    # exact exclusive-speaker timeline consumed by downstream conversion, then
    # fall back to older diarization artifacts for backwards compatibility.
    for name in [
        "segments.cleaned.direct_input_v1.json",
        "segments.cleaned.prepare_direct.json",
        "segments.cleaned.json",
        "diarization.direct_input_v1.json",
        "diarization.prepare_direct.json",
        "diarization.raw.json",
        "diarization.json",
    ]:
        data = load_json(run_dir / name, {})
        segs = data.get("segments") if isinstance(data, dict) else []
        out: list[dict[str, Any]] = []
        for seg in segs or []:
            try:
                out.append({
                    "start_sec": float(seg["start_sec"]),
                    "end_sec": float(seg["end_sec"]),
                    "speaker_id": str(seg.get("speaker_id") or ""),
                })
            except Exception:
                continue
        if out:
            return out
    return []


def speech_duration(intervals: list[dict[str, Any]]) -> float:
    return sum(max(0.0, float(x["end_sec"]) - float(x["start_sec"])) for x in intervals)


def frame_activity(intervals: list[dict[str, Any]], duration: float, frame_sec: float) -> set[int]:
    active: set[int] = set()
    if duration <= 0:
        return active
    max_frame = int(math.ceil(duration / frame_sec))
    for seg in intervals:
        start = max(0, int(math.floor(float(seg["start_sec"]) / frame_sec)))
        end = min(max_frame, int(math.ceil(float(seg["end_sec"]) / frame_sec)))
        active.update(range(start, end))
    return active


def frame_speaker_sets(intervals: list[dict[str, Any]], duration: float, frame_sec: float) -> list[set[str]]:
    max_frame = int(math.ceil(duration / frame_sec)) if duration > 0 else 0
    frames = [set() for _ in range(max_frame)]
    for seg in intervals:
        speaker = str(seg.get("speaker_id") or "")
        if not speaker:
            continue
        start = max(0, int(math.floor(float(seg["start_sec"]) / frame_sec)))
        end = min(max_frame, int(math.ceil(float(seg["end_sec"]) / frame_sec)))
        for frame_idx in range(start, end):
            frames[frame_idx].add(speaker)
    return frames
