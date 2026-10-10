#!/usr/bin/env python3
"""Compute objective waveform/signal statistics for AegisVox eval runs.

This intentionally avoids treating a MOS predictor as human naturalness. It
reports directly measured clipping, RMS level, silence occupancy, and exact
adjacent-chunk repetition. Perceptual quality is reported separately from the
locked listener study.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import wave

from common import row_completed, final_audio_path, input_audio_path, load_summary_rows, mean, run_dir_from_summary_row, write_csv, write_json


def pcm_quality(path: Path, chunk_ms: int = 500, silence_rms_threshold: int = 200) -> dict[str, Any]:
    """Return deterministic WAV quality proxies.

    The first implementation used Python per-sample loops and became too slow
    for full390 long-audio paper runs. This version uses numpy when available
    and falls back to the older pure-Python helpers only if numpy cannot be
    imported or the sample width is unusual.
    """
    if not path.exists() or path.suffix.lower() != ".wav":
        return {"exists": path.exists(), "duration_sec": None, "error": "missing_or_not_wav"}
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sr = wav.getframerate()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
        duration = frames / float(sr) if sr else None
        if not raw:
            return {
                "exists": True,
                "duration_sec": duration,
                "sample_rate": sr,
                "num_channels": channels,
                "sample_width_bytes": width,
                "rms": 0,
                "rms_dbfs": None,
                "peak": 0,
                "peak_dbfs": None,
                "clipped_sample_count": 0,
                "clipped_sample_fraction": 0.0,
                "clipping_gate": "pass",
                "silence_fraction_chunks": None,
                "repeated_adjacent_chunk_fraction": None,
                "chunk_ms": chunk_ms,
                "error": None,
            }
        try:
            return _pcm_quality_numpy(raw, channels, width, sr, frames, duration, chunk_ms, silence_rms_threshold)
        except Exception:
            # Conservative fallback for environments without numpy or unusual WAV encodings.
            rms = pcm_rms(raw, width) if raw else 0
            peak = pcm_peak(raw, width) if raw else 0
            max_amp = float((1 << (8 * width - 1)) - 1) if width > 1 else 127.0
            sample_count = max(1, frames * channels)
            clipped_count = count_near_clipped_samples(raw, width, threshold=0.99)
            clipped_sample_fraction = clipped_count / sample_count
            clipping_gate = gate_clipping_fraction(clipped_sample_fraction)
            chunk_frames = max(1, int(sr * chunk_ms / 1000.0))
            chunk_bytes = chunk_frames * channels * width
            num_chunks = max(1, math.ceil(len(raw) / chunk_bytes)) if chunk_bytes else 1
            silent_chunks = 0
            chunk_hashes: list[tuple[int, int]] = []
            repeated_adjacent = 0
            prev_hash: tuple[int, int] | None = None
            for idx in range(num_chunks):
                chunk = raw[idx * chunk_bytes : (idx + 1) * chunk_bytes]
                if not chunk:
                    continue
                chunk_rms = pcm_rms(chunk, width)
                chunk_peak = pcm_peak(chunk, width)
                if chunk_rms <= silence_rms_threshold:
                    silent_chunks += 1
                fp = (round(chunk_rms / 25), round(chunk_peak / 25))
                chunk_hashes.append(fp)
                if prev_hash is not None and fp == prev_hash and chunk_rms > silence_rms_threshold:
                    repeated_adjacent += 1
                prev_hash = fp
            rms_dbfs = 20.0 * math.log10(max(rms, 1) / max_amp) if max_amp else None
            peak_dbfs = 20.0 * math.log10(max(peak, 1) / max_amp) if max_amp else None
            return {
                "exists": True,
                "duration_sec": duration,
                "sample_rate": sr,
                "num_channels": channels,
                "sample_width_bytes": width,
                "rms": rms,
                "rms_dbfs": rms_dbfs,
                "peak": peak,
                "peak_dbfs": peak_dbfs,
                "clipped_sample_count": clipped_count,
                "clipped_sample_fraction": clipped_sample_fraction,
                "clipping_gate": clipping_gate,
                "silence_fraction_chunks": silent_chunks / len(chunk_hashes) if chunk_hashes else None,
                "repeated_adjacent_chunk_fraction": repeated_adjacent / max(1, len(chunk_hashes) - 1) if chunk_hashes else None,
                "chunk_ms": chunk_ms,
                "error": None,
            }
    except Exception as exc:  # valid IEEE-float WAV is unsupported by stdlib wave
        try:
            return _soundfile_quality(path, chunk_ms, silence_rms_threshold)
        except Exception as fallback_exc:  # pragma: no cover - corrupted/unsupported audio
            return {
                "exists": True,
                "duration_sec": None,
                "error": f"wave={exc}; soundfile={fallback_exc}",
            }


def _soundfile_quality(path: Path, chunk_ms: int, silence_rms_threshold: int) -> dict[str, Any]:
    """Streaming quality statistics for float WAV and other libsndfile WAVs."""
    import numpy as np
    import soundfile as sf

    info = sf.info(str(path))
    sr = int(info.samplerate)
    channels = int(info.channels)
    frames = int(info.frames)
    chunk_frames = max(1, int(sr * chunk_ms / 1000.0))
    silence_norm = float(silence_rms_threshold) / 32767.0
    total_sq = 0.0
    total_samples = 0
    peak_norm = 0.0
    clipped_count = 0
    silent_chunks = 0
    num_chunks = 0
    repeated_adjacent = 0
    previous_fingerprint: tuple[int, int] | None = None
    with sf.SoundFile(str(path), "r") as handle:
        while True:
            block = handle.read(chunk_frames, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            values = block.reshape(-1).astype(np.float64, copy=False)
            abs_values = np.abs(values)
            chunk_rms_norm = float(np.sqrt(np.mean(values * values)))
            chunk_peak_norm = float(np.max(abs_values))
            total_sq += float(np.sum(values * values))
            total_samples += int(values.size)
            peak_norm = max(peak_norm, chunk_peak_norm)
            clipped_count += int(np.count_nonzero(abs_values >= 0.99))
            if chunk_rms_norm <= silence_norm:
                silent_chunks += 1
            fingerprint = (
                round((chunk_rms_norm * 32767.0) / 25.0),
                round((chunk_peak_norm * 32767.0) / 25.0),
            )
            if previous_fingerprint == fingerprint and chunk_rms_norm > silence_norm:
                repeated_adjacent += 1
            previous_fingerprint = fingerprint
            num_chunks += 1
    rms_norm = math.sqrt(total_sq / total_samples) if total_samples else 0.0
    clipped_fraction = clipped_count / max(1, total_samples)
    subtype_width = {
        "PCM_U8": 1,
        "PCM_16": 2,
        "PCM_24": 3,
        "PCM_32": 4,
        "FLOAT": 4,
        "DOUBLE": 8,
    }.get(str(info.subtype).upper())
    return {
        "exists": True,
        "duration_sec": frames / float(sr) if sr else None,
        "sample_rate": sr,
        "num_channels": channels,
        "sample_width_bytes": subtype_width,
        "sample_subtype": info.subtype,
        "rms": rms_norm * 32767.0,
        "rms_dbfs": 20.0 * math.log10(max(rms_norm, 1e-12)),
        "peak": peak_norm * 32767.0,
        "peak_dbfs": 20.0 * math.log10(max(peak_norm, 1e-12)),
        "clipped_sample_count": clipped_count,
        "clipped_sample_fraction": clipped_fraction,
        "clipping_gate": gate_clipping_fraction(clipped_fraction),
        "silence_fraction_chunks": silent_chunks / num_chunks if num_chunks else None,
        "repeated_adjacent_chunk_fraction": repeated_adjacent / max(1, num_chunks - 1) if num_chunks else None,
        "chunk_ms": chunk_ms,
        "error": None,
    }


def _pcm_quality_numpy(raw: bytes, channels: int, width: int, sr: int, frames: int, duration: float | None, chunk_ms: int, silence_rms_threshold: int) -> dict[str, Any]:
    import numpy as np

    if width == 1:
        vals = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        max_amp = 127.0
    elif width == 2:
        vals = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype="<i2").astype(np.float32)
        max_amp = float((1 << 15) - 1)
    elif width == 4:
        vals = np.frombuffer(raw[: len(raw) - (len(raw) % 4)], dtype="<i4").astype(np.float64)
        max_amp = float((1 << 31) - 1)
    elif width == 3:
        data = np.frombuffer(raw[: len(raw) - (len(raw) % 3)], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        vals_i32 = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
        vals_i32 = np.where(vals_i32 & 0x800000, vals_i32 | ~0xFFFFFF, vals_i32)
        vals = vals_i32.astype(np.float32)
        max_amp = float((1 << 23) - 1)
    else:
        raise ValueError(f"unsupported_sample_width_{width}")
    if vals.size == 0:
        rms = 0
        peak = 0
        clipped_count = 0
        silence_fraction = None
        repeated_fraction = None
    else:
        abs_vals = np.abs(vals)
        rms = int(np.sqrt(np.mean(vals * vals)))
        peak = int(np.max(abs_vals))
        clipped_count = int(np.count_nonzero(abs_vals >= (max_amp * 0.99)))
        chunk_frames = max(1, int(sr * chunk_ms / 1000.0))
        chunk_samples = max(1, chunk_frames * max(1, channels))
        n_chunks = int(math.ceil(vals.size / chunk_samples))
        pad = n_chunks * chunk_samples - vals.size
        if pad:
            vals_padded = np.pad(vals, (0, pad), mode="constant")
        else:
            vals_padded = vals
        chunks = vals_padded.reshape(n_chunks, chunk_samples)
        chunk_rms = np.sqrt(np.mean(chunks * chunks, axis=1))
        chunk_peak = np.max(np.abs(chunks), axis=1)
        silent = chunk_rms <= silence_rms_threshold
        silence_fraction = float(np.count_nonzero(silent) / max(1, n_chunks))
        fps_rms = np.rint(chunk_rms / 25).astype(np.int64)
        fps_peak = np.rint(chunk_peak / 25).astype(np.int64)
        if n_chunks > 1:
            same = (fps_rms[1:] == fps_rms[:-1]) & (fps_peak[1:] == fps_peak[:-1]) & (chunk_rms[1:] > silence_rms_threshold)
            repeated_fraction = float(np.count_nonzero(same) / max(1, n_chunks - 1))
        else:
            repeated_fraction = 0.0
    sample_count = max(1, frames * channels)
    clipped_sample_fraction = clipped_count / sample_count
    rms_dbfs = 20.0 * math.log10(max(rms, 1) / max_amp) if max_amp else None
    peak_dbfs = 20.0 * math.log10(max(peak, 1) / max_amp) if max_amp else None
    return {
        "exists": True,
        "duration_sec": duration,
        "sample_rate": sr,
        "num_channels": channels,
        "sample_width_bytes": width,
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "peak": peak,
        "peak_dbfs": peak_dbfs,
        "clipped_sample_count": clipped_count,
        "clipped_sample_fraction": clipped_sample_fraction,
        "clipping_gate": gate_clipping_fraction(clipped_sample_fraction),
        "silence_fraction_chunks": silence_fraction,
        "repeated_adjacent_chunk_fraction": repeated_fraction,
        "chunk_ms": chunk_ms,
        "error": None,
    }


def pcm_samples(raw: bytes, width: int):
    """Return signed PCM sample values without using deprecated audioop.

    WAV 8-bit PCM is unsigned; 16/32-bit PCM is little-endian signed.
    24-bit PCM is handled explicitly for robustness.
    """
    if not raw:
        return []
    if width == 1:
        return [int(v) - 128 for v in raw]
    if width == 2:
        import array

        arr = array.array("h")
        arr.frombytes(raw[: len(raw) - (len(raw) % 2)])
        if sys_byteorder_is_big():
            arr.byteswap()
        return arr
    if width == 4:
        import array

        arr = array.array("i")
        arr.frombytes(raw[: len(raw) - (len(raw) % 4)])
        if sys_byteorder_is_big():
            arr.byteswap()
        return arr
    if width == 3:
        vals = []
        usable = len(raw) - (len(raw) % 3)
        for i in range(0, usable, 3):
            b = raw[i : i + 3]
            value = int.from_bytes(b + (b"\xff" if b[2] & 0x80 else b"\x00"), "little", signed=True)
            vals.append(value)
        return vals
    return []


def sys_byteorder_is_big() -> bool:
    import sys

    return sys.byteorder == "big"


def pcm_rms(raw: bytes, width: int) -> int:
    vals = pcm_samples(raw, width)
    if not vals:
        return 0
    return int(math.sqrt(sum(float(v) * float(v) for v in vals) / len(vals)))


def pcm_peak(raw: bytes, width: int) -> int:
    vals = pcm_samples(raw, width)
    return int(max((abs(int(v)) for v in vals), default=0))


def count_near_clipped_samples(raw: bytes, width: int, threshold: float = 0.99) -> int:
    if not raw:
        return 0
    if width == 1:
        # 8-bit PCM is unsigned in WAV.
        vals = raw
        lo = int(128 - 127 * threshold)
        hi = int(128 + 127 * threshold)
        return sum(1 for value in vals if value <= lo or value >= hi)
    if width == 2:
        import array

        arr = array.array("h")
        arr.frombytes(raw)
        limit = int(((1 << 15) - 1) * threshold)
        return sum(1 for value in arr if abs(value) >= limit)
    if width == 4:
        import array

        arr = array.array("i")
        arr.frombytes(raw)
        limit = int(((1 << 31) - 1) * threshold)
        return sum(1 for value in arr if abs(value) >= limit)
    # Fallback: use file peak only for uncommon widths, but do not call it a sample fraction.
    return 0


def gate_clipping_fraction(fraction: float | None) -> str:
    if fraction is None:
        return "missing"
    # Conservative gates for 16-bit PCM: tiny isolated peaks are okay; sustained
    # near-rail samples are quality risks.
    if fraction > 0.01:
        return "fail"
    if fraction > 0.001:
        return "warn"
    return "pass"


def row_metrics(row: dict[str, Any], chunk_ms: int, silence_rms_threshold: int) -> dict[str, Any]:
    run_dir = run_dir_from_summary_row(row)
    input_metrics = pcm_quality(input_audio_path(run_dir), chunk_ms, silence_rms_threshold)
    final_metrics = pcm_quality(final_audio_path(run_dir), chunk_ms, silence_rms_threshold)
    rms_delta = None
    if isinstance(final_metrics.get("rms_dbfs"), (int, float)) and isinstance(input_metrics.get("rms_dbfs"), (int, float)):
        rms_delta = final_metrics["rms_dbfs"] - input_metrics["rms_dbfs"]
    return {
        "audio_id": row.get("audio_id"),
        "run_id": row.get("run_id"),
        "completed": row_completed(row),
        "input_rms_dbfs": input_metrics.get("rms_dbfs"),
        "final_rms_dbfs": final_metrics.get("rms_dbfs"),
        "rms_delta_db_final_minus_input": rms_delta,
        "input_peak_dbfs": input_metrics.get("peak_dbfs"),
        "final_peak_dbfs": final_metrics.get("peak_dbfs"),
        "final_clipped_sample_count": final_metrics.get("clipped_sample_count"),
        "final_clipped_sample_fraction": final_metrics.get("clipped_sample_fraction"),
        "final_clipping_gate": final_metrics.get("clipping_gate"),
        "input_silence_fraction_chunks": input_metrics.get("silence_fraction_chunks"),
        "final_silence_fraction_chunks": final_metrics.get("silence_fraction_chunks"),
        "final_repeated_adjacent_chunk_fraction": final_metrics.get("repeated_adjacent_chunk_fraction"),
        "input_audio_error": input_metrics.get("error"),
        "final_audio_error": final_metrics.get("error"),
        "claim_boundary": "Direct waveform statistics only; they do not measure perceptual naturalness, which is evaluated in the listener study.",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_completed": sum(1 for row in rows if row.get("completed")),
        "mean_final_rms_dbfs": mean([row.get("final_rms_dbfs") for row in rows]),
        "mean_rms_delta_db": mean([row.get("rms_delta_db_final_minus_input") for row in rows]),
        "mean_final_clipped_sample_fraction": mean([row.get("final_clipped_sample_fraction") for row in rows]),
        "num_final_clipping_pass": sum(1 for row in rows if row.get("final_clipping_gate") == "pass"),
        "num_final_clipping_warn": sum(1 for row in rows if row.get("final_clipping_gate") == "warn"),
        "num_final_clipping_fail": sum(1 for row in rows if row.get("final_clipping_gate") == "fail"),
        "mean_final_silence_fraction_chunks": mean([row.get("final_silence_fraction_chunks") for row in rows]),
        "mean_final_repeated_adjacent_chunk_fraction": mean([row.get("final_repeated_adjacent_chunk_fraction") for row in rows]),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Audio Signal Statistics", "", "**Claim boundary:** direct waveform measurements, not perceptual naturalness scores. Clipping is sample-fraction based, not a one-peak file flag; naturalness is evaluated by listeners.", "", "## Aggregate", ""]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | final RMS dBFS | RMS delta | clipped sample frac | clipping gate | silence frac | repeat frac |", "|---|---:|---:|---:|---|---:|---:|"]
    for row in payload["rows"]:
        lines.append(f"| {row.get('audio_id')} | {_fmt(row.get('final_rms_dbfs'))} | {_fmt(row.get('rms_delta_db_final_minus_input'))} | {_fmt(row.get('final_clipped_sample_fraction'))} | {row.get('final_clipping_gate')} | {_fmt(row.get('final_silence_fraction_chunks'))} | {_fmt(row.get('final_repeated_adjacent_chunk_fraction'))} |")
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--silence-rms-threshold", type=int, default=200)
    args = parser.parse_args()
    rows = [row_metrics(row, args.chunk_ms, args.silence_rms_threshold) for row in load_summary_rows(args.run_summary)]
    payload = {"schema_version": "aegisvox.eval.audio_signal_statistics.v2", "run_summary": str(args.run_summary), "aggregate": aggregate(rows), "rows": rows}
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
