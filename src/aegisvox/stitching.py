"""Timeline-preserving stitching/DSP for converted segments."""

from __future__ import annotations

import json
import math
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import AegisVoxContractError, validate_artifact_uri
from .segmentation import wav_duration_sec
from .storage import LocalArtifactStore


class StitchingError(RuntimeError):
    """Raised when audio reconstruction cannot proceed."""


@dataclass(frozen=True)
class WavFormat:
    nchannels: int
    sampwidth: int
    framerate: int


def _wav_format(path: Path) -> WavFormat:
    with wave.open(str(path), "rb") as wav:
        return WavFormat(wav.getnchannels(), wav.getsampwidth(), wav.getframerate())


def _read_wav_frames(path: Path) -> tuple[WavFormat, bytes, int]:
    with wave.open(str(path), "rb") as wav:
        fmt = WavFormat(wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
        nframes = wav.getnframes()
        return fmt, wav.readframes(nframes), nframes


def _silence_bytes(nframes: int, fmt: WavFormat) -> bytearray:
    return bytearray(nframes * fmt.nchannels * fmt.sampwidth)


def _place_with_overlap_add(
    base: bytearray,
    occupied_frames: list[bool],
    *,
    start_frame: int,
    frames: bytes,
    fmt: WavFormat,
    enable_overlap_add: bool,
) -> tuple[int, int]:
    """Place frames on the output timeline with optional equal-power overlap-add.

    Earlier stitching simply overwrote already occupied timeline regions. That
    is audible when diarization padding or converted segments overlap by a few
    milliseconds. This helper keeps the dependency-light timeline model but uses
    equal-power overlap-add wherever a new converted segment lands on already
    occupied frames. True multi-speaker overlap handling is still a separate
    service; this is a boundary-smoothing safeguard, not speaker separation.
    """

    if start_frame < 0:
        start_frame = 0
    bytes_per_frame = fmt.nchannels * fmt.sampwidth
    if bytes_per_frame <= 0:
        raise StitchingError("invalid WAV format")
    max_frames = min(len(frames) // bytes_per_frame, max(0, len(occupied_frames) - start_frame))
    if max_frames <= 0:
        return 0, 0
    usable = frames[: max_frames * bytes_per_frame]
    start_byte = start_frame * bytes_per_frame
    if not enable_overlap_add:
        base[start_byte : start_byte + len(usable)] = usable
        for frame_idx in range(start_frame, start_frame + max_frames):
            occupied_frames[frame_idx] = True
        return len(usable), 0
    overlap_local_indices = [local_idx for local_idx in range(max_frames) if occupied_frames[start_frame + local_idx]]
    if not overlap_local_indices:
        base[start_byte : start_byte + len(usable)] = usable
        for frame_idx in range(start_frame, start_frame + max_frames):
            occupied_frames[frame_idx] = True
        return len(usable), 0
    _require_pcm16(fmt, "overlap-add")
    incoming_samples = _pcm16_samples(usable)
    overlap_rank = {local_idx: rank for rank, local_idx in enumerate(overlap_local_indices)}
    denom = max(1, len(overlap_local_indices) - 1)
    for local_idx in range(max_frames):
        out_frame_idx = start_frame + local_idx
        out_byte = out_frame_idx * bytes_per_frame
        in_sample_offset = local_idx * fmt.nchannels
        if occupied_frames[out_frame_idx]:
            rank = overlap_rank[local_idx]
            alpha = 0.5 if len(overlap_local_indices) == 1 else rank / float(denom)
            existing_gain = math.cos(alpha * math.pi / 2.0)
            incoming_gain = math.sin(alpha * math.pi / 2.0)
            existing = _pcm16_samples(bytes(base[out_byte : out_byte + bytes_per_frame]))
            blended = [
                existing[channel] * existing_gain + incoming_samples[in_sample_offset + channel] * incoming_gain
                for channel in range(fmt.nchannels)
            ]
            base[out_byte : out_byte + bytes_per_frame] = _samples_to_pcm16(blended)
        else:
            base[out_byte : out_byte + bytes_per_frame] = usable[
                local_idx * bytes_per_frame : (local_idx + 1) * bytes_per_frame
            ]
            occupied_frames[out_frame_idx] = True
    return len(usable), len(overlap_local_indices)


def _require_pcm16(fmt: WavFormat, operation: str) -> None:
    if fmt.sampwidth != 2:
        raise StitchingError(f"{operation} currently requires 16-bit PCM WAV; got sample width {fmt.sampwidth}")


def _pcm16_samples(frames: bytes) -> list[int]:
    if not frames:
        return []
    if len(frames) % 2:
        raise StitchingError("invalid 16-bit PCM frame buffer length")
    return list(struct.unpack("<" + "h" * (len(frames) // 2), frames))


def _samples_to_pcm16(samples: list[int]) -> bytes:
    clipped = [max(-32768, min(32767, int(round(value)))) for value in samples]
    return struct.pack("<" + "h" * len(clipped), *clipped)


def _rms_pcm16(frames: bytes) -> float:
    samples = _pcm16_samples(frames)
    if not samples:
        return 0.0
    return math.sqrt(sum(float(sample) * float(sample) for sample in samples) / len(samples))


def _scale_pcm16(frames: bytes, gain: float) -> bytes:
    if gain <= 0 or not math.isfinite(gain):
        raise StitchingError(f"invalid loudness gain {gain}")
    samples = _pcm16_samples(frames)
    return _samples_to_pcm16([sample * gain for sample in samples])


def _fit_frames_to_nframes(frames: bytes, *, current_nframes: int, target_nframes: int, fmt: WavFormat) -> tuple[bytes, str]:
    bytes_per_frame = fmt.nchannels * fmt.sampwidth
    if target_nframes < 0:
        raise StitchingError("target_nframes must be non-negative")
    target_bytes = target_nframes * bytes_per_frame
    if current_nframes == target_nframes:
        return frames, "unchanged"
    if current_nframes > target_nframes:
        return frames[:target_bytes], "trimmed"
    return frames + bytes(_silence_bytes(target_nframes - current_nframes, fmt)), "padded"


def _apply_linear_fades_pcm16(
    frames: bytes,
    *,
    nframes: int,
    fmt: WavFormat,
    fade_frames: int,
    fade_curve: str = "linear",
    fade_in: bool = True,
    fade_out: bool = True,
) -> bytes:
    _require_pcm16(fmt, "crossfade/fade")
    if fade_frames <= 0 or nframes <= 1 or not (fade_in or fade_out):
        return frames
    fade_frames = min(fade_frames, max(1, nframes // 2))
    samples = _pcm16_samples(frames)
    channels = fmt.nchannels
    total_frames = len(samples) // channels
    if total_frames <= 1:
        return frames
    for frame_idx in range(min(fade_frames, total_frames)):
        alpha = frame_idx / float(fade_frames)
        if fade_curve == "equal_power":
            fade_in_gain = math.sin(alpha * math.pi / 2.0)
            fade_out_gain = math.sin(alpha * math.pi / 2.0)
        else:
            fade_in_gain = alpha
            fade_out_gain = alpha
        tail_frame_idx = total_frames - frame_idx - 1
        for channel in range(channels):
            if fade_in:
                samples[frame_idx * channels + channel] *= fade_in_gain
            if fade_out:
                samples[tail_frame_idx * channels + channel] *= fade_out_gain
    return _samples_to_pcm16(samples)




def _normalized_pcm16_correlation(left: bytes, right: bytes) -> float | None:
    left_samples = _pcm16_samples(left)
    right_samples = _pcm16_samples(right)
    n = min(len(left_samples), len(right_samples))
    if n <= 0:
        return None
    left_samples = left_samples[:n]
    right_samples = right_samples[:n]
    left_energy = sum(float(x) * float(x) for x in left_samples)
    right_energy = sum(float(x) * float(x) for x in right_samples)
    if left_energy <= 0 or right_energy <= 0:
        return None
    dot = sum(float(a) * float(b) for a, b in zip(left_samples, right_samples))
    return dot / math.sqrt(left_energy * right_energy)


def _local_similarity_aligned_start_frame(
    base: bytearray,
    occupied_frames: list[bool],
    *,
    nominal_start_frame: int,
    frames: bytes,
    fmt: WavFormat,
    search_frames: int,
    min_overlap_frames: int,
    min_score: float,
) -> tuple[int, dict[str, Any]]:
    """Choose a near-boundary start frame using WSOLA-style similarity.

    This is a dependency-light local-similarity alignment, not a full WSOLA time
    stretcher. For a segment starting near the previous converted segment, it
    tries small earlier shifts and compares the incoming head against already
    occupied timeline samples. If a sufficiently correlated overlap is found,
    the segment is placed slightly earlier and the regular equal-power
    overlap-add path blends the overlap.
    """

    if search_frames <= 0 or min_overlap_frames <= 0:
        return nominal_start_frame, {"status": "disabled"}
    bytes_per_frame = fmt.nchannels * fmt.sampwidth
    incoming_nframes = len(frames) // bytes_per_frame
    if incoming_nframes <= 0:
        return nominal_start_frame, {"status": "empty_segment"}
    max_shift = min(search_frames, nominal_start_frame, incoming_nframes - 1)
    if max_shift <= 0:
        return nominal_start_frame, {"status": "no_search_window"}
    best: tuple[float, int, int] | None = None  # score, shift, overlap_frames
    for shift in range(1, max_shift + 1):
        candidate_start = nominal_start_frame - shift
        # Bound the scan to the local boundary window.  The previous
        # implementation scanned from candidate_start to the end of the full
        # timeline for every candidate shift when there were no occupied frames
        # near the boundary.  On long consultations with many short turns this
        # became O(num_segments * search_frames * full_timeline_frames) and
        # stalled paper-scale batches.  Local similarity should only inspect the
        # small overlap/search window by construction.
        max_local = min(incoming_nframes, len(occupied_frames) - candidate_start, search_frames)
        if max_local <= 0:
            continue
        existing = bytearray()
        incoming = bytearray()
        overlap_count = 0
        # Only compare occupied frames. In the common adjacent-boundary case,
        # this is the previous segment tail against the incoming segment head.
        for local_idx in range(max_local):
            out_idx = candidate_start + local_idx
            if out_idx < 0 or out_idx >= len(occupied_frames):
                continue
            if not occupied_frames[out_idx]:
                continue
            out_byte = out_idx * bytes_per_frame
            in_byte = local_idx * bytes_per_frame
            existing.extend(base[out_byte : out_byte + bytes_per_frame])
            incoming.extend(frames[in_byte : in_byte + bytes_per_frame])
            overlap_count += 1
            if overlap_count >= search_frames:
                break
        if overlap_count < min_overlap_frames:
            continue
        score = _normalized_pcm16_correlation(bytes(existing), bytes(incoming))
        if score is None or score < min_score:
            continue
        if best is None or score > best[0] or (math.isclose(score, best[0]) and shift < best[1]):
            best = (score, shift, overlap_count)
    if best is None:
        return nominal_start_frame, {
            "status": "no_sufficient_similarity_match",
            "search_frames": search_frames,
            "min_overlap_frames": min_overlap_frames,
            "min_score": min_score,
        }
    score, shift, overlap_count = best
    return nominal_start_frame - shift, {
        "status": "aligned",
        "shift_frames": shift,
        "shift_sec": shift / float(fmt.framerate),
        "score": round(score, 4),
        "overlap_frames_compared": overlap_count,
        "search_frames": search_frames,
    }

def _source_loudness_target(
    *,
    artifact_store: LocalArtifactStore,
    row: dict[str, Any],
    fmt: WavFormat,
) -> float | None:
    source_audio_uri = row.get("source_audio_uri") or row.get("segment_audio_uri")
    if not source_audio_uri:
        return None
    try:
        validate_artifact_uri(str(source_audio_uri))
        source_path = artifact_store.local_path(str(source_audio_uri))
        if not source_path.exists():
            return None
        source_fmt, source_frames, _ = _read_wav_frames(source_path)
        if source_fmt != fmt:
            return None
        return _rms_pcm16(source_frames)
    except Exception:
        return None


@dataclass(frozen=True)
class StitchingPolicy:
    """DSP policy for timeline reconstruction.

    The first implementation stays dependency-light and deterministic:
    converted segments are trimmed/padded to expected timeline duration, optional
    per-segment RMS matching uses the source segment as a coarse loudness target,
    and short linear fades are applied at segment boundaries to avoid clicks.
    """

    align_duration: bool = True
    loudness_normalize: bool = True
    fade_ms: float = 5.0
    fade_curve: str = "equal_power"
    overlap_add: bool = True
    boundary_alignment: str = "none"
    boundary_alignment_search_ms: float = 20.0
    boundary_alignment_min_overlap_ms: float = 5.0
    boundary_alignment_min_score: float = 0.2
    suppress_internal_boundary_fades: bool = True
    internal_boundary_gap_ms: float = 60.0
    max_duration_drift_sec: float = 0.25
    max_loudness_gain: float = 4.0
    min_loudness_gain: float = 0.25

    def __post_init__(self) -> None:
        if self.fade_ms < 0:
            raise AegisVoxContractError("fade_ms cannot be negative")
        if self.fade_curve not in {"linear", "equal_power"}:
            raise AegisVoxContractError("fade_curve must be 'linear' or 'equal_power'")
        if self.boundary_alignment not in {"none", "local_similarity"}:
            raise AegisVoxContractError("boundary_alignment must be 'none' or 'local_similarity'")
        if self.boundary_alignment_search_ms < 0 or self.boundary_alignment_min_overlap_ms < 0:
            raise AegisVoxContractError("boundary alignment windows cannot be negative")
        if not (0 <= self.boundary_alignment_min_score <= 1):
            raise AegisVoxContractError("boundary_alignment_min_score must be between 0 and 1")
        if self.internal_boundary_gap_ms < 0:
            raise AegisVoxContractError("internal_boundary_gap_ms cannot be negative")
        if self.max_duration_drift_sec < 0:
            raise AegisVoxContractError("max_duration_drift_sec cannot be negative")
        if self.min_loudness_gain <= 0 or self.max_loudness_gain < self.min_loudness_gain:
            raise AegisVoxContractError("invalid loudness gain bounds")


@dataclass
class StitchingService:
    artifact_store: LocalArtifactStore
    policy: StitchingPolicy = field(default_factory=StitchingPolicy)

    def stitch_to_artifact(
        self,
        *,
        converted_segments_uri: str,
        output_audio_uri: str,
        stitch_report_uri: str,
        audio_duration_sec: float | None = None,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        validate_artifact_uri(converted_segments_uri)
        validate_artifact_uri(output_audio_uri)
        validate_artifact_uri(stitch_report_uri)
        payload = json.loads(self.artifact_store.local_path(converted_segments_uri).read_text())
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise AegisVoxContractError("segments.converted.json must contain segments list")
        converted = [row for row in segments if row.get("conversion_status") == "converted"]
        if not converted:
            raise StitchingError("no converted segments available for stitching")
        first_path = self.artifact_store.local_path(converted[0]["converted_audio_uri"])
        if not first_path.exists():
            raise StitchingError(f"converted audio missing: {first_path}")
        fmt = _wav_format(first_path)
        if sample_rate is not None and sample_rate != fmt.framerate:
            raise StitchingError("resampling is not implemented; converted segment sample rate must match requested sample_rate")
        if audio_duration_sec is None:
            max_end = max(float(row.get("padded_end_sec") or row.get("end_sec") or 0.0) for row in segments)
            audio_duration_sec = max_end
        total_frames = max(1, int(round(audio_duration_sec * fmt.framerate)))
        timeline = _silence_bytes(total_frames, fmt)
        occupied_frames = [False] * total_frames
        fade_frames = int(round((self.policy.fade_ms / 1000.0) * fmt.framerate))
        internal_boundary_gap_sec = self.policy.internal_boundary_gap_ms / 1000.0
        boundary_alignment_search_frames = int(round((self.policy.boundary_alignment_search_ms / 1000.0) * fmt.framerate))
        boundary_alignment_min_overlap_frames = int(round((self.policy.boundary_alignment_min_overlap_ms / 1000.0) * fmt.framerate))
        converted_intervals: list[tuple[int, float, float, dict[str, Any]]] = []
        for idx, interval_row in enumerate(segments):
            if interval_row.get("conversion_status") != "converted":
                continue
            interval_start = float(
                interval_row.get("padded_start_sec")
                if interval_row.get("padded_start_sec") is not None
                else interval_row.get("start_sec")
                or 0.0
            )
            interval_end = float(
                interval_row.get("padded_end_sec")
                if interval_row.get("padded_end_sec") is not None
                else interval_row.get("end_sec")
                or interval_start
            )
            converted_intervals.append((idx, interval_start, interval_end, interval_row))
        converted_intervals.sort(key=lambda item: (item[1], item[2], item[0]))
        prev_end_by_idx: dict[int, float | None] = {}
        next_start_by_idx: dict[int, float | None] = {}
        for pos, (idx, start, end, _) in enumerate(converted_intervals):
            prev_end_by_idx[idx] = converted_intervals[pos - 1][2] if pos > 0 else None
            next_start_by_idx[idx] = converted_intervals[pos + 1][1] if pos + 1 < len(converted_intervals) else None
        stitched_rows = []
        excessive_drift = []
        for row_idx, row in enumerate(segments):
            status = row.get("conversion_status")
            out = dict(row)
            if status != "converted":
                out["stitch_status"] = "not_stitched"
                stitched_rows.append(out)
                continue
            path = self.artifact_store.local_path(row["converted_audio_uri"])
            if not path.exists():
                raise StitchingError(f"converted audio missing: {path}")
            seg_fmt, frames, nframes = _read_wav_frames(path)
            if seg_fmt != fmt:
                raise StitchingError(f"converted segment format mismatch for {path}")
            start_sec = float(row.get("padded_start_sec") if row.get("padded_start_sec") is not None else row.get("start_sec") or 0.0)
            expected_end = float(row.get("padded_end_sec") if row.get("padded_end_sec") is not None else row.get("end_sec") or start_sec)
            expected_duration = max(0.001, expected_end - start_sec)
            expected_nframes = max(1, int(round(expected_duration * fmt.framerate)))
            actual_duration = nframes / float(fmt.framerate)
            original_drift = actual_duration - expected_duration
            if abs(original_drift) > self.policy.max_duration_drift_sec:
                excessive_drift.append({"segment_id": row.get("segment_id"), "drift_sec": round(original_drift, 4)})
            duration_alignment = "unchanged"
            if self.policy.align_duration:
                frames, duration_alignment = _fit_frames_to_nframes(
                    frames,
                    current_nframes=nframes,
                    target_nframes=expected_nframes,
                    fmt=fmt,
                )
                nframes = expected_nframes
            loudness_gain = None
            loudness_normalization = "not_applied"
            if self.policy.loudness_normalize:
                _require_pcm16(fmt, "loudness normalization")
                source_rms = _source_loudness_target(artifact_store=self.artifact_store, row=row, fmt=fmt)
                converted_rms = _rms_pcm16(frames)
                if source_rms is not None and source_rms > 0 and converted_rms > 0:
                    raw_gain = source_rms / converted_rms
                    loudness_gain = max(self.policy.min_loudness_gain, min(self.policy.max_loudness_gain, raw_gain))
                    frames = _scale_pcm16(frames, loudness_gain)
                    loudness_normalization = "rms_matched"
                else:
                    loudness_normalization = "skipped_missing_or_silent_reference"
            fade_in_applied = fade_frames > 0
            fade_out_applied = fade_frames > 0
            boundary_smoothing = "edge_fades"
            prev_end = prev_end_by_idx.get(row_idx)
            next_start = next_start_by_idx.get(row_idx)
            if self.policy.suppress_internal_boundary_fades and fade_frames > 0:
                if prev_end is not None and (start_sec - prev_end) <= internal_boundary_gap_sec:
                    fade_in_applied = False
                if next_start is not None and (next_start - expected_end) <= internal_boundary_gap_sec:
                    fade_out_applied = False
                if not fade_in_applied or not fade_out_applied:
                    boundary_smoothing = "suppressed_internal_fade"
            if fade_frames > 0:
                frames = _apply_linear_fades_pcm16(
                    frames,
                    nframes=nframes,
                    fmt=fmt,
                    fade_frames=fade_frames,
                    fade_curve=self.policy.fade_curve,
                    fade_in=fade_in_applied,
                    fade_out=fade_out_applied,
                )
            nominal_start_frame = max(0, int(round(start_sec * fmt.framerate)))
            start_frame = nominal_start_frame
            boundary_alignment_result = {"status": "disabled"}
            if self.policy.boundary_alignment == "local_similarity":
                _require_pcm16(fmt, "local-similarity boundary alignment")
                start_frame, boundary_alignment_result = _local_similarity_aligned_start_frame(
                    timeline,
                    occupied_frames,
                    nominal_start_frame=nominal_start_frame,
                    frames=frames,
                    fmt=fmt,
                    search_frames=boundary_alignment_search_frames,
                    min_overlap_frames=max(1, boundary_alignment_min_overlap_frames),
                    min_score=self.policy.boundary_alignment_min_score,
                )
            bytes_written, overlap_added_frames = _place_with_overlap_add(
                timeline,
                occupied_frames,
                start_frame=start_frame,
                frames=frames,
                fmt=fmt,
                enable_overlap_add=self.policy.overlap_add,
            )
            placed_duration = (bytes_written // (fmt.nchannels * fmt.sampwidth)) / float(fmt.framerate)
            placed_drift = placed_duration - expected_duration
            out.update(
                {
                    "stitch_status": "stitched",
                    "actual_duration_sec": round(actual_duration, 4),
                    "expected_duration_sec": round(expected_duration, 4),
                    "original_duration_drift_sec": round(original_drift, 4),
                    "placed_duration_sec": round(placed_duration, 4),
                    "duration_drift_sec": round(placed_drift, 4),
                    "duration_alignment": duration_alignment,
                    "loudness_normalization": loudness_normalization,
                    "loudness_gain": round(loudness_gain, 4) if loudness_gain is not None else None,
                    "fade_ms": self.policy.fade_ms,
                    "fade_curve": self.policy.fade_curve,
                    "fade_in_applied": fade_in_applied,
                    "fade_out_applied": fade_out_applied,
                    "boundary_smoothing": boundary_smoothing,
                    "nominal_start_frame": nominal_start_frame,
                    "placed_start_frame": start_frame,
                    "boundary_alignment": self.policy.boundary_alignment,
                    "boundary_alignment_result": boundary_alignment_result,
                    "overlap_add_applied": self.policy.overlap_add and overlap_added_frames > 0,
                    "overlap_added_frames": overlap_added_frames,
                    "overlap_added_duration_sec": round(overlap_added_frames / float(fmt.framerate), 4),
                    "prev_converted_gap_sec": round(start_sec - prev_end, 4) if prev_end is not None else None,
                    "next_converted_gap_sec": round(next_start - expected_end, 4) if next_start is not None else None,
                }
            )
            stitched_rows.append(out)
        output_path = self.artifact_store.local_path(output_audio_uri)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(fmt.nchannels)
            wav.setsampwidth(fmt.sampwidth)
            wav.setframerate(fmt.framerate)
            wav.writeframes(bytes(timeline))
        report = {
            "changed_identity_audio_uri": output_audio_uri,
            "duration_sec": round(wav_duration_sec(output_path), 4),
            "dsp_policy": {
                "align_duration": self.policy.align_duration,
                "loudness_normalize": self.policy.loudness_normalize,
                "fade_ms": self.policy.fade_ms,
                "fade_curve": self.policy.fade_curve,
                "overlap_add": self.policy.overlap_add,
                "boundary_alignment": self.policy.boundary_alignment,
                "boundary_alignment_search_ms": self.policy.boundary_alignment_search_ms,
                "boundary_alignment_min_overlap_ms": self.policy.boundary_alignment_min_overlap_ms,
                "boundary_alignment_min_score": self.policy.boundary_alignment_min_score,
                "suppress_internal_boundary_fades": self.policy.suppress_internal_boundary_fades,
                "internal_boundary_gap_ms": self.policy.internal_boundary_gap_ms,
                "max_duration_drift_sec": self.policy.max_duration_drift_sec,
            },
            "num_segments": len(segments),
            "num_stitched": sum(1 for row in stitched_rows if row.get("stitch_status") == "stitched"),
            "num_not_stitched": sum(1 for row in stitched_rows if row.get("stitch_status") != "stitched"),
            "excessive_duration_drift": excessive_drift,
            "segments": stitched_rows,
        }
        self.artifact_store.write_json(stitch_report_uri, report)
        return report
