"""Deterministic diarization cleanup and segment extraction.

This module does not diarize audio. It consumes a real diarization artifact from
pyannote or another diarization backend and applies the AegisVox cleanup rules.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore

MIN_SEGMENT_SEC = 0.7
PRESERVE_SPEAKER_MIN_TURN_SEC = 0.10
PRESERVE_SPEAKER_MIN_TOTAL_SEC = 0.25
MAX_SEGMENT_SEC = 15.0
MERGE_GAP_SEC = 0.35
PAD_LEFT_SEC = 0.08
PAD_RIGHT_SEC = 0.12
TINY_SPEAKER_MAX_TOTAL_SEC = 15.0
TINY_SPEAKER_MAX_FRACTION = 0.02
TINY_SPEAKER_MAX_TURNS = 30


@dataclass(frozen=True)
class DiarizationTurn:
    speaker_id: str
    start_sec: float
    end_sec: float
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.speaker_id:
            raise AegisVoxContractError("speaker_id is required")
        if self.start_sec < 0:
            raise AegisVoxContractError("start_sec must be non-negative")
        if self.end_sec <= self.start_sec:
            raise AegisVoxContractError("end_sec must be greater than start_sec")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiarizationTurn":
        # Accept both direct rows and pyannote-like rows.
        speaker = payload.get("speaker_id") or payload.get("speaker") or payload.get("label")
        start = payload.get("start_sec", payload.get("start"))
        end = payload.get("end_sec", payload.get("end"))
        if speaker is None or start is None or end is None:
            raise AegisVoxContractError("diarization turn must include speaker/start/end")
        return cls(
            speaker_id=str(speaker),
            start_sec=float(start),
            end_sec=float(end),
            confidence=payload.get("confidence"),
            metadata={k: v for k, v in payload.items() if k not in {"speaker_id", "speaker", "label", "start_sec", "start", "end_sec", "end", "confidence"}},
        )


@dataclass(frozen=True)
class CleanSegment:
    segment_id: str
    speaker_id: str
    start_sec: float
    end_sec: float
    padded_start_sec: float
    padded_end_sec: float
    duration_sec: float
    overlap: bool = False
    source_audio_uri: str | None = None
    segment_audio_uri: str | None = None
    flags: list[str] = field(default_factory=list)
    overlap_speaker_ids: list[str] = field(default_factory=list)
    overlap_kind: str | None = None
    parent_turn_start_sec: float | None = None
    parent_turn_end_sec: float | None = None
    original_speaker_id: str | None = None
    speaker_merge_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "padded_start_sec": self.padded_start_sec,
            "padded_end_sec": self.padded_end_sec,
            "duration_sec": self.duration_sec,
            "overlap": self.overlap,
            "source_audio_uri": self.source_audio_uri,
            "segment_audio_uri": self.segment_audio_uri,
            "flags": list(self.flags),
        }
        if self.overlap_speaker_ids:
            payload["overlap_speaker_ids"] = list(self.overlap_speaker_ids)
        if self.overlap_kind:
            payload["overlap_kind"] = self.overlap_kind
        if self.parent_turn_start_sec is not None:
            payload["parent_turn_start_sec"] = self.parent_turn_start_sec
        if self.parent_turn_end_sec is not None:
            payload["parent_turn_end_sec"] = self.parent_turn_end_sec
        if self.original_speaker_id is not None:
            payload["original_speaker_id"] = self.original_speaker_id
        if self.speaker_merge_reason is not None:
            payload["speaker_merge_reason"] = self.speaker_merge_reason
        return payload


def load_diarization_turns(path: str | Path) -> list[DiarizationTurn]:
    payload = json.loads(Path(path).read_text())
    rows = payload.get("turns") or payload.get("segments") or payload
    if not isinstance(rows, list):
        raise AegisVoxContractError("diarization artifact must be a list or contain turns/segments list")
    return [DiarizationTurn.from_dict(row) for row in rows]


def _merge_adjacent_turns(turns: list[DiarizationTurn], merge_gap_sec: float) -> list[DiarizationTurn]:
    if not turns:
        return []
    sorted_turns = sorted(turns, key=lambda turn: (turn.start_sec, turn.end_sec, turn.speaker_id))
    merged: list[DiarizationTurn] = []
    current = sorted_turns[0]
    for turn in sorted_turns[1:]:
        same_speaker = turn.speaker_id == current.speaker_id
        gap = turn.start_sec - current.end_sec
        if same_speaker and 0 <= gap <= merge_gap_sec:
            current = DiarizationTurn(
                speaker_id=current.speaker_id,
                start_sec=current.start_sec,
                end_sec=max(current.end_sec, turn.end_sec),
                confidence=current.confidence,
                metadata={**current.metadata, "merged": True},
            )
        else:
            merged.append(current)
            current = turn
    merged.append(current)
    return merged


def _speaker_stats(turns: list[DiarizationTurn]) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for turn in turns:
        row = stats.setdefault(turn.speaker_id, {"duration_sec": 0.0, "num_turns": 0})
        row["duration_sec"] = float(row["duration_sec"]) + max(0.0, turn.end_sec - turn.start_sec)
        row["num_turns"] = int(row["num_turns"]) + 1
    return stats


def _overlap_duration(a: DiarizationTurn, b: DiarizationTurn) -> float:
    return max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))


def _nearest_main_speaker(turn: DiarizationTurn, main_turns: list[DiarizationTurn], speaker_totals: dict[str, float]) -> str:
    overlapping = []
    for other in main_turns:
        overlap = _overlap_duration(turn, other)
        if overlap > 0:
            overlapping.append((overlap, speaker_totals.get(other.speaker_id, 0.0), other.speaker_id))
    if overlapping:
        overlapping.sort(key=lambda row: (-row[0], -row[1], row[2]))
        return overlapping[0][2]

    nearest = []
    for other in main_turns:
        if other.end_sec <= turn.start_sec:
            gap = turn.start_sec - other.end_sec
        elif turn.end_sec <= other.start_sec:
            gap = other.start_sec - turn.end_sec
        else:
            gap = 0.0
        nearest.append((gap, -speaker_totals.get(other.speaker_id, 0.0), other.speaker_id))
    if nearest:
        nearest.sort(key=lambda row: (row[0], row[1], row[2]))
        return nearest[0][2]
    return sorted(speaker_totals.items(), key=lambda row: (-row[1], row[0]))[0][0]


def merge_tiny_speaker_fragments(
    turns: list[DiarizationTurn],
    *,
    expected_speaker_count: int | None,
    tiny_speaker_max_total_sec: float = TINY_SPEAKER_MAX_TOTAL_SEC,
    tiny_speaker_max_fraction: float = TINY_SPEAKER_MAX_FRACTION,
    tiny_speaker_max_turns: int = TINY_SPEAKER_MAX_TURNS,
) -> tuple[list[DiarizationTurn], dict[str, Any]]:
    """Collapse tiny extra diarization speakers before target-voice assignment.

    Diarization backends often create extra labels for short backchannels,
    breath/noise, or overlap fragments. If a run is bounded to an expected
    speaker count, tiny extra clusters should not receive independent target
    voices unless they have enough duration/turn evidence. This function maps
    those fragments onto the nearest or overlapping main speaker and preserves
    the original label in turn metadata for audit.
    """

    stats = _speaker_stats(turns)
    total_duration = sum(float(row["duration_sec"]) for row in stats.values()) or 1.0
    ranked = sorted(stats.items(), key=lambda item: (-float(item[1]["duration_sec"]), item[0]))
    auto_inferred = expected_speaker_count is None or expected_speaker_count <= 0
    if auto_inferred:
        substantial_speakers = [
            speaker
            for speaker, row in ranked
            if float(row["duration_sec"]) > tiny_speaker_max_total_sec
            or (float(row["duration_sec"]) / total_duration) > tiny_speaker_max_fraction
            or int(row["num_turns"]) > tiny_speaker_max_turns
        ]
        if len(substantial_speakers) < 2 or len(stats) <= len(substantial_speakers):
            return turns, {
                "enabled": True,
                "mode": "auto_over_speaker_likelihood",
                "inferred_expected_speaker_count": len(substantial_speakers) if len(substantial_speakers) >= 2 else None,
                "merged_speakers": [],
                "reason": "insufficient_substantial_main_speakers_or_no_extra_clusters",
            }
        expected_count = len(substantial_speakers)
        main_speakers = set(substantial_speakers)
        extra_speakers = [speaker for speaker, _ in ranked if speaker not in main_speakers]
    else:
        expected_count = int(expected_speaker_count)
        if len(stats) <= expected_count:
            return turns, {"enabled": True, "mode": "expected_speaker_count", "expected_speaker_count": expected_count, "merged_speakers": []}
        main_speakers = {speaker for speaker, _ in ranked[:expected_count]}
        extra_speakers = [speaker for speaker, _ in ranked[expected_count:]]

    merge_speakers = set()
    skipped_speakers = []
    for speaker in extra_speakers:
        row = stats[speaker]
        duration = float(row["duration_sec"])
        num_turns = int(row["num_turns"])
        fraction = duration / total_duration
        if auto_inferred:
            # Unknown-count mode must be conservative: only collapse clusters
            # that are both very short overall and low-fraction/low-turn.
            is_tiny = duration <= tiny_speaker_max_total_sec and fraction <= tiny_speaker_max_fraction and num_turns <= tiny_speaker_max_turns
        else:
            # Known-count mode can be more assertive because the operator or
            # diarizer has supplied a target speaker count.
            is_tiny = duration <= tiny_speaker_max_total_sec or (fraction <= tiny_speaker_max_fraction and num_turns <= tiny_speaker_max_turns)
        if is_tiny:
            merge_speakers.add(speaker)
        else:
            skipped_speakers.append({"speaker_id": speaker, "duration_sec": round(duration, 3), "num_turns": num_turns, "fraction": round(fraction, 5)})
    if not merge_speakers:
        return turns, {
            "enabled": True,
            "expected_speaker_count": expected_speaker_count,
            "merged_speakers": [],
            "skipped_extra_speakers": skipped_speakers,
        }

    main_turns = [turn for turn in turns if turn.speaker_id in main_speakers]
    speaker_totals = {speaker: float(row["duration_sec"]) for speaker, row in stats.items() if speaker in main_speakers}
    merged_turns: list[DiarizationTurn] = []
    merge_counts: dict[str, dict[str, Any]] = {}
    for turn in turns:
        if turn.speaker_id not in merge_speakers:
            merged_turns.append(turn)
            continue
        target = _nearest_main_speaker(turn, main_turns, speaker_totals)
        duration = max(0.0, turn.end_sec - turn.start_sec)
        row = merge_counts.setdefault(turn.speaker_id, {"target_speaker_counts": {}, "duration_sec": 0.0, "num_turns": 0})
        row["duration_sec"] = float(row["duration_sec"]) + duration
        row["num_turns"] = int(row["num_turns"]) + 1
        row["target_speaker_counts"][target] = int(row["target_speaker_counts"].get(target, 0)) + 1
        merged_turns.append(
            DiarizationTurn(
                speaker_id=target,
                start_sec=turn.start_sec,
                end_sec=turn.end_sec,
                confidence=turn.confidence,
                metadata={
                    **turn.metadata,
                    "speaker_fragment_merged": True,
                    "original_speaker_id": turn.speaker_id,
                    "speaker_merge_reason": "tiny_extra_speaker_cluster",
                    "expected_speaker_count": None if auto_inferred else expected_count,
                    "speaker_merge_mode": "auto_over_speaker_likelihood" if auto_inferred else "expected_speaker_count",
                },
            )
        )
    summary = {
        "enabled": True,
        "mode": "auto_over_speaker_likelihood" if auto_inferred else "expected_speaker_count",
        "expected_speaker_count": None if auto_inferred else expected_count,
        "inferred_expected_speaker_count": expected_count if auto_inferred else None,
        "main_speakers": sorted(main_speakers),
        "merged_speakers": [
            {
                "speaker_id": speaker,
                "duration_sec": round(float(row["duration_sec"]), 3),
                "num_turns": int(row["num_turns"]),
                "target_speaker_counts": dict(row["target_speaker_counts"]),
            }
            for speaker, row in sorted(merge_counts.items())
        ],
        "skipped_extra_speakers": skipped_speakers,
    }
    return merged_turns, summary



def make_exclusive_speaker_timeline(turns: list[DiarizationTurn]) -> tuple[list[DiarizationTurn], dict[str, Any]]:
    """Rewrite diarization to a non-overlapping single-speaker timeline.

    This is a controlled/debug reconstruction mode. Real overlapped speech is
    not separated here; each atomic time interval is assigned to exactly one
    dominant speaker by interval coverage, confidence, and speaker total
    duration. The output is safe for downstream single-speaker conversion and
    lexical editing because no segment can carry ``overlap=True``.
    """

    if not turns:
        return [], {"enabled": True, "mode": "exclusive_dominant_speaker", "num_input_turns": 0, "num_output_turns": 0, "num_resolved_overlap_intervals": 0}
    boundaries = sorted({float(turn.start_sec) for turn in turns} | {float(turn.end_sec) for turn in turns})
    speaker_totals: dict[str, float] = {}
    for turn in turns:
        speaker_totals[turn.speaker_id] = speaker_totals.get(turn.speaker_id, 0.0) + max(0.0, turn.end_sec - turn.start_sec)

    exclusive: list[DiarizationTurn] = []
    resolved_overlap_intervals = 0
    dropped_speaker_interval_sec: dict[str, float] = {}
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active: list[tuple[float, float, float, str, DiarizationTurn]] = []
        for turn in turns:
            cover = max(0.0, min(end, turn.end_sec) - max(start, turn.start_sec))
            if cover > 0:
                conf = float(turn.confidence) if turn.confidence is not None else 0.0
                total = speaker_totals.get(turn.speaker_id, 0.0)
                active.append((cover, conf, total, turn.speaker_id, turn))
        if not active:
            continue
        active.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        chosen = active[0][4]
        if len({row[3] for row in active}) > 1:
            resolved_overlap_intervals += 1
            for row in active[1:]:
                dropped_speaker_interval_sec[row[3]] = dropped_speaker_interval_sec.get(row[3], 0.0) + (end - start)
        metadata = {
            **chosen.metadata,
            "exclusive_timeline": True,
            "exclusive_source_start_sec": chosen.start_sec,
            "exclusive_source_end_sec": chosen.end_sec,
        }
        if len(active) > 1:
            metadata["exclusive_resolved_overlap"] = True
            metadata["exclusive_active_speakers"] = sorted({row[3] for row in active})
            metadata["exclusive_resolution"] = "dominant_speaker_by_interval_coverage_confidence_total_duration"
        current = DiarizationTurn(
            speaker_id=chosen.speaker_id,
            start_sec=start,
            end_sec=end,
            confidence=chosen.confidence,
            metadata=metadata,
        )
        if exclusive and exclusive[-1].speaker_id == current.speaker_id and exclusive[-1].end_sec == current.start_sec:
            prev = exclusive[-1]
            merged_meta = {**prev.metadata}
            if current.metadata.get("exclusive_resolved_overlap") or prev.metadata.get("exclusive_resolved_overlap"):
                merged_meta["exclusive_resolved_overlap"] = True
            if current.metadata.get("exclusive_active_speakers") or prev.metadata.get("exclusive_active_speakers"):
                merged_meta["exclusive_active_speakers"] = sorted(set(prev.metadata.get("exclusive_active_speakers") or []) | set(current.metadata.get("exclusive_active_speakers") or []))
            exclusive[-1] = DiarizationTurn(
                speaker_id=prev.speaker_id,
                start_sec=prev.start_sec,
                end_sec=current.end_sec,
                confidence=prev.confidence,
                metadata=merged_meta,
            )
        else:
            exclusive.append(current)
    summary = {
        "enabled": True,
        "mode": "exclusive_dominant_speaker",
        "num_input_turns": len(turns),
        "num_output_turns": len(exclusive),
        "num_resolved_overlap_intervals": resolved_overlap_intervals,
        "dropped_speaker_interval_sec": {k: round(v, 3) for k, v in sorted(dropped_speaker_interval_sec.items())},
        "claim_boundary": "overlapped source regions are forced to one dominant speaker for debugging/ablation; this does not preserve simultaneous secondary speech",
    }
    return exclusive, summary

def _mark_overlap(turn: DiarizationTurn, all_turns: list[DiarizationTurn]) -> bool:
    return any(
        other.speaker_id != turn.speaker_id
        and other.start_sec < turn.end_sec
        and turn.start_sec < other.end_sec
        for other in all_turns
    )



def _overlap_speaker_ids_for_interval(
    *,
    speaker_id: str,
    start_sec: float,
    end_sec: float,
    all_turns: list[DiarizationTurn],
) -> list[str]:
    ids = {speaker_id}
    for other in all_turns:
        if other.speaker_id == speaker_id:
            continue
        if other.start_sec < end_sec and start_sec < other.end_sec:
            ids.add(other.speaker_id)
    return sorted(ids)


def _split_boundaries_for_turn(turn: DiarizationTurn, all_turns: list[DiarizationTurn]) -> list[float]:
    boundaries = {turn.start_sec, turn.end_sec}
    for other in all_turns:
        if other.speaker_id == turn.speaker_id:
            continue
        start = max(turn.start_sec, other.start_sec)
        end = min(turn.end_sec, other.end_sec)
        if end > start:
            boundaries.add(start)
            boundaries.add(end)
    return sorted(boundaries)


def _append_split_segment(
    segments: list[CleanSegment],
    *,
    counter: int,
    speaker_id: str,
    start_sec: float,
    end_sec: float,
    parent_turn: DiarizationTurn,
    audio_duration_sec: float | None,
    pad_left_sec: float,
    pad_right_sec: float,
    source_audio_uri: str | None,
    flags: list[str],
    overlap: bool,
    overlap_speaker_ids: list[str] | None = None,
) -> int:
    padded_start = max(0.0, start_sec - pad_left_sec)
    padded_end = end_sec + pad_right_sec
    if audio_duration_sec is not None:
        padded_end = min(audio_duration_sec, padded_end)
    counter += 1
    segments.append(
        CleanSegment(
            segment_id=f"seg_{counter:06d}",
            speaker_id=speaker_id,
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            padded_start_sec=round(padded_start, 3),
            padded_end_sec=round(padded_end, 3),
            duration_sec=round(end_sec - start_sec, 3),
            overlap=overlap,
            source_audio_uri=source_audio_uri,
            flags=flags,
            overlap_speaker_ids=overlap_speaker_ids or [],
            overlap_kind=("multi_speaker" if overlap and len(overlap_speaker_ids or []) > 1 else ("single_speaker_flagged" if overlap else None)),
            parent_turn_start_sec=round(parent_turn.start_sec, 3),
            parent_turn_end_sec=round(parent_turn.end_sec, 3),
            original_speaker_id=parent_turn.metadata.get("original_speaker_id") if parent_turn.metadata.get("speaker_fragment_merged") else None,
            speaker_merge_reason=parent_turn.metadata.get("speaker_merge_reason") if parent_turn.metadata.get("speaker_fragment_merged") else None,
        )
    )
    return counter

def clean_diarization_turns(
    turns: list[DiarizationTurn],
    *,
    audio_duration_sec: float | None = None,
    min_segment_sec: float = MIN_SEGMENT_SEC,
    max_segment_sec: float = MAX_SEGMENT_SEC,
    preserve_speaker_min_turn_sec: float = PRESERVE_SPEAKER_MIN_TURN_SEC,
    preserve_speaker_min_total_sec: float = PRESERVE_SPEAKER_MIN_TOTAL_SEC,
    merge_gap_sec: float = MERGE_GAP_SEC,
    pad_left_sec: float = PAD_LEFT_SEC,
    pad_right_sec: float = PAD_RIGHT_SEC,
    source_audio_uri: str | None = None,
    split_on_overlap_intervals: bool = True,
    exclusive_speaker_timeline: bool = False,
    expected_speaker_count: int | None = None,
    tiny_speaker_max_total_sec: float = TINY_SPEAKER_MAX_TOTAL_SEC,
    tiny_speaker_max_fraction: float = TINY_SPEAKER_MAX_FRACTION,
    tiny_speaker_max_turns: int = TINY_SPEAKER_MAX_TURNS,
) -> list[CleanSegment]:
    if source_audio_uri is not None:
        validate_artifact_uri(source_audio_uri)
    merged = _merge_adjacent_turns(turns, merge_gap_sec=merge_gap_sec)
    exclusive_summary: dict[str, Any] = {"enabled": False}
    if exclusive_speaker_timeline:
        merged, exclusive_summary = make_exclusive_speaker_timeline(merged)
    merged, _merge_summary = merge_tiny_speaker_fragments(
        merged,
        expected_speaker_count=expected_speaker_count,
        tiny_speaker_max_total_sec=tiny_speaker_max_total_sec,
        tiny_speaker_max_fraction=tiny_speaker_max_fraction,
        tiny_speaker_max_turns=tiny_speaker_max_turns,
    )
    speaker_total_duration: dict[str, float] = {}
    speaker_has_normal_turn: dict[str, bool] = {}
    for turn in merged:
        duration = turn.end_sec - turn.start_sec
        speaker_total_duration[turn.speaker_id] = speaker_total_duration.get(turn.speaker_id, 0.0) + duration
        if duration >= min_segment_sec:
            speaker_has_normal_turn[turn.speaker_id] = True
    preserve_short_speakers = {
        speaker_id
        for speaker_id, total in speaker_total_duration.items()
        if total >= preserve_speaker_min_total_sec and not speaker_has_normal_turn.get(speaker_id, False)
    }
    segments: list[CleanSegment] = []
    counter = 0
    for turn in merged:
        duration = turn.end_sec - turn.start_sec
        preserve_short_turn = (
            turn.speaker_id in preserve_short_speakers
            and duration >= preserve_speaker_min_turn_sec
            and duration < min_segment_sec
        )
        preserve_merged_fragment = bool(turn.metadata.get("speaker_fragment_merged")) and duration >= preserve_speaker_min_turn_sec
        effective_min_segment_sec = preserve_speaker_min_turn_sec if preserve_short_turn or preserve_merged_fragment else min_segment_sec
        if duration < effective_min_segment_sec:
            continue
        if not split_on_overlap_intervals:
            piece_ranges = [(turn.start_sec, turn.end_sec)]
        else:
            boundaries = _split_boundaries_for_turn(turn, merged)
            piece_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

        for piece_start, piece_end in piece_ranges:
            overlap_speaker_ids = _overlap_speaker_ids_for_interval(
                speaker_id=turn.speaker_id,
                start_sec=piece_start,
                end_sec=piece_end,
                all_turns=merged,
            )
            overlap = len(overlap_speaker_ids) > 1
            piece_min_sec = preserve_speaker_min_turn_sec if overlap or preserve_short_turn or preserve_merged_fragment else min_segment_sec
            if piece_end - piece_start < piece_min_sec:
                continue
            split_start = piece_start
            split_idx = 0
            while split_start < piece_end:
                split_end = min(piece_end, split_start + max_segment_sec)
                if split_end - split_start < piece_min_sec:
                    break
                flags: list[str] = []
                if preserve_short_turn:
                    flags.extend(["short_speaker_turn", "low_duration_speaker", "preserved_to_avoid_speaker_collapse"])
                if split_idx > 0 or split_end < piece_end:
                    flags.append("split_long_turn")
                if split_on_overlap_intervals:
                    flags.append("overlap_interval_split")
                if turn.metadata.get("exclusive_timeline"):
                    flags.append("exclusive_speaker_timeline")
                if turn.metadata.get("exclusive_resolved_overlap"):
                    flags.append("exclusive_resolved_overlap_to_dominant_speaker")
                if turn.metadata.get("speaker_fragment_merged"):
                    flags.append("speaker_fragment_merged")
                    flags.append("tiny_speaker_cluster_collapsed")
                if overlap:
                    flags.append("overlap")
                    flags.append("true_overlap_interval")
                    if preserve_short_turn:
                        flags.append("short_overlap_speaker_turn")
                elif _mark_overlap(turn, merged):
                    flags.append("clean_subspan_from_overlap_turn")
                counter = _append_split_segment(
                    segments,
                    counter=counter,
                    speaker_id=turn.speaker_id,
                    start_sec=split_start,
                    end_sec=split_end,
                    parent_turn=turn,
                    audio_duration_sec=audio_duration_sec,
                    pad_left_sec=pad_left_sec,
                    pad_right_sec=pad_right_sec,
                    source_audio_uri=source_audio_uri,
                    flags=flags,
                    overlap=overlap,
                    overlap_speaker_ids=overlap_speaker_ids if overlap else [],
                )
                split_start = split_end
                split_idx += 1
    return segments

def segments_payload(segments: list[CleanSegment]) -> dict[str, Any]:
    merge_summary: dict[str, Any] = {"num_merged_segments": 0, "merged_original_speakers": []}
    merged_by_original: dict[str, dict[str, Any]] = {}
    for segment in segments:
        if not segment.original_speaker_id:
            continue
        merge_summary["num_merged_segments"] = int(merge_summary["num_merged_segments"]) + 1
        row = merged_by_original.setdefault(
            segment.original_speaker_id,
            {"original_speaker_id": segment.original_speaker_id, "duration_sec": 0.0, "num_segments": 0, "target_speaker_counts": {}},
        )
        row["duration_sec"] = float(row["duration_sec"]) + segment.duration_sec
        row["num_segments"] = int(row["num_segments"]) + 1
        row["target_speaker_counts"][segment.speaker_id] = int(row["target_speaker_counts"].get(segment.speaker_id, 0)) + 1
    merge_summary["merged_original_speakers"] = [
        {
            **row,
            "duration_sec": round(float(row["duration_sec"]), 3),
            "target_speaker_counts": dict(row["target_speaker_counts"]),
        }
        for _, row in sorted(merged_by_original.items())
    ]
    return {
        "segmentation_rules": {
            "MIN_SEGMENT_SEC": MIN_SEGMENT_SEC,
            "PRESERVE_SPEAKER_MIN_TURN_SEC": PRESERVE_SPEAKER_MIN_TURN_SEC,
            "PRESERVE_SPEAKER_MIN_TOTAL_SEC": PRESERVE_SPEAKER_MIN_TOTAL_SEC,
            "MAX_SEGMENT_SEC": MAX_SEGMENT_SEC,
            "MERGE_GAP_SEC": MERGE_GAP_SEC,
            "PAD_LEFT_SEC": PAD_LEFT_SEC,
            "PAD_RIGHT_SEC": PAD_RIGHT_SEC,
            "TINY_SPEAKER_MAX_TOTAL_SEC": TINY_SPEAKER_MAX_TOTAL_SEC,
            "TINY_SPEAKER_MAX_FRACTION": TINY_SPEAKER_MAX_FRACTION,
            "TINY_SPEAKER_MAX_TURNS": TINY_SPEAKER_MAX_TURNS,
        },
        "speaker_fragment_merge_summary": merge_summary,
        "num_segments": len(segments),
        "segments": [segment.to_dict() for segment in segments],
    }


def _file_path_from_uri_or_store(uri: str, store: LocalArtifactStore | None) -> Path:
    validate_artifact_uri(uri)
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if store is None:
        raise AegisVoxContractError("LocalArtifactStore is required to resolve s3:// segment audio URIs")
    return store.local_path(uri)


def wav_duration_sec(path: str | Path) -> float:
    with wave.open(str(path), "rb") as src:
        return src.getnframes() / float(src.getframerate())


def extract_wav_segment(
    source_audio_uri: str,
    output_audio_uri: str,
    start_sec: float,
    end_sec: float,
    *,
    store: LocalArtifactStore | None = None,
) -> Path:
    """Extract a PCM WAV segment while preserving channels/sample width/rate."""

    source_path = _file_path_from_uri_or_store(source_audio_uri, store)
    output_path = _file_path_from_uri_or_store(output_audio_uri, store)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise AegisVoxContractError(f"source WAV not found: {source_path}")
    with wave.open(str(source_path), "rb") as src:
        frame_rate = src.getframerate()
        start_frame = max(0, int(round(start_sec * frame_rate)))
        end_frame = min(src.getnframes(), int(round(end_sec * frame_rate)))
        if end_frame <= start_frame:
            raise AegisVoxContractError("empty segment after frame conversion")
        src.setpos(start_frame)
        frames = src.readframes(end_frame - start_frame)
        params = src.getparams()
    with wave.open(str(output_path), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
    return output_path
