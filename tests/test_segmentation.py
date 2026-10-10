import math
import wave

from aegisvox.segmentation import (
    clean_diarization_turns,
    extract_wav_segment,
    load_diarization_turns,
    segments_payload,
    wav_duration_sec,
)
from aegisvox.storage import LocalArtifactStore


def test_clean_diarization_merges_splits_and_marks_overlap():
    turns = load_diarization_turns_payload = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
        {"speaker": "SPEAKER_00", "start": 1.2, "end": 2.0},  # merge gap 0.2
        {"speaker": "SPEAKER_01", "start": 1.8, "end": 2.8},  # overlap
        {"speaker": "SPEAKER_02", "start": 3.0, "end": 3.2},  # too short
        {"speaker": "SPEAKER_00", "start": 4.0, "end": 20.2},  # split
    ]
    from aegisvox.segmentation import DiarizationTurn

    segments = clean_diarization_turns([DiarizationTurn.from_dict(row) for row in turns], audio_duration_sec=21.0)
    assert segments[0].speaker_id == "SPEAKER_00"
    assert segments[0].start_sec == 0.0
    assert segments[0].end_sec == 1.8
    assert segments[0].overlap is False
    assert "clean_subspan_from_overlap_turn" in segments[0].flags
    overlap_segments = [segment for segment in segments if segment.overlap]
    assert overlap_segments
    assert overlap_segments[0].start_sec == 1.8
    assert overlap_segments[0].end_sec == 2.0
    assert overlap_segments[0].overlap_speaker_ids == ["SPEAKER_00", "SPEAKER_01"]
    assert overlap_segments[0].overlap_kind == "multi_speaker"
    assert all(segment.speaker_id != "SPEAKER_02" for segment in segments)
    assert any("split_long_turn" in segment.flags for segment in segments)
    payload = segments_payload(segments)
    assert payload["num_segments"] == len(segments)


def test_clean_diarization_preserves_short_secondary_speaker_identity():
    from aegisvox.segmentation import DiarizationTurn

    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
        {"speaker": "SPEAKER_01", "start": 0.50, "end": 0.66},
        {"speaker": "SPEAKER_01", "start": 1.20, "end": 1.38},
        {"speaker": "SPEAKER_01", "start": 2.00, "end": 2.20},
        {"speaker": "SPEAKER_02", "start": 4.00, "end": 4.05},
    ]

    segments = clean_diarization_turns([DiarizationTurn.from_dict(row) for row in turns], audio_duration_sec=5.0)

    by_speaker = {}
    for segment in segments:
        by_speaker.setdefault(segment.speaker_id, []).append(segment)
    assert set(by_speaker) == {"SPEAKER_00", "SPEAKER_01"}
    preserved = by_speaker["SPEAKER_01"]
    assert len(preserved) == 3
    assert all("short_speaker_turn" in segment.flags for segment in preserved)
    assert all("preserved_to_avoid_speaker_collapse" in segment.flags for segment in preserved)
    assert all(segment.overlap is True for segment in preserved)
    assert all("short_overlap_speaker_turn" in segment.flags for segment in preserved)
    assert "SPEAKER_02" not in by_speaker


def test_overlap_interval_splitting_preserves_long_clean_subspans():
    from aegisvox.segmentation import DiarizationTurn

    turns = [
        {"speaker": "SPEAKER_01", "start": 84.001, "end": 123.91},
        {"speaker": "SPEAKER_02", "start": 112.098, "end": 113.077},
    ]

    segments = clean_diarization_turns([DiarizationTurn.from_dict(row) for row in turns], audio_duration_sec=130.0)

    s1 = [segment for segment in segments if segment.speaker_id == "SPEAKER_01"]
    assert [(segment.start_sec, segment.end_sec, segment.overlap) for segment in s1] == [
        (84.001, 99.001, False),
        (99.001, 112.098, False),
        (112.098, 113.077, True),
        (113.077, 123.91, False),
    ]
    assert all(segment.overlap is False for segment in s1[:2] + s1[3:])
    assert all("clean_subspan_from_overlap_turn" in segment.flags for segment in s1[:2] + s1[3:])
    assert s1[2].overlap_speaker_ids == ["SPEAKER_01", "SPEAKER_02"]
    assert "true_overlap_interval" in s1[2].flags


def test_load_diarization_turns_accepts_segments_key(tmp_path):
    path = tmp_path / "diarization.raw.json"
    path.write_text('{"segments":[{"speaker_id":"S0","start_sec":0,"end_sec":1.2}]}')
    turns = load_diarization_turns(path)
    assert turns[0].speaker_id == "S0"


def test_extract_wav_segment_preserves_duration(tmp_path):
    source = tmp_path / "source.wav"
    frame_rate = 8000
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(b"\x00\x00" * frame_rate * 2)
    store = LocalArtifactStore(tmp_path / "artifacts")
    out_uri = "s3://aegisvox/runs/run_001/segments/seg_000001.source.wav"
    extract_wav_segment(f"file://{source}", out_uri, 0.5, 1.25, store=store)
    out_path = store.local_path(out_uri)
    assert out_path.exists()
    assert wav_duration_sec(out_path) == pytest_approx(0.75, abs=1 / frame_rate)


def pytest_approx(value, **kwargs):
    # Tiny helper avoids exposing pytest import in wave-only environments.
    import pytest

    return pytest.approx(value, **kwargs)


def test_expected_speaker_count_collapses_tiny_extra_speaker_cluster():
    from aegisvox.segmentation import DiarizationTurn

    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 60.0},
        {"speaker": "SPEAKER_01", "start": 60.5, "end": 90.0},
        # Tiny third-speaker artifact overlapping a main speaker backchannel.
        {"speaker": "SPEAKER_02", "start": 10.0, "end": 10.6},
        {"speaker": "SPEAKER_02", "start": 70.0, "end": 70.8},
    ]

    segments = clean_diarization_turns(
        [DiarizationTurn.from_dict(row) for row in turns],
        audio_duration_sec=100.0,
        expected_speaker_count=2,
    )

    speakers = {segment.speaker_id for segment in segments}
    assert speakers == {"SPEAKER_00", "SPEAKER_01"}
    merged = [segment for segment in segments if segment.original_speaker_id == "SPEAKER_02"]
    assert merged
    assert all("speaker_fragment_merged" in segment.flags for segment in merged)
    assert all(segment.speaker_merge_reason == "tiny_extra_speaker_cluster" for segment in merged)
    payload = segments_payload(segments)
    summary = payload["speaker_fragment_merge_summary"]
    assert summary["num_merged_segments"] >= 2
    assert summary["merged_original_speakers"][0]["original_speaker_id"] == "SPEAKER_02"


def test_expected_speaker_count_does_not_collapse_substantial_extra_speaker():
    from aegisvox.segmentation import DiarizationTurn

    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 60.0},
        {"speaker": "SPEAKER_01", "start": 60.5, "end": 120.0},
        {"speaker": "SPEAKER_02", "start": 121.0, "end": 150.0},
    ]

    segments = clean_diarization_turns(
        [DiarizationTurn.from_dict(row) for row in turns],
        audio_duration_sec=151.0,
        expected_speaker_count=2,
    )

    assert "SPEAKER_02" in {segment.speaker_id for segment in segments}
    assert not [segment for segment in segments if segment.original_speaker_id == "SPEAKER_02"]


def test_unknown_count_auto_collapses_likely_over_speaker_tiny_cluster():
    from aegisvox.segmentation import DiarizationTurn

    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 600.0},
        {"speaker": "SPEAKER_01", "start": 601.0, "end": 780.0},
        {"speaker": "SPEAKER_02", "start": 10.0, "end": 10.6},
        {"speaker": "SPEAKER_02", "start": 120.0, "end": 121.0},
        {"speaker": "SPEAKER_02", "start": 720.0, "end": 721.0},
    ]

    segments = clean_diarization_turns(
        [DiarizationTurn.from_dict(row) for row in turns],
        audio_duration_sec=800.0,
        expected_speaker_count=None,
    )

    assert {segment.speaker_id for segment in segments} == {"SPEAKER_00", "SPEAKER_01"}
    merged = [segment for segment in segments if segment.original_speaker_id == "SPEAKER_02"]
    assert len(merged) == 3
    assert all("speaker_fragment_merged" in segment.flags for segment in merged)
    assert {segment.speaker_id for segment in merged} <= {"SPEAKER_00", "SPEAKER_01"}
    summary = segments_payload(segments)["speaker_fragment_merge_summary"]
    assert summary["num_merged_segments"] == 3
    assert summary["merged_original_speakers"][0]["original_speaker_id"] == "SPEAKER_02"


def test_unknown_count_auto_preserves_real_low_duration_third_speaker_above_fraction():
    from aegisvox.segmentation import DiarizationTurn

    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 80.0},
        {"speaker": "SPEAKER_01", "start": 81.0, "end": 140.0},
        # Short in absolute terms, but a meaningful share of the conversation.
        {"speaker": "SPEAKER_02", "start": 141.0, "end": 150.0},
    ]

    segments = clean_diarization_turns(
        [DiarizationTurn.from_dict(row) for row in turns],
        audio_duration_sec=151.0,
        expected_speaker_count=None,
    )

    assert "SPEAKER_02" in {segment.speaker_id for segment in segments}
    assert not [segment for segment in segments if segment.original_speaker_id == "SPEAKER_02"]
