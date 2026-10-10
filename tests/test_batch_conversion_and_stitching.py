import shutil
import struct
import wave
from pathlib import Path

import pytest

from aegisvox.batch_conversion import BatchVoiceConversionService
from aegisvox.stitching import StitchingError, StitchingPolicy, StitchingService
from aegisvox.storage import LocalArtifactStore


class CopyBackend:
    backend_name = "copy-test-backend"
    backend_version = "test-only"

    def preflight(self):
        return {"ok": True}

    def convert(self, *, source_wav: Path, target_wav: Path, output_wav: Path, options):
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_wav, output_wav)
        return {"test_only": True}


def write_wav(path: Path, seconds: float, amplitude: int = 1000, frame_rate: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = int(amplitude).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(sample * int(seconds * frame_rate))


def read_pcm16(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return list(struct.unpack("<" + "h" * (len(frames) // 2), frames)) if frames else []


def setup_batch_inputs(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    seg1 = "s3://aegisvox/runs/run_001/segments/seg_000001.wav"
    seg2 = "s3://aegisvox/runs/run_001/segments/seg_000002.wav"
    target = "s3://aegisvox/pool/target.wav"
    write_wav(store.local_path(seg1), 0.5, amplitude=1000)
    write_wav(store.local_path(seg2), 0.5, amplitude=2000)
    write_wav(store.local_path(target), 0.5, amplitude=500)
    segments_uri = "s3://aegisvox/runs/run_001/segments.cleaned.json"
    assignments_uri = "s3://aegisvox/runs/run_001/speaker_target_map.json"
    store.write_json(
        segments_uri,
        {
            "segments": [
                {
                    "segment_id": "seg_000001",
                    "speaker_id": "SPEAKER_00",
                    "segment_audio_uri": seg1,
                    "start_sec": 0.0,
                    "end_sec": 0.5,
                    "padded_start_sec": 0.0,
                    "padded_end_sec": 0.5,
                    "overlap": False,
                },
                {
                    "segment_id": "seg_000002",
                    "speaker_id": "SPEAKER_00",
                    "segment_audio_uri": seg2,
                    "start_sec": 0.5,
                    "end_sec": 1.0,
                    "padded_start_sec": 0.5,
                    "padded_end_sec": 1.0,
                    "overlap": True,
                },
            ]
        },
    )
    store.write_json(
        assignments_uri,
        {
            "assignments": {
                "SPEAKER_00": {
                    "target_reference_audio_uri": target,
                    "target_voice_id": "target_a",
                    "target_voice_sample_id": "sample_a",
                }
            }
        },
    )
    return store, segments_uri, assignments_uri


def test_batch_conversion_converts_non_overlap_and_skips_overlap(tmp_path):
    store, segments_uri, assignments_uri = setup_batch_inputs(tmp_path)
    service = BatchVoiceConversionService(store, CopyBackend())
    output_uri = "s3://aegisvox/runs/run_001/segments.converted.json"
    payload = service.convert_batch_to_artifact(
        segments_uri=segments_uri,
        assignments_uri=assignments_uri,
        output_segments_uri=output_uri,
        output_prefix_uri="s3://aegisvox/runs/run_001/converted_segments",
    )
    assert payload["num_converted"] == 1
    assert payload["num_skipped_overlap"] == 1
    assert payload["segments"][0]["conversion_status"] == "converted"
    assert payload["segments"][1]["conversion_status"] == "skipped_overlap"
    assert store.exists(payload["segments"][0]["converted_audio_uri"])




class CountingCopyBackend(CopyBackend):
    def __init__(self):
        self.calls = 0

    def convert(self, *, source_wav: Path, target_wav: Path, output_wav: Path, options):
        self.calls += 1
        return super().convert(source_wav=source_wav, target_wav=target_wav, output_wav=output_wav, options=options)


def test_batch_conversion_resumes_existing_outputs_without_recomputing(tmp_path):
    store, segments_uri, assignments_uri = setup_batch_inputs(tmp_path)
    existing_uri = "s3://aegisvox/runs/run_001/converted_segments/seg_000001.converted.wav"
    write_wav(store.local_path(existing_uri), 0.5, amplitude=1234)
    backend = CountingCopyBackend()

    payload = BatchVoiceConversionService(store, backend).convert_batch_to_artifact(
        segments_uri=segments_uri,
        assignments_uri=assignments_uri,
        output_segments_uri="s3://aegisvox/runs/run_001/segments.converted.json",
        output_prefix_uri="s3://aegisvox/runs/run_001/converted_segments",
    )

    assert backend.calls == 0
    assert payload["num_converted"] == 1
    assert payload["num_resumed_existing"] == 1
    assert payload["conversion_parallelism"]["resume_existing"] is True
    assert payload["segments"][0]["conversion_status"] == "converted"
    assert payload["segments"][0]["resumed_from_existing_output"] is True
    assert read_pcm16(store.local_path(existing_uri))[0] == 1234


def test_stitching_reconstructs_timeline(tmp_path):
    store, segments_uri, assignments_uri = setup_batch_inputs(tmp_path)
    batch = BatchVoiceConversionService(store, CopyBackend()).convert_batch_to_artifact(
        segments_uri=segments_uri,
        assignments_uri=assignments_uri,
        output_segments_uri="s3://aegisvox/runs/run_001/segments.converted.json",
        output_prefix_uri="s3://aegisvox/runs/run_001/converted_segments",
    )
    report = StitchingService(store).stitch_to_artifact(
        converted_segments_uri="s3://aegisvox/runs/run_001/segments.converted.json",
        output_audio_uri="s3://aegisvox/runs/run_001/changed_identity.wav",
        stitch_report_uri="s3://aegisvox/runs/run_001/stitch_report.json",
        audio_duration_sec=1.0,
    )
    assert report["num_stitched"] == 1
    assert report["num_not_stitched"] == 1
    out = store.local_path("s3://aegisvox/runs/run_001/changed_identity.wav")
    assert out.exists()
    with wave.open(str(out), "rb") as wav:
        assert wav.getnframes() == 8000


def test_stitching_aligns_duration_applies_fades_and_loudness_metadata(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_uri = "s3://aegisvox/runs/run_002/segments/seg_000001.source.wav"
    converted_uri = "s3://aegisvox/runs/run_002/converted_segments/seg_000001.converted.wav"
    converted_manifest_uri = "s3://aegisvox/runs/run_002/segments.converted.json"
    # Source is quieter and exactly 0.5s; converted is longer/louder so the DSP
    # path must trim, apply an RMS gain below 1.0, and fade segment boundaries.
    write_wav(store.local_path(source_uri), 0.5, amplitude=500)
    write_wav(store.local_path(converted_uri), 0.7, amplitude=2000)
    store.write_json(
        converted_manifest_uri,
        {
            "segments": [
                {
                    "segment_id": "seg_000001",
                    "speaker_id": "SPEAKER_00",
                    "source_audio_uri": source_uri,
                    "converted_audio_uri": converted_uri,
                    "conversion_status": "converted",
                    "start_sec": 0.0,
                    "end_sec": 0.5,
                    "padded_start_sec": 0.0,
                    "padded_end_sec": 0.5,
                }
            ]
        },
    )

    report = StitchingService(store).stitch_to_artifact(
        converted_segments_uri=converted_manifest_uri,
        output_audio_uri="s3://aegisvox/runs/run_002/changed_identity.wav",
        stitch_report_uri="s3://aegisvox/runs/run_002/stitch_report.json",
        audio_duration_sec=0.5,
    )

    row = report["segments"][0]
    assert row["duration_alignment"] == "trimmed"
    assert row["original_duration_drift_sec"] == pytest.approx(0.2)
    assert row["duration_drift_sec"] == pytest.approx(0.0)
    assert row["loudness_normalization"] == "rms_matched"
    assert row["loudness_gain"] < 1.0
    assert row["fade_ms"] == 5.0
    assert report["dsp_policy"]["align_duration"] is True
    with wave.open(str(store.local_path("s3://aegisvox/runs/run_002/changed_identity.wav")), "rb") as wav:
        assert wav.getnframes() == 4000


def test_stitching_suppresses_internal_fades_for_adjacent_segments(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source1_uri = "s3://aegisvox/runs/run_003/segments/seg_000001.source.wav"
    source2_uri = "s3://aegisvox/runs/run_003/segments/seg_000002.source.wav"
    converted1_uri = "s3://aegisvox/runs/run_003/converted_segments/seg_000001.converted.wav"
    converted2_uri = "s3://aegisvox/runs/run_003/converted_segments/seg_000002.converted.wav"
    manifest_uri = "s3://aegisvox/runs/run_003/segments.converted.json"
    write_wav(store.local_path(source1_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(source2_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(converted1_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(converted2_uri), 0.5, amplitude=1000)
    store.write_json(
        manifest_uri,
        {
            "segments": [
                {
                    "segment_id": "seg_000001",
                    "speaker_id": "SPEAKER_00",
                    "source_audio_uri": source1_uri,
                    "converted_audio_uri": converted1_uri,
                    "conversion_status": "converted",
                    "start_sec": 0.0,
                    "end_sec": 0.5,
                    "padded_start_sec": 0.0,
                    "padded_end_sec": 0.5,
                },
                {
                    "segment_id": "seg_000002",
                    "speaker_id": "SPEAKER_00",
                    "source_audio_uri": source2_uri,
                    "converted_audio_uri": converted2_uri,
                    "conversion_status": "converted",
                    "start_sec": 0.5,
                    "end_sec": 1.0,
                    "padded_start_sec": 0.5,
                    "padded_end_sec": 1.0,
                },
            ]
        },
    )

    report = StitchingService(store).stitch_to_artifact(
        converted_segments_uri=manifest_uri,
        output_audio_uri="s3://aegisvox/runs/run_003/changed_identity.wav",
        stitch_report_uri="s3://aegisvox/runs/run_003/stitch_report.json",
        audio_duration_sec=1.0,
    )

    first, second = report["segments"]
    assert first["boundary_smoothing"] == "suppressed_internal_fade"
    assert first["fade_in_applied"] is True
    assert first["fade_out_applied"] is False
    assert second["boundary_smoothing"] == "suppressed_internal_fade"
    assert second["fade_in_applied"] is False
    assert second["fade_out_applied"] is True
    assert report["dsp_policy"]["suppress_internal_boundary_fades"] is True
    assert report["dsp_policy"]["internal_boundary_gap_ms"] == 60.0
    samples = read_pcm16(store.local_path("s3://aegisvox/runs/run_003/changed_identity.wav"))
    # The internal boundary should stay at full amplitude instead of fading
    # both adjacent segments down and causing a perceptual dip/cut.
    assert samples[3999] == 1000
    assert samples[4000] == 1000


def test_stitching_fails_without_converted_segments(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    uri = "s3://aegisvox/runs/run_001/segments.converted.json"
    store.write_json(uri, {"segments": [{"conversion_status": "skipped_overlap"}]})
    with pytest.raises(StitchingError, match="no converted segments"):
        StitchingService(store).stitch_to_artifact(
            converted_segments_uri=uri,
            output_audio_uri="s3://aegisvox/runs/run_001/changed_identity.wav",
            stitch_report_uri="s3://aegisvox/runs/run_001/stitch_report.json",
        )

class FailingBackend:
    backend_name = "failing-test-backend"
    backend_version = "test-only"

    def preflight(self):
        return {"ok": True}

    def convert(self, *, source_wav: Path, target_wav: Path, output_wav: Path, options):
        raise RuntimeError("OPENAI_API_KEY=secret-test-value patient@example.com")


def test_batch_conversion_redacts_per_segment_failure_reason(tmp_path):
    store, segments_uri, assignments_uri = setup_batch_inputs(tmp_path)
    payload = BatchVoiceConversionService(store, FailingBackend()).convert_batch_to_artifact(
        segments_uri=segments_uri,
        assignments_uri=assignments_uri,
        output_segments_uri="s3://aegisvox/runs/run_001/segments.converted.json",
        output_prefix_uri="s3://aegisvox/runs/run_001/converted_segments",
    )

    failed = [row for row in payload["segments"] if row["conversion_status"] == "failed"][0]
    blob = str(failed)
    assert "sk-test" not in blob
    assert "patient@example.com" not in blob
    assert "[REDACTED]" in blob


def test_batch_conversion_can_convert_overlap_for_review_candidate(tmp_path):
    store, segments_uri, assignments_uri = setup_batch_inputs(tmp_path)
    payload = BatchVoiceConversionService(store, CopyBackend()).convert_batch_to_artifact(
        segments_uri=segments_uri,
        assignments_uri=assignments_uri,
        output_segments_uri="s3://aegisvox/runs/run_001/segments.converted.review.json",
        output_prefix_uri="s3://aegisvox/runs/run_001/converted_segments_review",
        overlap_policy="convert_and_flag",
    )

    assert payload["overlap_policy"] == "convert_and_flag"
    assert payload["num_converted"] == 2
    assert payload["num_skipped_overlap"] == 0
    assert payload["num_converted_overlap"] == 1
    overlap_row = [row for row in payload["segments"] if row.get("overlap_converted")][0]
    assert overlap_row["conversion_status"] == "converted"
    assert overlap_row["overlap_policy"] == "convert_and_flag"
    assert "engineering review" in overlap_row["overlap_claim_boundary"]


def test_batch_conversion_parallel_workers_preserve_order_and_assign_devices(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    target = "s3://aegisvox/pool/target.wav"
    write_wav(store.local_path(target), 0.5, amplitude=500)
    rows = []
    for idx in range(4):
        segment_id = f"seg_{idx + 1:06d}"
        uri = f"s3://aegisvox/runs/run_parallel/segments/{segment_id}.wav"
        write_wav(store.local_path(uri), 0.2, amplitude=1000 + idx)
        rows.append(
            {
                "segment_id": segment_id,
                "speaker_id": "SPEAKER_00",
                "segment_audio_uri": uri,
                "start_sec": idx * 0.2,
                "end_sec": (idx + 1) * 0.2,
                "overlap": False,
            }
        )
    segments_uri = "s3://aegisvox/runs/run_parallel/segments.cleaned.json"
    assignments_uri = "s3://aegisvox/runs/run_parallel/speaker_target_map.json"
    store.write_json(segments_uri, {"segments": rows})
    store.write_json(assignments_uri, {"assignments": {"SPEAKER_00": {"target_reference_audio_uri": target}}})

    payload = BatchVoiceConversionService(store, CopyBackend()).convert_batch_to_artifact(
        segments_uri=segments_uri,
        assignments_uri=assignments_uri,
        output_segments_uri="s3://aegisvox/runs/run_parallel/segments.converted.json",
        output_prefix_uri="s3://aegisvox/runs/run_parallel/converted_segments",
        options={"num_workers": 2, "cuda_devices": "0,1"},
    )

    assert payload["conversion_parallelism"] == {
        "num_workers": 2,
        "cuda_devices": ["0", "1"],
        "strategy": "segment_threadpool_preserve_order",
        "resume_existing": True,
    }
    assert [row["segment_id"] for row in payload["segments"]] == [row["segment_id"] for row in rows]
    assert [row["cuda_visible_devices"] for row in payload["segments"]] == ["0", "1", "0", "1"]
    assert payload["num_converted"] == 4


def test_stitching_local_similarity_boundary_alignment_shifts_into_overlap(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source1_uri = "s3://aegisvox/runs/run_align/segments/seg_000001.source.wav"
    source2_uri = "s3://aegisvox/runs/run_align/segments/seg_000002.source.wav"
    converted1_uri = "s3://aegisvox/runs/run_align/converted_segments/seg_000001.converted.wav"
    converted2_uri = "s3://aegisvox/runs/run_align/converted_segments/seg_000002.converted.wav"
    manifest_uri = "s3://aegisvox/runs/run_align/segments.converted.json"
    write_wav(store.local_path(source1_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(source2_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(converted1_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(converted2_uri), 0.5, amplitude=1000)
    store.write_json(
        manifest_uri,
        {
            "segments": [
                {
                    "segment_id": "seg_000001",
                    "speaker_id": "SPEAKER_00",
                    "source_audio_uri": source1_uri,
                    "converted_audio_uri": converted1_uri,
                    "conversion_status": "converted",
                    "start_sec": 0.0,
                    "end_sec": 0.5,
                    "padded_start_sec": 0.0,
                    "padded_end_sec": 0.5,
                },
                {
                    "segment_id": "seg_000002",
                    "speaker_id": "SPEAKER_00",
                    "source_audio_uri": source2_uri,
                    "converted_audio_uri": converted2_uri,
                    "conversion_status": "converted",
                    "start_sec": 0.5,
                    "end_sec": 1.0,
                    "padded_start_sec": 0.5,
                    "padded_end_sec": 1.0,
                },
            ]
        },
    )

    report = StitchingService(
        store,
        policy=StitchingPolicy(
            fade_ms=0.0,
            boundary_alignment="local_similarity",
            boundary_alignment_search_ms=10.0,
            boundary_alignment_min_overlap_ms=5.0,
            boundary_alignment_min_score=0.1,
        ),
    ).stitch_to_artifact(
        converted_segments_uri=manifest_uri,
        output_audio_uri="s3://aegisvox/runs/run_align/changed_identity.wav",
        stitch_report_uri="s3://aegisvox/runs/run_align/stitch_report.json",
        audio_duration_sec=1.0,
    )

    first, second = report["segments"]
    assert first["boundary_alignment_result"]["status"] == "no_search_window"
    assert second["boundary_alignment_result"]["status"] == "aligned"
    assert second["boundary_alignment_result"]["shift_frames"] == 40
    assert second["placed_start_frame"] == second["nominal_start_frame"] - 40
    assert second["overlap_add_applied"] is True
    assert second["overlap_added_frames"] == 40
    assert report["dsp_policy"]["boundary_alignment"] == "local_similarity"


def test_stitching_overlap_add_reports_timeline_overlap_without_alignment(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source1_uri = "s3://aegisvox/runs/run_overlap/segments/seg_000001.source.wav"
    source2_uri = "s3://aegisvox/runs/run_overlap/segments/seg_000002.source.wav"
    converted1_uri = "s3://aegisvox/runs/run_overlap/converted_segments/seg_000001.converted.wav"
    converted2_uri = "s3://aegisvox/runs/run_overlap/converted_segments/seg_000002.converted.wav"
    manifest_uri = "s3://aegisvox/runs/run_overlap/segments.converted.json"
    write_wav(store.local_path(source1_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(source2_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(converted1_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(converted2_uri), 0.5, amplitude=1000)
    store.write_json(
        manifest_uri,
        {
            "segments": [
                {
                    "segment_id": "seg_000001",
                    "speaker_id": "SPEAKER_00",
                    "source_audio_uri": source1_uri,
                    "converted_audio_uri": converted1_uri,
                    "conversion_status": "converted",
                    "padded_start_sec": 0.0,
                    "padded_end_sec": 0.5,
                },
                {
                    "segment_id": "seg_000002",
                    "speaker_id": "SPEAKER_00",
                    "source_audio_uri": source2_uri,
                    "converted_audio_uri": converted2_uri,
                    "conversion_status": "converted",
                    "padded_start_sec": 0.49,
                    "padded_end_sec": 0.99,
                },
            ]
        },
    )

    report = StitchingService(store, policy=StitchingPolicy(fade_ms=0.0)).stitch_to_artifact(
        converted_segments_uri=manifest_uri,
        output_audio_uri="s3://aegisvox/runs/run_overlap/changed_identity.wav",
        stitch_report_uri="s3://aegisvox/runs/run_overlap/stitch_report.json",
        audio_duration_sec=1.0,
    )

    second = report["segments"][1]
    assert second["overlap_add_applied"] is True
    assert second["overlap_added_frames"] == 80
    assert second["overlap_added_duration_sec"] == 0.01
    assert second["boundary_alignment"] == "none"
    assert report["dsp_policy"]["overlap_add"] is True
