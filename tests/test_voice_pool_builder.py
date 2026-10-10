from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from aegisvox.storage import LocalArtifactStore
from aegisvox.voice_pool import VoicePoolIndex, VoiceSearchRequest
from aegisvox.voice_pool_builder import VoicePoolBuildError, VoicePoolBuildService, load_source_records


class DeterministicEmbeddingBackend:
    backend_name = "deterministic-test-ecapa"
    backend_version = "test-only"

    def embed_wav(self, wav_path: Path) -> list[float]:
        value = (sum(wav_path.name.encode("utf-8")) % 17) + 1
        return [float(value), 1.0, 0.5]


def write_wav(path: Path, seconds: float = 1.2, frame_rate: int = 16000, amplitude: int = 1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = int(amplitude).to_bytes(2, byteorder="little", signed=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(frame * int(seconds * frame_rate))


def test_load_source_records_accepts_file_paths(tmp_path: Path):
    audio = tmp_path / "a.wav"
    write_wav(audio)
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(json.dumps({"voice_id": "v1", "path": str(audio), "language": "en"}) + "\n")
    rows = load_source_records(manifest)
    assert rows[0].voice_id == "v1"
    assert rows[0].audio_uri.startswith("file://")
    assert rows[0].sample_id.startswith("v1__sample")


def test_voice_pool_builder_writes_manifest_index_and_quality_report(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    audio1 = tmp_path / "source" / "a.wav"
    audio2 = tmp_path / "source" / "b.wav"
    write_wav(audio1, seconds=1.5)
    write_wav(audio2, seconds=1.6, amplitude=2000)
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"voice_id": "voice_a", "sample_id": "sample_a", "path": str(audio1), "language": "en", "license": "test"}),
                json.dumps({"voice_id": "voice_b", "sample_id": "sample_b", "path": str(audio2), "language": "en", "license": "test"}),
            ]
        )
        + "\n"
    )
    service = VoicePoolBuildService(store, embedding_backend=DeterministicEmbeddingBackend())
    payload = service.build_from_manifest_path(
        manifest,
        output_prefix_uri="s3://aegisvox/voice-pools/test_pool",
        manifest_uri="s3://aegisvox/voice-pools/test_pool/manifest.json",
        index_uri="s3://aegisvox/voice-pools/test_pool/embedding_index.json",
        quality_report_uri="s3://aegisvox/voice-pools/test_pool/quality_report.json",
    )
    assert payload["num_accepted"] == 2
    assert payload["num_rejected"] == 0
    manifest_path = store.local_path("s3://aegisvox/voice-pools/test_pool/manifest.json")
    index_path = store.local_path("s3://aegisvox/voice-pools/test_pool/embedding_index.json")
    quality = store.read_json("s3://aegisvox/voice-pools/test_pool/quality_report.json")
    assert quality["status"] == "completed"
    index = VoicePoolIndex.from_paths(manifest_path, index_path)
    result = index.search(VoiceSearchRequest(voice_pool_id="openswara_clinical_v1", query_embedding=[1.0, 0.0, 0.0], top_k=1))
    assert len(result) == 1
    assert store.exists("s3://aegisvox/voice-pools/test_pool/normalized_audio/sample_a.wav")
    assert store.exists("s3://aegisvox/voice-pools/test_pool/embeddings/sample_a.json")


def test_voice_pool_builder_rejects_bad_qc_and_reports(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    write_wav(good, seconds=1.2)
    write_wav(bad, seconds=0.1)
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(
        json.dumps({"voice_id": "good", "sample_id": "good_sample", "path": str(good)})
        + "\n"
        + json.dumps({"voice_id": "bad", "sample_id": "bad_sample", "path": str(bad)})
        + "\n"
    )
    payload = VoicePoolBuildService(store, embedding_backend=DeterministicEmbeddingBackend()).build_from_manifest_path(
        manifest,
        output_prefix_uri="s3://aegisvox/voice-pools/test_pool",
        manifest_uri="s3://aegisvox/voice-pools/test_pool/manifest.json",
        index_uri="s3://aegisvox/voice-pools/test_pool/embedding_index.json",
        quality_report_uri="s3://aegisvox/voice-pools/test_pool/quality_report.json",
    )
    assert payload["num_accepted"] == 1
    assert payload["num_rejected"] == 1
    report = store.read_json("s3://aegisvox/voice-pools/test_pool/quality_report.json")
    rejected = [row for row in report["rows"] if not row["qc"]["accepted"]]
    assert rejected[0]["qc"]["reason_codes"] == ["too_short"]


def test_voice_pool_builder_fails_when_all_rows_rejected(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    bad = tmp_path / "bad.wav"
    write_wav(bad, seconds=0.1)
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(json.dumps({"voice_id": "bad", "sample_id": "bad_sample", "path": str(bad)}) + "\n")
    with pytest.raises(VoicePoolBuildError, match="all voice-pool source records failed QC"):
        VoicePoolBuildService(store, embedding_backend=DeterministicEmbeddingBackend()).build_from_manifest_path(
            manifest,
            output_prefix_uri="s3://aegisvox/voice-pools/test_pool",
            manifest_uri="s3://aegisvox/voice-pools/test_pool/manifest.json",
            index_uri="s3://aegisvox/voice-pools/test_pool/embedding_index.json",
            quality_report_uri="s3://aegisvox/voice-pools/test_pool/quality_report.json",
        )
    assert store.read_json("s3://aegisvox/voice-pools/test_pool/quality_report.json")["status"] == "failed"


def test_native_wav_normalizer_resamples_without_ffmpeg(tmp_path: Path, monkeypatch):
    from aegisvox.voice_pool_builder import FFMpegAudioNormalizer, qc_wav
    monkeypatch.setattr("shutil.which", lambda name: None)
    src = tmp_path / "src_24k.wav"
    out = tmp_path / "out_16k.wav"
    write_wav(src, seconds=1.0, frame_rate=24000)
    metadata = FFMpegAudioNormalizer(sample_rate=16000).normalize_to_wav(src, out)
    assert metadata["mode"] == "native_pcm_wav"
    qc = qc_wav(out, min_duration_sec=0.5, max_duration_sec=2.0, required_sample_rate=16000)
    assert qc.accepted
    assert qc.sample_rate == 16000


def test_voice_pool_builder_reuses_existing_normalized_audio_and_embedding(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    audio = tmp_path / "source" / "a.wav"
    write_wav(audio, seconds=1.5)
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(json.dumps({"voice_id": "voice_a", "sample_id": "sample_a", "path": str(audio), "language": "en"}) + "\n")
    service = VoicePoolBuildService(store, embedding_backend=DeterministicEmbeddingBackend())
    kwargs = dict(
        output_prefix_uri="s3://aegisvox/voice-pools/test_pool",
        manifest_uri="s3://aegisvox/voice-pools/test_pool/manifest.json",
        index_uri="s3://aegisvox/voice-pools/test_pool/embedding_index.json",
        quality_report_uri="s3://aegisvox/voice-pools/test_pool/quality_report.json",
    )
    first = service.build_from_manifest_path(manifest, **kwargs)
    assert first["num_accepted"] == 1

    class FailingEmbeddingBackend:
        backend_name = "failing-test-ecapa"
        backend_version = "test-only"

        def embed_wav(self, wav_path: Path) -> list[float]:
            raise AssertionError("existing embedding should have been reused")

    second = VoicePoolBuildService(store, embedding_backend=FailingEmbeddingBackend()).build_from_manifest_path(manifest, **kwargs)
    assert second["num_accepted"] == 1
    quality = store.read_json("s3://aegisvox/voice-pools/test_pool/quality_report.json")
    assert quality["rows"][0]["reused_embedding"] is True
    assert quality["rows"][0]["reused_normalized_audio"] is True
