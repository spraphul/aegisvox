import wave
from pathlib import Path

import pytest

from aegisvox.privacy_metrics import SpeakerPrivacyMetricsService, SpeakerPrivacyMetricsSetupError
from aegisvox.storage import LocalArtifactStore


class PathEmbeddingBackend:
    backend_name = "fake-path-embedding"
    backend_version = "test"

    def embed_wav(self, wav_path: Path):
        name = wav_path.name.lower()
        if "target" in name or "s1.wav" in name:
            return [0.0, 1.0]
        return [1.0, 0.0]


def write_wav(path: Path, seconds: float = 0.5, frame_rate: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(b"\x01\x00" * int(seconds * frame_rate))


def test_privacy_metrics_requires_embedding_backend(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(SpeakerPrivacyMetricsSetupError, match="require a real speaker embedding backend"):
        SpeakerPrivacyMetricsService(store).compute_to_artifact(
            speaker_profiles_uri="s3://aegisvox/runs/run_x/speaker_profiles.json",
            assignments_uri="s3://aegisvox/runs/run_x/speaker_target_map.json",
            converted_segments_uri="s3://aegisvox/runs/run_x/segments.converted.json",
            output_metrics_uri="s3://aegisvox/runs/run_x/speaker_privacy_metrics.json",
            output_prefix_uri="s3://aegisvox/runs/run_x/privacy_metrics",
        )


def test_privacy_metrics_computes_source_and_target_similarity(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_embedding_uri = "s3://aegisvox/runs/run_x/speaker_embeddings/S1.json"
    target_audio_uri = "s3://aegisvox/runs/run_x/target_ref_target.wav"
    converted_audio_uri = "s3://aegisvox/runs/run_x/converted_segments/seg1.wav"
    store.write_json(source_embedding_uri, {"embedding": [1.0, 0.0]})
    write_wav(store.local_path(target_audio_uri))
    write_wav(store.local_path(converted_audio_uri))
    speaker_profiles_uri = "s3://aegisvox/runs/run_x/speaker_profiles.json"
    assignments_uri = "s3://aegisvox/runs/run_x/speaker_target_map.json"
    converted_segments_uri = "s3://aegisvox/runs/run_x/segments.converted.json"
    output_uri = "s3://aegisvox/runs/run_x/speaker_privacy_metrics.json"
    store.write_json(
        speaker_profiles_uri,
        {"speakers": {"S1": {"source_embedding_uri": source_embedding_uri, "source_audio_uri": "s3://aegisvox/runs/run_x/source.wav"}}},
    )
    store.write_json(
        assignments_uri,
        {"assignments": {"S1": {"target_reference_audio_uri": target_audio_uri}}},
    )
    store.write_json(
        converted_segments_uri,
        {"segments": [{"segment_id": "seg1", "speaker_id": "S1", "converted_audio_uri": converted_audio_uri, "conversion_status": "converted"}]},
    )

    payload = SpeakerPrivacyMetricsService(store, embedding_backend=PathEmbeddingBackend()).compute_to_artifact(
        speaker_profiles_uri=speaker_profiles_uri,
        assignments_uri=assignments_uri,
        converted_segments_uri=converted_segments_uri,
        output_metrics_uri=output_uri,
        output_prefix_uri="s3://aegisvox/runs/run_x/privacy_metrics",
    )

    row = payload["speaker_metrics"][0]
    assert row["source_converted_similarity"] == pytest.approx(0.0)
    assert row["target_converted_similarity"] == pytest.approx(1.0)
    assert payload["summary"]["num_computed"] == 1
    assert store.exists(row["converted_audio_uri"])
    assert store.exists(row["converted_embedding_uri"])
    assert store.exists(output_uri)
