import wave
from pathlib import Path

import pytest

from aegisvox.schema import AegisVoxContractError
from aegisvox.speaker_profile import SpeakerProfilingService, SpeakerProfilingSetupError, concatenate_wav_segments
from aegisvox.storage import LocalArtifactStore


class FakeEmbeddingBackend:
    backend_name = "fake-test-backend"
    backend_version = "test"

    def __init__(self):
        self.last_wav_path: Path | None = None

    def embed_wav(self, wav_path: Path):
        assert wav_path.exists()
        self.last_wav_path = wav_path
        return [0.1, 0.2, 0.3]


def write_wav(path: Path, seconds: float, amplitude: int = 1000, frame_rate: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = int(seconds * frame_rate)
    sample = int(amplitude).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(sample * nframes)


def test_concatenate_wav_segments(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    out = tmp_path / "out.wav"
    write_wav(a, 0.5)
    write_wav(b, 0.25)
    duration = concatenate_wav_segments([a, b], out)
    assert duration == pytest.approx(0.75)
    with wave.open(str(out), "rb") as wav:
        assert wav.getnframes() == 6000


def test_speaker_profile_requires_embedding_backend(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    service = SpeakerProfilingService(store)
    with pytest.raises(SpeakerProfilingSetupError, match="requires a real speaker embedding backend"):
        service.profile_to_artifact(
            "s3://aegisvox/runs/run_001/segments.cleaned.json",
            "s3://aegisvox/runs/run_001/speaker_profiles.json",
            output_prefix_uri="s3://aegisvox/runs/run_001",
        )


def test_speaker_profile_groups_segments_and_writes_embeddings(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    seg_a_uri = "s3://aegisvox/runs/run_001/segments/seg_a.wav"
    seg_b_uri = "s3://aegisvox/runs/run_001/segments/seg_b.wav"
    seg_c_uri = "s3://aegisvox/runs/run_001/segments/seg_c.wav"
    write_wav(store.local_path(seg_a_uri), 0.5, amplitude=1000)
    write_wav(store.local_path(seg_b_uri), 0.25, amplitude=1000)
    write_wav(store.local_path(seg_c_uri), 0.75, amplitude=50)
    segments_uri = "s3://aegisvox/runs/run_001/segments.cleaned.json"
    store.write_json(
        segments_uri,
        {
            "segments": [
                {"segment_id": "seg_a", "speaker_id": "SPEAKER_00", "segment_audio_uri": seg_a_uri},
                {"segment_id": "seg_b", "speaker_id": "SPEAKER_00", "segment_audio_uri": seg_b_uri},
                {"segment_id": "seg_c", "speaker_id": "SPEAKER_01", "segment_audio_uri": seg_c_uri},
            ]
        },
    )
    service = SpeakerProfilingService(store, embedding_backend=FakeEmbeddingBackend())
    payload = service.profile_to_artifact(
        segments_uri,
        "s3://aegisvox/runs/run_001/speaker_profiles.json",
        output_prefix_uri="s3://aegisvox/runs/run_001",
    )
    assert payload["num_speakers"] == 2
    s0 = payload["speakers"]["SPEAKER_00"]
    assert s0["duration_sec"] == pytest.approx(0.75)
    assert s0["energy"] == "calm"
    embedding = store.read_json(s0["source_embedding_uri"])
    assert embedding["embedding"] == [0.1, 0.2, 0.3]
    assert store.exists(s0["source_audio_uri"])
    s1 = payload["speakers"]["SPEAKER_01"]
    assert s1["metadata"]["profile_reliability"] == "insufficient_duration"
    assert s0["metadata"]["profile_reliability"] == "insufficient_duration"


def test_speaker_profile_records_short_turn_reliability_metadata(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    seg_uri = "s3://aegisvox/runs/run_short/segments/seg_short.wav"
    write_wav(store.local_path(seg_uri), 0.32, amplitude=1000)
    segments_uri = "s3://aegisvox/runs/run_short/segments.cleaned.json"
    store.write_json(
        segments_uri,
        {
            "segments": [
                {
                    "segment_id": "seg_short",
                    "speaker_id": "SPEAKER_01",
                    "segment_audio_uri": seg_uri,
                    "flags": ["short_speaker_turn", "preserved_to_avoid_speaker_collapse"],
                }
            ]
        },
    )
    service = SpeakerProfilingService(store, embedding_backend=FakeEmbeddingBackend())
    payload = service.profile_to_artifact(
        segments_uri,
        "s3://aegisvox/runs/run_short/speaker_profiles.json",
        output_prefix_uri="s3://aegisvox/runs/run_short",
    )

    profile = payload["speakers"]["SPEAKER_01"]
    assert profile["duration_sec"] == pytest.approx(0.32)
    assert profile["metadata"]["profile_reliability"] == "insufficient_duration"
    assert profile["metadata"]["num_short_speaker_turns"] == 1


def test_speaker_profile_caps_embedding_audio_for_long_speakers(monkeypatch, tmp_path):
    monkeypatch.setenv("AEGISVOX_SPEAKER_PROFILE_MAX_EMBED_SECONDS", "0.6")
    store = LocalArtifactStore(tmp_path / "artifacts")
    rows = []
    for idx in range(3):
        uri = f"s3://aegisvox/runs/run_cap/segments/seg_{idx}.wav"
        write_wav(store.local_path(uri), 0.5, amplitude=1000)
        rows.append({"segment_id": f"seg_{idx}", "speaker_id": "SPEAKER_00", "segment_audio_uri": uri})
    segments_uri = "s3://aegisvox/runs/run_cap/segments.cleaned.json"
    store.write_json(segments_uri, {"segments": rows})
    backend = FakeEmbeddingBackend()

    payload = SpeakerProfilingService(store, embedding_backend=backend).profile_to_artifact(
        segments_uri,
        "s3://aegisvox/runs/run_cap/speaker_profiles.json",
        output_prefix_uri="s3://aegisvox/runs/run_cap",
    )

    profile = payload["speakers"]["SPEAKER_00"]
    assert profile["duration_sec"] == pytest.approx(1.5)
    assert profile["metadata"]["embedding_duration_sec"] == pytest.approx(1.0)
    assert profile["metadata"]["embedding_max_seconds"] == pytest.approx(0.6)
    assert profile["metadata"]["embedding_audio_uri"].endswith("SPEAKER_00.embedding_sample.wav")
    assert backend.last_wav_path is not None
    assert backend.last_wav_path.name == "SPEAKER_00.embedding_sample.wav"
    with wave.open(str(backend.last_wav_path), "rb") as wav:
        assert wav.getnframes() == 8000


def test_speaker_profile_fails_on_missing_segment_audio(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    segments_uri = "s3://aegisvox/runs/run_001/segments.cleaned.json"
    store.write_json(segments_uri, {"segments": [{"segment_id": "seg_a", "speaker_id": "S0", "segment_audio_uri": "s3://aegisvox/runs/run_001/missing.wav"}]})
    service = SpeakerProfilingService(store, embedding_backend=FakeEmbeddingBackend())
    with pytest.raises(AegisVoxContractError, match="missing segment audio"):
        service.profile_to_artifact(
            segments_uri,
            "s3://aegisvox/runs/run_001/speaker_profiles.json",
            output_prefix_uri="s3://aegisvox/runs/run_001",
        )


def test_speechbrain_backend_supports_load_audio_encode_batch_api(monkeypatch, tmp_path):
    from aegisvox.speaker_profile import SpeechBrainECAPAEmbeddingBackend

    class FakeTensor:
        def unsqueeze(self, dim):
            assert dim == 0
            return self

    class FakeEmbedding:
        def detach(self):
            return self
        def cpu(self):
            return self
        def flatten(self):
            return self
        def tolist(self):
            return [0.1, 0.2, 0.3]

    class FakeClassifier:
        def load_audio(self, path):
            assert path.endswith("x.wav")
            return FakeTensor()
        def encode_batch(self, wav):
            return FakeEmbedding()

    backend = SpeechBrainECAPAEmbeddingBackend()
    monkeypatch.setattr(backend, "_load", lambda: FakeClassifier())
    wav_path = tmp_path / "x.wav"
    wav_path.write_bytes(b"not actually read by fake")
    assert backend.embed_wav(wav_path) == [0.1, 0.2, 0.3]


def test_speaker_profile_infers_document_language_from_sibling_asr_when_segments_unknown(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    seg_uri = "s3://aegisvox/runs/run_fr/segments/seg.wav"
    write_wav(store.local_path(seg_uri), 0.8, amplitude=1000)
    segments_uri = "s3://aegisvox/runs/run_fr/segments.cleaned.json"
    store.write_json(
        segments_uri,
        {"segments": [{"segment_id": "seg", "speaker_id": "SPEAKER_00", "segment_audio_uri": seg_uri}]},
    )
    store.write_json(
        "s3://aegisvox/runs/run_fr/asr.changed_identity.json",
        {
            "text": "Oui docteur, je suis dans la maison avec une urgence et il y a du sang partout.",
            "segments": [],
        },
    )

    payload = SpeakerProfilingService(store, embedding_backend=FakeEmbeddingBackend()).profile_to_artifact(
        segments_uri,
        "s3://aegisvox/runs/run_fr/speaker_profiles.json",
        output_prefix_uri="s3://aegisvox/runs/run_fr",
    )

    profile = payload["speakers"]["SPEAKER_00"]
    assert profile["language"] == "french"
    assert profile["metadata"]["language_source"] == "asr_document_text"
    assert profile["metadata"]["document_language_inference"]["method"] == "text_marker_count"
