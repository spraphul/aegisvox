import wave
from pathlib import Path

import pytest

from aegisvox.speaker_profile_traits import (
    SpeakerProfileTraitEnrichmentPolicy,
    enrich_speaker_profiles_with_audio_traits,
)
from aegisvox.storage import LocalArtifactStore


def write_wav(path: Path, seconds: float = 0.1, frame_rate: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = int(1000).to_bytes(2, "little", signed=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(frame_rate)
        wav.writeframes(sample * int(seconds * frame_rate))


def test_enrich_speaker_profiles_audio_traits(monkeypatch, tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    profile_audio_uri = "s3://aegisvox/runs/run_001/speaker_audio/SPEAKER_00.wav"
    emb_uri = "s3://aegisvox/runs/run_001/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_001/speaker_profiles.json"
    out_uri = "s3://aegisvox/runs/run_001/speaker_profiles.audio_traits.json"
    report_uri = "s3://aegisvox/runs/run_001/speaker_profiles.audio_traits_report.json"
    write_wav(store.local_path(profile_audio_uri))
    store.write_json(emb_uri, {"embedding": [0.1, 0.2]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "speaker_id": "SPEAKER_00",
                    "duration_sec": 1.0,
                    "source_audio_uri": profile_audio_uri,
                    "source_embedding_uri": emb_uri,
                    "language": "unknown",
                    "gender_presentation": "female",
                    "metadata": {},
                }
            }
        },
    )

    def fake_extract(path, *, model):
        assert path == store.local_path(profile_audio_uri)
        assert model == "gpt-audio-1.5"
        return {
            "model": model,
            "audio_traits": {
                "language": "english",
                "accent_group": "indian_english",
                "geography_region": "south_asia",
                "tone": "warm",
                "gender_presentation": "male",
            },
            "confidence": 0.7,
            "evidence": ["mock"],
        }

    monkeypatch.setattr("aegisvox.speaker_profile_traits.extract_audio_traits_with_openai", fake_extract)
    report = enrich_speaker_profiles_with_audio_traits(
        artifact_store=store,
        speaker_profiles_uri=profiles_uri,
        output_profiles_uri=out_uri,
        report_uri=report_uri,
        policy=SpeakerProfileTraitEnrichmentPolicy(),
    )
    assert report["num_enriched"] == 1
    assert report["num_guardrail_warnings"] == 1
    enriched = store.read_json(out_uri)["speakers"]["SPEAKER_00"]
    assert enriched["language"] == "english"
    assert enriched["accent_group"] == "indian_english"
    assert enriched["gender_presentation"] == "female"
    assert enriched["metadata"]["audio_traits"]["gender_presentation"] == "female"
    assert enriched["metadata"]["audio_trait_extraction"]["guardrail_warning"]["type"] == "source_profile_gender_guardrail"
    assert store.read_json(report_uri)["rows"][0]["status"] == "enriched"


def test_enrich_speaker_profiles_skips_existing(monkeypatch, tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    profile_audio_uri = "s3://aegisvox/runs/run_002/speaker_audio/SPEAKER_00.wav"
    emb_uri = "s3://aegisvox/runs/run_002/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_002/speaker_profiles.json"
    out_uri = "s3://aegisvox/runs/run_002/speaker_profiles.audio_traits.json"
    write_wav(store.local_path(profile_audio_uri))
    store.write_json(emb_uri, {"embedding": [0.1, 0.2]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "source_audio_uri": profile_audio_uri,
                    "source_embedding_uri": emb_uri,
                    "metadata": {"audio_traits": {"tone": "calm"}},
                }
            }
        },
    )

    def fail_extract(path, *, model):  # pragma: no cover - should not be called
        raise AssertionError("should skip existing traits")

    monkeypatch.setattr("aegisvox.speaker_profile_traits.extract_audio_traits_with_openai", fail_extract)
    report = enrich_speaker_profiles_with_audio_traits(
        artifact_store=store,
        speaker_profiles_uri=profiles_uri,
        output_profiles_uri=out_uri,
    )
    assert report["num_skipped_existing"] == 1
    assert store.read_json(out_uri)["speakers"]["SPEAKER_00"]["metadata"]["audio_traits"]["tone"] == "calm"


def test_f0_gender_conflict_prefers_audio_model_for_assignment_traits(monkeypatch, tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    profile_audio_uri = "s3://aegisvox/runs/run_003/speaker_audio/SPEAKER_00.wav"
    emb_uri = "s3://aegisvox/runs/run_003/speaker_embeddings/SPEAKER_00.json"
    profiles_uri = "s3://aegisvox/runs/run_003/speaker_profiles.json"
    out_uri = "s3://aegisvox/runs/run_003/speaker_profiles.audio_traits.json"
    report_uri = "s3://aegisvox/runs/run_003/speaker_profiles.audio_traits_report.json"
    write_wav(store.local_path(profile_audio_uri))
    store.write_json(emb_uri, {"embedding": [0.1, 0.2]})
    store.write_json(
        profiles_uri,
        {
            "speakers": {
                "SPEAKER_00": {
                    "speaker_id": "SPEAKER_00",
                    "source_audio_uri": profile_audio_uri,
                    "source_embedding_uri": emb_uri,
                    "gender_presentation": "female",
                    "metadata": {"gender_presentation_source": "f0_autocorrelation_heuristic"},
                }
            }
        },
    )

    def fake_extract(path, *, model):
        return {
            "model": model,
            "audio_traits": {
                "language": "english",
                "accent_group": "american_english",
                "geography_region": "north_america",
                "tone": "calm",
                "gender_presentation": "male",
                "vocal_weight": "thick",
                "resonance": "deep",
                "voice_maturity": "mature",
            },
            "confidence": 0.8,
            "evidence": ["mock"],
        }

    monkeypatch.setattr("aegisvox.speaker_profile_traits.extract_audio_traits_with_openai", fake_extract)
    report = enrich_speaker_profiles_with_audio_traits(
        artifact_store=store,
        speaker_profiles_uri=profiles_uri,
        output_profiles_uri=out_uri,
        report_uri=report_uri,
        policy=SpeakerProfileTraitEnrichmentPolicy(),
    )
    assert report["num_guardrail_warnings"] == 1
    enriched = store.read_json(out_uri)["speakers"]["SPEAKER_00"]
    assert enriched["gender_presentation"] == "male"
    assert enriched["metadata"]["audio_traits"]["gender_presentation"] == "male"
    assert enriched["metadata"]["audio_traits"]["voice_maturity"] == "mature"
    warning = enriched["metadata"]["audio_trait_extraction"]["guardrail_warning"]
    assert warning["type"] == "source_profile_gender_conflict_model_preferred"
