import pickle
import sys
import types

import pytest
from fastapi.testclient import TestClient

from aegisvox.api import create_app
from aegisvox.schema import AegisVoxContractError
from aegisvox.voice_pool import (
    VoicePoolIndex,
    VoicePoolManifest,
    VoicePoolSample,
    VoicePoolSetupError,
    VoiceSearchRequest,
    write_embedding_index,
    write_voice_pool_manifest,
)




class _FakeFaissIndexFlatIP:
    def __init__(self, dim: int):
        self.dim = dim
        self.matrix = None

    def add(self, matrix):
        self.matrix = matrix

    def search(self, query, k: int):
        import numpy as np
        scores = (query @ self.matrix.T)[0]
        order = np.argsort(-scores)[:k]
        return scores[order].reshape(1, -1), order.astype("int64").reshape(1, -1)


def _fake_faiss_module():
    module = types.SimpleNamespace()
    module.IndexFlatIP = _FakeFaissIndexFlatIP
    def write_index(index, path):
        with open(path, "wb") as f:
            pickle.dump(index, f)
    def read_index(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    module.write_index = write_index
    module.read_index = read_index
    return module


def make_voice_pool_files(tmp_path):
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="fixture-v1",
        source="unit-test-fixture",
        samples=[
            VoicePoolSample(
                voice_id="voice_a",
                sample_id="sample_a",
                audio_uri="s3://aegisvox/voice-pools/openswara/audio/a.wav",
                embedding_uri="s3://aegisvox/voice-pools/openswara/embeddings/a.json",
                language="en",
                pitch_band="mid",
                speaking_rate="moderate",
                energy="calm",
                tone="warm",
                geography_region="south_asia",
                license="fixture-only",
                metadata={"audio_traits": {"accent_group": "indian_english", "tone": "warm"}},
            ),
            VoicePoolSample(
                voice_id="voice_b",
                sample_id="sample_b",
                audio_uri="s3://aegisvox/voice-pools/openswara/audio/b.wav",
                embedding_uri="s3://aegisvox/voice-pools/openswara/embeddings/b.json",
                language="en",
                pitch_band="low",
                speaking_rate="slow",
                energy="calm",
                tone="neutral",
                geography_region="north_america",
                license="fixture-only",
                metadata={"audio_traits": {"accent_group": "american_english", "tone": "neutral"}},
            ),
        ],
    )
    manifest_path = tmp_path / "voice_pool.json"
    index_path = tmp_path / "voice_pool_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"sample_a": [1.0, 0.0, 0.0], "sample_b": [-1.0, 0.0, 0.0]})
    return manifest_path, index_path


def test_voice_pool_manifest_requires_artifact_uris():
    with pytest.raises(AegisVoxContractError):
        VoicePoolSample(voice_id="v", sample_id="s", audio_uri="/tmp/not-object-store.wav")


def test_voice_pool_index_search_uses_distance_threshold(tmp_path):
    manifest_path, index_path = make_voice_pool_files(tmp_path)
    index = VoicePoolIndex.from_paths(manifest_path, index_path)
    results = index.search(
        VoiceSearchRequest(
            voice_pool_id="openswara_clinical_v1",
            query_embedding=[1.0, 0.0, 0.0],
            language="en",
            min_source_distance=1.5,
            top_k=5,
        )
    )
    assert [r.sample_id for r in results] == ["sample_b"]
    assert results[0].source_target_distance == pytest.approx(2.0)


def test_voice_pool_index_fails_loudly_when_missing(tmp_path):
    manifest_path, _ = make_voice_pool_files(tmp_path)
    with pytest.raises(VoicePoolSetupError, match="embedding index not found"):
        VoicePoolIndex.from_paths(manifest_path, tmp_path / "missing.json")


def test_voice_pool_api_not_configured_fails_loudly(tmp_path):
    app = create_app(artifact_root=tmp_path / "artifacts", metadata_db_path=tmp_path / "meta.sqlite3")
    client = TestClient(app)
    response = client.post("/v1/voice-pool/search", json={"voice_pool_id": "openswara_clinical_v1"})
    assert response.status_code == 503
    assert "voice pool retrieval is not configured" in response.text


def test_voice_pool_api_search_with_configured_index(tmp_path):
    manifest_path, index_path = make_voice_pool_files(tmp_path)
    app = create_app(
        artifact_root=tmp_path / "artifacts",
        metadata_db_path=tmp_path / "meta.sqlite3",
        voice_pool_manifest_path=manifest_path,
        voice_pool_index_path=index_path,
    )
    client = TestClient(app)
    response = client.post(
        "/v1/voice-pool/search",
        json={
            "voice_pool_id": "openswara_clinical_v1",
            "query_embedding": [1.0, 0.0, 0.0],
            "language": "en",
            "min_source_distance": 1.5,
            "top_k": 3,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["candidates"][0]["sample_id"] == "sample_b"
    assert payload["candidates"][0]["audio_uri"].startswith("s3://")


def test_voice_pool_hybrid_search_uses_audio_traits(tmp_path):
    manifest_path, index_path = make_voice_pool_files(tmp_path)
    index = VoicePoolIndex.from_paths(manifest_path, index_path)
    results = index.search(
        VoiceSearchRequest(
            voice_pool_id="openswara_clinical_v1",
            query_embedding=[1.0, 0.0, 0.0],
            language="en",
            audio_traits={"accent_group": "indian english", "geography_region": "south asia", "tone": "warm"},
            trait_weights={"accent_group": 2.0, "geography_region": 1.5, "tone": 1.0},
            vector_weight=1.0,
            trait_weight=3.0,
            top_k=2,
        )
    )
    assert results[0].sample_id == "sample_a"
    assert results[0].metadata["audio_traits"]["accent_group"] == "indian_english"
    assert results[0].metadata["query_trait_match"]["matched"]["tone"] == "warm"
    explanation = results[0].metadata["hybrid_search_explanation"]
    assert explanation["hard_guardrails"]["language"]["status"] == "matched"
    assert explanation["hard_guardrails"]["accent_group"]["status"] == "matched"
    assert explanation["soft_rerank"]["geography_region"]["status"] == "matched"
    assert explanation["soft_rerank"]["tone"]["status"] == "matched"
    assert explanation["warnings"] == []
    assert "Operational voice-matching explanation only" in explanation["claim_boundary"]
    assert "trait_match=" in results[0].assignment_reason






def test_audio_model_traits_override_manifest_bucket_language(tmp_path):
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="fixture-v1",
        samples=[
            VoicePoolSample(
                voice_id="voice_a",
                sample_id="sample_a",
                audio_uri="s3://aegisvox/audio/a.wav",
                language="arabic",
                metadata={"audio_traits": {"language": "english", "tone": "calm"}},
            )
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    index_path = tmp_path / "index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"sample_a": [1.0, 0.0]})
    index = VoicePoolIndex.from_paths(manifest_path, index_path)
    results = index.search(
        VoiceSearchRequest(
            voice_pool_id="openswara_clinical_v1",
            query_embedding=[1.0, 0.0],
            language="english",
            top_k=1,
        )
    )
    assert [result.sample_id for result in results] == ["sample_a"]
    assert results[0].metadata["audio_traits"]["language"] == "english"


def test_voice_pool_explicit_accent_uses_nested_audio_traits(tmp_path):
    manifest_path, index_path = make_voice_pool_files(tmp_path)
    index = VoicePoolIndex.from_paths(manifest_path, index_path)
    results = index.search(
        VoiceSearchRequest(
            voice_pool_id="openswara_clinical_v1",
            query_embedding=[1.0, 0.0, 0.0],
            accent_group="indian english",
            top_k=5,
        )
    )
    assert [result.sample_id for result in results] == ["sample_a"]
    assert results[0].metadata["audio_traits"]["accent_group"] == "indian_english"


def test_voice_pool_canonicalizes_audio_model_trait_aliases_for_guardrails(tmp_path):
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="fixture-v1",
        samples=[
            VoicePoolSample(
                voice_id="voice_us",
                sample_id="sample_us",
                audio_uri="s3://aegisvox/voice-pools/openswara/audio/us.wav",
                metadata={
                    "audio_traits": {
                        "language": "American English",
                        "gender_presentation": "female sounding",
                        "accent_group": "General American",
                        "geography_region": "United States",
                        "tone": "ordinary",
                        "speaking_rate": "normal",
                        "pitch_band": "medium",
                    }
                },
            ),
            VoicePoolSample(
                voice_id="voice_india",
                sample_id="sample_india",
                audio_uri="s3://aegisvox/voice-pools/openswara/audio/india.wav",
                metadata={
                    "audio_traits": {
                        "language": "English",
                        "gender_presentation": "female",
                        "accent_group": "Indian English",
                    }
                },
            ),
        ],
    )
    manifest_path = tmp_path / "manifest_aliases.json"
    index_path = tmp_path / "index_aliases.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"sample_us": [1.0, 0.0], "sample_india": [0.0, 1.0]})
    index = VoicePoolIndex.from_paths(manifest_path, index_path)

    results = index.search(
        VoiceSearchRequest(
            voice_pool_id="openswara_clinical_v1",
            query_embedding=[1.0, 0.0],
            language="en",
            gender_presentation="woman",
            accent_group="US English",
            audio_traits={
                "geography_region": "north american",
                "tone": "neutral",
                "speaking_rate": "moderate",
                "pitch_band": "mid",
            },
            trait_weight=2.0,
            top_k=5,
        )
    )

    assert [result.sample_id for result in results] == ["sample_us"]
    traits = results[0].metadata["audio_traits"]
    assert traits["language"] == "english"
    assert traits["gender_presentation"] == "female"
    assert traits["accent_group"] == "american_english"
    assert traits["geography_region"] == "north_america"
    assert traits["tone"] == "neutral"
    assert traits["speaking_rate"] == "moderate"
    assert traits["pitch_band"] == "mid"
    query_match = results[0].metadata["query_trait_match"]
    assert query_match["matched"]["accent_group"] == "american_english"
    assert query_match["matched"]["geography_region"] == "north_america"


def test_voice_pool_api_accepts_hybrid_trait_payload(tmp_path):
    manifest_path, index_path = make_voice_pool_files(tmp_path)
    app = create_app(
        artifact_root=tmp_path / "artifacts",
        metadata_db_path=tmp_path / "meta.sqlite3",
        voice_pool_manifest_path=manifest_path,
        voice_pool_index_path=index_path,
    )
    response = TestClient(app).post(
        "/v1/voice-pool/search",
        json={
            "voice_pool_id": "openswara_clinical_v1",
            "query_embedding": [1.0, 0.0, 0.0],
            "audio_traits": {"accent_group": "indian_english", "tone": "warm"},
            "trait_weights": {"tone": 4.0},
            "trait_weight": 3.0,
            "top_k": 1,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["trait_weights"]["accent_group"] == 2.0
    assert payload["trait_weights"]["tone"] == 4.0
    assert payload["trait_weights"]["language"] == 0.75
    candidate = payload["candidates"][0]
    assert candidate["sample_id"] == "sample_a"
    assert candidate["metadata"]["query_trait_match"]["matched"]["accent_group"] == "indian_english"


def test_voice_pool_api_extracts_query_audio_traits(tmp_path, monkeypatch):
    manifest_path, index_path = make_voice_pool_files(tmp_path)
    artifact_root = tmp_path / "artifacts"
    from aegisvox.storage import LocalArtifactStore

    store = LocalArtifactStore(artifact_root)
    query_uri = "s3://aegisvox/queries/query.wav"
    store.local_path(query_uri).parent.mkdir(parents=True, exist_ok=True)
    store.local_path(query_uri).write_bytes(b"RIFFquery")

    def fake_extract(path, *, model):
        assert path == store.local_path(query_uri)
        assert model == "gpt-audio-1.5"
        return {
            "model": model,
            "audio_traits": {"accent_group": "indian_english", "tone": "warm"},
            "confidence": 0.8,
            "evidence": ["mocked audible traits"],
        }

    monkeypatch.setattr("aegisvox.api.extract_audio_traits_with_openai", fake_extract)
    app = create_app(
        artifact_root=artifact_root,
        metadata_db_path=tmp_path / "meta.sqlite3",
        voice_pool_manifest_path=manifest_path,
        voice_pool_index_path=index_path,
    )
    response = TestClient(app).post(
        "/v1/voice-pool/search",
        json={
            "voice_pool_id": "openswara_clinical_v1",
            "query_embedding": [1.0, 0.0, 0.0],
            "query_audio_uri": query_uri,
            "extract_audio_traits": True,
            "trait_weight": 3.0,
            "top_k": 1,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["query_audio_traits"]["audio_traits"]["accent_group"] == "indian_english"
    assert payload["effective_query_audio_traits"]["accent_group"] == "indian_english"
    assert payload["query_trait_precedence"] == ["top_level_fields", "audio_traits", "model_inferred_traits"]
    assert payload["query_trait_sources"]["accent_group"] == "model_inferred_traits"
    assert payload["query_trait_sources"]["tone"] == "model_inferred_traits"
    assert payload["candidates"][0]["sample_id"] == "sample_a"



def test_voice_pool_api_query_audio_traits_drive_guardrails_with_explicit_override(tmp_path, monkeypatch):
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="query-trait-guardrails",
        samples=[
            VoicePoolSample(
                voice_id="voice_indian_female",
                sample_id="sample_indian_female",
                audio_uri="s3://aegisvox/audio/indian_female.wav",
                metadata={
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "female",
                        "accent_group": "indian_english",
                        "tone": "calm",
                    }
                },
            ),
            VoicePoolSample(
                voice_id="voice_american_male",
                sample_id="sample_american_male",
                audio_uri="s3://aegisvox/audio/american_male.wav",
                metadata={
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "male",
                        "accent_group": "american_english",
                        "tone": "calm",
                    }
                },
            ),
        ],
    )
    manifest_path = tmp_path / "query_trait_manifest.json"
    index_path = tmp_path / "query_trait_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"sample_indian_female": [0.8, 0.6], "sample_american_male": [1.0, 0.0]})

    artifact_root = tmp_path / "artifacts"
    from aegisvox.storage import LocalArtifactStore

    store = LocalArtifactStore(artifact_root)
    query_uri = "s3://aegisvox/queries/query.wav"
    store.local_path(query_uri).parent.mkdir(parents=True, exist_ok=True)
    store.local_path(query_uri).write_bytes(b"RIFFquery")

    def fake_extract(path, *, model):
        return {
            "model": model,
            "audio_traits": {
                "language": "english",
                "gender_presentation": "male",
                "accent_group": "american_english",
                "tone": "calm",
            },
            "confidence": 0.8,
            "evidence": ["mocked audible traits"],
        }

    monkeypatch.setattr("aegisvox.api.extract_audio_traits_with_openai", fake_extract)
    app = create_app(
        artifact_root=artifact_root,
        metadata_db_path=tmp_path / "meta.sqlite3",
        voice_pool_manifest_path=manifest_path,
        voice_pool_index_path=index_path,
    )
    client = TestClient(app)

    extracted_response = client.post(
        "/v1/voice-pool/search",
        json={
            "voice_pool_id": "openswara_clinical_v1",
            "query_embedding": [1.0, 0.0],
            "query_audio_uri": query_uri,
            "extract_audio_traits": True,
            "top_k": 2,
        },
    )
    assert extracted_response.status_code == 200, extracted_response.text
    extracted_payload = extracted_response.json()
    assert extracted_payload["effective_query_audio_traits"]["gender_presentation"] == "male"
    assert extracted_payload["effective_query_audio_traits"]["accent_group"] == "american_english"
    assert extracted_payload["query_trait_sources"]["gender_presentation"] == "model_inferred_traits"
    assert extracted_payload["query_trait_sources"]["accent_group"] == "model_inferred_traits"
    extracted_candidates = extracted_payload["candidates"]
    assert [candidate["sample_id"] for candidate in extracted_candidates] == ["sample_american_male"]

    override_response = client.post(
        "/v1/voice-pool/search",
        json={
            "voice_pool_id": "openswara_clinical_v1",
            "query_embedding": [1.0, 0.0],
            "query_audio_uri": query_uri,
            "extract_audio_traits": True,
            "audio_traits": {"gender_presentation": "female", "accent_group": "indian_english"},
            "top_k": 2,
        },
    )
    assert override_response.status_code == 200, override_response.text
    override_payload = override_response.json()
    assert override_payload["query_audio_traits"]["audio_traits"]["accent_group"] == "american_english"
    assert override_payload["effective_query_audio_traits"]["gender_presentation"] == "female"
    assert override_payload["effective_query_audio_traits"]["accent_group"] == "indian_english"
    assert override_payload["query_trait_precedence"] == ["top_level_fields", "audio_traits", "model_inferred_traits"]
    assert override_payload["query_trait_sources"]["gender_presentation"] == "audio_traits"
    assert override_payload["query_trait_sources"]["accent_group"] == "audio_traits"
    assert override_payload["query_trait_sources"]["tone"] == "model_inferred_traits"
    override_candidates = override_payload["candidates"]
    assert [candidate["sample_id"] for candidate in override_candidates] == ["sample_indian_female"]
    trait = override_candidates[0]["metadata"]["query_trait_match"]
    assert trait["matched"]["gender_presentation"] == "female"
    assert trait["matched"]["accent_group"] == "indian_english"


def test_voice_pool_api_top_level_fields_override_audio_traits_and_model_traits(tmp_path, monkeypatch):
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="query-trait-top-level-precedence",
        samples=[
            VoicePoolSample(
                voice_id="voice_female_indian",
                sample_id="sample_female_indian",
                audio_uri="s3://aegisvox/audio/female_indian.wav",
                metadata={
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "female",
                        "accent_group": "indian_english",
                    }
                },
            ),
            VoicePoolSample(
                voice_id="voice_male_american",
                sample_id="sample_male_american",
                audio_uri="s3://aegisvox/audio/male_american.wav",
                metadata={
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "male",
                        "accent_group": "american_english",
                    }
                },
            ),
        ],
    )
    manifest_path = tmp_path / "query_trait_top_level_manifest.json"
    index_path = tmp_path / "query_trait_top_level_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"sample_female_indian": [0.8, 0.6], "sample_male_american": [1.0, 0.0]})

    artifact_root = tmp_path / "artifacts"
    from aegisvox.storage import LocalArtifactStore

    store = LocalArtifactStore(artifact_root)
    query_uri = "s3://aegisvox/queries/query.wav"
    store.local_path(query_uri).parent.mkdir(parents=True, exist_ok=True)
    store.local_path(query_uri).write_bytes(b"RIFFquery")

    def fake_extract(path, *, model):
        return {
            "model": model,
            "audio_traits": {"language": "english", "gender_presentation": "male", "accent_group": "american_english"},
            "confidence": 0.8,
            "evidence": [],
        }

    monkeypatch.setattr("aegisvox.api.extract_audio_traits_with_openai", fake_extract)
    app = create_app(
        artifact_root=artifact_root,
        metadata_db_path=tmp_path / "meta.sqlite3",
        voice_pool_manifest_path=manifest_path,
        voice_pool_index_path=index_path,
    )
    response = TestClient(app).post(
        "/v1/voice-pool/search",
        json={
            "voice_pool_id": "openswara_clinical_v1",
            "query_embedding": [1.0, 0.0],
            "query_audio_uri": query_uri,
            "extract_audio_traits": True,
            "audio_traits": {"gender_presentation": "male", "accent_group": "american_english"},
            "gender_presentation": "female",
            "accent_group": "indian_english",
            "top_k": 2,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["query_audio_traits"]["audio_traits"]["gender_presentation"] == "male"
    assert payload["effective_query_audio_traits"]["gender_presentation"] == "female"
    assert payload["effective_query_audio_traits"]["accent_group"] == "indian_english"
    assert payload["query_trait_precedence"] == ["top_level_fields", "audio_traits", "model_inferred_traits"]
    assert payload["query_trait_sources"]["gender_presentation"] == "top_level_fields"
    assert payload["query_trait_sources"]["accent_group"] == "top_level_fields"
    assert payload["query_trait_sources"]["language"] == "model_inferred_traits"
    assert [candidate["sample_id"] for candidate in payload["candidates"]] == ["sample_female_indian"]
    trait = payload["candidates"][0]["metadata"]["query_trait_match"]
    assert trait["matched"]["gender_presentation"] == "female"
    assert trait["matched"]["accent_group"] == "indian_english"


def test_voice_pool_api_effective_traits_are_canonical_search_tokens(tmp_path):
    manifest = VoicePoolManifest(
        voice_pool_id="openswara_clinical_v1",
        version="canonical-effective-traits",
        samples=[
            VoicePoolSample(
                voice_id="voice_us_female",
                sample_id="sample_us_female",
                audio_uri="s3://aegisvox/audio/us_female.wav",
                metadata={
                    "audio_traits": {
                        "language": "English",
                        "gender_presentation": "female",
                        "accent_group": "american_english",
                        "geography_region": "north_america",
                        "speaking_rate": "moderate",
                        "pitch_band": "mid",
                    }
                },
            )
        ],
    )
    manifest_path = tmp_path / "canonical_effective_manifest.json"
    index_path = tmp_path / "canonical_effective_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"sample_us_female": [1.0, 0.0]})
    app = create_app(
        artifact_root=tmp_path / "artifacts",
        metadata_db_path=tmp_path / "meta.sqlite3",
        voice_pool_manifest_path=manifest_path,
        voice_pool_index_path=index_path,
    )

    response = TestClient(app).post(
        "/v1/voice-pool/search",
        json={
            "voice_pool_id": "openswara_clinical_v1",
            "query_embedding": [1.0, 0.0],
            "language": "US English",
            "gender_presentation": "female sounding",
            "accent_group": "General American",
            "audio_traits": {
                "geography_region": "United States",
                "speaking_rate": "normal",
                "pitch_band": "medium",
            },
            "top_k": 1,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["trait_weights"]["accent_group"] == 2.0
    assert payload["trait_weights"]["geography_region"] == 1.0
    assert payload["trait_weights"]["speaking_rate"] == 0.75
    assert payload["trait_weights"]["pitch_band"] == 0.5
    assert payload["query_trait_sources"] == {
        "geography_region": "audio_traits",
        "speaking_rate": "audio_traits",
        "pitch_band": "audio_traits",
        "language": "top_level_fields",
        "gender_presentation": "top_level_fields",
        "accent_group": "top_level_fields",
    }
    assert payload["effective_query_audio_traits"] == {
        "geography_region": "north_america",
        "speaking_rate": "moderate",
        "pitch_band": "mid",
        "language": "english",
        "gender_presentation": "female",
        "accent_group": "american_english",
    }
    assert [candidate["sample_id"] for candidate in payload["candidates"]] == ["sample_us_female"]
    trait = payload["candidates"][0]["metadata"]["query_trait_match"]
    assert trait["matched"]["language"] == "english"
    assert trait["matched"]["accent_group"] == "american_english"


def test_voice_pool_api_search_with_faiss_index(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    from aegisvox.storage import LocalArtifactStore
    from aegisvox.voice_pool_faiss import build_faiss_index_from_exact

    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss_module())
    store = LocalArtifactStore(tmp_path / "artifacts")
    manifest_uri = "s3://aegisvox/voice-pools/faiss_api/manifest.json"
    exact_uri = "s3://aegisvox/voice-pools/faiss_api/embedding_index.json"
    metadata_uri = "s3://aegisvox/voice-pools/faiss_api/faiss_index.json"
    store.write_json(
        manifest_uri,
        {
            "voice_pool_id": "faiss_api",
            "version": "v1",
            "samples": [
                {"voice_id": "voice_a", "sample_id": "sample_a", "audio_uri": "s3://aegisvox/audio/a.wav", "language": "en"},
                {"voice_id": "voice_b", "sample_id": "sample_b", "audio_uri": "s3://aegisvox/audio/b.wav", "language": "en"},
            ],
        },
    )
    store.write_json(exact_uri, {"embeddings": {"sample_a": [1.0, 0.0], "sample_b": [0.0, 1.0]}})
    build_faiss_index_from_exact(
        artifact_store=store,
        manifest_uri=manifest_uri,
        exact_index_uri=exact_uri,
        faiss_index_uri="s3://aegisvox/voice-pools/faiss_api/faiss.index",
        metadata_uri=metadata_uri,
    )
    app = create_app(
        artifact_root=tmp_path / "artifacts",
        metadata_db_path=tmp_path / "meta.sqlite3",
        voice_pool_manifest_uri=manifest_uri,
        voice_pool_faiss_metadata_uri=metadata_uri,
    )
    client = TestClient(app)
    response = client.post(
        "/v1/voice-pool/search",
        json={"voice_pool_id": "faiss_api", "query_embedding": [0.9, 0.1], "language": "en", "top_k": 1},
    )
    assert response.status_code == 200, response.text
    assert response.json()["candidates"][0]["sample_id"] == "sample_a"



def test_voice_pool_api_faiss_uses_gender_accent_guardrails_and_soft_prosody(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    from aegisvox.storage import LocalArtifactStore
    from aegisvox.voice_pool_faiss import build_faiss_index_from_exact

    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss_module())
    store = LocalArtifactStore(tmp_path / "artifacts")
    manifest_uri = "s3://aegisvox/voice-pools/faiss_traits/manifest.json"
    exact_uri = "s3://aegisvox/voice-pools/faiss_traits/embedding_index.json"
    metadata_uri = "s3://aegisvox/voice-pools/faiss_traits/faiss_index.json"
    store.write_json(
        manifest_uri,
        {
            "voice_pool_id": "faiss_traits",
            "version": "v1",
            "samples": [
                {
                    "voice_id": "near_wrong_gender",
                    "sample_id": "near_wrong_gender",
                    "audio_uri": "s3://aegisvox/audio/near.wav",
                    "metadata": {
                        "audio_traits": {
                            "language": "english",
                            "gender_presentation": "male",
                            "accent_group": "indian_english",
                            "energy": "low",
                            "speaking_rate": "slow",
                            "pitch_band": "high",
                        }
                    },
                },
                {
                    "voice_id": "near_wrong_accent",
                    "sample_id": "near_wrong_accent",
                    "audio_uri": "s3://aegisvox/audio/accent.wav",
                    "metadata": {
                        "audio_traits": {
                            "language": "english",
                            "gender_presentation": "female",
                            "accent_group": "american_english",
                            "energy": "low",
                            "speaking_rate": "slow",
                            "pitch_band": "high",
                        }
                    },
                },
                {
                    "voice_id": "matching_guardrails_soft_mismatch",
                    "sample_id": "matching_guardrails_soft_mismatch",
                    "audio_uri": "s3://aegisvox/audio/match.wav",
                    "metadata": {
                        "audio_traits": {
                            "language": "english",
                            "gender_presentation": "female",
                            "accent_group": "indian_english",
                            "energy": "high",
                            "speaking_rate": "fast",
                            "pitch_band": "low",
                            "tone": "calm",
                        }
                    },
                },
            ],
        },
    )
    store.write_json(
        exact_uri,
        {
            "embeddings": {
                "near_wrong_gender": [1.0, 0.0],
                "near_wrong_accent": [0.98, 0.02],
                "matching_guardrails_soft_mismatch": [0.8, 0.6],
            }
        },
    )
    build_faiss_index_from_exact(
        artifact_store=store,
        manifest_uri=manifest_uri,
        exact_index_uri=exact_uri,
        faiss_index_uri="s3://aegisvox/voice-pools/faiss_traits/faiss.index",
        metadata_uri=metadata_uri,
    )
    app = create_app(
        artifact_root=tmp_path / "artifacts",
        metadata_db_path=tmp_path / "meta.sqlite3",
        voice_pool_manifest_uri=manifest_uri,
        voice_pool_faiss_metadata_uri=metadata_uri,
    )
    response = TestClient(app).post(
        "/v1/voice-pool/search",
        json={
            "voice_pool_id": "faiss_traits",
            "query_embedding": [1.0, 0.0],
            "language": "english",
            "gender_presentation": "female",
            "accent_group": "indian_english",
            "audio_traits": {
                "energy": "low",
                "speaking_rate": "slow",
                "pitch_band": "high",
                "tone": "conversational",
            },
            "trait_weights": {"gender_presentation": 2.0, "accent_group": 2.0, "energy": 0.5, "speaking_rate": 0.5, "pitch_band": 0.5},
            "trait_weight": 2.0,
            "top_k": 3,
        },
    )
    assert response.status_code == 200, response.text
    candidates = response.json()["candidates"]
    assert [candidate["sample_id"] for candidate in candidates] == ["matching_guardrails_soft_mismatch"]
    trait = candidates[0]["metadata"]["query_trait_match"]
    assert trait["matched"]["gender_presentation"] == "female"
    assert trait["matched"]["accent_group"] == "indian_english"
    assert trait["mismatched"]["energy"] == {"query": "low", "candidate": "high"}
    assert trait["mismatched"]["speaking_rate"] == {"query": "slow", "candidate": "fast"}
    assert trait["mismatched"]["pitch_band"] == {"query": "high", "candidate": "low"}
    assert trait["equivalent"]["tone"]["group"] == "neutral_conversational"
    explanation = candidates[0]["metadata"]["hybrid_search_explanation"]
    assert explanation["hard_guardrails"]["gender_presentation"]["status"] == "matched"
    assert explanation["hard_guardrails"]["accent_group"]["status"] == "matched"
    assert explanation["soft_rerank"]["tone"]["status"] == "equivalent"
    assert explanation["soft_rerank"]["tone"]["equivalence"]["group"] == "neutral_conversational"
    assert explanation["soft_rerank"]["energy"]["status"] == "mismatched"
    assert "mismatched_soft_trait:energy" in explanation["warnings"]

def test_voice_pool_api_faiss_requires_query_embedding(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    from aegisvox.storage import LocalArtifactStore
    from aegisvox.voice_pool_faiss import build_faiss_index_from_exact

    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss_module())
    store = LocalArtifactStore(tmp_path / "artifacts")
    manifest_uri = "s3://aegisvox/voice-pools/faiss_api/manifest.json"
    exact_uri = "s3://aegisvox/voice-pools/faiss_api/embedding_index.json"
    metadata_uri = "s3://aegisvox/voice-pools/faiss_api/faiss_index.json"
    store.write_json(
        manifest_uri,
        {
            "voice_pool_id": "faiss_api",
            "version": "v1",
            "samples": [{"voice_id": "voice_a", "sample_id": "sample_a", "audio_uri": "s3://aegisvox/audio/a.wav"}],
        },
    )
    store.write_json(exact_uri, {"embeddings": {"sample_a": [1.0, 0.0]}})
    build_faiss_index_from_exact(
        artifact_store=store,
        manifest_uri=manifest_uri,
        exact_index_uri=exact_uri,
        faiss_index_uri="s3://aegisvox/voice-pools/faiss_api/faiss.index",
        metadata_uri=metadata_uri,
    )
    app = create_app(
        artifact_root=tmp_path / "artifacts",
        metadata_db_path=tmp_path / "meta.sqlite3",
        voice_pool_manifest_uri=manifest_uri,
        voice_pool_faiss_metadata_uri=metadata_uri,
    )
    response = TestClient(app).post("/v1/voice-pool/search", json={"voice_pool_id": "faiss_api"})
    assert response.status_code == 503
    assert "requires query_embedding" in response.text


def test_voice_pool_geography_and_tone_are_soft_rerank_traits(tmp_path):
    manifest = VoicePoolManifest(
        voice_pool_id="soft_traits",
        version="v1",
        samples=[
            VoicePoolSample(
                voice_id="unknown_traits_far",
                sample_id="unknown_traits_far",
                audio_uri="s3://aegisvox/audio/unknown.wav",
                language="en",
                metadata={"audio_traits": {"language": "english", "gender_presentation": "female"}},
            ),
            VoicePoolSample(
                voice_id="geo_match_tone_neighbor",
                sample_id="geo_match_tone_neighbor",
                audio_uri="s3://aegisvox/audio/geo.wav",
                language="en",
                metadata={
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "female",
                        "geography_region": "north_america",
                        "tone": "calm",
                    }
                },
            ),
        ],
    )
    manifest_path = tmp_path / "soft_manifest.json"
    index_path = tmp_path / "soft_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"unknown_traits_far": [-1.0, 0.0], "geo_match_tone_neighbor": [-0.8, 0.6]})
    index = VoicePoolIndex.from_paths(manifest_path, index_path)
    results = index.search(
        VoiceSearchRequest(
            voice_pool_id="soft_traits",
            query_embedding=[1.0, 0.0],
            language="english",
            gender_presentation="female",
            audio_traits={"geography_region": "north_america", "tone": "conversational"},
            trait_weights={"geography_region": 2.0, "tone": 1.0, "gender_presentation": 1.0, "language": 1.0},
            vector_weight=1.0,
            trait_weight=3.0,
            top_k=2,
        )
    )
    assert results[0].sample_id == "geo_match_tone_neighbor"
    trait = results[0].metadata["query_trait_match"]
    assert trait["matched"]["geography_region"] == "north_america"
    assert trait["equivalent"]["tone"]["candidate"] == "calm"
    assert trait["equivalent"]["tone"]["query"] == "conversational"
    assert trait["equivalent"]["tone"]["group"] == "neutral_conversational"
    explanation = results[0].metadata["hybrid_search_explanation"]
    assert explanation["soft_rerank"]["tone"]["status"] == "equivalent"
    assert explanation["soft_rerank"]["tone"]["equivalence"]["group"] == "neutral_conversational"
    assert explanation["warnings"] == []
    assert trait["score"] > 0.9
    assert "tone" not in trait["mismatched"]


def test_voice_pool_can_fail_closed_on_missing_hard_guardrail_traits(tmp_path):
    manifest = VoicePoolManifest(
        voice_pool_id="hard_guardrail_missing",
        version="v1",
        samples=[
            VoicePoolSample(
                voice_id="missing_gender_accent_farther",
                sample_id="missing_gender_accent_farther",
                audio_uri="s3://aegisvox/audio/missing.wav",
                language="en",
                metadata={"audio_traits": {"language": "english", "accent_group": "american_english"}},
            ),
            VoicePoolSample(
                voice_id="known_match_nearer",
                sample_id="known_match_nearer",
                audio_uri="s3://aegisvox/audio/known.wav",
                language="en",
                metadata={
                    "audio_traits": {
                        "language": "english",
                        "gender_presentation": "female",
                        "accent_group": "american_english",
                    }
                },
            ),
        ],
    )
    manifest_path = tmp_path / "hard_guardrail_manifest.json"
    index_path = tmp_path / "hard_guardrail_index.json"
    write_voice_pool_manifest(manifest_path, manifest)
    write_embedding_index(index_path, {"missing_gender_accent_farther": [-1.0, 0.0], "known_match_nearer": [-0.8, 0.6]})
    index = VoicePoolIndex.from_paths(manifest_path, index_path)

    exploratory = index.search(
        VoiceSearchRequest(
            voice_pool_id="hard_guardrail_missing",
            query_embedding=[1.0, 0.0],
            language="english",
            gender_presentation="female",
            accent_group="american_english",
            vector_weight=1.0,
            trait_weight=0.0,
            top_k=2,
        )
    )
    assert exploratory[0].sample_id == "missing_gender_accent_farther"
    assert "gender_presentation" in exploratory[0].metadata["query_trait_match"]["missing"]
    assert exploratory[0].metadata["query_trait_match"]["matched"]["accent_group"] == "american_english"

    assignment_grade = index.search(
        VoiceSearchRequest(
            voice_pool_id="hard_guardrail_missing",
            query_embedding=[1.0, 0.0],
            language="english",
            gender_presentation="female",
            accent_group="american_english",
            vector_weight=1.0,
            trait_weight=0.0,
            require_known_hard_trait_matches=True,
            top_k=2,
        )
    )
    assert [result.sample_id for result in assignment_grade] == ["known_match_nearer"]
    explanation = assignment_grade[0].metadata["hybrid_search_explanation"]
    assert explanation["hard_guardrails"]["gender_presentation"]["status"] == "matched"
    assert explanation["hard_guardrails"]["accent_group"]["status"] == "matched"


def test_open_swara_catalog_traits_parse_repo_file_and_quality_fields():
    from aegisvox.voice_pool import open_swara_catalog_traits, sample_audio_traits

    native = VoicePoolSample(
        voice_id="english_male_open_swara_001",
        sample_id="english_male_open_swara_001",
        audio_uri="s3://aegisvox/pool/native.wav",
        metadata={
            "repo_file": "voices/english/male/english_male_open_swara_001.wav",
            "duration_seconds": "5.25",
            "sample_rate": "48000",
            "audio_traits": {"language": "english", "pitch_band": "high"},
        },
    )
    traits = open_swara_catalog_traits(native)
    assert traits["catalog_language"] == "english"
    assert traits["catalog_gender"] == "male"
    assert traits["catalog_type"] == "native"
    assert traits["is_english_accent"] is False
    assert traits["catalog_duration_seconds"] == pytest.approx(5.25)
    assert traits["catalog_sample_rate"] == 48000
    # Pitch-derived vocal body gives assignment a style guardrail even before a
    # full explicit vocal_weight enrichment pass exists.
    assert sample_audio_traits(native)["vocal_weight"] == "thin"
    assert sample_audio_traits(native)["resonance"] == "light"

    accented = VoicePoolSample(
        voice_id="german_female_english_open_swara_001",
        sample_id="german_female_english_open_swara_001",
        audio_uri="s3://aegisvox/pool/german_english.wav",
        provenance={"repo_file": "voices/german/female/german_female_english_open_swara_001.wav"},
    )
    accented_traits = open_swara_catalog_traits(accented)
    assert accented_traits["catalog_language"] == "german"
    assert accented_traits["catalog_gender"] == "female"
    assert accented_traits["catalog_type"] == "english_accent"
    assert accented_traits["is_english_accent"] is True


def test_voice_quality_profile_refines_body_and_maturity_traits():
    from aegisvox.voice_pool import sample_audio_traits

    sample = VoicePoolSample(
        voice_id="english_female_open_swara_123",
        sample_id="english_female_open_swara_123",
        audio_uri="s3://aegisvox/pool/female.wav",
        language="en",
        gender_presentation="female",
        metadata={
            "repo_file": "voices/english/female/english_female_open_swara_123.wav",
            "audio_traits": {"language": "english", "gender_presentation": "female", "accent_group": "american_english"},
            "voice_quality_profile": {
                "profile": {
                    "voice_maturity": "mature",
                    "vocal_weight": "thick",
                    "resonance": "deep",
                    "pitch_band": "low",
                    "child_like_risk": 0.01,
                },
                "audio_traits": {"voice_maturity": "mature", "vocal_weight": "thick", "resonance": "deep"},
            },
        },
    )
    traits = sample_audio_traits(sample)
    assert traits["language"] == "english"
    assert traits["gender_presentation"] == "female"
    assert traits["voice_maturity"] == "mature"
    assert traits["vocal_weight"] == "thick"
    assert traits["resonance"] == "deep"
