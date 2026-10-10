from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import pytest

from aegisvox.storage import LocalArtifactStore
from aegisvox.voice_pool_faiss import FaissVoicePoolIndex, FaissVoicePoolSetupError, build_faiss_index_from_exact


class FakeFaissIndexFlatIP:
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


def fake_faiss_module():
    module = types.SimpleNamespace()
    module.IndexFlatIP = FakeFaissIndexFlatIP

    def write_index(index, path):
        with open(path, "wb") as f:
            pickle.dump(index, f)

    def read_index(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    module.write_index = write_index
    module.read_index = read_index
    return module


def write_fixture(store: LocalArtifactStore):
    manifest_uri = "s3://aegisvox/voice-pools/test/manifest.json"
    exact_uri = "s3://aegisvox/voice-pools/test/embedding_index.json"
    store.write_json(
        manifest_uri,
        {
            "voice_pool_id": "test_pool",
            "version": "v1",
            "samples": [
                {
                    "voice_id": "voice_a",
                    "sample_id": "sample_a",
                    "audio_uri": "s3://aegisvox/audio/a.wav",
                    "language": "english",
                    "gender_presentation": "female",
                    "metadata": {
                        "audio_traits": {
                            "accent_group": "american_english",
                            "tone": "calm",
                        }
                    },
                },
                {
                    "voice_id": "voice_b",
                    "sample_id": "sample_b",
                    "audio_uri": "s3://aegisvox/audio/b.wav",
                    "language": "english",
                    "gender_presentation": "male",
                },
            ],
        },
    )
    store.write_json(
        exact_uri,
        {
            "voice_pool_id": "test_pool",
            "index_type": "exact_json_contract",
            "embeddings": {"sample_a": [1.0, 0.0], "sample_b": [0.0, 1.0]},
        },
    )
    return manifest_uri, exact_uri


def test_faiss_builder_fails_loudly_when_dependency_missing(tmp_path: Path, monkeypatch):
    store = LocalArtifactStore(tmp_path / "artifacts")
    manifest_uri, exact_uri = write_fixture(store)
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(FaissVoicePoolSetupError, match="requires numpy and faiss"):
        build_faiss_index_from_exact(
            artifact_store=store,
            manifest_uri=manifest_uri,
            exact_index_uri=exact_uri,
            faiss_index_uri="s3://aegisvox/voice-pools/test/faiss.index",
            metadata_uri="s3://aegisvox/voice-pools/test/faiss_index.json",
        )


def test_faiss_builder_and_search_with_injected_faiss(tmp_path: Path, monkeypatch):
    pytest.importorskip("numpy")
    store = LocalArtifactStore(tmp_path / "artifacts")
    manifest_uri, exact_uri = write_fixture(store)
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss_module())
    metadata = build_faiss_index_from_exact(
        artifact_store=store,
        manifest_uri=manifest_uri,
        exact_index_uri=exact_uri,
        faiss_index_uri="s3://aegisvox/voice-pools/test/faiss.index",
        metadata_uri="s3://aegisvox/voice-pools/test/faiss_index.json",
    )
    assert metadata["index_type"] == "faiss.IndexFlatIP"
    assert metadata["sample_ids"] == ["sample_a", "sample_b"]
    hybrid_metadata = metadata["hybrid_search_metadata"]
    assert hybrid_metadata["candidate_traits_source"] == "voice_pool_manifest"
    assert hybrid_metadata["hard_guardrail_traits"] == ["language", "gender_presentation", "accent_group"]
    assert "geography_region" in hybrid_metadata["soft_rerank_traits"]
    assert hybrid_metadata["num_samples_with_audio_traits"] == 2
    assert hybrid_metadata["trait_fill_counts"]["language"] == 2
    assert hybrid_metadata["trait_fill_counts"]["accent_group"] == 2
    assert hybrid_metadata["trait_fill_rates"]["tone"] == 0.5
    assert "not identity" in hybrid_metadata["claim_boundary"]
    assert store.exists("s3://aegisvox/voice-pools/test/faiss.index")
    index = FaissVoicePoolIndex.from_artifacts(
        store,
        manifest_uri=manifest_uri,
        metadata_uri="s3://aegisvox/voice-pools/test/faiss_index.json",
    )
    rows = index.search([0.9, 0.1], top_k=1)
    assert rows[0]["sample_id"] == "sample_a"
    assert rows[0]["assignment_reason"] == "faiss cosine candidate"
