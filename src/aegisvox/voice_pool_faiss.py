"""Optional FAISS index support for AegisVox voice pools.

The exact JSON embedding index remains the portable contract. This module turns
that contract into a production-scale FAISS index when `faiss` and `numpy` are
installed. Missing dependencies fail loudly; no approximate/fake replacement is
used.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore
from .voice_pool import (
    CANONICAL_AUDIO_TRAIT_KEYS,
    HARD_GUARDRAIL_TRAITS,
    SOFT_RERANK_TRAITS,
    VoicePoolManifest,
    VoicePoolSetupError,
    cosine_similarity,
    sample_audio_traits,
)


class FaissVoicePoolSetupError(VoicePoolSetupError):
    """Raised when FAISS indexing/search cannot be used."""


def _load_faiss_and_numpy() -> tuple[Any, Any]:
    try:
        np = importlib.import_module("numpy")
        faiss = importlib.import_module("faiss")
    except Exception as exc:
        raise FaissVoicePoolSetupError(
            "FAISS voice-pool indexing requires numpy and faiss/faiss-cpu. Install them in the AegisVox environment first."
        ) from exc
    return faiss, np


def _l2_normalize_matrix(matrix: Any, np: Any) -> Any:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise AegisVoxContractError("cannot build FAISS index with zero-norm embedding")
    return matrix / norms


def build_faiss_index_from_exact(
    *,
    artifact_store: LocalArtifactStore,
    manifest_uri: str,
    exact_index_uri: str,
    faiss_index_uri: str,
    metadata_uri: str,
) -> dict[str, Any]:
    """Build a cosine-similarity FAISS index from the exact JSON index."""

    validate_artifact_uri(manifest_uri)
    validate_artifact_uri(exact_index_uri)
    validate_artifact_uri(faiss_index_uri)
    validate_artifact_uri(metadata_uri)
    faiss, np = _load_faiss_and_numpy()
    manifest_payload = artifact_store.read_json(manifest_uri)
    manifest = VoicePoolManifest.from_dict(manifest_payload)
    exact_payload = artifact_store.read_json(exact_index_uri)
    embeddings = exact_payload.get("embeddings")
    if not isinstance(embeddings, dict) or not embeddings:
        raise VoicePoolSetupError("exact embedding index must contain a non-empty embeddings object")
    sample_ids = [sample.sample_id for sample in manifest.samples]
    missing = [sample_id for sample_id in sample_ids if sample_id not in embeddings]
    if missing:
        raise VoicePoolSetupError(f"exact embedding index missing sample ids, e.g. {missing[:3]}")
    vectors = [[float(x) for x in embeddings[sample_id]] for sample_id in sample_ids]
    dims = {len(vec) for vec in vectors}
    if len(dims) != 1:
        raise VoicePoolSetupError(f"embedding dimensions are inconsistent: {sorted(dims)}")
    dim = dims.pop()
    matrix = np.asarray(vectors, dtype="float32")
    matrix = _l2_normalize_matrix(matrix, np).astype("float32")
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss_path = artifact_store.local_path(faiss_index_uri)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    trait_fill_counts = {key: 0 for key in sorted(CANONICAL_AUDIO_TRAIT_KEYS)}
    num_samples_with_audio_traits = 0
    for sample in manifest.samples:
        traits = sample_audio_traits(sample)
        if traits:
            num_samples_with_audio_traits += 1
        for key, value in traits.items():
            if key in trait_fill_counts and value:
                trait_fill_counts[key] += 1
    trait_fill_rates = {
        key: (count / len(sample_ids) if sample_ids else 0.0)
        for key, count in trait_fill_counts.items()
    }
    metadata = {
        "voice_pool_id": manifest.voice_pool_id,
        "version": manifest.version,
        "index_type": "faiss.IndexFlatIP",
        "metric": "cosine_similarity_normalized_inner_product",
        "embedding_dim": dim,
        "num_vectors": len(sample_ids),
        "hybrid_search_metadata": {
            "candidate_traits_source": "voice_pool_manifest",
            "trait_fields": sorted(CANONICAL_AUDIO_TRAIT_KEYS),
            "hard_guardrail_traits": list(HARD_GUARDRAIL_TRAITS),
            "soft_rerank_traits": list(SOFT_RERANK_TRAITS),
            "num_samples_with_audio_traits": num_samples_with_audio_traits,
            "trait_fill_counts": trait_fill_counts,
            "trait_fill_rates": trait_fill_rates,
            "claim_boundary": (
                "Operational voice-matching metadata only; not identity, demographic, "
                "protected-class, clinical, or precise geography evidence."
            ),
        },
        "sample_ids": sample_ids,
        "faiss_index_uri": faiss_index_uri,
        "source_exact_index_uri": exact_index_uri,
        "manifest_uri": manifest_uri,
    }
    artifact_store.write_json(metadata_uri, metadata)
    return metadata


@dataclass
class FaissVoicePoolIndex:
    """FAISS-backed retrieval using manifest + FAISS metadata + binary index."""

    artifact_store: LocalArtifactStore
    manifest: VoicePoolManifest
    metadata: dict[str, Any]
    index: Any
    np: Any

    @classmethod
    def from_artifacts(cls, artifact_store: LocalArtifactStore, *, manifest_uri: str, metadata_uri: str) -> "FaissVoicePoolIndex":
        faiss, np = _load_faiss_and_numpy()
        manifest = VoicePoolManifest.from_dict(artifact_store.read_json(manifest_uri))
        metadata = artifact_store.read_json(metadata_uri)
        faiss_index_uri = metadata.get("faiss_index_uri")
        if not faiss_index_uri:
            raise VoicePoolSetupError("FAISS metadata missing faiss_index_uri")
        index_path = artifact_store.local_path(faiss_index_uri)
        if not index_path.exists():
            raise VoicePoolSetupError(f"FAISS index artifact missing: {index_path}")
        index = faiss.read_index(str(index_path))
        return cls(artifact_store=artifact_store, manifest=manifest, metadata=metadata, index=index, np=np)

    def search(self, query_embedding: list[float], *, top_k: int = 10, exclude_voice_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise AegisVoxContractError("top_k must be positive")
        sample_ids = self.metadata.get("sample_ids")
        if not isinstance(sample_ids, list) or not sample_ids:
            raise VoicePoolSetupError("FAISS metadata missing non-empty sample_ids")
        dim = int(self.metadata.get("embedding_dim") or 0)
        if len(query_embedding) != dim:
            raise AegisVoxContractError(f"query embedding dimension mismatch: {len(query_embedding)} != {dim}")
        query = self.np.asarray([[float(x) for x in query_embedding]], dtype="float32")
        query = _l2_normalize_matrix(query, self.np).astype("float32")
        k = min(len(sample_ids), max(top_k * 4, top_k))
        scores, indices = self.index.search(query, k)
        samples_by_id = {sample.sample_id: sample for sample in self.manifest.samples}
        excluded = set(exclude_voice_ids or [])
        rows = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(sample_ids):
                continue
            sample = samples_by_id[str(sample_ids[idx])]
            if sample.voice_id in excluded:
                continue
            distance = 1.0 - float(score)
            rows.append(
                {
                    "voice_id": sample.voice_id,
                    "sample_id": sample.sample_id,
                    "audio_uri": sample.audio_uri,
                    "score": distance,
                    "cosine_similarity": float(score),
                    "source_target_distance": distance,
                    "assignment_reason": "faiss cosine candidate",
                    "metadata": sample.to_dict(),
                }
            )
            if len(rows) >= top_k:
                break
        return rows
