"""Voice pool retrieval service wrappers.

Supports the original exact JSON contract index and the optional production FAISS
index metadata written by `build_voice_pool_faiss_index.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aegisvox.schema import AegisVoxContractError
from aegisvox.storage import LocalArtifactStore
from aegisvox.voice_pool import (
    VoicePoolIndex,
    VoicePoolSetupError,
    VoiceSearchRequest,
    VoiceSearchResult,
    build_hybrid_search_explanation,
    hard_guardrail_rejection_reason,
    request_audio_traits,
    sample_audio_traits,
    trait_match_score,
)
from aegisvox.voice_pool_faiss import FaissVoicePoolIndex


@dataclass
class VoicePoolRetrievalService:
    manifest_path: Path
    index_path: Path
    _index: VoicePoolIndex | None = None

    def load(self) -> VoicePoolIndex:
        if self._index is None:
            self._index = VoicePoolIndex.from_paths(self.manifest_path, self.index_path)
        return self._index

    def search(self, request: VoiceSearchRequest) -> list[VoiceSearchResult]:
        return self.load().search(request)


@dataclass
class FaissVoicePoolRetrievalService:
    artifact_store: LocalArtifactStore
    manifest_uri: str
    metadata_uri: str
    _index: FaissVoicePoolIndex | None = None

    def load(self) -> FaissVoicePoolIndex:
        if self._index is None:
            self._index = FaissVoicePoolIndex.from_artifacts(
                self.artifact_store,
                manifest_uri=self.manifest_uri,
                metadata_uri=self.metadata_uri,
            )
        return self._index

    def search(self, request: VoiceSearchRequest) -> list[VoiceSearchResult]:
        if request.query_embedding is None:
            raise VoicePoolSetupError("FAISS voice-pool retrieval requires query_embedding; use exact retrieval for metadata-only search")
        index = self.load()
        if request.voice_pool_id != index.manifest.voice_pool_id:
            raise AegisVoxContractError(
                f"request voice_pool_id={request.voice_pool_id!r} does not match loaded pool {index.manifest.voice_pool_id!r}"
            )
        # Overfetch because metadata filters and min distance are applied after FAISS search.
        top_k = max(request.top_k * 10, request.top_k)
        rows = index.search(request.query_embedding, top_k=top_k, exclude_voice_ids=request.exclude_voice_ids)
        results: list[VoiceSearchResult] = []
        samples_by_id = {sample.sample_id: sample for sample in index.manifest.samples}
        query_traits_for_rerank = request_audio_traits(request)
        has_soft_trait_rerank = bool(set(query_traits_for_rerank) - {"language"})
        for row in rows:
            metadata = row.get("metadata") or {}
            sample = samples_by_id[str(row["sample_id"])]
            sample_traits = sample_audio_traits(sample)
            # Keep FAISS retrieval aligned with the exact-search contract:
            # language, gender presentation, and accent are conservative
            # guardrails when known; geography, tone, energy, speaking rate,
            # and pitch are noisy audio-model descriptors used for reranking
            # evidence only.  Do not hard-filter on the soft descriptors or a
            # useful same-speaker-style candidate can disappear because of a
            # brittle GPT-audio label.
            if any(
                hard_guardrail_rejection_reason(
                    key=key,
                    requested_value=requested_value,
                    sample_traits=sample_traits,
                    require_known_match=request.require_known_hard_trait_matches,
                )
                for key, requested_value in (
                    ("language", request.language),
                    ("gender_presentation", request.gender_presentation),
                    ("accent_group", request.accent_group),
                )
            ):
                continue
            query_traits = query_traits_for_rerank
            trait_score, trait_debug = trait_match_score(sample_traits, query_traits, trait_weights=request.trait_weights)
            distance = row.get("source_target_distance")
            if request.min_source_distance is not None and distance is not None and float(distance) < request.min_source_distance:
                continue
            if request.min_source_distance is None and not has_soft_trait_rerank:
                hybrid_score = float(row.get("cosine_similarity") or 0.0)
            else:
                hybrid_score = float(row["score"]) * request.vector_weight + trait_score * request.trait_weight
            enriched_metadata = dict(metadata)
            enriched_metadata["audio_traits"] = sample_traits
            enriched_metadata["query_trait_match"] = trait_debug
            enriched_metadata["hybrid_search_explanation"] = build_hybrid_search_explanation(
                sample_traits,
                query_traits,
                trait_debug,
            )
            results.append(
                VoiceSearchResult(
                    voice_id=str(row["voice_id"]),
                    sample_id=str(row["sample_id"]),
                    audio_uri=str(row["audio_uri"]),
                    score=hybrid_score,
                    cosine_similarity=float(row["cosine_similarity"]) if row.get("cosine_similarity") is not None else None,
                    source_target_distance=float(distance) if distance is not None else None,
                    assignment_reason=(
                        str(row.get("assignment_reason") or "faiss candidate")
                        + (f", trait_match={trait_score:.3f}" if query_traits else "")
                    ),
                    metadata=enriched_metadata,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: request.top_k]
