"""Speaker privacy/linkability metrics for AegisVox evaluation.

This module computes real embedding-based metrics from existing artifacts. It
never fabricates embeddings: production use requires a configured speaker
embedding backend such as SpeechBrain ECAPA. Unit tests may inject a deterministic
backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import AegisVoxContractError, validate_artifact_uri
from .speaker_profile import SpeakerEmbeddingBackend, SpeechBrainECAPAEmbeddingBackend, concatenate_wav_segments
from .storage import LocalArtifactStore
from .voice_pool import cosine_similarity


class SpeakerPrivacyMetricsSetupError(RuntimeError):
    """Raised when real embedding dependencies or artifacts are unavailable."""


@dataclass(frozen=True)
class SpeakerPrivacyPolicy:
    """Thresholds for embedding-based privacy checks.

    `max_source_converted_similarity` is a privacy threshold: converted speech
    should not remain too similar to the source speaker. `min_target_converted`
    is a utility/assignment-consistency threshold: converted speech should be at
    least somewhat close to the assigned pseudo-speaker reference.
    """

    max_source_converted_similarity: float = 0.65
    min_target_converted_similarity: float = 0.25


def _load_json_object(store: LocalArtifactStore, uri: str) -> dict[str, Any]:
    validate_artifact_uri(uri)
    payload = store.read_json(uri)
    if not isinstance(payload, dict):
        raise AegisVoxContractError(f"expected JSON object at {uri}")
    return payload


def _load_embedding(store: LocalArtifactStore, uri: str) -> list[float]:
    payload = _load_json_object(store, uri)
    embedding = payload.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise AegisVoxContractError(f"embedding artifact {uri} must contain non-empty embedding list")
    return [float(x) for x in embedding]


def _converted_rows_by_speaker(converted_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = converted_payload.get("segments")
    if not isinstance(rows, list):
        raise AegisVoxContractError("segments.converted.json must contain segments list")
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("conversion_status") != "converted":
            continue
        speaker_id = row.get("speaker_id")
        converted_uri = row.get("converted_audio_uri")
        if not speaker_id or not converted_uri:
            raise AegisVoxContractError("converted segment row missing speaker_id/converted_audio_uri")
        validate_artifact_uri(str(converted_uri))
        by_speaker.setdefault(str(speaker_id), []).append(dict(row))
    return by_speaker


@dataclass
class SpeakerPrivacyMetricsService:
    artifact_store: LocalArtifactStore
    embedding_backend: SpeakerEmbeddingBackend | None = None
    policy: SpeakerPrivacyPolicy = field(default_factory=SpeakerPrivacyPolicy)

    def preflight(self) -> dict[str, Any]:
        if self.embedding_backend is None:
            SpeechBrainECAPAEmbeddingBackend()._load()
            backend = "speechbrain-ecapa"
        else:
            backend = self.embedding_backend.backend_name
        return {"ok": True, "backend": backend}

    def compute_to_artifact(
        self,
        *,
        speaker_profiles_uri: str,
        assignments_uri: str,
        converted_segments_uri: str,
        output_metrics_uri: str,
        output_prefix_uri: str,
    ) -> dict[str, Any]:
        validate_artifact_uri(output_metrics_uri)
        validate_artifact_uri(output_prefix_uri.rstrip("/") + "/")
        if self.embedding_backend is None:
            raise SpeakerPrivacyMetricsSetupError(
                "speaker privacy metrics require a real speaker embedding backend; configure SpeechBrain ECAPA first"
            )
        profiles_payload = _load_json_object(self.artifact_store, speaker_profiles_uri)
        assignments_payload = _load_json_object(self.artifact_store, assignments_uri)
        converted_payload = _load_json_object(self.artifact_store, converted_segments_uri)
        profiles = profiles_payload.get("speakers")
        assignments = assignments_payload.get("assignments")
        if not isinstance(profiles, dict) or not profiles:
            raise AegisVoxContractError("speaker_profiles.json must contain non-empty speakers object")
        if not isinstance(assignments, dict) or not assignments:
            raise AegisVoxContractError("speaker_target_map.json must contain non-empty assignments object")
        converted_by_speaker = _converted_rows_by_speaker(converted_payload)
        speaker_rows = []
        converted_embedding_uris: dict[str, str] = {}
        converted_audio_uris: dict[str, str] = {}
        for speaker_id, profile in sorted(profiles.items()):
            assignment = assignments.get(speaker_id)
            converted_rows = converted_by_speaker.get(str(speaker_id), [])
            if not assignment:
                raise AegisVoxContractError(f"missing pseudo-speaker assignment for {speaker_id}")
            if not converted_rows:
                speaker_rows.append(
                    {
                        "speaker_id": speaker_id,
                        "status": "missing_converted_audio",
                        "reason": "no converted segments for speaker",
                    }
                )
                continue
            source_embedding_uri = profile.get("source_embedding_uri")
            target_reference_audio_uri = assignment.get("target_reference_audio_uri")
            if not source_embedding_uri:
                raise AegisVoxContractError(f"speaker profile {speaker_id} missing source_embedding_uri")
            if not target_reference_audio_uri:
                raise AegisVoxContractError(f"assignment for {speaker_id} missing target_reference_audio_uri")
            source_embedding = _load_embedding(self.artifact_store, str(source_embedding_uri))
            converted_paths = [self.artifact_store.local_path(str(row["converted_audio_uri"])) for row in converted_rows]
            missing = [str(path) for path in converted_paths if not path.exists()]
            if missing:
                raise AegisVoxContractError(f"missing converted audio for speaker {speaker_id}: {missing[:3]}")
            converted_audio_uri = f"{output_prefix_uri.rstrip('/')}/speaker_converted_audio/{speaker_id}.wav"
            converted_audio_path = self.artifact_store.local_path(converted_audio_uri)
            duration = concatenate_wav_segments(converted_paths, converted_audio_path)
            converted_embedding = self.embedding_backend.embed_wav(converted_audio_path)
            converted_embedding_uri = f"{output_prefix_uri.rstrip('/')}/speaker_converted_embeddings/{speaker_id}.json"
            self.artifact_store.write_json(
                converted_embedding_uri,
                {"embedding": converted_embedding, "backend": self.embedding_backend.backend_name},
            )
            target_reference_path = self.artifact_store.local_path(str(target_reference_audio_uri))
            if not target_reference_path.exists():
                raise AegisVoxContractError(f"target reference audio not found for {speaker_id}: {target_reference_path}")
            target_embedding = self.embedding_backend.embed_wav(target_reference_path)
            source_converted_similarity = cosine_similarity(source_embedding, converted_embedding)
            target_converted_similarity = cosine_similarity(target_embedding, converted_embedding)
            row = {
                "speaker_id": speaker_id,
                "status": "computed",
                "converted_duration_sec": round(duration, 4),
                "num_converted_segments": len(converted_rows),
                "source_embedding_uri": source_embedding_uri,
                "converted_audio_uri": converted_audio_uri,
                "converted_embedding_uri": converted_embedding_uri,
                "target_reference_audio_uri": target_reference_audio_uri,
                "source_converted_similarity": round(source_converted_similarity, 6),
                "source_converted_distance": round(1.0 - source_converted_similarity, 6),
                "target_converted_similarity": round(target_converted_similarity, 6),
                "target_converted_distance": round(1.0 - target_converted_similarity, 6),
                "source_similarity_threshold": self.policy.max_source_converted_similarity,
                "target_similarity_threshold": self.policy.min_target_converted_similarity,
                "source_linkability_flag": source_converted_similarity > self.policy.max_source_converted_similarity,
                "target_similarity_flag": target_converted_similarity < self.policy.min_target_converted_similarity,
            }
            speaker_rows.append(row)
            converted_embedding_uris[str(speaker_id)] = converted_embedding_uri
            converted_audio_uris[str(speaker_id)] = converted_audio_uri
        computed = [row for row in speaker_rows if row.get("status") == "computed"]
        source_sims = [float(row["source_converted_similarity"]) for row in computed]
        target_sims = [float(row["target_converted_similarity"]) for row in computed]
        payload = {
            "service": "speaker_privacy_metrics",
            "backend": self.embedding_backend.backend_name,
            "backend_version": self.embedding_backend.backend_version,
            "policy": {
                "max_source_converted_similarity": self.policy.max_source_converted_similarity,
                "min_target_converted_similarity": self.policy.min_target_converted_similarity,
            },
            "speaker_metrics": speaker_rows,
            "summary": {
                "num_speakers": len(speaker_rows),
                "num_computed": len(computed),
                "num_missing_converted_audio": sum(1 for row in speaker_rows if row.get("status") == "missing_converted_audio"),
                "max_source_converted_similarity": max(source_sims) if source_sims else None,
                "min_target_converted_similarity": min(target_sims) if target_sims else None,
                "num_source_linkability_flags": sum(1 for row in computed if row.get("source_linkability_flag")),
                "num_target_similarity_flags": sum(1 for row in computed if row.get("target_similarity_flag")),
            },
            "artifacts": {
                "converted_audio_uris": converted_audio_uris,
                "converted_embedding_uris": converted_embedding_uris,
            },
            "claim_boundary": "embedding-based privacy/utility proxy; requires backend validation and adversarial privacy review before clinical claims",
        }
        self.artifact_store.write_json(output_metrics_uri, payload)
        return payload
