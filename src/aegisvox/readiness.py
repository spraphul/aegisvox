"""Readiness checks for AegisVox API deployments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audit import redact_text, sanitize_for_audit
from .services.voice_pool_retrieval import FaissVoicePoolRetrievalService, VoicePoolRetrievalService
from .storage import LocalArtifactStore


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "status": self.status}
        if self.detail:
            payload["detail"] = redact_text(self.detail, max_length=1000)
        if self.metadata:
            payload["metadata"] = sanitize_for_audit(dict(self.metadata), max_string_length=1000)
        return payload


def check_artifact_store(store: LocalArtifactStore) -> ReadinessCheck:
    probe_uri = f"s3://aegisvox/readiness/probe_{uuid4().hex}.json"
    try:
        path = store.write_json(probe_uri, {"ok": True})
        payload = store.read_json(probe_uri)
        try:
            path.unlink(missing_ok=True)
        except TypeError:  # pragma: no cover - py<3.8 compatibility guard
            if path.exists():
                path.unlink()
        if payload != {"ok": True}:
            return ReadinessCheck("artifact_store", "fail", "read-after-write probe returned unexpected payload")
        return ReadinessCheck(
            "artifact_store",
            "pass",
            metadata={"backend": "local", "root": str(store.root)},
        )
    except Exception as exc:
        return ReadinessCheck(
            "artifact_store",
            "fail",
            redact_text(str(exc), max_length=1000),
            metadata={"backend": "local", "root": str(getattr(store, "root", ""))},
        )


def check_metadata_store(metadata_store: object, *, backend: str) -> ReadinessCheck:
    try:
        jobs = metadata_store.list_queue_jobs(limit=1)
        return ReadinessCheck(
            "metadata_store",
            "pass",
            metadata={"backend": backend, "probe_jobs_seen": len(jobs)},
        )
    except Exception as exc:
        return ReadinessCheck("metadata_store", "fail", redact_text(str(exc), max_length=1000), metadata={"backend": backend})


def check_voice_pool(retrieval_service: object | None) -> ReadinessCheck:
    if retrieval_service is None:
        return ReadinessCheck(
            "voice_pool",
            "warn",
            "voice-pool retrieval is not configured; assignment services requiring pool search will fail loudly",
            metadata={"configured": False},
        )
    if isinstance(retrieval_service, VoicePoolRetrievalService):
        manifest_path = Path(retrieval_service.manifest_path)
        index_path = Path(retrieval_service.index_path)
        missing = [str(path) for path in [manifest_path, index_path] if not path.exists()]
        if missing:
            return ReadinessCheck(
                "voice_pool",
                "fail",
                "configured exact voice-pool files are missing",
                metadata={"configured": True, "mode": "exact", "missing": missing},
            )
        return ReadinessCheck(
            "voice_pool",
            "pass",
            metadata={"configured": True, "mode": "exact", "manifest_path": str(manifest_path), "index_path": str(index_path)},
        )
    if isinstance(retrieval_service, FaissVoicePoolRetrievalService):
        missing = []
        for uri in [retrieval_service.manifest_uri, retrieval_service.metadata_uri]:
            try:
                if not retrieval_service.artifact_store.exists(uri):
                    missing.append(uri)
            except Exception:
                missing.append(uri)
        if missing:
            return ReadinessCheck(
                "voice_pool",
                "fail",
                "configured FAISS voice-pool artifacts are missing",
                metadata={"configured": True, "mode": "faiss", "missing": missing},
            )
        sidecar_metadata: dict[str, Any] = {}
        hybrid_metadata: dict[str, Any] = {}
        try:
            payload = retrieval_service.artifact_store.read_json(retrieval_service.metadata_uri)
            if isinstance(payload, dict):
                sidecar_metadata = payload
                raw_hybrid_metadata = payload.get("hybrid_search_metadata")
                if isinstance(raw_hybrid_metadata, dict):
                    hybrid_metadata = raw_hybrid_metadata
        except Exception as exc:
            return ReadinessCheck(
                "voice_pool",
                "fail",
                "configured FAISS metadata sidecar could not be read",
                metadata={
                    "configured": True,
                    "mode": "faiss",
                    "manifest_uri": retrieval_service.manifest_uri,
                    "metadata_uri": retrieval_service.metadata_uri,
                    "error": str(exc),
                },
            )
        return ReadinessCheck(
            "voice_pool",
            "pass",
            metadata={
                "configured": True,
                "mode": "faiss",
                "manifest_uri": retrieval_service.manifest_uri,
                "metadata_uri": retrieval_service.metadata_uri,
                "faiss_index_uri": sidecar_metadata.get("faiss_index_uri"),
                "index_type": sidecar_metadata.get("index_type"),
                "metric": sidecar_metadata.get("metric"),
                "embedding_dim": sidecar_metadata.get("embedding_dim"),
                "num_vectors": sidecar_metadata.get("num_vectors"),
                "sidecar_manifest_uri": sidecar_metadata.get("manifest_uri"),
                "source_exact_index_uri": sidecar_metadata.get("source_exact_index_uri"),
                "hybrid_metadata_present": bool(hybrid_metadata),
                "hybrid_search_metadata": {
                    key: hybrid_metadata.get(key)
                    for key in [
                        "candidate_traits_source",
                        "trait_fields",
                        "hard_guardrail_traits",
                        "soft_rerank_traits",
                        "num_samples_with_audio_traits",
                        "trait_fill_rates",
                        "claim_boundary",
                    ]
                    if key in hybrid_metadata
                },
            },
        )
    return ReadinessCheck(
        "voice_pool",
        "warn",
        "unknown voice-pool retrieval service type",
        metadata={"configured": True, "type": type(retrieval_service).__name__},
    )


def readiness_snapshot(
    *,
    artifact_store: LocalArtifactStore,
    metadata_store: object,
    metadata_backend: str,
    voice_pool_retrieval: object | None,
) -> dict[str, Any]:
    checks = [
        check_artifact_store(artifact_store),
        check_metadata_store(metadata_store, backend=metadata_backend),
        check_voice_pool(voice_pool_retrieval),
    ]
    failed = [check for check in checks if check.status == "fail"]
    warned = [check for check in checks if check.status == "warn"]
    return {
        "ok": not failed,
        "status": "fail" if failed else ("warn" if warned else "pass"),
        "checks": [check.to_dict() for check in checks],
    }
