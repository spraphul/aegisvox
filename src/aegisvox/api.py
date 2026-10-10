"""FastAPI app for the AegisVox orchestrator service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .audit import redact_text
from .auth import APIAuthConfig
from .audio_trait_extraction import AudioTraitExtractionError, extract_audio_traits_with_openai
from .events import InMemoryEventBus
from .manifest import manifest_uri
from .metadata_factory import create_metadata_store
from .queue import AegisVoxQueue
from .readiness import readiness_snapshot
from .schema import AegisVoxContractError, BackgroundPreservationMode, OverlapMode, PrivacyMode, RunRequest
from .schema import validate_artifact_uri
from .services.orchestrator import Orchestrator
from .services.voice_pool_retrieval import FaissVoicePoolRetrievalService, VoicePoolRetrievalService
from .storage import LocalArtifactStore
from .voice_pool import CANONICAL_AUDIO_TRAIT_KEYS, VoicePoolSetupError, VoiceSearchRequest, effective_trait_weights, request_audio_traits


class CreateRunPayload(BaseModel):
    input_audio_uri: str
    privacy_mode: PrivacyMode = PrivacyMode.MEDIUM
    voice_pool_id: str = "openswara_clinical_v1"
    language_hint: str | None = "en"
    enable_lexical_deid: bool = True
    enable_speech_editing: bool = True
    overlap_mode: OverlapMode = OverlapMode.SKIP_AND_FLAG
    background_preservation_mode: BackgroundPreservationMode = BackgroundPreservationMode.DISABLED
    experiment_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Audit-safe per-run experiment YAML slice. Do not include secrets or raw transcripts.",
    )
    run_id: str | None = Field(default=None, description="Optional deterministic run id for tests/replays.")


class CreateRunResponse(BaseModel):
    run_id: str
    status: str
    manifest_uri: str
    events_emitted: list[str]


class VoicePoolSearchPayload(BaseModel):
    voice_pool_id: str = "openswara_clinical_v1"
    query_embedding: list[float] | None = None
    language: str | None = None
    pitch_band: str | None = None
    speaking_rate: str | None = None
    energy: str | None = None
    gender_presentation: str | None = None
    accent_group: str | None = None
    geography_region: str | None = None
    tone: str | None = None
    audio_traits: dict[str, Any] = Field(default_factory=dict)
    query_audio_uri: str | None = Field(
        default=None,
        description="Optional artifact URI for query-time audio trait extraction with the configured audio model.",
    )
    extract_audio_traits: bool = False
    audio_trait_model: str | None = None
    trait_weights: dict[str, float] = Field(default_factory=dict)
    vector_weight: float = 1.0
    trait_weight: float = 0.35
    exclude_voice_ids: list[str] = Field(default_factory=list)
    min_source_distance: float | None = None
    require_known_hard_trait_matches: bool = Field(
        default=False,
        description=(
            "When true, candidates missing a known query language/gender/accent hard guardrail are rejected "
            "instead of merely warning. Use for assignment/candidate generation; leave false for exploratory search."
        ),
    )
    top_k: int = 10


class VoicePoolSearchResponse(BaseModel):
    voice_pool_id: str
    candidates: list[dict[str, Any]]
    query_audio_traits: dict[str, Any] = Field(default_factory=dict)
    trait_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Effective hybrid trait weights after default weights and request overrides.",
    )
    effective_query_audio_traits: dict[str, Any] = Field(
        default_factory=dict,
        description="Merged traits that actually drive search after model extraction and request overrides.",
    )
    query_trait_sources: dict[str, str] = Field(
        default_factory=dict,
        description="Per-trait source after precedence: top_level_fields, audio_traits, or model_inferred_traits.",
    )
    query_trait_precedence: list[str] = Field(
        default_factory=lambda: ["top_level_fields", "audio_traits", "model_inferred_traits"],
        description="Trait precedence from strongest to weakest.",
    )


class ServiceCompletionPayload(BaseModel):
    event_type: str = Field(description="Canonical completed event, e.g. diarization.completed")
    service: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ClaimJobPayload(BaseModel):
    service: str
    worker_id: str | None = None
    timeout_seconds: int | None = Field(default=None, ge=1)


class UpdateJobStatusPayload(BaseModel):
    status: str
    output_artifacts: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None


class HeartbeatJobPayload(BaseModel):
    worker_id: str | None = None


class RequeueStaleJobsPayload(BaseModel):
    stale_after_seconds: int = Field(default=900, ge=1)
    service: str | None = None


class RunFailurePayload(BaseModel):
    service: str
    error: dict[str, Any] | str
    job_id: str | None = None


class RunRetryPayload(BaseModel):
    service: str | None = Field(default=None, description="Service to retry; defaults to latest failed job service.")
    job_id: str | None = Field(default=None, description="Specific failed job to retry.")
    reason: str | None = Field(default=None, description="Operator-visible retry reason.")


def _safe_http_detail(exc: Exception | str) -> str:
    return redact_text(str(exc), max_length=1000)


def create_app(
    artifact_root: str | Path | None = None,
    metadata_db_path: str | Path | None = None,
    metadata_backend: str | None = None,
    postgres_dsn: str | None = None,
    bucket: str = "aegisvox",
    voice_pool_manifest_path: str | Path | None = None,
    voice_pool_index_path: str | Path | None = None,
    voice_pool_manifest_uri: str | None = None,
    voice_pool_faiss_metadata_uri: str | None = None,
    api_auth_required: bool | None = None,
    api_keys: list[str] | tuple[str, ...] | None = None,
    bearer_tokens: list[str] | tuple[str, ...] | None = None,
) -> FastAPI:
    artifact_root = artifact_root or os.environ.get("AEGISVOX_ARTIFACT_ROOT", "/tmp/aegisvox_artifacts")
    metadata_db_path = metadata_db_path or os.environ.get("AEGISVOX_METADATA_DB", str(Path(artifact_root) / "metadata.sqlite3"))
    selected_metadata_backend = (metadata_backend or os.environ.get("AEGISVOX_METADATA_BACKEND") or "sqlite").strip().lower()
    store = LocalArtifactStore(artifact_root)
    event_bus = InMemoryEventBus()
    metadata = create_metadata_store(
        backend=metadata_backend,
        sqlite_path=metadata_db_path,
        postgres_dsn=postgres_dsn,
    )
    queue = AegisVoxQueue(metadata)
    orchestrator = Orchestrator(store, event_bus, bucket=bucket, metadata_store=metadata)
    voice_pool_manifest_path = voice_pool_manifest_path or os.environ.get("AEGISVOX_VOICE_POOL_MANIFEST")
    voice_pool_index_path = voice_pool_index_path or os.environ.get("AEGISVOX_VOICE_POOL_INDEX")
    voice_pool_manifest_uri = voice_pool_manifest_uri or os.environ.get("AEGISVOX_VOICE_POOL_MANIFEST_URI")
    voice_pool_faiss_metadata_uri = voice_pool_faiss_metadata_uri or os.environ.get("AEGISVOX_VOICE_POOL_FAISS_METADATA_URI")
    api_auth = APIAuthConfig.from_env(required=api_auth_required, api_keys=api_keys, bearer_tokens=bearer_tokens)
    retrieval_service = None
    if voice_pool_manifest_uri and voice_pool_faiss_metadata_uri:
        retrieval_service = FaissVoicePoolRetrievalService(store, voice_pool_manifest_uri, voice_pool_faiss_metadata_uri)
    elif voice_pool_manifest_path and voice_pool_index_path:
        retrieval_service = VoicePoolRetrievalService(Path(voice_pool_manifest_path), Path(voice_pool_index_path))

    app = FastAPI(title="AegisVox Orchestrator", version=__version__)
    app.state.artifact_store = store
    app.state.event_bus = event_bus
    app.state.metadata = metadata
    app.state.metadata_backend = selected_metadata_backend
    app.state.queue = queue
    app.state.orchestrator = orchestrator
    app.state.voice_pool_retrieval = retrieval_service
    app.state.api_auth = api_auth

    @app.middleware("http")
    async def api_auth_middleware(request: Request, call_next):
        auth_config: APIAuthConfig = request.app.state.api_auth
        if auth_config.protects(request.url.path):
            if not auth_config.authorize(
                authorization_header=request.headers.get("authorization"),
                api_key_header=request.headers.get("x-api-key"),
            ):
                return JSONResponse(status_code=401, content={"detail": "unauthorized"}, headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True, "service": "orchestrator", "version": __version__}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        payload = readiness_snapshot(
            artifact_store=store,
            metadata_store=metadata,
            metadata_backend=selected_metadata_backend,
            voice_pool_retrieval=app.state.voice_pool_retrieval,
        )
        payload["checks"].append({"name": "api_auth", "status": "pass", "metadata": app.state.api_auth.status()})
        return payload

    @app.post("/v1/runs", response_model=CreateRunResponse)
    def create_run(payload: CreateRunPayload) -> CreateRunResponse:
        try:
            request = RunRequest(
                input_audio_uri=payload.input_audio_uri,
                privacy_mode=payload.privacy_mode,
                voice_pool_id=payload.voice_pool_id,
                language_hint=payload.language_hint,
                enable_lexical_deid=payload.enable_lexical_deid,
                enable_speech_editing=payload.enable_speech_editing,
                overlap_mode=payload.overlap_mode,
                background_preservation_mode=payload.background_preservation_mode,
                experiment_config=payload.experiment_config,
            )
            manifest = orchestrator.create_run(request, run_id=payload.run_id)
        except AegisVoxContractError as exc:
            raise HTTPException(status_code=422, detail=_safe_http_detail(exc)) from exc
        uri = manifest_uri(manifest.run_id, bucket=bucket)
        return CreateRunResponse(
            run_id=manifest.run_id,
            status=manifest.status.value,
            manifest_uri=uri,
            events_emitted=[event.event_type for event in event_bus.by_run(manifest.run_id)],
        )


    @app.post("/v1/voice-pool/search", response_model=VoicePoolSearchResponse)
    def search_voice_pool(payload: VoicePoolSearchPayload) -> VoicePoolSearchResponse:
        if app.state.voice_pool_retrieval is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "voice pool retrieval is not configured; set exact AEGISVOX_VOICE_POOL_MANIFEST "
                    "and AEGISVOX_VOICE_POOL_INDEX, or FAISS AEGISVOX_VOICE_POOL_MANIFEST_URI "
                    "and AEGISVOX_VOICE_POOL_FAISS_METADATA_URI after running the real voice pool builder"
                ),
            )
        try:
            extracted_traits: dict[str, Any] = {}
            model_inferred_audio_traits: dict[str, Any] = {}
            merged_audio_traits = dict(payload.audio_traits)
            if payload.extract_audio_traits:
                if not payload.query_audio_uri:
                    raise AegisVoxContractError("query_audio_uri is required when extract_audio_traits=true")
                validate_artifact_uri(payload.query_audio_uri)
                local_query_audio = store.local_path(payload.query_audio_uri)
                extracted_traits = extract_audio_traits_with_openai(
                    local_query_audio,
                    model=payload.audio_trait_model or os.environ.get("AEGISVOX_AUDIO_TRAIT_MODEL", "gpt-audio-1.5"),
                )
                model_inferred_audio_traits = (
                    dict(extracted_traits.get("audio_traits", {})) if isinstance(extracted_traits.get("audio_traits"), dict) else {}
                )
                # Explicit request traits override model-inferred values.
                merged_audio_traits = {**model_inferred_audio_traits, **merged_audio_traits}
            # Promote query-time audio traits into the explicit request fields
            # used by retrieval guardrails/reranking. Explicit top-level fields
            # still win over model-inferred or audio_traits values.
            request = VoiceSearchRequest(
                voice_pool_id=payload.voice_pool_id,
                query_embedding=payload.query_embedding,
                language=payload.language if payload.language is not None else merged_audio_traits.get("language"),
                pitch_band=payload.pitch_band if payload.pitch_band is not None else merged_audio_traits.get("pitch_band"),
                speaking_rate=payload.speaking_rate if payload.speaking_rate is not None else merged_audio_traits.get("speaking_rate"),
                energy=payload.energy if payload.energy is not None else merged_audio_traits.get("energy"),
                gender_presentation=payload.gender_presentation
                if payload.gender_presentation is not None
                else merged_audio_traits.get("gender_presentation"),
                accent_group=payload.accent_group if payload.accent_group is not None else merged_audio_traits.get("accent_group"),
                geography_region=payload.geography_region if payload.geography_region is not None else merged_audio_traits.get("geography_region"),
                tone=payload.tone if payload.tone is not None else merged_audio_traits.get("tone"),
                audio_traits=merged_audio_traits,
                trait_weights=payload.trait_weights,
                vector_weight=payload.vector_weight,
                trait_weight=payload.trait_weight,
                exclude_voice_ids=payload.exclude_voice_ids,
                min_source_distance=payload.min_source_distance,
                require_known_hard_trait_matches=payload.require_known_hard_trait_matches,
                top_k=payload.top_k,
            )
            results = app.state.voice_pool_retrieval.search(request)
        except AegisVoxContractError as exc:
            raise HTTPException(status_code=422, detail=_safe_http_detail(exc)) from exc
        except AudioTraitExtractionError as exc:
            raise HTTPException(status_code=503, detail=_safe_http_detail(exc)) from exc
        except VoicePoolSetupError as exc:
            raise HTTPException(status_code=503, detail=_safe_http_detail(exc)) from exc
        # Return the canonical trait map used by retrieval, not just the raw
        # request/model surface forms.  This keeps the API/debug contract aligned
        # with guardrail behavior after alias normalization (for example,
        # `US English` -> `english`, `General American` -> `american_english`).
        effective_query_audio_traits = request_audio_traits(request)
        query_trait_sources: dict[str, str] = {}
        for key in CANONICAL_AUDIO_TRAIT_KEYS:
            if key in model_inferred_audio_traits and key in effective_query_audio_traits:
                query_trait_sources[key] = "model_inferred_traits"
            if key in payload.audio_traits and key in effective_query_audio_traits:
                query_trait_sources[key] = "audio_traits"
            if getattr(payload, key, None) is not None and key in effective_query_audio_traits:
                query_trait_sources[key] = "top_level_fields"
        return VoicePoolSearchResponse(
            voice_pool_id=payload.voice_pool_id,
            candidates=[result.to_dict() for result in results],
            query_audio_traits=extracted_traits,
            trait_weights=effective_trait_weights(payload.trait_weights),
            effective_query_audio_traits=effective_query_audio_traits,
            query_trait_sources=query_trait_sources,
        )

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        row = metadata.get_run_row(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run": row, "events": metadata.list_events(run_id)}

    @app.post("/v1/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            manifest = orchestrator.cancel_run(run_id)
        except AegisVoxContractError as exc:
            status_code = 404 if "not found" in str(exc) else 409
            raise HTTPException(status_code=status_code, detail=_safe_http_detail(exc)) from exc
        return {
            "run_id": manifest.run_id,
            "status": manifest.status.value,
            "manifest_uri": manifest_uri(manifest.run_id, bucket=bucket),
            "events_emitted": [event.event_type for event in event_bus.by_run(manifest.run_id)],
        }

    @app.post("/v1/runs/{run_id}/fail")
    def fail_run(run_id: str, payload: RunFailurePayload) -> dict[str, Any]:
        try:
            manifest = orchestrator.record_service_failure(
                run_id=run_id,
                service=payload.service,
                error=payload.error,
                job_id=payload.job_id,
            )
        except AegisVoxContractError as exc:
            status_code = 404 if "not found" in str(exc) else 409
            raise HTTPException(status_code=status_code, detail=_safe_http_detail(exc)) from exc
        return {
            "run_id": manifest.run_id,
            "status": manifest.status.value,
            "failure_reason": manifest.failure_reason,
            "events": metadata.list_events(run_id),
            "jobs": metadata.list_jobs(run_id),
        }

    @app.post("/v1/runs/{run_id}/retry")
    def retry_run(run_id: str, payload: RunRetryPayload) -> dict[str, Any]:
        try:
            manifest = orchestrator.retry_failed_service(
                run_id=run_id,
                service=payload.service,
                job_id=payload.job_id,
                reason=payload.reason,
            )
        except AegisVoxContractError as exc:
            status_code = 404 if "not found" in str(exc) else 409
            raise HTTPException(status_code=status_code, detail=_safe_http_detail(exc)) from exc
        return {
            "run_id": manifest.run_id,
            "status": manifest.status.value,
            "events": metadata.list_events(run_id),
            "jobs": metadata.list_jobs(run_id),
        }

    @app.get("/v1/runs/{run_id}/events")
    def get_run_events(run_id: str) -> dict[str, Any]:
        if metadata.get_run_row(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run_id": run_id, "events": metadata.list_events(run_id)}

    @app.post("/v1/runs/{run_id}/events")
    def record_run_event(run_id: str, payload: ServiceCompletionPayload) -> dict[str, Any]:
        try:
            manifest = orchestrator.record_service_completion(
                run_id=run_id,
                event_type=payload.event_type,
                service=payload.service,
                payload=payload.payload,
            )
        except AegisVoxContractError as exc:
            status_code = 404 if "not found" in str(exc) else 422
            raise HTTPException(status_code=status_code, detail=_safe_http_detail(exc)) from exc
        return {
            "run_id": manifest.run_id,
            "status": manifest.status.value,
            "events": metadata.list_events(run_id),
            "jobs": metadata.list_jobs(run_id),
        }

    @app.get("/v1/runs/{run_id}/artifacts")
    def get_run_artifacts(run_id: str) -> dict[str, Any]:
        if metadata.get_run_row(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run_id": run_id, "artifacts": metadata.list_artifacts(run_id)}

    @app.get("/v1/runs/{run_id}/jobs")
    def get_run_jobs(run_id: str) -> dict[str, Any]:
        if metadata.get_run_row(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run_id": run_id, "jobs": metadata.list_jobs(run_id)}

    @app.get("/v1/jobs")
    def list_jobs(service: str | None = None, status: str | None = None, limit: int = 100) -> dict[str, Any]:
        return {"jobs": queue.list_jobs(service=service, status=status, limit=limit)}

    @app.post("/v1/jobs/claim")
    def claim_job(payload: ClaimJobPayload) -> dict[str, Any]:
        claim = queue.claim(
            payload.service,
            worker_id=payload.worker_id,
            timeout_seconds=payload.timeout_seconds,
        )
        if claim is None:
            raise HTTPException(status_code=404, detail=f"no created jobs available for service {payload.service!r}")
        return {"job": claim.job}

    @app.post("/v1/jobs/requeue-stale")
    def requeue_stale_jobs(payload: RequeueStaleJobsPayload) -> dict[str, Any]:
        try:
            jobs = queue.requeue_stale(
                stale_after_seconds=payload.stale_after_seconds,
                service=payload.service,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=_safe_http_detail(exc)) from exc
        return {"requeued": jobs, "count": len(jobs)}

    @app.post("/v1/jobs/{job_id}/heartbeat")
    def heartbeat_job(job_id: str, payload: HeartbeatJobPayload) -> dict[str, Any]:
        job = queue.heartbeat(job_id, worker_id=payload.worker_id)
        if job is None:
            raise HTTPException(status_code=404, detail="running job not found for heartbeat")
        return {"job": job}

    @app.post("/v1/jobs/{job_id}/status")
    def update_job_status(job_id: str, payload: UpdateJobStatusPayload) -> dict[str, Any]:
        queue.update_status(
            job_id,
            payload.status,
            output_artifacts=payload.output_artifacts,
            error=payload.error,
        )
        job = queue.get_job(job_id)
        if job is not None:
            return {"job": job}
        raise HTTPException(status_code=404, detail="job not found")

    return app


app = create_app()
