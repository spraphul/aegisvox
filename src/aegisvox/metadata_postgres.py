"""Postgres metadata store for production-like AegisVox deployments.

The SQLite store remains the local smoke path. This adapter is explicit and
fail-loud: constructing it requires psycopg and a reachable DSN. It implements
AegisVox's metadata-store surface with PostgreSQL parameter syntax and row
returning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .audit import sanitize_for_audit
from .metadata import _parse_iso_timestamp
from .metadata_migrations import MetadataMigrationError, migrate_postgres
from .schema import ArtifactRecord, EventEnvelope, RunManifest, utc_now_iso


class PostgresMetadataSetupError(RuntimeError):
    """Raised when the Postgres metadata store cannot be configured."""


@dataclass(frozen=True)
class PostgresMetadataStore:
    dsn: str
    initialize_schema: bool = True

    def __init__(self, dsn: str, *, initialize_schema: bool = True):
        if not dsn:
            raise PostgresMetadataSetupError("Postgres metadata backend requires AEGISVOX_POSTGRES_DSN")
        try:
            import psycopg  # type: ignore[import-not-found]
            from psycopg.rows import dict_row  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - environment-dependent import failure
            raise PostgresMetadataSetupError(
                "Postgres metadata backend requires psycopg. Install with `python -m pip install '.[postgres]'`."
            ) from exc
        object.__setattr__(self, "dsn", dsn)
        object.__setattr__(self, "initialize_schema", initialize_schema)
        object.__setattr__(self, "_psycopg", psycopg)
        object.__setattr__(self, "_row_factory", dict_row)
        if initialize_schema:
            try:
                migrate_postgres(dsn)
            except MetadataMigrationError as exc:
                raise PostgresMetadataSetupError(str(exc)) from exc

    def connect(self):
        return self._psycopg.connect(self.dsn, row_factory=self._row_factory)

    def upsert_run(self, manifest: RunManifest, manifest_uri: str) -> None:
        request = manifest.request.to_dict()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runs (
                      run_id, status, input_audio_uri, privacy_mode, voice_pool_id,
                      language_hint, enable_lexical_deid, enable_speech_editing,
                      overlap_mode, manifest_uri, created_at, updated_at,
                      failure_reason, request_json, service_versions_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(run_id) DO UPDATE SET
                      status=excluded.status,
                      updated_at=excluded.updated_at,
                      failure_reason=excluded.failure_reason,
                      service_versions_json=excluded.service_versions_json
                    """,
                    (
                        manifest.run_id,
                        manifest.status.value,
                        request["input_audio_uri"],
                        request["privacy_mode"],
                        request["voice_pool_id"],
                        request.get("language_hint"),
                        int(bool(request["enable_lexical_deid"])),
                        int(bool(request["enable_speech_editing"])),
                        request["overlap_mode"],
                        manifest_uri,
                        manifest.created_at,
                        manifest.updated_at,
                        manifest.failure_reason,
                        json.dumps(request, sort_keys=True),
                        json.dumps(manifest.service_versions, sort_keys=True),
                    ),
                )
                for event in manifest.events:
                    cur.execute(
                        """
                        INSERT INTO events (
                          event_id, run_id, event_type, service, emitted_at, payload_json
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT(event_id) DO NOTHING
                        """,
                        (
                            event.event_id,
                            event.run_id,
                            event.event_type,
                            event.service,
                            event.emitted_at,
                            json.dumps(event.payload, sort_keys=True),
                        ),
                    )
                for artifact in manifest.artifacts.values():
                    cur.execute(
                        """
                        INSERT INTO artifacts (run_id, name, uri, artifact_type, created_at, metadata_json)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT(run_id, name) DO UPDATE SET
                          uri=excluded.uri,
                          artifact_type=excluded.artifact_type,
                          created_at=excluded.created_at,
                          metadata_json=excluded.metadata_json
                        """,
                        (
                            manifest.run_id,
                            artifact.name,
                            artifact.uri,
                            artifact.artifact_type,
                            artifact.created_at,
                            json.dumps(artifact.metadata, sort_keys=True),
                        ),
                    )

    def insert_event(self, event: EventEnvelope) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events (event_id, run_id, event_type, service, emitted_at, payload_json)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    (
                        event.event_id,
                        event.run_id,
                        event.event_type,
                        event.service,
                        event.emitted_at,
                        json.dumps(event.payload, sort_keys=True),
                    ),
                )

    def upsert_artifact(self, run_id: str, artifact: ArtifactRecord) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO artifacts (run_id, name, uri, artifact_type, created_at, metadata_json)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(run_id, name) DO UPDATE SET
                      uri=excluded.uri,
                      artifact_type=excluded.artifact_type,
                      created_at=excluded.created_at,
                      metadata_json=excluded.metadata_json
                    """,
                    (
                        run_id,
                        artifact.name,
                        artifact.uri,
                        artifact.artifact_type,
                        artifact.created_at,
                        json.dumps(artifact.metadata, sort_keys=True),
                    ),
                )

    def get_run_row(self, run_id: str) -> dict | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def list_events(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM events WHERE run_id = %s ORDER BY emitted_at, event_id",
                    (run_id,),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def list_artifacts(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM artifacts WHERE run_id = %s ORDER BY created_at, name",
                    (run_id,),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def list_jobs(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM jobs WHERE run_id = %s ORDER BY created_at, job_id",
                    (run_id,),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def list_queue_jobs(
        self,
        *,
        service: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[Any] = []
        if service:
            clauses.append("service = %s")
            params.append(service)
        if status:
            clauses.append("status = %s")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 1000)))
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM jobs {where} ORDER BY created_at, job_id LIMIT %s", params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def create_job(self, job_id: str, run_id: str, service: str, input_artifacts: Iterable[str] = ()) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO jobs (job_id, run_id, service, status, created_at, updated_at, input_artifacts_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (job_id, run_id, service, "created", now, now, json.dumps(list(input_artifacts))),
                )

    def update_job_status(
        self,
        job_id: str,
        status: str,
        output_artifacts: Iterable[str] = (),
        error: dict | None = None,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = %s,
                        updated_at = %s,
                        output_artifacts_json = %s,
                        error_json = %s,
                        heartbeat_at = CASE WHEN %s IN ('completed', 'failed', 'canceled') THEN NULL ELSE heartbeat_at END
                    WHERE job_id = %s
                    """,
                    (
                        status,
                        now,
                        json.dumps(list(output_artifacts)),
                        json.dumps(sanitize_for_audit(error), sort_keys=True) if error is not None else None,
                        status,
                        job_id,
                    ),
                )

    def update_latest_job_for_service(
        self,
        run_id: str,
        service: str,
        status: str,
        output_artifacts: Iterable[str] = (),
        error: dict | None = None,
    ) -> str | None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT job_id
                    FROM jobs
                    WHERE run_id = %s AND service = %s
                    ORDER BY created_at DESC, job_id DESC
                    LIMIT 1
                    """,
                    (run_id, service),
                )
                row = cur.fetchone()
        if row is None:
            return None
        job_id = str(row["job_id"])
        self.update_job_status(job_id, status, output_artifacts=output_artifacts, error=error)
        return job_id

    def claim_next_job(
        self,
        service: str,
        worker_id: str | None = None,
        timeout_seconds: int | None = None,
    ) -> dict | None:
        now = utc_now_iso()
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT job_id
                    FROM jobs
                    WHERE service = %s AND status = 'created'
                    ORDER BY created_at, job_id
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                    """,
                    (service,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                job_id = str(row["job_id"])
                cur.execute(
                    """
                    UPDATE jobs
                    SET status = 'running',
                        updated_at = %s,
                        worker_id = %s,
                        claimed_at = %s,
                        heartbeat_at = %s,
                        attempt = COALESCE(attempt, 0) + 1,
                        timeout_seconds = %s,
                        error_json = NULL
                    WHERE job_id = %s AND status = 'created'
                    RETURNING *
                    """,
                    (now, worker_id, now, now, timeout_seconds, job_id),
                )
                claimed = cur.fetchone()
        return dict(claimed) if claimed else None

    def heartbeat_job(self, job_id: str, worker_id: str | None = None) -> dict | None:
        now = utc_now_iso()
        with self.connect() as conn:
            with conn.cursor() as cur:
                if worker_id is None:
                    cur.execute(
                        """
                        UPDATE jobs
                        SET heartbeat_at = %s, updated_at = %s
                        WHERE job_id = %s AND status = 'running'
                        RETURNING *
                        """,
                        (now, now, job_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE jobs
                        SET heartbeat_at = %s, updated_at = %s
                        WHERE job_id = %s AND status = 'running' AND worker_id = %s
                        RETURNING *
                        """,
                        (now, now, job_id, worker_id),
                    )
                row = cur.fetchone()
        return dict(row) if row else None

    def requeue_stale_jobs(self, *, stale_after_seconds: int, service: str | None = None) -> list[dict]:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        now_iso = utc_now_iso()
        now_dt = _parse_iso_timestamp(now_iso)
        if now_dt is None:
            return []
        clauses = ["jobs.status = 'running'", "runs.status NOT IN ('completed', 'failed', 'canceled')"]
        params: list[Any] = []
        if service:
            clauses.append("jobs.service = %s")
            params.append(service)
        where = " AND ".join(clauses)
        requeued: list[dict] = []
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT jobs.*
                    FROM jobs
                    JOIN runs ON jobs.run_id = runs.run_id
                    WHERE {where}
                    ORDER BY jobs.created_at, jobs.job_id
                    FOR UPDATE OF jobs
                    """,
                    params,
                )
                rows = cur.fetchall()
                for row in rows:
                    row_dict = dict(row)
                    last_seen = (
                        _parse_iso_timestamp(row_dict.get("heartbeat_at"))
                        or _parse_iso_timestamp(row_dict.get("claimed_at"))
                        or _parse_iso_timestamp(row_dict.get("updated_at"))
                    )
                    if last_seen is None or (now_dt - last_seen).total_seconds() <= stale_after_seconds:
                        continue
                    error = {
                        "reason": "worker_heartbeat_timeout",
                        "previous_worker_id": row_dict.get("worker_id"),
                        "stale_after_seconds": stale_after_seconds,
                        "last_heartbeat_at": row_dict.get("heartbeat_at"),
                        "requeued_at": now_iso,
                    }
                    cur.execute(
                        """
                        UPDATE jobs
                        SET status = 'created',
                            updated_at = %s,
                            worker_id = NULL,
                            claimed_at = NULL,
                            heartbeat_at = NULL,
                            timeout_seconds = NULL,
                            error_json = %s
                        WHERE job_id = %s AND status = 'running'
                        RETURNING *
                        """,
                        (now_iso, json.dumps(sanitize_for_audit(error), sort_keys=True), row_dict["job_id"]),
                    )
                    updated = cur.fetchone()
                    if updated is not None:
                        requeued.append(dict(updated))
        return requeued
