"""Metadata persistence for AegisVox runs, artifacts, and events.

Phase 1 uses SQLite for local/ORD smoke. The schema is intentionally close to
Postgres so a later Postgres adapter can share the same run/artifact/event
contract.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .audit import sanitize_for_audit
from .schema import ArtifactRecord, EventEnvelope, RunManifest, utc_now_iso


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  input_audio_uri TEXT NOT NULL,
  privacy_mode TEXT NOT NULL,
  voice_pool_id TEXT NOT NULL,
  language_hint TEXT,
  enable_lexical_deid INTEGER NOT NULL,
  enable_speech_editing INTEGER NOT NULL,
  overlap_mode TEXT NOT NULL,
  manifest_uri TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  failure_reason TEXT,
  request_json TEXT NOT NULL,
  service_versions_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  service TEXT NOT NULL,
  emitted_at TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
  run_id TEXT NOT NULL,
  name TEXT NOT NULL,
  uri TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  PRIMARY KEY(run_id, name),
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  service TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  input_artifacts_json TEXT NOT NULL DEFAULT '[]',
  output_artifacts_json TEXT NOT NULL DEFAULT '[]',
  error_json TEXT,
  worker_id TEXT,
  claimed_at TEXT,
  heartbeat_at TEXT,
  attempt INTEGER NOT NULL DEFAULT 0,
  timeout_seconds INTEGER,
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
"""


JOB_COLUMN_MIGRATIONS: dict[str, str] = {
    "worker_id": "ALTER TABLE jobs ADD COLUMN worker_id TEXT",
    "claimed_at": "ALTER TABLE jobs ADD COLUMN claimed_at TEXT",
    "heartbeat_at": "ALTER TABLE jobs ADD COLUMN heartbeat_at TEXT",
    "attempt": "ALTER TABLE jobs ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0",
    "timeout_seconds": "ALTER TABLE jobs ADD COLUMN timeout_seconds INTEGER",
}


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class SQLiteMetadataStore:
    """Small metadata store used for local smoke and API development."""

    db_path: Path

    def __init__(self, db_path: str | Path):
        path = Path(db_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "db_path", path)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._ensure_job_columns(conn)

    def _ensure_job_columns(self, conn: sqlite3.Connection) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        for column, ddl in JOB_COLUMN_MIGRATIONS.items():
            if column not in existing:
                conn.execute(ddl)

    def upsert_run(self, manifest: RunManifest, manifest_uri: str) -> None:
        request = manifest.request.to_dict()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                  run_id, status, input_audio_uri, privacy_mode, voice_pool_id,
                  language_hint, enable_lexical_deid, enable_speech_editing,
                  overlap_mode, manifest_uri, created_at, updated_at,
                  failure_reason, request_json, service_versions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                conn.execute(
                    """
                    INSERT OR IGNORE INTO events (
                      event_id, run_id, event_type, service, emitted_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
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
                conn.execute(
                    """
                    INSERT INTO artifacts (run_id, name, uri, artifact_type, created_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
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
            conn.execute(
                """
                INSERT OR IGNORE INTO events (
                  event_id, run_id, event_type, service, emitted_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
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
            conn.execute(
                """
                INSERT INTO artifacts (run_id, name, uri, artifact_type, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
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
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_events(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_artifacts(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_jobs(self, run_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_queue_jobs(
        self,
        *,
        service: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[object] = []
        if service:
            clauses.append("service = ?")
            params.append(service)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 1000)))
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs {where} ORDER BY rowid LIMIT ?",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def create_job(self, job_id: str, run_id: str, service: str, input_artifacts: Iterable[str] = ()) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, run_id, service, status, created_at, updated_at, input_artifacts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
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
            conn.execute(
                """
                UPDATE jobs
                SET status = ?,
                    updated_at = ?,
                    output_artifacts_json = ?,
                    error_json = ?,
                    heartbeat_at = CASE WHEN ? IN ('completed', 'failed', 'canceled') THEN NULL ELSE heartbeat_at END
                WHERE job_id = ?
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
            row = conn.execute(
                """
                SELECT job_id
                FROM jobs
                WHERE run_id = ? AND service = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (run_id, service),
            ).fetchone()
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
            row = conn.execute(
                """
                SELECT *
                FROM jobs
                WHERE service = ? AND status = 'created'
                ORDER BY rowid
                LIMIT 1
                """,
                (service,),
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["job_id"])
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    updated_at = ?,
                    worker_id = ?,
                    claimed_at = ?,
                    heartbeat_at = ?,
                    attempt = COALESCE(attempt, 0) + 1,
                    timeout_seconds = ?,
                    error_json = NULL
                WHERE job_id = ? AND status = 'created'
                """,
                (now, worker_id, now, now, timeout_seconds, job_id),
            )
            claimed = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(claimed) if claimed else None

    def heartbeat_job(self, job_id: str, worker_id: str | None = None) -> dict | None:
        now = utc_now_iso()
        with self.connect() as conn:
            if worker_id is None:
                row = conn.execute(
                    """
                    UPDATE jobs
                    SET heartbeat_at = ?, updated_at = ?
                    WHERE job_id = ? AND status = 'running'
                    RETURNING *
                    """,
                    (now, now, job_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    UPDATE jobs
                    SET heartbeat_at = ?, updated_at = ?
                    WHERE job_id = ? AND status = 'running' AND worker_id = ?
                    RETURNING *
                    """,
                    (now, now, job_id, worker_id),
                ).fetchone()
        return dict(row) if row else None

    def requeue_stale_jobs(self, *, stale_after_seconds: int, service: str | None = None) -> list[dict]:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be positive")
        now_iso = utc_now_iso()
        now_dt = _parse_iso_timestamp(now_iso)
        if now_dt is None:
            return []

        clauses = ["jobs.status = 'running'", "runs.status NOT IN ('completed', 'failed', 'canceled')"]
        params: list[object] = []
        if service:
            clauses.append("jobs.service = ?")
            params.append(service)
        where = " AND ".join(clauses)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT jobs.*
                FROM jobs
                JOIN runs ON jobs.run_id = runs.run_id
                WHERE {where}
                ORDER BY jobs.rowid
                """,
                params,
            ).fetchall()
            stale_rows: list[dict] = []
            for row in rows:
                row_dict = dict(row)
                last_seen = (
                    _parse_iso_timestamp(row_dict.get("heartbeat_at"))
                    or _parse_iso_timestamp(row_dict.get("claimed_at"))
                    or _parse_iso_timestamp(row_dict.get("updated_at"))
                )
                if last_seen is None:
                    continue
                if (now_dt - last_seen).total_seconds() <= stale_after_seconds:
                    continue
                stale_rows.append(row_dict)

            requeued: list[dict] = []
            for row in stale_rows:
                error = {
                    "reason": "worker_heartbeat_timeout",
                    "previous_worker_id": row.get("worker_id"),
                    "stale_after_seconds": stale_after_seconds,
                    "last_heartbeat_at": row.get("heartbeat_at"),
                    "requeued_at": now_iso,
                }
                updated = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'created',
                        updated_at = ?,
                        worker_id = NULL,
                        claimed_at = NULL,
                        heartbeat_at = NULL,
                        timeout_seconds = NULL,
                        error_json = ?
                    WHERE job_id = ? AND status = 'running'
                    RETURNING *
                    """,
                    (now_iso, json.dumps(sanitize_for_audit(error), sort_keys=True), row["job_id"]),
                ).fetchone()
                if updated is not None:
                    requeued.append(dict(updated))
        return requeued
