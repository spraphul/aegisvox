"""Lightweight queue abstraction over the AegisVox metadata store.

This is the shared control-plane interface used before introducing Redis/RQ,
Celery, Kafka, or a production workflow engine. It deliberately stores only job
state and artifact URIs; it never fabricates model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class QueueClaim:
    job: dict[str, Any]

    @property
    def job_id(self) -> str:
        return str(self.job["job_id"])

    @property
    def run_id(self) -> str:
        return str(self.job["run_id"])

    @property
    def service(self) -> str:
        return str(self.job["service"])


@dataclass(frozen=True)
class AegisVoxQueue:
    """Small queue facade backed by a metadata store implementation."""

    metadata_store: object

    def list_jobs(
        self,
        *,
        service: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.metadata_store.list_queue_jobs(service=service, status=status, limit=limit)

    def claim(
        self,
        service: str,
        *,
        worker_id: str | None = None,
        timeout_seconds: int | None = None,
    ) -> QueueClaim | None:
        job = self.metadata_store.claim_next_job(
            service,
            worker_id=worker_id,
            timeout_seconds=timeout_seconds,
        )
        return QueueClaim(job=dict(job)) if job else None

    def heartbeat(self, job_id: str, *, worker_id: str | None = None) -> dict[str, Any] | None:
        job = self.metadata_store.heartbeat_job(job_id, worker_id=worker_id)
        return dict(job) if job else None

    def complete(self, job_id: str, *, output_artifacts: Iterable[str] = ()) -> dict[str, Any] | None:
        self.metadata_store.update_job_status(job_id, "completed", output_artifacts=output_artifacts)
        return self.get_job(job_id)

    def fail(self, job_id: str, *, error: dict[str, Any]) -> dict[str, Any] | None:
        self.metadata_store.update_job_status(job_id, "failed", error=error)
        return self.get_job(job_id)

    def update_status(
        self,
        job_id: str,
        status: str,
        *,
        output_artifacts: Iterable[str] = (),
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        self.metadata_store.update_job_status(
            job_id,
            status,
            output_artifacts=output_artifacts,
            error=error,
        )
        return self.get_job(job_id)

    def requeue_stale(self, *, stale_after_seconds: int, service: str | None = None) -> list[dict[str, Any]]:
        return [dict(job) for job in self.metadata_store.requeue_stale_jobs(
            stale_after_seconds=stale_after_seconds,
            service=service,
        )]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        for job in self.metadata_store.list_queue_jobs(limit=1000):
            if job["job_id"] == job_id:
                return dict(job)
        return None
