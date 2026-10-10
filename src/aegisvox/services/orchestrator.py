"""Minimal orchestrator service logic.

The orchestrator creates and updates run state only. It must not perform audio
inference; downstream services consume emitted events and artifact references.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from aegisvox import __version__
from aegisvox.audit import sanitize_for_audit
from aegisvox.events import InMemoryEventBus
from aegisvox.manifest import manifest_uri
from aegisvox.schema import (
    AegisVoxContractError,
    ArtifactRecord,
    EventEnvelope,
    RunManifest,
    RunRequest,
    RunStatus,
    new_run_id,
)
from aegisvox.storage import LocalArtifactStore
from aegisvox.workflow import input_artifact_uris_for_event, plan_transition, request_event_type_for_service


@dataclass
class Orchestrator:
    artifact_store: LocalArtifactStore
    event_bus: InMemoryEventBus
    bucket: str = "aegisvox"
    metadata_store: object | None = None

    def _manifest_uri(self, run_id: str) -> str:
        return manifest_uri(run_id, bucket=self.bucket)

    def _load_manifest(self, run_id: str) -> RunManifest:
        uri = self._manifest_uri(run_id)
        if not self.artifact_store.exists(uri):
            raise AegisVoxContractError(f"run manifest not found for {run_id}")
        return RunManifest.from_dict(self.artifact_store.read_json(uri))

    def _persist_manifest(self, manifest: RunManifest) -> None:
        uri = self._manifest_uri(manifest.run_id)
        self.artifact_store.write_json(uri, manifest.to_dict())
        if self.metadata_store is not None:
            self.metadata_store.upsert_run(manifest, manifest_uri=uri)

    def _create_job_for_request(self, event: EventEnvelope, target_service: str | None) -> None:
        if self.metadata_store is None or target_service is None:
            return
        self.metadata_store.create_job(
            job_id=f"job_{event.event_id}",
            run_id=event.run_id,
            service=target_service,
            input_artifacts=event.payload.get("input_artifacts", []),
        )

    def _artifact_records_from_completion_payload(self, payload: dict) -> list[ArtifactRecord]:
        raw_artifacts = payload.get("artifacts") or []
        if isinstance(raw_artifacts, dict):
            raw_artifacts = [
                {"name": name, **value} if isinstance(value, dict) else {"name": name, "uri": value}
                for name, value in raw_artifacts.items()
            ]
        if not isinstance(raw_artifacts, list):
            raise AegisVoxContractError("completion payload artifacts must be a list or object")

        records: list[ArtifactRecord] = []
        for row in raw_artifacts:
            if not isinstance(row, dict):
                raise AegisVoxContractError("each completion artifact must be an object")
            name = str(row.get("name") or "")
            uri = str(row.get("uri") or "")
            artifact_type = str(row.get("artifact_type") or row.get("type") or name)
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise AegisVoxContractError("completion artifact metadata must be an object")
            records.append(ArtifactRecord(name=name, uri=uri, artifact_type=artifact_type, metadata=metadata))
        return records

    def _service_versions_from_completion_payload(self, payload: dict) -> dict[str, str]:
        raw_versions: dict[str, object] = {}
        for key in ("service_versions", "model_versions"):
            value = payload.get(key) or {}
            if not isinstance(value, dict):
                raise AegisVoxContractError(f"completion payload {key} must be an object")
            raw_versions.update(value)
        versions: dict[str, str] = {}
        for name, version in raw_versions.items():
            if not name:
                raise AegisVoxContractError("service/model version key cannot be empty")
            if version is None or isinstance(version, (dict, list)):
                raise AegisVoxContractError(f"service/model version for {name!r} must be a scalar value")
            versions[str(name)] = str(version)
        return versions

    def _append_transition_request(
        self,
        manifest: RunManifest,
        from_event_type: str,
        *,
        create_job: bool = True,
    ) -> EventEnvelope:
        transition = plan_transition(from_event_type, manifest.request)
        input_artifacts = input_artifact_uris_for_event(transition.next_event_type, manifest)
        event = EventEnvelope(
            event_type=transition.next_event_type,
            run_id=manifest.run_id,
            service="orchestrator",
            payload={
                "from_event_type": from_event_type,
                "target_service": transition.target_service,
                "manifest_uri": self._manifest_uri(manifest.run_id),
                "input_artifacts": input_artifacts,
            },
        )
        if event.event_type == "run.completed":
            manifest.set_status(RunStatus.COMPLETED)
        manifest.add_event(event)
        self.event_bus.emit(event)
        if create_job:
            self._create_job_for_request(event, transition.target_service)
        return event

    def create_run(self, request: RunRequest, run_id: str | None = None) -> RunManifest:
        run_id = run_id or new_run_id()
        manifest = RunManifest(run_id=run_id, request=request, service_versions={"orchestrator": __version__})
        event = EventEnvelope(
            event_type="run.created",
            run_id=run_id,
            service="orchestrator",
            payload={"manifest_uri": manifest_uri(run_id, bucket=self.bucket)},
        )
        manifest.add_event(event)
        self.event_bus.emit(event)
        request_event = self._append_transition_request(manifest, "run.created", create_job=False)
        self._persist_manifest(manifest)
        self._create_job_for_request(request_event, request_event.payload.get("target_service"))
        return manifest

    def cancel_run(self, run_id: str) -> RunManifest:
        uri = self._manifest_uri(run_id)
        manifest = self._load_manifest(run_id)
        if manifest.status == RunStatus.COMPLETED:
            raise AegisVoxContractError(f"cannot cancel completed run {run_id}")
        if manifest.status != RunStatus.CANCELED:
            manifest.set_status(RunStatus.CANCELED, failure_reason="cancelled by request")
            event = EventEnvelope(
                event_type="run.canceled",
                run_id=run_id,
                service="orchestrator",
                payload={"manifest_uri": uri, "reason": "cancelled by request"},
            )
            manifest.add_event(event)
            self.event_bus.emit(event)
            self._persist_manifest(manifest)
        return manifest

    def record_service_failure(
        self,
        *,
        run_id: str,
        service: str,
        error: dict | str,
        job_id: str | None = None,
    ) -> RunManifest:
        manifest = self._load_manifest(run_id)
        if manifest.status == RunStatus.COMPLETED:
            raise AegisVoxContractError(f"cannot fail completed run {run_id}")
        if manifest.status == RunStatus.CANCELED:
            raise AegisVoxContractError(f"cannot fail canceled run {run_id}")
        error_payload = sanitize_for_audit(error if isinstance(error, dict) else {"message": str(error)})
        failure_reason = str(
            error_payload.get("reason")
            or error_payload.get("message")
            or error_payload.get("error")
            or f"{service} failed"
        )
        if manifest.status != RunStatus.FAILED:
            manifest.set_status(RunStatus.FAILED, failure_reason=failure_reason)
            event = EventEnvelope(
                event_type="run.failed",
                run_id=run_id,
                service=service,
                payload={
                    "manifest_uri": self._manifest_uri(run_id),
                    "job_id": job_id,
                    "error": error_payload,
                },
            )
            manifest.add_event(event)
            self.event_bus.emit(event)
            if self.metadata_store is not None:
                if job_id:
                    self.metadata_store.update_job_status(job_id, "failed", error=error_payload)
                else:
                    self.metadata_store.update_latest_job_for_service(run_id, service, "failed", error=error_payload)
            self._persist_manifest(manifest)
        return manifest

    def _find_failed_job_for_retry(
        self,
        *,
        run_id: str,
        service: str | None = None,
        job_id: str | None = None,
    ) -> dict | None:
        if self.metadata_store is None:
            return None
        jobs = self.metadata_store.list_jobs(run_id)
        if job_id:
            for job in jobs:
                if job["job_id"] == job_id:
                    if service and job["service"] != service:
                        raise AegisVoxContractError(
                            f"job {job_id!r} belongs to service {job['service']!r}, not {service!r}"
                        )
                    if job["status"] != "failed":
                        raise AegisVoxContractError(f"cannot retry non-failed job {job_id!r}")
                    return job
            raise AegisVoxContractError(f"job {job_id!r} not found for run {run_id}")
        for job in reversed(jobs):
            if job["status"] == "failed" and (service is None or job["service"] == service):
                return job
        return None

    def retry_failed_service(
        self,
        *,
        run_id: str,
        service: str | None = None,
        job_id: str | None = None,
        reason: str | None = None,
    ) -> RunManifest:
        """Re-open a failed run and enqueue an explicit retry job.

        Runtime failures remain loud and terminal by default. This method is an
        auditable operator action: it records a new service `*.requested` event
        with retry metadata, clears the run-level failure reason, and creates a
        fresh queue job using the failed job's original input artifact URIs.
        """

        manifest = self._load_manifest(run_id)
        if manifest.status == RunStatus.COMPLETED:
            raise AegisVoxContractError(f"cannot retry completed run {run_id}")
        if manifest.status == RunStatus.CANCELED:
            raise AegisVoxContractError(f"cannot retry canceled run {run_id}")
        if manifest.status != RunStatus.FAILED:
            raise AegisVoxContractError(f"can only retry failed runs, got status {manifest.status.value!r}")
        if self.metadata_store is None:
            raise AegisVoxContractError("cannot retry without a metadata/job store")

        failed_job = self._find_failed_job_for_retry(run_id=run_id, service=service, job_id=job_id)
        retry_service = service or (str(failed_job["service"]) if failed_job else None)
        if not retry_service:
            raise AegisVoxContractError("retry requires a service or failed job")
        request_event_type = request_event_type_for_service(retry_service)

        if failed_job is None:
            input_artifacts = input_artifact_uris_for_event(request_event_type, manifest)
        else:
            try:
                input_artifacts = json.loads(failed_job.get("input_artifacts_json") or "[]")
            except json.JSONDecodeError as exc:
                raise AegisVoxContractError(
                    f"failed job {failed_job['job_id']!r} has invalid input_artifacts_json"
                ) from exc
            if not isinstance(input_artifacts, list):
                raise AegisVoxContractError(
                    f"failed job {failed_job['job_id']!r} input_artifacts_json must decode to a list"
                )

        manifest.set_status(RunStatus.RUNNING)
        event = EventEnvelope(
            event_type=request_event_type,
            run_id=run_id,
            service="orchestrator",
            payload={
                "retry": True,
                "retry_of_job_id": failed_job["job_id"] if failed_job else job_id,
                "retry_reason": reason or "operator_requested_retry",
                "target_service": retry_service,
                "manifest_uri": self._manifest_uri(run_id),
                "input_artifacts": input_artifacts,
            },
        )
        manifest.add_event(event)
        self.event_bus.emit(event)
        self._persist_manifest(manifest)
        self._create_job_for_request(event, retry_service)
        return manifest

    def record_service_completion(
        self,
        *,
        run_id: str,
        event_type: str,
        service: str,
        payload: dict | None = None,
    ) -> RunManifest:
        if not event_type.endswith(".completed"):
            raise AegisVoxContractError(f"completion event_type must end with .completed, got {event_type!r}")
        manifest = self._load_manifest(run_id)
        if manifest.status in {RunStatus.CANCELED, RunStatus.COMPLETED, RunStatus.FAILED}:
            raise AegisVoxContractError(f"cannot record service completion for terminal run {run_id}")
        completed = EventEnvelope(
            event_type=event_type,
            run_id=run_id,
            service=service,
            payload=dict(payload or {}),
        )
        artifact_records = self._artifact_records_from_completion_payload(completed.payload)
        service_versions = self._service_versions_from_completion_payload(completed.payload)
        for artifact in artifact_records:
            manifest.add_artifact(artifact)
        for name, version in service_versions.items():
            manifest.service_versions[name] = version
        manifest.add_event(completed)
        self.event_bus.emit(completed)
        if self.metadata_store is not None:
            self.metadata_store.update_latest_job_for_service(
                run_id,
                service,
                "completed",
                output_artifacts=[artifact.uri for artifact in artifact_records],
            )
        self._append_transition_request(manifest, event_type)
        self._persist_manifest(manifest)
        return manifest
