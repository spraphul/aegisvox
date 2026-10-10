"""Generic one-shot worker runner for local/service development.

The worker is intentionally command-backed. It never fabricates model outputs:
each service must configure a real command for its job. Commands can either
return a completion event payload for the orchestrator, or only update the job
status when they are pure control-plane/smoke commands.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .audit import sanitize_for_audit
from .events import InMemoryEventBus
from .logging_utils import write_audit_log_event
from .queue import AegisVoxQueue
from .services.orchestrator import Orchestrator
from .storage import LocalArtifactStore


class WorkerSetupError(RuntimeError):
    """Raised when a worker backend command is not configured."""


class WorkerExecutionError(RuntimeError):
    """Raised when a worker command returns invalid output."""


def service_command_env_name(service: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in service.upper())
    return f"AEGISVOX_WORKER_COMMAND_{normalized}"


def command_for_service(service: str, explicit_command: str | None = None) -> str:
    if explicit_command:
        return explicit_command
    service_specific = os.environ.get(service_command_env_name(service))
    generic = os.environ.get("AEGISVOX_WORKER_COMMAND")
    command = service_specific or generic
    if not command:
        raise WorkerSetupError(
            f"no worker command configured for service {service!r}; set {service_command_env_name(service)} "
            "or AEGISVOX_WORKER_COMMAND"
        )
    return command


def parse_worker_stdout(stdout: str) -> dict[str, Any]:
    """Parse the last non-empty stdout line as JSON, or return empty payload."""

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        parsed = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise WorkerExecutionError(f"worker command last stdout line was not JSON: {lines[-1]!r}") from exc
    if not isinstance(parsed, dict):
        raise WorkerExecutionError("worker command JSON result must be an object")
    return parsed


@dataclass
class CommandWorker:
    metadata_store: object
    artifact_store: LocalArtifactStore
    bucket: str = "aegisvox"
    command: str | None = None
    heartbeat_interval_seconds: float | None = None
    job_timeout_seconds: int | None = None
    audit_log_stream: TextIO | None = None

    def _log(self, *, event_type: str, service: str, level: str = "info", job: dict[str, Any] | None = None, **fields: Any) -> None:
        if self.audit_log_stream is None:
            return
        write_audit_log_event(
            self.audit_log_stream,
            event_type=event_type,
            service=service,
            level=level,
            run_id=str(job["run_id"]) if job is not None and job.get("run_id") is not None else None,
            job_id=str(job["job_id"]) if job is not None and job.get("job_id") is not None else None,
            **fields,
        )

    def _queue(self) -> AegisVoxQueue:
        return AegisVoxQueue(self.metadata_store)

    def _orchestrator(self) -> Orchestrator:
        return Orchestrator(
            self.artifact_store,
            InMemoryEventBus(),
            bucket=self.bucket,
            metadata_store=self.metadata_store,
        )

    def _fail_run_for_job(self, job: dict[str, Any], service: str, error: dict[str, Any]) -> None:
        self._queue().fail(str(job["job_id"]), error=error)
        self._orchestrator().record_service_failure(
            run_id=str(job["run_id"]),
            service=service,
            job_id=str(job["job_id"]),
            error=error,
        )

    def run_once(self, *, service: str, worker_id: str | None = None) -> dict[str, Any]:
        command = command_for_service(service, self.command)
        queue = self._queue()
        claim = queue.claim(service, worker_id=worker_id, timeout_seconds=self.job_timeout_seconds)
        if claim is None:
            self._log(event_type="worker.no_job", service=service, worker_id=worker_id)
            return {"claimed": False, "service": service}
        job = claim.job
        self._log(
            event_type="worker.job_claimed",
            service=service,
            job=job,
            worker_id=worker_id,
            attempt=job.get("attempt"),
            timeout_seconds=job.get("timeout_seconds"),
        )

        with tempfile.TemporaryDirectory(prefix="aegisvox_worker_") as tmpdir:
            job_json_path = Path(tmpdir) / "job.json"
            job_json_path.write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
            env = dict(os.environ)
            env.update(
                {
                    "AEGISVOX_JOB_JSON": str(job_json_path),
                    "AEGISVOX_JOB_ID": str(job["job_id"]),
                    "AEGISVOX_RUN_ID": str(job["run_id"]),
                    "AEGISVOX_SERVICE": service,
                    "AEGISVOX_ARTIFACT_ROOT": str(self.artifact_store.root),
                    "AEGISVOX_METADATA_DB": str(getattr(self.metadata_store, "db_path", "")),
                    "AEGISVOX_JOB_ATTEMPT": str(job.get("attempt") or 0),
                    "AEGISVOX_JOB_WORKER_ID": str(job.get("worker_id") or ""),
                    "AEGISVOX_JOB_CLAIMED_AT": str(job.get("claimed_at") or ""),
                }
            )
            process = subprocess.Popen(command, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stop_heartbeat = threading.Event()
            heartbeat_error: list[str] = []

            def heartbeat_loop() -> None:
                interval = float(self.heartbeat_interval_seconds or 0)
                if interval <= 0:
                    return
                while not stop_heartbeat.wait(interval):
                    try:
                        updated = queue.heartbeat(str(job["job_id"]), worker_id=worker_id)
                        if updated is None:
                            heartbeat_error.append("heartbeat target job not found or not running")
                            return
                    except Exception as exc:  # noqa: BLE001 - preserve command result and report heartbeat failure separately
                        heartbeat_error.append(str(sanitize_for_audit(str(exc))))
                        return

            heartbeat_thread = threading.Thread(target=heartbeat_loop, name=f"aegisvox-heartbeat-{job['job_id']}", daemon=True)
            heartbeat_thread.start()
            stdout, stderr = process.communicate()
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1.0)

        result = subprocess.CompletedProcess(command, process.returncode, stdout=stdout or "", stderr=stderr or "")
        if heartbeat_error and result.returncode == 0:
            error = sanitize_for_audit({"reason": "worker_heartbeat_failed", "message": heartbeat_error[-1]})
            self._fail_run_for_job(job, service, error)
            self._log(event_type="worker.heartbeat_failed", service=service, level="error", job=job, error=error)
            return {"claimed": True, "job": job, "status": "failed", "error": error}

        if result.returncode != 0:
            error = sanitize_for_audit(
                {
                    "reason": "worker_command_failed",
                    "returncode": result.returncode,
                    "stderr_tail": result.stderr[-4000:],
                    "stdout_tail": result.stdout[-4000:],
                }
            )
            self._fail_run_for_job(job, service, error)
            self._log(event_type="worker.command_failed", service=service, level="error", job=job, error=error)
            return {"claimed": True, "job": job, "status": "failed", "error": error}

        try:
            payload = parse_worker_stdout(result.stdout)
        except WorkerExecutionError as exc:
            error = sanitize_for_audit({"reason": "invalid_worker_output", "message": str(exc)})
            self._fail_run_for_job(job, service, error)
            self._log(event_type="worker.invalid_output", service=service, level="error", job=job, error=error)
            return {"claimed": True, "job": job, "status": "failed", "error": error}

        completion_event_type = payload.get("completion_event_type")
        if completion_event_type:
            manifest = self._orchestrator().record_service_completion(
                run_id=str(job["run_id"]),
                event_type=str(completion_event_type),
                service=service,
                payload=dict(payload.get("completion_payload") or {}),
            )
            self._log(
                event_type="worker.completion_processed",
                service=service,
                job=job,
                completion_event_type=completion_event_type,
                run_status=manifest.status.value,
            )
            return {
                "claimed": True,
                "job": job,
                "status": "completed",
                "completion_event_type": completion_event_type,
                "run_status": manifest.status.value,
            }

        output_artifacts = payload.get("output_artifacts") or []
        if not isinstance(output_artifacts, list):
            error = sanitize_for_audit({"reason": "invalid_worker_output", "message": "output_artifacts must be a list"})
            self._fail_run_for_job(job, service, error)
            self._log(event_type="worker.invalid_output", service=service, level="error", job=job, error=error)
            return {"claimed": True, "job": job, "status": "failed", "error": error}
        status = str(payload.get("status") or "completed")
        self._queue().update_status(str(job["job_id"]), status, output_artifacts=output_artifacts)
        self._log(
            event_type="worker.status_updated",
            service=service,
            job=job,
            status=status,
            output_artifact_count=len(output_artifacts),
        )
        return {"claimed": True, "job": job, "status": status, "output_artifacts": output_artifacts}
