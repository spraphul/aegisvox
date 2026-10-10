"""ORD/tmux worker launch-plan generation.

Launch plans are safe handoff artifacts: they contain commands and required env
variable *names*, never secret values. Operators can review the JSON/Markdown or
shell script before launching service workers.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit import sanitize_for_audit
from .preflight import PREFLIGHT_DEFINITIONS
from .schema import utc_now_iso
from .service_job import SERVICE_COMPLETION_EVENTS

DEFAULT_REMOTE_ROOT = "./artifacts"
DEFAULT_VENV_PYTHON = f"{DEFAULT_REMOTE_ROOT}/venvs/aegisvox-py311/bin/python"
DEFAULT_ARTIFACT_ROOT = f"{DEFAULT_REMOTE_ROOT}/artifacts"
DEFAULT_METADATA_DB = f"{DEFAULT_ARTIFACT_ROOT}/metadata.sqlite3"
DEFAULT_LOG_ROOT = f"{DEFAULT_REMOTE_ROOT}/logs"
DEFAULT_WORKER_COMMAND = "python scripts/run_service_job_command.py"

GPU_SERVICES = {"diarization", "speaker_profiling", "speaker_privacy_metrics", "voice_conversion", "speech_editing"}
API_SERVICES = {"asr_alignment", "background_leakage_check"}
CPU_SERVICES = {"source_separation", "segmentation", "assignment", "stitching", "text_deid", "final_background_remix", "overlap_remix", "evaluation"}

SERVICE_ORDER = [
    "source_separation",
    "background_leakage_check",
    "diarization",
    "segmentation",
    "speaker_profiling",
    "assignment",
    "voice_conversion",
    "stitching",
    "asr_alignment",
    "text_deid",
    "speech_editing",
    "final_background_remix",
    "speaker_privacy_metrics",
    "overlap_remix",
    "evaluation",
]

DEFAULT_TIMEOUTS = {
    "source_separation": 3600,
    "background_leakage_check": 3600,
    "diarization": 3600,
    "segmentation": 1800,
    "speaker_profiling": 3600,
    "assignment": 900,
    "voice_conversion": 7200,
    "stitching": 1800,
    "asr_alignment": 3600,
    "text_deid": 1800,
    "speech_editing": 7200,
    "final_background_remix": 1800,
    "speaker_privacy_metrics": 3600,
    "overlap_remix": 7200,
    "evaluation": 1800,
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


@dataclass(frozen=True)
class WorkerLaunchSpec:
    service: str
    workers: int = 1
    resource_profile: str = "cpu"
    required_env_any: tuple[str, ...] = ()
    required_env_all: tuple[str, ...] = ()
    optional_env: tuple[str, ...] = ()
    job_timeout_seconds: int = 1800
    heartbeat_interval_seconds: int = 30
    command: str = DEFAULT_WORKER_COMMAND
    enabled: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "workers": self.workers,
            "resource_profile": self.resource_profile,
            "required_env_any": list(self.required_env_any),
            "required_env_all": list(self.required_env_all),
            "optional_env": list(self.optional_env),
            "job_timeout_seconds": self.job_timeout_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "command": self.command,
            "enabled": self.enabled,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class LaunchPlanConfig:
    remote_root: str = DEFAULT_REMOTE_ROOT
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    metadata_db: str = DEFAULT_METADATA_DB
    log_root: str = DEFAULT_LOG_ROOT
    python_bin: str = DEFAULT_VENV_PYTHON
    metadata_backend: str = "sqlite"
    postgres_dsn_env: str = "AEGISVOX_POSTGRES_DSN"
    heartbeat_interval_seconds: int = 30
    poll_seconds: float = 5.0
    worker_command: str = DEFAULT_WORKER_COMMAND
    include_disabled: bool = False
    max_jobs: int | None = None
    exit_when_idle: bool = False


def default_launch_specs(*, services: list[str] | None = None, workers_per_service: dict[str, int] | None = None) -> list[WorkerLaunchSpec]:
    selected = services or SERVICE_ORDER
    unknown = [service for service in selected if service not in SERVICE_COMPLETION_EVENTS]
    if unknown:
        raise ValueError("unknown worker service(s): " + ", ".join(unknown))
    workers_per_service = workers_per_service or {}
    specs: list[WorkerLaunchSpec] = []
    for service in selected:
        definition = PREFLIGHT_DEFINITIONS.get(service)
        if service in GPU_SERVICES:
            resource = "gpu"
        elif service in API_SERVICES:
            resource = "api"
        else:
            resource = "cpu"
        specs.append(
            WorkerLaunchSpec(
                service=service,
                workers=max(1, int(workers_per_service.get(service, 1))),
                resource_profile=resource,
                required_env_any=tuple(definition.required_env_any if definition else ()),
                required_env_all=tuple(definition.required_env_all if definition else ()),
                optional_env=tuple(definition.optional_env if definition else ()),
                job_timeout_seconds=DEFAULT_TIMEOUTS.get(service, 1800),
                heartbeat_interval_seconds=30,
                notes=("model/API-backed; configure runtime-only dependencies first" if resource in {"gpu", "api"} else "deterministic/control-plane service"),
            )
        )
    return specs


def _env_prefix(config: LaunchPlanConfig, spec: WorkerLaunchSpec) -> str:
    exports = [
        f"AEGISVOX_WORKER_COMMAND={shlex.quote(spec.command)}",
        f"AEGISVOX_ARTIFACT_ROOT={shlex.quote(config.artifact_root)}",
    ]
    if config.metadata_backend == "postgres":
        exports.extend(
            [
                "AEGISVOX_METADATA_BACKEND=postgres",
                f"AEGISVOX_POSTGRES_DSN=${{{config.postgres_dsn_env}}}",
            ]
        )
    return " ".join(exports)


def worker_command(config: LaunchPlanConfig, spec: WorkerLaunchSpec, worker_index: int) -> str:
    service_slug = _slug(spec.service)
    worker_id = f"{service_slug}_worker_{worker_index}"
    parts = [
        shlex.quote(config.python_bin),
        "scripts/run_worker_loop.py",
        "--service",
        shlex.quote(spec.service),
        "--artifact-root",
        shlex.quote(config.artifact_root),
        "--metadata-backend",
        shlex.quote(config.metadata_backend),
        "--worker-id",
        shlex.quote(worker_id),
        "--heartbeat-interval-seconds",
        str(spec.heartbeat_interval_seconds or config.heartbeat_interval_seconds),
        "--job-timeout-seconds",
        str(spec.job_timeout_seconds),
        "--poll-seconds",
        str(config.poll_seconds),
        "--audit-log-jsonl",
        shlex.quote(f"{config.log_root}/{service_slug}_worker_{worker_index}.jsonl"),
    ]
    if config.metadata_backend == "sqlite":
        parts.extend(["--metadata-db", shlex.quote(config.metadata_db)])
    if config.metadata_backend == "postgres":
        parts.extend(["--postgres-dsn", f"\"${{{config.postgres_dsn_env}}}\""])
    if config.max_jobs is not None:
        parts.extend(["--max-jobs", str(config.max_jobs)])
    if config.exit_when_idle:
        parts.append("--exit-when-idle")
    command = " ".join(parts)
    return f"{_env_prefix(config, spec)} {command}"


def tmux_command(config: LaunchPlanConfig, spec: WorkerLaunchSpec, worker_index: int) -> str:
    service_slug = _slug(spec.service)
    session = f"aegisvox_{service_slug}_w{worker_index}"
    stdout_log = f"{config.log_root}/{service_slug}_worker_{worker_index}.stdout.log"
    inner = f"cd {shlex.quote(config.remote_root)} && {worker_command(config, spec, worker_index)} 2>&1 | tee -a {shlex.quote(stdout_log)}"
    return f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(inner)}"


def build_launch_plan(
    *,
    config: LaunchPlanConfig | None = None,
    services: list[str] | None = None,
    workers_per_service: dict[str, int] | None = None,
) -> dict[str, Any]:
    config = config or LaunchPlanConfig()
    specs = default_launch_specs(services=services, workers_per_service=workers_per_service)
    entries = []
    for spec in specs:
        if not spec.enabled and not config.include_disabled:
            continue
        commands = [tmux_command(config, spec, idx) for idx in range(spec.workers)]
        entries.append({**spec.to_dict(), "tmux_commands": commands})
    return sanitize_for_audit(
        {
            "plan_type": "aegisvox_worker_launch_plan",
            "generated_at": utc_now_iso(),
            "remote_root": config.remote_root,
            "artifact_root": config.artifact_root,
            "metadata_backend": config.metadata_backend,
            "metadata_db": config.metadata_db if config.metadata_backend == "sqlite" else None,
            "log_root": config.log_root,
            "python_bin": config.python_bin,
            "services": entries,
            "secret_policy": "contains env var names and shell commands only; do not write token values into this plan",
            "claim_boundary": "launch plans are operational commands only; they do not prove backend readiness or privacy/utility evidence",
        }
    )


def launch_plan_shell(plan: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Auto-generated AegisVox worker launch plan.",
        "# Review env requirements before running. Do not paste secrets into this file.",
        "mkdir -p " + shlex.quote(str(plan.get("log_root") or DEFAULT_LOG_ROOT)),
        "",
    ]
    for service in plan.get("services") or []:
        lines.append(f"# Service: {service.get('service')} ({service.get('resource_profile')})")
        req_any = service.get("required_env_any") or []
        req_all = service.get("required_env_all") or []
        opt = service.get("optional_env") or []
        if req_any:
            lines.append("# Requires one of: " + ", ".join(req_any))
        if req_all:
            lines.append("# Requires all: " + ", ".join(req_all))
        if opt:
            lines.append("# Optional env: " + ", ".join(opt))
        for command in service.get("tmux_commands") or []:
            lines.append(command)
        lines.append("")
    return "\n".join(lines) + "\n"


def launch_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# AegisVox worker launch plan",
        "",
        f"- Generated at: `{plan.get('generated_at')}`",
        f"- Remote root: `{plan.get('remote_root')}`",
        f"- Artifact root: `{plan.get('artifact_root')}`",
        f"- Metadata backend: `{plan.get('metadata_backend')}`",
        "",
        "## Services",
        "",
        "| Service | Workers | Resource | Required env |",
        "|---|---:|---|---|",
    ]
    for service in plan.get("services") or []:
        reqs = []
        if service.get("required_env_any"):
            reqs.append("one of: " + ", ".join(service["required_env_any"]))
        if service.get("required_env_all"):
            reqs.append("all: " + ", ".join(service["required_env_all"]))
        lines.append(
            f"| `{service.get('service')}` | {service.get('workers')} | `{service.get('resource_profile')}` | {'; '.join(reqs) or 'none'} |"
        )
    lines.extend(["", "## Claim boundary", "", str(plan.get("claim_boundary"))])
    return "\n".join(lines) + "\n"
