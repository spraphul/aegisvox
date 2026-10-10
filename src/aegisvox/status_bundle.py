"""Compact redacted status bundles for AegisVox operations.

A status bundle is a handoff artifact: it combines readiness, runtime preflight,
queue/job summaries, and expected provenance keys without exposing secrets or
large raw artifacts. It is intended for ORD tmux handoffs and local debugging.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .audit import sanitize_for_audit
from .preflight import expected_version_registry, run_preflight_checks
from .readiness import readiness_snapshot
from .schema import utc_now_iso
from .storage import LocalArtifactStore


def _queue_summary(metadata_store: object, *, recent_limit: int = 25) -> dict[str, Any]:
    jobs = metadata_store.list_queue_jobs(limit=1000)
    by_status = Counter(str(job.get("status")) for job in jobs)
    by_service = Counter(str(job.get("service")) for job in jobs)
    by_service_status: dict[str, dict[str, int]] = defaultdict(dict)
    for job in jobs:
        service = str(job.get("service"))
        status = str(job.get("status"))
        by_service_status[service][status] = int(by_service_status[service].get(status, 0)) + 1
    recent_jobs = jobs[-max(0, min(int(recent_limit), 100)):]
    return sanitize_for_audit(
        {
            "total_jobs_seen": len(jobs),
            "by_status": dict(sorted(by_status.items())),
            "by_service": dict(sorted(by_service.items())),
            "by_service_status": {service: dict(sorted(counts.items())) for service, counts in sorted(by_service_status.items())},
            "recent_jobs": recent_jobs,
            "recent_limit": recent_limit,
            "truncated": len(jobs) >= 1000,
        }
    )


def _run_summary(metadata_store: object) -> dict[str, Any]:
    connect = getattr(metadata_store, "connect", None)
    if connect is None:
        return {"available": False, "reason": "metadata backend does not expose direct run summary query"}
    try:
        with connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS count FROM runs GROUP BY status ORDER BY status").fetchall()
            total = conn.execute("SELECT COUNT(*) AS count FROM runs").fetchone()[0]
        by_status = {str(row[0]): int(row[1]) for row in rows}
        return {"available": True, "total_runs": int(total), "by_status": by_status}
    except Exception as exc:  # noqa: BLE001 - status bundle should report, not crash, on optional summary failure
        return sanitize_for_audit({"available": False, "reason": str(exc)})


def build_status_bundle(
    *,
    artifact_store: LocalArtifactStore,
    metadata_store: object,
    metadata_backend: str = "sqlite",
    voice_pool_retrieval: object | None = None,
    preflight_services: list[str] | None = None,
    include_preflight: bool = True,
    preflight_deep: bool = False,
    recent_job_limit: int = 25,
) -> dict[str, Any]:
    """Build a redacted status bundle dictionary."""

    readiness = readiness_snapshot(
        artifact_store=artifact_store,
        metadata_store=metadata_store,
        metadata_backend=metadata_backend,
        voice_pool_retrieval=voice_pool_retrieval,
    )
    preflight = (
        run_preflight_checks(
            services=preflight_services,
            artifact_root=artifact_store.root,
            deep=preflight_deep,
        )
        if include_preflight
        else {"ok": None, "status": "skipped", "reason": "include_preflight=false"}
    )
    readiness_failed = readiness.get("status") == "fail"
    preflight_failed = preflight.get("status") == "fail"
    bundle = {
        "bundle_type": "aegisvox_status_bundle",
        "generated_at": utc_now_iso(),
        "overall_status": "fail" if readiness_failed or preflight_failed else ("warn" if readiness.get("status") == "warn" else "pass"),
        "artifact_root": str(Path(artifact_store.root)),
        "metadata_backend": metadata_backend,
        "readiness": readiness,
        "preflight": preflight,
        "runs": _run_summary(metadata_store),
        "queue": _queue_summary(metadata_store, recent_limit=recent_job_limit),
        "version_registry": expected_version_registry(),
        "claim_boundary": (
            "Status bundles are operational readiness/handoff artifacts only; they do not prove privacy, "
            "utility, PHI recall, speaker unlinkability, or end-to-end clinical safety."
        ),
    }
    return sanitize_for_audit(bundle)


def status_bundle_markdown(bundle: dict[str, Any]) -> str:
    """Render a compact Markdown status bundle."""

    lines = [
        "# AegisVox status bundle",
        "",
        f"- Generated at: `{bundle.get('generated_at')}`",
        f"- Overall status: **{bundle.get('overall_status')}**",
        f"- Artifact root: `{bundle.get('artifact_root')}`",
        f"- Metadata backend: `{bundle.get('metadata_backend')}`",
        "",
        "## Readiness",
        "",
        f"- Status: `{(bundle.get('readiness') or {}).get('status')}`",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in (bundle.get("readiness") or {}).get("checks") or []:
        detail = check.get("detail") or check.get("metadata") or ""
        detail_text = json.dumps(detail, sort_keys=True) if isinstance(detail, (dict, list)) else str(detail)
        detail_text = detail_text.replace('|', '\\|')[:300]
        lines.append(f"| `{check.get('name')}` | `{check.get('status')}` | {detail_text} |")
    preflight = bundle.get("preflight") or {}
    lines.extend(
        [
            "",
            "## Preflight",
            "",
            f"- Status: `{preflight.get('status')}`",
            f"- Failed required: `{(preflight.get('summary') or {}).get('failed_required', [])}`",
            "",
            "## Queue",
            "",
        ]
    )
    queue = bundle.get("queue") or {}
    lines.append(f"- Total jobs seen: {queue.get('total_jobs_seen')}")
    lines.append(f"- By status: `{json.dumps(queue.get('by_status') or {}, sort_keys=True)}`")
    lines.extend(["", "## Runs", ""])
    runs = bundle.get("runs") or {}
    lines.append(f"- Total runs: {runs.get('total_runs') if runs.get('available') else 'unavailable'}")
    lines.append(f"- By status: `{json.dumps(runs.get('by_status') or {}, sort_keys=True)}`")
    lines.extend(["", "## Claim boundary", "", str(bundle.get("claim_boundary"))])
    return "\n".join(lines) + "\n"
