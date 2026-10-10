"""Runtime evidence validation for AegisVox milestones.

The plan-compliance report answers "what should exist?". This module answers
"does the runtime artifact look structurally valid?" without logging raw audio,
transcripts, embeddings, or PHI-like payloads.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .audit import sanitize_for_audit
from .plan_compliance import MILESTONES, EvidenceSpec
from .schema import utc_now_iso

ValidationStatus = Literal["pass", "fail", "missing", "skipped"]


@dataclass(frozen=True)
class RuntimeEvidencePolicy:
    require_nonempty_json: bool = True
    require_nonempty_wav: bool = True
    max_preview_keys: int = 12


def _safe_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _validate_json(path: Path, *, policy: RuntimeEvidencePolicy) -> dict[str, Any]:
    size = _safe_size(path)
    if policy.require_nonempty_json and (size is None or size <= 0):
        return {"status": "fail", "reason": "empty_json_file", "size_bytes": size}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - report should aggregate structural failures
        return {"status": "fail", "reason": "invalid_json", "error": str(exc), "size_bytes": size}
    detail: dict[str, Any] = {"status": "pass", "size_bytes": size, "json_type": type(payload).__name__}
    if isinstance(payload, dict):
        detail["top_level_keys"] = sorted(map(str, payload.keys()))[: policy.max_preview_keys]
    elif isinstance(payload, list):
        detail["num_items"] = len(payload)
    return detail


def _validate_wav(path: Path, *, policy: RuntimeEvidencePolicy) -> dict[str, Any]:
    size = _safe_size(path)
    if policy.require_nonempty_wav and (size is None or size <= 44):
        return {"status": "fail", "reason": "empty_or_too_small_wav", "size_bytes": size}
    try:
        with wave.open(str(path), "rb") as reader:
            frames = reader.getnframes()
            frame_rate = reader.getframerate()
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
    except Exception as exc:  # noqa: BLE001 - report should aggregate structural failures
        return {"status": "fail", "reason": "invalid_wav", "error": str(exc), "size_bytes": size}
    duration = float(frames) / float(frame_rate) if frame_rate else 0.0
    if policy.require_nonempty_wav and frames <= 0:
        return {"status": "fail", "reason": "zero_frame_wav", "size_bytes": size, "sample_rate_hz": frame_rate}
    return {
        "status": "pass",
        "size_bytes": size,
        "num_frames": frames,
        "sample_rate_hz": frame_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "duration_seconds": round(duration, 6),
    }


def _validate_by_suffix(path: Path, *, policy: RuntimeEvidencePolicy) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _validate_json(path, policy=policy)
    if suffix == ".wav":
        return _validate_wav(path, policy=policy)
    return {"status": "pass", "size_bytes": _safe_size(path), "validator": "exists_only"}


def validate_evidence_spec(spec: EvidenceSpec, *, root: str | Path, policy: RuntimeEvidencePolicy | None = None) -> dict[str, Any]:
    """Validate one runtime evidence path structurally."""

    policy = policy or RuntimeEvidencePolicy()
    project_root = Path(root).expanduser().resolve()
    path = project_root / spec.path
    if not path.exists():
        return sanitize_for_audit(
            {
                "path": spec.path,
                "description": spec.description,
                "required": spec.required,
                "status": "missing" if spec.required else "skipped",
                "exists": False,
            }
        )
    detail = _validate_by_suffix(path, policy=policy)
    return sanitize_for_audit(
        {
            "path": spec.path,
            "description": spec.description,
            "required": spec.required,
            "exists": True,
            **detail,
        }
    )


def build_runtime_evidence_report(*, root: str | Path, policy: RuntimeEvidencePolicy | None = None) -> dict[str, Any]:
    """Validate all runtime evidence entries from the plan-compliance registry."""

    policy = policy or RuntimeEvidencePolicy()
    project_root = Path(root).expanduser().resolve()
    milestones: list[dict[str, Any]] = []
    aggregate = {"pass": 0, "fail": 0, "missing": 0, "skipped": 0}
    for milestone in MILESTONES:
        checks = [validate_evidence_spec(spec, root=project_root, policy=policy) for spec in milestone.runtime_evidence]
        for check in checks:
            status = str(check.get("status"))
            aggregate[status] = int(aggregate.get(status, 0)) + 1
        required_checks = [check for check in checks if check.get("required")]
        required_bad = [check for check in required_checks if check.get("status") != "pass"]
        if not checks:
            status = "no_runtime_evidence_declared"
        elif required_bad:
            status = "not_verified"
        else:
            status = "verified"
        milestones.append(
            sanitize_for_audit(
                {
                    "milestone_id": milestone.milestone_id,
                    "title": milestone.title,
                    "runtime_status": status,
                    "num_runtime_checks": len(checks),
                    "num_required_runtime_checks": len(required_checks),
                    "num_required_runtime_not_passed": len(required_bad),
                    "runtime_blockers": list(milestone.runtime_blockers),
                    "checks": checks,
                }
            )
        )
    report = {
        "report_type": "aegisvox_runtime_evidence_report",
        "generated_at": utc_now_iso(),
        "project_root": str(project_root),
        "summary": {
            "num_milestones": len(milestones),
            "aggregate_check_status": dict(sorted(aggregate.items())),
            "runtime_verified_milestones": [row["milestone_id"] for row in milestones if row["runtime_status"] == "verified"],
            "runtime_pending_milestones": [row["milestone_id"] for row in milestones if row["runtime_status"] == "not_verified"],
        },
        "milestones": milestones,
        "claim_boundary": (
            "Runtime evidence validation checks structural artifact presence/parseability only. It does not prove privacy, "
            "utility, PHI safety, speaker unlinkability, audio quality, or clinical readiness."
        ),
    }
    return sanitize_for_audit(report)


def runtime_evidence_markdown(report: dict[str, Any]) -> str:
    """Render a compact runtime evidence validation report."""

    summary = report.get("summary") or {}
    lines = [
        "# AegisVox runtime evidence report",
        "",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Project root: `{report.get('project_root')}`",
        f"- Check statuses: `{json.dumps(summary.get('aggregate_check_status') or {}, sort_keys=True)}`",
        f"- Runtime verified milestones: `{summary.get('runtime_verified_milestones')}`",
        f"- Runtime pending milestones: `{summary.get('runtime_pending_milestones')}`",
        "",
        "## Milestone matrix",
        "",
        "| ID | Milestone | Runtime status | Checks | Required not passed |",
        "|---:|---|---|---:|---:|",
    ]
    for milestone in report.get("milestones") or []:
        lines.append(
            "| {id} | {title} | `{status}` | {checks} | {bad} |".format(
                id=milestone.get("milestone_id"),
                title=str(milestone.get("title", "")).replace("|", "\\|"),
                status=milestone.get("runtime_status"),
                checks=milestone.get("num_runtime_checks"),
                bad=milestone.get("num_required_runtime_not_passed"),
            )
        )
    lines.extend(["", "## Missing or failed required runtime evidence", ""])
    for milestone in report.get("milestones") or []:
        bad = [check for check in milestone.get("checks") or [] if check.get("required") and check.get("status") != "pass"]
        if not bad:
            continue
        lines.append(f"### M{milestone.get('milestone_id')} {milestone.get('title')}")
        for check in bad:
            lines.append(f"- `{check.get('path')}`: `{check.get('status')}` ({check.get('description')})")
        lines.append("")
    lines.extend(["## Claim boundary", "", str(report.get("claim_boundary"))])
    return "\n".join(lines) + "\n"
