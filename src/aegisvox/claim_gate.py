"""Conservative claim/action gate for AegisVox control bundles.

The gate consumes compact reports produced by the local/ORD control-plane scripts
and translates them into explicit allowed/blocked claims and next actions. It is
not an empirical evaluator; it prevents operational and paper-positioning drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audit import sanitize_for_audit
from .schema import utc_now_iso


DEFAULT_REPORT_DIR = Path("outputs/local_control")


REPORT_FILENAMES = {
    "status_bundle": "status_bundle.json",
    "preflight_report": "preflight_report.json",
    "worker_launch_plan": "worker_launch_plan.json",
    "plan_compliance_report": "plan_compliance_report.json",
    "runtime_evidence_report": "runtime_evidence_report.json",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _report_status(name: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"name": name, "present": False, "status": "missing"}
    if name == "status_bundle":
        status = payload.get("overall_status") or "unknown"
    elif name == "preflight_report":
        status = payload.get("status") or ("pass" if payload.get("ok") else "fail")
    elif name == "worker_launch_plan":
        status = "present" if payload.get("plan_type") == "aegisvox_worker_launch_plan" else "invalid"
    elif name == "plan_compliance_report":
        missing_impl = int((payload.get("summary") or {}).get("missing_required_implementation_evidence") or 0)
        status = "pass" if missing_impl == 0 else "fail"
    elif name == "runtime_evidence_report":
        pending = (payload.get("summary") or {}).get("runtime_pending_milestones") or []
        status = "pass" if not pending else "pending"
    else:
        status = "present"
    return {"name": name, "present": True, "status": status}


def _has_runtime_pending(runtime_report: dict[str, Any] | None) -> bool:
    if not runtime_report:
        return True
    return bool((runtime_report.get("summary") or {}).get("runtime_pending_milestones") or [])


def _missing_impl_count(plan_report: dict[str, Any] | None) -> int | None:
    if not plan_report:
        return None
    return int((plan_report.get("summary") or {}).get("missing_required_implementation_evidence") or 0)


def _preflight_failed(preflight_report: dict[str, Any] | None) -> bool | None:
    if preflight_report is None:
        return None
    status = preflight_report.get("status") or ("pass" if preflight_report.get("ok") else "fail")
    return status == "fail"


def build_claim_gate(
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    status_bundle: dict[str, Any] | None = None,
    preflight_report: dict[str, Any] | None = None,
    worker_launch_plan: dict[str, Any] | None = None,
    plan_compliance_report: dict[str, Any] | None = None,
    runtime_evidence_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a conservative claim/action gate from control reports."""

    base = Path(report_dir)
    reports = {
        "status_bundle": status_bundle if status_bundle is not None else _load_json(base / REPORT_FILENAMES["status_bundle"]),
        "preflight_report": preflight_report if preflight_report is not None else _load_json(base / REPORT_FILENAMES["preflight_report"]),
        "worker_launch_plan": worker_launch_plan if worker_launch_plan is not None else _load_json(base / REPORT_FILENAMES["worker_launch_plan"]),
        "plan_compliance_report": plan_compliance_report
        if plan_compliance_report is not None
        else _load_json(base / REPORT_FILENAMES["plan_compliance_report"]),
        "runtime_evidence_report": runtime_evidence_report
        if runtime_evidence_report is not None
        else _load_json(base / REPORT_FILENAMES["runtime_evidence_report"]),
    }
    report_statuses = [_report_status(name, reports[name]) for name in REPORT_FILENAMES]
    missing_reports = [row["name"] for row in report_statuses if not row["present"]]
    missing_impl = _missing_impl_count(reports["plan_compliance_report"])
    runtime_pending = _has_runtime_pending(reports["runtime_evidence_report"])
    preflight_fail = _preflight_failed(reports["preflight_report"])
    launch_plan_present = bool(reports["worker_launch_plan"])

    allowed_claims: list[str] = []
    blocked_claims: list[dict[str, str]] = []
    recommended_actions: list[dict[str, str]] = []

    if missing_impl == 0:
        allowed_claims.append("local_implementation_evidence_complete")
    else:
        blocked_claims.append({"claim": "local_implementation_evidence_complete", "reason": "missing_plan_compliance_report_or_implementation_evidence"})

    if not runtime_pending:
        allowed_claims.append("runtime_artifacts_structurally_verified")
    else:
        blocked_claims.append({"claim": "runtime_artifacts_structurally_verified", "reason": "runtime_evidence_missing_or_pending"})

    if not runtime_pending and preflight_fail is False:
        allowed_claims.append("candidate_for_full_stack_smoke_review")
    else:
        blocked_claims.append({"claim": "candidate_for_full_stack_smoke_review", "reason": "runtime_pending_or_preflight_missing_or_failed"})

    blocked_claims.extend(
        [
            {"claim": "privacy_or_phi_safety_proven", "reason": "requires evaluation artifacts and safety thresholds, not just structural reports"},
            {"claim": "clinical_readiness", "reason": "requires real backend evidence, privacy/utility/safety gates, and human/compliance review"},
            {"claim": "paper_ready_results", "reason": "runtime/full-stack evidence and evaluation reports remain required"},
        ]
    )

    if missing_reports:
        recommended_actions.append({"action": "build_missing_control_reports", "reason": ",".join(missing_reports)})
    if preflight_fail is None:
        recommended_actions.append({"action": "run_preflight_report", "reason": "preflight_report_missing"})
    elif preflight_fail:
        recommended_actions.append({"action": "fix_preflight_failures_service_by_service", "reason": "required_runtime_config_missing_or_failed"})
    if runtime_pending:
        recommended_actions.append({"action": "produce_real_runtime_artifacts_in_plan_order", "reason": "runtime_evidence_report_has_pending_milestones"})
    if launch_plan_present and preflight_fail is False:
        recommended_actions.append({"action": "review_worker_launch_plan_before_tmux_start", "reason": "launch_plan_available_and_preflight_passed"})
    elif launch_plan_present:
        recommended_actions.append({"action": "do_not_start_model_workers_until_preflight_review", "reason": "launch_plan_exists_but_preflight_not_green"})

    overall = "pass" if not runtime_pending and preflight_fail is False and missing_impl == 0 else "blocked"
    if missing_impl == 0 and runtime_pending:
        overall = "implementation_ready_runtime_pending"
    if missing_reports:
        overall = "control_reports_incomplete"

    gate = {
        "gate_type": "aegisvox_claim_gate",
        "generated_at": utc_now_iso(),
        "report_dir": str(base),
        "overall_status": overall,
        "reports": report_statuses,
        "allowed_claims": allowed_claims,
        "blocked_claims": blocked_claims,
        "recommended_actions": recommended_actions,
        "launch_allowed": bool(not missing_reports and preflight_fail is False and launch_plan_present),
        "runtime_claims_allowed": bool(not runtime_pending),
        "clinical_claims_allowed": False,
        "claim_boundary": (
            "The claim gate is an operational/positioning guardrail. It never proves privacy, utility, PHI safety, "
            "speaker unlinkability, audio quality, or clinical readiness by itself."
        ),
    }
    return sanitize_for_audit(gate)


def claim_gate_markdown(gate: dict[str, Any]) -> str:
    """Render a compact Markdown claim gate."""

    lines = [
        "# AegisVox claim gate",
        "",
        f"- Generated at: `{gate.get('generated_at')}`",
        f"- Overall status: **{gate.get('overall_status')}**",
        f"- Launch allowed: `{gate.get('launch_allowed')}`",
        f"- Runtime claims allowed: `{gate.get('runtime_claims_allowed')}`",
        f"- Clinical claims allowed: `{gate.get('clinical_claims_allowed')}`",
        "",
        "## Reports",
        "",
        "| Report | Present | Status |",
        "|---|---:|---|",
    ]
    for row in gate.get("reports") or []:
        lines.append(f"| `{row.get('name')}` | {row.get('present')} | `{row.get('status')}` |")
    lines.extend(["", "## Allowed claims", ""])
    for claim in gate.get("allowed_claims") or []:
        lines.append(f"- `{claim}`")
    lines.extend(["", "## Blocked claims", ""])
    for row in gate.get("blocked_claims") or []:
        lines.append(f"- `{row.get('claim')}`: {row.get('reason')}")
    lines.extend(["", "## Recommended actions", ""])
    for row in gate.get("recommended_actions") or []:
        lines.append(f"- `{row.get('action')}`: {row.get('reason')}")
    lines.extend(["", "## Claim boundary", "", str(gate.get("claim_boundary"))])
    return "\n".join(lines) + "\n"
