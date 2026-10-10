"""Provenance-preserving policy for manually approved output candidates.

This policy is intentionally separate from `output_policy.py`: canonical output
remains the default product artifact. If a manual review approves a diarization
candidate, this module records a promoted *candidate* download decision with
explicit provenance and claim boundaries, without copying over or mutating
`final_sanitized.wav`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import utc_now_iso


PROMOTION_BLOCKING_REVIEW_CRITERIA = {
    "speaker_preservation",
    "voice_quality",
    "overlap_handling",
    "background_or_noise",
    "privacy_or_phi_risk",
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _candidate_summaries(run_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for summary_path in sorted((run_dir / "candidates").glob("*/candidate_summary.json")) if (run_dir / "candidates").exists() else []:
        payload = _load_json(summary_path)
        if not payload:
            continue
        candidate_id = str(payload.get("candidate_id") or summary_path.parent.name)
        if not isinstance(payload.get("assignment_targets"), dict):
            copied_assignment = _load_json(summary_path.parent / "speaker_target_map.json")
            assignments = copied_assignment.get("assignments") if isinstance(copied_assignment.get("assignments"), dict) else None
            if assignments:
                payload = {**payload, "assignment_targets": assignments}
        out[candidate_id] = payload
    return out


def _review_rows(candidate_reviews: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = candidate_reviews.get("reviews") if isinstance(candidate_reviews, dict) else []
    if not isinstance(rows, list):
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def _candidate_uri(summary: dict[str, Any]) -> str | None:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    uri = artifacts.get("changed_identity_audio_uri")
    return str(uri) if uri else None


def _has_hybrid_trait_evidence(summary: dict[str, Any]) -> bool:
    targets = summary.get("assignment_targets") if isinstance(summary.get("assignment_targets"), dict) else {}
    for target in targets.values():
        row = target if isinstance(target, dict) else {}
        if isinstance(row.get("hybrid_search_explanation"), dict) or isinstance(row.get("query_trait_match"), dict):
            return True
    return False


def _blocking_hybrid_warnings(summary: dict[str, Any]) -> list[str]:
    """Return hybrid-search warnings that should block candidate promotion.

    Soft trait warnings remain reviewer-visible caution. Hard guardrail warnings
    mean the selected pseudo-speaker may violate language/gender/accent matching
    constraints, so an approve click alone must not expose it as promoted output.
    """

    warnings: list[str] = []
    targets = summary.get("assignment_targets") if isinstance(summary.get("assignment_targets"), dict) else {}
    for target in targets.values():
        row = target if isinstance(target, dict) else {}
        explanation = row.get("hybrid_search_explanation") if isinstance(row.get("hybrid_search_explanation"), dict) else {}
        raw_warnings = explanation.get("warnings") if isinstance(explanation.get("warnings"), list) else []
        for warning in raw_warnings:
            warning_text = str(warning)
            if warning_text.startswith("missing_hard_guardrail:") or warning_text.startswith("mismatched_hard_guardrail:"):
                warnings.append(warning_text)
    # Preserve deterministic reporting while de-duplicating across speakers.
    return sorted(set(warnings))


def _failed_review_criteria(review: dict[str, Any]) -> dict[str, str]:
    criteria = review.get("criteria") if isinstance(review.get("criteria"), dict) else {}
    failed: dict[str, str] = {}
    for key, value in criteria.items():
        key_str = str(key)
        value_str = str(value).strip().lower()
        if key_str in PROMOTION_BLOCKING_REVIEW_CRITERIA and value_str == "fail":
            failed[key_str] = value_str
    return failed


def build_promoted_output_policy(
    *,
    run_id: str,
    output_policy: dict[str, Any] | None,
    candidate_reviews: dict[str, Any] | None,
    candidate_summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build an auditable policy report for approved candidate downloads."""

    output_policy = output_policy or {}
    canonical_uri = (output_policy.get("canonical") or {}).get("uri") or output_policy.get("recommended_for_download")
    approved_reviews = [row for row in _review_rows(candidate_reviews) if row.get("decision") == "approve_for_promotion"]
    rejected_or_pending = [row for row in _review_rows(candidate_reviews) if row.get("decision") != "approve_for_promotion"]
    reasons: list[str] = []
    promoted: dict[str, Any] | None = None
    blocked_approved_reviews: list[dict[str, Any]] = []
    for review in sorted(approved_reviews, key=lambda row: str(row.get("created_at") or ""), reverse=True):
        candidate_id = str(review.get("candidate_id") or "")
        failed_criteria = _failed_review_criteria(review)
        if failed_criteria:
            blocked_approved_reviews.append(
                {
                    "candidate_id": candidate_id,
                    "decision_created_at": review.get("created_at"),
                    "failed_criteria": failed_criteria,
                }
            )
            reasons.append(
                f"approved candidate {candidate_id!r} is blocked by failed review criteria: "
                + ", ".join(sorted(failed_criteria))
            )
            continue
        summary = candidate_summaries.get(candidate_id)
        if not summary:
            reasons.append(f"approved candidate {candidate_id!r} has no candidate_summary.json")
            continue
        if not _has_hybrid_trait_evidence(summary):
            blocked_approved_reviews.append(
                {
                    "candidate_id": candidate_id,
                    "decision_created_at": review.get("created_at"),
                    "missing_hybrid_trait_evidence": True,
                }
            )
            reasons.append(
                f"approved candidate {candidate_id!r} is blocked because it lacks hybrid trait evidence; "
                "regenerate with the current hybrid assignment or review it as a non-promotable baseline"
            )
            continue
        blocking_hybrid_warnings = _blocking_hybrid_warnings(summary)
        if blocking_hybrid_warnings:
            blocked_approved_reviews.append(
                {
                    "candidate_id": candidate_id,
                    "decision_created_at": review.get("created_at"),
                    "hybrid_blocking_warnings": blocking_hybrid_warnings,
                }
            )
            reasons.append(
                f"approved candidate {candidate_id!r} is blocked by hard hybrid-search warnings: "
                + ", ".join(blocking_hybrid_warnings)
            )
            continue
        uri = _candidate_uri(summary)
        if not uri:
            reasons.append(f"approved candidate {candidate_id!r} has no changed_identity_audio_uri")
            continue
        if summary.get("status") != "completed":
            reasons.append(f"approved candidate {candidate_id!r} is not completed")
            continue
        promoted = {
            "candidate_id": candidate_id,
            "uri": uri,
            "decision_created_at": review.get("created_at"),
            "reviewer_id": review.get("reviewer_id"),
            "backend": summary.get("backend"),
            "model_id": summary.get("model_id"),
            "overlap_policy": summary.get("overlap_policy"),
            "num_clean_speakers": summary.get("num_clean_speakers"),
            "num_clean_segments": summary.get("num_clean_segments"),
            "num_stitched": summary.get("num_stitched"),
            "num_not_stitched": summary.get("num_not_stitched"),
            "num_converted_overlap": summary.get("num_converted_overlap"),
            "num_skipped_overlap": summary.get("num_skipped_overlap"),
            "review_criteria": review.get("criteria") if isinstance(review.get("criteria"), dict) else {},
        }
        reasons.append(f"manual review approved candidate {candidate_id!r} for promotion follow-up")
        break
    if promoted is None:
        reasons.append("no completed candidate has an approve_for_promotion manual review")

    return {
        "service": "promoted_output_policy",
        "schema_version": "aegisvox_promoted_output_policy.v1",
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "canonical": {
            "uri": canonical_uri,
            "remains_default": True,
            "canonical_output_mutated": False,
        },
        "promoted_candidate": promoted,
        "promoted_download_allowed": promoted is not None,
        "recommended_for_download": canonical_uri,
        "promoted_candidate_download_uri": promoted.get("uri") if promoted else None,
        "num_approved_reviews": len(approved_reviews),
        "num_blocked_approved_reviews": len(blocked_approved_reviews),
        "num_nonapproved_reviews": len(rejected_or_pending),
        "blocked_approved_reviews": blocked_approved_reviews,
        "approval_safety_policy": {
            "blocking_failed_criteria": sorted(PROMOTION_BLOCKING_REVIEW_CRITERIA),
            "requires_hybrid_trait_evidence": True,
            "blocking_hybrid_warning_prefixes": ["missing_hard_guardrail:", "mismatched_hard_guardrail:"],
            "rule": "An approve_for_promotion review is not enough if hybrid trait evidence is absent, any blocking review criterion is marked fail, or a hard hybrid-search guardrail warning is present.",
        },
        "reasons": reasons,
        "claim_boundary": (
            "This report can expose an approved candidate for manual/review download, but canonical final_sanitized.wav "
            "remains the default and is not overwritten. Clinical/privacy readiness still requires separate release gates."
        ),
    }


def build_promoted_output_policy_from_run(run_dir: Path) -> dict[str, Any]:
    output_policy = _load_json(run_dir / "output_policy_report.json")
    if not output_policy and (run_dir / "final_sanitized.wav").exists():
        canonical_uri = f"s3://aegisvox/runs/{run_dir.name}/final_sanitized.wav"
        output_policy = {
            "canonical": {"uri": canonical_uri},
            "recommended_for_download": canonical_uri,
        }
    return build_promoted_output_policy(
        run_id=run_dir.name,
        output_policy=output_policy,
        candidate_reviews=_load_json(run_dir / "diarization_candidate_reviews.json"),
        candidate_summaries=_candidate_summaries(run_dir),
    )


def write_promoted_output_policy_report(run_dir: Path) -> dict[str, Any]:
    payload = build_promoted_output_policy_from_run(run_dir)
    _write_json(run_dir / "promoted_output_policy_report.json", payload)
    return payload
