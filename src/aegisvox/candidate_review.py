"""Manual review decisions for diarization rerun candidates.

These artifacts are deliberately lightweight and audit-safe. A decision can mark
a candidate as promising or rejected, but it never mutates the canonical output
or silently promotes a candidate into the product default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .audit import sanitize_for_audit
from .schema import AegisVoxContractError, utc_now_iso


class CandidateReviewDecision(str, Enum):
    APPROVE_FOR_PROMOTION = "approve_for_promotion"
    REJECT = "reject"
    NEEDS_MORE_WORK = "needs_more_work"


class CandidateReviewError(RuntimeError):
    """Raised when a candidate review decision cannot be written."""


_ALLOWED_CRITERIA_STATUS = {"pass", "warn", "fail", "unknown", None}
_DEFAULT_CRITERIA = {
    "speaker_preservation": "unknown",
    "voice_quality": "unknown",
    "overlap_handling": "unknown",
    "background_or_noise": "unknown",
    "privacy_or_phi_risk": "unknown",
}


@dataclass(frozen=True)
class CandidateReviewRequest:
    run_id: str
    candidate_id: str
    decision: CandidateReviewDecision | str
    reviewer_id: str = "manual_reviewer"
    rationale: str = ""
    criteria: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise AegisVoxContractError("run_id is required")
        if not self.candidate_id:
            raise AegisVoxContractError("candidate_id is required")
        if isinstance(self.decision, str):
            object.__setattr__(self, "decision", CandidateReviewDecision(self.decision))
        if not self.reviewer_id:
            object.__setattr__(self, "reviewer_id", "manual_reviewer")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _sanitize_criteria(criteria: dict[str, Any]) -> dict[str, str]:
    merged = dict(_DEFAULT_CRITERIA)
    for key, value in (criteria or {}).items():
        key_str = str(key)
        value_str = str(value).strip().lower() if value is not None else "unknown"
        if value_str not in _ALLOWED_CRITERIA_STATUS:
            value_str = "unknown"
        merged[key_str] = value_str
    return merged


def build_candidate_review_payload(*, run_dir: Path, request: CandidateReviewRequest) -> dict[str, Any]:
    candidate_dir = run_dir / "candidates" / request.candidate_id
    summary = _load_json(candidate_dir / "candidate_summary.json")
    if not summary:
        raise CandidateReviewError(f"candidate summary not found for {request.candidate_id}")
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    audio_path = candidate_dir / "changed_identity.wav"
    if not audio_path.exists():
        raise CandidateReviewError(f"candidate audio not found for {request.candidate_id}")
    decision = request.decision.value if isinstance(request.decision, CandidateReviewDecision) else str(request.decision)
    promotion_recommendation = {
        CandidateReviewDecision.APPROVE_FOR_PROMOTION.value: "candidate_selected_for_followup_policy_review",
        CandidateReviewDecision.REJECT.value: "do_not_promote_candidate",
        CandidateReviewDecision.NEEDS_MORE_WORK.value: "needs_more_candidate_work_before_promotion",
    }[decision]
    return {
        "schema_version": "aegisvox_candidate_review.v1",
        "created_at": utc_now_iso(),
        "run_id": request.run_id,
        "candidate_id": request.candidate_id,
        "decision": decision,
        "reviewer_id": sanitize_for_audit(request.reviewer_id, max_string_length=120),
        "rationale": sanitize_for_audit(request.rationale, max_string_length=1000),
        "criteria": _sanitize_criteria(request.criteria),
        "candidate": {
            "status": summary.get("status"),
            "backend": summary.get("backend"),
            "model_id": summary.get("model_id"),
            "overlap_policy": summary.get("overlap_policy"),
            "num_clean_speakers": summary.get("num_clean_speakers"),
            "num_clean_segments": summary.get("num_clean_segments"),
            "num_stitched": summary.get("num_stitched"),
            "num_not_stitched": summary.get("num_not_stitched"),
            "num_converted_overlap": summary.get("num_converted_overlap"),
            "num_skipped_overlap": summary.get("num_skipped_overlap"),
            "changed_identity_audio_uri": artifacts.get("changed_identity_audio_uri"),
            "candidate_summary_uri": artifacts.get("candidate_summary_uri"),
        },
        "promotion_recommendation": promotion_recommendation,
        "canonical_output_mutated": False,
        "claim_boundary": (
            "Manual review decision over a diarization candidate. This artifact does not mutate the canonical output; "
            "promotion still requires explicit output-policy work and provenance-preserving release gates."
        ),
    }


def load_candidate_reviews(run_dir: Path) -> dict[str, Any]:
    candidates_dir = run_dir / "candidates"
    reviews = []
    history = _load_jsonl(run_dir / "diarization_candidate_review_history.jsonl")
    if candidates_dir.exists():
        for review_path in sorted(candidates_dir.glob("*/manual_review_decision.json")):
            payload = _load_json(review_path)
            if payload:
                reviews.append(payload)
    history_by_candidate: dict[str, int] = {}
    for row in history:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            history_by_candidate[candidate_id] = history_by_candidate.get(candidate_id, 0) + 1
    return {
        "schema_version": "aegisvox_candidate_reviews.v1",
        "run_id": run_dir.name,
        "num_reviews": len(reviews),
        "num_history_events": len(history),
        "reviews": reviews,
        "history": history,
        "history_by_candidate": history_by_candidate,
        "latest_reviews_are_current_state": True,
        "claim_boundary": (
            "Review decisions are manual evidence and do not automatically promote candidates. "
            "The reviews field contains the latest decision per candidate; history is append-only."
        ),
    }


def write_candidate_review(*, run_dir: Path, request: CandidateReviewRequest) -> dict[str, Any]:
    payload = build_candidate_review_payload(run_dir=run_dir, request=request)
    candidate_dir = run_dir / "candidates" / request.candidate_id
    _append_jsonl(candidate_dir / "manual_review_history.jsonl", payload)
    _append_jsonl(run_dir / "diarization_candidate_review_history.jsonl", payload)
    _write_json(candidate_dir / "manual_review_decision.json", payload)
    aggregate = load_candidate_reviews(run_dir)
    _write_json(run_dir / "diarization_candidate_reviews.json", aggregate)
    return payload
