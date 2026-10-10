"""Side-by-side listening review packet for AegisVox output candidates.

The packet is artifact-only and audit-safe: it lists canonical and candidate
outputs, the exact manual-review criteria, and claim boundaries. It does not
copy audio, expose raw transcript text, or promote any candidate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .candidate_review import load_candidate_reviews
from .promoted_output_policy import build_promoted_output_policy_from_run
from .schema import utc_now_iso
from .voice_pool import build_hybrid_search_explanation


REVIEW_CRITERIA: list[dict[str, Any]] = [
    {
        "key": "speaker_preservation",
        "label": "Speaker preservation",
        "question": "Does the candidate preserve the number of audible speakers and short secondary-speaker turns better than canonical?",
        "allowed_values": ["pass", "warn", "fail", "unknown"],
        "promotion_guidance": "Prefer candidates that preserve audible speakers without collapsing them into one pseudo-speaker.",
    },
    {
        "key": "voice_quality",
        "label": "Voice quality",
        "question": "Is the transformed speech intelligible, natural enough for review, and free of severe artifacts?",
        "allowed_values": ["pass", "warn", "fail", "unknown"],
        "promotion_guidance": "Do not approve candidates with severe conversion or stitching artifacts even if diarization is better.",
    },
    {
        "key": "overlap_handling",
        "label": "Overlap handling",
        "question": "Are overlapping/interrupting speaker regions handled acceptably, and are risky overlap-converted regions understood as review-only?",
        "allowed_values": ["pass", "warn", "fail", "unknown"],
        "promotion_guidance": "Overlap-converted candidates can be approved for follow-up listening, but remain engineering-review artifacts unless separately validated.",
    },
    {
        "key": "background_or_noise",
        "label": "Background/noise realism",
        "question": "Does the candidate preserve useful ambience without making speech/background leakage worse?",
        "allowed_values": ["pass", "warn", "fail", "unknown"],
        "promotion_guidance": "Background-preserved outputs are separate candidates; do not conflate ambience realism with privacy readiness.",
    },
    {
        "key": "privacy_or_phi_risk",
        "label": "Privacy/PHI risk",
        "question": "Does anything sound like original-speaker identity leakage, unedited PHI, or unsafe background speech leakage?",
        "allowed_values": ["pass", "warn", "fail", "unknown"],
        "promotion_guidance": "Any suspected privacy/PHI issue should be marked warn/fail and block product-style promotion pending deeper gates.",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _file_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "size_bytes": None, "sha256": None}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"exists": True, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _review_by_candidate(run_dir: Path) -> dict[str, dict[str, Any]]:
    reviews = load_candidate_reviews(run_dir).get("reviews")
    out: dict[str, dict[str, Any]] = {}
    if isinstance(reviews, list):
        for row in reviews:
            if isinstance(row, dict) and row.get("candidate_id"):
                out[str(row["candidate_id"])] = row
    return out


def _canonical_uri(run_dir: Path) -> str | None:
    output_policy = _load_json(run_dir / "output_policy_report.json")
    canonical = output_policy.get("canonical") if isinstance(output_policy.get("canonical"), dict) else {}
    uri = canonical.get("uri") or output_policy.get("recommended_for_download")
    if uri:
        return str(uri)
    if (run_dir / "final_sanitized.wav").exists():
        return f"s3://aegisvox/runs/{run_dir.name}/final_sanitized.wav"
    return None


TRAIT_EVIDENCE_KEYS = {
    "language",
    "gender_presentation",
    "accent_group",
    "geography_region",
    "tone",
    "energy",
    "speaking_rate",
    "pitch_band",
}


def _safe_trait_map(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {str(key): payload.get(key) for key in sorted(TRAIT_EVIDENCE_KEYS) if payload.get(key) not in {None, "", "unknown"}}


def _safe_query_trait_match(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    out: dict[str, Any] = {}
    if isinstance(payload.get("score"), (int, float)):
        out["score"] = float(payload["score"])
    for key in ["matched", "equivalent", "mismatched"]:
        value = payload.get(key)
        if isinstance(value, dict):
            out[key] = value
    missing = payload.get("missing")
    if isinstance(missing, list):
        out["missing"] = [str(item) for item in missing]
    for key in ["total_weight", "matched_weight"]:
        if isinstance(payload.get(key), (int, float)):
            out[key] = float(payload[key])
    return out or None


def _query_traits_from_match(trait_match: dict[str, Any] | None, source_traits: dict[str, Any], target_traits: dict[str, Any]) -> dict[str, str]:
    query_traits: dict[str, str] = {str(key): str(value) for key, value in source_traits.items() if value not in {None, "", "unknown"}}
    if not isinstance(trait_match, dict):
        return query_traits
    matched = trait_match.get("matched") if isinstance(trait_match.get("matched"), dict) else {}
    equivalent = trait_match.get("equivalent") if isinstance(trait_match.get("equivalent"), dict) else {}
    mismatched = trait_match.get("mismatched") if isinstance(trait_match.get("mismatched"), dict) else {}
    for key, value in matched.items():
        if key in TRAIT_EVIDENCE_KEYS and key not in query_traits and value not in {None, "", "unknown"}:
            query_traits[str(key)] = str(value)
    for key, value in equivalent.items():
        if key not in TRAIT_EVIDENCE_KEYS or key in query_traits or not isinstance(value, dict):
            continue
        query = value.get("query")
        if query not in {None, "", "unknown"}:
            query_traits[str(key)] = str(query)
    for key, value in mismatched.items():
        if key not in TRAIT_EVIDENCE_KEYS or key in query_traits or not isinstance(value, dict):
            continue
        query = value.get("query")
        if query not in {None, "", "unknown"}:
            query_traits[str(key)] = str(query)
    for key, value in target_traits.items():
        # For exact matches, query and candidate are equivalent. This recovers
        # older assignment maps that kept query_trait_match but not source_audio_traits.
        if key in matched and key not in query_traits and value not in {None, "", "unknown"}:
            query_traits[str(key)] = str(value)
    return query_traits


def _derive_hybrid_search_explanation(
    *,
    trait_match: dict[str, Any] | None,
    source_traits: dict[str, Any],
    target_traits: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(trait_match, dict) or not trait_match:
        return None
    query_traits = _query_traits_from_match(trait_match, source_traits, target_traits)
    sample_traits = {str(key): str(value) for key, value in target_traits.items() if value not in {None, "", "unknown"}}
    if not query_traits or not sample_traits:
        return None
    explanation = build_hybrid_search_explanation(sample_traits, query_traits, trait_match)
    explanation["evidence_source"] = "reconstructed_from_query_trait_match"
    explanation["reconstruction_note"] = (
        "Derived for review from stored query_trait_match and source/target trait evidence; "
        "candidate audio and assignments were not regenerated."
    )
    return explanation


def _safe_hybrid_search_explanation(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    out: dict[str, Any] = {}
    if isinstance(payload.get("evidence_source"), str):
        out["evidence_source"] = payload["evidence_source"]
    if isinstance(payload.get("reconstruction_note"), str):
        out["reconstruction_note"] = payload["reconstruction_note"]
    for key in ["hard_guardrails", "soft_rerank"]:
        value = payload.get(key)
        if isinstance(value, dict):
            out[key] = value
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        out["warnings"] = [str(item) for item in warnings]
    if isinstance(payload.get("trait_score"), (int, float)):
        out["trait_score"] = float(payload["trait_score"])
    if isinstance(payload.get("claim_boundary"), str):
        out["claim_boundary"] = payload["claim_boundary"]
    return out or None


def _safe_assignment_targets(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    targets = summary.get("assignment_targets") if isinstance(summary.get("assignment_targets"), dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for speaker_id, raw_row in sorted(targets.items()):
        row = raw_row if isinstance(raw_row, dict) else {}
        trait_match = _safe_query_trait_match(row.get("query_trait_match"))
        target_traits = _safe_trait_map(row.get("target_audio_traits"))
        source_traits = _safe_trait_map(row.get("source_audio_traits"))
        explanation = _safe_hybrid_search_explanation(row.get("hybrid_search_explanation"))
        if explanation is None:
            explanation = _derive_hybrid_search_explanation(
                trait_match=trait_match,
                source_traits=source_traits,
                target_traits=target_traits,
            )
        out[str(speaker_id)] = {
            "target_voice_id": row.get("target_voice_id"),
            "target_voice_sample_id": row.get("target_voice_sample_id"),
            "source_audio_traits": source_traits,
            "target_audio_traits": target_traits,
            "query_trait_match": trait_match,
            "hybrid_search_explanation": explanation,
        }
    return out


def _hybrid_warning_status_for_warnings(warnings: list[str], *, saw_explanation: bool) -> str:
    for warning in warnings:
        if warning.startswith("missing_hard_guardrail:") or warning.startswith("mismatched_hard_guardrail:"):
            return "hard_guardrail_warning"
    if warnings:
        return "soft_trait_warning"
    if saw_explanation:
        return "no_hybrid_warnings"
    return "no_hybrid_explanation"


def _hybrid_warning_summary(assignment_targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    warnings_by_speaker: dict[str, list[str]] = {}
    all_warnings: list[str] = []
    num_explanations = 0
    for speaker_id, row in assignment_targets.items():
        explanation = row.get("hybrid_search_explanation") if isinstance(row.get("hybrid_search_explanation"), dict) else None
        if explanation is None:
            continue
        num_explanations += 1
        warnings = explanation.get("warnings") if isinstance(explanation.get("warnings"), list) else []
        safe_warnings = [str(item) for item in warnings]
        warnings_by_speaker[str(speaker_id)] = safe_warnings
        all_warnings.extend(safe_warnings)
    warning_counts: dict[str, int] = {}
    for warning in all_warnings:
        warning_counts[warning] = warning_counts.get(warning, 0) + 1
    status = _hybrid_warning_status_for_warnings(all_warnings, saw_explanation=num_explanations > 0)
    return {
        "status": status,
        "num_speakers_with_explanation": num_explanations,
        "num_warnings": len(all_warnings),
        "warning_counts": dict(sorted(warning_counts.items())),
        "warnings_by_speaker": warnings_by_speaker,
    }


def _hybrid_evidence_coverage(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    num_with_explanation = 0
    num_with_trait_match = 0
    num_promotion_eligible = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        assignment_targets = candidate.get("assignment_targets") if isinstance(candidate.get("assignment_targets"), dict) else {}
        speaker_ids_with_explanation: list[str] = []
        speaker_ids_with_trait_match: list[str] = []
        evidence_sources: set[str] = set()
        for speaker_id, target in assignment_targets.items():
            row = target if isinstance(target, dict) else {}
            if isinstance(row.get("query_trait_match"), dict):
                speaker_ids_with_trait_match.append(str(speaker_id))
            explanation = row.get("hybrid_search_explanation") if isinstance(row.get("hybrid_search_explanation"), dict) else None
            if explanation is not None:
                speaker_ids_with_explanation.append(str(speaker_id))
                evidence_sources.add(str(explanation.get("evidence_source") or "native_hybrid_search_explanation"))
        has_explanation = bool(speaker_ids_with_explanation)
        has_trait_match = bool(speaker_ids_with_trait_match)
        if has_explanation:
            num_with_explanation += 1
        if has_trait_match:
            num_with_trait_match += 1
        promotion_eligible = has_explanation or has_trait_match
        if promotion_eligible:
            num_promotion_eligible += 1
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "status": "hybrid_evidence_present" if promotion_eligible else "missing_hybrid_evidence",
                "promotion_eligible_by_hybrid_evidence": promotion_eligible,
                "num_assignment_targets": len(assignment_targets),
                "speaker_ids_with_explanation": speaker_ids_with_explanation,
                "speaker_ids_with_trait_match": speaker_ids_with_trait_match,
                "evidence_sources": sorted(evidence_sources),
                "suggested_action": (
                    "manual_review_allowed"
                    if promotion_eligible
                    else "treat_as_non_hybrid_baseline_or_regenerate_with_current_hybrid_assignment"
                ),
            }
        )
    total = len(rows)
    return {
        "status": "pass" if total and num_promotion_eligible == total else ("partial" if num_promotion_eligible else "missing"),
        "num_candidates": total,
        "num_with_hybrid_explanation": num_with_explanation,
        "num_with_trait_match": num_with_trait_match,
        "num_promotion_eligible_by_hybrid_evidence": num_promotion_eligible,
        "num_missing_hybrid_evidence": total - num_promotion_eligible,
        "coverage_rate": round(num_promotion_eligible / total, 4) if total else 0.0,
        "candidate_rows": rows,
        "promotion_policy": "approve_for_promotion requires hybrid trait evidence; missing-evidence candidates may be reviewed as baselines but are not promotable.",
        "claim_boundary": "Hybrid evidence coverage is an operational review-readiness gate only; it does not validate identity, privacy, clinical quality, or protected attributes.",
    }


def _aggregate_hybrid_warning_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    total_warnings = 0
    for row in candidates:
        summary = row.get("hybrid_warning_summary") if isinstance(row.get("hybrid_warning_summary"), dict) else {}
        status = str(summary.get("status") or "no_hybrid_explanation")
        status_counts[status] = status_counts.get(status, 0) + 1
        counts = summary.get("warning_counts") if isinstance(summary.get("warning_counts"), dict) else {}
        for warning, count in counts.items():
            try:
                numeric_count = int(count)
            except (TypeError, ValueError):
                continue
            warning_counts[str(warning)] = warning_counts.get(str(warning), 0) + numeric_count
            total_warnings += numeric_count
    return {
        "status_counts": dict(sorted(status_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "num_warnings": total_warnings,
    }


def _trait_match_summary(assignment_targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores: list[float] = []
    matched_fields: set[str] = set()
    equivalent_fields: set[str] = set()
    mismatched_fields: set[str] = set()
    missing_fields: set[str] = set()
    for row in assignment_targets.values():
        trait = row.get("query_trait_match") if isinstance(row.get("query_trait_match"), dict) else {}
        score = trait.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        matched = trait.get("matched") if isinstance(trait.get("matched"), dict) else {}
        equivalent = trait.get("equivalent") if isinstance(trait.get("equivalent"), dict) else {}
        mismatched = trait.get("mismatched") if isinstance(trait.get("mismatched"), dict) else {}
        missing = trait.get("missing") if isinstance(trait.get("missing"), list) else []
        matched_fields.update(str(key) for key in matched)
        equivalent_fields.update(str(key) for key in equivalent)
        mismatched_fields.update(str(key) for key in mismatched)
        missing_fields.update(str(key) for key in missing)
    return {
        "num_speakers_with_trait_match": len(scores),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        "min_score": round(min(scores), 4) if scores else None,
        "matched_fields": sorted(matched_fields),
        "equivalent_fields": sorted(equivalent_fields),
        "mismatched_fields": sorted(mismatched_fields),
        "missing_fields": sorted(missing_fields),
    }


def _candidate_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reviews = _review_by_candidate(run_dir)
    candidates_dir = run_dir / "candidates"
    for summary_path in sorted(candidates_dir.glob("*/candidate_summary.json")) if candidates_dir.exists() else []:
        summary = _load_json(summary_path)
        if not summary:
            continue
        candidate_id = str(summary.get("candidate_id") or summary_path.parent.name)
        artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
        audio_path = summary_path.parent / "changed_identity.wav"
        review = reviews.get(candidate_id)
        assignment_targets = _safe_assignment_targets(summary)
        if not assignment_targets:
            copied_assignment = _load_json(summary_path.parent / "speaker_target_map.json")
            if isinstance(copied_assignment.get("assignments"), dict):
                assignment_targets = _safe_assignment_targets({"assignment_targets": copied_assignment["assignments"]})
        hybrid_warning_summary = _hybrid_warning_summary(assignment_targets)
        rows.append(
            {
                "kind": "diarization_candidate",
                "candidate_id": candidate_id,
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
                "assignments_source_uri": summary.get("assignments_source_uri"),
                "assignment_evidence_refresh": summary.get("assignment_evidence_refresh")
                if isinstance(summary.get("assignment_evidence_refresh"), dict)
                else None,
                "assignment_evidence_refresh_history": summary.get("assignment_evidence_refresh_history")
                if isinstance(summary.get("assignment_evidence_refresh_history"), list)
                else [],
                "uri": artifacts.get("changed_identity_audio_uri"),
                "download_url": f"/download/diarization-candidate/{candidate_id}",
                "review_action_urls": {
                    "approve_for_promotion": f"/review/diarization-candidate/{candidate_id}/approve_for_promotion",
                    "needs_more_work": f"/review/diarization-candidate/{candidate_id}/needs_more_work",
                    "reject": f"/review/diarization-candidate/{candidate_id}/reject",
                },
                "manual_review_decision": review.get("decision") if review else None,
                "manual_review_criteria": review.get("criteria") if isinstance(review, dict) else None,
                "assignment_targets": assignment_targets,
                "trait_match_summary": _trait_match_summary(assignment_targets),
                "hybrid_warning_summary": hybrid_warning_summary,
                "audio": _file_metadata(audio_path),
                "claim_boundary": summary.get("claim_boundary"),
            }
        )
    return rows


def build_candidate_review_packet_from_run(run_dir: Path) -> dict[str, Any]:
    """Build a structured packet for side-by-side candidate listening."""

    canonical_path = run_dir / "final_sanitized.wav"
    candidates = _candidate_rows(run_dir)
    promoted_policy = build_promoted_output_policy_from_run(run_dir)
    canonical_uri = _canonical_uri(run_dir)
    review_summary = load_candidate_reviews(run_dir)
    return {
        "service": "candidate_review_packet",
        "schema_version": "aegisvox_candidate_review_packet.v1",
        "created_at": utc_now_iso(),
        "run_id": run_dir.name,
        "canonical_track": {
            "kind": "canonical",
            "label": "Canonical final_sanitized.wav",
            "uri": canonical_uri,
            "download_url": "/download/recommended",
            "remains_default": True,
            "audio": _file_metadata(canonical_path),
        },
        "candidate_tracks": candidates,
        "num_candidate_tracks": len(candidates),
        "hybrid_warning_summary": _aggregate_hybrid_warning_summary(candidates),
        "hybrid_evidence_coverage": _hybrid_evidence_coverage(candidates),
        "review_criteria": REVIEW_CRITERIA,
        "required_decision_values": ["approve_for_promotion", "needs_more_work", "reject"],
        "suggested_listening_order": ["canonical"] + [row["candidate_id"] for row in candidates],
        "manual_reviews": review_summary,
        "promoted_output_policy": promoted_policy,
        "promotion_ready": bool(promoted_policy.get("promoted_download_allowed")),
        "promoted_candidate_download_url": "/download/promoted-candidate"
        if promoted_policy.get("promoted_download_allowed")
        else None,
        "review_protocol": [
            "Listen to canonical first to establish baseline intelligibility and speaker count.",
            "Listen to each diarization candidate side-by-side, focusing on speaker preservation before audio polish.",
            "Mark overlap-converted improvements as review evidence, not validated overlap separation.",
            "Approve only one candidate for promotion follow-up after explicit listening; otherwise use needs_more_work or reject.",
            "Never overwrite final_sanitized.wav from this review packet.",
        ],
        "claim_boundary": (
            "This packet structures manual listening review for canonical and candidate outputs. It does not certify privacy, "
            "clinical readiness, or validated overlap separation, and it never mutates canonical final_sanitized.wav."
        ),
    }


def _md_value(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value).replace("\n", " ").strip()
    return text or "—"


def render_candidate_review_report_markdown(packet: dict[str, Any]) -> str:
    """Render a human-readable, audit-safe candidate review report."""

    canonical = packet.get("canonical_track") if isinstance(packet.get("canonical_track"), dict) else {}
    candidates = packet.get("candidate_tracks") if isinstance(packet.get("candidate_tracks"), list) else []
    manual_reviews = packet.get("manual_reviews") if isinstance(packet.get("manual_reviews"), dict) else {}
    promoted_policy = (
        packet.get("promoted_output_policy") if isinstance(packet.get("promoted_output_policy"), dict) else {}
    )
    lines: list[str] = [
        f"# AegisVox Candidate Review Report — {_md_value(packet.get('run_id'))}",
        "",
        "## Status",
        "",
        f"- Promotion ready: `{_md_value(packet.get('promotion_ready'))}`",
        f"- Promoted candidate download: `{_md_value(packet.get('promoted_candidate_download_url'))}`",
        f"- Latest reviews: `{_md_value(manual_reviews.get('num_reviews'))}`",
        f"- Hybrid warning summary: `{_md_value(json.dumps(packet.get('hybrid_warning_summary') or {}, sort_keys=True))}`",
        f"- Hybrid evidence coverage: `{_md_value(json.dumps(packet.get('hybrid_evidence_coverage') or {}, sort_keys=True))}`",
        f"- Append-only review events: `{_md_value(manual_reviews.get('num_history_events'))}`",
        f"- Canonical remains default: `{_md_value(canonical.get('remains_default'))}`",
        f"- Canonical download: `{_md_value(canonical.get('download_url'))}`",
        "",
        "## Listening order",
        "",
    ]
    for idx, item in enumerate(packet.get("suggested_listening_order") or [], start=1):
        lines.append(f"{idx}. `{_md_value(item)}`")
    lines.extend(["", "## Candidate tracks", ""])
    if candidates:
        lines.extend(
            [
                "| Candidate | Decision | Speakers | Segments | Overlap policy | Stitched | Download |",
                "|---|---:|---:|---:|---|---:|---|",
            ]
        )
        for row in candidates:
            if not isinstance(row, dict):
                continue
            stitched = f"{_md_value(row.get('num_stitched'))}/{_md_value(row.get('num_not_stitched'))}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{_md_value(row.get('candidate_id'))}`",
                        f"`{_md_value(row.get('manual_review_decision'))}`",
                        _md_value(row.get("num_clean_speakers")),
                        _md_value(row.get("num_clean_segments")),
                        f"`{_md_value(row.get('overlap_policy'))}`",
                        stitched,
                        f"`{_md_value(row.get('download_url'))}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No candidate tracks found.")
    lines.extend(["", "## Review criteria", ""])
    for criterion in packet.get("review_criteria") or []:
        if not isinstance(criterion, dict):
            continue
        lines.extend(
            [
                f"### {_md_value(criterion.get('label') or criterion.get('key'))}",
                f"- Key: `{_md_value(criterion.get('key'))}`",
                f"- Question: {_md_value(criterion.get('question'))}",
                f"- Promotion guidance: {_md_value(criterion.get('promotion_guidance'))}",
                "",
            ]
        )
    lines.extend(["## Review history", ""])
    history = manual_reviews.get("history") if isinstance(manual_reviews.get("history"), list) else []
    if history:
        lines.extend(
            [
                "| Time | Candidate | Decision | Speaker | Quality | Overlap | Background | Privacy |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in history:
            if not isinstance(row, dict):
                continue
            criteria = row.get("criteria") if isinstance(row.get("criteria"), dict) else {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_value(row.get("created_at")),
                        f"`{_md_value(row.get('candidate_id'))}`",
                        f"`{_md_value(row.get('decision'))}`",
                        f"`{_md_value(criteria.get('speaker_preservation'))}`",
                        f"`{_md_value(criteria.get('voice_quality'))}`",
                        f"`{_md_value(criteria.get('overlap_handling'))}`",
                        f"`{_md_value(criteria.get('background_or_noise'))}`",
                        f"`{_md_value(criteria.get('privacy_or_phi_risk'))}`",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No manual review events have been recorded yet.")
    lines.extend(
        [
            "",
            "## Promotion policy",
            "",
            f"- Promoted download allowed: `{_md_value(promoted_policy.get('promoted_download_allowed'))}`",
            f"- Approved reviews: `{_md_value(promoted_policy.get('num_approved_reviews'))}`",
            f"- Blocked approved reviews: `{_md_value(promoted_policy.get('num_blocked_approved_reviews'))}`",
            "",
            "### Reasons",
            "",
        ]
    )
    for reason in promoted_policy.get("reasons") or []:
        lines.append(f"- {_md_value(reason)}")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            _md_value(packet.get("claim_boundary")),
            "",
            "> This report is generated from run artifacts only. It does not certify clinical/privacy readiness and does not mutate `final_sanitized.wav`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_candidate_review_packet(run_dir: Path) -> dict[str, Any]:
    payload = build_candidate_review_packet_from_run(run_dir)
    out_path = run_dir / "candidate_review_packet.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (run_dir / "candidate_review_report.md").write_text(render_candidate_review_report_markdown(payload) + "\n")
    return payload
