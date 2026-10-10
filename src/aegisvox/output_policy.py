"""Output-selection policy for canonical vs candidate AegisVox audio artifacts.

This module is intentionally artifact-only. It does not run inference and does
not repair unsafe outputs. It summarizes product semantics after Phase-9 gates:
``final_sanitized.wav`` remains canonical unless a future stricter product
policy explicitly promotes a candidate. Background-preserved audio is classified
as a candidate with reasons derived from leakage and remix-policy gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CANONICAL_AUDIO_NAME = "final_sanitized.wav"
BACKGROUND_CANDIDATE_AUDIO_NAME = "final_sanitized_with_background.wav"


@dataclass(frozen=True)
class OutputPolicyInputs:
    canonical_audio_uri: str | None = None
    background_candidate_audio_uri: str | None = None
    evaluation_report: dict[str, Any] | None = None
    pass_fail: dict[str, Any] | None = None
    background_leakage_report: dict[str, Any] | None = None
    final_background_remix_report: dict[str, Any] | None = None


def _gate_status(report: dict[str, Any] | None, gate_name: str) -> str | None:
    gates = report.get("gates") if isinstance(report, dict) else None
    if not isinstance(gates, list):
        return None
    for gate in gates:
        if isinstance(gate, dict) and gate.get("name") == gate_name:
            return str(gate.get("status") or "unknown")
    return None


def _safe_int(payload: dict[str, Any] | None, key: str) -> int:
    if not isinstance(payload, dict):
        return 0
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def build_output_policy(inputs: OutputPolicyInputs) -> dict[str, Any]:
    """Return an auditable product decision for output artifacts.

    The policy is conservative: it never promotes background-preserved audio to
    canonical. Instead, it classifies it as absent/clean-candidate/
    mitigated-candidate/rejected-candidate and records the exact reason.
    """

    evaluation = inputs.evaluation_report or {}
    pass_fail = inputs.pass_fail or {}
    leakage = inputs.background_leakage_report or {}
    remix = inputs.final_background_remix_report or {}
    leakage_gate = _gate_status(evaluation, "background_leakage_safety")
    remix_gate = _gate_status(evaluation, "background_remix_policy")
    high_conf_phi = _safe_int(leakage or remix.get("background_leakage_summary"), "num_high_confidence_phi")
    text_chars = _safe_int(leakage or remix.get("background_leakage_summary"), "text_chars")
    num_words = _safe_int(leakage or remix.get("background_leakage_summary"), "num_words")
    action = str(remix.get("background_policy_action") or "unavailable")
    candidate_uri = inputs.background_candidate_audio_uri or remix.get("output_audio_uri")

    reasons: list[str] = []
    candidate_status = "unavailable"
    if not candidate_uri or not remix:
        reasons.append("background-preserved candidate was not produced")
    elif remix_gate == "fail":
        candidate_status = "rejected_candidate"
        reasons.append("background remix policy gate failed")
    elif high_conf_phi > 0:
        if action == "drop_background_due_phi_leakage":
            candidate_status = "mitigated_candidate"
            reasons.append("background contained high-confidence PHI-like leakage; background was dropped")
        else:
            candidate_status = "rejected_candidate"
            reasons.append("background contained high-confidence PHI-like leakage without required drop action")
    elif text_chars > 0 or num_words > 0 or leakage_gate in {"warn", "fail"}:
        if action in {"duck_background_due_speech_leakage", "drop_background_due_speech_leakage"}:
            candidate_status = "mitigated_candidate"
            reasons.append("background contained speech-like leakage; background was ducked/dropped")
        else:
            candidate_status = "rejected_candidate"
            reasons.append("background contained speech-like leakage without mitigation")
    else:
        candidate_status = "clean_candidate"
        reasons.append("background leakage screen found no speech/PHI-like content")

    canonical_uri = inputs.canonical_audio_uri
    canonical_status = "available" if canonical_uri else "missing"
    if pass_fail.get("overall_pass") is False:
        reasons.append("overall evaluation has failed gates; inspect pass_fail before product use")

    return {
        "service": "output_policy",
        "canonical": {
            "name": CANONICAL_AUDIO_NAME,
            "uri": canonical_uri,
            "status": canonical_status,
            "decision": "canonical_output",
            "reason": "canonical output excludes preserved background candidate by default",
        },
        "background_candidate": {
            "name": BACKGROUND_CANDIDATE_AUDIO_NAME,
            "uri": candidate_uri,
            "status": candidate_status,
            "background_policy_action": action,
            "background_leakage_gate": leakage_gate,
            "background_remix_policy_gate": remix_gate,
            "effective_background_gain": remix.get("effective_background_gain"),
            "num_high_confidence_phi": high_conf_phi,
            "text_chars": text_chars,
            "num_words": num_words,
        },
        "recommended_for_download": canonical_uri,
        "candidate_allowed_for_review": candidate_status in {"clean_candidate", "mitigated_candidate"},
        "reasons": reasons,
        "claim_boundary": (
            "Only canonical output should be treated as the safe product default. Background-preserved audio is a review "
            "candidate unless stronger leakage, privacy, and utility gates explicitly promote it in a future policy."
        ),
    }
