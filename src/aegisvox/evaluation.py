"""Evaluation and safety gates for AegisVox pipeline artifacts.

Phase 9 is intentionally artifact-driven: it does not run inference or repair
missing outputs. It reads the artifacts produced by earlier services and emits a
machine-readable gate report plus a compact markdown summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore


class GateStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class EvaluationInputs:
    """Artifact URIs consumed by the evaluation service."""

    speaker_profiles_uri: str | None = None
    assignments_uri: str | None = None
    converted_segments_uri: str | None = None
    stitch_report_uri: str | None = None
    asr_uri: str | None = None
    deid_text_uri: str | None = None
    voicecraft_edit_jobs_uri: str | None = None
    final_sanitized_audio_uri: str | None = None
    speaker_privacy_metrics_uri: str | None = None
    changed_identity_audio_uri: str | None = None
    run_manifest_uri: str | None = None
    overlap_remix_apply_report_uri: str | None = None
    overlap_remixed_audio_uri: str | None = None
    background_leakage_report_uri: str | None = None
    final_sanitized_with_background_audio_uri: str | None = None
    final_background_remix_report_uri: str | None = None
    diarization_comparison_report_uri: str | None = None

    def required_items(self, policy: "EvaluationPolicy | None" = None) -> dict[str, str | None]:
        policy = policy or EvaluationPolicy()
        items = {
            "speaker_profiles": self.speaker_profiles_uri,
            "assignments": self.assignments_uri,
            "converted_segments": self.converted_segments_uri,
            "stitch_report": self.stitch_report_uri,
            "asr": self.asr_uri,
        }
        if policy.require_lexical_deid_artifacts:
            items["deid_text"] = self.deid_text_uri
        if policy.require_speech_editing_artifacts:
            items["voicecraft_edit_jobs"] = self.voicecraft_edit_jobs_uri
            items["final_sanitized_audio"] = self.final_sanitized_audio_uri
        elif self.final_sanitized_audio_uri is not None:
            items["final_sanitized_audio"] = self.final_sanitized_audio_uri
        return items

    def optional_items(self) -> dict[str, str | None]:
        return {
            "speaker_privacy_metrics": self.speaker_privacy_metrics_uri,
            "changed_identity_audio": self.changed_identity_audio_uri,
            "deid_text": self.deid_text_uri,
            "voicecraft_edit_jobs": self.voicecraft_edit_jobs_uri,
            "final_sanitized_audio": self.final_sanitized_audio_uri,
            "run_manifest": self.run_manifest_uri,
            "overlap_remix_apply_report": self.overlap_remix_apply_report_uri,
            "overlap_remixed_audio": self.overlap_remixed_audio_uri,
            "background_leakage_report": self.background_leakage_report_uri,
            "final_sanitized_with_background_audio": self.final_sanitized_with_background_audio_uri,
            "final_background_remix_report": self.final_background_remix_report_uri,
            "diarization_comparison_report": self.diarization_comparison_report_uri,
        }


@dataclass(frozen=True)
class EvaluationPolicy:
    """Thresholds for safety/utility gate decisions.

    Defaults are deliberately conservative for clinical de-identification: any
    unresolved high-confidence PHI is a failure, conversion failures above 5%
    fail, and substantial stitch drift fails. Overlap skipping is allowed only
    as a warning by default because the current MVP intentionally skips overlap
    regions; downstream product readiness should tighten this later.
    """

    max_conversion_failure_rate: float = 0.05
    max_overlap_skipped_rate_fail: float = 0.20
    max_overlap_skipped_rate_warn: float = 0.0
    max_abs_duration_drift_sec: float = 0.25
    max_unresolved_high_confidence_phi: int = 0
    require_asr_words: bool = False
    require_all_artifacts: bool = True
    require_speaker_privacy_metrics: bool = False
    require_lexical_deid_artifacts: bool = True
    require_speech_editing_artifacts: bool = True
    require_service_versions: bool = False
    max_source_converted_similarity: float = 0.65
    min_target_converted_similarity_warn: float = 0.25
    fail_on_known_gender_mismatch: bool = True
    fail_on_known_accent_mismatch: bool = False
    require_overlap_remix_for_skipped_overlaps: bool = False
    fail_on_incomplete_overlap_remix: bool = False
    fail_on_background_phi_leakage: bool = True
    fail_on_background_speech_leakage: bool = False
    max_background_text_chars_warn: int = 0
    warn_single_speaker_min_segments: int = 12
    warn_single_speaker_min_duration_sec: float = 45.0
    warn_diarization_comparison_extra_speakers: bool = True
    warn_diarization_comparison_profile_speaker_delta: int = 1


@dataclass(frozen=True)
class GateResult:
    name: str
    status: GateStatus
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "metrics": self.metrics,
            "evidence": self.evidence,
        }


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _load_optional(store: LocalArtifactStore, uri: str | None) -> dict[str, Any] | None:
    if uri is None:
        return None
    validate_artifact_uri(uri)
    payload = store.read_json(uri)
    if not isinstance(payload, dict):
        raise AegisVoxContractError(f"expected JSON object at {uri}")
    return payload


def _artifact_completeness_gate(store: LocalArtifactStore, inputs: EvaluationInputs, policy: EvaluationPolicy) -> GateResult:
    rows = []
    missing_required = []
    for name, uri in inputs.required_items(policy).items():
        exists = False
        reason = None
        if uri is None:
            reason = "uri_missing"
        else:
            try:
                validate_artifact_uri(uri)
                exists = store.exists(uri)
                if not exists:
                    reason = "artifact_missing"
            except Exception as exc:  # noqa: BLE001 - gate should summarize contract failures
                reason = f"invalid_uri:{exc}"
        row = {"artifact": name, "uri": uri, "exists": exists, "reason": reason}
        rows.append(row)
        if not exists:
            missing_required.append(row)
    if missing_required and policy.require_all_artifacts:
        return GateResult(
            "artifact_completeness",
            GateStatus.FAIL,
            f"{len(missing_required)} required artifacts are missing or invalid",
            {"num_missing": len(missing_required), "num_expected": len(rows)},
            rows,
        )
    if missing_required:
        return GateResult(
            "artifact_completeness",
            GateStatus.WARN,
            f"{len(missing_required)} optional artifacts are missing or invalid",
            {"num_missing": len(missing_required), "num_expected": len(rows)},
            rows,
        )
    return GateResult(
        "artifact_completeness",
        GateStatus.PASS,
        "all expected artifacts are present",
        {"num_missing": 0, "num_expected": len(rows)},
        rows,
    )


def _required_service_version_keys(policy: EvaluationPolicy) -> list[str]:
    keys = [
        "orchestrator",
        "diarization_service",
        "pyannote",
        "segmentation",
        "speaker_profiling",
        "speechbrain-ecapa",
        "assignment",
        "voice_pool",
        "voice_conversion",
        "seed-vc",
        "stitching",
        "asr_alignment",
        "openai-asr",
    ]
    if policy.require_lexical_deid_artifacts:
        keys.append("text_deid")
    if policy.require_speech_editing_artifacts:
        keys.extend(["speech_editing", "speech_editing_backend"])
    return keys


def _service_version_gate(manifest_payload: dict[str, Any] | None, policy: EvaluationPolicy) -> GateResult:
    required_keys = _required_service_version_keys(policy)
    if manifest_payload is None:
        status = GateStatus.FAIL if policy.require_service_versions else GateStatus.PASS
        return GateResult(
            "service_version_audit",
            status,
            "run manifest is unavailable for service/model version audit"
            if policy.require_service_versions
            else "run manifest unavailable; service/model version audit is informational",
            {"required": policy.require_service_versions, "manifest_available": False, "num_required_keys": len(required_keys)},
        )
    versions = manifest_payload.get("service_versions")
    if not isinstance(versions, dict):
        status = GateStatus.FAIL if policy.require_service_versions else GateStatus.WARN
        return GateResult(
            "service_version_audit",
            status,
            "run manifest service_versions must be an object",
            {"required": policy.require_service_versions, "manifest_available": True, "num_required_keys": len(required_keys)},
        )
    missing = [key for key in required_keys if not versions.get(key)]
    metrics = {
        "required": policy.require_service_versions,
        "manifest_available": True,
        "num_required_keys": len(required_keys),
        "num_recorded_versions": len(versions),
        "num_missing_required_versions": len(missing),
    }
    if missing:
        status = GateStatus.FAIL if policy.require_service_versions else GateStatus.WARN
        return GateResult(
            "service_version_audit",
            status,
            "service/model version provenance is incomplete",
            metrics,
            [{"missing_key": key} for key in missing],
        )
    return GateResult(
        "service_version_audit",
        GateStatus.PASS,
        "required service/model versions are recorded",
        metrics,
        [{"key": key, "version": str(versions[key])} for key in required_keys],
    )



def _speaker_count_sanity_gate(payload: dict[str, Any] | None, policy: EvaluationPolicy) -> GateResult:
    if payload is None:
        return GateResult("speaker_count_sanity", GateStatus.FAIL, "speaker profiles artifact is unavailable")
    speakers = payload.get("speakers") if isinstance(payload.get("speakers"), dict) else {}
    num_speakers = int(payload.get("num_speakers") or len(speakers))
    total_segments = 0
    total_duration = 0.0
    for profile in speakers.values():
        if not isinstance(profile, dict):
            continue
        metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
        try:
            total_segments += int(metadata.get("num_segments") or profile.get("num_segments") or 0)
        except (TypeError, ValueError):
            pass
        try:
            total_duration += float(profile.get("duration_sec") or profile.get("total_duration_sec") or 0.0)
        except (TypeError, ValueError):
            pass
    metrics = {
        "num_speakers": num_speakers,
        "total_segments": total_segments,
        "total_duration_sec": round(total_duration, 3),
        "warn_single_speaker_min_segments": policy.warn_single_speaker_min_segments,
        "warn_single_speaker_min_duration_sec": policy.warn_single_speaker_min_duration_sec,
    }
    if num_speakers <= 0:
        return GateResult("speaker_count_sanity", GateStatus.FAIL, "speaker profiling found no speakers", metrics)
    if (
        num_speakers == 1
        and total_segments >= policy.warn_single_speaker_min_segments
        and total_duration >= policy.warn_single_speaker_min_duration_sec
    ):
        return GateResult(
            "speaker_count_sanity",
            GateStatus.WARN,
            "long many-segment clip collapsed to one speaker; manually verify diarization and pseudo-speaker assignment",
            metrics,
            [
                {
                    "reason": "possible_speaker_collapse",
                    "claim_boundary": (
                        "This is a conservative QA warning, not proof that multiple speakers are present. "
                        "If human review hears multiple speakers, rerun diarization/segmentation or force speaker separation before accepting output."
                    ),
                }
            ],
        )
    return GateResult("speaker_count_sanity", GateStatus.PASS, "speaker count sanity check passed", metrics)


def _diarization_backend_disagreement_gate(
    comparison_payload: dict[str, Any] | None,
    speaker_profiles_payload: dict[str, Any] | None,
    policy: EvaluationPolicy,
) -> GateResult:
    if comparison_payload is None:
        return GateResult(
            "diarization_backend_disagreement",
            GateStatus.PASS,
            "no diarization backend comparison artifact provided; disagreement audit not run",
            {"comparison_available": False},
        )
    summaries = comparison_payload.get("summaries") if isinstance(comparison_payload.get("summaries"), list) else []
    profile_speakers = 0
    if isinstance(speaker_profiles_payload, dict):
        speakers = speaker_profiles_payload.get("speakers") if isinstance(speaker_profiles_payload.get("speakers"), dict) else {}
        profile_speakers = int(speaker_profiles_payload.get("num_speakers") or len(speakers))
    max_detected = int(comparison_payload.get("max_detected_speakers") or 0)
    best_variants = comparison_payload.get("best_variants") if isinstance(comparison_payload.get("best_variants"), list) else []
    max_total_duration = 0.0
    evidence: list[dict[str, Any]] = []
    for row in summaries:
        if not isinstance(row, dict):
            continue
        try:
            total_duration = float(row.get("total_speech_duration_sec") or 0.0)
        except (TypeError, ValueError):
            total_duration = 0.0
        max_total_duration = max(max_total_duration, total_duration)
        evidence.append(
            {
                "label": row.get("label"),
                "num_speaker_ids": row.get("num_speaker_ids"),
                "speaker_ids": row.get("speaker_ids"),
                "duration_by_speaker_sec": row.get("duration_by_speaker_sec"),
                "output_uri": row.get("output_uri"),
            }
        )
    delta = max_detected - profile_speakers
    metrics = {
        "comparison_available": True,
        "num_comparison_variants": len(summaries),
        "profile_num_speakers": profile_speakers,
        "max_detected_speakers": max_detected,
        "speaker_count_delta": delta,
        "best_variants": best_variants,
        "max_total_speech_duration_sec": round(max_total_duration, 3),
    }
    if (
        policy.warn_diarization_comparison_extra_speakers
        and delta >= policy.warn_diarization_comparison_profile_speaker_delta
    ):
        return GateResult(
            "diarization_backend_disagreement",
            GateStatus.WARN,
            "diarization backend comparison found more speakers than the profiled pipeline; route to manual diarization review or rerun candidate",
            metrics,
            evidence,
        )
    if max_detected <= 0:
        return GateResult(
            "diarization_backend_disagreement",
            GateStatus.WARN,
            "diarization backend comparison artifact is present but found no speakers",
            metrics,
            evidence,
        )
    return GateResult(
        "diarization_backend_disagreement",
        GateStatus.PASS,
        "diarization backend comparison is consistent with profiled speaker count",
        metrics,
        evidence,
    )


def _assignment_gate(payload: dict[str, Any] | None) -> GateResult:
    if payload is None:
        return GateResult("assignment_privacy", GateStatus.FAIL, "speaker_target_map artifact is unavailable")
    assignments = payload.get("assignments")
    policy = payload.get("policy") or {}
    if not isinstance(assignments, dict) or not assignments:
        return GateResult("assignment_privacy", GateStatus.FAIL, "assignments object is missing or empty")
    min_source_target = float(policy.get("min_source_target_distance", 0.0))
    min_target_target = float(policy.get("min_target_target_distance", 0.0))
    failed = []
    warnings = []
    distances = []
    target_voice_ids = []
    for speaker_id, row in assignments.items():
        distance = row.get("source_target_distance")
        status = row.get("status")
        target_voice_id = row.get("target_voice_id")
        target_voice_ids.append(target_voice_id)
        if distance is not None:
            distances.append(float(distance))
        if status != "assigned":
            failed.append({"speaker_id": speaker_id, "reason": f"status={status}"})
        elif distance is None:
            failed.append({"speaker_id": speaker_id, "reason": "missing_source_target_distance"})
        elif float(distance) < min_source_target:
            failed.append(
                {
                    "speaker_id": speaker_id,
                    "source_target_distance": float(distance),
                    "min_source_target_distance": min_source_target,
                }
            )
    duplicate_targets = len([x for x in target_voice_ids if x is not None]) - len(set(x for x in target_voice_ids if x is not None))
    pairwise_rows = payload.get("target_target_distances")
    pairwise_distances = []
    target_target_failures = []
    expected_pairs = max(0, len(assignments) * (len(assignments) - 1) // 2)
    if pairwise_rows is None:
        if expected_pairs:
            warnings.append({"reason": "missing_target_target_distances", "expected_pairs": expected_pairs})
    elif not isinstance(pairwise_rows, list):
        failed.append({"reason": "target_target_distances_malformed"})
    else:
        for row in pairwise_rows:
            if not isinstance(row, dict) or row.get("target_target_distance") is None:
                failed.append({"reason": "target_target_distance_row_malformed", "row": row})
                continue
            distance = float(row["target_target_distance"])
            pairwise_distances.append(distance)
            if distance < min_target_target:
                target_target_failures.append(
                    {
                        "left_target_sample_id": row.get("left_target_sample_id"),
                        "right_target_sample_id": row.get("right_target_sample_id"),
                        "target_target_distance": distance,
                        "min_target_target_distance": min_target_target,
                    }
                )
    metrics = {
        "num_speakers": len(assignments),
        "min_source_target_distance": min(distances) if distances else None,
        "policy_min_source_target_distance": min_source_target,
        "duplicate_target_voice_count": duplicate_targets,
        "num_failed_assignments": len(failed),
        "target_target_num_pairs": len(pairwise_distances),
        "target_target_expected_pairs": expected_pairs,
        "min_target_target_distance": min(pairwise_distances) if pairwise_distances else None,
        "policy_min_target_target_distance": min_target_target,
        "num_target_target_failures": len(target_target_failures),
    }
    if failed or duplicate_targets or target_target_failures:
        evidence = (
            failed
            + target_target_failures
            + ([{"reason": "duplicate_target_voice_ids", "count": duplicate_targets}] if duplicate_targets else [])
        )
        return GateResult("assignment_privacy", GateStatus.FAIL, "assignment privacy constraints are not satisfied", metrics, evidence)
    if warnings:
        return GateResult("assignment_privacy", GateStatus.WARN, "assignment passed source-target checks but target-target evidence is incomplete", metrics, warnings)
    return GateResult("assignment_privacy", GateStatus.PASS, "source-target and target-target privacy constraints are satisfied", metrics)




def _attribute_status(row: dict[str, Any], *, attribute: str) -> tuple[str, dict[str, Any]]:
    """Return normalized assignment-attribute status and compact evidence.

    Assignment artifacts written by newer runs include explicit match-status fields.
    Older artifacts may only have source/target values, so infer a mismatch when both
    sides are known and different. Unknown/not-applicable values should not become
    demographic claims or hard failures.
    """

    status_key = f"{attribute}_match_status"
    source_key = f"source_{attribute}"
    target_key = f"target_{attribute}"
    status = str(row.get(status_key) or "").strip().lower()
    source = str(row.get(source_key) or "unknown").strip().lower()
    target = str(row.get(target_key) or "unknown").strip().lower()
    evidence = {
        "speaker_id": row.get("speaker_id"),
        "target_voice_id": row.get("target_voice_id"),
        status_key: status or None,
        source_key: source,
        target_key: target,
    }
    unknown_values = {"", "unknown", "not_applicable", "none", "null"}
    if status in {"match", "mismatch"} or status.startswith("not_applicable"):
        return status, evidence
    if source in unknown_values:
        return "not_applicable_source_unknown", evidence
    if target in unknown_values:
        return "unknown_target", evidence
    if source == target:
        return "match", evidence
    return "mismatch", evidence


def _assignment_attribute_gate(payload: dict[str, Any] | None, policy: EvaluationPolicy) -> GateResult:
    if payload is None:
        return GateResult("assignment_attribute_match", GateStatus.FAIL, "speaker_target_map artifact is unavailable")
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        return GateResult("assignment_attribute_match", GateStatus.FAIL, "assignments object is missing or empty")

    gender_counts = {"match": 0, "mismatch": 0, "not_applicable": 0, "unknown_target": 0}
    accent_counts = {"match": 0, "mismatch": 0, "not_applicable": 0, "unknown_target": 0}
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for speaker_id, raw_row in assignments.items():
        row = dict(raw_row) if isinstance(raw_row, dict) else {}
        row.setdefault("speaker_id", speaker_id)
        gender_status, gender_evidence = _attribute_status(row, attribute="gender_presentation")
        accent_status, accent_evidence = _attribute_status(row, attribute="accent_group")

        if gender_status == "mismatch":
            gender_counts["mismatch"] += 1
            evidence = {**gender_evidence, "reason": "known_gender_presentation_mismatch"}
            if policy.fail_on_known_gender_mismatch:
                failures.append(evidence)
            else:
                warnings.append(evidence)
        elif gender_status == "match":
            gender_counts["match"] += 1
        elif gender_status == "unknown_target":
            gender_counts["unknown_target"] += 1
            warnings.append({**gender_evidence, "reason": "known_source_gender_target_unknown"})
        else:
            gender_counts["not_applicable"] += 1

        if accent_status == "mismatch":
            accent_counts["mismatch"] += 1
            evidence = {**accent_evidence, "reason": "known_accent_group_mismatch"}
            if policy.fail_on_known_accent_mismatch:
                failures.append(evidence)
            else:
                warnings.append(evidence)
        elif accent_status == "match":
            accent_counts["match"] += 1
        elif accent_status == "unknown_target":
            accent_counts["unknown_target"] += 1
            warnings.append({**accent_evidence, "reason": "known_source_accent_target_unknown"})
        else:
            accent_counts["not_applicable"] += 1

    metrics = {
        "num_speakers": len(assignments),
        "gender_matches": gender_counts["match"],
        "gender_mismatches": gender_counts["mismatch"],
        "gender_not_applicable": gender_counts["not_applicable"],
        "gender_unknown_target": gender_counts["unknown_target"],
        "accent_matches": accent_counts["match"],
        "accent_mismatches": accent_counts["mismatch"],
        "accent_not_applicable": accent_counts["not_applicable"],
        "accent_unknown_target": accent_counts["unknown_target"],
        "fail_on_known_gender_mismatch": policy.fail_on_known_gender_mismatch,
        "fail_on_known_accent_mismatch": policy.fail_on_known_accent_mismatch,
    }
    if failures:
        return GateResult("assignment_attribute_match", GateStatus.FAIL, "known source/target assignment attributes mismatched", metrics, failures[:20])
    if warnings:
        return GateResult("assignment_attribute_match", GateStatus.WARN, "assignment attribute preservation has warnings", metrics, warnings[:20])
    if gender_counts["match"] or accent_counts["match"]:
        return GateResult("assignment_attribute_match", GateStatus.PASS, "known assignment attributes are preserved", metrics)
    return GateResult("assignment_attribute_match", GateStatus.PASS, "no known assignment attributes required matching", metrics)

def _speaker_privacy_metrics_gate(payload: dict[str, Any] | None, policy: EvaluationPolicy) -> GateResult:
    if payload is None:
        status = GateStatus.FAIL if policy.require_speaker_privacy_metrics else GateStatus.WARN
        return GateResult(
            "speaker_similarity_privacy",
            status,
            "speaker privacy metrics artifact is unavailable",
            {"required": policy.require_speaker_privacy_metrics},
        )
    rows = payload.get("speaker_metrics")
    summary = payload.get("summary") or {}
    if not isinstance(rows, list) or not rows:
        return GateResult("speaker_similarity_privacy", GateStatus.FAIL, "speaker privacy metrics are missing or empty")
    failures = []
    warnings = []
    source_sims = []
    target_sims = []
    for row in rows:
        if not isinstance(row, dict):
            failures.append({"reason": "malformed_metric_row", "row": row})
            continue
        if row.get("status") != "computed":
            warnings.append({"speaker_id": row.get("speaker_id"), "reason": row.get("status") or "not_computed"})
            continue
        source_sim = row.get("source_converted_similarity")
        target_sim = row.get("target_converted_similarity")
        if source_sim is None or target_sim is None:
            failures.append({"speaker_id": row.get("speaker_id"), "reason": "missing_similarity"})
            continue
        source_sim = float(source_sim)
        target_sim = float(target_sim)
        source_sims.append(source_sim)
        target_sims.append(target_sim)
        if source_sim > policy.max_source_converted_similarity:
            failures.append(
                {
                    "speaker_id": row.get("speaker_id"),
                    "source_converted_similarity": source_sim,
                    "max_source_converted_similarity": policy.max_source_converted_similarity,
                    "reason": "source_linkability_too_high",
                }
            )
        if target_sim < policy.min_target_converted_similarity_warn:
            warnings.append(
                {
                    "speaker_id": row.get("speaker_id"),
                    "target_converted_similarity": target_sim,
                    "min_target_converted_similarity_warn": policy.min_target_converted_similarity_warn,
                    "reason": "target_similarity_low",
                }
            )
    metrics = {
        "num_speakers": int(summary.get("num_speakers") or len(rows)),
        "num_computed": int(summary.get("num_computed") or len(source_sims)),
        "max_source_converted_similarity": max(source_sims) if source_sims else summary.get("max_source_converted_similarity"),
        "min_target_converted_similarity": min(target_sims) if target_sims else summary.get("min_target_converted_similarity"),
        "max_source_converted_similarity_threshold": policy.max_source_converted_similarity,
        "min_target_converted_similarity_warn_threshold": policy.min_target_converted_similarity_warn,
        "num_source_linkability_flags": len([row for row in failures if row.get("reason") == "source_linkability_too_high"]),
        "num_target_similarity_warnings": len([row for row in warnings if row.get("reason") == "target_similarity_low"]),
    }
    if failures:
        return GateResult("speaker_similarity_privacy", GateStatus.FAIL, "speaker similarity privacy thresholds failed", metrics, failures[:20])
    if warnings:
        return GateResult("speaker_similarity_privacy", GateStatus.WARN, "speaker similarity metrics have warnings", metrics, warnings[:20])
    return GateResult("speaker_similarity_privacy", GateStatus.PASS, "speaker similarity privacy metrics are within thresholds", metrics)

def _conversion_gate(payload: dict[str, Any] | None, policy: EvaluationPolicy) -> GateResult:
    if payload is None:
        return GateResult("conversion_integrity", GateStatus.FAIL, "segments.converted artifact is unavailable")
    rows = payload.get("segments")
    if not isinstance(rows, list) or not rows:
        return GateResult("conversion_integrity", GateStatus.FAIL, "segments.converted.json must contain non-empty segments list")
    num_total = len(rows)
    num_converted = sum(1 for row in rows if row.get("conversion_status") == "converted")
    num_failed = sum(1 for row in rows if row.get("conversion_status") == "failed")
    num_skipped_overlap = sum(1 for row in rows if row.get("conversion_status") == "skipped_overlap")
    failure_rate = _safe_rate(num_failed, num_total)
    overlap_rate = _safe_rate(num_skipped_overlap, num_total)
    metrics = {
        "num_total": num_total,
        "num_converted": num_converted,
        "num_failed": num_failed,
        "num_skipped_overlap": num_skipped_overlap,
        "conversion_failure_rate": round(failure_rate, 6),
        "overlap_skipped_rate": round(overlap_rate, 6),
    }
    failed_rows = [
        {"segment_id": row.get("segment_id"), "failure_reason": row.get("failure_reason")}
        for row in rows
        if row.get("conversion_status") == "failed"
    ]
    if failure_rate > policy.max_conversion_failure_rate:
        return GateResult("conversion_integrity", GateStatus.FAIL, "conversion failure rate exceeds threshold", metrics, failed_rows[:20])
    if num_failed:
        return GateResult("conversion_integrity", GateStatus.WARN, "some conversion failures occurred but rate is below threshold", metrics, failed_rows[:20])
    return GateResult("conversion_integrity", GateStatus.PASS, "conversion completed within failure threshold", metrics)


def _skipped_overlap_segment_ids(converted_payload: dict[str, Any] | None) -> list[str]:
    rows = converted_payload.get("segments") if isinstance(converted_payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("segment_id"))
        for row in rows
        if isinstance(row, dict) and row.get("conversion_status") == "skipped_overlap" and row.get("segment_id") is not None
    ]


def _overlap_remix_handling_metrics(remix_payload: dict[str, Any] | None) -> dict[str, Any]:
    if remix_payload is None:
        return {
            "remix_report_available": False,
            "num_regions": None,
            "num_remixed_regions": 0,
            "num_protected_multi_speaker_regions": 0,
            "num_unresolved_regions": None,
            "num_handled_regions": 0,
            "remix_complete_or_protected": False,
        }
    status_text = str(remix_payload.get("status") or "")
    num_regions = int(remix_payload.get("num_regions") or 0)
    num_converted_vocals = int(remix_payload.get("num_converted_vocals") or 0)
    num_remixed = int(remix_payload.get("num_remixed_regions") or 0)
    num_protected = int(remix_payload.get("num_protected_multi_speaker_regions") or 0)
    if remix_payload.get("num_unresolved_regions") is None:
        num_unresolved = max(0, num_regions - num_remixed - num_protected)
    else:
        num_unresolved = int(remix_payload.get("num_unresolved_regions") or 0)
    handled = num_remixed + num_protected
    complete_statuses = {
        "completed_overlap_vocal_conversion_remix",
        "completed_overlap_vocal_conversion_remix_with_protected_multi_speaker_regions",
        "completed",
        "pass",
    }
    return {
        "remix_report_available": True,
        "status": status_text,
        "num_regions": num_regions,
        "num_converted_vocals": num_converted_vocals,
        "num_remixed_regions": num_remixed,
        "num_protected_multi_speaker_regions": num_protected,
        "num_unresolved_regions": num_unresolved,
        "num_handled_regions": handled,
        "output_audio_uri": remix_payload.get("output_audio_uri"),
        "remix_complete_or_protected": status_text in complete_statuses and num_regions > 0 and handled == num_regions and num_unresolved == 0,
        "has_protected_multi_speaker_regions": num_protected > 0,
    }


def _overlap_gate(
    payload: dict[str, Any] | None,
    policy: EvaluationPolicy,
    remix_payload: dict[str, Any] | None = None,
) -> GateResult:
    if payload is None:
        return GateResult("overlap_handling", GateStatus.FAIL, "segments.converted artifact is unavailable")
    rows = payload.get("segments")
    if not isinstance(rows, list) or not rows:
        return GateResult("overlap_handling", GateStatus.FAIL, "segments.converted.json must contain non-empty segments list")
    skipped_segment_ids = _skipped_overlap_segment_ids(payload)
    num_skipped = len(skipped_segment_ids)
    rate = _safe_rate(num_skipped, len(rows))
    remix_metrics = _overlap_remix_handling_metrics(remix_payload)
    remediated_or_protected = bool(remix_metrics["remix_complete_or_protected"]) and int(remix_metrics["num_handled_regions"] or 0) >= num_skipped
    metrics = {
        "num_segments": len(rows),
        "num_skipped_overlap": num_skipped,
        "overlap_skipped_rate": round(rate, 6),
        "posthoc_remix_or_protection_available": remix_metrics["remix_report_available"],
        "posthoc_num_handled_regions": remix_metrics["num_handled_regions"],
        "posthoc_num_remixed_regions": remix_metrics["num_remixed_regions"],
        "posthoc_num_protected_multi_speaker_regions": remix_metrics["num_protected_multi_speaker_regions"],
        "posthoc_num_unresolved_regions": remix_metrics["num_unresolved_regions"],
        "posthoc_complete_or_protected": remix_metrics["remix_complete_or_protected"],
    }
    evidence = [{"segment_id": segment_id} for segment_id in skipped_segment_ids[:20]]
    if num_skipped and remediated_or_protected:
        evidence.append(
            {
                "reason": "skipped_overlap_segments_have_posthoc_handling",
                "output_audio_uri": remix_metrics.get("output_audio_uri"),
                "has_protected_multi_speaker_regions": remix_metrics.get("has_protected_multi_speaker_regions"),
                "claim_boundary": (
                    "single-speaker overlaps may be remixed; multi-speaker overlaps are protected by preserving the "
                    "already de-identified changed_identity timeline and require manual listening review"
                ),
            }
        )
        return GateResult(
            "overlap_handling",
            GateStatus.WARN,
            "overlap skips were handled by post-hoc remix/protected multi-speaker policy; manual review required",
            metrics,
            evidence,
        )
    if rate > policy.max_overlap_skipped_rate_fail:
        return GateResult("overlap_handling", GateStatus.FAIL, "too many overlap regions were skipped", metrics, evidence)
    if rate > policy.max_overlap_skipped_rate_warn:
        return GateResult("overlap_handling", GateStatus.WARN, "overlap regions were skipped by current MVP policy", metrics, evidence)
    return GateResult("overlap_handling", GateStatus.PASS, "no overlap regions were skipped", metrics)


def _overlap_remix_coverage_gate(
    converted_payload: dict[str, Any] | None,
    remix_payload: dict[str, Any] | None,
    policy: EvaluationPolicy,
) -> GateResult:
    """Audit post-hoc overlap remediation/protection coverage without hiding warnings.

    The canonical conversion artifact can still record skipped overlap segments.
    A post-hoc overlap artifact can make those regions reviewable by either
    remixing single-speaker vocal/background stems or explicitly protecting
    multi-speaker overlap intervals by keeping the already de-identified
    changed-identity timeline. The latter is a product safety policy, not a
    claim of speaker-separated overlap editing.
    """

    if converted_payload is None:
        return GateResult("overlap_remix_coverage", GateStatus.FAIL, "segments.converted artifact is unavailable")
    rows = converted_payload.get("segments")
    if not isinstance(rows, list) or not rows:
        return GateResult("overlap_remix_coverage", GateStatus.FAIL, "segments.converted.json must contain non-empty segments list")
    skipped_segment_ids = _skipped_overlap_segment_ids(converted_payload)
    num_skipped = len(skipped_segment_ids)
    if num_skipped == 0:
        return GateResult(
            "overlap_remix_coverage",
            GateStatus.PASS,
            "no skipped overlap regions require post-hoc remix coverage",
            {"num_skipped_overlap": 0, "remix_report_available": remix_payload is not None},
        )

    if remix_payload is None:
        status = GateStatus.FAIL if policy.require_overlap_remix_for_skipped_overlaps else GateStatus.WARN
        return GateResult(
            "overlap_remix_coverage",
            status,
            "skipped overlap regions exist but no overlap-remix apply report was provided",
            {
                "num_skipped_overlap": num_skipped,
                "num_regions": None,
                "num_remixed_regions": 0,
                "num_protected_multi_speaker_regions": 0,
                "num_handled_regions": 0,
                "remix_report_available": False,
                "required": policy.require_overlap_remix_for_skipped_overlaps,
            },
            [{"segment_id": segment_id} for segment_id in skipped_segment_ids[:20]],
        )

    remix_metrics = _overlap_remix_handling_metrics(remix_payload)
    covers_skipped = int(remix_metrics["num_handled_regions"] or 0) >= num_skipped
    complete = bool(remix_metrics["remix_complete_or_protected"])
    metrics = {
        "num_skipped_overlap": num_skipped,
        **remix_metrics,
        "covers_all_skipped_overlaps": covers_skipped,
        "required": policy.require_overlap_remix_for_skipped_overlaps,
        "fail_on_incomplete_overlap_remix": policy.fail_on_incomplete_overlap_remix,
    }
    evidence = [
        {
            "reason": "overlap_remix_is_post_hoc_review_artifact",
            "claim_boundary": remix_payload.get("claim_boundary")
            or "post-hoc overlap remix/protection coverage does not replace clinical/privacy validation",
        }
    ]
    if complete and covers_skipped:
        status = GateStatus.WARN if int(remix_metrics["num_protected_multi_speaker_regions"] or 0) > 0 else GateStatus.PASS
        message = (
            "post-hoc overlap handling covers skipped regions with protected multi-speaker intervals"
            if status is GateStatus.WARN
            else "post-hoc overlap remix covers skipped overlap regions"
        )
        return GateResult(
            "overlap_remix_coverage",
            status,
            message,
            metrics,
            evidence,
        )
    status = GateStatus.FAIL if policy.fail_on_incomplete_overlap_remix or policy.require_overlap_remix_for_skipped_overlaps else GateStatus.WARN
    evidence.append(
        {
            "reason": "incomplete_overlap_remix_coverage",
            "skipped_segment_ids": skipped_segment_ids[:20],
        }
    )
    return GateResult(
        "overlap_remix_coverage",
        status,
        "post-hoc overlap remix/protection is incomplete for skipped overlap regions",
        metrics,
        evidence,
    )

def _stitch_gate(payload: dict[str, Any] | None, policy: EvaluationPolicy) -> GateResult:
    if payload is None:
        return GateResult("stitching_integrity", GateStatus.FAIL, "stitch_report artifact is unavailable")
    rows = payload.get("segments")
    if not isinstance(rows, list) or not rows:
        return GateResult("stitching_integrity", GateStatus.FAIL, "stitch_report.json must contain non-empty segments list")
    stitched = [row for row in rows if row.get("stitch_status") == "stitched"]
    not_stitched = [row for row in rows if row.get("stitch_status") != "stitched"]
    drift_rows = []
    drift_values = []
    for row in stitched:
        drift = row.get("duration_drift_sec")
        if drift is None:
            continue
        drift = float(drift)
        drift_values.append(abs(drift))
        if abs(drift) > policy.max_abs_duration_drift_sec:
            drift_rows.append({"segment_id": row.get("segment_id"), "duration_drift_sec": drift})
    explicit_excessive = payload.get("excessive_duration_drift") if isinstance(payload.get("excessive_duration_drift"), list) else []
    metrics = {
        "num_segments": len(rows),
        "num_stitched": len(stitched),
        "num_not_stitched": len(not_stitched),
        "max_abs_duration_drift_sec": max(drift_values) if drift_values else 0.0,
        "num_excessive_duration_drift": len(drift_rows) + len(explicit_excessive),
    }
    evidence = drift_rows + explicit_excessive[:20]
    if drift_rows or explicit_excessive:
        return GateResult("stitching_integrity", GateStatus.FAIL, "stitched audio has excessive duration drift", metrics, evidence)
    if not_stitched:
        return GateResult("stitching_integrity", GateStatus.WARN, "some segments were not stitched, usually skipped overlaps", metrics, not_stitched[:20])
    return GateResult("stitching_integrity", GateStatus.PASS, "stitched timeline is within drift threshold", metrics)


def _asr_gate(payload: dict[str, Any] | None, policy: EvaluationPolicy) -> GateResult:
    if payload is None:
        return GateResult("asr_availability", GateStatus.FAIL, "ASR artifact is unavailable")
    text = str(payload.get("text") or "").strip()
    segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    words = payload.get("words") if isinstance(payload.get("words"), list) else []
    metrics = {"text_chars": len(text), "num_segments": len(segments), "num_words": len(words)}
    if not text and not segments:
        return GateResult("asr_availability", GateStatus.FAIL, "ASR transcript has no text or segments", metrics)
    if policy.require_asr_words and not words:
        return GateResult("asr_availability", GateStatus.FAIL, "ASR word timestamps are required but missing", metrics)
    if not words:
        return GateResult("asr_availability", GateStatus.WARN, "ASR transcript exists but word timestamps are missing", metrics)
    return GateResult("asr_availability", GateStatus.PASS, "ASR transcript and word timestamps are available", metrics)


def _deid_gate(payload: dict[str, Any] | None, policy: EvaluationPolicy) -> GateResult:
    if payload is None:
        return GateResult("text_deid_safety", GateStatus.FAIL, "deid_text artifact is unavailable")
    summary = payload.get("validator_summary") if isinstance(payload.get("validator_summary"), dict) else {}
    unresolved = int(summary.get("unresolved_high_confidence_phi") or 0)
    edits = payload.get("edits") if isinstance(payload.get("edits"), list) else []
    deidentified_text = str(payload.get("deidentified_text") or "")
    metrics = {
        "num_edits": len(edits),
        "unresolved_high_confidence_phi": unresolved,
        "deidentified_text_chars": len(deidentified_text),
        "deterministic_edits": int(summary.get("deterministic_edits") or 0),
        "llm_edits": int(summary.get("llm_edits") or 0),
    }
    if unresolved > policy.max_unresolved_high_confidence_phi:
        return GateResult("text_deid_safety", GateStatus.FAIL, "unresolved high-confidence PHI exceeds threshold", metrics)
    return GateResult("text_deid_safety", GateStatus.PASS, "text de-identification validator summary is within threshold", metrics)


def _background_leakage_gate(payload: dict[str, Any] | None, policy: EvaluationPolicy) -> GateResult:
    if payload is None:
        return GateResult(
            "background_leakage_safety",
            GateStatus.PASS,
            "background preservation was not used or no background leakage report was provided",
            {"required": False, "report_available": False},
        )
    text_chars = int(payload.get("text_chars") or 0)
    high_conf_phi = int(payload.get("num_high_confidence_phi") or 0)
    detected_phi = int(payload.get("num_detected_phi") or 0)
    metrics = {
        "report_available": True,
        "status": payload.get("status"),
        "text_chars": text_chars,
        "num_segments": int(payload.get("num_segments") or 0),
        "num_words": int(payload.get("num_words") or 0),
        "num_detected_phi": detected_phi,
        "num_high_confidence_phi": high_conf_phi,
        "high_confidence_phi_types": payload.get("high_confidence_phi_types") or [],
        "fail_on_background_phi_leakage": policy.fail_on_background_phi_leakage,
        "fail_on_background_speech_leakage": policy.fail_on_background_speech_leakage,
        "max_background_text_chars_warn": policy.max_background_text_chars_warn,
    }
    evidence = [
        {
            "reason": "preserved_background_requires_leakage_audit",
            "background_audio_uri": payload.get("background_audio_uri"),
            "asr_artifact_uri": payload.get("asr_artifact_uri"),
            "claim_boundary": payload.get("claim_boundary"),
        }
    ]
    if high_conf_phi > 0:
        status = GateStatus.FAIL if policy.fail_on_background_phi_leakage else GateStatus.WARN
        return GateResult(
            "background_leakage_safety",
            status,
            "preserved background contains high-confidence PHI-like leakage",
            metrics,
            evidence,
        )
    if text_chars > policy.max_background_text_chars_warn:
        status = GateStatus.FAIL if policy.fail_on_background_speech_leakage else GateStatus.WARN
        return GateResult(
            "background_leakage_safety",
            status,
            "preserved background contains intelligible speech-like ASR text",
            metrics,
            evidence,
        )
    return GateResult("background_leakage_safety", GateStatus.PASS, "preserved background leakage screen passed", metrics)



def _background_remix_policy_gate(
    leakage_payload: dict[str, Any] | None, remix_payload: dict[str, Any] | None
) -> GateResult:
    if remix_payload is None:
        return GateResult(
            "background_remix_policy",
            GateStatus.PASS,
            "background-preserving remix was not produced for this run",
            {"required": False, "report_available": False},
        )
    action = str(remix_payload.get("background_policy_action") or "")
    try:
        effective_gain = float(remix_payload.get("effective_background_gain"))
    except (TypeError, ValueError):
        effective_gain = float("nan")
    requested_gain = remix_payload.get("requested_background_gain")
    leakage_summary = remix_payload.get("background_leakage_summary") if isinstance(remix_payload.get("background_leakage_summary"), dict) else {}
    high_conf_phi = int((leakage_payload or leakage_summary or {}).get("num_high_confidence_phi") or 0)
    text_chars = int((leakage_payload or leakage_summary or {}).get("text_chars") or 0)
    num_words = int((leakage_payload or leakage_summary or {}).get("num_words") or 0)
    metrics = {
        "report_available": True,
        "background_policy_action": action,
        "effective_background_gain": effective_gain,
        "requested_background_gain": requested_gain,
        "leakage_report_available": leakage_payload is not None or bool(leakage_summary),
        "num_high_confidence_phi": high_conf_phi,
        "text_chars": text_chars,
        "num_words": num_words,
    }
    evidence = [
        {
            "reason": "background_candidate_policy_audit",
            "output_audio_uri": remix_payload.get("output_audio_uri"),
            "background_leakage_report_uri": remix_payload.get("background_leakage_report_uri"),
            "claim_boundary": remix_payload.get("claim_boundary"),
        }
    ]
    if high_conf_phi > 0:
        if action == "drop_background_due_phi_leakage" and effective_gain <= 0.0:
            return GateResult(
                "background_remix_policy",
                GateStatus.PASS,
                "background remix dropped background after high-confidence PHI-like leakage",
                metrics,
                evidence,
            )
        return GateResult(
            "background_remix_policy",
            GateStatus.FAIL,
            "background remix did not drop background despite high-confidence PHI-like leakage",
            metrics,
            evidence,
        )
    if text_chars > 0 or num_words > 0:
        if action in {"duck_background_due_speech_leakage", "drop_background_due_speech_leakage"} and effective_gain <= 0.10:
            return GateResult(
                "background_remix_policy",
                GateStatus.PASS,
                "background remix mitigated speech-like background leakage by ducking or dropping",
                metrics,
                evidence,
            )
        return GateResult(
            "background_remix_policy",
            GateStatus.FAIL,
            "background remix preserved background despite speech-like leakage",
            metrics,
            evidence,
        )
    if action in {"preserve_background", "preserve_background_without_leakage_report"}:
        status = GateStatus.WARN if action == "preserve_background_without_leakage_report" else GateStatus.PASS
        message = (
            "background remix preserved background without a leakage report"
            if action == "preserve_background_without_leakage_report"
            else "background remix preserved clean background"
        )
        return GateResult("background_remix_policy", status, message, metrics, evidence)
    return GateResult(
        "background_remix_policy",
        GateStatus.WARN,
        "background remix policy action is not recognized for a clean/unknown background",
        metrics,
        evidence,
    )

def _speech_editing_gate(payload: dict[str, Any] | None) -> GateResult:
    if payload is None:
        return GateResult("speech_editing_integrity", GateStatus.FAIL, "voicecraft_edit_jobs artifact is unavailable")
    status = payload.get("status")
    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
    unlocalized = payload.get("unlocalized_edits") if isinstance(payload.get("unlocalized_edits"), list) else []
    failed_jobs = [job for job in jobs if job.get("status") == "failed"]
    metrics = {
        "status": status,
        "num_jobs": int(payload.get("num_jobs") or len(jobs)),
        "num_unlocalized_edits": int(payload.get("num_unlocalized_edits") or len(unlocalized)),
        "num_failed_jobs": int(payload.get("num_failed_jobs") or len(failed_jobs)),
        "num_fallback_jobs": int(payload.get("num_fallback_jobs") or 0),
    }
    if status in {"completed", "no_edits_passthrough"}:
        return GateResult("speech_editing_integrity", GateStatus.PASS, "speech editing completed or no lexical edits were required", metrics)
    if status == "completed_with_fallback" and not failed_jobs:
        return GateResult(
            "speech_editing_integrity",
            GateStatus.PASS,
            "speech editing completed with documented fallback repairs for failed/suspicious VoiceCraft spans",
            metrics,
            {"fallback_report": payload.get("fallback_report"), "fallback_renderer": payload.get("fallback_renderer")},
        )
    if unlocalized:
        return GateResult("speech_editing_integrity", GateStatus.FAIL, "speech editing has unlocalized edits", metrics, unlocalized[:20])
    if failed_jobs:
        return GateResult("speech_editing_integrity", GateStatus.FAIL, "speech editing jobs failed", metrics, failed_jobs[:20])
    if status == "planned":
        return GateResult("speech_editing_integrity", GateStatus.WARN, "speech editing jobs are planned but synthesis has not run", metrics)
    return GateResult("speech_editing_integrity", GateStatus.FAIL, "speech editing status is not complete", metrics)



GATE_SECTION_MAP = {
    "artifact_completeness": "audit",
    "service_version_audit": "audit",
    "assignment_privacy": "privacy",
    "speaker_count_sanity": "utility",
    "diarization_backend_disagreement": "utility",
    "assignment_attribute_match": "utility",
    "speaker_similarity_privacy": "privacy",
    "conversion_integrity": "utility",
    "overlap_handling": "utility",
    "overlap_remix_coverage": "utility",
    "stitching_integrity": "utility",
    "asr_availability": "utility",
    "text_deid_safety": "safety",
    "background_leakage_safety": "safety",
    "background_remix_policy": "safety",
    "speech_editing_integrity": "safety",
}


def build_report_sections(gates: list[GateResult]) -> dict[str, Any]:
    sections: dict[str, Any] = {
        "privacy": {"status": "pass", "gates": [], "summary": "speaker assignment and unlinkability checks"},
        "utility": {"status": "pass", "gates": [], "summary": "conversion, overlap, stitching, and ASR availability checks"},
        "safety": {"status": "pass", "gates": [], "summary": "lexical de-identification and speech-editing integrity checks"},
        "audit": {"status": "pass", "gates": [], "summary": "artifact completeness and pipeline audit checks"},
    }
    for gate in gates:
        section_name = GATE_SECTION_MAP.get(gate.name, "audit")
        sections[section_name]["gates"].append(gate.name)
    for section in sections.values():
        statuses = [gate.status for gate in gates if gate.name in section["gates"]]
        if any(status is GateStatus.FAIL for status in statuses):
            section["status"] = "fail"
        elif any(status is GateStatus.WARN for status in statuses):
            section["status"] = "warn"
    return sections

def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# AegisVox Evaluation Report",
        "",
        f"Overall status: **{report['overall_status'].upper()}**",
        "",
        "## Privacy / Utility / Safety Sections",
        "",
        "| Section | Status | Summary | Gates |",
        "|---|---:|---|---|",
    ]
    for name, section in (report.get("sections") or {}).items():
        gates = ", ".join(f"`{gate}`" for gate in section.get("gates", []))
        lines.append(f"| {name} | **{section.get('status')}** | {section.get('summary')} | {gates} |")
    lines.extend([
        "",
        "## Gates",
        "",
        "| Gate | Status | Message |",
        "|---|---:|---|",
    ])
    for gate in report["gates"]:
        lines.append(f"| `{gate['name']}` | **{gate['status']}** | {gate['message']} |")
    lines.extend(["", "## Service/model versions", ""])
    versions = report.get("service_versions") or {}
    if versions:
        for key, value in sorted(versions.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No service/model versions were available in the evaluated run manifest.")
    lines.extend(["", "## Key metrics", ""])
    for gate in report["gates"]:
        lines.append(f"### {gate['name']}")
        metrics = gate.get("metrics") or {}
        if metrics:
            for key, value in metrics.items():
                lines.append(f"- `{key}`: `{value}`")
        else:
            lines.append("- No metrics recorded.")
        lines.append("")
    lines.extend(
        [
            "## Claim boundary",
            "",
            "This report is a pipeline safety/evaluation gate over existing artifacts. It is not a substitute for real model dependency smokes, human clinical review, PHI recall benchmarking, or adversarial privacy evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass
class EvaluationService:
    artifact_store: LocalArtifactStore
    policy: EvaluationPolicy = field(default_factory=EvaluationPolicy)

    def evaluate_to_artifacts(
        self,
        *,
        inputs: EvaluationInputs,
        output_report_uri: str,
        output_markdown_uri: str,
        output_pass_fail_uri: str,
    ) -> dict[str, Any]:
        validate_artifact_uri(output_report_uri)
        validate_artifact_uri(output_markdown_uri)
        validate_artifact_uri(output_pass_fail_uri)
        completeness = _artifact_completeness_gate(self.artifact_store, inputs, self.policy)
        artifacts_available = completeness.status is GateStatus.PASS or not self.policy.require_all_artifacts
        payloads: dict[str, dict[str, Any] | None] = {}
        if artifacts_available:
            payloads = {
                "speaker_profiles": _load_optional(self.artifact_store, inputs.speaker_profiles_uri),
                "assignments": _load_optional(self.artifact_store, inputs.assignments_uri),
                "converted_segments": _load_optional(self.artifact_store, inputs.converted_segments_uri),
                "stitch_report": _load_optional(self.artifact_store, inputs.stitch_report_uri),
                "asr": _load_optional(self.artifact_store, inputs.asr_uri),
                "deid_text": _load_optional(self.artifact_store, inputs.deid_text_uri),
                "voicecraft_edit_jobs": _load_optional(self.artifact_store, inputs.voicecraft_edit_jobs_uri),
                "speaker_privacy_metrics": _load_optional(self.artifact_store, inputs.speaker_privacy_metrics_uri),
                "run_manifest": _load_optional(self.artifact_store, inputs.run_manifest_uri),
                "overlap_remix_apply_report": _load_optional(self.artifact_store, inputs.overlap_remix_apply_report_uri),
                "background_leakage_report": _load_optional(self.artifact_store, inputs.background_leakage_report_uri),
                "final_background_remix_report": _load_optional(self.artifact_store, inputs.final_background_remix_report_uri),
                "diarization_comparison_report": _load_optional(self.artifact_store, inputs.diarization_comparison_report_uri),
            }
        deid_gate = (
            _deid_gate(payloads.get("deid_text"), self.policy)
            if self.policy.require_lexical_deid_artifacts or payloads.get("deid_text") is not None
            else GateResult(
                "text_deid_safety",
                GateStatus.PASS,
                "lexical de-identification was disabled for this run",
                {"required": False},
            )
        )
        speech_edit_gate = (
            _speech_editing_gate(payloads.get("voicecraft_edit_jobs"))
            if self.policy.require_speech_editing_artifacts or payloads.get("voicecraft_edit_jobs") is not None
            else GateResult(
                "speech_editing_integrity",
                GateStatus.PASS,
                "speech editing was disabled for this run",
                {"required": False},
            )
        )
        gates = [
            completeness,
            _service_version_gate(payloads.get("run_manifest"), self.policy),
            _assignment_gate(payloads.get("assignments")),
            _speaker_count_sanity_gate(payloads.get("speaker_profiles"), self.policy),
            _diarization_backend_disagreement_gate(
                payloads.get("diarization_comparison_report"),
                payloads.get("speaker_profiles"),
                self.policy,
            ),
            _assignment_attribute_gate(payloads.get("assignments"), self.policy),
            _speaker_privacy_metrics_gate(payloads.get("speaker_privacy_metrics"), self.policy),
            _conversion_gate(payloads.get("converted_segments"), self.policy),
            _overlap_gate(payloads.get("converted_segments"), self.policy),
            _overlap_remix_coverage_gate(
                payloads.get("converted_segments"),
                payloads.get("overlap_remix_apply_report"),
                self.policy,
            ),
            _stitch_gate(payloads.get("stitch_report"), self.policy),
            _asr_gate(payloads.get("asr"), self.policy),
            deid_gate,
            _background_leakage_gate(payloads.get("background_leakage_report"), self.policy),
            _background_remix_policy_gate(
                payloads.get("background_leakage_report"),
                payloads.get("final_background_remix_report"),
            ),
            speech_edit_gate,
        ]
        failed = [gate for gate in gates if gate.status is GateStatus.FAIL]
        warned = [gate for gate in gates if gate.status is GateStatus.WARN]
        overall_status = "fail" if failed else ("warn" if warned else "pass")
        sections = build_report_sections(gates)
        service_versions = {}
        manifest_payload = payloads.get("run_manifest")
        if isinstance(manifest_payload, dict) and isinstance(manifest_payload.get("service_versions"), dict):
            service_versions = {str(key): str(value) for key, value in manifest_payload["service_versions"].items()}
        report = {
            "service": "evaluation",
            "overall_status": overall_status,
            "overall_pass": not failed,
            "sections": sections,
            "inputs": {"required": inputs.required_items(self.policy), "optional": inputs.optional_items()},
            "service_versions": service_versions,
            "policy": {
                "max_conversion_failure_rate": self.policy.max_conversion_failure_rate,
                "max_overlap_skipped_rate_fail": self.policy.max_overlap_skipped_rate_fail,
                "max_overlap_skipped_rate_warn": self.policy.max_overlap_skipped_rate_warn,
                "max_abs_duration_drift_sec": self.policy.max_abs_duration_drift_sec,
                "max_unresolved_high_confidence_phi": self.policy.max_unresolved_high_confidence_phi,
                "require_asr_words": self.policy.require_asr_words,
                "require_all_artifacts": self.policy.require_all_artifacts,
                "require_speaker_privacy_metrics": self.policy.require_speaker_privacy_metrics,
                "require_lexical_deid_artifacts": self.policy.require_lexical_deid_artifacts,
                "require_speech_editing_artifacts": self.policy.require_speech_editing_artifacts,
                "require_service_versions": self.policy.require_service_versions,
                "max_source_converted_similarity": self.policy.max_source_converted_similarity,
                "min_target_converted_similarity_warn": self.policy.min_target_converted_similarity_warn,
                "fail_on_known_gender_mismatch": self.policy.fail_on_known_gender_mismatch,
                "fail_on_known_accent_mismatch": self.policy.fail_on_known_accent_mismatch,
                "require_overlap_remix_for_skipped_overlaps": self.policy.require_overlap_remix_for_skipped_overlaps,
                "fail_on_incomplete_overlap_remix": self.policy.fail_on_incomplete_overlap_remix,
                "fail_on_background_phi_leakage": self.policy.fail_on_background_phi_leakage,
                "fail_on_background_speech_leakage": self.policy.fail_on_background_speech_leakage,
                "max_background_text_chars_warn": self.policy.max_background_text_chars_warn,
                "warn_diarization_comparison_extra_speakers": self.policy.warn_diarization_comparison_extra_speakers,
                "warn_diarization_comparison_profile_speaker_delta": self.policy.warn_diarization_comparison_profile_speaker_delta,
            },
            "gates": [gate.to_dict() for gate in gates],
            "claim_boundary": "artifact-level pipeline safety gate; real dependency smokes and clinical validation still required",
        }
        pass_fail = {
            "overall_pass": report["overall_pass"],
            "overall_status": overall_status,
            "failed_gates": [gate.name for gate in failed],
            "warned_gates": [gate.name for gate in warned],
        }
        self.artifact_store.write_json(output_report_uri, report)
        self.artifact_store.write_bytes(output_markdown_uri, build_markdown_report(report).encode("utf-8") + b"\n")
        self.artifact_store.write_json(output_pass_fail_uri, pass_fail)
        return report
