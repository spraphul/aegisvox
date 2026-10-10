"""Plan-compliance reporting for AegisVox.

This module maps the user-approved service execution plan into explicit local
implementation/runtime evidence checks. It is intentionally conservative:
source files and tests can prove implementation coverage, but model-backed
runtime/full-stack evidence remains unverified until real ORD artifacts exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .audit import sanitize_for_audit
from .schema import utc_now_iso

EvidenceKind = Literal["code", "script", "test", "doc", "config", "schema", "runtime_artifact"]


@dataclass(frozen=True)
class EvidenceSpec:
    path: str
    kind: EvidenceKind
    description: str
    required: bool = True

    def check(self, root: Path) -> dict[str, Any]:
        target = root / self.path
        return {
            "path": self.path,
            "kind": self.kind,
            "description": self.description,
            "required": self.required,
            "exists": target.exists(),
        }


@dataclass(frozen=True)
class MilestoneSpec:
    milestone_id: int
    title: str
    goal: str
    deliverables: tuple[str, ...]
    implementation_evidence: tuple[EvidenceSpec, ...]
    runtime_evidence: tuple[EvidenceSpec, ...] = ()
    runtime_blockers: tuple[str, ...] = ()
    notes: str = ""

    def evaluate(self, root: Path) -> dict[str, Any]:
        implementation = [spec.check(root) for spec in self.implementation_evidence]
        runtime = [spec.check(root) for spec in self.runtime_evidence]
        required_impl = [row for row in implementation if row["required"]]
        required_runtime = [row for row in runtime if row["required"]]
        missing_impl = [row for row in required_impl if not row["exists"]]
        missing_runtime = [row for row in required_runtime if not row["exists"]]
        optional_missing = [row for row in implementation + runtime if not row["required"] and not row["exists"]]
        implementation_status = "complete" if not missing_impl else ("partial" if len(missing_impl) < len(required_impl) else "missing")
        if required_runtime:
            runtime_status = "verified" if not missing_runtime else "not_verified"
        else:
            runtime_status = "not_required_for_local_control_plane"
        if implementation_status == "complete" and runtime_status in {"verified", "not_required_for_local_control_plane"}:
            overall = "complete"
        elif implementation_status == "complete":
            overall = "implementation_complete_runtime_pending"
        elif implementation_status == "partial":
            overall = "partial"
        else:
            overall = "missing"
        return sanitize_for_audit(
            {
                "milestone_id": self.milestone_id,
                "title": self.title,
                "goal": self.goal,
                "deliverables": list(self.deliverables),
                "overall_status": overall,
                "implementation_status": implementation_status,
                "runtime_status": runtime_status,
                "implementation_evidence": implementation,
                "runtime_evidence": runtime,
                "missing_required_implementation": missing_impl,
                "missing_required_runtime": missing_runtime,
                "missing_optional": optional_missing,
                "runtime_blockers": list(self.runtime_blockers),
                "notes": self.notes,
            }
        )


def ev(path: str, kind: EvidenceKind, description: str, *, required: bool = True) -> EvidenceSpec:
    return EvidenceSpec(path=path, kind=kind, description=description, required=required)


MILESTONES: tuple[MilestoneSpec, ...] = (
    MilestoneSpec(
        milestone_id=0,
        title="Contracts and Infrastructure",
        goal="Create shared contracts, artifact store, metadata, queue/events, base service wiring, and local infra docs.",
        deliverables=("contracts package", "artifact store", "metadata schema", "queue/event abstraction", "Docker Compose stack", "audit/log format"),
        implementation_evidence=(
            ev("src/aegisvox/schema.py", "code", "Run/artifact/event/schema contracts"),
            ev("src/aegisvox/storage.py", "code", "Artifact store client"),
            ev("src/aegisvox/metadata.py", "code", "SQLite metadata store"),
            ev("src/aegisvox/metadata_postgres.py", "code", "Postgres metadata adapter"),
            ev("schemas/metadata_schema.sql", "schema", "Metadata migration schema"),
            ev("src/aegisvox/queue.py", "code", "Queue lifecycle facade"),
            ev("src/aegisvox/events.py", "code", "Event envelope/bus"),
            ev("src/aegisvox/audit.py", "code", "Audit redaction format"),
            ev("deployments/docker-compose.cpu.yaml", "config", "CPU Compose stack"),
            ev("tests/test_contracts.py", "test", "Contract tests"),
            ev("tests/test_metadata_and_api.py", "test", "Metadata/API tests"),
            ev("tests/test_queue.py", "test", "Queue lifecycle tests"),
        ),
        runtime_evidence=(
            ev("outputs/local_control/status_bundle.json", "runtime_artifact", "Local status bundle", required=False),
        ),
        notes="Compose runtime on ORD remains an infrastructure dependency; direct Python service path is supported.",
    ),
    MilestoneSpec(
        milestone_id=1,
        title="Voice Pool v1",
        goal="Build Open Swara-based voice pool, searchable metadata/speaker indexes, and retrieval API.",
        deliverables=("Open Swara voice pool", "normalized audio references", "QC manifest", "speaker embeddings", "text metadata embeddings", "FAISS index", "retrieval API"),
        implementation_evidence=(
            ev("src/aegisvox/openswara.py", "code", "Open Swara manifest/export helpers"),
            ev("src/aegisvox/voice_pool.py", "code", "Voice-pool contracts and retrieval"),
            ev("src/aegisvox/voice_pool_builder.py", "code", "Voice-pool builder/QC"),
            ev("src/aegisvox/voice_pool_faiss.py", "code", "FAISS index builder"),
            ev("src/aegisvox/services/voice_pool_builder.py", "script", "Voice-pool builder service wrapper"),
            ev("src/aegisvox/services/voice_pool_retrieval.py", "script", "Retrieval service wrapper"),
            ev("scripts/export_openswara_source_manifest.py", "script", "Open Swara source exporter"),
            ev("scripts/build_voice_pool.py", "script", "Voice-pool build CLI"),
            ev("scripts/build_voice_pool_faiss_index.py", "script", "FAISS build CLI"),
            ev("tests/test_voice_pool.py", "test", "Voice-pool retrieval tests"),
            ev("tests/test_voice_pool_builder.py", "test", "Builder/QC tests"),
            ev("tests/test_voice_pool_faiss.py", "test", "FAISS tests"),
        ),
        runtime_evidence=(
            ev("outputs/openswara_voicepool_validation/manifest.json", "runtime_artifact", "Validated Open Swara manifest"),
            ev("outputs/openswara_voicepool_validation/embedding_index.json", "runtime_artifact", "Validated embedding index"),
        ),
        runtime_blockers=("ORD auth is currently required to inspect/finish full Open Swara build.",),
    ),
    MilestoneSpec(
        milestone_id=2,
        title="Diarization and Segmentation",
        goal="Produce speaker turns and clean timeline-preserving segment WAVs.",
        deliverables=("diarization service", "segmentation service", "segments.cleaned.json", "segment WAV files"),
        implementation_evidence=(
            ev("src/aegisvox/diarization.py", "code", "pyannote-backed diarization adapter"),
            ev("src/aegisvox/segmentation.py", "code", "Segment cleanup/extraction"),
            ev("src/aegisvox/services/diarization.py", "script", "Diarization service wrapper"),
            ev("src/aegisvox/services/segmentation.py", "script", "Segmentation service wrapper"),
            ev("scripts/run_pyannote_diarization.py", "script", "Diarization CLI/preflight"),
            ev("scripts/run_segmentation_smoke.py", "script", "Segmentation smoke CLI"),
            ev("tests/test_diarization.py", "test", "Diarization fail-loud tests"),
            ev("tests/test_segmentation.py", "test", "Segmentation tests"),
        ),
        runtime_evidence=(
            ev("outputs/runtime_smokes/diarization.raw.json", "runtime_artifact", "Real pyannote diarization output"),
            ev("outputs/runtime_smokes/segments.cleaned.json", "runtime_artifact", "Extracted segment manifest"),
        ),
        runtime_blockers=("Requires HF/Pyannote token and pyannote runtime on ORD.",),
    ),
    MilestoneSpec(
        milestone_id=3,
        title="Speaker Profiling and Assignment",
        goal="Profile source speakers and assign unique, privacy-adjusted pseudo-speakers.",
        deliverables=("speaker_profiles.json", "speaker_target_map.json", "assignment_report.md"),
        implementation_evidence=(
            ev("src/aegisvox/speaker_profile.py", "code", "ECAPA profile backend"),
            ev("src/aegisvox/assignment.py", "code", "Privacy-adjusted assignment"),
            ev("src/aegisvox/services/speaker_profiling.py", "script", "Speaker profiling service wrapper"),
            ev("src/aegisvox/services/assignment.py", "script", "Assignment service wrapper"),
            ev("scripts/run_speaker_profiling_preflight.py", "script", "Speaker profiling preflight"),
            ev("scripts/run_assignment_smoke.py", "script", "Assignment smoke CLI"),
            ev("scripts/run_voice_pool_assignment_probe.py", "script", "Real voice-pool assignment probe"),
            ev("tests/test_speaker_profile.py", "test", "Speaker profile tests"),
            ev("tests/test_assignment.py", "test", "Assignment tests"),
            ev("tests/test_assignment_probe_script.py", "test", "Assignment probe script tests"),
        ),
        runtime_evidence=(
            ev("outputs/runtime_smokes/speaker_profiles.json", "runtime_artifact", "Real speaker profile output"),
            ev("outputs/runtime_smokes/speaker_target_map.json", "runtime_artifact", "Real target assignment output"),
        ),
        runtime_blockers=("Requires validated voice pool and SpeechBrain ECAPA runtime.",),
    ),
    MilestoneSpec(
        milestone_id=4,
        title="Single-Segment Voice Conversion",
        goal="Validate a real Seed-VC-backed conversion for one segment.",
        deliverables=("voice conversion service", "single segment conversion endpoint", "conversion_report.json"),
        implementation_evidence=(
            ev("src/aegisvox/voice_conversion.py", "code", "Seed-VC command backend"),
            ev("src/aegisvox/services/voice_conversion.py", "script", "Voice conversion service wrapper"),
            ev("scripts/run_seedvc_preflight.py", "script", "Seed-VC preflight CLI"),
            ev("tests/test_voice_conversion.py", "test", "Voice conversion fail-loud tests"),
        ),
        runtime_evidence=(
            ev("outputs/runtime_smokes/conversion_report.json", "runtime_artifact", "Real Seed-VC conversion report"),
        ),
        runtime_blockers=("Requires Seed-VC command/checkpoints on ORD; no mock output is allowed.",),
    ),
    MilestoneSpec(
        milestone_id=5,
        title="Batch Conversion and Stitching",
        goal="Convert all non-overlap segments and stitch a changed-identity timeline.",
        deliverables=("segments.converted.json", "converted segment WAVs", "changed_identity.wav", "stitch_report.json"),
        implementation_evidence=(
            ev("src/aegisvox/batch_conversion.py", "code", "Batch conversion orchestration"),
            ev("src/aegisvox/stitching.py", "code", "Timeline stitching/DSP"),
            ev("src/aegisvox/services/stitching.py", "script", "Stitching service wrapper"),
            ev("scripts/run_stitching_smoke.py", "script", "Stitching smoke CLI"),
            ev("tests/test_batch_conversion_and_stitching.py", "test", "Batch/stitching tests"),
        ),
        runtime_evidence=(
            ev("outputs/runtime_smokes/segments.converted.json", "runtime_artifact", "Converted segment manifest"),
            ev("outputs/runtime_smokes/changed_identity.wav", "runtime_artifact", "Changed-identity audio"),
        ),
        runtime_blockers=("Requires real converted segment outputs from Milestone 4/5.",),
    ),
    MilestoneSpec(
        milestone_id=6,
        title="ASR and Text De-ID",
        goal="Transcribe changed-identity audio and produce structured PHI edits.",
        deliverables=("asr.changed_identity.json", "deid_text.json", "edited_transcript.json"),
        implementation_evidence=(
            ev("src/aegisvox/asr_alignment.py", "code", "OpenAI ASR alignment backend"),
            ev("src/aegisvox/text_deid.py", "code", "Text de-id/validators"),
            ev("src/aegisvox/services/asr_alignment.py", "script", "ASR service wrapper"),
            ev("src/aegisvox/services/text_deid.py", "script", "Text de-id service wrapper"),
            ev("scripts/run_asr_preflight.py", "script", "ASR preflight CLI"),
            ev("scripts/run_text_deid_smoke.py", "script", "Text de-id smoke CLI"),
            ev("tests/test_asr_alignment.py", "test", "ASR tests"),
            ev("tests/test_text_deid.py", "test", "Text de-id tests"),
        ),
        runtime_evidence=(
            ev("outputs/runtime_smokes/asr.changed_identity.json", "runtime_artifact", "Real ASR output"),
            ev("outputs/runtime_smokes/deid_text.json", "runtime_artifact", "Structured text de-id output"),
        ),
        runtime_blockers=("Requires OpenAI ASR key/runtime and changed-identity audio.",),
    ),
    MilestoneSpec(
        milestone_id=7,
        title="Edited-Span Speech Re-Synthesis",
        goal="Regenerate edited transcript spans in pseudo-speaker voice.",
        deliverables=("voicecraft_edit_jobs.json", "replacement span WAVs", "final_sanitized.wav", "edit_integration_report.json"),
        implementation_evidence=(
            ev("src/aegisvox/speech_editing.py", "code", "VoiceCraft command backend and span patching"),
            ev("src/aegisvox/services/speech_editing.py", "script", "Speech editing service wrapper"),
            ev("scripts/run_voicecraft_preflight.py", "script", "VoiceCraft preflight CLI"),
            ev("scripts/run_speech_editing.py", "script", "Speech editing CLI"),
            ev("tests/test_speech_editing.py", "test", "Speech editing tests"),
        ),
        runtime_evidence=(
            ev("outputs/runtime_smokes/final_sanitized.wav", "runtime_artifact", "Final sanitized audio"),
        ),
        runtime_blockers=("Requires VoiceCraft command/checkpoints and edited spans; this is required for the full AegisVox pipeline, with disable flags allowed only for ablation/debug runs.",),
    ),
    MilestoneSpec(
        milestone_id=8,
        title="Evaluation and Safety Gates",
        goal="Generate privacy, utility, safety, and audit reports for every run.",
        deliverables=("evaluation_report.json", "evaluation_report.md", "pass_fail.json"),
        implementation_evidence=(
            ev("src/aegisvox/evaluation.py", "code", "Evaluation gates"),
            ev("src/aegisvox/privacy_metrics.py", "code", "Speaker privacy metrics"),
            ev("src/aegisvox/services/evaluation.py", "script", "Evaluation service wrapper"),
            ev("scripts/run_evaluation_smoke.py", "script", "Evaluation smoke CLI"),
            ev("scripts/run_speaker_privacy_metrics.py", "script", "Speaker privacy metrics CLI"),
            ev("tests/test_evaluation.py", "test", "Evaluation tests"),
            ev("tests/test_privacy_metrics.py", "test", "Privacy metrics tests"),
        ),
        runtime_evidence=(
            ev("outputs/runtime_smokes/evaluation_report.json", "runtime_artifact", "End-to-end evaluation report"),
            ev("outputs/runtime_smokes/pass_fail.json", "runtime_artifact", "End-to-end pass/fail report"),
        ),
        runtime_blockers=("Requires outputs from prior milestones; privacy metrics need real speaker embeddings.",),
    ),
    MilestoneSpec(
        milestone_id=9,
        title="Overlap Handling",
        goal="Extract, flag, and optionally process overlapped speech regions.",
        deliverables=("overlap_segments.json", "overlap converted stems", "overlap_remix_report.json"),
        implementation_evidence=(
            ev("src/aegisvox/overlap_handling.py", "code", "Overlap extraction/skip-and-flag/separation command adapter"),
            ev("src/aegisvox/services/overlap_handling.py", "script", "Overlap service wrapper"),
            ev("scripts/run_overlap_handling.py", "script", "Overlap CLI"),
            ev("tests/test_overlap_handling.py", "test", "Overlap tests"),
        ),
        runtime_evidence=(
            ev("outputs/runtime_smokes/overlap_segments.json", "runtime_artifact", "Overlap extraction output"),
            ev("outputs/runtime_smokes/overlap_remix_report.json", "runtime_artifact", "Production overlap remix report"),
        ),
        runtime_blockers=("Production overlap processing requires OVERLAP_SEPARATION_COMMAND; MVP skip-and-flag is implemented.",),
    ),
    MilestoneSpec(
        milestone_id=10,
        title="Product Hardening",
        goal="Prepare for production formalization: auth, secret discipline, logs/tracing, retries, health, model registry, metrics, runbooks.",
        deliverables=("service-level tests", "load tests", "audit logs", "security review notes", "model cards", "operational runbook"),
        implementation_evidence=(
            ev("src/aegisvox/auth.py", "code", "Optional fail-closed API auth"),
            ev("src/aegisvox/logging_utils.py", "code", "Structured JSONL audit logs"),
            ev("src/aegisvox/preflight.py", "code", "Preflight/provenance registry"),
            ev("src/aegisvox/status_bundle.py", "code", "Status bundle builder"),
            ev("src/aegisvox/worker_launch_plan.py", "code", "Worker launch-plan generator"),
            ev("scripts/run_worker_loop.py", "script", "Continuous worker loop with heartbeat"),
            ev("scripts/build_status_bundle.py", "script", "Status bundle CLI"),
            ev("scripts/build_worker_launch_plan.py", "script", "Worker launch-plan CLI"),
            ev("docs/ord_runbook.md", "doc", "Operational ORD runbook"),
            ev("docs/audit_and_redaction.md", "doc", "Audit/redaction policy"),
            ev("docs/api_auth.md", "doc", "API auth notes"),
            ev("docs/status_bundle.md", "doc", "Status bundle docs"),
            ev("tests/test_api_auth.py", "test", "API auth tests"),
            ev("tests/test_logging_utils.py", "test", "Structured logging tests"),
            ev("tests/test_preflight.py", "test", "Preflight tests"),
            ev("tests/test_status_bundle.py", "test", "Status bundle tests"),
            ev("tests/test_worker_launch_plan.py", "test", "Worker launch-plan tests"),
            ev("tests/test_worker_loop_script.py", "test", "Worker loop tests"),
        ),
        runtime_evidence=(
            ev("outputs/local_control/worker_launch_plan.json", "runtime_artifact", "Generated local worker launch plan", required=False),
            ev("outputs/local_control/status_bundle.json", "runtime_artifact", "Generated local status bundle", required=False),
        ),
        runtime_blockers=("Load tests, model cards, and security review notes remain future production formalization items.",),
    ),
)


def build_plan_compliance_report(*, root: str | Path) -> dict[str, Any]:
    """Evaluate current project evidence against the service execution milestones."""

    project_root = Path(root).expanduser().resolve()
    milestones = [milestone.evaluate(project_root) for milestone in MILESTONES]
    counts: dict[str, int] = {}
    for row in milestones:
        counts[row["overall_status"]] = counts.get(row["overall_status"], 0) + 1
    missing_required_impl = sum(len(row["missing_required_implementation"]) for row in milestones)
    missing_required_runtime = sum(len(row["missing_required_runtime"]) for row in milestones)
    report = {
        "report_type": "aegisvox_plan_compliance_report",
        "generated_at": utc_now_iso(),
        "project_root": str(project_root),
        "plan_source": "docs/aegisvox_service_execution_plan.md",
        "summary": {
            "num_milestones": len(milestones),
            "by_overall_status": dict(sorted(counts.items())),
            "missing_required_implementation_evidence": missing_required_impl,
            "missing_required_runtime_evidence": missing_required_runtime,
            "implementation_complete_milestones": [row["milestone_id"] for row in milestones if row["implementation_status"] == "complete"],
            "runtime_verified_milestones": [row["milestone_id"] for row in milestones if row["runtime_status"] in {"verified", "not_required_for_local_control_plane"}],
        },
        "milestones": milestones,
        "claim_boundary": (
            "This report verifies local evidence paths only. Implementation evidence does not prove real model execution, "
            "privacy, utility, PHI safety, or clinical readiness. Runtime claims require ORD artifacts from real backends."
        ),
    }
    return sanitize_for_audit(report)


def plan_compliance_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown compliance matrix."""

    summary = report.get("summary") or {}
    lines = [
        "# AegisVox plan compliance report",
        "",
        f"- Generated at: `{report.get('generated_at')}`",
        f"- Project root: `{report.get('project_root')}`",
        f"- Plan source: `{report.get('plan_source')}`",
        f"- Milestones: {summary.get('num_milestones')}",
        f"- Overall statuses: `{json.dumps(summary.get('by_overall_status') or {}, sort_keys=True)}`",
        "",
        "## Milestone matrix",
        "",
        "| ID | Milestone | Overall | Impl | Runtime | Missing impl | Missing runtime |",
        "|---:|---|---|---|---|---:|---:|",
    ]
    for milestone in report.get("milestones") or []:
        lines.append(
            "| {id} | {title} | `{overall}` | `{impl}` | `{runtime}` | {mi} | {mr} |".format(
                id=milestone.get("milestone_id"),
                title=str(milestone.get("title", "")).replace("|", "\\|"),
                overall=milestone.get("overall_status"),
                impl=milestone.get("implementation_status"),
                runtime=milestone.get("runtime_status"),
                mi=len(milestone.get("missing_required_implementation") or []),
                mr=len(milestone.get("missing_required_runtime") or []),
            )
        )
    lines.extend(["", "## Runtime blockers", ""])
    for milestone in report.get("milestones") or []:
        blockers = milestone.get("runtime_blockers") or []
        if blockers:
            lines.append(f"- **M{milestone.get('milestone_id')} {milestone.get('title')}**: " + "; ".join(map(str, blockers)))
    lines.extend(["", "## Claim boundary", "", str(report.get("claim_boundary"))])
    return "\n".join(lines) + "\n"
