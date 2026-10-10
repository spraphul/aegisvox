"""Background speech/PHI leakage checks for background-preserving mode."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .asr_alignment import ASRBackend, ASRAlignmentService, create_asr_backend
from .schema import validate_artifact_uri
from .storage import LocalArtifactStore
from .text_deid import DeIDPolicy, OpenAILLMTextDeIDBackend, TextDeIDBackend, _coerce_llm_detected_edits


def _background_leakage_llm_failure_policy() -> str:
    return os.environ.get("AEGISVOX_BACKGROUND_LEAKAGE_LLM_FAILURE_POLICY", "").strip().lower()


def _allow_asr_only_llm_failure_fallback() -> bool:
    return _background_leakage_llm_failure_policy() in {
        "asr_only",
        "asr_only_warn",
        "continue_asr_only",
        "conservative_asr_only",
    }


@dataclass
class BackgroundLeakageService:
    """Transcribe preserved background and summarize speech/PHI leakage risk.

    The report intentionally stores enough evidence for audit, but downstream UI
    should not display the raw background transcript by default. Evaluation uses
    the aggregate counts to decide whether the background can be remixed.
    """

    artifact_store: LocalArtifactStore
    backend: ASRBackend = field(default_factory=create_asr_backend)
    policy: DeIDPolicy = field(default_factory=lambda: DeIDPolicy(mode="typed_mask"))
    llm_backend: TextDeIDBackend = field(default_factory=OpenAILLMTextDeIDBackend)

    def check_to_artifact(
        self,
        *,
        background_audio_uri: str,
        output_report_uri: str,
    ) -> dict[str, Any]:
        validate_artifact_uri(background_audio_uri)
        validate_artifact_uri(output_report_uri)
        asr_payload = ASRAlignmentService(self.artifact_store, backend=self.backend).transcribe_to_artifact(
            audio_uri=background_audio_uri,
            output_asr_uri=output_report_uri.rsplit("/", 1)[0] + "/background_leakage_asr.json",
        )
        text = str(asr_payload.get("text") or "").strip()
        phi_detector_status = "completed"
        phi_detector_error: dict[str, str] | None = None
        llm_detector_used = True
        try:
            detected_rows = self.llm_backend.detect_edits(asr_payload, self.policy)
            edits = _coerce_llm_detected_edits(detected_rows, asr_payload, self.policy)
        except Exception as exc:
            if not _allow_asr_only_llm_failure_fallback():
                raise
            # This is deliberately not a deterministic PHI detector. It is a
            # conservative continuity mode for long internal/product batches
            # when the live LLM endpoint is unavailable: record ASR evidence,
            # report that PHI screening did not run, and let downstream
            # background policy duck/drop speech-like background rather than
            # silently accepting it as PHI-safe.
            detected_rows = []
            edits = []
            llm_detector_used = False
            phi_detector_status = "unavailable_asr_only_fallback"
            phi_detector_error = {
                "class": exc.__class__.__name__,
                "message": str(exc)[:500],
            }
        high_confidence = [edit for edit in edits if float(edit.confidence) >= 0.90]
        if phi_detector_status == "unavailable_asr_only_fallback":
            status = (
                "background_speech_detected_phi_detector_unavailable"
                if text
                else "no_background_speech_detected_phi_detector_unavailable"
            )
            claim_boundary = (
                "Background leakage ASR completed, but the live LLM HIPAA-PHI detector was unavailable. "
                "This artifact is an ASR-only conservative fallback for pipeline continuity and must not be "
                "used as evidence that background PHI screening passed."
            )
        else:
            status = (
                "background_phi_leakage_detected"
                if high_confidence
                else ("background_speech_detected" if text else "no_background_speech_detected")
            )
            claim_boundary = (
                "Background leakage check is an ASR/LLM-HIPAA-PHI screen over the preserved background stem. "
                "It is a safety gate for remix eligibility, not a clinical PHI recall benchmark."
            )
        report = {
            "service": "background_leakage_check",
            "status": status,
            "background_audio_uri": background_audio_uri,
            "asr_artifact_uri": output_report_uri.rsplit("/", 1)[0] + "/background_leakage_asr.json",
            "backend": self.backend.backend_name,
            "backend_version": self.backend.backend_version,
            "phi_detector_backend": self.llm_backend.backend_name,
            "phi_detector_backend_version": self.llm_backend.backend_version,
            "phi_detector_status": phi_detector_status,
            "phi_detector_error": phi_detector_error,
            "llm_detector_used": llm_detector_used,
            "regex_detector_used": False,
            "text_chars": len(text),
            "num_segments": len(asr_payload.get("segments") or []),
            "num_words": len(asr_payload.get("words") or []),
            "num_detected_phi": len(edits),
            "num_high_confidence_phi": len(high_confidence),
            "phi_types": sorted({edit.phi_type for edit in edits}),
            "high_confidence_phi_types": sorted({edit.phi_type for edit in high_confidence}),
            "edits": [edit.to_dict() for edit in edits],
            "claim_boundary": claim_boundary,
        }
        self.artifact_store.write_json(output_report_uri, report)
        return report
