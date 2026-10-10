"""Run-lifecycle transition helpers.

The orchestrator should not execute model inference, but it does need a small,
auditable mapping from completed lifecycle events to the next requested service.
This module keeps that DAG explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import AegisVoxContractError, BackgroundPreservationMode, RunManifest, RunRequest


COMPLETION_TO_NEXT_REQUEST: dict[str, str] = {
    "run.created": "diarization.requested",
    "source_separation.completed": "background_leakage_check.requested",
    "background_leakage_check.completed": "diarization.requested",
    "diarization.completed": "segmentation.requested",
    "segmentation.completed": "speaker_profile.requested",
    "speaker_profile.completed": "pseudo_assignment.requested",
    "pseudo_assignment.completed": "voice_conversion.requested",
    "voice_conversion.completed": "stitching.requested",
    "stitching.completed": "asr.requested",
    "asr.completed": "text_deid.requested",
    "text_deid.completed": "speech_edit.requested",
    "speech_edit.completed": "speaker_privacy_metrics.requested",
    "final_background_remix.completed": "speaker_privacy_metrics.requested",
    "speaker_privacy_metrics.completed": "evaluation.requested",
    "overlap_remix.completed": "evaluation.requested",
    "evaluation.completed": "run.completed",
}

REQUEST_EVENT_TO_SERVICE: dict[str, str] = {
    "source_separation.requested": "source_separation",
    "background_leakage_check.requested": "background_leakage_check",
    "diarization.requested": "diarization",
    "segmentation.requested": "segmentation",
    "speaker_profile.requested": "speaker_profiling",
    "pseudo_assignment.requested": "assignment",
    "voice_conversion.requested": "voice_conversion",
    "stitching.requested": "stitching",
    "asr.requested": "asr_alignment",
    "text_deid.requested": "text_deid",
    "speech_edit.requested": "speech_editing",
    "final_background_remix.requested": "final_background_remix",
    "speaker_privacy_metrics.requested": "speaker_privacy_metrics",
    "overlap_remix.requested": "overlap_remix",
    "evaluation.requested": "evaluation",
}

SERVICE_TO_REQUEST_EVENT: dict[str, str] = {service: event for event, service in REQUEST_EVENT_TO_SERVICE.items()}

REQUEST_EVENT_INPUT_ARTIFACT_NAMES: dict[str, list[str]] = {
    "source_separation.requested": [],
    "background_leakage_check.requested": ["preserved_background.wav"],
    "segmentation.requested": ["diarization.raw.json"],
    "speaker_profile.requested": ["segments.cleaned.json"],
    "pseudo_assignment.requested": ["speaker_profiles.json"],
    "voice_conversion.requested": ["segments.cleaned.json", "speaker_target_map.json"],
    "stitching.requested": ["segments.converted.json"],
    "asr.requested": ["changed_identity.wav"],
    "text_deid.requested": ["asr.changed_identity.json"],
    "speech_edit.requested": ["changed_identity.wav", "deid_text.json"],
    "final_background_remix.requested": [
        "final_sanitized.wav",
        "preserved_background.wav",
        "source_separation_report.json",
        "background_leakage_report.json",
    ],
    "speaker_privacy_metrics.requested": ["speaker_profiles.json", "speaker_target_map.json", "segments.converted.json"],
    "overlap_remix.requested": ["segments.cleaned.json", "speaker_target_map.json", "changed_identity.wav"],
}


@dataclass(frozen=True)
class PlannedTransition:
    from_event_type: str
    next_event_type: str
    target_service: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "from_event_type": self.from_event_type,
            "next_event_type": self.next_event_type,
            "target_service": self.target_service,
        }


def next_event_type(from_event_type: str, request: RunRequest) -> str:
    """Return the next canonical event for a completed lifecycle event.

    The production AegisVox plan requires lexical de-id and VoiceCraft speech
    editing. Feature flags remain available only for controlled ablation/debug
    runs:
    - If lexical de-id is explicitly disabled, ASR completion goes straight to evaluation.
    - If speech editing is explicitly disabled, text de-id completion goes straight to
      evaluation. The text edit artifact is still inspectable, but this path is not
      claim-ready for the full AegisVox pipeline.
    """

    if from_event_type == "run.created" and request.background_preservation_mode is BackgroundPreservationMode.SEPARATE_AND_REMIX:
        return "source_separation.requested"
    if from_event_type == "asr.completed" and not request.enable_lexical_deid:
        return "speaker_privacy_metrics.requested"
    if from_event_type == "text_deid.completed" and not request.enable_speech_editing:
        return "speaker_privacy_metrics.requested"
    if (
        from_event_type == "speech_edit.completed"
        and request.background_preservation_mode is BackgroundPreservationMode.SEPARATE_AND_REMIX
    ):
        return "final_background_remix.requested"
    if (
        from_event_type == "speaker_privacy_metrics.completed"
        and request.overlap_mode.value == "separate_convert_remix"
    ):
        return "overlap_remix.requested"
    try:
        return COMPLETION_TO_NEXT_REQUEST[from_event_type]
    except KeyError as exc:
        raise AegisVoxContractError(f"no workflow transition defined for event_type {from_event_type!r}") from exc


def plan_transition(from_event_type: str, request: RunRequest) -> PlannedTransition:
    next_event = next_event_type(from_event_type, request)
    return PlannedTransition(
        from_event_type=from_event_type,
        next_event_type=next_event,
        target_service=REQUEST_EVENT_TO_SERVICE.get(next_event),
    )


def request_event_type_for_service(service: str) -> str:
    """Return the canonical requested-event type for a service name."""

    try:
        return SERVICE_TO_REQUEST_EVENT[service]
    except KeyError as exc:
        raise AegisVoxContractError(f"no request event defined for service {service!r}") from exc


def input_artifact_uris_for_event(event_type: str, manifest: RunManifest) -> list[str]:
    """Resolve canonical input artifact URIs for a requested service event.

    The first diarization job consumes the registered input audio URI. Other
    jobs consume artifacts produced by earlier services. Missing required
    upstream artifacts fail loudly so the orchestrator does not enqueue
    un-runnable work.
    """

    if event_type == "source_separation.requested":
        return [manifest.request.input_audio_uri]
    if event_type == "diarization.requested":
        foreground = manifest.artifacts.get("foreground_vocals.wav")
        if foreground is not None:
            return [foreground.uri]
        return [manifest.request.input_audio_uri]
    if event_type == "evaluation.requested":
        return [artifact.uri for artifact in manifest.artifacts.values()]
    names = REQUEST_EVENT_INPUT_ARTIFACT_NAMES.get(event_type, [])
    uris: list[str] = []
    missing: list[str] = []
    for name in names:
        artifact = manifest.artifacts.get(name)
        if artifact is None:
            missing.append(name)
        else:
            uris.append(artifact.uri)
    if missing:
        raise AegisVoxContractError(
            f"cannot dispatch {event_type}: missing required upstream artifacts {missing}"
        )
    return uris
