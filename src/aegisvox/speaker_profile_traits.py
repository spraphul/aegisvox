"""Source-speaker audio trait enrichment for hybrid pseudo-speaker search.

This module is the query-side companion to ``enrich_voice_pool_audio_traits``:
it runs the real audio model on each extracted source-speaker profile audio and
persists operational matching traits under ``metadata.audio_traits``.  The
traits are used only to improve voice-pool retrieval constraints such as broad
accent, language, tone, energy, rate, pitch band, and gender-presentation
guardrails.  They are not identity, demographic, clinical, or precise geography
claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_trait_extraction import AudioTraitExtractionError, extract_audio_traits_with_openai
from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore
from .voice_pool import _normalize_trait_value


class SpeakerProfileTraitEnrichmentError(RuntimeError):
    """Raised when source-speaker trait enrichment cannot be completed."""


PROFILE_AUDIO_TRAIT_KEYS = [
    "language",
    "accent_group",
    "geography_region",
    "tone",
    "energy",
    "speaking_rate",
    "pitch_band",
    "gender_presentation",
    "vocal_weight",
    "resonance",
    "voice_maturity",
]


@dataclass(frozen=True)
class SpeakerProfileTraitEnrichmentPolicy:
    model: str = "gpt-audio-1.5"
    overwrite_existing: bool = False
    continue_on_error: bool = False
    promote_to_top_level_when_unknown: bool = True


def _known_trait(value: Any) -> str | None:
    normalized = _normalize_trait_value(value)
    if normalized and normalized != "unknown":
        return normalized
    return None


def _has_existing_model_traits(profile: dict[str, Any]) -> bool:
    metadata = profile.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return isinstance(metadata.get("audio_trait_extraction"), dict) or bool(metadata.get("audio_traits"))


def _apply_source_gender_guardrail(
    *,
    speaker_id: str,
    profile: dict[str, Any],
    traits: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve source-profile gender conflicts conservatively for matching.

    Catalog/manual source gender should remain a hard guardrail.  The top-level
    source profile may also come from the lightweight F0 heuristic, and that can
    be wrong for older, breathy, high-pitch male voices or low-pitch female
    voices.  When the source is explicitly marked as the F0 heuristic and the
    audio model disagrees, prefer the audio-model trait for assignment and audit
    the conflict instead of forcing an obvious male<->female pseudo-voice flip.
    """

    source_gender = _known_trait(profile.get("gender_presentation"))
    model_gender = _known_trait(traits.get("gender_presentation"))
    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
    source_method = str(metadata.get("gender_presentation_source") or "").lower()
    if source_gender in {"female", "male"} and model_gender in {"female", "male"} and model_gender != source_gender:
        if "f0" in source_method or "heuristic" in source_method:
            return {
                "type": "source_profile_gender_conflict_model_preferred",
                "speaker_id": speaker_id,
                "source_gender_presentation": source_gender,
                "model_reported_gender_presentation": model_gender,
                "selected_gender_presentation": model_gender,
                "source_gender_presentation_source": source_method or "unknown",
            }
        traits["gender_presentation"] = source_gender
        return {
            "type": "source_profile_gender_guardrail",
            "speaker_id": speaker_id,
            "source_gender_presentation": source_gender,
            "model_reported_gender_presentation": model_gender,
            "corrected_gender_presentation": source_gender,
            "source_gender_presentation_source": source_method or "unknown",
        }
    return None


def _local_audio_path(store: LocalArtifactStore, profile: dict[str, Any], speaker_id: str) -> Path:
    uri = profile.get("source_audio_uri")
    if not uri:
        raise SpeakerProfileTraitEnrichmentError(f"speaker profile {speaker_id} missing source_audio_uri")
    validate_artifact_uri(str(uri))
    path = store.local_path(str(uri))
    if not path.exists():
        raise SpeakerProfileTraitEnrichmentError(f"speaker profile audio missing for {speaker_id}: {uri} -> {path}")
    return path


def enrich_speaker_profiles_with_audio_traits(
    *,
    artifact_store: LocalArtifactStore,
    speaker_profiles_uri: str,
    output_profiles_uri: str,
    report_uri: str | None = None,
    policy: SpeakerProfileTraitEnrichmentPolicy | None = None,
) -> dict[str, Any]:
    """Enrich source speaker profiles with real audio-model traits.

    The input profile artifact is never overwritten unless the caller passes the
    same URI as input and output.  Missing keys/model errors fail loudly by
    default; ``continue_on_error`` records per-speaker errors without fabricating
    traits.
    """

    validate_artifact_uri(speaker_profiles_uri)
    validate_artifact_uri(output_profiles_uri)
    if report_uri is not None:
        validate_artifact_uri(report_uri)
    policy = policy or SpeakerProfileTraitEnrichmentPolicy()
    payload = artifact_store.read_json(speaker_profiles_uri)
    speakers = payload.get("speakers") if isinstance(payload, dict) else None
    if not isinstance(speakers, dict) or not speakers:
        raise AegisVoxContractError("speaker_profiles.json must contain a non-empty speakers object")

    output_speakers: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    guardrail_warnings: list[dict[str, Any]] = []
    num_enriched = 0
    num_skipped_existing = 0
    num_errors = 0

    for speaker_id, raw_profile in sorted(speakers.items()):
        if not isinstance(raw_profile, dict):
            raise AegisVoxContractError(f"speaker profile {speaker_id} must be an object")
        profile = dict(raw_profile)
        metadata = dict(profile.get("metadata") or {})
        existing_traits = metadata.get("audio_traits") if isinstance(metadata.get("audio_traits"), dict) else {}
        if _has_existing_model_traits(profile) and not policy.overwrite_existing:
            num_skipped_existing += 1
            output_speakers[str(speaker_id)] = profile
            rows.append(
                {
                    "speaker_id": str(speaker_id),
                    "status": "skipped_existing_traits",
                    "audio_traits": existing_traits,
                }
            )
            continue
        try:
            audio_path = _local_audio_path(artifact_store, profile, str(speaker_id))
            extraction = extract_audio_traits_with_openai(audio_path, model=policy.model)
        except (AudioTraitExtractionError, SpeakerProfileTraitEnrichmentError) as exc:
            if not policy.continue_on_error:
                raise
            num_errors += 1
            output_speakers[str(speaker_id)] = profile
            rows.append({"speaker_id": str(speaker_id), "status": "error", "error": str(exc)})
            continue
        traits = dict(extraction.get("audio_traits") if isinstance(extraction.get("audio_traits"), dict) else {})
        warning = _apply_source_gender_guardrail(speaker_id=str(speaker_id), profile=profile, traits=traits)
        if warning:
            guardrail_warnings.append(warning)
        metadata["audio_traits"] = {**existing_traits, **traits}
        metadata["audio_trait_extraction"] = {
            "model": extraction.get("model") or policy.model,
            "confidence": extraction.get("confidence"),
            "evidence": extraction.get("evidence") if isinstance(extraction.get("evidence"), list) else [],
            "guardrail_warning": warning,
            "claim_boundary": (
                "Operational voice-matching traits only; not identity, protected-class, "
                "clinical, or precise geography claims."
            ),
        }
        profile["metadata"] = metadata
        if policy.promote_to_top_level_when_unknown:
            for key in PROFILE_AUDIO_TRAIT_KEYS:
                model_value = traits.get(key)
                if not model_value:
                    continue
                if not _known_trait(profile.get(key)):
                    profile[key] = model_value
                elif key == "gender_presentation" and warning and warning.get("type") == "source_profile_gender_conflict_model_preferred":
                    profile[key] = model_value
        output_speakers[str(speaker_id)] = profile
        num_enriched += 1
        rows.append(
            {
                "speaker_id": str(speaker_id),
                "status": "enriched",
                "source_audio_uri": profile.get("source_audio_uri"),
                "audio_traits": traits,
                "confidence": extraction.get("confidence"),
            }
        )

    output_payload = dict(payload)
    output_payload["speakers"] = output_speakers
    output_payload["audio_trait_enrichment"] = {
        "status": "completed_with_errors" if num_errors else "completed",
        "model": policy.model,
        "input_profiles_uri": speaker_profiles_uri,
        "output_profiles_uri": output_profiles_uri,
        "num_speakers": len(output_speakers),
        "num_enriched": num_enriched,
        "num_skipped_existing": num_skipped_existing,
        "num_errors": num_errors,
        "num_guardrail_warnings": len(guardrail_warnings),
        "claim_boundary": (
            "Source profile traits are query-side operational matching metadata only. "
            "They are not identity, demographic, clinical, or precise geography claims."
        ),
    }
    artifact_store.write_json(output_profiles_uri, output_payload)
    report = {
        "service": "speaker_profile_audio_trait_enrichment",
        **output_payload["audio_trait_enrichment"],
        "guardrail_warnings": guardrail_warnings,
        "rows": rows,
    }
    if report_uri is not None:
        artifact_store.write_json(report_uri, report)
    return report
