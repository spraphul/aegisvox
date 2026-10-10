"""Pseudo-speaker assignment for AegisVox.

Assigns one unique voice-pool target to each source speaker while enforcing
source-target distance and target-target distinctiveness thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .schema import AegisVoxContractError, PrivacyMode, validate_artifact_uri
from .storage import LocalArtifactStore
from .voice_pool import (
    DEFAULT_AUDIO_TRAIT_WEIGHTS,
    VoicePoolIndex,
    VoicePoolSample,
    VoiceSearchRequest,
    VoiceSearchResult,
    cosine_similarity,
    normalize_audio_trait_value,
    open_swara_catalog_traits,
    sample_audio_traits,
)


class AssignmentError(RuntimeError):
    """Raised when no valid privacy-preserving assignment can be produced."""


class AssignmentStatus(str, Enum):
    ASSIGNED = "assigned"
    FAILED = "failed"


@dataclass(frozen=True)
class AssignmentPolicy:
    privacy_mode: PrivacyMode = PrivacyMode.MEDIUM
    candidate_top_k: int = 200
    min_source_target_distance: float = 0.55
    min_target_target_distance: float = 0.30
    preserve_language: bool = True
    preserve_gender_presentation: bool = True
    preserve_accent_group: bool = True
    preserve_pitch_band: bool = False
    preserve_speaking_rate: bool = False
    preserve_energy: bool = False
    preserve_geography_region: bool = True
    preserve_tone: bool = True
    preserve_vocal_weight: bool = True
    preserve_resonance: bool = True
    preserve_voice_maturity: bool = True
    use_hybrid_audio_traits: bool = True
    hard_filter_source_audio_traits: bool = False
    reject_child_like_targets: bool = True
    accent_style_constrained: bool = True
    min_reference_duration_sec: float = 4.5
    min_reference_sample_rate_hz: int = 16000
    require_native_english_targets: bool = True
    require_native_english_for_north_america: bool = True
    vector_weight: float = 1.0
    trait_weight: float = 2.0
    trait_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_AUDIO_TRAIT_WEIGHTS))

    @classmethod
    def for_privacy_mode(cls, privacy_mode: PrivacyMode | str) -> "AssignmentPolicy":
        mode = PrivacyMode(privacy_mode)
        if mode is PrivacyMode.LOW:
            return cls(
                privacy_mode=mode,
                min_source_target_distance=0.40,
                min_target_target_distance=0.25,
                preserve_language=True,
                preserve_gender_presentation=True,
                preserve_accent_group=True,
                preserve_geography_region=True,
                preserve_pitch_band=True,
                preserve_speaking_rate=True,
                preserve_energy=True,
                preserve_voice_maturity=True,
            )
        if mode is PrivacyMode.HIGH:
            return cls(
                privacy_mode=mode,
                min_source_target_distance=0.70,
                min_target_target_distance=0.45,
                preserve_language=True,
                preserve_gender_presentation=True,
                preserve_accent_group=True,
                preserve_geography_region=True,
                preserve_pitch_band=False,
                preserve_speaking_rate=False,
                preserve_energy=False,
                preserve_voice_maturity=True,
            )
        return cls(
            privacy_mode=mode,
            min_source_target_distance=0.55,
            min_target_target_distance=0.30,
            preserve_language=True,
            preserve_gender_presentation=True,
            preserve_accent_group=True,
            preserve_geography_region=True,
            preserve_pitch_band=False,
            preserve_speaking_rate=True,
            preserve_energy=False,
            preserve_voice_maturity=True,
        )


def load_speaker_profiles(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    speakers = payload.get("speakers")
    if not isinstance(speakers, dict) or not speakers:
        raise AegisVoxContractError("speaker_profiles.json must contain a non-empty speakers object")
    return {str(k): dict(v) for k, v in speakers.items()}


def _load_embedding_from_uri(store: LocalArtifactStore, uri: str) -> list[float]:
    validate_artifact_uri(uri)
    payload = store.read_json(uri)
    embedding = payload.get("embedding") if isinstance(payload, dict) else None
    if not isinstance(embedding, list) or not embedding:
        raise AegisVoxContractError(f"embedding artifact {uri} must contain non-empty embedding list")
    return [float(x) for x in embedding]


def _target_target_distance(index: VoicePoolIndex, sample_a: str, sample_b: str) -> float:
    return 1.0 - cosine_similarity(index.embeddings[sample_a], index.embeddings[sample_b])


def _candidate_hard_rejection(
    *,
    profile: dict[str, Any],
    candidate: VoiceSearchResult,
    index: VoicePoolIndex,
    policy: AssignmentPolicy,
) -> str | None:
    query_traits = _query_audio_traits_from_profile(profile, policy)
    target_sample = index.samples_by_id[candidate.sample_id]
    return _assignment_v2_rejection_reason(
        profile=profile,
        query_traits=query_traits,
        target_sample=target_sample,
        target_traits=sample_audio_traits(target_sample),
        policy=policy,
    )


def _select_assignment_candidates(
    *,
    profiles: dict[str, dict[str, Any]],
    source_embeddings: dict[str, list[float]],
    index: VoicePoolIndex,
    policy: AssignmentPolicy,
) -> tuple[dict[str, VoiceSearchResult], dict[str, list[dict[str, Any]]], str]:
    """Select targets without relaxing any frozen assignment constraint.

    The first pass is the historical deterministic greedy procedure.  If that
    reaches a dead end, a bounded depth-first search revisits earlier choices
    using the same retrieval order, top-k inventory, hard metadata gates, and
    distance thresholds.  Consequently every previously successful greedy
    assignment remains unchanged; backtracking only repairs a greedy ordering
    failure when an exact feasible assignment already exists.
    """

    speaker_ids = sorted(profiles)
    rejected: dict[str, list[dict[str, Any]]] = {speaker_id: [] for speaker_id in speaker_ids}
    greedy: dict[str, VoiceSearchResult] = {}
    used_voice_ids: list[str] = []
    used_sample_ids: list[str] = []

    for speaker_id in speaker_ids:
        profile = profiles[speaker_id]
        request = _query_from_profile(
            speaker_id=speaker_id,
            profile=profile,
            source_embedding=source_embeddings[speaker_id],
            policy=policy,
            exclude_voice_ids=used_voice_ids,
            voice_pool_id=index.manifest.voice_pool_id,
        )
        chosen = None
        for candidate in index.search(request):
            hard_reason = _candidate_hard_rejection(
                profile=profile,
                candidate=candidate,
                index=index,
                policy=policy,
            )
            if hard_reason:
                rejected[speaker_id].append(
                    {"target_voice_id": candidate.voice_id, "target_sample_id": candidate.sample_id, "reason": hard_reason}
                )
                continue
            conflicts = []
            for sample_id in used_sample_ids:
                distance = _target_target_distance(index, candidate.sample_id, sample_id)
                if distance < policy.min_target_target_distance:
                    conflicts.append(
                        f"target_target_distance_to_{sample_id}={distance:.3f}<min={policy.min_target_target_distance:.3f}"
                    )
            if not conflicts:
                chosen = candidate
                break
            rejected[speaker_id].append(
                {"target_voice_id": candidate.voice_id, "target_sample_id": candidate.sample_id, "reason": "; ".join(conflicts)}
            )
        if chosen is None:
            break
        greedy[speaker_id] = chosen
        used_voice_ids.append(chosen.voice_id)
        used_sample_ids.append(chosen.sample_id)

    if len(greedy) == len(speaker_ids):
        return greedy, rejected, "greedy_exact_constraints_v1"

    # Greedy failed. Search the assignment tree while issuing the same top-k
    # query at each node with the voices already selected on that branch
    # excluded. This matters when a voice contributes many samples: a static
    # no-exclusion top-k list can hide valid samples that the historical
    # conditional retrieval would expose after another voice is selected.
    global_rejected: dict[str, list[dict[str, Any]]] = {speaker_id: [] for speaker_id in speaker_ids}
    selected: dict[str, VoiceSearchResult] = {}
    selected_voice_ids: set[str] = set()
    selected_sample_ids: list[str] = []
    search_nodes = 0
    max_search_nodes = 250_000

    def search(depth: int) -> bool:
        nonlocal search_nodes
        if depth == len(speaker_ids):
            return True
        search_nodes += 1
        if search_nodes > max_search_nodes:
            return False
        speaker_id = speaker_ids[depth]
        profile = profiles[speaker_id]
        request = _query_from_profile(
            speaker_id=speaker_id,
            profile=profile,
            source_embedding=source_embeddings[speaker_id],
            policy=policy,
            exclude_voice_ids=sorted(selected_voice_ids),
            voice_pool_id=index.manifest.voice_pool_id,
        )
        for candidate in index.search(request):
            hard_reason = _candidate_hard_rejection(
                profile=profile,
                candidate=candidate,
                index=index,
                policy=policy,
            )
            if hard_reason:
                global_rejected[speaker_id].append(
                    {"target_voice_id": candidate.voice_id, "target_sample_id": candidate.sample_id, "reason": hard_reason}
                )
                continue
            if any(
                _target_target_distance(index, candidate.sample_id, sample_id) < policy.min_target_target_distance
                for sample_id in selected_sample_ids
            ):
                continue
            selected[speaker_id] = candidate
            selected_voice_ids.add(candidate.voice_id)
            selected_sample_ids.append(candidate.sample_id)
            if search(depth + 1):
                return True
            selected_sample_ids.pop()
            selected_voice_ids.remove(candidate.voice_id)
            del selected[speaker_id]
        return False

    if search(0):
        return dict(selected), global_rejected, "global_backtracking_exact_constraints_v1"

    failed_speaker = speaker_ids[len(greedy)] if len(greedy) < len(speaker_ids) else speaker_ids[-1]
    raise AssignmentError(
        f"no valid pseudo-speaker candidate for {failed_speaker}; "
        f"privacy_mode={policy.privacy_mode.value}, min_source_target_distance={policy.min_source_target_distance}, "
        f"min_target_target_distance={policy.min_target_target_distance}; exact global assignment search exhausted "
        f"after {search_nodes} nodes"
    )


def _profile_trait(profile: dict[str, Any], key: str) -> Any:
    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
    traits = metadata.get("audio_traits") if isinstance(metadata.get("audio_traits"), dict) else {}
    if key in {"vocal_weight", "resonance", "voice_maturity"}:
        value = traits.get(key)
        if value not in {None, "", "unknown"}:
            return value
        value = profile.get(key)
        if value not in {None, "", "unknown"}:
            return value
        return None
    if key == "gender_presentation":
        metadata_source = str(metadata.get("gender_presentation_source") or "").lower()
        value = traits.get(key)
        if ("f0" in metadata_source or "heuristic" in metadata_source) and value not in {None, "", "unknown"}:
            return value
    value = profile.get(key)
    if value not in {None, "", "unknown"}:
        return value
    value = traits.get(key)
    if value not in {None, "", "unknown"}:
        return value
    return None


def _query_audio_traits_from_profile(profile: dict[str, Any], policy: AssignmentPolicy) -> dict[str, Any]:
    if not policy.use_hybrid_audio_traits:
        return {}
    mapping = {
        "language": policy.preserve_language,
        "pitch_band": policy.preserve_pitch_band,
        "speaking_rate": policy.preserve_speaking_rate,
        "energy": policy.preserve_energy,
        "gender_presentation": policy.preserve_gender_presentation,
        "accent_group": policy.preserve_accent_group,
        "geography_region": policy.preserve_geography_region,
        "tone": policy.preserve_tone,
        "vocal_weight": policy.preserve_vocal_weight,
        "resonance": policy.preserve_resonance,
        "voice_maturity": policy.preserve_voice_maturity,
    }
    return {key: _profile_trait(profile, key) for key, enabled in mapping.items() if enabled and _profile_trait(profile, key) is not None}



AMERICAN_ENGLISH_ACCENTS = {"american_english", "general_american", "american", "english", "us_english"}
NORTH_AMERICA_REGIONS = {"north_america", "united_states", "canada"}
VOICE_WEIGHT_ORDER = {"thin": 0, "medium": 1, "thick": 2}
RESONANCE_ORDER = {"light": 0, "balanced": 1, "deep": 2}
VOICE_MATURITY_ORDER = {"youthful": 0, "adult": 1, "mature": 2}


def _normalized_profile_trait(profile: dict[str, Any], key: str) -> str | None:
    value = _profile_trait(profile, key)
    normalized = normalize_audio_trait_value(key, value)
    if normalized and normalized != "unknown":
        return normalized
    return None


def _voice_quality_payload_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = metadata.get("voice_quality_profile")
    if not isinstance(payload, dict):
        return {}
    if isinstance(payload.get("profile"), dict) or isinstance(payload.get("audio_traits"), dict):
        return payload
    result = payload.get("result")
    return dict(result) if isinstance(result, dict) else {}


def _profile_voice_quality_payload(profile: dict[str, Any]) -> dict[str, Any]:
    metadata = profile.get("metadata") if isinstance(profile.get("metadata"), dict) else {}
    return _voice_quality_payload_from_metadata(metadata)


def _sample_voice_quality_payload(sample: VoicePoolSample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return _voice_quality_payload_from_metadata(metadata)


def _voice_quality_profile_value(payload: dict[str, Any], key: str) -> Any:
    for container_key in ("profile", "audio_traits"):
        container = payload.get(container_key)
        if isinstance(container, dict) and container.get(key) not in (None, "", "unknown"):
            return container.get(key)
    return None


def _voice_quality_child_like_risk(payload: dict[str, Any]) -> float | None:
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        return None
    try:
        return float(profile.get("child_like_risk"))
    except (TypeError, ValueError):
        return None


def _source_requires_adult_target(profile: dict[str, Any], query_traits: dict[str, Any]) -> bool:
    maturity = normalize_audio_trait_value(
        "voice_maturity",
        query_traits.get("voice_maturity") or _profile_trait(profile, "voice_maturity"),
    )
    if maturity in {"adult", "mature"}:
        return True
    payload = _profile_voice_quality_payload(profile)
    vq_maturity = normalize_audio_trait_value("voice_maturity", _voice_quality_profile_value(payload, "voice_maturity"))
    return vq_maturity in {"adult", "mature"}


def _adult_source_target_maturity_rejection(
    *,
    source_profile: dict[str, Any],
    query_traits: dict[str, Any],
    target_sample: VoicePoolSample,
    target_traits: dict[str, str],
) -> str | None:
    if not _source_requires_adult_target(source_profile, query_traits):
        return None
    target_maturity = normalize_audio_trait_value("voice_maturity", target_traits.get("voice_maturity"))
    target_vq = _sample_voice_quality_payload(target_sample)
    vq_maturity = normalize_audio_trait_value("voice_maturity", _voice_quality_profile_value(target_vq, "voice_maturity"))
    child_like_risk = _voice_quality_child_like_risk(target_vq)
    if target_maturity == "youthful" or vq_maturity == "youthful":
        return "target_child_like_for_adult_source:voice_maturity=youthful"
    if child_like_risk is not None and child_like_risk >= 0.25:
        return f"target_child_like_for_adult_source:child_like_risk={child_like_risk:.3f}"
    return None


def _target_child_like_rejection(target_sample: VoicePoolSample, target_traits: dict[str, str]) -> str | None:
    """Reject clearly child-like targets without trusting source GPT-audio traits."""

    target_maturity = normalize_audio_trait_value("voice_maturity", target_traits.get("voice_maturity"))
    target_vq = _sample_voice_quality_payload(target_sample)
    vq_maturity = normalize_audio_trait_value("voice_maturity", _voice_quality_profile_value(target_vq, "voice_maturity"))
    child_like_risk = _voice_quality_child_like_risk(target_vq)
    if target_maturity == "youthful" or vq_maturity == "youthful":
        return "target_child_like:voice_maturity=youthful"
    if child_like_risk is not None and child_like_risk >= 0.25:
        return f"target_child_like:child_like_risk={child_like_risk:.3f}"
    return None


def _source_language(profile: dict[str, Any], query_traits: dict[str, Any]) -> str | None:
    return normalize_audio_trait_value("language", query_traits.get("language") or _profile_trait(profile, "language"))


def _source_is_english_or_unknown(profile: dict[str, Any], query_traits: dict[str, Any]) -> bool:
    language = _source_language(profile, query_traits)
    return language in {None, "english", "unknown"}


def _source_is_north_american_english(profile: dict[str, Any], query_traits: dict[str, Any]) -> bool:
    language = _source_language(profile, query_traits)
    if language != "english":
        return False
    accent = normalize_audio_trait_value("accent_group", query_traits.get("accent_group") or _profile_trait(profile, "accent_group"))
    geography = normalize_audio_trait_value("geography_region", query_traits.get("geography_region") or _profile_trait(profile, "geography_region"))
    return accent in AMERICAN_ENGLISH_ACCENTS or geography in NORTH_AMERICA_REGIONS


def _ordered_style_rejection(
    *,
    key: str,
    source_value: Any,
    target_value: Any,
    order: dict[str, int],
    max_distance: int = 1,
) -> str | None:
    source = normalize_audio_trait_value(key, source_value)
    target = normalize_audio_trait_value(key, target_value)
    if not source or source == "unknown":
        return None
    if not target or target == "unknown":
        # Missing values should not kill the whole pool, but they are recorded
        # as non-ideal candidates by the search explanation.
        return None
    if source not in order or target not in order:
        return None
    if abs(order[source] - order[target]) > max_distance:
        return f"mismatched_style_trait:{key}:{target}!~{source}"
    return None


def _assignment_v2_rejection_reason(
    *,
    profile: dict[str, Any],
    query_traits: dict[str, Any],
    target_sample: VoicePoolSample,
    target_traits: dict[str, str],
    policy: AssignmentPolicy,
) -> str | None:
    if not policy.accent_style_constrained:
        return None
    catalog = open_swara_catalog_traits(target_sample)
    duration = catalog.get("catalog_duration_seconds")
    if isinstance(duration, (int, float)) and duration < policy.min_reference_duration_sec:
        return f"reference_too_short:{duration:.2f}<min={policy.min_reference_duration_sec:.2f}"
    sample_rate = catalog.get("catalog_sample_rate")
    if isinstance(sample_rate, int) and sample_rate < policy.min_reference_sample_rate_hz:
        return f"reference_sample_rate_too_low:{sample_rate}<min={policy.min_reference_sample_rate_hz}"
    source_language = _source_language(profile, query_traits)
    if source_language and source_language not in {"unknown", "english"}:
        target_language = normalize_audio_trait_value("language", target_traits.get("language") or catalog.get("catalog_language"))
        if target_language and target_language != source_language:
            return f"target_language_not_source_language:{target_language}!={source_language}"
    if policy.require_native_english_targets and _source_is_english_or_unknown(profile, query_traits):
        catalog_language = catalog.get("catalog_language")
        catalog_type = catalog.get("catalog_type")
        is_english_accent = catalog.get("is_english_accent")
        if catalog_language and catalog_language != "english":
            return f"catalog_language_not_native_english:{catalog_language}"
        if catalog_type and catalog_type != "native":
            return f"catalog_type_not_native:{catalog_type}"
        if is_english_accent is True:
            return "catalog_is_english_accent"
    if policy.reject_child_like_targets:
        reason = _target_child_like_rejection(target_sample, target_traits)
        if reason:
            return reason
    if (
        policy.hard_filter_source_audio_traits
        and policy.require_native_english_for_north_america
        and _source_is_north_american_english(profile, query_traits)
    ):
        catalog_language = catalog.get("catalog_language")
        catalog_type = catalog.get("catalog_type")
        is_english_accent = catalog.get("is_english_accent")
        if catalog_language and catalog_language != "english":
            return f"catalog_language_not_native_english:{catalog_language}"
        if catalog_type and catalog_type != "native":
            return f"catalog_type_not_native:{catalog_type}"
        if is_english_accent is True:
            return "catalog_is_english_accent"
        target_language = target_traits.get("language")
        if target_language and target_language != "english":
            return f"target_language_not_english:{target_language}"
        target_geo = target_traits.get("geography_region")
        if target_geo and target_geo not in NORTH_AMERICA_REGIONS:
            return f"target_geography_not_north_america:{target_geo}"
        target_accent = target_traits.get("accent_group")
        if target_accent and target_accent not in AMERICAN_ENGLISH_ACCENTS:
            return f"target_accent_not_american_english:{target_accent}"
    if policy.hard_filter_source_audio_traits and policy.preserve_vocal_weight:
        reason = _ordered_style_rejection(
            key="vocal_weight",
            source_value=query_traits.get("vocal_weight") or _profile_trait(profile, "vocal_weight"),
            target_value=target_traits.get("vocal_weight"),
            order=VOICE_WEIGHT_ORDER,
            max_distance=1,
        )
        if reason:
            return reason
    if policy.hard_filter_source_audio_traits and policy.preserve_resonance:
        reason = _ordered_style_rejection(
            key="resonance",
            source_value=query_traits.get("resonance") or _profile_trait(profile, "resonance"),
            target_value=target_traits.get("resonance"),
            order=RESONANCE_ORDER,
            max_distance=1,
        )
        if reason:
            return reason
    if policy.hard_filter_source_audio_traits and policy.preserve_voice_maturity:
        reason = _adult_source_target_maturity_rejection(
            source_profile=profile,
            query_traits=query_traits,
            target_sample=target_sample,
            target_traits=target_traits,
        )
        if reason:
            return reason
        reason = _ordered_style_rejection(
            key="voice_maturity",
            source_value=query_traits.get("voice_maturity") or _profile_trait(profile, "voice_maturity"),
            target_value=target_traits.get("voice_maturity"),
            order=VOICE_MATURITY_ORDER,
            max_distance=1,
        )
        if reason:
            return reason
    return None

def _query_from_profile(
    *,
    speaker_id: str,
    profile: dict[str, Any],
    source_embedding: list[float],
    policy: AssignmentPolicy,
    exclude_voice_ids: list[str],
    voice_pool_id: str,
) -> VoiceSearchRequest:
    query_audio_traits = _query_audio_traits_from_profile(profile, policy)
    hard_source_traits_enabled = policy.hard_filter_source_audio_traits
    # Language is transcript-level operational context, not a noisy voice trait.
    # Preserve it whenever known so multilingual public evaluations do not route
    # French conversations through English-only target voices. Other source audio
    # traits remain soft unless hard filtering is explicitly enabled.
    language = query_audio_traits.get("language") if policy.preserve_language else None
    pitch_band = query_audio_traits.get("pitch_band") if policy.preserve_pitch_band else None
    speaking_rate = query_audio_traits.get("speaking_rate") if policy.preserve_speaking_rate else None
    energy = query_audio_traits.get("energy") if policy.preserve_energy else None
    gender_presentation = query_audio_traits.get("gender_presentation") if policy.preserve_gender_presentation and hard_source_traits_enabled else None
    accent_group = query_audio_traits.get("accent_group") if policy.preserve_accent_group and hard_source_traits_enabled else None
    return VoiceSearchRequest(
        voice_pool_id=voice_pool_id,
        query_embedding=source_embedding,
        language=language,
        pitch_band=pitch_band,
        speaking_rate=speaking_rate,
        energy=energy,
        gender_presentation=gender_presentation,
        accent_group=accent_group,
        geography_region=None,
        tone=None,
        audio_traits=query_audio_traits,
        trait_weights=policy.trait_weights,
        vector_weight=policy.vector_weight,
        trait_weight=policy.trait_weight,
        exclude_voice_ids=exclude_voice_ids,
        min_source_distance=policy.min_source_target_distance,
        # GPT-audio source traits are too noisy for hard filtering.  Assignment
        # keeps them in `audio_traits` for soft scoring/explanations only; hard
        # exclusions are deterministic metadata/acoustic gates applied outside
        # the voice-pool search.
        require_known_hard_trait_matches=False,
        top_k=policy.candidate_top_k,
    )


def assignment_report_markdown(payload: dict[str, Any]) -> str:
    """Render a compact human-auditable assignment report.

    The report deliberately avoids embedding vectors or raw transcript/audio
    content. It explains policy thresholds, selected pseudo-speakers, rejected
    candidates, and target-target distinctiveness evidence.
    """

    assignments = payload.get("assignments") or {}
    policy = payload.get("policy") or {}
    summary = payload.get("target_target_summary") or {}
    lines = [
        "# Pseudo-Speaker Assignment Report",
        "",
        "## Summary",
        "",
        f"- Voice pool: `{payload.get('voice_pool_id', 'unknown')}`",
        f"- Privacy mode: `{payload.get('privacy_mode', 'unknown')}`",
        f"- Assigned speakers: {len(assignments)}",
        f"- Target-target pairs checked: {summary.get('num_pairs', 0)}",
        f"- Minimum target-target distance: {summary.get('min_target_target_distance', 'n/a')}",
        "",
        "## Policy",
        "",
        f"- Minimum source-target distance: {policy.get('min_source_target_distance', 'n/a')}",
        f"- Minimum target-target distance: {policy.get('min_target_target_distance', 'n/a')}",
        f"- Candidate top-k: {policy.get('candidate_top_k', 'n/a')}",
        f"- Require native English targets: {policy.get('require_native_english_targets', 'n/a')}",
        f"- Preserve language: {policy.get('preserve_language', 'n/a')}",
        f"- Preserve gender presentation: {policy.get('preserve_gender_presentation', 'n/a')}",
        f"- Preserve accent group: {policy.get('preserve_accent_group', 'n/a')}",
        f"- Preserve pitch band: {policy.get('preserve_pitch_band', 'n/a')}",
        f"- Preserve speaking rate: {policy.get('preserve_speaking_rate', 'n/a')}",
        f"- Preserve energy: {policy.get('preserve_energy', 'n/a')}",
        f"- Preserve geography region: {policy.get('preserve_geography_region', 'n/a')}",
        f"- Preserve tone: {policy.get('preserve_tone', 'n/a')}",
        f"- Use hybrid audio traits: {policy.get('use_hybrid_audio_traits', 'n/a')}",
        f"- Hard-filter source audio traits: {policy.get('hard_filter_source_audio_traits', 'n/a')}",
        f"- Reject child-like targets: {policy.get('reject_child_like_targets', 'n/a')}",
        f"- Vector weight: {policy.get('vector_weight', 'n/a')}",
        f"- Trait weight: {policy.get('trait_weight', 'n/a')}",
        f"- Trait weights: `{json.dumps(policy.get('trait_weights') or {}, sort_keys=True)}`",
        "",
        "Source GPT-audio traits are soft scoring/explanation signals only by default. Hard filtering is limited to deterministic target metadata/acoustic gates and embedding distance thresholds unless explicitly overridden.",
        "",
        "## Assignments",
        "",
        "| Source speaker | Source gender | Source accent | Target voice | Target gender | Target accent | Target sample | Source-target distance | Cosine similarity | Status |",
        "|---|---|---|---|---|---|---|---:|---:|---|",
    ]
    for speaker_id, row in sorted(assignments.items()):
        lines.append(
            "| {speaker} | {source_gender} | {source_accent} | {voice} | {target_gender} | {target_accent} | {sample} | {distance} | {similarity} | {status} |".format(
                speaker=speaker_id,
                source_gender=row.get("source_gender_presentation", ""),
                source_accent=row.get("source_accent_group", ""),
                voice=row.get("target_voice_id", ""),
                target_gender=row.get("target_gender_presentation", ""),
                target_accent=row.get("target_accent_group", ""),
                sample=row.get("target_voice_sample_id", ""),
                distance=row.get("source_target_distance", ""),
                similarity=row.get("cosine_similarity", ""),
                status=row.get("status", ""),
            )
        )
    lines.extend(["", "## Assignment rationales", ""])
    for speaker_id, row in sorted(assignments.items()):
        lines.extend(
            [
                f"### {speaker_id}",
                "",
                f"- Target voice: `{row.get('target_voice_id', 'unknown')}`",
                f"- Source gender presentation: `{row.get('source_gender_presentation', 'unknown')}`",
                f"- Target gender presentation: `{row.get('target_gender_presentation', 'unknown')}`",
                f"- Source accent group: `{row.get('source_accent_group', 'unknown')}`",
                f"- Target accent group: `{row.get('target_accent_group', 'unknown')}`",
                f"- Target sample: `{row.get('target_voice_sample_id', 'unknown')}`",
                f"- Source audio traits: `{json.dumps(row.get('source_audio_traits') or {}, sort_keys=True)}`",
                f"- Target audio traits: `{json.dumps(row.get('target_audio_traits') or {}, sort_keys=True)}`",
                f"- Trait match: `{json.dumps(row.get('query_trait_match') or {}, sort_keys=True)}`",
                f"- Reference audio URI: `{row.get('target_reference_audio_uri', 'unknown')}`",
                f"- Reason: {row.get('assignment_reason', 'n/a')}",
                "",
            ]
        )
    target_distances = payload.get("target_target_distances") or []
    lines.extend(["## Target-target distinctiveness", ""])
    if target_distances:
        lines.extend(
            [
                "| Left target sample | Right target sample | Distance |",
                "|---|---|---:|",
            ]
        )
        for row in target_distances:
            lines.append(
                "| {left} | {right} | {distance} |".format(
                    left=row.get("left_target_sample_id", ""),
                    right=row.get("right_target_sample_id", ""),
                    distance=row.get("target_target_distance", ""),
                )
            )
        lines.append("")
    else:
        lines.extend(["No target-target pairs were needed for this run.", ""])
    rejected = payload.get("rejected_candidates") or {}
    lines.extend(["## Rejected candidates", ""])
    any_rejected = False
    for speaker_id, rows in sorted(rejected.items()):
        if not rows:
            continue
        any_rejected = True
        lines.extend([f"### {speaker_id}", "", "| Target voice | Target sample | Reason |", "|---|---|---|"])
        for row in rows:
            lines.append(
                "| {voice} | {sample} | {reason} |".format(
                    voice=row.get("target_voice_id", ""),
                    sample=row.get("target_sample_id", ""),
                    reason=str(row.get("reason", "")).replace("|", "\\|"),
                )
            )
        lines.append("")
    if not any_rejected:
        lines.extend(["No candidates were rejected after retrieval filtering.", ""])
    lines.extend(
        [
            "## Audit boundary",
            "",
            "- This report explains assignment decisions and policy thresholds.",
            "- It is not an ASV/linkability evaluation; speaker privacy metrics are produced by the evaluation/privacy services.",
            "- It does not include raw embeddings or PHI-bearing transcript/audio content.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass
class AssignmentService:
    artifact_store: LocalArtifactStore
    voice_pool_index: VoicePoolIndex
    policy: AssignmentPolicy = field(default_factory=AssignmentPolicy)

    def assign_to_artifact(
        self,
        speaker_profiles_uri: str,
        output_assignment_uri: str,
        output_report_uri: str | None = None,
    ) -> dict[str, Any]:
        validate_artifact_uri(output_assignment_uri)
        if output_report_uri is not None:
            validate_artifact_uri(output_report_uri)
        profiles_path = self.artifact_store.local_path(speaker_profiles_uri)
        profiles = load_speaker_profiles(profiles_path)
        source_embeddings: dict[str, list[float]] = {}
        for speaker_id, profile in sorted(profiles.items()):
            embedding_uri = profile.get("source_embedding_uri")
            if not embedding_uri:
                raise AegisVoxContractError(f"speaker profile {speaker_id} missing source_embedding_uri")
            source_embeddings[speaker_id] = _load_embedding_from_uri(self.artifact_store, embedding_uri)

        selected_candidates, rejected, assignment_solver = _select_assignment_candidates(
            profiles=profiles,
            source_embeddings=source_embeddings,
            index=self.voice_pool_index,
            policy=self.policy,
        )
        assignments: dict[str, dict[str, Any]] = {}
        used_voice_ids: list[str] = []
        used_sample_ids: list[str] = []
        for speaker_id, profile in sorted(profiles.items()):
            chosen = selected_candidates[speaker_id]
            query_traits = _query_audio_traits_from_profile(profile, self.policy)
            used_voice_ids.append(chosen.voice_id)
            used_sample_ids.append(chosen.sample_id)
            target_sample = self.voice_pool_index.samples_by_id[chosen.sample_id]
            target_traits = sample_audio_traits(target_sample)
            query_traits = _query_audio_traits_from_profile(profile, self.policy)
            target_catalog_traits = open_swara_catalog_traits(target_sample)
            source_gender = str(query_traits.get("gender_presentation") or profile.get("gender_presentation") or "unknown")
            target_gender = target_traits.get("gender_presentation", "unknown")
            source_accent = str(query_traits.get("accent_group") or profile.get("accent_group") or "unknown")
            target_accent = target_traits.get("accent_group", "unknown")
            gender_match_status = (
                "not_applicable_source_unknown"
                if source_gender == "unknown"
                else ("match" if source_gender == target_gender else "mismatch")
            )
            accent_match_status = (
                "not_applicable_source_unknown"
                if source_accent == "unknown"
                else ("match" if source_accent == target_accent else "mismatch")
            )
            source_language = str(query_traits.get("language") or profile.get("language") or "unknown")
            target_language = target_traits.get("language", "unknown")
            language_match_status = (
                "not_applicable_source_unknown"
                if source_language == "unknown"
                else ("match" if normalize_audio_trait_value("language", source_language) == normalize_audio_trait_value("language", target_language) else "mismatch")
            )
            assignments[speaker_id] = {
                "target_voice_id": chosen.voice_id,
                "target_voice_sample_id": chosen.sample_id,
                "target_reference_audio_uri": chosen.audio_uri,
                "source_language": source_language,
                "target_language": target_language,
                "language_match_status": language_match_status,
                "source_gender_presentation": source_gender,
                "target_gender_presentation": target_gender,
                "gender_match_status": gender_match_status,
                "source_accent_group": source_accent,
                "target_accent_group": target_accent,
                "accent_match_status": accent_match_status,
                "source_audio_traits": query_traits,
                "target_audio_traits": target_traits,
                "query_trait_match": chosen.metadata.get("query_trait_match") if isinstance(chosen.metadata, dict) else None,
                "hybrid_search_explanation": chosen.metadata.get("hybrid_search_explanation") if isinstance(chosen.metadata, dict) else None,
                "target_catalog_traits": target_catalog_traits,
                "assignment_strategy": "accent_style_constrained_v2" if self.policy.accent_style_constrained else "hybrid_trait_search_v1",
                "source_target_distance": chosen.source_target_distance,
                "cosine_similarity": chosen.cosine_similarity,
                "assignment_reason": (
                    f"privacy_mode={self.policy.privacy_mode.value}; {chosen.assignment_reason}; "
                    f"unique target; source-target and target-target thresholds satisfied; solver={assignment_solver}"
                ),
                "status": AssignmentStatus.ASSIGNED.value,
            }
        target_distances = []
        for left_idx, left_sample in enumerate(used_sample_ids):
            for right_sample in used_sample_ids[left_idx + 1 :]:
                distance = _target_target_distance(self.voice_pool_index, left_sample, right_sample)
                left_voice = self.voice_pool_index.samples_by_id[left_sample].voice_id
                right_voice = self.voice_pool_index.samples_by_id[right_sample].voice_id
                target_distances.append(
                    {
                        "left_target_voice_id": left_voice,
                        "left_target_sample_id": left_sample,
                        "right_target_voice_id": right_voice,
                        "right_target_sample_id": right_sample,
                        "target_target_distance": round(distance, 6),
                    }
                )
        min_target_distance = min((row["target_target_distance"] for row in target_distances), default=None)
        payload = {
            "voice_pool_id": self.voice_pool_index.manifest.voice_pool_id,
            "privacy_mode": self.policy.privacy_mode.value,
            "assignment_solver": assignment_solver,
            "policy": {
                "min_source_target_distance": self.policy.min_source_target_distance,
                "min_target_target_distance": self.policy.min_target_target_distance,
                "candidate_top_k": self.policy.candidate_top_k,
                "preserve_language": self.policy.preserve_language,
                "preserve_gender_presentation": self.policy.preserve_gender_presentation,
                "preserve_accent_group": self.policy.preserve_accent_group,
                "preserve_pitch_band": self.policy.preserve_pitch_band,
                "preserve_speaking_rate": self.policy.preserve_speaking_rate,
                "preserve_energy": self.policy.preserve_energy,
                "preserve_geography_region": self.policy.preserve_geography_region,
                "preserve_tone": self.policy.preserve_tone,
                "preserve_vocal_weight": self.policy.preserve_vocal_weight,
                "preserve_resonance": self.policy.preserve_resonance,
                "preserve_voice_maturity": self.policy.preserve_voice_maturity,
                "use_hybrid_audio_traits": self.policy.use_hybrid_audio_traits,
                "hard_filter_source_audio_traits": self.policy.hard_filter_source_audio_traits,
                "reject_child_like_targets": self.policy.reject_child_like_targets,
                "accent_style_constrained": self.policy.accent_style_constrained,
                "min_reference_duration_sec": self.policy.min_reference_duration_sec,
                "min_reference_sample_rate_hz": self.policy.min_reference_sample_rate_hz,
                "require_native_english_targets": self.policy.require_native_english_targets,
                "require_native_english_for_north_america": self.policy.require_native_english_for_north_america,
                "vector_weight": self.policy.vector_weight,
                "trait_weight": self.policy.trait_weight,
                "trait_weights": self.policy.trait_weights,
            },
            "assignments": assignments,
            "target_target_distances": target_distances,
            "target_target_summary": {
                "num_pairs": len(target_distances),
                "min_target_target_distance": min_target_distance,
                "policy_min_target_target_distance": self.policy.min_target_target_distance,
            },
            "rejected_candidates": rejected,
            "assignment_strategy": "accent_style_constrained_v2" if self.policy.accent_style_constrained else "hybrid_trait_search_v1",
            "retrieval_contract": {
                "hard_filters": [
                    "target_reference_duration",
                    "target_reference_sample_rate",
                    "target_native_english_catalog_metadata",
                    "target_child_like_risk_or_youthful_maturity",
                ],
                "source_gpt_audio_traits": "soft_scoring_only",
                "source_embedding": "privacy_distance_scoring_and_min_distance",
                "target_metadata": "deterministic_catalog_and_manifest_traits",
                "claim_boundary": "Do not use model-inferred source traits as hard filters; retrieval hard filters are deterministic target metadata/acoustic gates plus embedding distance thresholds.",
            },
        }
        self.artifact_store.write_json(output_assignment_uri, payload)
        if output_report_uri is not None:
            self.artifact_store.write_bytes(output_report_uri, assignment_report_markdown(payload).encode("utf-8"))
        return payload
