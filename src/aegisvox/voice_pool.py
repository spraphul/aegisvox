"""Voice Pool v1 contracts and retrieval.

The retrieval service consumes a prebuilt voice-pool manifest and embedding index.
It never fabricates voices or embeddings: if an index is missing, malformed, or
empty, it raises an actionable setup error.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schema import AegisVoxContractError, validate_artifact_uri


class VoicePoolSetupError(RuntimeError):
    """Raised when the real voice pool/index artifacts are unavailable."""


@dataclass(frozen=True)
class VoicePoolSample:
    voice_id: str
    sample_id: str
    audio_uri: str
    embedding_uri: str | None = None
    language: str | None = None
    pitch_band: str | None = None
    speaking_rate: str | None = None
    energy: str | None = None
    gender_presentation: str | None = None
    accent_group: str | None = None
    geography_region: str | None = None
    tone: str | None = None
    license: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    qc: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.voice_id:
            raise AegisVoxContractError("voice_id is required")
        if not self.sample_id:
            raise AegisVoxContractError("sample_id is required")
        validate_artifact_uri(self.audio_uri)
        if self.embedding_uri is not None:
            validate_artifact_uri(self.embedding_uri)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VoicePoolSample":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "sample_id": self.sample_id,
            "audio_uri": self.audio_uri,
            "embedding_uri": self.embedding_uri,
            "language": self.language,
            "pitch_band": self.pitch_band,
            "speaking_rate": self.speaking_rate,
            "energy": self.energy,
            "gender_presentation": self.gender_presentation,
            "accent_group": self.accent_group,
            "geography_region": self.geography_region,
            "tone": self.tone,
            "license": self.license,
            "provenance": dict(self.provenance),
            "qc": dict(self.qc),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VoicePoolManifest:
    voice_pool_id: str
    version: str
    samples: list[VoicePoolSample]
    created_at: str | None = None
    source: str | None = None
    quality_report_uri: str | None = None
    embedding_index_uri: str | None = None

    def __post_init__(self) -> None:
        if not self.voice_pool_id:
            raise AegisVoxContractError("voice_pool_id is required")
        if not self.version:
            raise AegisVoxContractError("version is required")
        if not self.samples:
            raise AegisVoxContractError("voice pool manifest must contain at least one sample")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise AegisVoxContractError("sample_id values must be unique within a voice pool")
        if self.quality_report_uri:
            validate_artifact_uri(self.quality_report_uri)
        if self.embedding_index_uri:
            validate_artifact_uri(self.embedding_index_uri)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VoicePoolManifest":
        samples = [VoicePoolSample.from_dict(row) for row in payload.get("samples", [])]
        return cls(
            voice_pool_id=payload["voice_pool_id"],
            version=payload["version"],
            samples=samples,
            created_at=payload.get("created_at"),
            source=payload.get("source"),
            quality_report_uri=payload.get("quality_report_uri"),
            embedding_index_uri=payload.get("embedding_index_uri"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "VoicePoolManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_pool_id": self.voice_pool_id,
            "version": self.version,
            "created_at": self.created_at,
            "source": self.source,
            "quality_report_uri": self.quality_report_uri,
            "embedding_index_uri": self.embedding_index_uri,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class VoiceSearchRequest:
    voice_pool_id: str
    query_embedding: list[float] | None = None
    language: str | None = None
    pitch_band: str | None = None
    speaking_rate: str | None = None
    energy: str | None = None
    gender_presentation: str | None = None
    accent_group: str | None = None
    geography_region: str | None = None
    tone: str | None = None
    audio_traits: dict[str, Any] = field(default_factory=dict)
    trait_weights: dict[str, float] = field(default_factory=dict)
    vector_weight: float = 1.0
    trait_weight: float = 0.35
    exclude_voice_ids: list[str] = field(default_factory=list)
    min_source_distance: float | None = None
    require_known_hard_trait_matches: bool = False
    top_k: int = 10

    def __post_init__(self) -> None:
        if not self.voice_pool_id:
            raise AegisVoxContractError("voice_pool_id is required")
        if self.top_k <= 0:
            raise AegisVoxContractError("top_k must be positive")
        if self.query_embedding is not None:
            if not self.query_embedding:
                raise AegisVoxContractError("query_embedding cannot be empty")
            if any(not isinstance(x, (int, float)) or math.isnan(float(x)) for x in self.query_embedding):
                raise AegisVoxContractError("query_embedding must contain finite numbers")
        if self.vector_weight < 0 or self.trait_weight < 0:
            raise AegisVoxContractError("vector_weight and trait_weight must be non-negative")
        for key, value in self.trait_weights.items():
            if not isinstance(value, (int, float)) or math.isnan(float(value)) or float(value) < 0:
                raise AegisVoxContractError(f"trait weight for {key!r} must be a non-negative finite number")


@dataclass(frozen=True)
class VoiceSearchResult:
    voice_id: str
    sample_id: str
    audio_uri: str
    score: float
    cosine_similarity: float | None
    source_target_distance: float | None
    assignment_reason: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "sample_id": self.sample_id,
            "audio_uri": self.audio_uri,
            "score": self.score,
            "cosine_similarity": self.cosine_similarity,
            "source_target_distance": self.source_target_distance,
            "assignment_reason": self.assignment_reason,
            "metadata": dict(self.metadata),
        }


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise AegisVoxContractError(f"embedding dimension mismatch: {len(a)} != {len(b)}")
    denom = _norm(a) * _norm(b)
    if denom <= 0:
        raise AegisVoxContractError("zero-norm embedding encountered")
    return _dot(a, b) / denom


CANONICAL_AUDIO_TRAIT_KEYS = {
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
}

DEFAULT_AUDIO_TRAIT_WEIGHTS: dict[str, float] = {
    "accent_group": 2.0,
    "geography_region": 1.0,
    "tone": 1.0,
    "language": 0.75,
    "gender_presentation": 1.0,
    "speaking_rate": 0.75,
    "pitch_band": 0.5,
    "energy": 0.5,
    "vocal_weight": 1.25,
    "resonance": 1.0,
    "voice_maturity": 0.4,
}

HARD_GUARDRAIL_TRAITS = ("language", "gender_presentation", "accent_group")
SOFT_RERANK_TRAITS = ("geography_region", "tone", "energy", "speaking_rate", "pitch_band")

TRAIT_EQUIVALENCE_GROUPS: dict[str, dict[str, set[str]]] = {
    # Audio LLMs often use these tone labels interchangeably for ordinary
    # clean speech. Treat them as close operational matches for reranking, not
    # as exact semantic equality.
    "tone": {
        "neutral_conversational": {"neutral", "calm", "conversational", "natural", "steady"},
        "positive_warm": {"warm", "friendly", "positive", "pleasant"},
        "tense_high_arousal": {"tense", "urgent", "stressed", "strained", "anxious"},
        "low_arousal": {"flat", "monotone", "subdued", "tired"},
    },
}

TRAIT_EQUIVALENCE_CREDIT = 0.75


TRAIT_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "language": {
        "en": "english",
        "eng": "english",
        "en_us": "english",
        "en_uk": "english",
        "us_english": "english",
        "uk_english": "english",
        "english_us": "english",
        "english_uk": "english",
        "american_english": "english",
        "british_english": "english",
        "indian_english": "english",
        "fr": "french",
        "fra": "french",
        "fre": "french",
        "fr_fr": "french",
        "french_france": "french",
        "francais": "french",
        "français": "french",
    },
    "gender_presentation": {
        "woman": "female",
        "female_sounding": "female",
        "feminine": "female",
        "girl": "female",
        "man": "male",
        "male_sounding": "male",
        "masculine": "male",
        "boy": "male",
    },
    "accent_group": {
        "american": "american_english",
        "american_accent": "american_english",
        "general_american": "american_english",
        "us": "american_english",
        "usa": "american_english",
        "us_english": "american_english",
        "united_states_english": "american_english",
        "north_american": "american_english",
        "north_american_english": "american_english",
        "indian": "indian_english",
        "india": "indian_english",
        "indian_accent": "indian_english",
        "south_asian": "indian_english",
        "south_asian_english": "indian_english",
        "uk": "british_english",
        "british": "british_english",
        "uk_english": "british_english",
        "british_accent": "british_english",
        "england_english": "british_english",
    },
    "geography_region": {
        "north_american": "north_america",
        "us": "north_america",
        "usa": "north_america",
        "u_s": "north_america",
        "united_states": "north_america",
        "canada": "north_america",
        "south_asian": "south_asia",
        "india": "south_asia",
        "pakistan": "south_asia",
        "bangladesh": "south_asia",
        "uk": "uk_ireland",
        "britain": "uk_ireland",
        "great_britain": "uk_ireland",
        "ireland": "uk_ireland",
        "east_asian": "east_asia",
        "china": "east_asia",
        "japan": "east_asia",
        "korea": "east_asia",
        "middle_eastern": "middle_east",
        "gulf": "middle_east",
    },
    "tone": {
        "matter_of_fact": "neutral",
        "ordinary": "neutral",
        "relaxed": "calm",
        "casual": "conversational",
    },
    "speaking_rate": {
        "medium": "moderate",
        "normal": "moderate",
        "average": "moderate",
    },
    "pitch_band": {
        "medium": "mid",
        "middle": "mid",
    },
    "vocal_weight": {
        "light": "thin",
        "thin_voice": "thin",
        "light_voice": "thin",
        "slim": "thin",
        "average": "medium",
        "moderate": "medium",
        "balanced": "medium",
        "heavy": "thick",
        "full": "thick",
        "deep": "thick",
        "thick_voice": "thick",
    },
    "resonance": {
        "bright": "light",
        "thin": "light",
        "average": "balanced",
        "medium": "balanced",
        "neutral": "balanced",
        "low": "deep",
        "heavy": "deep",
        "chesty": "deep",
    },
    "voice_maturity": {
        "child": "youthful",
        "child_like": "youthful",
        "teen": "youthful",
        "teenager": "youthful",
        "young": "youthful",
        "young_adult": "adult",
        "middle_aged": "adult",
        "adult": "adult",
        "older": "mature",
        "elderly": "mature",
        "senior": "mature",
    },
}


def _normalize_trait_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        return cleaned or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def normalize_audio_trait_value(key: str, value: Any) -> str | None:
    """Normalize GPT-audio/candidate trait values into canonical search tokens.

    Audio models and manifests often describe the same operational trait with
    slightly different strings (for example `en`, `US English`, `American`, or
    `north american English`).  Hard guardrails should not fail because of this
    surface-form variation, so retrieval canonicalizes known aliases before
    matching.  This is still an operational matching normalization, not a claim
    about identity, protected attributes, or precise geography.
    """

    normalized = _normalize_trait_value(value)
    if not normalized:
        return None
    aliases = TRAIT_VALUE_ALIASES.get(key, {})
    return aliases.get(normalized, normalized)


def effective_trait_weights(custom_weights: dict[str, float] | None = None) -> dict[str, float]:
    """Return default hybrid-search trait weights plus explicit overrides.

    Direct API callers often provide query audio traits but omit weights.  The
    default weights encode the product behavior we want for pseudo-speaker
    matching: accent is strongest; geography/tone/gender/language matter;
    prosody refines the rank.  Callers can override any key, including setting a
    weight to 0.0 to disable that trait's contribution.
    """

    weights = dict(DEFAULT_AUDIO_TRAIT_WEIGHTS)
    for key, value in (custom_weights or {}).items():
        weights[str(key)] = float(value)
    return weights


def equivalent_trait_match(key: str, query_value: str, sample_value: str) -> dict[str, Any] | None:
    """Return partial-match metadata for intentionally coarse trait groups.

    Equivalence is only used for soft reranking evidence. Hard filters still use
    explicit exact matching before this scorer is called.
    """

    groups = TRAIT_EQUIVALENCE_GROUPS.get(key)
    if not groups:
        return None
    for group_name, values in groups.items():
        if query_value in values and sample_value in values:
            return {
                "query": query_value,
                "candidate": sample_value,
                "group": group_name,
                "credit": TRAIT_EQUIVALENCE_CREDIT,
            }
    return None


def hard_guardrail_rejection_reason(
    *,
    key: str,
    requested_value: Any,
    sample_traits: dict[str, str],
    require_known_match: bool = False,
) -> str | None:
    """Return the rejection reason for a hard guardrail, if any.

    Language, gender presentation, and accent group are used as conservative
    operational guardrails.  The older search behavior only rejected explicit
    mismatches and allowed candidates with missing hard traits to pass through
    with a warning.  That is useful for exploratory search over sparse
    manifests, but it is too weak for assignment-time pseudo-speaker routing:
    a missing candidate gender/accent can win by vector score and then produce
    obvious voice drift.  ``require_known_match`` upgrades missing candidate
    hard traits into a fail-closed rejection.
    """

    normalized_request = normalize_audio_trait_value(key, requested_value)
    if not normalized_request:
        return None
    candidate_value = sample_traits.get(key)
    if candidate_value is None:
        if require_known_match:
            return f"missing_required_hard_guardrail:{key}"
        return None
    if candidate_value != normalized_request:
        return f"mismatched_hard_guardrail:{key}:{candidate_value}!={normalized_request}"
    return None


def _nested_dict(payload: Any, key: str) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get(key), dict):
        return dict(payload[key])
    return {}


def _voice_quality_profile_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    payload = metadata.get("voice_quality_profile")
    if not isinstance(payload, dict):
        return {}
    # Service responses are stored either directly or as an audit wrapper.
    if isinstance(payload.get("profile"), dict) or isinstance(payload.get("audio_traits"), dict):
        return payload
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return {}


def _metadata_first(sample: VoicePoolSample, keys: Sequence[str]) -> Any:
    """Return the first non-empty value from sample metadata/provenance/qc."""

    containers: list[dict[str, Any]] = []
    for payload in (sample.metadata, sample.provenance, sample.qc):
        if isinstance(payload, dict):
            containers.append(payload)
            nested = payload.get("audio_traits")
            if isinstance(nested, dict):
                containers.append(nested)
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value not in (None, "", "unknown"):
                return value
    return None


def _repo_file_candidates(sample: VoicePoolSample) -> list[str]:
    texts = [sample.voice_id, sample.sample_id]
    for payload in (sample.metadata, sample.provenance):
        if isinstance(payload, dict):
            texts.extend(str(payload.get(key, "")) for key in ["repo_file", "file", "path", "source_path"])
    return [text for text in texts if text]


def open_swara_catalog_traits(sample: VoicePoolSample) -> dict[str, Any]:
    """Infer trusted OpenSwara catalog fields from repo metadata.

    OpenSwara exposes stable metadata in repo paths / ids: language bucket,
    gender bucket, and whether the utterance is native language speech or
    English spoken with that native-language accent. These fields are used as
    product guardrails for voice replacement. They are catalog provenance, not
    identity or protected-attribute claims.
    """

    traits: dict[str, Any] = {}
    repo_file = ""
    for text in _repo_file_candidates(sample):
        normalized_path = str(text).replace("\\", "/")
        parts = [part for part in normalized_path.lower().split("/") if part]
        if len(parts) >= 4 and parts[0] == "voices":
            repo_file = normalized_path
            traits["catalog_language"] = normalize_audio_trait_value("language", parts[1]) or parts[1]
            traits["catalog_gender"] = normalize_audio_trait_value("gender_presentation", parts[2]) or parts[2]
            stem = Path(parts[-1]).stem.lower()
            traits["catalog_type"] = "english_accent" if "_english_open_swara_" in stem else "native"
            traits["is_english_accent"] = traits["catalog_type"] == "english_accent"
            break
    if not traits:
        # Fall back to filename semantics when only sample_id/voice_id survived.
        for text in _repo_file_candidates(sample):
            stem = Path(str(text).replace("\\", "/")).stem.lower()
            parts = stem.split("_")
            if len(parts) >= 4 and parts[-3:] == ["open", "swara", parts[-1]]:
                # Defensive fallback; the exact repo path parser above is preferred.
                pass
            if "_open_swara_" in stem:
                if "_english_open_swara_" in stem:
                    prefix = stem.split("_english_open_swara_")[0]
                    catalog_type = "english_accent"
                else:
                    prefix = stem.split("_open_swara_")[0]
                    catalog_type = "native"
                prefix_parts = prefix.split("_")
                if len(prefix_parts) >= 2:
                    traits["catalog_language"] = normalize_audio_trait_value("language", "_".join(prefix_parts[:-1])) or "_".join(prefix_parts[:-1])
                    traits["catalog_gender"] = normalize_audio_trait_value("gender_presentation", prefix_parts[-1]) or prefix_parts[-1]
                    traits["catalog_type"] = catalog_type
                    traits["is_english_accent"] = catalog_type == "english_accent"
                    break
    if repo_file:
        traits["repo_file"] = repo_file
    duration = _metadata_first(sample, ["duration_seconds", "duration_sec", "duration", "catalog_duration_seconds"])
    if duration is not None:
        try:
            traits["catalog_duration_seconds"] = float(duration)
        except (TypeError, ValueError):
            traits["catalog_duration_seconds"] = duration
    sample_rate = _metadata_first(sample, ["sample_rate", "sample_rate_hz", "catalog_sample_rate"])
    if sample_rate is not None:
        try:
            traits["catalog_sample_rate"] = int(float(sample_rate))
        except (TypeError, ValueError):
            traits["catalog_sample_rate"] = sample_rate
    return traits


def _derive_voice_body_traits(raw: dict[str, Any]) -> None:
    """Fill coarse vocal-body traits from available operational audio traits."""

    pitch = normalize_audio_trait_value("pitch_band", raw.get("pitch_band"))
    if raw.get("vocal_weight") in (None, "", "unknown"):
        if pitch == "high":
            raw["vocal_weight"] = "thin"
        elif pitch == "low":
            raw["vocal_weight"] = "thick"
        elif pitch == "mid":
            raw["vocal_weight"] = "medium"
    if raw.get("resonance") in (None, "", "unknown"):
        if pitch == "high":
            raw["resonance"] = "light"
        elif pitch == "low":
            raw["resonance"] = "deep"
        elif pitch == "mid":
            raw["resonance"] = "balanced"

def sample_audio_traits(sample: VoicePoolSample) -> dict[str, str]:
    """Return normalized searchable audio traits for a candidate voice.

    Traits can come from explicit manifest columns, metadata keys, or
    `metadata.audio_traits`. They are operational matching features only, not
    demographic or clinical claims.
    """

    metadata = dict(sample.metadata) if isinstance(sample.metadata, dict) else {}
    metadata_traits = _nested_dict(metadata, "audio_traits")
    voice_quality = _voice_quality_profile_payload(metadata)
    voice_quality_traits = _nested_dict(voice_quality, "audio_traits")
    voice_quality_profile = _nested_dict(voice_quality, "profile")
    raw: dict[str, Any] = {
        "language": sample.language,
        "pitch_band": sample.pitch_band,
        "speaking_rate": sample.speaking_rate,
        "energy": sample.energy,
        "gender_presentation": sample.gender_presentation,
        "accent_group": sample.accent_group,
        "geography_region": sample.geography_region,
        "tone": sample.tone,
    }
    for key in CANONICAL_AUDIO_TRAIT_KEYS:
        if raw.get(key) is None and key in metadata:
            raw[key] = metadata[key]
        # Audio-model-derived traits are the highest-precedence operational
        # search fields because source manifests may encode dataset buckets
        # rather than the trait actually audible in the clip.
        if key in metadata_traits and metadata_traits[key] is not None:
            raw[key] = metadata_traits[key]
        # Voice-quality singleton-derived traits refine the perceptual body/age
        # dimensions that GPT-audio captions can miss (e.g. child-like target,
        # thin-vs-thick resonance).  Do not let these silently override
        # language/accent/tone semantics; those remain query/caption fields.
        if key in {"pitch_band", "vocal_weight", "resonance", "voice_maturity"}:
            if key in voice_quality_traits and voice_quality_traits[key] is not None:
                raw[key] = voice_quality_traits[key]
            elif key in voice_quality_profile and voice_quality_profile[key] is not None:
                raw[key] = voice_quality_profile[key]
        elif key == "gender_presentation" and raw.get(key) in (None, "", "unknown"):
            if key in voice_quality_traits and voice_quality_traits[key] is not None:
                raw[key] = voice_quality_traits[key]
            elif key in voice_quality_profile and voice_quality_profile[key] is not None:
                raw[key] = voice_quality_profile[key]
    _derive_voice_body_traits(raw)
    # Keep existing conservative inference helpers in the searchable trait map.
    raw["gender_presentation"] = raw.get("gender_presentation") or infer_sample_gender_presentation(sample)
    raw["accent_group"] = raw.get("accent_group") or infer_sample_accent_group(sample)
    traits = {}
    for key, value in raw.items():
        normalized = normalize_audio_trait_value(key, value)
        if normalized and normalized != "unknown":
            traits[key] = normalized
    return traits


def request_audio_traits(request: VoiceSearchRequest) -> dict[str, str]:
    raw: dict[str, Any] = {}
    if isinstance(request.audio_traits, dict):
        raw.update(request.audio_traits)
    explicit = {
        "language": request.language,
        "pitch_band": request.pitch_band,
        "speaking_rate": request.speaking_rate,
        "energy": request.energy,
        "gender_presentation": request.gender_presentation,
        "accent_group": request.accent_group,
        "geography_region": request.geography_region,
        "tone": request.tone,
    }
    for key, value in explicit.items():
        if value is not None:
            raw[key] = value
    traits = {}
    for key, value in raw.items():
        normalized = normalize_audio_trait_value(str(key), value)
        if normalized and normalized != "unknown":
            traits[str(key)] = normalized
    return traits


def build_hybrid_search_explanation(
    sample_traits: dict[str, str],
    query_traits: dict[str, str],
    trait_debug: dict[str, Any],
) -> dict[str, Any]:
    """Build a review-facing explanation for a hybrid voice-pool candidate.

    `query_trait_match` is intentionally machine-oriented.  This companion
    summary separates conservative hard guardrails from noisy soft rerank traits
    and emits warnings that should be visible in review packets/UI.  It does not
    convert operational trait labels into demographic, identity, geography, or
    clinical claims.
    """

    matched = trait_debug.get("matched") if isinstance(trait_debug.get("matched"), dict) else {}
    equivalent = trait_debug.get("equivalent") if isinstance(trait_debug.get("equivalent"), dict) else {}
    mismatched = trait_debug.get("mismatched") if isinstance(trait_debug.get("mismatched"), dict) else {}
    missing_values = trait_debug.get("missing") if isinstance(trait_debug.get("missing"), list) else []
    missing = {str(key) for key in missing_values}
    hard_guardrails: dict[str, dict[str, Any]] = {}
    soft_rerank: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for key in HARD_GUARDRAIL_TRAITS:
        if key not in query_traits:
            continue
        status = "matched"
        if key in missing:
            status = "candidate_missing"
            warnings.append(f"missing_hard_guardrail:{key}")
        elif key in mismatched:
            status = "mismatched"
            warnings.append(f"mismatched_hard_guardrail:{key}")
        hard_guardrails[key] = {
            "status": status,
            "query": query_traits.get(key),
            "candidate": sample_traits.get(key),
        }
    for key in SOFT_RERANK_TRAITS:
        if key not in query_traits:
            continue
        status = "matched"
        detail: dict[str, Any] = {}
        if key in equivalent:
            status = "equivalent"
            detail = dict(equivalent[key]) if isinstance(equivalent[key], dict) else {}
        elif key in missing:
            status = "candidate_missing"
            warnings.append(f"missing_soft_trait:{key}")
        elif key in mismatched:
            status = "mismatched"
            warnings.append(f"mismatched_soft_trait:{key}")
        soft_rerank[key] = {
            "status": status,
            "query": query_traits.get(key),
            "candidate": sample_traits.get(key),
            **({"equivalence": detail} if detail else {}),
        }
    return {
        "hard_guardrails": hard_guardrails,
        "soft_rerank": soft_rerank,
        "warnings": warnings,
        "trait_score": trait_debug.get("score"),
        "claim_boundary": "Operational voice-matching explanation only; not identity, demographic, clinical, or precise geography evidence.",
    }


def trait_match_score(
    sample_traits: dict[str, str],
    query_traits: dict[str, str],
    *,
    trait_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    weights = effective_trait_weights(trait_weights)
    total_weight = 0.0
    matched_weight = 0.0
    matched: dict[str, str] = {}
    equivalent: dict[str, dict[str, Any]] = {}
    mismatched: dict[str, dict[str, str | None]] = {}
    missing: list[str] = []
    for key, query_value in query_traits.items():
        weight = weights.get(key, 1.0)
        total_weight += weight
        sample_value = sample_traits.get(key)
        if sample_value is None:
            missing.append(key)
        elif sample_value == query_value:
            matched_weight += weight
            matched[key] = query_value
        else:
            equivalent_match = equivalent_trait_match(key, query_value, sample_value)
            if equivalent_match is not None:
                matched_weight += weight * float(equivalent_match["credit"])
                equivalent[key] = equivalent_match
            else:
                mismatched[key] = {"query": query_value, "candidate": sample_value}
    score = matched_weight / total_weight if total_weight > 0 else 0.0
    return score, {
        "score": score,
        "matched": matched,
        "equivalent": equivalent,
        "mismatched": mismatched,
        "missing": missing,
        "total_weight": total_weight,
        "matched_weight": matched_weight,
    }


class VoicePoolIndex:
    """Simple exact-search index for Phase 2 contract testing.

    The index JSON format is deliberately portable:

    ```json
    {"embeddings": {"sample_id": [0.1, 0.2, ...]}}
    ```

    A later FAISS-backed index can implement the same search API.
    """

    def __init__(self, manifest: VoicePoolManifest, embeddings: dict[str, list[float]]):
        if not embeddings:
            raise VoicePoolSetupError("voice pool embedding index is empty; build ECAPA/FAISS index first")
        self.manifest = manifest
        self.embeddings = embeddings
        self.samples_by_id = {sample.sample_id: sample for sample in manifest.samples}
        missing = sorted(set(self.samples_by_id) - set(embeddings))
        if missing:
            raise VoicePoolSetupError(
                f"embedding index missing {len(missing)} sample(s), e.g. {missing[:3]}; rebuild voice pool index"
            )

    @classmethod
    def from_paths(cls, manifest_path: str | Path, index_path: str | Path) -> "VoicePoolIndex":
        manifest_path = Path(manifest_path)
        index_path = Path(index_path)
        if not manifest_path.exists():
            raise VoicePoolSetupError(f"voice pool manifest not found: {manifest_path}")
        if not index_path.exists():
            raise VoicePoolSetupError(f"voice pool embedding index not found: {index_path}; run voice pool builder first")
        manifest = VoicePoolManifest.from_path(manifest_path)
        payload = json.loads(index_path.read_text())
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, dict):
            raise VoicePoolSetupError("embedding index must contain an 'embeddings' object")
        return cls(manifest, {str(k): [float(x) for x in v] for k, v in embeddings.items()})

    def search(self, request: VoiceSearchRequest) -> list[VoiceSearchResult]:
        if request.voice_pool_id != self.manifest.voice_pool_id:
            raise AegisVoxContractError(
                f"request voice_pool_id={request.voice_pool_id!r} does not match loaded pool {self.manifest.voice_pool_id!r}"
            )
        candidates: list[VoiceSearchResult] = []
        for sample in self.manifest.samples:
            if sample.voice_id in request.exclude_voice_ids:
                continue
            sample_traits = sample_audio_traits(sample)
            query_traits = request_audio_traits(request)
            if any(
                hard_guardrail_rejection_reason(
                    key=key,
                    requested_value=requested_value,
                    sample_traits=sample_traits,
                    require_known_match=request.require_known_hard_trait_matches,
                )
                for key, requested_value in (
                    ("language", request.language),
                    ("gender_presentation", request.gender_presentation),
                    ("accent_group", request.accent_group),
                )
            ):
                continue
            trait_score, trait_debug = trait_match_score(
                sample_traits,
                query_traits,
                trait_weights=request.trait_weights,
            )
            # Geography and tone are deliberately soft reranking traits.
            # GPT-audio may use slightly different broad labels across query and
            # pool samples (e.g. conversational vs calm), so hard-filtering them
            # can eliminate otherwise good candidates. Language/gender/accent
            # remain the primary guardrail filters above.
            reasons = []
            if request.language:
                reasons.append(f"language={request.language}")
            if request.gender_presentation:
                reasons.append(f"gender_presentation={request.gender_presentation}")
            if request.accent_group:
                reasons.append(f"accent_group={request.accent_group}")
            if request.geography_region:
                reasons.append(f"geography_region={request.geography_region}")
            if request.tone:
                reasons.append(f"tone={request.tone}")
            if request.pitch_band and sample.pitch_band == request.pitch_band:
                reasons.append(f"pitch_band={request.pitch_band}")
            if request.speaking_rate and sample.speaking_rate == request.speaking_rate:
                reasons.append(f"speaking_rate={request.speaking_rate}")
            if request.energy and sample.energy == request.energy:
                reasons.append(f"energy={request.energy}")
            if query_traits:
                reasons.append(f"trait_match={trait_score:.3f}")
            sim: float | None = None
            distance: float | None = None
            score = 0.0
            if request.query_embedding is not None:
                sim = cosine_similarity(request.query_embedding, self.embeddings[sample.sample_id])
                distance = 1.0 - sim
                if request.min_source_distance is not None and distance < request.min_source_distance:
                    continue
                score = distance * request.vector_weight + trait_score * request.trait_weight
            else:
                # Retrieval without a source embedding is allowed only as metadata filtering.
                # It is deterministic and explicit; it is not a privacy assignment decision.
                score = trait_score * request.trait_weight if query_traits else float(len(reasons))
            enriched_metadata = sample.to_dict()
            enriched_metadata["audio_traits"] = sample_traits
            enriched_metadata["open_swara_catalog_traits"] = open_swara_catalog_traits(sample)
            enriched_metadata["query_trait_match"] = trait_debug
            enriched_metadata["hybrid_search_explanation"] = build_hybrid_search_explanation(
                sample_traits,
                query_traits,
                trait_debug,
            )
            candidates.append(
                VoiceSearchResult(
                    voice_id=sample.voice_id,
                    sample_id=sample.sample_id,
                    audio_uri=sample.audio_uri,
                    score=score,
                    cosine_similarity=sim,
                    source_target_distance=distance,
                    assignment_reason=", ".join(reasons) if reasons else "metadata/index candidate",
                    metadata=enriched_metadata,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: request.top_k]


def _tokenize_metadata_text(text: str) -> set[str]:
    return {part for part in text.lower().replace("-", "_").replace("/", "_").split("_") if part}


def infer_sample_gender_presentation(sample: VoicePoolSample) -> str:
    """Infer target voice gender presentation from explicit or source metadata.

    Open Swara currently encodes gender in repository paths and sample IDs rather
    than in a top-level manifest field. This helper is intentionally conservative
    and returns only `female`, `male`, or `unknown`.
    """

    explicit = sample.gender_presentation
    if not explicit and isinstance(sample.metadata, dict):
        explicit = sample.metadata.get("gender_presentation")
    if isinstance(explicit, str) and explicit.lower() in {"female", "male"}:
        return explicit.lower()
    texts = [sample.voice_id, sample.sample_id]
    if isinstance(sample.metadata, dict):
        texts.extend(str(sample.metadata.get(key, "")) for key in ["repo_file", "path", "source_path"])
    if isinstance(sample.provenance, dict):
        texts.extend(str(sample.provenance.get(key, "")) for key in ["repo_file", "path", "source_path"])
    tokens = set()
    for text in texts:
        tokens |= _tokenize_metadata_text(str(text))
    if "female" in tokens:
        return "female"
    if "male" in tokens:
        return "male"
    return "unknown"


def infer_sample_accent_group(sample: VoicePoolSample) -> str:
    """Infer a coarse target accent group from explicit or Open Swara metadata.

    Open Swara paths encode coarse language/accent buckets (for example
    `voices/arabic/male/arabic_male_english_open_swara_001.wav`). For product
    assignment we use this only as a matching constraint, not as a demographic
    claim.
    """

    explicit = sample.accent_group
    if not explicit and isinstance(sample.metadata, dict):
        explicit = sample.metadata.get("accent_group")
    if isinstance(explicit, str) and explicit and explicit.lower() != "unknown":
        return explicit.lower()
    if isinstance(sample.metadata, dict):
        repo_file = str(sample.metadata.get("repo_file", ""))
        parts = [part for part in repo_file.lower().replace("\\", "/").split("/") if part]
        if len(parts) >= 2 and parts[0] == "voices":
            return parts[1]
    if isinstance(sample.provenance, dict):
        repo_file = str(sample.provenance.get("repo_file", ""))
        parts = [part for part in repo_file.lower().replace("\\", "/").split("/") if part]
        if len(parts) >= 2 and parts[0] == "voices":
            return parts[1]
    if sample.language and sample.language.lower() != "unknown":
        return sample.language.lower()
    return "unknown"


def write_voice_pool_manifest(path: str | Path, manifest: VoicePoolManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")


def write_embedding_index(path: str | Path, embeddings: dict[str, Iterable[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"embeddings": {key: [float(x) for x in value] for key, value in embeddings.items()}}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
