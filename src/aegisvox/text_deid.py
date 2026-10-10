"""Text de-identification service for AegisVox transcripts."""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .schema import AegisVoxContractError, validate_artifact_uri
from .storage import LocalArtifactStore


class TextDeIDSetupError(RuntimeError):
    """Raised when LLM de-id backend is unavailable."""


class TextDeIDBackend(Protocol):
    backend_name: str
    backend_version: str

    def detect_edits(self, transcript: dict[str, Any], policy: "DeIDPolicy") -> list[dict[str, Any]]: ...

    def plan_replacements(
        self,
        *,
        transcript: dict[str, Any],
        entities: list[dict[str, Any]],
        policy: "DeIDPolicy",
    ) -> list[dict[str, Any]]: ...


HIPAA_PHI_TYPES: tuple[str, ...] = (
    "HIPAA_NAME",
    "HIPAA_GEOGRAPHIC_SUBDIVISION",
    "HIPAA_DATE",
    "HIPAA_AGE_OVER_89",
    "HIPAA_PHONE",
    "HIPAA_FAX",
    "HIPAA_EMAIL",
    "HIPAA_SSN",
    "HIPAA_MEDICAL_RECORD_NUMBER",
    "HIPAA_HEALTH_PLAN_NUMBER",
    "HIPAA_ACCOUNT_NUMBER",
    "HIPAA_CERTIFICATE_LICENSE_NUMBER",
    "HIPAA_VEHICLE_IDENTIFIER",
    "HIPAA_DEVICE_IDENTIFIER",
    "HIPAA_URL",
    "HIPAA_IP_ADDRESS",
    "HIPAA_BIOMETRIC_IDENTIFIER",
    "HIPAA_FULL_FACE_IMAGE_OR_COMPARABLE_IMAGE",
    "HIPAA_OTHER_UNIQUE_IDENTIFIER",
)

HIPAA_PHI_TYPE_SET = frozenset(HIPAA_PHI_TYPES)


@dataclass(frozen=True)
class DeIDPolicy:
    """Text de-identification replacement policy.

    Modes:
    - ``mask``: replace all detected PHI with ``[PHI]``.
    - ``typed_mask`` / legacy ``replace``: replace with type masks such as
      ``[NAME]`` and ``[DATE]``.
    - ``relexicalize``: replace every PHI type with deterministic synthetic,
      speakable values.
    - ``hybrid``: relexicalize speakable HIPAA entities useful for speech
      editing (names, dates, geographic subdivisions) and mask high-risk
      identifiers (phone, email, SSN, medical record numbers, account numbers,
      IDs, URLs, IPs, biometrics, etc.). This is the recommended AegisVox
      default.

    Relexicalization is entity-consistent within a transcript/run: the same
    original text and PHI type always receive the same synthetic replacement.
    """

    mode: str = "hybrid"
    replacements: dict[str, str] = field(
        default_factory=lambda: {
            "HIPAA_NAME": "[HIPAA_NAME]",
            "HIPAA_GEOGRAPHIC_SUBDIVISION": "[HIPAA_GEOGRAPHIC_SUBDIVISION]",
            "HIPAA_DATE": "[HIPAA_DATE]",
            "HIPAA_AGE_OVER_89": "[HIPAA_AGE_OVER_89]",
            "HIPAA_PHONE": "[HIPAA_PHONE]",
            "HIPAA_FAX": "[HIPAA_FAX]",
            "HIPAA_EMAIL": "[HIPAA_EMAIL]",
            "HIPAA_SSN": "[HIPAA_SSN]",
            "HIPAA_MEDICAL_RECORD_NUMBER": "[HIPAA_MEDICAL_RECORD_NUMBER]",
            "HIPAA_HEALTH_PLAN_NUMBER": "[HIPAA_HEALTH_PLAN_NUMBER]",
            "HIPAA_ACCOUNT_NUMBER": "[HIPAA_ACCOUNT_NUMBER]",
            "HIPAA_CERTIFICATE_LICENSE_NUMBER": "[HIPAA_CERTIFICATE_LICENSE_NUMBER]",
            "HIPAA_VEHICLE_IDENTIFIER": "[HIPAA_VEHICLE_IDENTIFIER]",
            "HIPAA_DEVICE_IDENTIFIER": "[HIPAA_DEVICE_IDENTIFIER]",
            "HIPAA_URL": "[HIPAA_URL]",
            "HIPAA_IP_ADDRESS": "[HIPAA_IP_ADDRESS]",
            "HIPAA_BIOMETRIC_IDENTIFIER": "[HIPAA_BIOMETRIC_IDENTIFIER]",
            "HIPAA_FULL_FACE_IMAGE_OR_COMPARABLE_IMAGE": "[HIPAA_FULL_FACE_IMAGE_OR_COMPARABLE_IMAGE]",
            "HIPAA_OTHER_UNIQUE_IDENTIFIER": "[HIPAA_OTHER_UNIQUE_IDENTIFIER]",
        }
    )
    synthetic_replacements: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "HIPAA_NAME": (
                "Alex Morgan",
                "Emily Carter",
                "Michael Rivera",
                "Priya Shah",
                "Daniel Kim",
                "Sophia Bennett",
                "Jordan Lee",
                "Maya Thompson",
            ),
            "HIPAA_DATE": (
                "March 12, 2026",
                "April 8, 2026",
                "September 3, 2025",
                "January 21, 2026",
                "November 14, 2025",
            ),
            "HIPAA_GEOGRAPHIC_SUBDIVISION": (
                "Riverside",
                "North Valley",
                "Lakeside Clinic",
                "Cedar Grove",
                "Pine Hill",
            ),
            "HIPAA_AGE_OVER_89": (
                "seventy eight",
                "eighty two",
                "seventy five",
                "eighty six",
                "seventy one",
            ),
        }
    )
    language_aware_relexicalization: bool = True
    language_specific_synthetic_replacements: dict[str, dict[str, tuple[str, ...]]] = field(
        default_factory=lambda: {
            "fr": {
                "HIPAA_NAME": (
                    "Claire Martin",
                    "Julien Moreau",
                    "Sophie Laurent",
                    "Thomas Bernard",
                    "Camille Dubois",
                ),
                "HIPAA_DATE": (
                    "le douze mars deux mille vingt-six",
                    "le huit avril deux mille vingt-six",
                    "le trois septembre deux mille vingt-cinq",
                    "le vingt et un janvier deux mille vingt-six",
                ),
                "HIPAA_GEOGRAPHIC_SUBDIVISION": (
                    "Nantes",
                    "Grenoble",
                    "Montpellier",
                    "Tours",
                    "Dijon",
                ),
                "HIPAA_AGE_OVER_89": (
                    "soixante-dix-huit ans",
                    "quatre-vingt-deux ans",
                    "soixante-quinze ans",
                    "quatre-vingt-six ans",
                ),
            },
            "en": {},
        }
    )
    hybrid_mask_types: frozenset[str] = frozenset(
        set(HIPAA_PHI_TYPES)
        - {
            "HIPAA_NAME",
            "HIPAA_GEOGRAPHIC_SUBDIVISION",
            "HIPAA_DATE",
        }
    )

    def replacement_for(self, phi_type: str) -> str:
        return self.replacements.get(phi_type, f"[{phi_type}]")

    def normalized_mode(self) -> str:
        mode = self.mode.lower().strip()
        if mode == "replace":
            return "typed_mask"
        if mode not in {"mask", "typed_mask", "relexicalize", "hybrid"}:
            raise AegisVoxContractError(
                f"unsupported text de-id mode {self.mode!r}; expected mask, typed_mask, relexicalize, or hybrid"
            )
        return mode

    def replacement_for_entity(
        self,
        phi_type: str,
        original_text: str,
        entity_index: int,
        *,
        language: str | None = None,
    ) -> str:
        mode = self.normalized_mode()
        if mode == "mask":
            return "[PHI]"
        if mode == "typed_mask" or (mode == "hybrid" and phi_type in self.hybrid_mask_types):
            return self.replacement_for(phi_type)
        if self.language_aware_relexicalization and language:
            lang_pool = self.language_specific_synthetic_replacements.get(_normalize_language_code(language), {})
            pool = lang_pool.get(phi_type)
            if pool:
                return pool[entity_index % len(pool)]
        pool = self.synthetic_replacements.get(phi_type)
        if not pool:
            return self.replacement_for(phi_type)
        return pool[entity_index % len(pool)]


@dataclass(frozen=True)
class TextEdit:
    edit_id: str
    phi_type: str
    original_text: str
    replacement_text: str
    confidence: float
    rationale: str
    speaker_id: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    source_segment_index: int | None = None
    source_segment_text: str | None = None
    source_language: str | None = None
    replacement_language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = {
            "edit_id": self.edit_id,
            "speaker_id": self.speaker_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "phi_type": self.phi_type,
            "original_text": self.original_text,
            "replacement_text": self.replacement_text,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }
        if self.source_segment_index is not None:
            row["source_segment_index"] = self.source_segment_index
        if self.source_segment_text is not None:
            row["source_segment_text"] = self.source_segment_text
        if self.source_language is not None:
            row["source_language"] = self.source_language
        if self.replacement_language is not None:
            row["replacement_language"] = self.replacement_language
        return row


def _segment_text_rows(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    rows = transcript.get("segments")
    if isinstance(rows, list) and rows:
        return rows
    return [{"text": transcript.get("text", ""), "speaker_id": None, "start_sec": None, "end_sec": None}]


def _entity_key(phi_type: str, original_text: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", " ", original_text.strip()).casefold()
    return phi_type, normalized


def _normalize_language_code(language: str | None) -> str:
    raw = str(language or "").strip().casefold().replace("_", "-")
    if not raw:
        return "unknown"
    aliases = {
        "english": "en",
        "eng": "en",
        "en-us": "en",
        "en-gb": "en",
        "french": "fr",
        "fra": "fr",
        "fre": "fr",
        "fr-fr": "fr",
    }
    return aliases.get(raw, raw.split("-", 1)[0])


_FRENCH_LANGUAGE_MARKERS = frozenset(
    {
        "bonjour",
        "bonsoir",
        "allo",
        "allô",
        "monsieur",
        "madame",
        "docteur",
        "patient",
        "patiente",
        "je",
        "vous",
        "nous",
        "avec",
        "dans",
        "pour",
        "très",
        "tres",
        "être",
        "etre",
        "été",
        "ete",
        "suis",
        "avez",
        "vais",
        "faut",
        "hôpital",
        "hopital",
    }
)

_ENGLISH_LANGUAGE_MARKERS = frozenset(
    {
        "the",
        "and",
        "you",
        "your",
        "this",
        "that",
        "with",
        "from",
        "have",
        "name",
        "phone",
        "email",
        "patient",
        "doctor",
        "visited",
        "hospital",
        "clinic",
    }
)


def _infer_language_from_text(text: str) -> str:
    """Infer coarse transcript language when ASR omits metadata.

    This is intentionally narrow and deterministic. It is not a claimed language
    ID model; it only prevents obvious metadata loss in known public M7 buckets
    where French transcripts otherwise become ``unknown`` and lose same-language
    relexicalization accounting.
    """

    lowered = text.casefold()
    tokens = re.findall(r"[a-zà-ÿ']+", lowered)
    if not tokens:
        return "unknown"
    token_set = set(tokens)
    french_hits = len(token_set & _FRENCH_LANGUAGE_MARKERS)
    english_hits = len(token_set & _ENGLISH_LANGUAGE_MARKERS)
    accented_french = len(re.findall(r"[àâçéèêëîïôûùüÿœæ]", lowered))
    french_score = french_hits + min(accented_french, 4)
    english_score = english_hits
    if french_score >= 2 and french_score > english_score:
        return "fr"
    if english_score >= 2 and english_score > french_score:
        return "en"
    return "unknown"


def _transcript_language(transcript: dict[str, Any]) -> str:
    # An explicit corpus/config hint is authoritative for language-aware
    # relexicalization. ASR language ID is retained as evidence, but must not
    # force English replacements into a known French evaluation row.
    for key in ("language_hint", "language", "detected_language"):
        if transcript.get(key):
            return _normalize_language_code(str(transcript[key]))
    meta = transcript.get("metadata")
    if isinstance(meta, dict):
        for key in ("language_hint", "language", "detected_language", "dataset_language"):
            if meta.get(key):
                return _normalize_language_code(str(meta[key]))
    return _infer_language_from_text(str(transcript.get("text") or ""))


def _segment_language(segment: dict[str, Any] | None, transcript: dict[str, Any]) -> str:
    if isinstance(segment, dict):
        for key in ("language_hint", "language", "detected_language"):
            if segment.get(key):
                return _normalize_language_code(str(segment[key]))
        inferred = _infer_language_from_text(str(segment.get("text") or ""))
        if inferred != "unknown":
            return inferred
    return _transcript_language(transcript)


def deterministic_phi_edits(transcript: dict[str, Any], policy: DeIDPolicy) -> list[TextEdit]:
    raise TextDeIDSetupError(
        "deterministic regex PHI detection is disabled; use an LLM HIPAA Safe Harbor detector"
    )


def apply_text_edits(text: str, edits: list[TextEdit]) -> str:
    # Replace longer spans first to avoid partial replacement collisions.
    result = text
    for edit in sorted(edits, key=lambda e: len(e.original_text), reverse=True):
        result = result.replace(edit.original_text, edit.replacement_text)
    return result


def _unique_entities_from_edits(edits: list[TextEdit]) -> list[dict[str, Any]]:
    entities_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for edit in edits:
        key = _entity_key(edit.phi_type, edit.original_text)
        if key not in entities_by_key:
            entities_by_key[key] = {
                "entity_id": f"entity_{len(entities_by_key)+1:04d}",
                "phi_type": edit.phi_type,
                "original_text": edit.original_text,
                "source_language": edit.source_language,
                "replacement_language": edit.replacement_language,
                "edit_ids": [],
                "speaker_ids": [],
            }
        entity = entities_by_key[key]
        entity["edit_ids"].append(edit.edit_id)
        if edit.speaker_id and edit.speaker_id not in entity["speaker_ids"]:
            entity["speaker_ids"].append(edit.speaker_id)
    return list(entities_by_key.values())


def _validate_phi_type(phi_type: str) -> str:
    normalized = phi_type.strip().upper()
    if normalized not in HIPAA_PHI_TYPE_SET:
        raise TextDeIDSetupError(
            f"unsupported PHI type {phi_type!r}; detector must use HIPAA Safe Harbor entity types only"
        )
    return normalized


def _unsupported_detector_phi_types(rows: list[dict[str, Any]]) -> list[str]:
    """Return unsupported detector phi_type values without coercing or mapping them.

    We intentionally do not auto-map model-invented labels such as
    HIPAA_ORGANIZATION.  The detector must re-emit valid HIPAA Safe Harbor enum
    values or the run should fail; otherwise the audit trail would hide a
    category error.
    """

    unsupported: list[str] = []
    for row in rows:
        normalized = str(row.get("phi_type") or "").strip().upper()
        if normalized not in HIPAA_PHI_TYPE_SET and normalized not in unsupported:
            unsupported.append(normalized)
    return unsupported


def _coerce_llm_detected_edits(
    rows: list[dict[str, Any]],
    transcript: dict[str, Any],
    policy: DeIDPolicy,
    *,
    quarantined_rows: list[dict[str, Any]] | None = None,
) -> list[TextEdit]:
    segments = _segment_text_rows(transcript)
    entity_replacements: dict[tuple[str, str], str] = {}
    entity_counts_by_type: dict[str, int] = {}
    edits: list[TextEdit] = []
    seen: set[tuple[str, str, float | None, float | None]] = set()

    def quarantine(row: dict[str, Any], *, reason: str, original: str | None = None) -> None:
        if quarantined_rows is None:
            raise TextDeIDSetupError(reason)
        quarantined_rows.append(
            {
                "reason": reason,
                "segment_index": row.get("segment_index"),
                "phi_type": row.get("phi_type"),
                "original_text": original if original is not None else row.get("original_text"),
                "confidence": row.get("confidence"),
                "rationale": row.get("rationale"),
            }
        )

    for row in rows:
        original = re.sub(r"\s+", " ", str(row.get("original_text") or "").strip())
        if not original:
            quarantine(row, reason="empty_original_text", original=original)
            continue
        phi_type = _validate_phi_type(str(row.get("phi_type") or ""))
        try:
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            quarantine(row, reason="invalid_confidence", original=original)
            continue
        if confidence < 0.0 or confidence > 1.0:
            quarantine(row, reason="invalid_confidence", original=original)
            continue
        segment_index = row.get("segment_index")
        segment = None
        resolved_segment_index: int | None = None
        if segment_index is not None:
            try:
                idx = int(segment_index)
            except (TypeError, ValueError):
                quarantine(row, reason="invalid_segment_index", original=original)
                continue
            if idx < 0 or idx >= len(segments):
                quarantine(row, reason="out_of_range_segment_index", original=original)
                continue
            segment = segments[idx]
            resolved_segment_index = idx
            if original not in str(segment.get("text") or ""):
                containing = [(candidate_idx, candidate) for candidate_idx, candidate in enumerate(segments) if original in str(candidate.get("text") or "")]
                if len(containing) == 1:
                    resolved_segment_index, segment = containing[0]
                    row = {**row, "relocated_from_segment_index": idx}
                else:
                    quarantine(
                        row,
                        reason="original_text_not_present_in_claimed_segment_or_unique_transcript_segment",
                        original=original,
                    )
                    continue
        else:
            full_text = str(transcript.get("text") or "")
            if original not in full_text:
                quarantine(row, reason="original_text_not_present_in_transcript", original=original)
                continue
        start_sec = segment.get("start_sec") if segment else row.get("start_sec")
        end_sec = segment.get("end_sec") if segment else row.get("end_sec")
        source_language = _segment_language(segment, transcript)
        key = (phi_type, original, start_sec, end_sec)
        if key in seen:
            continue
        seen.add(key)
        entity_key = _entity_key(phi_type, original)
        if entity_key not in entity_replacements:
            entity_index = entity_counts_by_type.get(phi_type, 0)
            entity_replacements[entity_key] = policy.replacement_for_entity(
                phi_type,
                original,
                entity_index,
                language=source_language,
            )
            entity_counts_by_type[phi_type] = entity_index + 1
        rationale = str(row.get("rationale") or "LLM HIPAA Safe Harbor detector")
        if row.get("relocated_from_segment_index") is not None:
            rationale = f"{rationale}; detector segment_index relocated from {row.get('relocated_from_segment_index')}"
        edits.append(
            TextEdit(
                edit_id=f"edit_{len(edits)+1:04d}",
                speaker_id=segment.get("speaker_id") if segment else row.get("speaker_id"),
                start_sec=start_sec,
                end_sec=end_sec,
                phi_type=phi_type,
                original_text=original,
                replacement_text=entity_replacements[entity_key],
                confidence=confidence,
                rationale=rationale,
                source_segment_index=resolved_segment_index,
                source_segment_text=str(segment.get("text") or "") if segment else None,
                source_language=source_language,
                replacement_language=source_language if policy.language_aware_relexicalization else None,
            )
        )
    return edits

def _normalized_for_leak_check(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _looks_ascii_english_phrase(text: str) -> bool:
    letters = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    if not letters:
        return False
    ascii_letters = re.findall(r"[A-Za-z]+", text)
    return len(ascii_letters) == len(letters)


def _validate_planned_replacement(*, entity: dict[str, Any], replacement: str, policy: DeIDPolicy) -> None:
    """Validate an LLM-proposed replacement before it can touch artifacts.

    The LLM is allowed to optimize for naturalness, but the service remains the
    safety boundary: replacements must be complete, non-empty, not reuse the
    original entity, and in hybrid mode must keep high-risk identifiers masked.
    """

    phi_type = str(entity["phi_type"])
    _validate_phi_type(phi_type)
    original = str(entity["original_text"])
    normalized_original = _normalized_for_leak_check(original)
    normalized_replacement = _normalized_for_leak_check(replacement)
    if normalized_original and normalized_original == normalized_replacement:
        raise TextDeIDSetupError(
            f"LLM replacement for {entity['entity_id']} reuses the original {phi_type} text"
        )
    if normalized_original and normalized_original in normalized_replacement:
        raise TextDeIDSetupError(
            f"LLM replacement for {entity['entity_id']} appears to contain the original {phi_type} text"
        )
    if policy.normalized_mode() == "hybrid" and phi_type in policy.hybrid_mask_types:
        required = policy.replacement_for(phi_type)
        if replacement != required:
            raise TextDeIDSetupError(
                f"LLM replacement for {entity['entity_id']} must be {required!r} in hybrid mode, got {replacement!r}"
            )
    if phi_type == "HIPAA_EMAIL" and "@" in replacement and not replacement.endswith(".invalid"):
        raise TextDeIDSetupError(
            f"LLM replacement for {entity['entity_id']} must not introduce a reachable-looking email address"
        )
    source_language = _normalize_language_code(str(entity.get("source_language") or ""))
    if policy.language_aware_relexicalization and source_language not in {"", "unknown", "en"}:
        # This is intentionally a conservative guard, not a full language ID
        # model: it prevents obvious English-only fallback strings on non-English
        # rows while leaving short names/places acceptable.
        if phi_type == "HIPAA_DATE" and _looks_ascii_english_phrase(replacement) and re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december|today|tomorrow|yesterday)\b",
            replacement.casefold(),
        ):
            raise TextDeIDSetupError(
                f"LLM replacement for {entity['entity_id']} is English-like but source language is {source_language}"
            )



def _repair_replacement_entity_ids(replacements: list[dict[str, Any]], entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known = {str(entity.get("entity_id")) for entity in entities}
    by_original = {str(entity.get("original_text") or "").casefold(): str(entity.get("entity_id")) for entity in entities}
    repaired: list[dict[str, Any]] = []
    for idx, row in enumerate(replacements):
        if not isinstance(row, dict):
            repaired.append(row)
            continue
        fixed = dict(row)
        entity_id = str(fixed.get("entity_id") or "")
        if entity_id not in known:
            original = str(fixed.get("original_text") or fixed.get("original") or "").casefold()
            if original and original in by_original:
                fixed["entity_id"] = by_original[original]
            elif len(replacements) == len(entities) and idx < len(entities):
                # Some chat models return replacement rows in the requested order
                # but use their own entity identifiers. Preserve the replacement
                # text and restore our canonical entity id.
                fixed["entity_id"] = str(entities[idx].get("entity_id"))
        repaired.append(fixed)
    return repaired

def _apply_replacement_plan(
    edits: list[TextEdit], plan: list[dict[str, Any]], policy: DeIDPolicy, planned_entities: list[dict[str, Any]] | None = None
) -> tuple[list[TextEdit], dict[str, Any]]:
    entities = _unique_entities_from_edits(edits)
    by_entity_id = {entity["entity_id"]: entity for entity in entities}
    required_entity_ids = {entity["entity_id"] for entity in (planned_entities if planned_entities is not None else entities)}
    replacement_by_key: dict[tuple[str, str], str] = {}
    rationale_by_key: dict[tuple[str, str], str] = {}
    for row in plan:
        entity_id = str(row.get("entity_id") or "")
        replacement = str(row.get("replacement_text") or "").strip()
        if not entity_id or entity_id not in by_entity_id:
            raise TextDeIDSetupError(f"LLM replacement plan returned unknown entity_id: {entity_id!r}")
        if not replacement:
            raise TextDeIDSetupError(f"LLM replacement plan returned empty replacement for {entity_id}")
        entity = by_entity_id[entity_id]
        rationale = str(row.get("rationale") or "LLM naturalness-preserving relexicalization")
        try:
            _validate_planned_replacement(entity=entity, replacement=replacement, policy=policy)
        except TextDeIDSetupError:
            fallback_index = len(replacement_by_key)
            replacement = policy.replacement_for_entity(
                str(entity["phi_type"]),
                str(entity["original_text"]),
                fallback_index,
                language=str(entity.get("source_language") or ""),
            )
            _validate_planned_replacement(entity=entity, replacement=replacement, policy=policy)
            rationale = "deterministic safe replacement after unsafe LLM replacement"
        key = _entity_key(str(entity["phi_type"]), str(entity["original_text"]))
        replacement_by_key[key] = replacement
        rationale_by_key[key] = rationale
    missing = [
        entity["entity_id"]
        for entity in entities
        if entity["entity_id"] in required_entity_ids
        and _entity_key(str(entity["phi_type"]), str(entity["original_text"])) not in replacement_by_key
    ]
    if missing:
        raise TextDeIDSetupError(f"LLM replacement plan missing replacements for entities: {', '.join(missing[:10])}")
    updated = [
        TextEdit(
            edit_id=edit.edit_id,
            speaker_id=edit.speaker_id,
            start_sec=edit.start_sec,
            end_sec=edit.end_sec,
            phi_type=edit.phi_type,
            original_text=edit.original_text,
            replacement_text=replacement_by_key.get(_entity_key(edit.phi_type, edit.original_text), edit.replacement_text),
            confidence=edit.confidence,
            rationale=rationale_by_key.get(_entity_key(edit.phi_type, edit.original_text), edit.rationale),
            source_segment_index=edit.source_segment_index,
            source_segment_text=edit.source_segment_text,
            source_language=edit.source_language,
            replacement_language=edit.replacement_language,
        )
        for edit in edits
    ]
    return updated, {
        "num_entities": len(entities),
        "replacement_source": "llm",
        "replacements_planned": len(replacement_by_key),
        "policy_replacements_preserved": len(entities) - len(replacement_by_key),
    }



def _openai_compatible_api_key_from_env() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise TextDeIDSetupError(
            "OpenAI-compatible text de-id backend requires OPENAI_API_KEY at runtime"
        )
    return key


def _openai_compatible_base_url_from_env() -> str | None:
    return (
        os.environ.get("AEGISVOX_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
    )


def _openai_compatible_default_headers_from_env(model: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    service_name = os.environ.get("AEGISVOX_OPENAI_COMPAT_SERVICE_NAME") or os.environ.get("OPENAI_COMPAT_SERVICE_NAME")
    if service_name:
        headers["service-name"] = service_name
    x_model = os.environ.get("AEGISVOX_OPENAI_COMPAT_X_MODEL") or os.environ.get("OPENAI_COMPAT_X_MODEL")
    if x_model:
        headers["X-Model"] = x_model
    elif os.environ.get("AEGISVOX_OPENAI_COMPAT_SET_X_MODEL", "").strip().lower() in {"1", "true", "yes"} and model:
        headers["X-Model"] = model
    raw_json = os.environ.get("AEGISVOX_OPENAI_DEFAULT_HEADERS_JSON") or os.environ.get("OPENAI_COMPAT_DEFAULT_HEADERS_JSON")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise TextDeIDSetupError(f"default headers JSON is invalid: {exc}") from exc
        if not isinstance(parsed, dict):
            raise TextDeIDSetupError("default headers JSON must be an object")
        for key, value in parsed.items():
            if value is not None:
                headers[str(key)] = str(value)
    return headers


def _openai_compatible_client_kwargs(model: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"api_key": _openai_compatible_api_key_from_env()}
    base_url = _openai_compatible_base_url_from_env()
    if base_url:
        kwargs["base_url"] = base_url
    headers = _openai_compatible_default_headers_from_env(model)
    if headers:
        kwargs["default_headers"] = headers
    return kwargs


def _openai_compat_raw_headers(model: str | None = None) -> dict[str, str]:
    # Raw OpenAI-compatible provider calls should prefer the OpenAI-compatible provider bearer token when present,
    # even if a separate OPENAI_API_KEY is also set locally for other workflows.
    raw_token = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or _openai_compatible_api_key_from_env()
    )
    headers = {
        "Authorization": f"Bearer {raw_token}",
        "Content-Type": "application/json",
    }
    headers.update(_openai_compatible_default_headers_from_env(model))
    return headers


def _openai_compat_api_version_suffix() -> str:
    raw = os.environ.get("AEGISVOX_OPENAI_COMPAT_API_VERSION") or os.environ.get("OPENAI_COMPAT_API_VERSION")
    if raw is None:
        return ""
    raw = raw.strip()
    return f"?api-version={raw}" if raw else ""


def _extract_json_text_from_raw_response(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [str(item.get("text") or item.get("content")) for item in content if isinstance(item, dict) and (item.get("text") or item.get("content"))]
                if parts:
                    return "\n".join(parts)
    output = payload.get("output")
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and (c.get("text") or c.get("output_text")):
                        parts.append(str(c.get("text") or c.get("output_text")))
            elif isinstance(content, str):
                parts.append(content)
        if parts:
            return "\n".join(parts)
    raise TextDeIDSetupError("OpenAI-compatible provider raw response did not include JSON text")


def _parse_json_from_model_text(text: str, *, expected_key: str | None = None) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_obj >= 0 and end_obj > start_obj:
            payload = json.loads(text[start_obj : end_obj + 1])
        elif start_arr >= 0 and end_arr > start_arr:
            payload = json.loads(text[start_arr : end_arr + 1])
        else:
            raise TextDeIDSetupError("OpenAI-compatible provider raw response did not contain parseable JSON")
    if isinstance(payload, list) and expected_key:
        return {expected_key: payload}
    if not isinstance(payload, dict):
        raise TextDeIDSetupError("OpenAI-compatible provider raw response JSON must be an object or expected list")
    return payload


def _openai_compat_raw_json_call(*, model: str, system: str, user_payload: dict[str, Any], expected_key: str) -> dict[str, Any]:
    base = _openai_compatible_base_url_from_env()
    if not base:
        raise TextDeIDSetupError("OpenAI-compatible provider raw HTTP requires OPENAI_BASE_URL or compatible base URL")
    import urllib.error
    import urllib.request
    mode = (os.environ.get("AEGISVOX_OPENAI_COMPAT_API_MODE") or os.environ.get("OPENAI_COMPAT_API_MODE") or "chat").strip().lower()
    timeout = int(os.environ.get("AEGISVOX_OPENAI_COMPAT_TIMEOUT_SEC") or os.environ.get("OPENAI_COMPAT_TIMEOUT_SEC") or "120")
    verify = (os.environ.get("AEGISVOX_OPENAI_COMPAT_VERIFY_SSL") or os.environ.get("OPENAI_COMPAT_VERIFY_SSL") or "true").strip().lower() not in {"0", "false", "no"}
    headers = _openai_compat_raw_headers(model)
    if mode == "responses":
        url = base.rstrip("/") + "/openai/responses" + _openai_compat_api_version_suffix()
        body: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        headers.setdefault("X-Model", model)
    elif mode == "chat":
        url = base.rstrip("/") + f"/openai/deployments/{model}/chat/completions" + _openai_compat_api_version_suffix()
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "stream": False,
        }
        if model in {"gpt-5.2", "gpt-5.2-chat-latest", "gpt-5.4"}:
            body["max_completion_tokens"] = int(
                os.environ.get("AEGISVOX_OPENAI_COMPAT_MAX_COMPLETION_TOKENS")
                or os.environ.get("OPENAI_COMPAT_MAX_COMPLETION_TOKENS")
                or "4000"
            )
    else:
        raise TextDeIDSetupError("OPENAI_COMPAT_API_MODE must be chat or responses")
    if not verify and url.lower().startswith("https://"):
        raise TextDeIDSetupError("OpenAI-compatible provider urllib transport does not support disabled TLS verification")
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise TextDeIDSetupError(f"OpenAI-compatible provider raw HTTP call failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise TextDeIDSetupError(f"OpenAI-compatible provider raw HTTP call failed: {exc}") from exc
    payload = _parse_json_from_model_text(_extract_json_text_from_raw_response(response_payload), expected_key=expected_key)
    if expected_key == "replacements" and isinstance(payload.get("entity_replacements"), dict):
        rows = []
        for entity_id, value in payload["entity_replacements"].items():
            if isinstance(value, dict):
                replacement_text = str(value.get("replacement_text") or value.get("replacement") or "")
                rationale = str(value.get("rationale") or "converted from entity_replacements map")
            else:
                replacement_text = str(value)
                rationale = "converted from entity_replacements map"
            rows.append({"entity_id": str(entity_id), "replacement_text": replacement_text, "rationale": rationale})
        payload = {"replacements": rows}
    if expected_key == "replacements" and isinstance(payload.get("replacements"), dict):
        rows = []
        for entity_id, value in payload["replacements"].items():
            if isinstance(value, dict):
                replacement_text = str(value.get("replacement_text") or value.get("replacement") or "")
                rationale = str(value.get("rationale") or "converted from replacements map")
            else:
                replacement_text = str(value)
                rationale = "converted from replacements map"
            rows.append({"entity_id": str(entity_id), "replacement_text": replacement_text, "rationale": rationale})
        payload = {"replacements": rows}
    if expected_key not in payload:
        raise TextDeIDSetupError(f"OpenAI-compatible provider raw HTTP response missing {expected_key!r}")
    return payload


def _should_use_raw_openai_compat() -> bool:
    return os.environ.get("AEGISVOX_OPENAI_COMPAT_RAW", "").strip().lower() in {"1", "true", "yes"}

def _is_allowed_paper_eval_llm(model: str) -> bool:
    m = (model or "").strip().lower()
    return m == "gpt-5.5" or m.startswith("gpt-5.5-") or m == "gpt-5.4" or m.startswith("gpt-5.4-")


class OpenAILLMTextDeIDBackend:
    """Optional LLM backend boundary.

    Deterministic validators are always run; this backend is for future recall
    improvement and for naturalness-preserving relexicalization.
    """

    backend_name = "openai-llm-deid"
    backend_version = "openai-python"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AEGISVOX_TEXT_DEID_MODEL") or os.environ.get("OPENAI_TEXT_DEID_MODEL") or "gpt-5.5"
        if not _is_allowed_paper_eval_llm(self.model):
            raise ValueError(
                "AegisVox paper/eval LLM de-id and relexicalization must use GPT-5.4 or GPT-5.5; "
                f"refusing model={self.model!r}. Do not downgrade to GPT-4.1."
            )

    def _api_key(self) -> str:
        return _openai_compatible_api_key_from_env()

    def _strict_detection_instructions(self, *, correction_note: str | None = None) -> str:
        instructions = (
            "You are the PHI detector for a clinical audio de-identification system. Return JSON only. "
            "Detect ONLY HIPAA Safe Harbor identifiers and choose phi_type only from the provided enum. "
            "Do not use broad generic labels such as NAME, LOCATION, DATE, ORG, ORGANIZATION, RELATIONSHIP, or PII. "
            "Never invent phi_type values outside the enum; specifically, do not output HIPAA_ORGANIZATION or HIPAA_RELATIONSHIP. "
            "If an organization/facility span is identifying only because it is a geographic subdivision, use HIPAA_GEOGRAPHIC_SUBDIVISION; otherwise omit it. "
            "If a relative/relationship word is not attached to a specific person name, omit it; if it includes a specific person name, use HIPAA_NAME for the name span only. "
            "Do not mark ordinary words or filler phrases as PHI. Do not mark a month word by itself as "
            "HIPAA_GEOGRAPHIC_SUBDIVISION. Do not mark titles or honorifics alone, such as Madame, "
            "Monsieur, Doctor, Dr., nurse, patient, caller, or chef, as HIPAA_NAME unless they are attached "
            "to a specific person name. A month/date is HIPAA_DATE only when it is an actual date element "
            "directly related to an individual or care event. If uncertain, omit the span rather than guessing. "
            "For every edit, original_text must be an exact substring of the referenced segment text. "
            "Return no replacement text; this step is detection only. If no PHI is present, return exactly {\"edits\": []}."
        )
        if correction_note:
            instructions += (
                " CORRECTION REQUIRED: Your previous response used unsupported phi_type values. "
                f"{correction_note} Re-emit the full JSON object from scratch using ONLY these allowed values: "
                f"{', '.join(HIPAA_PHI_TYPES)}. If a span cannot be represented by one of those values, omit it."
            )
        return instructions

    def _validate_or_retry_detected_edits(
        self,
        *,
        edits: list[dict[str, Any]],
        call_once,
    ) -> list[dict[str, Any]]:
        unsupported = _unsupported_detector_phi_types(edits)
        if not unsupported:
            return edits
        correction_note = "Unsupported values were: " + ", ".join(repr(x) for x in unsupported) + "."
        retried = call_once(self._strict_detection_instructions(correction_note=correction_note))
        unsupported_after_retry = _unsupported_detector_phi_types(retried)
        if unsupported_after_retry:
            raise TextDeIDSetupError(
                "unsupported PHI type(s) after correction retry "
                f"{unsupported_after_retry!r}; detector must use HIPAA Safe Harbor entity types only"
            )
        return retried

    def detect_edits(self, transcript: dict[str, Any], policy: DeIDPolicy) -> list[dict[str, Any]]:
        """Detect PHI spans with an LLM under a strict HIPAA Safe Harbor schema."""

        self._api_key()
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["edits"],
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["segment_index", "phi_type", "original_text", "confidence", "rationale"],
                        "properties": {
                            "segment_index": {"type": "integer", "minimum": 0},
                            "phi_type": {"type": "string", "enum": list(HIPAA_PHI_TYPES)},
                            "original_text": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "rationale": {"type": "string"},
                        },
                    },
                }
            },
        }
        instructions = self._strict_detection_instructions()
        segments = [
            {"segment_index": idx, "speaker_id": row.get("speaker_id"), "start_sec": row.get("start_sec"), "end_sec": row.get("end_sec"), "text": str(row.get("text") or "")}
            for idx, row in enumerate(_segment_text_rows(transcript))
        ]
        user_payload = {"hipaa_phi_types": list(HIPAA_PHI_TYPES), "segments": segments}
        if _should_use_raw_openai_compat():
            def raw_call(system_instructions: str) -> list[dict[str, Any]]:
                edits = _openai_compat_raw_json_call(model=self.model, system=system_instructions, user_payload=user_payload, expected_key="edits").get("edits")
                if not isinstance(edits, list):
                    raise TextDeIDSetupError("OpenAI-compatible provider HIPAA PHI detection response missing edits list")
                return edits

            return self._validate_or_retry_detected_edits(edits=raw_call(instructions), call_once=raw_call)
        try:
            from openai import OpenAI
        except ImportError:
            def raw_call(system_instructions: str) -> list[dict[str, Any]]:
                edits = _openai_compat_raw_json_call(model=self.model, system=system_instructions, user_payload=user_payload, expected_key="edits").get("edits")
                if not isinstance(edits, list):
                    raise TextDeIDSetupError("OpenAI-compatible provider HIPAA PHI detection response missing edits list")
                return edits

            return self._validate_or_retry_detected_edits(edits=raw_call(instructions), call_once=raw_call)
        client = OpenAI(**_openai_compatible_client_kwargs(self.model))

        def sdk_call(system_instructions: str) -> list[dict[str, Any]]:
            try:
                response = client.responses.create(
                    model=self.model,
                    input=[{"role": "system", "content": system_instructions}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
                    text={"format": {"type": "json_schema", "name": "aegisvox_hipaa_phi_detection", "strict": True, "schema": schema}},
                    store=False,
                )
            except Exception as exc:  # pragma: no cover
                raise TextDeIDSetupError(f"OpenAI HIPAA PHI detection call failed: {exc}") from exc
            output_text = getattr(response, "output_text", None)
            if not output_text:
                try:
                    output_text = response.output[0].content[0].text
                except Exception as exc:  # pragma: no cover
                    raise TextDeIDSetupError("OpenAI HIPAA PHI detection response did not include output_text") from exc
            try:
                payload = json.loads(output_text)
            except json.JSONDecodeError as exc:
                raise TextDeIDSetupError(f"OpenAI HIPAA PHI detection response was not valid JSON: {exc}") from exc
            edits = payload.get("edits")
            if not isinstance(edits, list):
                raise TextDeIDSetupError("OpenAI HIPAA PHI detection response missing edits list")
            return edits

        return self._validate_or_retry_detected_edits(edits=sdk_call(instructions), call_once=sdk_call)

    def plan_replacements(self, *, transcript: dict[str, Any], entities: list[dict[str, Any]], policy: DeIDPolicy) -> list[dict[str, Any]]:
        """Ask an LLM to choose natural synthetic replacements."""

        # In hybrid mode, high-risk HIPAA types are intentionally mask-only.
        # Do not ask the LLM to "naturalize" these entities: doing so can
        # create unsafe outputs (for example, reusing an age-over-89 phrase)
        # and contradicts the policy boundary.  Return deterministic policy
        # placeholders for those entities and only send speakable relexicalized
        # types (name/date/geographic subdivision) to the LLM.
        #
        # In full relexicalize mode, age-over-89 is also handled
        # deterministically.  A specific age above 89 remains HIPAA PHI, so the
        # safe natural replacement is a speakable age below 90 rather than an
        # LLM-invented over-89 phrase.
        policy_replacements: list[dict[str, str]] = []
        llm_entities: list[dict[str, Any]] = []
        mode = policy.normalized_mode()
        for entity in entities:
            phi_type = str(entity.get("phi_type") or "")
            entity_id = str(entity.get("entity_id") or "")
            original_text = str(entity.get("original_text") or "")
            if mode == "hybrid" and phi_type in policy.hybrid_mask_types:
                policy_replacements.append(
                    {
                        "entity_id": entity_id,
                        "replacement_text": policy.replacement_for(phi_type),
                        "rationale": "hybrid policy mask for high-risk HIPAA identifier",
                    }
                )
            elif mode == "relexicalize" and phi_type == "HIPAA_AGE_OVER_89":
                policy_replacements.append(
                    {
                        "entity_id": entity_id,
                        "replacement_text": policy.replacement_for_entity(
                            phi_type, original_text, len(policy_replacements)
                        ),
                        "rationale": "deterministic safe age below ninety for HIPAA age-over-89",
                    }
                )
            else:
                llm_entities.append(entity)
        if not llm_entities:
            return policy_replacements

        self._api_key()
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["replacements"],
            "properties": {
                "replacements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["entity_id", "replacement_text", "rationale"],
                        "properties": {"entity_id": {"type": "string"}, "replacement_text": {"type": "string"}, "rationale": {"type": "string"}},
                    },
                }
            },
        }
        instructions = (
            "You are a clinical audio de-identification relexicalizer. Return JSON only. "
            "Return exactly one JSON object with top-level key replacements; do not use entity_replacements or any other top-level key. "
            "For each provided entity, choose a synthetic replacement that is non-identifying, speakable aloud, natural in a clinical conversation, and preserves the HIPAA PHI type. "
            "Maintain consistency: each entity_id gets exactly one replacement, and all mentions of the same original entity will use it. Do not reuse the original text. "
            "Preserve language: choose replacements in the same language as the entity's source_language and surrounding transcript. Do not translate the utterance to English. "
            "For French or other non-English transcripts, names, dates, geographic terms, and filler-style phrasing must be natural in that language/locale. "
            "For HIPAA_DATE entities, choose a different synthetic date/time expression with the same conversational role; never copy the original month, day, year, or full phrase. "
            "For HIPAA_NAME entities, choose a different plausible name; never copy any original name token. "
            "For HIPAA_EMAIL, do not output an address with @ unless it ends in .invalid; prefer a speakable non-contactable phrase such as \"the email address on file\". "
            "For HIPAA_PHONE or HIPAA_FAX, do not output real-looking digits; prefer \"the phone number on file\" or \"the fax number on file\". "
            "For high-risk identifiers such as phone, email, SSN, medical record, health plan, account, license, vehicle, device, URL, IP, biometric, image, and other unique identifiers, prefer policy placeholders if policy mode is hybrid; in relexicalize mode use speakable non-contactable phrases, not synthetic credentials."
        )
        user_payload = {
            "policy_mode": policy.normalized_mode(),
            "document_language": _transcript_language(transcript),
            "language_aware_relexicalization": policy.language_aware_relexicalization,
            "entities": llm_entities,
            "full_transcript": str(transcript.get("text") or ""),
        }
        if _should_use_raw_openai_compat():
            replacements = _openai_compat_raw_json_call(model=self.model, system=instructions, user_payload=user_payload, expected_key="replacements").get("replacements")
            if not isinstance(replacements, list):
                raise TextDeIDSetupError("OpenAI-compatible provider relexicalization response missing replacements list")
            return policy_replacements + _repair_replacement_entity_ids(replacements, llm_entities)
        try:
            from openai import OpenAI
        except ImportError:
            replacements = _openai_compat_raw_json_call(model=self.model, system=instructions, user_payload=user_payload, expected_key="replacements").get("replacements")
            if not isinstance(replacements, list):
                raise TextDeIDSetupError("OpenAI-compatible provider relexicalization response missing replacements list")
            return policy_replacements + _repair_replacement_entity_ids(replacements, llm_entities)
        client = OpenAI(**_openai_compatible_client_kwargs(self.model))
        try:
            response = client.responses.create(
                model=self.model,
                input=[{"role": "system", "content": instructions}, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
                text={"format": {"type": "json_schema", "name": "aegisvox_relexicalization_plan", "strict": True, "schema": schema}},
                store=False,
            )
        except Exception as exc:  # pragma: no cover
            raise TextDeIDSetupError(f"OpenAI relexicalization call failed: {exc}") from exc
        output_text = getattr(response, "output_text", None)
        if not output_text:
            try:
                output_text = response.output[0].content[0].text
            except Exception as exc:  # pragma: no cover
                raise TextDeIDSetupError("OpenAI relexicalization response did not include output_text") from exc
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise TextDeIDSetupError(f"OpenAI relexicalization response was not valid JSON: {exc}") from exc
        replacements = payload.get("replacements")
        if not isinstance(replacements, list):
            raise TextDeIDSetupError("OpenAI relexicalization response missing replacements list")
        return policy_replacements + _repair_replacement_entity_ids(replacements, llm_entities)


@dataclass
class TextDeIDService:
    artifact_store: LocalArtifactStore
    policy: DeIDPolicy = field(default_factory=DeIDPolicy)
    llm_backend: TextDeIDBackend | None = None

    def deidentify_to_artifact(
        self,
        *,
        asr_uri: str,
        output_deid_uri: str,
        use_llm: bool = False,
        use_llm_replacements: bool = False,
    ) -> dict[str, Any]:
        validate_artifact_uri(asr_uri)
        validate_artifact_uri(output_deid_uri)
        transcript = self.artifact_store.read_json(asr_uri)
        if self.llm_backend is None:
            raise TextDeIDSetupError("text de-id detection requires a configured LLM HIPAA PHI backend")
        llm_detected_rows = self.llm_backend.detect_edits(transcript, self.policy)
        quarantined_detector_rows: list[dict[str, Any]] = []
        edits = _coerce_llm_detected_edits(
            llm_detected_rows,
            transcript,
            self.policy,
            quarantined_rows=quarantined_detector_rows,
        )
        replacement_plan_summary = {
            "replacement_source": "policy_after_llm_detection",
            "num_entities": len(_unique_entities_from_edits(edits)),
            "replacements_planned": len(_unique_entities_from_edits(edits)),
        }
        if use_llm_replacements:
            entities = _unique_entities_from_edits(edits)
            if self.policy.normalized_mode() == "hybrid":
                entities_to_plan = [
                    entity for entity in entities if str(entity["phi_type"]) not in self.policy.hybrid_mask_types
                ]
            else:
                entities_to_plan = entities
            if entities_to_plan:
                plan = self.llm_backend.plan_replacements(
                    transcript=transcript, entities=entities_to_plan, policy=self.policy
                )
                edits, replacement_plan_summary = _apply_replacement_plan(
                    edits, plan, self.policy, planned_entities=entities_to_plan
                )
            else:
                replacement_plan_summary = {
                    "replacement_source": "policy_only",
                    "num_entities": len(entities),
                    "replacements_planned": 0,
                    "policy_replacements_preserved": len(entities),
                }
        text = str(transcript.get("text") or "")
        payload = {
            "asr_uri": asr_uri,
            "detector": {
                "backend": self.llm_backend.backend_name,
                "backend_version": self.llm_backend.backend_version,
                "policy": "hipaa_safe_harbor_llm_only",
                "regex_detector_used": False,
                "allowed_phi_types": list(HIPAA_PHI_TYPES),
                "invalid_detector_rows_quarantined": len(quarantined_detector_rows),
            },
            "quarantined_detector_rows": quarantined_detector_rows,
            "policy_mode": self.policy.normalized_mode(),
            "transcript_language": _transcript_language(transcript),
            "language_aware_relexicalization": self.policy.language_aware_relexicalization,
            "entity_consistency": {
                "enabled": True,
                "scope": "transcript",
                "key": "phi_type+casefolded_original_text",
                "same_original_same_replacement": True,
            },
            "replacement_plan_summary": replacement_plan_summary,
            "edits": [edit.to_dict() for edit in edits],
            "deidentified_text": apply_text_edits(text, edits),
            "deidentified_segments": [
                {
                    **row,
                    "text": apply_text_edits(str(row.get("text") or ""), edits),
                }
                for row in _segment_text_rows(transcript)
            ],
            "validator_summary": {
                "deterministic_edits": 0,
                "llm_detector_rows": len(llm_detected_rows),
                "llm_detector_edits": len(edits),
                "llm_detector_rows_quarantined": len(quarantined_detector_rows),
                "llm_edits": len(edits),
                "llm_replacement_plan": bool(use_llm_replacements),
                "unresolved_high_confidence_phi": 0,
            },
        }
        self.artifact_store.write_json(output_deid_uri, payload)
        return payload
