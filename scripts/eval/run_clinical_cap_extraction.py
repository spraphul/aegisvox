#!/usr/bin/env python3
"""Extract evidence-linked clinical atomic propositions from source transcripts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gpt_concept_preservation import (  # noqa: E402
    call_openai_compat_bearer_chat,
    call_chat_completions,
    extract_json_text_chat,
    group_pairs,
    read_jsonl,
    safe_parse_model_json,
)


SYSTEM_PROMPT = """You extract clinical atomic propositions (CAPs) from a complete source clinical conversation.

Extract source facts only. Every CAP must be atomic, clinically material, and
supported by one exact contiguous source quote. Preserve negation, uncertainty,
temporality, verification state, speaker/role, severity/value, action type, and
the final state after corrections. Atomize conjunctions when parts could fail
independently. Names and direct identifiers are not clinical CAPs unless their
relationship or replacement changes clinical meaning. Do not infer diagnoses,
roles, or actions not stated in the transcript. Return strict JSON only.
Evidence must be literal and discriminative: never use an isolated generic
answer such as "yes", "no", "okay", or a repeated one-word fragment. For a
short answer, quote enough contiguous question-and-answer context to uniquely
locate the evidence in the complete transcript.
"""


USER_TEMPLATE = """audio_id: {audio_id}
dataset_id: {dataset_id}
language: {language}

COMPLETE SOURCE TRANSCRIPT:
{transcript}

Return:
{{
  "audio_id": "{audio_id}",
  "language": "en|fr|mixed|unknown",
  "caps": [
    {{
      "cap_id": "C1",
      "category": "symptom|diagnosis|medication|allergy|test|procedure|measurement|history|plan|other",
      "canonical_fact": "one atomic source fact",
      "source_quote": "exact contiguous transcript quote",
      "source_speaker": "speaker label or unknown",
      "source_role": "patient|clinician|dispatcher|caller|caregiver|third_party|unknown",
      "polarity": "affirmed|negated|uncertain|conditional",
      "temporality": "past|current|future|duration|trend|unknown",
      "verification": "reported|observed|assessed|planned|ruled_out",
      "severity_or_value": "verbatim value/modifier or empty",
      "action_type": "none|test|medication|procedure|followup|escalation",
      "clinical_importance": "critical|major|minor"
    }}
  ]
}}
"""


ENUMS = {
    "category": {"symptom", "diagnosis", "medication", "allergy", "test", "procedure", "measurement", "history", "plan", "other"},
    "source_role": {"patient", "clinician", "dispatcher", "caller", "caregiver", "third_party", "unknown"},
    "polarity": {"affirmed", "negated", "uncertain", "conditional"},
    "temporality": {"past", "current", "future", "duration", "trend", "unknown"},
    "verification": {"reported", "observed", "assessed", "planned", "ruled_out"},
    "action_type": {"none", "test", "medication", "procedure", "followup", "escalation"},
    "clinical_importance": {"critical", "major", "minor"},
}

ENUM_ALIASES = {
    "temporality": {
        "historical": "past", "recent": "past", "previous": "past",
        "ongoing": "current", "chronic": "current", "planned": "future",
    },
    "verification": {"suspected": "reported", "confirmed": "assessed"},
}


class ModelJSONDecodeError(ValueError):
    """Preserve a malformed/truncated model response for fail-closed recovery."""

    def __init__(self, message: str, *, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def load_env(path: Path | None) -> None:
    if not path or not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip("'").strip('"')


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def locate_quote(transcript: str, quote: str) -> tuple[int, int]:
    # First attempt exact offsets. Whitespace-normalized validation is accepted,
    # but normalized-only quotes retain -1 offsets and fail the paper gate until
    # repaired to literal evidence.
    start = transcript.find(quote)
    return (start, start + len(quote)) if start >= 0 else (-1, -1)


TOKEN_PATTERN = re.compile(r"\w+(?:['’\-]\w+)*", flags=re.UNICODE)


def literalize_quote(transcript: str, quote: str) -> tuple[str, str] | None:
    """Recover an exact source substring from a lexically identical model quote.

    Models sometimes copy the right words while normalizing case, punctuation,
    apostrophes, or whitespace.  This is evidence alignment rather than a
    clinical judgment: only a *unique, exact token-sequence* match is accepted.
    The returned quote is copied byte-for-character from the transcript.  Typos,
    paraphrases, omissions, and ambiguous matches still fail closed and go back
    through model repair.
    """
    quote_tokens = [match.group(0).casefold().replace("’", "'") for match in TOKEN_PATTERN.finditer(quote)]
    if len(quote_tokens) < 2:
        return None
    transcript_matches = list(TOKEN_PATTERN.finditer(transcript))
    transcript_tokens = [match.group(0).casefold().replace("’", "'") for match in transcript_matches]
    width = len(quote_tokens)
    matches = [
        index for index in range(len(transcript_tokens) - width + 1)
        if transcript_tokens[index:index + width] == quote_tokens
    ]
    if len(matches) != 1:
        return None
    index = matches[0]
    start = transcript_matches[index].start()
    end = transcript_matches[index + width - 1].end()
    return transcript[start:end], "unique_exact_token_sequence"


def expand_unique_literal_context(transcript: str, quote: str, *, radius: int = 120) -> str | None:
    """Expand a unique but underspecified literal quote to nearby source text."""
    if not quote or transcript.count(quote) != 1:
        return None
    start = transcript.find(quote)
    end = start + len(quote)
    left_floor = max(0, start - radius)
    right_ceiling = min(len(transcript), end + radius)
    left_candidates = [transcript.rfind(mark, left_floor, start) for mark in ("\n", ". ", "? ", "! ", ": ")]
    left = max(left_candidates)
    left = left + (1 if left >= 0 and transcript[left] == "\n" else 2) if left >= 0 else left_floor
    right_candidates = [transcript.find(mark, end, right_ceiling) for mark in ("\n", ". ", "? ", "! ")]
    right_candidates = [value for value in right_candidates if value >= 0]
    right = min(right_candidates) + 1 if right_candidates else right_ceiling
    expanded = transcript[left:right].strip()
    words = [token for token in TOKEN_PATTERN.findall(expanded) if any(ch.isalnum() for ch in token)]
    if len(words) < 2:
        token_matches = list(TOKEN_PATTERN.finditer(transcript))
        overlapping = [i for i, match in enumerate(token_matches) if match.start() <= start < match.end()]
        if overlapping:
            index = overlapping[0]
            token_left = token_matches[max(0, index - 3)].start()
            token_right = token_matches[min(len(token_matches) - 1, index + 3)].end()
            expanded = transcript[token_left:token_right].strip()
            words = [token for token in TOKEN_PATTERN.findall(expanded) if any(ch.isalnum() for ch in token)]
    if len(words) < 2 or transcript.count(expanded) != 1 or quote not in expanded:
        return None
    return expanded


def evidence_candidate_contexts(
    transcript: str,
    quote: str,
    *,
    canonical_fact: str = "",
    radius: int = 140,
) -> list[dict[str, Any]]:
    """Return exact source candidates for model-selected evidence repair.

    Candidate retrieval is not the clinical decision.  It only presents exact
    source substrings; the same LLM must select a candidate that supports its
    CAP, or return null.  This avoids asking a model to reproduce punctuation
    and whitespace perfectly over a long conversation.
    """
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(text: str, start: int, end: int, kind: str, score: float | None = None) -> None:
        literal = text.strip()
        if not literal or literal in seen or transcript.count(literal) != 1:
            return
        if len(TOKEN_PATTERN.findall(literal)) < 2:
            return
        seen.add(literal)
        contexts.append({
            "candidate_id": f"E{len(contexts) + 1}",
            "start": transcript.find(literal),
            "end": transcript.find(literal) + len(literal),
            "candidate_text": literal,
            "retrieval_kind": kind,
            **({"retrieval_score": round(score, 6)} if score is not None else {}),
        })

    start = 0
    exact_occurrences = []
    while quote and len(exact_occurrences) < 12:
        found = transcript.find(quote, start)
        if found < 0:
            break
        end = found + len(quote)
        exact_occurrences.append((found, end))
        start = end
    for found, end in exact_occurrences:
        # Prefer a compact question-plus-answer candidate when the repeated
        # phrase is a short answer. This keeps nearby repeated answers (e.g.,
        # smoking versus recreational-drug questions) clinically separable.
        question_mark = transcript.rfind("?", max(0, found - radius), found)
        if question_mark >= 0:
            prior_boundaries = [
                transcript.rfind(mark, max(0, question_mark - radius), question_mark)
                for mark in (". ", "? ", "! ", "\n")
            ]
            boundary = max(prior_boundaries)
            question_start = boundary + (1 if boundary >= 0 and transcript[boundary] == "\n" else 2) if boundary >= 0 else max(0, question_mark - radius)
            add(transcript[question_start:end], question_start, end, "question_answer_occurrence")
        expanded = transcript[max(0, found - radius):min(len(transcript), end + radius)].strip()
        add(expanded, found, end, "exact_occurrence_context")

    query = " ".join((quote + " " + canonical_fact).casefold().split())
    sentence_matches = list(re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", transcript))
    scored = []
    query_tokens = set(TOKEN_PATTERN.findall(query))
    for match in sentence_matches:
        text = match.group(0).strip()
        if len(TOKEN_PATTERN.findall(text)) < 2 or transcript.count(text) != 1:
            continue
        normalized = " ".join(text.casefold().split())
        sequence = difflib.SequenceMatcher(None, query, normalized, autojunk=False).ratio() if query else 0.0
        text_tokens = set(TOKEN_PATTERN.findall(normalized))
        overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens | text_tokens))
        score = max(sequence, overlap)
        if score > 0:
            scored.append((score, match.start(), match.end(), text))
    for score, found, end, text in sorted(scored, reverse=True)[:12]:
        if len(contexts) >= 12:
            break
        add(text, found, end, "lexical_candidate", score)
    return contexts


def normalize_declared_aliases(result: dict[str, Any]) -> None:
    ledger = result.setdefault("schema_normalizations", [])
    if not isinstance(ledger, list):
        ledger = []
        result["schema_normalizations"] = ledger
    for cap in result.get("caps") or []:
        if not isinstance(cap, dict):
            continue
        for field, aliases in ENUM_ALIASES.items():
            raw = str(cap.get(field) or "").strip().casefold().replace(" ", "_").replace("-", "_")
            mapped = aliases.get(raw)
            if mapped:
                cap[field] = mapped
                entry = {"cap_id": cap.get("cap_id"), "field": field, "from": raw, "to": mapped}
                if entry not in ledger:
                    ledger.append(entry)


def validate_and_enrich(result: dict[str, Any], *, audio_id: str, transcript: str) -> None:
    normalize_declared_aliases(result)
    if result.get("audio_id") != audio_id:
        raise ValueError("audio_id mismatch")
    caps = result.get("caps")
    if not isinstance(caps, list) or not caps:
        raise ValueError("caps must be a non-empty list")
    ids: set[str] = set()
    for index, cap in enumerate(caps):
        if not isinstance(cap, dict):
            raise ValueError(f"caps[{index}] must be an object")
        cap_id = str(cap.get("cap_id") or "").strip()
        if not cap_id or cap_id in ids:
            raise ValueError(f"invalid/duplicate cap_id: {cap_id!r}")
        ids.add(cap_id)
        if not str(cap.get("canonical_fact") or "").strip():
            raise ValueError(f"{cap_id}: empty canonical_fact")
        quote = str(cap.get("source_quote") or "").strip()
        start, end = locate_quote(transcript, quote)
        if start < 0:
            aligned = literalize_quote(transcript, quote)
            if aligned is None:
                raise ValueError(f"{cap_id}: source_quote must be a literal contiguous transcript substring")
            literal_quote, alignment_method = aligned
            cap["source_quote_model_copy"] = quote
            cap["source_quote"] = literal_quote
            cap["evidence_alignment_method"] = alignment_method
            quote = literal_quote
            start, end = locate_quote(transcript, quote)
            if start < 0:  # Defensive: literalize_quote must return source text.
                raise ValueError(f"{cap_id}: recovered source_quote is not literal")
        words = [token for token in quote.replace("'", " ").split() if any(ch.isalnum() for ch in token)]
        if len(words) < 2 or len(quote) < 6:
            expanded = expand_unique_literal_context(transcript, quote)
            if expanded is None:
                raise ValueError(f"{cap_id}: source_quote is too short/ambiguous; include contiguous context")
            cap["source_quote_model_copy"] = quote
            cap["source_quote"] = expanded
            cap["evidence_alignment_method"] = "unique_literal_context_expansion"
            quote = expanded
            start, end = locate_quote(transcript, quote)
        if transcript.count(quote) != 1:
            raise ValueError(
                f"{cap_id}: source_quote is repeated; include more contiguous context; "
                f"quote={quote[:240]!r}; canonical_fact={str(cap.get('canonical_fact') or '')[:240]!r}; "
                f"source_speaker={str(cap.get('source_speaker') or '')[:80]!r}"
            )
        for field, allowed in ENUMS.items():
            if cap.get(field) not in allowed:
                raise ValueError(f"{cap_id}: invalid {field}={cap.get(field)!r}")
        if not str(cap.get("source_speaker") or "").strip():
            raise ValueError(f"{cap_id}: source_speaker must be explicit or unknown")
        cap["evidence_start"] = start
        cap["evidence_end"] = end


def build_prompt(pair: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            audio_id=pair["audio_id"], dataset_id=pair.get("source_dataset_id") or "unknown",
            language=pair.get("source_language") or "unknown", transcript=pair["reference"],
        )},
    ]


def call_model(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict[str, Any]:
    if args.backend == "openai_compat-bearer":
        token = os.environ.get("OPENAI_API_KEY", "")
        if not token:
            raise RuntimeError("missing OPENAI_API_KEY")
        response = call_openai_compat_bearer_chat(
            base_url=args.base_url, bearer_token=token, model=args.model, messages=messages,
            temperature=0.0, max_tokens=args.max_tokens, timeout=args.timeout,
            verify_ssl=not args.no_verify_ssl, api_version=args.api_version,
            omit_api_version=args.omit_api_version, service_name=None,
            completion_token_param="max_completion_tokens",
        )
    else:
        response = call_chat_completions(
            base_url=args.base_url, api_key=args.api_key or "not-required", model=args.model,
            messages=messages, temperature=0.0, timeout=args.timeout, max_retries=2,
            verify_ssl=not args.no_verify_ssl, max_tokens=args.max_tokens,
        )
    raw_text = extract_json_text_chat(response)
    try:
        return safe_parse_model_json(raw_text)
    except json.JSONDecodeError as exc:
        raise ModelJSONDecodeError(str(exc), raw_text=raw_text) from exc


def salvage_complete_caps(raw_text: str) -> list[dict[str, Any]]:
    """Read only fully formed CAP objects from a truncated JSON response.

    This never repairs or guesses a partial object.  Parsing stops at the first
    incomplete/non-object item so only syntactically complete model output can
    enter the continuation inventory.
    """
    match = re.search(r'"caps"\s*:\s*\[', raw_text)
    if not match:
        return []
    decoder = json.JSONDecoder()
    position = match.end()
    caps: list[dict[str, Any]] = []
    while position < len(raw_text):
        while position < len(raw_text) and (raw_text[position].isspace() or raw_text[position] == ","):
            position += 1
        if position >= len(raw_text) or raw_text[position] == "]":
            break
        if raw_text[position] != "{":
            break
        try:
            item, end = decoder.raw_decode(raw_text, position)
        except json.JSONDecodeError:
            break
        if not isinstance(item, dict):
            break
        caps.append(item)
        position = end
    return caps


def merge_cap_batches(batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge continuation batches, dropping only byte-semantic duplicates."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for batch in batches:
        for cap in batch:
            canonical = " ".join(str(cap.get("canonical_fact") or "").casefold().split())
            quote = str(cap.get("source_quote") or "")
            key = (canonical, quote)
            if not canonical or key in seen:
                continue
            seen.add(key)
            copied = dict(cap)
            copied["cap_id"] = f"C{len(merged) + 1}"
            merged.append(copied)
    return merged


def continue_truncated_extraction(
    args: argparse.Namespace,
    *,
    pair: dict[str, Any],
    initial_error: ModelJSONDecodeError,
    max_rounds: int = 8,
) -> dict[str, Any]:
    """Continue a long full-context extraction rather than dropping the row.

    Each continuation sees the complete source transcript and an inventory of
    already completed CAPs.  The transcript is never chunked, and partial JSON
    objects are never accepted.  A valid response must explicitly mark the
    extraction complete; otherwise another bounded continuation is required.
    """
    batches = [salvage_complete_caps(initial_error.raw_text)]
    if not batches[0]:
        raise initial_error
    for _ in range(max_rounds):
        existing = merge_cap_batches(batches)
        inventory = [
            {
                "cap_id": cap["cap_id"],
                "canonical_fact": cap.get("canonical_fact"),
                "source_quote": cap.get("source_quote"),
            }
            for cap in existing
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\nThe previous response reached the output limit. Continue the same extraction. "
                    "Do not repeat any fact in the completed inventory. Return only additional CAPs and set "
                    "complete=true only after every remaining clinically material source fact has been extracted."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"audio_id: {pair['audio_id']}\n"
                    f"dataset_id: {pair.get('source_dataset_id') or 'unknown'}\n"
                    f"language: {pair.get('source_language') or 'unknown'}\n\n"
                    f"COMPLETED CAP INVENTORY (do not repeat):\n{json.dumps(inventory, ensure_ascii=False)}\n\n"
                    f"COMPLETE SOURCE TRANSCRIPT:\n{pair['reference']}\n\n"
                    "Return strict JSON only as {\"audio_id\": \"...\", \"language\": \"en|fr|mixed|unknown\", "
                    "\"caps\": [additional CAP objects using the original closed schema], \"complete\": true}."
                ),
            },
        ]
        try:
            continuation = call_model(args, messages)
            additional = continuation.get("caps")
            if not isinstance(additional, list):
                raise ValueError("continuation caps must be a list")
            batches.append([dict(cap) for cap in additional if isinstance(cap, dict)])
            if continuation.get("complete") is True:
                return {
                    "audio_id": pair["audio_id"],
                    "language": continuation.get("language") or pair.get("source_language") or "unknown",
                    "caps": merge_cap_batches(batches),
                    "extraction_continuations": len(batches) - 1,
                    "truncated_response_recovery": True,
                }
        except ModelJSONDecodeError as exc:
            salvaged = salvage_complete_caps(exc.raw_text)
            if not salvaged:
                raise
            batches.append(salvaged)
    raise ValueError(f"long extraction did not complete after {max_rounds} continuation rounds")


def repair_one_cap(
    args: argparse.Namespace,
    *,
    pair: dict[str, Any],
    result: dict[str, Any],
    cap_id: str,
    error: Exception,
) -> dict[str, Any]:
    """Repair one invalid CAP without asking the model to regenerate 50+ valid CAPs."""
    caps = result.get("caps") or []
    index = next((i for i, cap in enumerate(caps) if str(cap.get("cap_id")) == cap_id), None)
    if index is None:
        raise ValueError(f"cannot locate invalid CAP {cap_id}")
    failed_cap = caps[index]
    candidate_contexts = evidence_candidate_contexts(
        pair["reference"],
        str(failed_cap.get("source_quote") or ""),
        canonical_fact=str(failed_cap.get("canonical_fact") or ""),
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You repair one clinical atomic proposition. Preserve its clinical meaning and every valid field. "
                "Return strict JSON only as {\"cap\": {...}, \"selected_candidate_id\": \"E1|...|null\"}. "
                "When an exact evidence candidate supports the CAP, select its candidate_id; the program will copy "
                "candidate_text exactly. Otherwise use null and copy source_quote verbatim as one "
                "unique contiguous substring from the complete transcript; do not normalize punctuation, spelling, "
                "speaker labels, or whitespace. Keep every field inside these closed enums: category=symptom|diagnosis|"
                "medication|allergy|test|procedure|measurement|history|plan|other; source_role=patient|clinician|"
                "dispatcher|caller|caregiver|third_party|unknown; polarity=affirmed|negated|uncertain|conditional; "
                "temporality=past|current|future|duration|trend|unknown; verification=reported|observed|assessed|planned|"
                "ruled_out; action_type=none|test|medication|procedure|followup|escalation; "
                "clinical_importance=critical|major|minor."
            ),
        },
        {
            "role": "user",
            "content": (
                f"audio_id: {pair['audio_id']}\nvalidation_error: {type(error).__name__}: {error}\n\n"
                f"INVALID CAP:\n{json.dumps(failed_cap, ensure_ascii=False)}\n\n"
                f"EXACT SOURCE EVIDENCE CANDIDATES (may be empty; select only a clinically supporting candidate):\n"
                f"{json.dumps(candidate_contexts, ensure_ascii=False)}\n\n"
                f"COMPLETE SOURCE TRANSCRIPT:\n{pair['reference']}\n\n"
                "Return only the repaired cap wrapper."
            ),
        },
    ]
    repaired_payload = call_model(args, messages)
    repaired = repaired_payload.get("cap", repaired_payload)
    if not isinstance(repaired, dict):
        raise ValueError("targeted CAP repair did not return an object")
    repaired["cap_id"] = cap_id
    selected_candidate_id = repaired_payload.get("selected_candidate_id")
    if selected_candidate_id in (None, "null", ""):
        # Schema-only normalization for models that place the declared wrapper
        # field inside ``cap``.  The candidate is still checked against the
        # exact precomputed candidate list before any source text is copied.
        selected_candidate_id = repaired.pop("selected_candidate_id", None)
    repaired_quote = str(repaired.get("source_quote") or "")
    repaired_quote_is_discriminative = (
        repaired_quote in pair["reference"]
        and pair["reference"].count(repaired_quote) == 1
        and len(TOKEN_PATTERN.findall(repaired_quote)) >= 2
    )
    if selected_candidate_id in (None, "null", "") and candidate_contexts and not repaired_quote_is_discriminative:
        selection = call_model(
            args,
            [
                {
                    "role": "system",
                    "content": (
                        "Select exact source evidence for one clinical atomic proposition. "
                        "Return strict JSON only. Choose one candidate_id only when its candidate_text directly "
                        "supports the proposition with the declared polarity, temporality, speaker, and role; "
                        "otherwise return null. Do not rewrite candidate text."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "canonical_fact": repaired.get("canonical_fact"),
                            "source_speaker": repaired.get("source_speaker"),
                            "source_role": repaired.get("source_role"),
                            "polarity": repaired.get("polarity"),
                            "temporality": repaired.get("temporality"),
                            "candidates": candidate_contexts,
                            "return": {"selected_candidate_id": "candidate ID or null", "reason": "brief rationale"},
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        selected_candidate_id = selection.get("selected_candidate_id")
        repaired["evidence_candidate_selection_reason"] = selection.get("reason")
    if selected_candidate_id not in (None, "null", ""):
        selected = next(
            (candidate for candidate in candidate_contexts if candidate["candidate_id"] == selected_candidate_id),
            None,
        )
        if selected is None:
            raise ValueError(f"unknown selected_candidate_id={selected_candidate_id!r}")
        repaired["source_quote_model_copy"] = repaired.get("source_quote")
        repaired["source_quote"] = selected["candidate_text"]
        repaired["evidence_alignment_method"] = "model_selected_exact_source_candidate"
        repaired["evidence_candidate_selection"] = {
            "candidate_id": selected["candidate_id"],
            "retrieval_kind": selected["retrieval_kind"],
            "retrieval_score": selected.get("retrieval_score"),
        }
    updated = dict(result)
    updated["caps"] = [dict(cap) for cap in caps]
    updated["caps"][index] = repaired
    return updated


def evaluate(pair: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages = build_prompt(pair)
    try:
        result = call_model(args, messages)
    except ModelJSONDecodeError as exc:
        result = continue_truncated_extraction(args, pair=pair, initial_error=exc)
    for attempt in range(args.schema_retries + 1):
        try:
            validate_and_enrich(result, audio_id=pair["audio_id"], transcript=pair["reference"])
            return result
        except Exception as exc:  # noqa: BLE001
            if attempt >= args.schema_retries:
                raise
            match = re.match(r"^(C[^: ]+):", str(exc))
            if match:
                # Long conversations routinely contain dozens of otherwise
                # valid CAPs.  Repair the one invalid evidence field against
                # the same complete transcript instead of regenerating the
                # full list and moving the error elsewhere.
                result = repair_one_cap(
                    args, pair=pair, result=result, cap_id=match.group(1), error=exc,
                )
                continue
            # Show the model the exact object that failed.  Without this
            # assistant turn, a "correct the preceding response" retry merely
            # regenerated an entire long CAP list from scratch and could move
            # the schema/evidence error to another CAP.
            messages = [*messages, {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)}]
            messages = [*messages, {"role": "user", "content": (
                f"The preceding response failed validation: {type(exc).__name__}: {exc}. "
                "Repair only the invalid field(s), preserving all other valid CAPs, and return the corrected complete JSON object only. "
                "Use the closed enums. Every source_quote must be a unique, "
                "literal contiguous substring with at least two words; expand short yes/no evidence to include enough "
                "contiguous question-and-answer context."
            )}]
            result = call_model(args, messages)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-transcripts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["openai_compat-bearer", "openai-compatible"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--omit-api-version", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--no-verify-ssl", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_env(args.env_file)
    pairs = group_pairs(read_jsonl(args.paired_transcripts))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    pair_by_id = {pair["audio_id"]: pair for pair in pairs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A resume-safe append log is not safe when two processes target the same
    # directory. Fail loudly instead of issuing duplicate model calls or
    # interleaving JSONL writes.
    lock_handle = (args.output_dir / ".source_caps.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            f"another CAP extraction process already owns {args.output_dir}"
        ) from exc
    output_path = args.output_dir / "source_caps.jsonl"
    error_path = args.output_dir / "errors.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        for line in output_path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            pair = pair_by_id.get(item.get("audio_id"))
            if pair is None:
                continue
            try:
                validate_and_enrich(item, audio_id=pair["audio_id"], transcript=pair["reference"])
            except ValueError:
                continue
            completed[pair["audio_id"]] = item
    pending = [pair for pair in pairs if pair["audio_id"] not in completed]
    errors: list[dict[str, Any]] = []
    with output_path.open("a", encoding="utf-8") as output_log, error_path.open("a", encoding="utf-8") as error_log:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            future_map = {pool.submit(evaluate, pair, args): pair for pair in pending}
            for future in as_completed(future_map):
                pair = future_map[future]
                try:
                    item = future.result()
                    completed[pair["audio_id"]] = item
                    output_log.write(json.dumps(item, ensure_ascii=False) + "\n")
                    output_log.flush()
                except Exception as exc:  # noqa: BLE001
                    error = {"audio_id": pair["audio_id"], "error_type": type(exc).__name__, "error": str(exc)[:3000]}
                    errors.append(error)
                    error_log.write(json.dumps(error, ensure_ascii=False) + "\n")
                    error_log.flush()
                print(json.dumps({"completed": len(completed), "errors": len(errors), "total": len(pairs)}), flush=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            if pair["audio_id"] in completed:
                handle.write(json.dumps(completed[pair["audio_id"]], ensure_ascii=False) + "\n")
    total_caps = sum(len(item["caps"]) for item in completed.values())
    literal_offsets = sum(cap["evidence_start"] >= 0 for item in completed.values() for cap in item["caps"])
    summary = {
        "schema_version": "aegisvox_clinical_caps.v1",
        "model": args.model,
        "backend": args.backend,
        "num_source_conversations": len(pairs),
        "num_completed": len(completed),
        "num_errors": len(errors),
        "num_caps": total_caps,
        "literal_evidence_offset_rate": literal_offsets / total_caps if total_caps else None,
        "input_sha256": hashlib.sha256(args.paired_transcripts.read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not errors and len(completed) == len(pairs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
