#!/usr/bin/env python3
"""Generate structured clinical notes from complete source or target transcripts."""
from __future__ import annotations
import argparse, copy, fcntl, hashlib, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from run_gpt_concept_preservation import call_openai_compat_bearer_chat,call_chat_completions,extract_json_text_chat,group_pairs,read_jsonl as read_paired_jsonl,safe_parse_model_json  # noqa:E402

SYSTEM_PROMPT="""You are a conservative clinical documentation model.

Convert the complete supplied clinical conversation into a concise structured
note. The source-derived clinical atomic propositions (CAPs) were frozen before
this note was generated and are the only allowed clinical content inventory.
Include a CAP only when the supplied transcript itself supports it. Preserve
negation, uncertainty, temporality, speaker roles, medication/allergy details,
and the final state after corrections. Never infer missing diagnoses, exams,
tests, treatments, or recommendations from world knowledge. Every note item
must cite one or more frozen CAP IDs. For transcript_quote, copy the complete
source_quote field of the first supporting CAP byte-for-byte; never paraphrase,
normalize punctuation, or construct a new quote. Return strict JSON only in
the transcript's language."""
TARGET_SYSTEM_PROMPT="""You are a conservative clinical documentation model.

Convert the complete transformed clinical conversation into a concise
structured note. The source-derived clinical atomic propositions (CAPs) are an
allow-list of clinical content, not evidence that every proposition survived
the transformation. Include a CAP only when the supplied transformed
transcript itself supports it. Preserve negation, uncertainty, temporality,
speaker roles, medication/allergy details, and the final state after
corrections. Never infer missing content from the source CAP inventory or world
knowledge. Every note item must cite one or more frozen CAP IDs and one
target_evidence_id from the supplied deterministic evidence-unit registry. The
selected evidence unit must directly support the item. Never invent, alter, or
combine evidence IDs. Return strict JSON only in the transcript's language."""
USER_TEMPLATE="""audio_id: {audio_id}
dataset_id: {dataset_id}
language: {language}

FROZEN SOURCE-DERIVED CAPS:
{caps}

COMPLETE TRANSCRIPT:
{transcript}

Return:
{{
 "audio_id":"{audio_id}",
 "note":{{
   "chief_concern":[{{"text":"concise item","supporting_cap_ids":["C1"],"transcript_quote":"exact quote"}}],
   "history_of_present_illness":[],
   "medications":[],
   "allergies":[],
   "pertinent_history_and_review_of_systems":[],
   "assessment_and_problem_list":[],
   "plan_and_follow_up":[]
 }}
}}
Every section is a list of zero or more objects with exactly text,
supporting_cap_ids, and transcript_quote. Empty sections are allowed.
The transcript_quote must be the byte-for-byte source_quote of the first listed
supporting CAP. The runner canonically grounds this evidence pointer back to
that frozen source_quote; this does not alter the generated clinical text.
"""
TARGET_USER_TEMPLATE="""audio_id: {audio_id}
dataset_id: {dataset_id}
language: {language}

FROZEN SOURCE-DERIVED CAP ALLOW-LIST:
{caps}

COMPLETE TRANSFORMED TRANSCRIPT:
{transcript}

DETERMINISTIC TARGET EVIDENCE UNITS:
{target_evidence_units}

Return:
{{
 "audio_id":"{audio_id}",
 "note":{{
   "chief_concern":[{{"text":"concise item","supporting_cap_ids":["C1"],"target_evidence_id":"E0001"}}],
   "history_of_present_illness":[],
   "medications":[],
   "allergies":[],
   "pertinent_history_and_review_of_systems":[],
   "assessment_and_problem_list":[],
   "plan_and_follow_up":[]
 }}
}}
Every section is a list of zero or more objects with exactly text,
supporting_cap_ids, and target_evidence_id. Empty sections are allowed. A CAP
may be used only when the transformed transcript supports it. Select exactly
one listed target_evidence_id whose literal text directly supports the item.
The runner resolves that ID to its immutable exact transcript substring; it
never substitutes source evidence or performs fuzzy/semantic quote repair.
"""
SECTIONS=("chief_concern","history_of_present_illness","medications","allergies","pertinent_history_and_review_of_systems","assessment_and_problem_list","plan_and_follow_up")
TARGET_EVIDENCE_POLICY = "deterministic_transformed_transcript_evidence_ids_v1"
TARGET_EVIDENCE_MAX_CHARS = 320


def note_response_format(
    audio_id: str, *, max_items_per_section: int,
    transcript_source: str = "reference",
) -> dict[str, Any]:
    """Return a bounded JSON schema for vLLM structured decoding.

    This is a decoding constraint, not a clinical-content repair.  It prevents
    pathological Gemma repetitions from consuming the entire generation budget
    while retaining every required note section and the same CAP-grounded item
    contract used by the validator.
    """
    evidence_field = (
        "transcript_quote" if transcript_source == "reference"
        else "target_evidence_id"
    )
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["text", "supporting_cap_ids", evidence_field],
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 1000},
            "supporting_cap_ids": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "items": {"type": "string", "minLength": 1, "maxLength": 64},
            },
            evidence_field: {"type": "string", "minLength": 1, "maxLength": 2000},
        },
    }
    note = {
        "type": "object",
        "additionalProperties": False,
        "required": list(SECTIONS),
        "properties": {
            section: {
                "type": "array", "maxItems": max_items_per_section,
                "items": item,
            }
            for section in SECTIONS
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["audio_id", "note"],
        "properties": {
            "audio_id": {"type": "string", "const": audio_id},
            "note": note,
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "clinical_note", "strict": True, "schema": schema},
    }

def read_jsonl(path:Path)->list[dict[str,Any]]:
    """Read generic JSONL without applying the paired-transcript schema."""
    rows=[]
    with path.open('r',encoding='utf-8') as handle:
        for line_no,line in enumerate(handle,1):
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row,dict): raise ValueError(f'{path}:{line_no}: expected JSON object')
            rows.append(row)
    return rows


def build_target_evidence_units(
    transcript: str, *, max_chars: int = TARGET_EVIDENCE_MAX_CHARS,
) -> list[dict[str, Any]]:
    """Partition the full transformed transcript into immutable literal units.

    This is an evidence-addressing layer, not transcript truncation or clinical
    inference. The complete transcript is still supplied. Sentence punctuation
    is preferred; unusually long ASR sentences are split deterministically at
    a whitespace boundary.
    """
    if max_chars < 80:
        raise ValueError("target evidence max_chars must be at least 80")
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?=\s|$)", transcript, flags=re.UNICODE):
        end = match.end()
        if transcript[start:end].strip():
            left = start
            while left < end and transcript[left].isspace():
                left += 1
            spans.append((left, end))
        start = end
    if transcript[start:].strip():
        left = start
        while left < len(transcript) and transcript[left].isspace():
            left += 1
        spans.append((left, len(transcript)))

    bounded: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        cursor = span_start
        while span_end - cursor > max_chars:
            limit = cursor + max_chars
            split = transcript.rfind(" ", cursor + 1, limit + 1)
            if split <= cursor:
                split = limit
            bounded.append((cursor, split))
            cursor = split
            while cursor < span_end and transcript[cursor].isspace():
                cursor += 1
        if cursor < span_end:
            bounded.append((cursor, span_end))

    units: list[dict[str, Any]] = []
    for index, (unit_start, unit_end) in enumerate(bounded, 1):
        text = transcript[unit_start:unit_end]
        if not text or transcript[unit_start:unit_end] != text:
            raise AssertionError("invalid deterministic target evidence unit")
        units.append({
            "evidence_id": f"E{index:04d}",
            "start_char": unit_start,
            "end_char": unit_end,
            "text": text,
        })
    if transcript.strip() and not units:
        raise ValueError("nonempty transcript produced no target evidence units")
    return units

def load_env(path):
    if path and path.is_file():
        for raw in path.read_text().splitlines():
            line=raw.strip()
            if line and not line.startswith('#') and '=' in line:
                k,v=line.split('=',1); os.environ.setdefault(k.strip(),v.strip().strip("'").strip('"'))
def validate(result:dict[str,Any],audio_id:str,*,transcript:str,cap_ids:set[str],min_quote_chars:int=6)->None:
    if result.get('audio_id')!=audio_id: raise ValueError('audio_id mismatch')
    note=result.get('note')
    if not isinstance(note,dict) or set(note)!=set(SECTIONS): raise ValueError('note section contract mismatch')
    for field in SECTIONS:
        if not isinstance(note[field],list): raise ValueError(f'{field} must be a list')
        for index,item in enumerate(note[field]):
            if not isinstance(item,dict) or set(item)!={'text','supporting_cap_ids','transcript_quote'}: raise ValueError(f'{field}[{index}] contract mismatch')
            if not str(item['text']).strip(): raise ValueError(f'{field}[{index}] empty text')
            ids=item['supporting_cap_ids']
            if not isinstance(ids,list) or not ids:
                raise ValueError(f'{field}[{index}] invalid CAP support: expected nonempty list')
            invalid_ids=sorted(set(map(str,ids))-cap_ids)
            if invalid_ids:
                raise ValueError(
                    f'{field}[{index}] invalid CAP support: unlisted IDs {invalid_ids}'
                )
            quote=str(item['transcript_quote'])
            if len(quote)<min_quote_chars or quote not in transcript: raise ValueError(f'{field}[{index}] transcript quote is not literal evidence')


def canonicalize_evidence_quotes(
    result: dict[str, Any], *, caps: list[dict[str, Any]], transcript: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ground evidence pointers to immutable, source-audited CAP quotations.

    The model still generates every clinical note item and chooses its CAP IDs.
    Only ``transcript_quote`` is normalized: it is copied from the frozen
    ``source_quote`` of the first supporting CAP. Original and final hashes are
    retained so this mechanical evidence normalization is never silent.
    """
    grounded = copy.deepcopy(result)
    note = grounded.get('note')
    if not isinstance(note, dict):
        return grounded, []
    cap_by_id = {str(cap.get('cap_id')): cap for cap in caps}
    repairs: list[dict[str, Any]] = []
    for section, items in note.items():
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            supporting = item.get('supporting_cap_ids')
            if not isinstance(supporting, list) or not supporting:
                continue
            selected_cap_id = str(supporting[0])
            cap = cap_by_id.get(selected_cap_id)
            if cap is None:
                continue
            source_quote = str(cap.get('source_quote') or '')
            if len(source_quote) < 6 or source_quote not in transcript:
                raise ValueError(
                    f'{section}[{index}] frozen CAP {selected_cap_id} lacks literal source_quote'
                )
            original_quote = str(item.get('transcript_quote') or '')
            item['transcript_quote'] = source_quote
            if original_quote != source_quote:
                repairs.append({
                    'section': section,
                    'item_index': index,
                    'selected_cap_id': selected_cap_id,
                    'original_quote_sha256': hashlib.sha256(original_quote.encode()).hexdigest(),
                    'grounded_quote_sha256': hashlib.sha256(source_quote.encode()).hexdigest(),
                })
    return grounded, repairs


def resolve_target_evidence_ids(
    result: dict[str, Any], *, transcript: str,
    evidence_units: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve model-selected evidence IDs to exact transformed substrings."""
    grounded = copy.deepcopy(result)
    unit_by_id = {str(unit["evidence_id"]): unit for unit in evidence_units}
    if len(unit_by_id) != len(evidence_units):
        raise ValueError("duplicate target evidence IDs")
    repairs: list[dict[str, Any]] = []
    note = grounded.get('note')
    if not isinstance(note, dict):
        return grounded, repairs
    for section, items in note.items():
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            if set(item) != {'text', 'supporting_cap_ids', 'target_evidence_id'}:
                raise ValueError(f'{section}[{index}] target evidence-ID contract mismatch')
            evidence_id = str(item.pop('target_evidence_id') or '')
            unit = unit_by_id.get(evidence_id)
            if unit is None:
                raise ValueError(f'{section}[{index}] unknown target evidence ID {evidence_id!r}')
            start, end = int(unit['start_char']), int(unit['end_char'])
            literal = transcript[start:end]
            if literal != str(unit['text']) or not literal:
                raise ValueError(f'{section}[{index}] target evidence registry mismatch')
            item['transcript_quote'] = literal
            repairs.append({
                'section': section,
                'item_index': index,
                'normalization': TARGET_EVIDENCE_POLICY,
                'target_evidence_id': evidence_id,
                'grounded_quote_sha256': hashlib.sha256(literal.encode()).hexdigest(),
                'target_start_char': start,
                'target_end_char': end,
            })
    return grounded, repairs


def caps_for_generation(
    caps: list[dict[str, Any]], *, transcript_source: str,
) -> list[dict[str, Any]]:
    """Keep CAP semantics but hide immutable source evidence from target notes."""
    if transcript_source == 'reference':
        return caps
    hidden = {'source_quote', 'evidence_start', 'evidence_end'}
    return [
        {key: value for key, value in cap.items() if key not in hidden}
        for cap in caps
    ]


def validate_persisted_row(
    row: dict[str, Any], *, pair: dict[str, Any], caps: list[dict[str, Any]],
    transcript_source: str,
) -> None:
    """Fail closed on stale/mixed rows before resume counts them complete."""
    audio_id = str(pair["audio_id"])
    if str(row.get("audio_id")) != audio_id or row.get("transcript_source") != transcript_source:
        raise ValueError("persisted row identity/contract mismatch")
    transcript = str(pair[transcript_source])
    result = row.get("result")
    if not isinstance(result, dict):
        raise ValueError("persisted result missing")
    validate(
        result, audio_id, transcript=transcript,
        cap_ids={str(cap["cap_id"]) for cap in caps},
        min_quote_chars=6 if transcript_source == "reference" else 1,
    )
    registry = row.get("evidence_quote_normalization")
    expected_policy = (
        "first_supporting_cap_source_quote_v1"
        if transcript_source == "reference" else TARGET_EVIDENCE_POLICY
    )
    if not isinstance(registry, dict) or registry.get("policy") != expected_policy:
        raise ValueError("persisted evidence policy mismatch")
    repairs = registry.get("repairs")
    if not isinstance(repairs, list) or registry.get("num_grounded_quotes") != len(repairs):
        raise ValueError("persisted evidence repair registry malformed")
    if transcript_source == "reference":
        return
    by_location: dict[tuple[str, int], dict[str, Any]] = {}
    for repair in repairs:
        if not isinstance(repair, dict):
            raise ValueError("persisted target evidence repair malformed")
        location = (str(repair.get("section")), int(repair.get("item_index", -1)))
        if location in by_location:
            raise ValueError("duplicate persisted target evidence repair")
        by_location[location] = repair
    item_count = 0
    for section in SECTIONS:
        for index, item in enumerate(result["note"][section]):
            item_count += 1
            repair = by_location.get((section, index))
            if repair is None or repair.get("normalization") != TARGET_EVIDENCE_POLICY:
                raise ValueError("persisted target item lacks evidence-ID resolution")
            if not re.fullmatch(r"E\d{4,}", str(repair.get("target_evidence_id") or "")):
                raise ValueError("persisted target evidence ID malformed")
            start = int(repair.get("target_start_char", -1))
            end = int(repair.get("target_end_char", -1))
            quote = str(item["transcript_quote"])
            if start < 0 or end <= start or transcript[start:end] != quote:
                raise ValueError("persisted target evidence offsets mismatch")
            if repair.get("grounded_quote_sha256") != hashlib.sha256(quote.encode()).hexdigest():
                raise ValueError("persisted target evidence hash mismatch")
    if len(by_location) != item_count:
        raise ValueError("persisted target evidence repair cardinality mismatch")


def select_base_url(args, audio_id: str) -> str:
    pool = [value.strip().rstrip('/') for value in str(getattr(args, 'base_url_pool', '') or '').split(',') if value.strip()]
    if not pool:
        return args.base_url
    if args.backend != 'openai-compatible':
        raise ValueError('--base-url-pool is supported only for openai-compatible identical replicas')
    # Python's hash() is process-randomized.  Use a stable digest so resume and
    # retries route each audio to the same operational endpoint.
    index = int(hashlib.sha256(audio_id.encode()).hexdigest()[:16], 16) % len(pool)
    return pool[index]


def persist_failed_model_response(
    args: argparse.Namespace, *, audio_id: str, text: str, error: Exception,
) -> Path | None:
    """Persist malformed model output without prompts, headers, or secrets."""
    output_dir = getattr(args, 'failed_response_dir', None)
    if output_dir is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    path = output_dir / f'{audio_id}.{digest[:16]}.json'
    payload = {
        'schema_version': 'aegisvox.failed_model_response.v1',
        'audio_id': audio_id,
        'response_sha256': digest,
        'response_text': text,
        'error_type': type(error).__name__,
        'error': str(error)[:4000],
        'request_content_included': False,
        'credentials_included': False,
    }
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)
    return path


def call(
    args, messages, *, audio_id: str,
    max_items_per_section: int | None = None,
):
    base_url = select_base_url(args, audio_id)
    if args.backend=='openai_compat-bearer':
        token=os.environ.get('OPENAI_API_KEY','')
        if not token: raise RuntimeError('missing OPENAI_API_KEY')
        response=call_openai_compat_bearer_chat(base_url=base_url,bearer_token=token,model=args.model,messages=messages,temperature=0.0,max_tokens=args.max_tokens,timeout=args.timeout,verify_ssl=not args.no_verify_ssl,api_version=None,omit_api_version=False,service_name=None,completion_token_param='max_completion_tokens')
    else:
        response_format = (
            note_response_format(
                audio_id,
                max_items_per_section=(
                    args.max_items_per_section
                    if max_items_per_section is None
                    else max_items_per_section
                ),
                transcript_source=args.transcript_source,
            )
            if args.structured_output == 'json_schema' else None
        )
        response=call_chat_completions(base_url=base_url,api_key=args.api_key or 'not-required',model=args.model,messages=messages,temperature=0.0,timeout=args.timeout,max_retries=2,verify_ssl=not args.no_verify_ssl,max_tokens=args.max_tokens,response_format=response_format)
    response_text = extract_json_text_chat(response)
    try:
        result = safe_parse_model_json(response_text)
    except json.JSONDecodeError as exc:
        persist_failed_model_response(args, audio_id=audio_id, text=response_text, error=exc)
        raise
    return result, base_url


def repair_messages(
    base_messages:list[dict[str,str]], error:Exception, *,
    allowed_cap_ids: list[str] | None = None,
)->list[dict[str,str]]:
    """Request a full regeneration without accumulating prior long outputs.

    Long clinical transcripts plus CAP inventories can already occupy most of a
    model's context window.  Appending every invalid multi-thousand-token note
    to the next repair request makes the retry itself exceed the context limit.
    The immutable source prompt contains all evidence needed to regenerate the
    note, so retain it verbatim and add only the latest validation failure.
    """
    # Gemma-family OpenAI-compatible servers require strict role alternation.
    # Appending a second user message after the immutable user prompt yields
    # user/user and is rejected before inference.  Add the latest repair
    # instruction to that same immutable user turn instead.  This still avoids
    # accumulating prior multi-thousand-token model outputs and preserves the
    # complete transcript/CAP inventory byte-for-byte as a prefix.
    messages=[dict(message) for message in base_messages]
    malformed_json_guidance = ''
    if isinstance(error, json.JSONDecodeError):
        # Three long public rows entered a whitespace-only dead end after an
        # unterminated evidence string.  Stay within the original max-six-item
        # schema but ask the bounded repair generation to be materially shorter.
        # This changes neither the clinical inventory nor the final validation:
        # every retained item still needs valid CAP support and a literal quote.
        malformed_json_guidance = (
            ' The response was malformed or truncated JSON. Use at most three '
            'high-priority items per section, keep note text concise, close '
            'every JSON string and container, and stop immediately after the '
            'single complete JSON object.'
        )
    target_quote_guidance = ''
    error_text = str(error)
    if 'target evidence' in error_text or 'transcript quote is not literal evidence' in error_text:
        target_quote_guidance = (
            ' A target-note item is permitted only when COMPLETE TRANSFORMED '
            'TRANSCRIPT contains supporting words. Select exactly one valid '
            'target_evidence_id from DETERMINISTIC TARGET EVIDENCE UNITS that '
            'directly supports the item. If no listed unit supports it, omit '
            'the item instead of inventing or altering an evidence ID.'
        )
    cap_support_guidance = ''
    if 'invalid CAP support' in error_text:
        exact_allow_list = (
            ' The complete allowed CAP-ID list for this audio is: '
            + json.dumps(sorted(set(map(str, allowed_cap_ids or []))))
            + '.'
        )
        cap_support_guidance = (
            ' Every supporting_cap_ids entry must be copied exactly from a '
            'cap_id present in FROZEN SOURCE-DERIVED CAP ALLOW-LIST. Do not '
            'invent, rename, translate, shorten, or infer a CAP ID. If no '
            'listed CAP is supported by the transformed transcript, omit that '
            'note item rather than emitting an unlisted CAP ID.'
            f'{exact_allow_list}'
        )
    instruction=(
        '\n\nREPAIR REQUIREMENT: Your previous response failed the required '
        f'output contract: {error}. Regenerate the complete JSON from the '
        'supplied transcript and frozen CAP inventory. Return strict JSON only; '
        'do not omit any required section and do not add unsupported content.'
        f'{malformed_json_guidance}{target_quote_guidance}{cap_support_guidance}'
    )
    messages[-1]['content']=messages[-1]['content']+instruction
    return messages


def evaluate(pair,caps,args):
    transcript=pair[args.transcript_source]
    cap_ids={str(row['cap_id']) for row in caps}
    target_evidence_units: list[dict[str, Any]] = []
    if args.transcript_source == 'reference':
        system_prompt, user_template = SYSTEM_PROMPT, USER_TEMPLATE
        user_values = {
            'audio_id': pair['audio_id'],
            'dataset_id': pair.get('source_dataset_id') or 'unknown',
            'language': pair.get('source_language') or 'unknown',
            'caps': json.dumps(caps_for_generation(caps, transcript_source=args.transcript_source), ensure_ascii=False),
            'transcript': transcript,
        }
    else:
        system_prompt, user_template = TARGET_SYSTEM_PROMPT, TARGET_USER_TEMPLATE
        target_evidence_units = build_target_evidence_units(
            transcript,
            max_chars=getattr(args, 'target_evidence_max_chars', TARGET_EVIDENCE_MAX_CHARS),
        )
        prompt_units = [
            {'target_evidence_id': unit['evidence_id'], 'text': unit['text']}
            for unit in target_evidence_units
        ]
        user_values = {
            'audio_id': pair['audio_id'],
            'dataset_id': pair.get('source_dataset_id') or 'unknown',
            'language': pair.get('source_language') or 'unknown',
            'caps': json.dumps(caps_for_generation(caps, transcript_source=args.transcript_source), ensure_ascii=False),
            'transcript': transcript,
            'target_evidence_units': json.dumps(prompt_units, ensure_ascii=False),
        }
    base_messages=[{'role':'system','content':system_prompt},{'role':'user','content':user_template.format(**user_values)}]
    messages=base_messages
    requested_max_items_per_section = getattr(args, 'max_items_per_section', 6)
    effective_max_items_per_section = requested_max_items_per_section
    adaptive_malformed_json_repair_applied = False
    for attempt in range(args.schema_retries+1):
        try:
            # Parsing is part of the output contract.  Keep it inside the same
            # bounded retry loop as schema/evidence validation so malformed
            # JSON receives the frozen, non-accumulating repair instruction
            # instead of escaping before ``schema_retries`` can apply.
            result, base_url=call(
                args, messages, audio_id=pair['audio_id'],
                max_items_per_section=effective_max_items_per_section,
            )
            if args.transcript_source == 'reference':
                grounded, repairs = canonicalize_evidence_quotes(result, caps=caps, transcript=transcript)
            else:
                # For a target note, evidence must be literal in the transformed
                # transcript. Source CAP quotations are provenance/allow-list
                # fields and cannot be substituted as target evidence.
                grounded, repairs = resolve_target_evidence_ids(
                    result, transcript=transcript,
                    evidence_units=target_evidence_units,
                )
            validate(grounded,pair['audio_id'],transcript=transcript,cap_ids=cap_ids,min_quote_chars=6 if args.transcript_source=='reference' else 1)
            return (
                grounded, repairs, base_url,
                effective_max_items_per_section,
                adaptive_malformed_json_repair_applied,
            )
        except Exception as exc:
            if attempt>=args.schema_retries: raise
            if isinstance(exc, json.JSONDecodeError):
                # The prose repair instruction already requests at most three
                # high-priority items per section.  Enforce the same bounded
                # contract in structured decoding on the retry so a long row
                # cannot deterministically exhaust its output budget again.
                # This is fail-visible, outcome-independent repair: it is
                # triggered only by malformed/truncated JSON, and three is a
                # strict subset of the frozen six-item maximum.
                effective_max_items_per_section = min(
                    requested_max_items_per_section, 3,
                )
                adaptive_malformed_json_repair_applied = True
            messages=repair_messages(
                base_messages, exc, allowed_cap_ids=sorted(cap_ids),
            )
    raise AssertionError('unreachable')
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--paired-transcripts',type=Path,required=True); ap.add_argument('--source-caps',type=Path,required=True); ap.add_argument('--transcript-source',choices=['reference','hypothesis'],required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--backend',choices=['openai_compat-bearer','openai-compatible'],required=True); ap.add_argument('--model',required=True); ap.add_argument('--base-url',required=True); ap.add_argument('--base-url-pool',default='',help='comma-separated identical OpenAI-compatible replicas; stable audio-ID routing'); ap.add_argument('--env-file',type=Path); ap.add_argument('--api-key',default=''); ap.add_argument('--max-tokens',type=int,default=12000); ap.add_argument('--timeout',type=float,default=300); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--schema-retries',type=int,default=8); ap.add_argument('--limit',type=int); ap.add_argument('--audio-id',action='append',default=[]); ap.add_argument('--structured-output',choices=['json_object','json_schema'],default='json_object'); ap.add_argument('--max-items-per-section',type=int,default=6); ap.add_argument('--target-evidence-max-chars',type=int,default=TARGET_EVIDENCE_MAX_CHARS); ap.add_argument('--failed-response-dir',type=Path,help='optional content-addressed diagnostics for malformed model responses; prompts and credentials are never stored'); ap.add_argument('--resume',action='store_true'); ap.add_argument('--no-verify-ssl',action='store_true'); args=ap.parse_args(); load_env(args.env_file)
    if args.max_items_per_section < 1: raise ValueError('--max-items-per-section must be positive')
    if args.target_evidence_max_chars < 80: raise ValueError('--target-evidence-max-chars must be at least 80')
    pairs=group_pairs(read_paired_jsonl(args.paired_transcripts))
    if args.audio_id:
        selected=set(args.audio_id); pairs=[p for p in pairs if p['audio_id'] in selected]
        missing=selected-{p['audio_id'] for p in pairs}
        if missing: raise ValueError(f'unknown --audio-id values: {sorted(missing)}')
    pairs=pairs[:args.limit] if args.limit is not None else pairs
    caps_by_audio={str(row['audio_id']):row['caps'] for row in read_jsonl(args.source_caps)}
    if args.audio_id or args.limit is not None:
        caps_by_audio={p['audio_id']:caps_by_audio[p['audio_id']] for p in pairs}
    if {p['audio_id'] for p in pairs} != set(caps_by_audio): raise ValueError('source CAP population does not match note population')
    args.output_dir.mkdir(parents=True,exist_ok=True); lock=(args.output_dir/'.note_generation.lock').open('w')
    try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc: raise RuntimeError('another note generator owns this output') from exc
    output=args.output_dir/'notes.jsonl'; errors_path=args.output_dir/'errors.jsonl'; completed={}
    if args.resume and output.exists():
        for row in read_jsonl(output):
            try:
                pair=next(p for p in pairs if p['audio_id']==str(row['audio_id']))
                validate_persisted_row(
                    row, pair=pair,
                    caps=caps_by_audio[str(row['audio_id'])],
                    transcript_source=args.transcript_source,
                )
                completed[str(row['audio_id'])]=row
            except Exception: pass
    pending=[p for p in pairs if p['audio_id'] not in completed]; errors=[]
    with output.open('a') as out,errors_path.open('a') as err,ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures={pool.submit(evaluate,p,caps_by_audio[p['audio_id']],args):p for p in pending}
        for future in as_completed(futures):
            pair=futures[future]
            try:
                result, quote_repairs, inference_endpoint, effective_max_items, adaptive_repair=future.result(); evidence_policy='first_supporting_cap_source_quote_v1' if args.transcript_source=='reference' else TARGET_EVIDENCE_POLICY; item={'audio_id':pair['audio_id'],'dataset_id':pair.get('source_dataset_id'),'language':pair.get('source_language'),'transcript_source':args.transcript_source,'inference_endpoint':inference_endpoint,'result':result,'decoding_contract':{'requested_max_items_per_section':args.max_items_per_section,'effective_max_items_per_section':effective_max_items,'adaptive_malformed_json_repair_applied':adaptive_repair},'evidence_quote_normalization':{'policy':evidence_policy,'num_grounded_quotes':len(quote_repairs),'repairs':quote_repairs}}; completed[pair['audio_id']]=item; out.write(json.dumps(item,ensure_ascii=False)+'\n'); out.flush()
            except Exception as exc:
                item={'audio_id':pair['audio_id'],'error_type':type(exc).__name__,'error':str(exc)[:4000]}; errors.append(item); err.write(json.dumps(item,ensure_ascii=False)+'\n'); err.flush()
            print(json.dumps({'completed':len(completed),'errors':len(errors),'total':len(pairs)}),flush=True)
    with output.open('w') as out:
        for p in pairs:
            if p['audio_id'] in completed: out.write(json.dumps(completed[p['audio_id']],ensure_ascii=False)+'\n')
    total_quote_repairs=sum(int(row.get('evidence_quote_normalization',{}).get('num_grounded_quotes',0)) for row in completed.values())
    active_system_prompt = SYSTEM_PROMPT if args.transcript_source == 'reference' else TARGET_SYSTEM_PROMPT
    evidence_policy = 'first_supporting_cap_source_quote_v1' if args.transcript_source == 'reference' else TARGET_EVIDENCE_POLICY
    schema_version = 'aegisvox.clinical_note_generation.v3' if args.transcript_source == 'reference' else 'aegisvox.clinical_note_generation.v4'
    endpoint_pool=sorted({str(row.get('inference_endpoint') or args.base_url) for row in completed.values()})
    max_items_enforced = args.structured_output == 'json_schema'
    summary={
        'schema_version':schema_version,
        'model':args.model,
        'backend':args.backend,
        'transcript_source':args.transcript_source,
        'num_rows':len(pairs),
        'num_completed':len(completed),
        'num_errors':len(errors),
        'input_sha256':hashlib.sha256(args.paired_transcripts.read_bytes()).hexdigest(),
        'source_caps_sha256':hashlib.sha256(args.source_caps.read_bytes()).hexdigest(),
        'prompt_sha256':hashlib.sha256(active_system_prompt.encode()).hexdigest(),
        'full_transcript_used':True,
        'source_cap_anchored':True,
        'source_evidence_quotes_exposed_to_generator':args.transcript_source=='reference',
        'deterministic_proxy_used':False,
        'repair_context_policy':'immutable_source_prompt_plus_latest_validation_error',
        'evidence_quote_normalization_policy':evidence_policy,
        'target_evidence_contract':None if args.transcript_source=='reference' else TARGET_EVIDENCE_POLICY,
        'target_evidence_max_chars':None if args.transcript_source=='reference' else args.target_evidence_max_chars,
        'minimum_literal_quote_characters':6 if args.transcript_source=='reference' else 1,
        'num_grounded_evidence_quotes':total_quote_repairs,
        'structured_output':args.structured_output,
        # Keep the legacy field but state whether it was actually enforced.
        # OpenAI-compatible provider json_object mode cannot enforce JSON-schema maxItems,
        # whereas the vLLM json_schema path does.  Reporting both as a hard
        # six-item cap would silently misdescribe the generator factorial.
        'max_items_per_section':args.max_items_per_section,
        'max_items_per_section_requested':args.max_items_per_section,
        'max_items_per_section_enforced':max_items_enforced,
        'item_count_constraint':(
            'json_schema_max_items'
            if max_items_enforced
            else 'prompt_conciseness_only_no_hard_item_cap'
        ),
        'adaptive_malformed_json_repair_policy':'structured_max_items_3_after_json_decode_failure_v1',
        'num_adaptive_malformed_json_repairs':sum(
            bool(row.get('decoding_contract',{}).get('adaptive_malformed_json_repair_applied'))
            for row in completed.values()
        ),
        'inference_endpoint_count':len(endpoint_pool),
        'inference_endpoint_pool_sha256':hashlib.sha256(json.dumps(endpoint_pool,separators=(',',':')).encode()).hexdigest(),
    }
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary))
    return 0 if len(completed)==len(pairs) and not errors else 2
if __name__=='__main__': raise SystemExit(main())
