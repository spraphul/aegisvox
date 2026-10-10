#!/usr/bin/env python3
"""Run and score an evidence-constrained clinical judge qualification suite."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gpt_concept_preservation import (  # noqa: E402
    call_openai_compat_bearer_chat,
    call_chat_completions,
    extract_json_text_chat,
    safe_parse_model_json,
)


SYSTEM_PROMPT = """You are a strict clinical source-to-target integrity judge.

Compare the complete SOURCE and TARGET transcripts. Determine whether the
target preserves the source or contains one clinically meaningful corruption.
Treat a name-only synthetic replacement and a meaning-preserving paraphrase as
preserved. Be strict about negation, uncertainty, time, medication, dose,
measurement, severity/trend, speaker role, plan attribution, unsupported
insertions, and chronology. Quote exact evidence from both transcripts. Do not
use external medical knowledge to invent facts. Fail closed: if either complete
transcript is withheld, explicitly marked truncated, or otherwise insufficient
to support a whole-pair decision, return outcome=uncertain,
clinical_impact=uncertain, corruption_family=unknown, and empty evidence fields.
Never infer the answer from identifiers or dataset context. Return one JSON
object only.
"""

PARAPHRASED_SYSTEM_PROMPT = """Act as a conservative reviewer of clinical transcript fidelity.

Using only the complete labeled SOURCE and TARGET transcripts, classify whether
the TARGET keeps the SOURCE meaning or introduces one clinically relevant
change. A synthetic name substitution and a meaning-equivalent rewording are
preserved. Check polarity, uncertainty, timing, medicines, dose, measurements,
severity or trend, speaker/plan attribution, unsupported additions, and event
order. Support the decision with literal transcript excerpts. If the supplied
evidence is insufficient, return uncertain rather than guessing. Output exactly
one JSON object.
"""


SYSTEM_PROMPT_V2 = """You are a strict clinical source-to-target integrity judge.

SOURCE is the original clinical conversation and TARGET is the transformed
conversation. Compare only the supplied evidence segments. If the only change
is a person's direct identifier (for example, one name replaced by another),
set outcome=preserved, clinical_impact=none, and
corruption_family=phi_only_replacement; a changed name is not a speaker-role
change. A meaning-preserving paraphrase is also preserved. A change in
negation, certainty/uncertainty, duration, medication, dose, measurement,
severity/trend, speaker role, plan attribution, event order, or an unsupported
clinical addition is not preserved. In particular, changing uncertain language
such as "might" or "perhaps" into definite language such as "definitely" or
"certainly" is a contradiction even when the underlying event is unchanged.
Use outcome=missing when a source clinical proposition is absent from TARGET;
outcome=contradicted when TARGET expresses an incompatible value, polarity,
certainty, timing, or chronology; outcome=role_changed for speaker or plan
attribution changes; and outcome=unsupported_insertion for a target-only
clinical proposition.

Use only segment IDs supplied in the prompt; never copy or rewrite transcript
text. If either transcript is WITHHELD or TRUNCATED, set decision_status to
insufficient_evidence, outcome/impact/family to uncertain/uncertain/unknown,
and return empty evidence-ID lists. Otherwise set decision_status to
adjudicated and cite the smallest relevant source and target segment IDs.
Never infer the answer from case IDs or dataset context. Return one JSON object
only.
"""


PARAPHRASED_SYSTEM_PROMPT_V2 = """Conservatively assess clinical fidelity from SOURCE to TARGET.

The source is the reference conversation; the target is its transformed
version. Judge only from the numbered evidence segments. When only a direct
identifier such as a person's name changes, label the pair preserved with
family phi_only_replacement and impact none; do not call that a role change.
Equivalent wording also preserves meaning. Treat changes to polarity, epistemic
certainty, timing, medicines, dosage, measurements, severity, chronology,
speaker/plan ownership, deletion, or unsupported addition as clinically
meaningful. A shift from possible to definite (or the reverse) is a
contradiction. Classify a deleted source proposition as missing, an incompatible
target proposition as contradicted, a speaker/plan ownership change as
role_changed, and a target-only clinical assertion as unsupported_insertion.
Select supplied segment IDs instead of reproducing quotations.
If either side is withheld or explicitly truncated, abstain with
decision_status=insufficient_evidence and empty evidence lists. Return exactly
one JSON object.
"""


SYSTEM_PROMPT_V3 = """You are a conservative judge of one predeclared clinical proposition.

The REFERENCE PROPOSITION was selected before this judgment. Do not search for
additional source propositions and do not grade the whole conversation. For a
source_cap_preservation task, decide whether TARGET preserves that source
proposition with its negation, uncertainty, temporality, dose/value, severity,
speaker role, plan attribution, and event order intact. For a
target_assertion_support task, decide whether the target assertion is supported
by SOURCE; otherwise label it unsupported_insertion. A direct-identifier
replacement and a meaning-equivalent paraphrase are preserved when clinical
meaning and role are unchanged.

SOURCE and TARGET labels are authoritative regardless of which block appears
first. Never infer semantic direction from display order. Read both labeled
blocks before deciding. Use only supplied segment IDs. If either evidence side is WITHHELD or
TRUNCATED, abstain with decision_status=insufficient_evidence,
outcome/impact/family=uncertain/uncertain/unknown, and empty evidence-ID lists.
Apply this decision sequence:
1. If the only difference is the literal value of a direct identifier, return
   preserved/none/phi_only_replacement. The identifier value itself is not a
   clinical contradiction.
2. For target_assertion_support, return unsupported_insertion when SOURCE does
   not support the TARGET reference assertion, and cite at least one TARGET
   evidence ID. The assertion's presence in TARGET cannot support itself.
3. For source_cap_preservation, return missing when the proposition is absent,
   contradicted when a clinical modifier or value conflicts, and role_changed
   when speaker or plan ownership changes.
4. Use the most specific declared corruption family. A speaker or plan owner
   change is not a deletion merely because the original speaker no longer says
   the proposition.
5. Before returning missing, search TARGET for a statement about the same
   clinical event, entity, action, or speaker. If one exists but changes tense,
   chronology, polarity, uncertainty, medication, dose, measurement, severity,
   or attribution, it is contradicted or role_changed rather than missing.
Preserved, contradicted, and role_changed outcomes require evidence IDs from
both sides. Missing requires source evidence. Unsupported insertion requires
target evidence. Do not leave required evidence lists empty.
Return exactly one JSON object.
"""


PARAPHRASED_SYSTEM_PROMPT_V3 = """Assess one frozen clinical proposition against labeled evidence.

Evaluate only the declared REFERENCE PROPOSITION. In source_cap_preservation,
check whether TARGET retains the proposition and every clinically meaningful
modifier and attribution. In target_assertion_support, check whether SOURCE
supports the target assertion; unsupported target-only clinical content is an
unsupported_insertion. Names may be replaced without clinical change, and
equivalent wording is preserved. Treat SOURCE and TARGET labels as fixed even
when their display order changes. Select only supplied evidence IDs. Abstain
when either side is withheld or truncated. Literal direct-identifier changes
are preserved; target-only assertions require target evidence; speaker or plan
ownership changes are role_changed rather than missing. Display order never
changes SOURCE/TARGET semantics. For source preservation, compare the SOURCE
anchor against TARGET and treat a conflicting same-event statement as a
contradiction, not a deletion. For target assertion support, test the TARGET
claim only against SOURCE; a claim cannot support itself merely because it is
present in TARGET. Return one JSON object only.
"""

VARIANTS = (
    "repeat_1",
    "repeat_2",
    "repeat_3",
    "target_first",
    "rubric_permuted",
    "prompt_paraphrase",
    "evidence_removed",
    "context_truncated",
)


USER_TEMPLATE = """case_id: {case_id}
dataset_context: {dataset_context}
language: {language}

SOURCE TRANSCRIPT:
{source}

TARGET TRANSCRIPT:
{target}

Return exactly:
{{
  "case_id": "{case_id}",
  "outcome": "preserved|missing|contradicted|role_changed|unsupported_insertion|uncertain",
  "clinical_impact": "none|minor|major|potentially_harmful|uncertain",
  "corruption_family": "exact_control|benign_paraphrase|phi_only_replacement|critical_fact_deletion|negation_flip|uncertainty_flip|duration_change|medication_substitution|dose_change|measurement_change|severity_trend_change|speaker_role_swap|plan_attribution_swap|unsupported_insertion|chronology_reversal|unknown",
  "source_evidence": "exact source quote or empty only for exact control",
  "target_evidence": "exact target quote or empty only when the relevant fact is missing",
  "rationale": "brief evidence-grounded explanation"
}}
"""


USER_TEMPLATE_V2 = """case_id: {case_id}
dataset_context: {dataset_context}
language: {language}

{first_block}

{second_block}

Return exactly:
{{
  "case_id": "{case_id}",
  "decision_status": "adjudicated|insufficient_evidence",
  "outcome": "{outcome_options}",
  "clinical_impact": "{impact_options}",
  "corruption_family": "{family_options}",
  "source_evidence_ids": ["S01"],
  "target_evidence_ids": ["T01"],
  "rationale": "brief evidence-grounded explanation"
}}
"""


USER_TEMPLATE_V3 = """case_id: {case_id}
dataset_context: {dataset_context}
language: {language}
reference_direction: {reference_direction}
reference_proposition: {reference_proposition}

OPERAND CONTRACT:
{operand_contract}

{first_block}

{second_block}

{decision_directive}
Return exactly:
{{
  "case_id": "{case_id}",
  "decision_status": "adjudicated|insufficient_evidence",
  "outcome": "{outcome_options}",
  "clinical_impact": "{impact_options}",
  "corruption_family": "{family_options}",
  "source_evidence_ids": ["S01"],
  "target_evidence_ids": ["T01"],
  "rationale": "brief evidence-grounded explanation of this proposition only"
}}
"""


OUTCOMES = {"preserved", "missing", "contradicted", "role_changed", "unsupported_insertion", "uncertain"}
IMPACTS = {"none", "minor", "major", "potentially_harmful", "uncertain"}
FAMILIES = {
    "exact_control", "benign_paraphrase", "phi_only_replacement", "critical_fact_deletion",
    "negation_flip", "uncertainty_flip", "duration_change", "medication_substitution",
    "dose_change", "measurement_change", "severity_trend_change", "speaker_role_swap",
    "plan_attribution_swap", "unsupported_insertion", "chronology_reversal", "unknown",
}


def read_env_file(path: Path | None) -> None:
    if not path or not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_result(
    result: dict[str, Any],
    case_id: str,
    *,
    source_transcript: str | None = None,
    target_transcript: str | None = None,
) -> None:
    if str(result.get("case_id")) != case_id:
        raise ValueError(f"case_id mismatch: {result.get('case_id')!r}")
    if result.get("outcome") not in OUTCOMES:
        raise ValueError(f"invalid outcome: {result.get('outcome')!r}")
    if result.get("clinical_impact") not in IMPACTS:
        raise ValueError(f"invalid clinical_impact: {result.get('clinical_impact')!r}")
    if result.get("corruption_family") not in FAMILIES:
        raise ValueError(f"invalid corruption_family: {result.get('corruption_family')!r}")
    if not str(result.get("rationale") or "").strip():
        raise ValueError("rationale must be non-empty")
    if result["outcome"] != "preserved" and not (
        str(result.get("source_evidence") or "").strip() or str(result.get("target_evidence") or "").strip()
    ):
        raise ValueError("non-preserved judgment requires transcript evidence")
    if source_transcript is not None and not normalized_contains(source_transcript, str(result.get("source_evidence") or "")):
        raise ValueError("source_evidence is not an exact source-transcript substring")
    if target_transcript is not None and not normalized_contains(target_transcript, str(result.get("target_evidence") or "")):
        raise ValueError("target_evidence is not an exact target-transcript substring")


def _variant_row(row: dict[str, Any], variant: str) -> dict[str, Any]:
    varied = dict(row)
    if variant == "evidence_removed":
        # The expected response is fail-closed uncertainty: this deliberately
        # withholds the transcript evidence rather than asking the judge to
        # ignore evidence that remains visible.
        varied["source_transcript"] = "[SOURCE EVIDENCE WITHHELD]"
        varied["target_transcript"] = "[TARGET EVIDENCE WITHHELD]"
    elif variant == "context_truncated":
        # Preserve only neutral role/context fragments and remove the decisive
        # proposition.  This tests whether the judge guesses preservation.
        source = str(row.get("source_transcript") or "")
        target = str(row.get("target_transcript") or "")
        varied["source_transcript"] = source[: max(1, len(source) // 5)] + " [CONTEXT TRUNCATED]"
        varied["target_transcript"] = target[: max(1, len(target) // 5)] + " [CONTEXT TRUNCATED]"
    return varied


def _model_case_id(row: dict[str, Any]) -> str:
    """Return an opaque ID that cannot reveal the controlled corruption label."""
    return "case_" + hashlib.sha256(str(row["case_id"]).encode("utf-8")).hexdigest()[:16]


def _evidence_segments(text: str, prefix: str) -> list[dict[str, str]]:
    """Create stable, literal evidence candidates without semantic inference.

    The controlled suite uses punctuated speaker-attributed transcripts.  This
    splitter keeps every candidate a literal contiguous substring and therefore
    avoids grading a model on punctuation-perfect quote transcription.
    """
    value = " ".join(str(text or "").split())
    if not value:
        return []
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+(?=(?:PATIENT|CLINICIAN|CLINICIEN|CALLER|DISPATCHER)\s*:|[A-ZÀ-ÖØ-Þ])", value)
        if part.strip()
    ]
    if not parts:
        parts = [value]
    return [{"id": f"{prefix}{index:02d}", "text": part} for index, part in enumerate(parts, 1)]


def _v2_evidence_state(row: dict[str, Any], variant: str) -> dict[str, Any]:
    varied = _variant_row(row, variant)
    if variant == "evidence_removed":
        return {
            "source_status": "WITHHELD",
            "target_status": "WITHHELD",
            "source_segments": [],
            "target_segments": [],
        }
    if variant == "context_truncated":
        return {
            "source_status": "TRUNCATED",
            "target_status": "TRUNCATED",
            "source_segments": _evidence_segments(varied["source_transcript"], "S"),
            "target_segments": _evidence_segments(varied["target_transcript"], "T"),
        }
    return {
        "source_status": "COMPLETE",
        "target_status": "COMPLETE",
        "source_segments": _evidence_segments(varied["source_transcript"], "S"),
        "target_segments": _evidence_segments(varied["target_transcript"], "T"),
    }


def _format_evidence_block(label: str, status: str, segments: list[dict[str, str]]) -> str:
    lines = [f"{label} STATUS: {status}", f"{label} EVIDENCE SEGMENTS:"]
    if not segments:
        lines.append("[NO EVIDENCE AVAILABLE]")
    else:
        lines.extend(f"[{item['id']}] {item['text']}" for item in segments)
    return "\n".join(lines)


def make_messages_v2(row: dict[str, Any], variant: str = "repeat_1") -> list[dict[str, str]]:
    """Versioned evidence-ID/fail-closed contract for post-v1 qualification."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown qualification variant: {variant}")
    evidence = _v2_evidence_state(row, variant)
    outcome_options = "preserved|missing|contradicted|role_changed|unsupported_insertion|uncertain"
    impact_options = "none|minor|major|potentially_harmful|uncertain"
    family_options = (
        "exact_control|benign_paraphrase|phi_only_replacement|critical_fact_deletion|negation_flip|"
        "uncertainty_flip|duration_change|medication_substitution|dose_change|measurement_change|"
        "severity_trend_change|speaker_role_swap|plan_attribution_swap|unsupported_insertion|"
        "chronology_reversal|unknown"
    )
    if variant == "rubric_permuted":
        outcome_options = "uncertain|unsupported_insertion|role_changed|contradicted|missing|preserved"
        impact_options = "uncertain|potentially_harmful|major|minor|none"
        family_options = "|".join(reversed(family_options.split("|")))
    source_block = _format_evidence_block(
        "SOURCE", evidence["source_status"], evidence["source_segments"],
    )
    target_block = _format_evidence_block(
        "TARGET", evidence["target_status"], evidence["target_segments"],
    )
    first_block, second_block = (target_block, source_block) if variant == "target_first" else (source_block, target_block)
    user = USER_TEMPLATE_V2.format(
        case_id=_model_case_id(row),
        dataset_context=row["dataset_context"],
        language=row["language"],
        first_block=first_block,
        second_block=second_block,
        outcome_options=outcome_options,
        impact_options=impact_options,
        family_options=family_options,
    )
    system = PARAPHRASED_SYSTEM_PROMPT_V2 if variant == "prompt_paraphrase" else SYSTEM_PROMPT_V2
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _v3_evidence_state(row: dict[str, Any], variant: str) -> dict[str, Any]:
    state = _v2_evidence_state(row, variant)
    direction = str(row.get("reference_direction") or "")
    if direction not in {"source_cap_preservation", "target_assertion_support"}:
        raise ValueError(f"invalid reference_direction: {direction!r}")
    proposition = str(row.get("reference_proposition") or "").strip()
    if not proposition:
        raise ValueError("reference_proposition must be non-empty")
    state["reference_direction"] = direction
    state["reference_proposition"] = proposition
    return state


def _v3_operand_contract(direction: str) -> str:
    if direction == "source_cap_preservation":
        return (
            "REFERENCE SIDE = SOURCE. SOURCE contains the anchor proposition. TARGET is the candidate rendering. "
            "Locate the same clinical event/entity/action in TARGET even when its wording or modifier changed. "
            "A conflicting same-event TARGET statement is contradicted (or role_changed), not missing."
        )
    if direction == "target_assertion_support":
        return (
            "REFERENCE SIDE = TARGET. TARGET contains the claim under test. SOURCE is the only support corpus. "
            "Do not treat the claim's presence in TARGET as evidence that SOURCE supports it. If SOURCE does not "
            "support the TARGET claim, return unsupported_insertion."
        )
    raise ValueError(f"invalid reference_direction: {direction!r}")


def _v3_decision_directive(evidence: dict[str, Any]) -> str:
    incomplete = evidence["source_status"] != "COMPLETE" or evidence["target_status"] != "COMPLETE"
    if incomplete:
        return (
            "FAIL-CLOSED OVERRIDE: At least one labeled evidence block is WITHHELD or TRUNCATED. "
            "Do not adjudicate the proposition and do not infer from visible fragments. Set decision_status to "
            "insufficient_evidence, outcome to uncertain, clinical_impact to uncertain, corruption_family to "
            "unknown, and both evidence-ID lists to []."
        )
    return (
        "Judge only the reference proposition above. First apply the OPERAND CONTRACT, then read both complete "
        "labeled blocks. The block shown first has no semantic priority. Before choosing missing, check for a "
        "conflicting statement about the same event in TARGET."
    )


def make_messages_v3(row: dict[str, Any], variant: str = "repeat_1") -> list[dict[str, str]]:
    """CAP-local, role-explicit contract designed for order robustness."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown qualification variant: {variant}")
    evidence = _v3_evidence_state(row, variant)
    outcome_options = "preserved|missing|contradicted|role_changed|unsupported_insertion|uncertain"
    impact_options = "none|minor|major|potentially_harmful|uncertain"
    family_options = (
        "exact_control|benign_paraphrase|phi_only_replacement|critical_fact_deletion|negation_flip|"
        "uncertainty_flip|duration_change|medication_substitution|dose_change|measurement_change|"
        "severity_trend_change|speaker_role_swap|plan_attribution_swap|unsupported_insertion|"
        "chronology_reversal|unknown"
    )
    if variant == "rubric_permuted":
        outcome_options = "uncertain|unsupported_insertion|role_changed|contradicted|missing|preserved"
        impact_options = "uncertain|potentially_harmful|major|minor|none"
        family_options = "|".join(reversed(family_options.split("|")))
    source_block = _format_evidence_block(
        "SOURCE", evidence["source_status"], evidence["source_segments"],
    )
    target_block = _format_evidence_block(
        "TARGET", evidence["target_status"], evidence["target_segments"],
    )
    first_block, second_block = (
        (target_block, source_block) if variant == "target_first" else (source_block, target_block)
    )
    user = USER_TEMPLATE_V3.format(
        case_id=_model_case_id(row),
        dataset_context=row["dataset_context"],
        language=row["language"],
        reference_direction=evidence["reference_direction"],
        reference_proposition=evidence["reference_proposition"],
        operand_contract=_v3_operand_contract(evidence["reference_direction"]),
        first_block=first_block,
        second_block=second_block,
        decision_directive=_v3_decision_directive(evidence),
        outcome_options=outcome_options,
        impact_options=impact_options,
        family_options=family_options,
    )
    system = PARAPHRASED_SYSTEM_PROMPT_V3 if variant == "prompt_paraphrase" else SYSTEM_PROMPT_V3
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_result_v2(
    result: dict[str, Any],
    case_id: str,
    *,
    evidence_state: dict[str, Any],
) -> dict[str, Any]:
    """Validate IDs and materialize exact quotes for downstream compatibility."""
    if str(result.get("case_id")) != case_id:
        raise ValueError(f"case_id mismatch: {result.get('case_id')!r}")
    status = result.get("decision_status")
    if status not in {"adjudicated", "insufficient_evidence"}:
        raise ValueError(f"invalid decision_status: {status!r}")
    if result.get("outcome") not in OUTCOMES:
        raise ValueError(f"invalid outcome: {result.get('outcome')!r}")
    if result.get("clinical_impact") not in IMPACTS:
        raise ValueError(f"invalid clinical_impact: {result.get('clinical_impact')!r}")
    if result.get("corruption_family") not in FAMILIES:
        raise ValueError(f"invalid corruption_family: {result.get('corruption_family')!r}")
    if not str(result.get("rationale") or "").strip():
        raise ValueError("rationale must be non-empty")
    source_ids = result.get("source_evidence_ids")
    target_ids = result.get("target_evidence_ids")
    if not isinstance(source_ids, list) or not all(isinstance(v, str) for v in source_ids):
        raise ValueError("source_evidence_ids must be a string list")
    if not isinstance(target_ids, list) or not all(isinstance(v, str) for v in target_ids):
        raise ValueError("target_evidence_ids must be a string list")
    if len(source_ids) != len(set(source_ids)) or len(target_ids) != len(set(target_ids)):
        raise ValueError("evidence IDs must be unique")
    if len(source_ids) > 3 or len(target_ids) > 3:
        raise ValueError("cite at most three evidence segments per side")

    source_map = {item["id"]: item["text"] for item in evidence_state["source_segments"]}
    target_map = {item["id"]: item["text"] for item in evidence_state["target_segments"]}
    unknown_source = [value for value in source_ids if value not in source_map]
    unknown_target = [value for value in target_ids if value not in target_map]
    if unknown_source or unknown_target:
        raise ValueError(f"unknown evidence IDs: source={unknown_source}, target={unknown_target}")

    incomplete_input = evidence_state["source_status"] != "COMPLETE" or evidence_state["target_status"] != "COMPLETE"
    if status == "insufficient_evidence":
        if not incomplete_input:
            raise ValueError("insufficient_evidence is only valid for withheld/truncated qualification inputs")
        if (result["outcome"], result["clinical_impact"], result["corruption_family"]) != (
            "uncertain", "uncertain", "unknown",
        ):
            raise ValueError("insufficient_evidence requires uncertain/uncertain/unknown")
        if source_ids or target_ids:
            raise ValueError("insufficient_evidence requires empty evidence-ID lists")
    else:
        if incomplete_input:
            raise ValueError("withheld/truncated input must fail closed")
        if result["outcome"] == "uncertain":
            raise ValueError("adjudicated output cannot have uncertain outcome")
        if result["outcome"] == "missing":
            if not source_ids:
                raise ValueError("missing judgment requires source evidence")
        elif result["outcome"] == "unsupported_insertion":
            if not target_ids:
                raise ValueError("unsupported insertion requires target evidence")
        elif not source_ids or not target_ids:
            raise ValueError("adjudicated judgment requires source and target evidence")

    materialized = dict(result)
    materialized["source_evidence"] = source_map[source_ids[0]] if source_ids else ""
    materialized["target_evidence"] = target_map[target_ids[0]] if target_ids else ""
    materialized["source_evidence_quotes"] = [source_map[value] for value in source_ids]
    materialized["target_evidence_quotes"] = [target_map[value] for value in target_ids]
    return materialized


def make_messages(row: dict[str, Any], variant: str = "repeat_1") -> list[dict[str, str]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown qualification variant: {variant}")
    varied = _variant_row(row, variant)
    system_prompt = PARAPHRASED_SYSTEM_PROMPT if variant == "prompt_paraphrase" else SYSTEM_PROMPT
    outcome_options = "preserved|missing|contradicted|role_changed|unsupported_insertion|uncertain"
    impact_options = "none|minor|major|potentially_harmful|uncertain"
    family_options = (
        "exact_control|benign_paraphrase|phi_only_replacement|critical_fact_deletion|negation_flip|"
        "uncertainty_flip|duration_change|medication_substitution|dose_change|measurement_change|"
        "severity_trend_change|speaker_role_swap|plan_attribution_swap|unsupported_insertion|"
        "chronology_reversal|unknown"
    )
    if variant == "rubric_permuted":
        outcome_options = "uncertain|unsupported_insertion|role_changed|contradicted|missing|preserved"
        impact_options = "uncertain|potentially_harmful|major|minor|none"
        family_options = "|".join(reversed(family_options.split("|")))
    template = USER_TEMPLATE.replace(
        "preserved|missing|contradicted|role_changed|unsupported_insertion|uncertain", outcome_options,
    ).replace(
        "none|minor|major|potentially_harmful|uncertain", impact_options,
    ).replace(
        "exact_control|benign_paraphrase|phi_only_replacement|critical_fact_deletion|negation_flip|uncertainty_flip|duration_change|medication_substitution|dose_change|measurement_change|severity_trend_change|speaker_role_swap|plan_attribution_swap|unsupported_insertion|chronology_reversal|unknown",
        family_options,
    )
    user = template.format(
        case_id=_model_case_id(row),
        dataset_context=row["dataset_context"],
        language=row["language"],
        source=varied["source_transcript"],
        target=varied["target_transcript"],
    )
    if variant == "target_first":
        source_marker = "SOURCE TRANSCRIPT:\n"
        target_marker = "TARGET TRANSCRIPT:\n"
        prefix, rest = user.split(source_marker, 1)
        source_text, rest = rest.split("\n\n" + target_marker, 1)
        target_text, suffix = rest.split("\n\nReturn exactly:", 1)
        user = (
            prefix + target_marker + target_text + "\n\n" + source_marker + source_text
            + "\n\nReturn exactly:" + suffix
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def normalized_contains(container: str, quote: str) -> bool:
    """Check verbatim evidence while tolerating whitespace-only differences."""
    if not quote.strip():
        return True
    norm = lambda value: " ".join(value.casefold().split())
    return norm(quote) in norm(container)


def call_model(args: argparse.Namespace, messages: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    if args.backend == "openai_compat-bearer":
        token = os.environ.get("OPENAI_API_KEY", "")
        if not token:
            raise RuntimeError("missing OPENAI_API_KEY")
        response = call_openai_compat_bearer_chat(
            base_url=args.base_url,
            bearer_token=token,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            verify_ssl=not args.no_verify_ssl,
            api_version=args.api_version,
            omit_api_version=args.omit_api_version,
            service_name=None,
            completion_token_param="max_completion_tokens",
        )
    else:
        response = call_chat_completions(
            base_url=args.base_url,
            api_key=args.api_key or "not-required",
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=2,
            verify_ssl=not args.no_verify_ssl,
            max_tokens=args.max_tokens,
        )
    raw = extract_json_text_chat(response)
    provenance = {
        "latency_sec": time.monotonic() - started,
        "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "usage": response.get("usage"),
        "response_model": response.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
    }
    return safe_parse_model_json(raw), provenance


def evaluate_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    contract_version = getattr(args, "contract_version", "v1")
    if contract_version == "v3":
        messages = make_messages_v3(row, args.variant)
    elif contract_version == "v2":
        messages = make_messages_v2(row, args.variant)
    else:
        messages = make_messages(row, args.variant)
    model_case_id = _model_case_id(row)
    last_error: Exception | None = None
    for attempt in range(args.schema_retries + 1):
        result: dict[str, Any] | None = None
        try:
            result, provenance = call_model(args, messages)
            varied = _variant_row(row, args.variant)
            if contract_version == "v3":
                result = validate_result_v2(
                    result,
                    model_case_id,
                    evidence_state=_v3_evidence_state(row, args.variant),
                )
            elif contract_version == "v2":
                result = validate_result_v2(
                    result,
                    model_case_id,
                    evidence_state=_v2_evidence_state(row, args.variant),
                )
            else:
                validate_result(
                    result,
                    model_case_id,
                    source_transcript=varied["source_transcript"],
                    target_transcript=varied["target_transcript"],
                )
            # Restore the suite identifier only after validating the model's
            # opaque response ID; retain both identifiers for auditability.
            result["model_case_id"] = model_case_id
            result["case_id"] = row["case_id"]
            provenance["model_case_id"] = model_case_id
            return {"result": result, "provenance": provenance}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= args.schema_retries:
                raise
            # Return the invalid object to the model.  Without this assistant
            # turn, "correct your response" asks for a fresh judgment and can
            # move an otherwise-local evidence error to another field.
            if result is not None:
                messages = [
                    *messages,
                    {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)},
                ]
            messages = [
                *messages,
                {"role": "user", "content": (
                    f"Your response failed validation: {type(exc).__name__}: {exc}. "
                    "Repair only the invalid field(s), preserve the remaining judgment, and return one corrected "
                    "complete JSON object using only the declared enums. "
                    + (
                        "Evidence fields must use only the declared segment IDs. Do not copy transcript text. "
                        "For WITHHELD or TRUNCATED inputs, return decision_status=insufficient_evidence, "
                        "outcome=uncertain, clinical_impact=uncertain, corruption_family=unknown, and empty "
                        "source_evidence_ids and target_evidence_ids."
                        if contract_version in {"v2", "v3"} else
                        "Evidence fields must be literal contiguous transcript substrings: do not add role prefixes, "
                        "quotation marks, ellipses, or words that are absent from the corresponding transcript."
                    )
                )},
            ]
    raise RuntimeError(str(last_error))


def summarize(
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    errors: list[dict[str, Any]],
    variant: str = "repeat_1",
) -> dict[str, Any]:
    scored = []
    for row in rows:
        item = results.get(row["case_id"])
        if item is None:
            continue
        result = item.get("result", item)
        known = row["known_label"]
        stress = variant in {"evidence_removed", "context_truncated"}
        expected_outcome = "uncertain" if stress else known["expected_outcome"]
        scored.append({
            "case_id": row["case_id"],
            "dataset_id": row["dataset_id"],
            "language": row["language"],
            "family": known["corruption_family"],
            "expected_outcome": expected_outcome,
            "predicted_outcome": result["outcome"],
            "outcome_correct": result["outcome"] == expected_outcome,
            "family_correct": None if stress else result["corruption_family"] == known["corruption_family"],
            "is_harmful": known["clinical_impact"] in {"major", "potentially_harmful"},
            "harmful_detected": result["outcome"] not in {"preserved", "uncertain"},
            "source_evidence_valid": normalized_contains(row.get("source_transcript", ""), str(result.get("source_evidence") or "")),
            "target_evidence_valid": normalized_contains(row.get("target_transcript", ""), str(result.get("target_evidence") or "")),
        })
    n = len(scored)
    controls = [x for x in scored if x["expected_outcome"] == "preserved"]
    harmful = [x for x in scored if x["is_harmful"]] if variant not in {"evidence_removed", "context_truncated"} else []
    return {
        "schema_version": "aegisvox_clinical_judge_qualification.v1",
        "num_cases": len(rows),
        "num_completed": n,
        "num_errors": len(errors),
        "outcome_accuracy": sum(x["outcome_correct"] for x in scored) / n if n else None,
        "corruption_family_accuracy": (
            sum(bool(x["family_correct"]) for x in scored) / n
            if n and variant not in {"evidence_removed", "context_truncated"} else None
        ),
        "benign_specificity": sum(x["predicted_outcome"] == "preserved" for x in controls) / len(controls) if controls else None,
        "harmful_corruption_sensitivity": sum(x["harmful_detected"] for x in harmful) / len(harmful) if harmful else None,
        "exact_evidence_validity": (
            sum(x["source_evidence_valid"] and x["target_evidence_valid"] for x in scored) / n if n else None
        ),
        "fail_closed_uncertain_rate": (
            sum(x["predicted_outcome"] == "uncertain" for x in scored) / n
            if n and variant in {"evidence_removed", "context_truncated"} else None
        ),
        "scored_cases": scored,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["openai_compat-bearer", "openai-compatible"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reported-model",
        help="Exact registry/checkpoint identity when the serving API uses an alias.",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--omit-api-version", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--language", action="append", help="Optional language filter; may be repeated")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--no-verify-ssl", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, default="repeat_1")
    parser.add_argument(
        "--contract-version", choices=("v1", "v2", "v3"), default="v1",
        help="Versioned response/evidence contract; v1 remains available to reproduce frozen runs.",
    )
    args = parser.parse_args()
    read_env_file(args.env_file)
    rows = read_jsonl(args.suite)
    if args.language:
        allowed_languages = set(args.language)
        rows = [row for row in rows if str(row.get("language")) in allowed_languages]
    if args.start_index:
        rows = rows[args.start_index:]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "item_judgments.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    results: dict[str, dict[str, Any]] = {}
    row_by_case = {row["case_id"]: row for row in rows}
    if args.resume and results_path.exists():
        for item in read_jsonl(results_path):
            payload = item.get("payload") or {"result": item["result"], "provenance": item.get("provenance", {})}
            result = payload["result"]
            row = row_by_case.get(item["case_id"])
            if row is None:
                continue
            try:
                if args.contract_version == "v3":
                    validate_result_v2(
                        result,
                        item["case_id"],
                        evidence_state=_v3_evidence_state(row, args.variant),
                    )
                elif args.contract_version == "v2":
                    validate_result_v2(
                        result,
                        item["case_id"],
                        evidence_state=_v2_evidence_state(row, args.variant),
                    )
                else:
                    validate_result(
                        result,
                        item["case_id"],
                        source_transcript=_variant_row(row, args.variant)["source_transcript"],
                        target_transcript=_variant_row(row, args.variant)["target_transcript"],
                    )
            except ValueError:
                # Stale schema/evidence failures are deliberately retried.
                continue
            results[item["case_id"]] = payload

    pending = [row for row in rows if row["case_id"] not in results]
    errors: list[dict[str, Any]] = []
    # Persist each completed item before scheduling the next status update.  A
    # local bearer session or terminal can disappear during a long run; resume
    # must never replay already completed paper-evaluation calls.
    with results_path.open("a", encoding="utf-8") as result_log, errors_path.open("a", encoding="utf-8") as error_log:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            future_map = {pool.submit(evaluate_one, row, args): row for row in pending}
            for future in as_completed(future_map):
                row = future_map[future]
                try:
                    payload = future.result()
                    results[row["case_id"]] = payload
                    result_log.write(json.dumps({"case_id": row["case_id"], "payload": payload}, ensure_ascii=False) + "\n")
                    result_log.flush()
                except Exception as exc:  # noqa: BLE001
                    error = {"case_id": row["case_id"], "error_type": type(exc).__name__, "error": str(exc)[:3000]}
                    errors.append(error)
                    error_log.write(json.dumps(error, ensure_ascii=False) + "\n")
                    error_log.flush()
                print(json.dumps({"completed": len(results), "errors": len(errors), "total": len(rows)}), flush=True)

    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["case_id"] in results:
                handle.write(json.dumps({"case_id": row["case_id"], "payload": results[row["case_id"]]}, ensure_ascii=False) + "\n")
    with errors_path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")

    summary = summarize(rows, results, errors, args.variant)
    summary["schema_version"] = f"aegisvox_clinical_judge_qualification.{args.contract_version}"
    if args.contract_version == "v3":
        prompt_builder = make_messages_v3
        system_prompt = SYSTEM_PROMPT_V3
    elif args.contract_version == "v2":
        prompt_builder = make_messages_v2
        system_prompt = SYSTEM_PROMPT_V2
    else:
        prompt_builder = make_messages
        system_prompt = SYSTEM_PROMPT
    summary.update({
        "model": args.reported_model or args.model,
        "api_model": args.model,
        "backend": args.backend,
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "contract_version": args.contract_version,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "variant": args.variant,
        "prompt_sha256": hashlib.sha256(json.dumps(prompt_builder(rows[0], args.variant), sort_keys=True).encode()).hexdigest() if rows else None,
        "language_filter": args.language or [],
    })
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "scored_cases"}, indent=2))
    return 0 if not errors and len(results) == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
