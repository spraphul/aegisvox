#!/usr/bin/env python3
"""Aggregate the complete note-study design, including frozen judge exclusions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_clinical_note_pair_adjudication import (  # noqa: E402
    SCHEMA_REPAIR_POLICY,
    SCHEMA_REPAIR_PROMPT,
    SCHEMA_REPAIR_POLICY_V2,
    SCHEMA_REPAIR_PROMPT_V2,
)


FIELDS = (
    "num_rows", "num_completed", "section_results",
    "unsupported_note_assertion_count", "unsupported_note_assertion_impacts",
    "modifier_error_counts", "section_leakage_count", "exactness_counts",
    "conversation_assessment_counts", "note_no_major_error_rate",
    "mean_structural_section_jaccard", "exact_structural_section_agreement_rate",
    "source_note_word_count", "target_note_word_count",
    "target_source_note_compression_ratio",
)

EXPECTED_CONDITIONS = {
    "seedvc_only", "stock_voicecraft", "aegisvox_f5",
    "localized_redaction", "full_regeneration",
}
EXPECTED_MODELS = {"gpt54", "medgemma27b"}
JUDGE_CONTRACTS = {
    "gpt54": {
        "model": "gpt-5.4",
        "backend": "openai_compat-bearer",
    },
    "medgemma27b": {
        "model": "google/medgemma-27b-text-it",
        "backend": "openai-compatible",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object")
        rows.append(row)
    return rows


def audit_adjudication_rows(summary_path: Path, expected_rows: int) -> dict[str, Any]:
    """Bind a cell summary to its exact row-level adjudication evidence."""
    rows_path = summary_path.parent / "note_pair_adjudications.jsonl"
    if not rows_path.is_file():
        raise ValueError(f"missing row-level adjudications beside {summary_path}")
    rows = read_jsonl(rows_path)
    audio_ids: list[str] = []
    for index, row in enumerate(rows):
        audio_id = str(row.get("audio_id") or "")
        result = row.get("result")
        if not audio_id or not isinstance(result, dict):
            raise ValueError(f"invalid row-level adjudication {rows_path}:{index + 1}")
        if str(result.get("audio_id") or "") != audio_id:
            raise ValueError(f"row/result audio_id mismatch {rows_path}:{index + 1}")
        if not isinstance(result.get("cap_adjudications"), list):
            raise ValueError(f"missing CAP adjudications {rows_path}:{index + 1}")
        if result.get("conversation_assessment") not in {
            "no_major_error", "major_omission", "major_contradiction",
            "major_unsupported_addition", "uncertain",
        }:
            raise ValueError(f"invalid conversation assessment {rows_path}:{index + 1}")
        audio_ids.append(audio_id)
    if len(rows) != expected_rows or len(set(audio_ids)) != expected_rows:
        raise ValueError(
            f"row-level adjudication population mismatch for {rows_path}: "
            f"rows={len(rows)} unique_audio_ids={len(set(audio_ids))} expected={expected_rows}"
        )
    return {
        "path": str(rows_path),
        "sha256": sha256(rows_path),
        "num_rows": len(rows),
        "num_unique_audio_ids": len(set(audio_ids)),
    }


def schema_repair_contract(data: dict[str, Any]) -> dict[str, Any]:
    """Validate legacy or evidence-preserving invalid-output repair provenance."""
    policy = data.get("schema_repair_policy")
    prompt_hash = data.get("schema_repair_prompt_sha256")
    if policy is None and prompt_hash is None:
        return {
            "policy": "generic_complete_json_repair_v1",
            "prompt_sha256": None,
            "scope": "invalid model outputs only; accepted rows satisfy the unchanged strict validator",
        }
    expected_scope = (
        "invalid model outputs only; every accepted row satisfies the unchanged strict validator"
    )
    contracts = {
        SCHEMA_REPAIR_POLICY_V2: {
            "prompt_sha256": hashlib.sha256(SCHEMA_REPAIR_PROMPT_V2.encode()).hexdigest(),
            "history": ["generic_complete_json_repair_v1", SCHEMA_REPAIR_POLICY_V2],
        },
        SCHEMA_REPAIR_POLICY: {
            "prompt_sha256": hashlib.sha256(SCHEMA_REPAIR_PROMPT.encode()).hexdigest(),
            "history": [
                "generic_complete_json_repair_v1",
                SCHEMA_REPAIR_POLICY_V2,
                SCHEMA_REPAIR_POLICY,
            ],
        },
    }
    contract = contracts.get(policy)
    if (
        contract is None
        or prompt_hash != contract["prompt_sha256"]
        or data.get("schema_repair_policy_history") != contract["history"]
        or data.get("schema_repair_scope") != expected_scope
    ):
        raise ValueError("unknown or incomplete schema-repair provenance")
    return {
        "policy": policy,
        "prompt_sha256": prompt_hash,
        "scope": expected_scope,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell", action="append", required=True,
        help="condition|generator|judge=summary.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=390)
    parser.add_argument("--expected-cells", type=int, default=20)
    parser.add_argument(
        "--generation-audit", action="append", required=True,
        help="generator=clinical_note_generation_panel_audit.json",
    )
    parser.add_argument(
        "--excluded-judge", action="append", default=[],
        help="judge=exclusion.json for a judge excluded by a frozen qualification gate",
    )
    args = parser.parse_args()

    generation_audits = {}
    for raw in args.generation_audit:
        generator, raw_path = raw.split("=", 1)
        if generator in generation_audits:
            raise ValueError(f"duplicate generation audit: {generator}")
        path = Path(raw_path)
        data = json.loads(path.read_text())
        if (
            data.get("schema_version") != "aegisvox.clinical_note_generation_panel_audit.v1"
            or data.get("status") != "complete"
            or data.get("generator") != generator
            or data.get("expected_rows_per_cell") != args.expected_rows
            or data.get("num_cells") != 6
            or data.get("source_evidence_quotes_exposed_to_target_generator") is not False
        ):
            raise ValueError(f"invalid generation audit for {generator}")
        audit_cells = data.get("cells")
        if not isinstance(audit_cells, dict) or set(audit_cells) != {
            "source", "seedvc_only", "stock_voicecraft", "aegisvox_f5",
            "localized_redaction", "full_regeneration",
        }:
            raise ValueError(f"invalid generation-audit cells for {generator}")
        item_count_contracts = {}
        for cell_name, cell in sorted(audit_cells.items()):
            contract = {
                "structured_output": cell.get("structured_output"),
                "max_items_per_section_requested": cell.get("max_items_per_section_requested"),
                "max_items_per_section_enforced": cell.get("max_items_per_section_enforced"),
                "item_count_constraint": cell.get("item_count_constraint"),
                "maximum_items_observed_in_one_section": cell.get("maximum_items_observed_in_one_section"),
                "num_sections_above_requested_max": cell.get("num_sections_above_requested_max"),
            }
            if (
                not isinstance(contract["max_items_per_section_requested"], int)
                or contract["max_items_per_section_requested"] < 1
                or not isinstance(contract["max_items_per_section_enforced"], bool)
                or contract["item_count_constraint"] not in {
                    "json_schema_max_items", "prompt_conciseness_only_no_hard_item_cap",
                }
                or not isinstance(contract["maximum_items_observed_in_one_section"], int)
                or not isinstance(contract["num_sections_above_requested_max"], int)
            ):
                raise ValueError(
                    f"invalid item-count contract for {generator}/{cell_name}"
                )
            if (
                contract["max_items_per_section_enforced"]
                and contract["num_sections_above_requested_max"] != 0
            ):
                raise ValueError(
                    f"enforced item cap violated for {generator}/{cell_name}"
                )
            item_count_contracts[cell_name] = contract
        generation_audits[generator] = {
            "path": str(path),
            "sha256": sha256(path),
            "target_evidence_contract": data.get("target_evidence_contract"),
            "item_count_contracts": item_count_contracts,
            "cells": audit_cells,
        }

    cells = {}
    for raw in args.cell:
        label, raw_path = raw.split("=", 1)
        if label in cells:
            raise ValueError(f"duplicate cell label: {label}")
        parts = label.split("|")
        if len(parts) != 3:
            raise ValueError(f"invalid cell label: {label}")
        condition, generator, judge = parts
        path = Path(raw_path); data = json.loads(path.read_text())
        if data.get("schema_version") != "aegisvox.clinical_note_pair_adjudication.v1":
            raise ValueError(f"unexpected note adjudication schema for {label}")
        if (
            data.get("num_rows") != args.expected_rows
            or data.get("num_completed") != args.expected_rows
            or data.get("num_errors") != 0
        ):
            raise ValueError(f"incomplete note-study cell {label}")
        contract = JUDGE_CONTRACTS.get(judge)
        if contract is None or data.get("judge_model") != contract["model"] or data.get("judge_backend") != contract["backend"]:
            raise ValueError(f"judge contract mismatch for {label}")
        if data.get("full_transcript_used") is not True or data.get("deterministic_proxy_used") is not False:
            raise ValueError(f"invalid evidence contract for {label}")
        if condition not in EXPECTED_CONDITIONS or generator not in EXPECTED_MODELS:
            raise ValueError(f"unexpected condition/generator for {label}")
        generator_audit = generation_audits.get(generator)
        if generator_audit is None:
            raise ValueError(f"missing generation audit for {label}")
        source_cell = generator_audit["cells"]["source"]
        target_cell = generator_audit["cells"][condition]
        expected_hashes = {
            "input_sha256": target_cell.get("paired_transcripts_sha256"),
            "source_caps_sha256": target_cell.get("source_caps_sha256"),
            "source_notes_sha256": source_cell.get("notes_sha256"),
            "target_notes_sha256": target_cell.get("notes_sha256"),
        }
        for key, expected_hash in expected_hashes.items():
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"generation audit lacks {key} linkage for {label}")
            if data.get(key) != expected_hash:
                raise ValueError(f"{key} mismatch for {label}")
        row_evidence = audit_adjudication_rows(path, args.expected_rows)
        repair_contract = schema_repair_contract(data)
        cells[label] = {field: data.get(field) for field in FIELDS}
        cells[label].update(
            {
                "condition": condition,
                "generator": generator,
                "judge": judge,
                "evaluator_affinity": "same_family" if generator == judge else "cross_family",
                "summary_sha256": sha256(path),
                "row_evidence": row_evidence,
                "schema_repair_contract": repair_contract,
            }
        )
    expected_conditions = EXPECTED_CONDITIONS
    expected_models = EXPECTED_MODELS
    if set(generation_audits) != expected_models:
        raise ValueError(
            f"generation audit mismatch: expected={sorted(expected_models)} "
            f"observed={sorted(generation_audits)}"
        )
    observed = {(row["condition"], row["generator"], row["judge"]) for row in cells.values()}
    expected = {(condition, generator, judge) for condition in expected_conditions for generator in expected_models for judge in expected_models}
    exclusions = {}
    excluded_judges = set()
    for raw in args.excluded_judge:
        judge, raw_path = raw.split("=", 1)
        if judge not in expected_models or judge in excluded_judges:
            raise ValueError(f"invalid or duplicate excluded judge: {judge}")
        path = Path(raw_path); data = json.loads(path.read_text())
        if data.get("status") != "excluded_by_frozen_gate" or data.get("judge") != judge:
            raise ValueError(f"invalid frozen exclusion artifact for {judge}")
        excluded_judges.add(judge)
        exclusions[judge] = {
            "status": data["status"],
            "reason": data.get("reason"),
            "artifact_sha256": sha256(path),
        }
    excluded = {
        (condition, generator, judge)
        for condition in expected_conditions
        for generator in expected_models
        for judge in excluded_judges
    }
    if len(expected) != args.expected_cells:
        raise ValueError(f"expected design has {len(expected)} cells, configured {args.expected_cells}")
    if observed & excluded:
        raise ValueError(f"observed cells cannot use an excluded judge: {sorted(observed & excluded)}")
    if observed | excluded != expected:
        missing = sorted(expected - observed - excluded); extra = sorted(observed - expected)
        raise ValueError(f"note-study factorial mismatch: missing={missing} extra={extra}")
    excluded_cells = {
        "|".join(cell): {
            "condition": cell[0], "generator": cell[1], "judge": cell[2],
            "status": "excluded_by_frozen_gate",
            "exclusion": exclusions[cell[2]],
        }
        for cell in sorted(excluded)
    }
    contract_signatures = {
        generator: sorted({
            (
                cell["max_items_per_section_requested"],
                cell["max_items_per_section_enforced"],
                cell["item_count_constraint"],
            )
            for cell in audit["item_count_contracts"].values()
        })
        for generator, audit in generation_audits.items()
    }
    cross_generator_item_count_contract_equivalent = (
        len({json.dumps(value, sort_keys=True) for value in contract_signatures.values()}) == 1
    )
    report = {
        "schema_version": "aegisvox.clinical_note_studies.v1",
        "status": "complete",
        "expected_rows_per_cell": args.expected_rows,
        "num_cells": len(expected),
        "num_observed_cells": len(cells),
        "num_excluded_cells": len(excluded_cells),
        "factorial": {"conditions": sorted(expected_conditions), "generators": sorted(expected_models), "judges": sorted(expected_models)},
        "primary_comparison": {"system": "aegisvox_f5", "baseline": "stock_voicecraft"},
        "primary_estimand_scope": "within_generator_condition_contrasts",
        "generator_item_count_contract_signatures": contract_signatures,
        "cross_generator_item_count_contract_equivalent": cross_generator_item_count_contract_equivalent,
        "direct_cross_generator_note_length_comparison_allowed": cross_generator_item_count_contract_equivalent,
        "cells": cells,
        "excluded_cells": excluded_cells,
        "excluded_judges": exclusions,
        "generation_audits": generation_audits,
        "claim_boundary": "source-CAP-anchored model-generated notes and qualification-gated model-adjudicated invariance; primary system contrasts are within generator; direct cross-generator note-length comparisons are prohibited when item-count contracts differ; excluded judge cells are explicit protocol outcomes, not imputed results; not human gold documentation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "num_cells": len(cells), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
