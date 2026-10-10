#!/usr/bin/env python3
"""Build exact five-condition recording registries for Study-9 outcomes.

Each output row is anchored to one frozen public recording and one condition.
The script supports CAP/role, chronology, and action outcome families.  It does
not convert deterministic text similarity into a clinical label.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable


CONDITIONS = (
    "seedvc_only", "stock_voicecraft", "aegisvox_f5",
    "localized_redaction", "full_regeneration",
)


def module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


CAP = module("frontend_cap_registry", "build_frontend_cap_role_registry.py")
ACTION = module("frontend_action_registry", "build_frontend_action_registry.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        key = str(row.get("audio_id") or "")
        if not key or key in out: raise ValueError(f"{label}: blank/duplicate audio_id={key!r}")
        out[key] = row
    return out


def duration_bin(value: float) -> str:
    if value == 0: return "no_edit"
    if value < 1: return "lt_1s"
    if value < 2: return "1_to_lt_2s"
    if value < 5: return "2_to_lt_5s"
    if value < 10: return "5_to_lt_10s"
    return "ge_10s"


def chronology_fields(item: dict[str, Any]) -> dict[str, Any]:
    result = item.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("adjudications"), list):
        raise ValueError("chronology row lacks result/adjudications")
    assessment = str(result.get("conversation_assessment") or "")
    if assessment not in {"chronology_preserved", "meaningful_chronology_error", "insufficient_evidence", "not_applicable"}:
        raise ValueError(f"invalid chronology assessment {assessment!r}")
    rows = result["adjudications"]
    impacts = [str(row.get("clinical_impact") or "") for row in rows]
    statuses = [str(row.get("status") or "") for row in rows]
    final = [str(row.get("final_state_status") or "") for row in rows]
    return {
        "relation_count": len(rows),
        "any_chronology_not_preserved": int(any(value != "preserved" for value in statuses)),
        "any_major_or_harmful_chronology_error": int(any(value in {"major", "potentially_harmful"} for value in impacts)),
        "any_final_state_error": int(any(value in {"changed", "missing"} for value in final)),
        "conversation_meaningful_chronology_error": int(assessment == "meaningful_chronology_error"),
        "conversation_chronology_uncertain": int(assessment == "insufficient_evidence"),
        "conversation_chronology_not_applicable": int(assessment == "not_applicable"),
    }


FAMILIES: dict[str, tuple[str, Callable[[dict[str, Any]], dict[str, Any]], tuple[str, ...]]] = {
    "cap_role": (
        "cap_target_adjudications.jsonl", CAP.clinical_fields,
        ("any_critical_major_cap_error", "any_role_attribution_error", "any_major_unsupported_target_assertion", "conversation_major_error", "conversation_uncertain"),
    ),
    "chronology": (
        "target_chronology.jsonl", chronology_fields,
        ("any_chronology_not_preserved", "any_major_or_harmful_chronology_error", "any_final_state_error", "conversation_meaningful_chronology_error", "conversation_chronology_uncertain"),
    ),
    "action": (
        "target_actions.jsonl", ACTION.outcomes,
        ("any_action_not_strictly_preserved", "any_action_state_error", "any_action_role_error", "any_unsafe_action_change", "any_potentially_harmful_action_change", "any_unsupported_target_action", "conversation_major_or_harmful_change", "conversation_plan_uncertain"),
    ),
}


def build(strata_path: Path, family_root: Path, family: str, expected_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filename, extractor, endpoints = FAMILIES[family]
    strata = index(read_jsonl(strata_path), "strata")
    if len(strata) != expected_rows: raise ValueError("strata population mismatch")
    output = []; hashes = {}
    for condition in CONDITIONS:
        path = family_root / condition / filename
        if not path.is_file(): raise ValueError(f"{condition}: missing {filename}")
        rows = index(read_jsonl(path), f"{family}/{condition}")
        if set(rows) != set(strata): raise ValueError(f"{condition}: population mismatch")
        for audio_id in sorted(strata):
            s = strata[audio_id]
            fields = extractor(rows[audio_id])
            if any(fields.get(endpoint) not in {0, 1} for endpoint in endpoints):
                raise ValueError(f"{condition}/{audio_id}: incomplete binary endpoint")
            output.append({
                "audio_id": audio_id, "condition": condition, "outcome_family": family,
                "dataset_id": s["dataset_id"], "language": s["language"],
                "duration_sec": s["duration_sec"], "duration_quartile": s["duration_quartile"],
                "speaker_count": s["speaker_count"], "speaker_count_bin": s["speaker_count_bin"],
                "edit_count": s["edit_count"], "edit_count_bin": s["edit_count_bin"],
                "max_edit_duration_sec": s["max_edit_duration_sec"],
                "max_edit_duration_bin": duration_bin(float(s["max_edit_duration_sec"])),
                "phi_type_counts": s["phi_type_counts"], **fields,
            })
        hashes[condition] = sha(path)
    if len(output) != len(CONDITIONS) * expected_rows:
        raise ValueError("registry cardinality mismatch")
    return output, {
        "schema_version": "aegisvox.condition_clinical_equity_registry.v1",
        "status": "complete", "outcome_family": family,
        "num_conditions": len(CONDITIONS), "num_rows_per_condition": expected_rows,
        "num_rows": len(output), "endpoints": list(endpoints),
        "strata_sha256": sha(strata_path), "outcome_sha256_by_condition": hashes,
        "deterministic_clinical_proxy_used": False,
        "claim_boundary": "Qualified GPT-5.4 source-anchored clinical adjudication; not human ground truth.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strata-jsonl", type=Path, required=True)
    ap.add_argument("--family-root", type=Path, required=True)
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    ap.add_argument("--expected-rows", type=int, default=390)
    args = ap.parse_args()
    rows, summary = build(args.strata_jsonl, args.family_root, args.family, args.expected_rows)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["output_sha256"] = sha(args.output_jsonl)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "family": args.family, "num_rows": len(rows)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
