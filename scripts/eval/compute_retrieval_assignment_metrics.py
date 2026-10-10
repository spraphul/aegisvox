#!/usr/bin/env python3
"""Measure pseudo-speaker retrieval and assignment behavior from frozen artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from common import load_summary_rows, mean, percentile, row_completed, write_csv, write_json


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def distribution(values: list[Any], prefix: str) -> dict[str, Any]:
    nums = numeric(values)
    return {
        f"num_{prefix}": len(nums),
        f"min_{prefix}": min(nums) if nums else None,
        f"p05_{prefix}": percentile(nums, 0.05),
        f"mean_{prefix}": mean(nums),
        f"median_{prefix}": percentile(nums, 0.50),
        f"p95_{prefix}": percentile(nums, 0.95),
        f"max_{prefix}": max(nums) if nums else None,
    }


def run_metrics(summary_row: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    run_id = str(summary_row.get("run_id") or "")
    path = artifact_root / "aegisvox" / "runs" / run_id / "speaker_target_map.json"
    base = {
        "audio_id": summary_row.get("audio_id"),
        "run_id": run_id,
        "completed": row_completed(summary_row),
        "assignment_artifact": str(path),
    }
    if not path.is_file():
        return {**base, "status": "missing_assignment_artifact", "assignment_available": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "malformed_assignment_artifact", "assignment_available": False, "error": f"{type(exc).__name__}: {exc}"}
    assignments = payload.get("assignments") if isinstance(payload.get("assignments"), dict) else {}
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    target_pairs = payload.get("target_target_distances") if isinstance(payload.get("target_target_distances"), list) else []
    assigned = [row for row in assignments.values() if isinstance(row, dict) and row.get("status") == "assigned"]
    source_distances = numeric([row.get("source_target_distance") for row in assigned])
    pair_distances = numeric([row.get("target_target_distance") for row in target_pairs if isinstance(row, dict)])
    source_threshold = policy.get("min_source_target_distance")
    target_threshold = policy.get("min_target_target_distance")
    source_failures = sum(float(value) < float(source_threshold) for value in source_distances) if isinstance(source_threshold, (int, float)) else None
    target_failures = sum(float(value) < float(target_threshold) for value in pair_distances) if isinstance(target_threshold, (int, float)) else None
    target_ids = [
        str(row.get("target_voice_sample_id") or row.get("target_voice_id") or row.get("target_reference_audio_uri") or "")
        for row in assigned
    ]
    duplicate_count = len([value for value in target_ids if value]) - len(set(value for value in target_ids if value))
    candidate_ranks = numeric([row.get("selected_candidate_rank") for row in assigned])
    trait_scores = numeric([(row.get("query_trait_match") or {}).get("score") for row in assigned if isinstance(row.get("query_trait_match"), dict)])
    expected_pairs = len(assigned) * (len(assigned) - 1) // 2
    all_constraints_pass = (
        bool(assignments)
        and len(assigned) == len(assignments)
        and len(source_distances) == len(assigned)
        and source_failures == 0
        and len(pair_distances) == expected_pairs
        and target_failures == 0
        and duplicate_count == 0
    )
    match_statuses: dict[str, dict[str, int]] = {}
    for field in ("gender_match_status", "language_match_status", "accent_match_status"):
        match_statuses[field] = dict(Counter(str(row.get(field) or "missing") for row in assigned))
    return {
        **base,
        "status": "ok" if all_constraints_pass else "constraint_failure",
        "assignment_available": True,
        "voice_pool_id": payload.get("voice_pool_id"),
        "assignment_strategy": payload.get("assignment_strategy"),
        "num_source_speakers": len(assignments),
        "num_assigned_speakers": len(assigned),
        "source_target_distances": source_distances,
        "target_target_distances": pair_distances,
        "policy_min_source_target_distance": source_threshold,
        "policy_min_target_target_distance": target_threshold,
        "num_source_target_threshold_failures": source_failures,
        "num_target_target_threshold_failures": target_failures,
        "num_expected_target_target_pairs": expected_pairs,
        "num_observed_target_target_pairs": len(pair_distances),
        "num_duplicate_target_assignments": duplicate_count,
        "duplicate_target_assignment_rate": duplicate_count / len(target_ids) if target_ids else None,
        "all_assignment_constraints_pass": all_constraints_pass,
        "selected_candidate_ranks": candidate_ranks,
        "num_assignments_with_selected_candidate_rank": len(candidate_ranks),
        "candidate_top_k": policy.get("candidate_top_k"),
        "query_trait_scores": trait_scores,
        "match_status_counts": match_statuses,
        "retrieval_contract": payload.get("retrieval_contract") or {},
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_distances = [value for row in rows for value in (row.get("source_target_distances") or [])]
    target_distances = [value for row in rows for value in (row.get("target_target_distances") or [])]
    candidate_ranks = [value for row in rows for value in (row.get("selected_candidate_ranks") or [])]
    trait_scores = [value for row in rows for value in (row.get("query_trait_scores") or [])]
    available = [row for row in rows if row.get("assignment_available")]
    constraints_pass = [row for row in available if row.get("all_assignment_constraints_pass")]
    out = {
        "num_rows": len(rows),
        "num_completed": sum(1 for row in rows if row.get("completed")),
        "num_rows_with_assignment_artifact": len(available),
        "num_rows_all_assignment_constraints_pass": len(constraints_pass),
        "assignment_constraint_pass_rate": len(constraints_pass) / len(available) if available else None,
        "num_assigned_speakers": sum(int(row.get("num_assigned_speakers") or 0) for row in rows),
        "num_source_target_threshold_failures": sum(int(row.get("num_source_target_threshold_failures") or 0) for row in available),
        "num_target_target_threshold_failures": sum(int(row.get("num_target_target_threshold_failures") or 0) for row in available),
        "num_duplicate_target_assignments": sum(int(row.get("num_duplicate_target_assignments") or 0) for row in available),
        "mean_duplicate_target_assignment_rate": mean([row.get("duplicate_target_assignment_rate") for row in available]),
        "num_assignments_with_selected_candidate_rank": len(candidate_ranks),
        "candidate_rank_availability_note": (
            "selected candidate rank was not persisted in this assignment artifact schema"
            if not candidate_ranks else "selected candidate rank available"
        ),
    }
    out.update(distribution(source_distances, "source_target_distance"))
    out.update(distribution(target_distances, "target_target_distance"))
    out.update(distribution(candidate_ranks, "selected_candidate_rank"))
    out.update(distribution(trait_scores, "query_trait_score"))
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Pseudo-speaker Retrieval and Assignment Metrics",
        "",
        "**Protocol:** deterministic assignment artifacts from the frozen multi-embedding/metadata retrieval system.",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | speakers | min source-target | min target-target | duplicates | constraints |", "|---|---:|---:|---:|---:|---|"]
    for row in payload["rows"]:
        src = min(row.get("source_target_distances") or []) if row.get("source_target_distances") else None
        tgt = min(row.get("target_target_distances") or []) if row.get("target_target_distances") else None
        lines.append(f"| {row.get('audio_id')} | {row.get('num_assigned_speakers')} | {src} | {tgt} | {row.get('num_duplicate_target_assignments')} | {row.get('all_assignment_constraints_pass')} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-summary", type=Path, required=True)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()
    rows = [run_metrics(row, args.artifact_root) for row in load_summary_rows(args.run_summary)]
    payload = {
        "schema_version": "aegisvox.eval.retrieval_assignment.v1",
        "run_summary": str(args.run_summary),
        "artifact_root": str(args.artifact_root),
        "aggregate": aggregate(rows),
        "rows": rows,
        "claim_boundary": "Assignment-level retrieval behavior from deterministic metadata/embedding artifacts; not an ASV privacy result.",
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
