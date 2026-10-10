#!/usr/bin/env python3
"""Compute PHI relexicalization/residual-audit proxy metrics for AegisVox runs.

This script uses existing de-id, validation, and fallback artifacts. It does not
run a fresh ASR or LLM PHI detector; therefore outputs are protocol-level until
paired with a final-audio ASR pass and gold PHI labels.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any

from common import row_completed, load_json, load_summary_rows, norm_text, run_dir_from_summary_row, write_csv, write_json


def norm_phrase(text: str | None) -> str:
    return norm_text(text or "")


def phrase_present(needle: str | None, haystack: str) -> bool:
    n = norm_phrase(needle)
    h = norm_phrase(haystack)
    return bool(n) and n in h


def max_similarity(needle: str | None, haystack: str) -> float | None:
    n = norm_phrase(needle)
    h_tokens = norm_phrase(haystack).split()
    n_tokens = n.split()
    if not n_tokens or not h_tokens:
        return None
    width = len(n_tokens)
    best = 0.0
    for i in range(0, max(1, len(h_tokens) - width + 1)):
        cand = " ".join(h_tokens[i : i + width])
        best = max(best, SequenceMatcher(None, n, cand).ratio())
    return best


def load_validation_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ["voicecraft_latest_final_validation_fasterwhisper.json", "voicecraft_direct_input_latest_validation_fasterwhisper.json"]:
        data = load_json(run_dir / name, {})
        if isinstance(data, dict):
            for row in data.get("rows") or []:
                if isinstance(row, dict):
                    rows.append({**row, "validation_source": name})
    return rows


def validation_matches(edit: dict[str, Any], validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edit_id = str(edit.get("edit_id") or "")
    original = norm_phrase(edit.get("original_text"))
    replacement = norm_phrase(edit.get("replacement_text"))
    matched = []
    for row in validation_rows:
        source_ids = [str(x) for x in row.get("source_edit_ids") or []]
        if edit_id and edit_id in source_ids:
            matched.append(row)
            continue
        if original and norm_phrase(row.get("original_text")) == original:
            matched.append(row)
            continue
        if replacement and norm_phrase(row.get("replacement_text")) == replacement:
            matched.append(row)
            continue
    return matched


def fallback_edit_ids(run_dir: Path) -> set[str]:
    data = load_json(run_dir / "lexical_fallback_after_voicecraft_validation_latest_report.json", {})
    ids: set[str] = set()
    if not isinstance(data, dict):
        return ids
    for row in data.get("rows") or []:
        for key in ["source_edit_ids", "edit_ids"]:
            for value in row.get(key) or []:
                ids.add(str(value))
        if row.get("edit_id"):
            ids.add(str(row["edit_id"]))
    return ids


def final_validation_text(validation_rows: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for row in validation_rows:
        for key in ["final_asr_text", "edited_span_asr_text"]:
            if isinstance(row.get(key), str):
                chunks.append(row[key])
    return " ".join(chunks)


def load_deid_artifact(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    # Full390 B5 writes canonical de-id as deid_text.json; older direct-input
    # pilot runs used deid_text_hipaa_direct_input_v1.json. Metrics must handle
    # both or the paper-scale run falsely reports zero PHI edits.
    for name in ["deid_text_hipaa_direct_input_v1.json", "deid_text.json"]:
        path = run_dir / name
        data = load_json(path, None)
        if isinstance(data, dict) and isinstance(data.get("edits"), list):
            return path, data
    for path in sorted(run_dir.glob("deid_text*.json")):
        data = load_json(path, None)
        if isinstance(data, dict) and isinstance(data.get("edits"), list):
            return path, data
    return None, {}


def row_metrics(summary_row: dict[str, Any]) -> dict[str, Any]:
    run_dir = run_dir_from_summary_row(summary_row)
    deid_path, deid = load_deid_artifact(run_dir)
    validation_report = load_json(run_dir / "voicecraft_latest_final_validation_fasterwhisper.json", {})
    fallback_report = load_json(run_dir / "lexical_fallback_after_voicecraft_validation_latest_report.json", {})
    validation_rows = load_validation_rows(run_dir)
    validation_text = final_validation_text(validation_rows)
    fallback_ids = fallback_edit_ids(run_dir)
    edits = deid.get("edits") if isinstance(deid, dict) else []
    edit_rows: list[dict[str, Any]] = []
    for edit in edits or []:
        matches = validation_matches(edit, validation_rows)
        statuses = [str(m.get("status") or "") for m in matches]
        source_text = " ".join(str(m.get("final_asr_text") or "") for m in matches) or validation_text
        original_present = phrase_present(edit.get("original_text"), source_text)
        replacement_present = phrase_present(edit.get("replacement_text"), source_text)
        replacement_similarity = max_similarity(edit.get("replacement_text"), source_text)
        original_similarity = max_similarity(edit.get("original_text"), source_text)
        edit_rows.append({
            "edit_id": edit.get("edit_id"),
            "phi_type": edit.get("phi_type"),
            "speaker_id": edit.get("speaker_id"),
            "start_sec": edit.get("start_sec"),
            "end_sec": edit.get("end_sec"),
            "original_text": edit.get("original_text"),
            "replacement_text": edit.get("replacement_text"),
            "validation_statuses": statuses,
            "replacement_present_proxy": replacement_present,
            "original_present_proxy": original_present,
            "max_replacement_similarity_proxy": replacement_similarity,
            "max_original_similarity_proxy": original_similarity,
            "used_fallback": str(edit.get("edit_id")) in fallback_ids,
            "num_validation_matches": len(matches),
        })
    residual_original = sum(1 for row in edit_rows if row["original_present_proxy"])
    replacement_missing = sum(1 for row in edit_rows if not row["replacement_present_proxy"])
    fail_like = sum(1 for row in edit_rows if any(s.startswith("fail") or s.startswith("suspicious") for s in row["validation_statuses"]))
    return {
        "audio_id": summary_row.get("audio_id"),
        "run_id": summary_row.get("run_id"),
        "completed": row_completed(summary_row),
        "deid_path": str(deid_path) if deid_path else None,
        "num_phi_edits": len(edit_rows),
        "phi_types": sorted({str(row.get("phi_type")) for row in edit_rows if row.get("phi_type")}),
        "num_quarantined_detector_rows": len(deid.get("quarantined_detector_rows") or []) if isinstance(deid, dict) else None,
        "num_residual_original_phi_proxy": residual_original,
        "num_replacement_missing_proxy": replacement_missing,
        "num_validation_fail_like_proxy": fail_like,
        "num_fallback_insertions": fallback_report.get("num_fallback_insertions") if isinstance(fallback_report, dict) else None,
        "num_validation_pass_like": validation_report.get("num_pass_like") if isinstance(validation_report, dict) else None,
        "num_validation_fail_like": validation_report.get("num_fail_like") if isinstance(validation_report, dict) else None,
        "validation_status_counts": validation_report.get("status_counts") if isinstance(validation_report, dict) else None,
        "replacement_recall_proxy": (sum(1 for row in edit_rows if row["replacement_present_proxy"]) / len(edit_rows)) if edit_rows else None,
        "residual_original_rate_proxy": (residual_original / len(edit_rows)) if edit_rows else None,
        "edit_rows": edit_rows,
        "claim_boundary": "ASR-validation-artifact proxy only; final paper needs fresh final-audio ASR, phonetic matching, LLM PHI detector, and gold/manual PHI labels.",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_edits = sum(int(row.get("num_phi_edits") or 0) for row in rows)
    total_residual = sum(int(row.get("num_residual_original_phi_proxy") or 0) for row in rows)
    total_missing = sum(int(row.get("num_replacement_missing_proxy") or 0) for row in rows)
    return {
        "num_rows": len(rows),
        "num_completed": sum(1 for row in rows if row.get("completed")),
        "total_phi_edits": total_edits,
        "total_residual_original_phi_proxy": total_residual,
        "total_replacement_missing_proxy": total_missing,
        "total_fallback_insertions": sum(int(row.get("num_fallback_insertions") or 0) for row in rows),
        "total_validation_fail_like": sum(int(row.get("num_validation_fail_like") or 0) for row in rows),
        "replacement_recall_proxy": ((total_edits - total_missing) / total_edits) if total_edits else None,
        "residual_original_rate_proxy": (total_residual / total_edits) if total_edits else None,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# PHI Relexicalization / Residual Proxy Metrics", "", "**Claim boundary:** uses existing validation artifacts only; fresh final-audio ASR + gold PHI labels still required.", "", "## Aggregate", ""]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | edits | residual original proxy | replacement missing proxy | fallback insertions | validation fail-like |", "|---|---:|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        lines.append(f"| {row.get('audio_id')} | {row.get('num_phi_edits')} | {row.get('num_residual_original_phi_proxy')} | {row.get('num_replacement_missing_proxy')} | {row.get('num_fallback_insertions')} | {row.get('num_validation_fail_like')} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    rows = [row_metrics(row) for row in load_summary_rows(args.run_summary)]
    flat_rows = [{k: v for k, v in row.items() if k != "edit_rows"} for row in rows]
    payload = {"schema_version": "aegisvox.eval.phi_relexicalization.v1", "run_summary": str(args.run_summary), "aggregate": aggregate(rows), "rows": rows}
    write_json(args.output_json, payload)
    write_csv(args.output_csv, flat_rows)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
