#!/usr/bin/env python3
"""Run/consume fresh final-audio ASR and audit PHI/replacement realization.

For every text de-identification edit, this script searches the final-audio ASR
transcript for original PHI and replacement text using normalized exact, fuzzy,
and phonetic matching. It reports global evidence and timestamp-local evidence
around the intended edit window when ASR words/segments have timestamps.

The scanner is deterministic. ``--compute-missing`` is model-backed only for
creating a fresh ASR artifact; matching and scoring remain deterministic.  The
same edit rows also carry factual renderer-route, language, PHI-type, and edit
duration metadata, allowing the final paper tables to report the required
subgroups without relying on the legacy validation-artifact proxy metric.
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
from typing import Any

from common import row_completed, final_audio_path, load_json, load_summary_rows, mean, norm_text, run_dir_from_summary_row, write_csv, write_json

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_DEID_NAMES = [
    "deid_text_hipaa_direct_input_v1.json",
    "deid_text.json",
]
DEFAULT_ASR_NAME = "asr.final_fresh_phi_audit.json"
RECONSTRUCTION_PROVENANCE_NAME = "hash_bound_reconstruction.json"

VOICECRAFT_JOBS_NAMES = [
    "voicecraft_edit_jobs.json",
    "voicecraft_edit_jobs_direct_input_v1_utterance_window_v4.json",
]
FALLBACK_REPORT_NAMES = [
    "lexical_fallback_after_voicecraft_validation_latest_report.json",
]


_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def canonical_phrase(text: str | None) -> str:
    normalized = norm_text(text or "")
    if not normalized:
        return ""
    # Make numeric PHI slightly more robust to ASR spelling single digits.
    tokens = []
    for token in normalized.split():
        if token.isdigit() and len(token) <= 4:
            tokens.append(" ".join(_DIGIT_WORDS.get(ch, ch) for ch in token))
        else:
            tokens.append(token)
    return " ".join(tokens)


def normalize_language(value: Any) -> str:
    raw = str(value or "").strip().casefold().replace("_", "-")
    if not raw:
        return "unknown"
    if raw in {"english", "eng"} or raw.startswith("en"):
        return "en"
    if raw in {"french", "fra", "fre"} or raw.startswith("fr"):
        return "fr"
    return raw.split("-", 1)[0]


def source_dataset_id(summary_row: dict[str, Any]) -> str:
    explicit = summary_row.get("source_dataset_id")
    if explicit:
        return str(explicit)
    identifier = str(summary_row.get("audio_id") or summary_row.get("run_id") or "").lower()
    if "primock57" in identifier:
        return "primock57_mixed"
    if "fareez" in identifier:
        return "fareez_respiratory_interviews"
    if "simsamu" in identifier:
        return "simsamu"
    return "unknown"


def edit_duration_seconds(edit: dict[str, Any]) -> float | None:
    start = edit.get("mask_start_sec", edit.get("start_sec"))
    end = edit.get("mask_end_sec", edit.get("end_sec"))
    try:
        duration = float(end) - float(start)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def duration_bin(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 1.0:
        return "<1s"
    if seconds < 2.0:
        return "1-2s"
    if seconds <= 5.0:
        return "2-5s"
    return ">5s"


def _first_json(run_dir: Path, names: list[str]) -> tuple[Path | None, dict[str, Any]]:
    for name in names:
        path = run_dir / name
        value = load_json(path, None)
        if isinstance(value, dict):
            return path, value
    return None, {}


def _ids_from_row(row: dict[str, Any]) -> set[str]:
    values = [row.get("edit_id")]
    source_ids = row.get("source_edit_ids")
    if isinstance(source_ids, list):
        values.extend(source_ids)
    return {str(value) for value in values if value not in {None, ""}}


def renderer_metadata(run_dir: Path) -> tuple[dict[str, dict[str, Any]], set[str], dict[str, Any]]:
    jobs_path, jobs_payload = _first_json(run_dir, VOICECRAFT_JOBS_NAMES)
    fallback_path, fallback_payload = _first_json(run_dir, FALLBACK_REPORT_NAMES)
    jobs_by_edit: dict[str, dict[str, Any]] = {}
    for job in jobs_payload.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        for edit_id in _ids_from_row(job):
            jobs_by_edit[edit_id] = job
    fallback_ids: set[str] = set()
    for row in fallback_payload.get("rows") or []:
        if isinstance(row, dict):
            fallback_ids.update(_ids_from_row(row))
    return jobs_by_edit, fallback_ids, {
        "voicecraft_jobs_path": str(jobs_path) if jobs_path else None,
        "fallback_report_path": str(fallback_path) if fallback_path else None,
        "voicecraft_jobs_present": bool(jobs_payload),
        "fallback_report_present": bool(fallback_payload),
    }


def renderer_route(edit_id: str, job: dict[str, Any] | None, fallback_ids: set[str]) -> str:
    if edit_id in fallback_ids:
        return "fallback_tts_seedvc"
    if not job:
        return "missing_renderer_job"
    status = str(job.get("status") or "unknown")
    if status == "synthesized":
        validation = job.get("edited_span_validation") if isinstance(job.get("edited_span_validation"), dict) else {}
        if validation:
            return "voicecraft_primary_validated" if validation.get("replacement_found") else "voicecraft_primary_validation_warning"
        return "voicecraft_primary"
    if status.startswith("failed_validation"):
        return "voicecraft_failed_validation_source_preserved"
    if status.startswith("failed"):
        return "voicecraft_failed_source_preserved"
    return f"voicecraft_{status}"


def soundex(token: str) -> str:
    token = re.sub(r"[^a-z]", "", token.lower())
    if not token:
        return ""
    first = token[0].upper()
    mapping = {
        **dict.fromkeys(list("bfpv"), "1"),
        **dict.fromkeys(list("cgjkqsxz"), "2"),
        **dict.fromkeys(list("dt"), "3"),
        "l": "4",
        **dict.fromkeys(list("mn"), "5"),
        "r": "6",
    }
    digits: list[str] = []
    prev = mapping.get(token[0], "")
    for ch in token[1:]:
        code = mapping.get(ch, "")
        if code and code != prev:
            digits.append(code)
        prev = code
    return (first + "".join(digits) + "000")[:4]


def phrase_soundex(phrase: str) -> list[str]:
    return [code for code in (soundex(tok) for tok in canonical_phrase(phrase).split()) if code]


def sliding_windows(tokens: list[str], width: int) -> list[tuple[int, str]]:
    if width <= 0:
        return []
    if not tokens:
        return []
    if len(tokens) < width:
        return [(0, " ".join(tokens))]
    return [(idx, " ".join(tokens[idx : idx + width])) for idx in range(0, len(tokens) - width + 1)]


def match_phrase(
    needle: str | None,
    haystack: str,
    *,
    fuzzy_threshold: float = 0.88,
    phonetic_min_similarity: float = 0.70,
) -> dict[str, Any]:
    phrase = canonical_phrase(needle)
    text = canonical_phrase(haystack)
    if not phrase:
        return {
            "found": False,
            "exact": False,
            "fuzzy": False,
            "phonetic": False,
            "max_similarity": None,
            "best_window": "",
            "matched_by": [],
        }
    exact = bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))
    phrase_tokens = phrase.split()
    text_tokens = text.split()
    best = 0.0
    best_window = ""
    for _, window in sliding_windows(text_tokens, len(phrase_tokens)):
        score = SequenceMatcher(None, phrase, window).ratio()
        if score > best:
            best = score
            best_window = window
    fuzzy = best >= fuzzy_threshold

    target_codes = phrase_soundex(phrase)
    phonetic = False
    phonetic_window = ""
    if target_codes and text_tokens:
        text_codes = [soundex(tok) for tok in text_tokens]
        width = len(target_codes)
        for idx in range(0, max(1, len(text_codes) - width + 1)):
            if text_codes[idx : idx + width] == target_codes:
                candidate = " ".join(text_tokens[idx : idx + width])
                # Soundex is useful for ASR spelling variants (e.g. names), but
                # it is too permissive for short/common clinical terms: "today"
                # can collide with unrelated words such as "that". Require a
                # minimum textual similarity before accepting phonetic-only
                # evidence so the audit does not inflate false residual-PHI
                # failures.
                similarity = SequenceMatcher(None, phrase, candidate).ratio()
                if similarity >= phonetic_min_similarity:
                    phonetic = True
                    phonetic_window = candidate
                    break
    matched_by = []
    if exact:
        matched_by.append("exact")
    if fuzzy and not exact:
        matched_by.append("fuzzy")
    if phonetic and not exact:
        matched_by.append("phonetic")
    return {
        "found": bool(exact or fuzzy or phonetic),
        "exact": exact,
        "fuzzy": fuzzy,
        "phonetic": phonetic,
        "phonetic_window": phonetic_window,
        "phonetic_min_similarity": phonetic_min_similarity,
        "max_similarity": best,
        "best_window": best_window,
        "matched_by": matched_by,
    }


def words_text(words: list[dict[str, Any]], *, start_sec: float | None = None, end_sec: float | None = None) -> str:
    chunks: list[str] = []
    for row in words:
        word_start = row.get("start_sec", row.get("start"))
        word_end = row.get("end_sec", row.get("end"))
        if start_sec is not None or end_sec is not None:
            midpoint = None
            if word_start is not None and word_end is not None:
                midpoint = (float(word_start) + float(word_end)) / 2.0
            elif word_start is not None:
                midpoint = float(word_start)
            if midpoint is None:
                continue
            if start_sec is not None and midpoint < start_sec:
                continue
            if end_sec is not None and midpoint > end_sec:
                continue
        value = str(row.get("word") or row.get("text") or "").strip()
        if value:
            chunks.append(value)
    return " ".join(chunks)


def segments_text(segments: list[dict[str, Any]], *, start_sec: float | None = None, end_sec: float | None = None) -> str:
    chunks: list[str] = []
    for row in segments:
        seg_start = row.get("start_sec", row.get("start"))
        seg_end = row.get("end_sec", row.get("end"))
        if start_sec is not None or end_sec is not None:
            if seg_start is None or seg_end is None:
                continue
            if float(seg_end) < float(start_sec if start_sec is not None else -1e9):
                continue
            if float(seg_start) > float(end_sec if end_sec is not None else 1e9):
                continue
        value = str(row.get("text") or "").strip()
        if value:
            chunks.append(value)
    return " ".join(chunks)


def local_asr_text(asr: dict[str, Any], start_sec: Any, end_sec: Any, pad_sec: float) -> tuple[str, dict[str, Any]]:
    if start_sec is None or end_sec is None:
        return "", {"available": False, "reason": "edit_missing_timestamp"}
    start = max(0.0, float(start_sec) - pad_sec)
    end = float(end_sec) + pad_sec
    words = asr.get("words") if isinstance(asr.get("words"), list) else []
    text = words_text(words, start_sec=start, end_sec=end)
    source = "words"
    if not text:
        segments = asr.get("segments") if isinstance(asr.get("segments"), list) else []
        text = segments_text(segments, start_sec=start, end_sec=end)
        source = "segments" if text else "none"
    return text, {"available": bool(text), "start_sec": round(start, 3), "end_sec": round(end, 3), "source": source}


def load_deid(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    for name in DEFAULT_DEID_NAMES:
        path = run_dir / name
        data = load_json(path, None)
        if isinstance(data, dict):
            return path, data
    candidates = sorted(run_dir.glob("deid_text*.json"))
    for path in candidates:
        data = load_json(path, None)
        if isinstance(data, dict) and isinstance(data.get("edits"), list):
            return path, data
    return None, {}


def hash_bound_edit_windows(run_dir: Path) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
    """Map each planned edit to its actual generated-turn interval in final audio."""
    path = run_dir / RECONSTRUCTION_PROVENANCE_NAME
    payload = load_json(path, None)
    if not isinstance(payload, dict):
        return {}, {"path": str(path), "present": False, "policy": None}
    windows: dict[str, tuple[float, float]] = {}
    regions = payload.get("edit_timeline_regions") or payload.get("insertions") or []
    for insertion in regions:
        if not isinstance(insertion, dict):
            continue
        start = insertion.get("output_turn_start_sec")
        end = insertion.get("output_turn_end_sec")
        if start is None or end is None or float(end) <= float(start):
            continue
        for edit_id in insertion.get("grouped_edit_ids") or []:
            key = str(edit_id)
            window = (float(start), float(end))
            if key in windows and windows[key] != window:
                raise ValueError(f"conflicting hash-bound output windows for {key} in {path}")
            windows[key] = window
    return windows, {
        "path": str(path),
        "present": True,
        "policy": payload.get("reconstruction_policy"),
        "post_edit_validation_applied": payload.get("post_edit_validation_applied"),
        "num_mapped_edits": len(windows),
    }


def compute_fresh_asr(run_dir: Path, asr_path: Path, backend_kind: str | None) -> dict[str, Any]:
    from aegisvox.asr_alignment import create_asr_backend, normalize_asr_output

    audio = final_audio_path(run_dir)
    backend = create_asr_backend(backend_kind)
    raw = backend.transcribe(audio)
    normalized = normalize_asr_output(raw, [])
    payload = {
        "audio_path": str(audio),
        "backend": getattr(backend, "backend_name", None),
        "backend_version": getattr(backend, "backend_version", None),
        "fresh_asr_for": "phi_residual_audit",
        **normalized,
    }
    asr_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return payload


def load_or_compute_asr(run_dir: Path, *, asr_name: str, compute_missing: bool, backend_kind: str | None) -> tuple[Path, dict[str, Any], bool]:
    asr_path = run_dir / asr_name
    data = load_json(asr_path, None)
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        return asr_path, data, False
    if not compute_missing:
        return asr_path, {}, False
    payload = compute_fresh_asr(run_dir, asr_path, backend_kind)
    return asr_path, payload, True


def status_for(global_original: dict[str, Any], local_original: dict[str, Any], global_replacement: dict[str, Any], local_replacement: dict[str, Any]) -> str:
    original_any = bool(global_original.get("found") or local_original.get("found"))
    replacement_any = bool(global_replacement.get("found") or local_replacement.get("found"))
    # Local evidence around the intended edit window is the strongest evidence.
    # Global original matches are retained as warnings because common date words
    # such as "today" can occur elsewhere in a long consultation.
    if local_original.get("found") and replacement_any:
        return "fail_mixed_original_and_replacement_local"
    if local_original.get("found"):
        return "fail_residual_phi_local"
    if global_original.get("found") and replacement_any:
        return "warn_mixed_original_and_replacement_global"
    if global_original.get("found"):
        return "warn_residual_phi_global"
    if local_replacement.get("found"):
        return "pass_replacement_localized"
    if global_replacement.get("found"):
        return "warn_global_only_replacement"
    if not original_any and not replacement_any:
        return "warn_replacement_missing"
    return "needs_manual_review"


def edit_metric_rows(
    summary_row: dict[str, Any],
    *,
    asr_name: str,
    compute_missing: bool,
    backend_kind: str | None,
    local_pad_sec: float,
    fuzzy_threshold: float,
    phonetic_min_similarity: float = 0.70,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = run_dir_from_summary_row(summary_row)
    deid_path, deid = load_deid(run_dir)
    asr_path, asr, asr_computed = load_or_compute_asr(run_dir, asr_name=asr_name, compute_missing=compute_missing, backend_kind=backend_kind)
    edits = deid.get("edits") if isinstance(deid, dict) else []
    if not isinstance(edits, list):
        edits = []
    jobs_by_edit, fallback_ids, renderer_artifacts = renderer_metadata(run_dir)
    stitched_windows, reconstruction_artifact = hash_bound_edit_windows(run_dir)
    dataset_id = source_dataset_id(summary_row)
    global_text = str(asr.get("text") or "") if isinstance(asr, dict) else ""
    rows: list[dict[str, Any]] = []
    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict):
            continue
        edit_id = str(edit.get("edit_id") or f"edit_{idx:04d}")
        duration = edit_duration_seconds(edit)
        source_language = normalize_language(edit.get("source_language"))
        replacement_language = normalize_language(edit.get("replacement_language") or edit.get("source_language"))
        job = jobs_by_edit.get(edit_id)
        mapped_window = stitched_windows.get(edit_id)
        local_start, local_end = mapped_window if mapped_window is not None else (edit.get("start_sec"), edit.get("end_sec"))
        local_text, window = local_asr_text(asr, local_start, local_end, local_pad_sec) if asr else ("", {"available": False, "reason": "missing_asr"})
        window["timeline"] = "hash_bound_final_output_turn" if mapped_window is not None else "source_edit_fallback"
        global_original = match_phrase(edit.get("original_text"), global_text, fuzzy_threshold=fuzzy_threshold, phonetic_min_similarity=phonetic_min_similarity)
        local_original = match_phrase(edit.get("original_text"), local_text, fuzzy_threshold=fuzzy_threshold, phonetic_min_similarity=phonetic_min_similarity)
        global_replacement = match_phrase(edit.get("replacement_text"), global_text, fuzzy_threshold=fuzzy_threshold, phonetic_min_similarity=phonetic_min_similarity)
        local_replacement = match_phrase(edit.get("replacement_text"), local_text, fuzzy_threshold=fuzzy_threshold, phonetic_min_similarity=phonetic_min_similarity)
        status = status_for(global_original, local_original, global_replacement, local_replacement)
        rows.append(
            {
                "audio_id": summary_row.get("audio_id"),
                "run_id": summary_row.get("run_id"),
                "source_dataset_id": dataset_id,
                "edit_index": idx,
                "edit_id": edit_id,
                "phi_type": edit.get("phi_type"),
                "speaker_id": edit.get("speaker_id"),
                "start_sec": edit.get("start_sec"),
                "end_sec": edit.get("end_sec"),
                "edit_duration_sec": duration,
                "edit_duration_bin": duration_bin(duration),
                "source_language": source_language,
                "replacement_language": replacement_language,
                "replacement_language_matches_source": (
                    None
                    if source_language == "unknown" or replacement_language == "unknown"
                    else source_language == replacement_language
                ),
                "renderer_route": renderer_route(edit_id, job, fallback_ids),
                "renderer_job_status": job.get("status") if job else None,
                "original_text": edit.get("original_text"),
                "replacement_text": edit.get("replacement_text"),
                "global_original_found": global_original["found"],
                "local_original_found": local_original["found"],
                "global_replacement_found": global_replacement["found"],
                "local_replacement_found": local_replacement["found"],
                "global_original_match": global_original,
                "local_original_match": local_original,
                "global_replacement_match": global_replacement,
                "local_replacement_match": local_replacement,
                "local_window": window,
                "hash_bound_output_window_used": mapped_window is not None,
                "local_asr_text": local_text[:500],
                "status": status,
            }
        )
    row_summary = {
        "audio_id": summary_row.get("audio_id"),
        "run_id": summary_row.get("run_id"),
        "source_dataset_id": dataset_id,
        "completed": row_completed(summary_row),
        "run_dir": str(run_dir),
        "deid_path": str(deid_path) if deid_path else None,
        "fresh_asr_path": str(asr_path),
        "fresh_asr_computed": asr_computed,
        "fresh_asr_present": bool(global_text),
        "fresh_asr_word_count": len((asr.get("words") if isinstance(asr, dict) else []) or []),
        "fresh_asr_segment_count": len((asr.get("segments") if isinstance(asr, dict) else []) or []),
        **renderer_artifacts,
        "reconstruction_artifact": reconstruction_artifact,
        "num_phi_edits": len(rows),
        "num_fail_residual_phi_local": sum(1 for row in rows if row["status"] == "fail_residual_phi_local"),
        "num_fail_residual_phi_global": sum(1 for row in rows if row["status"] == "fail_residual_phi_global"),
        "num_warn_residual_phi_global": sum(1 for row in rows if row["status"] == "warn_residual_phi_global"),
        "num_fail_mixed_original_and_replacement_local": sum(1 for row in rows if row["status"] == "fail_mixed_original_and_replacement_local"),
        "num_warn_mixed_original_and_replacement_global": sum(1 for row in rows if row["status"] == "warn_mixed_original_and_replacement_global"),
        "num_pass_replacement_localized": sum(1 for row in rows if row["status"] == "pass_replacement_localized"),
        "num_warn_global_only_replacement": sum(1 for row in rows if row["status"] == "warn_global_only_replacement"),
        "num_warn_replacement_missing": sum(1 for row in rows if row["status"] == "warn_replacement_missing"),
        "num_needs_manual_review": sum(1 for row in rows if row["status"] == "needs_manual_review"),
        "residual_phi_local_rate": (sum(1 for row in rows if row["local_original_found"]) / len(rows)) if rows else None,
        "residual_phi_global_rate": (sum(1 for row in rows if row["global_original_found"]) / len(rows)) if rows else None,
        "replacement_local_recall": (sum(1 for row in rows if row["local_replacement_found"]) / len(rows)) if rows else None,
        "replacement_global_recall": (sum(1 for row in rows if row["global_replacement_found"]) / len(rows)) if rows else None,
        "renderer_route_counts": dict(Counter(str(row["renderer_route"]) for row in rows)),
        "duration_bin_counts": dict(Counter(str(row["edit_duration_bin"]) for row in rows)),
        "source_language_counts": dict(Counter(str(row["source_language"]) for row in rows)),
        "num_known_replacement_languages": sum(1 for row in rows if row["replacement_language_matches_source"] is not None),
        "replacement_language_match_rate_known": (
            sum(1 for row in rows if row["replacement_language_matches_source"] is True)
            / sum(1 for row in rows if row["replacement_language_matches_source"] is not None)
        ) if any(row["replacement_language_matches_source"] is not None for row in rows) else None,
        "claim_boundary": "Automatic fresh-final-ASR audit with factual renderer metadata. Residual/replacement warnings require manual adjudication for ASR errors and common-term ambiguity.",
    }
    return row_summary, rows


def _subgroup_metrics(edit_rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in edit_rows:
        groups.setdefault(str(row.get(field) or "unknown"), []).append(row)
    output: dict[str, dict[str, Any]] = {}
    for key, subset in sorted(groups.items()):
        count = len(subset)
        local_residual = sum(1 for row in subset if row.get("local_original_found"))
        global_residual = sum(1 for row in subset if row.get("global_original_found"))
        local_replacement = sum(1 for row in subset if row.get("local_replacement_found"))
        any_replacement = sum(1 for row in subset if row.get("local_replacement_found") or row.get("global_replacement_found"))
        output[key] = {
            "num_edits": count,
            "local_residual_original_rate": local_residual / count if count else None,
            "global_residual_original_rate": global_residual / count if count else None,
            "local_replacement_evidence_rate": local_replacement / count if count else None,
            "global_or_local_replacement_evidence_rate": any_replacement / count if count else None,
        }
    return output


def aggregate(rows: list[dict[str, Any]], edit_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    edit_rows = list(edit_rows or [])
    total_edits = sum(int(row.get("num_phi_edits") or 0) for row in rows)
    local_resid = sum(int(row.get("num_fail_residual_phi_local") or 0) + int(row.get("num_fail_mixed_original_and_replacement_local") or 0) for row in rows)
    global_resid = sum(
        int(row.get("num_fail_residual_phi_local") or 0)
        + int(row.get("num_fail_mixed_original_and_replacement_local") or 0)
        + int(row.get("num_warn_residual_phi_global") or 0)
        + int(row.get("num_warn_mixed_original_and_replacement_global") or 0)
        for row in rows
    )
    local_repl = sum(int(row.get("num_pass_replacement_localized") or 0) for row in rows)
    global_repl_any = sum(
        int(row.get("num_pass_replacement_localized") or 0)
        + int(row.get("num_warn_global_only_replacement") or 0)
        + int(row.get("num_fail_mixed_original_and_replacement_local") or 0)
        + int(row.get("num_warn_mixed_original_and_replacement_global") or 0)
        for row in rows
    )
    audio_with_local_residual = sum(
        1 for row in rows
        if int(row.get("num_fail_residual_phi_local") or 0)
        + int(row.get("num_fail_mixed_original_and_replacement_local") or 0) > 0
    )
    renderer_counts = Counter(str(row.get("renderer_route") or "unknown") for row in edit_rows)
    return {
        "num_rows": len(rows),
        "num_completed": sum(1 for row in rows if row.get("completed")),
        "num_rows_with_fresh_asr": sum(1 for row in rows if row.get("fresh_asr_present")),
        "num_rows_asr_computed": sum(1 for row in rows if row.get("fresh_asr_computed")),
        "total_phi_edits": total_edits,
        "total_local_residual_phi_failures": local_resid,
        "total_global_residual_phi_failures": global_resid,
        "total_local_mixed_original_and_replacement_failures": sum(int(row.get("num_fail_mixed_original_and_replacement_local") or 0) for row in rows),
        "total_global_mixed_original_and_replacement_warnings": sum(int(row.get("num_warn_mixed_original_and_replacement_global") or 0) for row in rows),
        "total_localized_replacement_passes": local_repl,
        "total_global_or_local_replacement_evidence": global_repl_any,
        "total_replacement_missing_warnings": sum(int(row.get("num_warn_replacement_missing") or 0) for row in rows),
        "num_audio_with_local_residual_risk": audio_with_local_residual,
        "audio_local_residual_risk_rate": (audio_with_local_residual / len(rows)) if rows else None,
        "residual_phi_local_rate": (local_resid / total_edits) if total_edits else None,
        "residual_phi_global_rate": (global_resid / total_edits) if total_edits else None,
        "replacement_local_recall": (local_repl / total_edits) if total_edits else None,
        "replacement_global_or_local_recall": (global_repl_any / total_edits) if total_edits else None,
        "mean_row_replacement_local_recall": mean([row.get("replacement_local_recall") for row in rows]),
        "renderer_route_counts": dict(renderer_counts),
        "renderer_route_percentages": {
            key: count / len(edit_rows) for key, count in sorted(renderer_counts.items())
        } if edit_rows else {},
        "num_known_replacement_languages": sum(1 for row in edit_rows if row.get("replacement_language_matches_source") is not None),
        "replacement_language_match_rate_known": (
            sum(1 for row in edit_rows if row.get("replacement_language_matches_source") is True)
            / sum(1 for row in edit_rows if row.get("replacement_language_matches_source") is not None)
        ) if any(row.get("replacement_language_matches_source") is not None for row in edit_rows) else None,
        "by_phi_type": _subgroup_metrics(edit_rows, "phi_type"),
        "by_renderer_route": _subgroup_metrics(edit_rows, "renderer_route"),
        "by_edit_duration_bin": _subgroup_metrics(edit_rows, "edit_duration_bin"),
        "by_source_dataset": _subgroup_metrics(edit_rows, "source_dataset_id"),
        "by_source_language": _subgroup_metrics(edit_rows, "source_language"),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fresh Final-ASR PHI Residual Audit",
        "",
        "**Claim boundary:** automatic scanner over fresh final-audio ASR. Residual/replacement warnings are manually adjudicated for ASR errors and common-term ambiguity before paper reporting.",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | edits | local residual fail | global residual warn | local mixed fail | global mixed warn | localized replacement pass | missing replacement |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        lines.append(
            f"| {row.get('audio_id')} | {row.get('num_phi_edits')} | {row.get('num_fail_residual_phi_local')} | {row.get('num_warn_residual_phi_global')} | {row.get('num_fail_mixed_original_and_replacement_local')} | {row.get('num_warn_mixed_original_and_replacement_global')} | {row.get('num_pass_replacement_localized')} | {row.get('num_warn_replacement_missing')} |"
        )
    lines += [
        "",
        "## Required paper subgroups",
        "",
        f"- Renderer route: `{payload['aggregate'].get('by_renderer_route')}`",
        f"- Edit duration: `{payload['aggregate'].get('by_edit_duration_bin')}`",
        f"- PHI/PII type: `{payload['aggregate'].get('by_phi_type')}`",
        f"- Source dataset: `{payload['aggregate'].get('by_source_dataset')}`",
        f"- Source language: `{payload['aggregate'].get('by_source_language')}`",
        "",
        "## Status definitions",
        "",
        "- `pass_replacement_localized`: replacement found in the ASR window around the intended edit and no original PHI found.",
        "- `fail_residual_phi_local`: original PHI found in the intended local ASR window.",
        "- `fail_mixed_original_and_replacement_local`: both original and replacement were found with local original evidence; high-risk fail.",
        "- `warn_residual_phi_global`: original PHI found somewhere in final ASR but not localized; warning because long-audio common terms can recur.",
        "- `warn_mixed_original_and_replacement_global`: original found only globally while replacement evidence exists; manual-review warning.",
        "- `warn_global_only_replacement`: replacement found globally but not in the local window.",
        "- `warn_replacement_missing`: neither original nor replacement found; utility/edit-realization warning.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--edit-csv", type=Path, required=True)
    parser.add_argument("--asr-name", default=DEFAULT_ASR_NAME)
    parser.add_argument("--compute-missing", action="store_true")
    parser.add_argument("--asr-backend", default=None, help="Optional backend override, e.g. local_faster_whisper")
    parser.add_argument("--local-pad-sec", type=float, default=3.0)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.88)
    parser.add_argument("--phonetic-min-similarity", type=float, default=0.70)
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of row shards for parallel PHI-ASR audit.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to process.")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")
    rows: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    all_summary_rows = load_summary_rows(args.run_summary)
    selected_summary_rows = [row for idx, row in enumerate(all_summary_rows) if idx % args.num_shards == args.shard_index]
    for summary_row in selected_summary_rows:
        row_summary, edit_rows = edit_metric_rows(
            summary_row,
            asr_name=args.asr_name,
            compute_missing=args.compute_missing,
            backend_kind=args.asr_backend,
            local_pad_sec=args.local_pad_sec,
            fuzzy_threshold=args.fuzzy_threshold,
            phonetic_min_similarity=args.phonetic_min_similarity,
        )
        rows.append(row_summary)
        edits.extend(edit_rows)
    payload = {
        "schema_version": "aegisvox.eval.phi_residual_asr.v2",
        "run_summary": str(args.run_summary),
        "asr_name": args.asr_name,
        "local_pad_sec": args.local_pad_sec,
        "fuzzy_threshold": args.fuzzy_threshold,
        "phonetic_min_similarity": args.phonetic_min_similarity,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_total_summary_rows": len(all_summary_rows),
        "num_selected_summary_rows": len(selected_summary_rows),
        "aggregate": aggregate(rows, edits),
        "rows": rows,
        "edit_rows": edits,
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    write_csv(args.edit_csv, edits)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
