#!/usr/bin/env python3
"""Compute official source DER/JER and before/after speaker-structure metrics."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from common import audio_id_for_summary_row, config_input_for_summary_row, config_inputs, frame_activity, frame_speaker_sets, input_audio_path, load_json, load_summary_rows, mean, parse_rttm, predicted_diarization_segments, row_completed, run_dir_from_summary_row, speech_duration, wav_info, write_csv, write_json


def artifact_segments(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path, {})
    rows = payload.get("segments") if isinstance(payload, dict) else []
    out = []
    for row in rows or []:
        try:
            out.append({"start_sec": float(row["start_sec"]), "end_sec": float(row["end_sec"]), "speaker_id": str(row.get("speaker_id") or "")})
        except Exception:
            continue
    return out


def official_der_jer(ref: list[dict[str, Any]], pred: list[dict[str, Any]], *, collar: float, skip_overlap: bool) -> dict[str, Any]:
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    except ImportError as exc:
        raise RuntimeError("official pyannote.metrics backend is unavailable") from exc
    reference = Annotation(uri="reference")
    hypothesis = Annotation(uri="hypothesis")
    for idx, row in enumerate(ref):
        reference[Segment(float(row["start_sec"]), float(row["end_sec"])), idx] = str(row["speaker_id"])
    for idx, row in enumerate(pred):
        hypothesis[Segment(float(row["start_sec"]), float(row["end_sec"])), idx] = str(row["speaker_id"])
    der = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    jer = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
    detail = der(reference, hypothesis, detailed=True)
    total = float(detail.get("total", 0.0))
    error = float(detail.get("confusion", 0.0)) + float(detail.get("false alarm", 0.0)) + float(detail.get("missed detection", 0.0))
    return {
        "official_der": error / total if total > 0 else None,
        "official_jer": float(jer(reference, hypothesis)),
        "official_der_confusion": float(detail.get("confusion", 0.0)),
        "official_der_false_alarm": float(detail.get("false alarm", 0.0)),
        "official_der_missed_detection": float(detail.get("missed detection", 0.0)),
        "official_der_total": float(detail.get("total", 0.0)),
        "official_metric_backend": "pyannote.metrics",
        "official_metric_collar_sec": collar,
        "official_metric_skip_overlap": skip_overlap,
    }


def best_speaker_mapping(ref_frames: list[set[str]], pred_frames: list[set[str]]) -> dict[str, str]:
    ref_speakers = sorted({s for frame in ref_frames for s in frame})
    pred_speakers = sorted({s for frame in pred_frames for s in frame})
    if not ref_speakers or not pred_speakers:
        return {}
    # Most public/clinical conversations in current eval are small speaker-count;
    # brute force avoids scipy dependency and is deterministic.
    best_score = -1
    best_map: dict[str, str] = {}
    if len(pred_speakers) <= len(ref_speakers) and len(pred_speakers) <= 8:
        for refs in itertools.permutations(ref_speakers, len(pred_speakers)):
            mapping = dict(zip(pred_speakers, refs))
            score = speaker_overlap_score(ref_frames, pred_frames, mapping)
            if score > best_score:
                best_score = score
                best_map = mapping
        return best_map
    # Greedy fallback when over-speaker prediction makes brute force too costly.
    remaining_refs = set(ref_speakers)
    mapping: dict[str, str] = {}
    overlaps: list[tuple[int, str, str]] = []
    for pred in pred_speakers:
        for ref in ref_speakers:
            overlaps.append((sum(1 for rf, pf in zip(ref_frames, pred_frames) if ref in rf and pred in pf), pred, ref))
    for _, pred, ref in sorted(overlaps, reverse=True):
        if pred not in mapping and ref in remaining_refs:
            mapping[pred] = ref
            remaining_refs.remove(ref)
    return mapping


def speaker_overlap_score(ref_frames: list[set[str]], pred_frames: list[set[str]], mapping: dict[str, str]) -> int:
    score = 0
    for ref_set, pred_set in zip(ref_frames, pred_frames):
        mapped_pred = {mapping[p] for p in pred_set if p in mapping}
        score += len(ref_set & mapped_pred)
    return score


def approximate_der(ref: list[dict[str, Any]], pred: list[dict[str, Any]], duration: float, frame_sec: float) -> dict[str, Any]:
    ref_frames = frame_speaker_sets(ref, duration, frame_sec)
    pred_frames = frame_speaker_sets(pred, duration, frame_sec)
    mapping = best_speaker_mapping(ref_frames, pred_frames)
    ref_total = sum(len(frame) for frame in ref_frames)
    correct = speaker_overlap_score(ref_frames, pred_frames, mapping)
    pred_total = sum(len(frame) for frame in pred_frames)
    missed = max(0, ref_total - correct)
    false_alarm = max(0, pred_total - correct)
    der = (missed + false_alarm) / ref_total if ref_total else None
    return {
        "approx_frame_der": der,
        "approx_frame_miss_rate": missed / ref_total if ref_total else None,
        "approx_frame_false_alarm_rate": false_alarm / ref_total if ref_total else None,
        "approx_frame_correct_speaker_time_sec": correct * frame_sec,
        "approx_frame_ref_speaker_time_sec": ref_total * frame_sec,
        "speaker_mapping_pred_to_ref": mapping,
    }


def approximate_jer(ref: list[dict[str, Any]], pred: list[dict[str, Any]], duration: float, frame_sec: float) -> float | None:
    ref_frames = frame_speaker_sets(ref, duration, frame_sec)
    pred_frames = frame_speaker_sets(pred, duration, frame_sec)
    mapping = best_speaker_mapping(ref_frames, pred_frames)
    ref_speakers = sorted({s for frame in ref_frames for s in frame})
    if not ref_speakers:
        return None
    errors = []
    for ref_spk in ref_speakers:
        ref_active = {i for i, frame in enumerate(ref_frames) if ref_spk in frame}
        mapped_preds = [pred for pred, ref_name in mapping.items() if ref_name == ref_spk]
        pred_active = {i for i, frame in enumerate(pred_frames) if any(pred in frame for pred in mapped_preds)}
        union = len(ref_active | pred_active)
        jaccard = len(ref_active & pred_active) / union if union else 1.0
        errors.append(1.0 - jaccard)
    return sum(errors) / len(errors) if errors else None


def activity_metrics(ref: list[dict[str, Any]], pred: list[dict[str, Any]], duration: float, frame_sec: float) -> dict[str, Any]:
    ref_active = frame_activity(ref, duration, frame_sec)
    pred_active = frame_activity(pred, duration, frame_sec)
    tp = len(ref_active & pred_active)
    fp = len(pred_active - ref_active)
    fn = len(ref_active - pred_active)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) else None
    return {"speech_activity_precision": precision, "speech_activity_recall": recall, "speech_activity_f1": f1}


def row_metrics(row: dict[str, Any], config_by_audio: dict[str, dict[str, Any]], frame_sec: float, *, collar: float, skip_overlap: bool, require_official: bool) -> dict[str, Any]:
    audio_id = audio_id_for_summary_row(row, config_by_audio)
    run_dir = run_dir_from_summary_row(row)
    metadata = (config_input_for_summary_row(row, config_by_audio).get("metadata") or {})
    ref = parse_rttm(Path(metadata["diarization_path"]) if metadata.get("diarization_path") else None)
    pred = predicted_diarization_segments(run_dir)
    final_pred = artifact_segments(run_dir / "diarization.final.json")
    input_duration = wav_info(input_audio_path(run_dir))["duration_sec"]
    duration = input_duration or max([0.0] + [float(seg["end_sec"]) for seg in ref + pred])
    ref_speakers = {seg.get("speaker_id") for seg in ref if seg.get("speaker_id")}
    pred_speakers = {seg.get("speaker_id") for seg in pred if seg.get("speaker_id")}
    final_speakers = {seg.get("speaker_id") for seg in final_pred if seg.get("speaker_id")}
    out = {
        "audio_id": audio_id,
        "run_id": row.get("run_id"),
        "source_dataset_id": metadata.get("source_dataset_id"),
        "completed": row_completed(row),
        "reference_rttm_path": metadata.get("diarization_path"),
        "has_reference_rttm": bool(ref),
        "reference_speaker_count": len(ref_speakers) if ref else None,
        "predicted_speaker_count": len(pred_speakers) if pred else None,
        "source_detected_speaker_count": len(pred_speakers) if pred else None,
        "final_detected_speaker_count": len(final_speakers) if final_pred else None,
        "source_final_speaker_count_abs_diff": abs(len(pred_speakers) - len(final_speakers)) if pred and final_pred else None,
        "source_final_speaker_count_exact_match": len(pred_speakers) == len(final_speakers) if pred and final_pred else None,
        "source_turn_count": len(pred),
        "final_turn_count": len(final_pred) if final_pred else None,
        "source_final_turn_count_ratio": len(final_pred) / len(pred) if pred and final_pred else None,
        "final_speech_sec": speech_duration(final_pred) if final_pred else None,
        "has_final_diarization": bool(final_pred),
        "speaker_count_abs_diff": abs(len(ref_speakers) - len(pred_speakers)) if ref and pred else None,
        "reference_speech_sec": speech_duration(ref) if ref else None,
        "predicted_speech_sec": speech_duration(pred) if pred else None,
        "frame_sec": frame_sec,
        "claim_boundary": "Official source-audio DER/JER where reference RTTM exists; source-versus-final counts are same-backend structural measurements.",
    }
    if ref and pred:
        out.update(activity_metrics(ref, pred, duration, frame_sec))
        try:
            out.update(official_der_jer(ref, pred, collar=collar, skip_overlap=skip_overlap))
        except Exception:
            if require_official:
                raise
            out.update({"official_der": None, "official_jer": None, "official_metric_backend": None})
    else:
        out.update({
            "speech_activity_precision": None,
            "speech_activity_recall": None,
            "speech_activity_f1": None,
            "official_der": None,
            "official_jer": None,
            "official_metric_backend": None,
            "speaker_mapping_pred_to_ref": {},
        })
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_with_reference_rttm": sum(1 for row in rows if row.get("has_reference_rttm")),
        "num_with_official_der_jer": sum(1 for row in rows if row.get("official_metric_backend") == "pyannote.metrics"),
        "num_with_final_diarization": sum(1 for row in rows if row.get("has_final_diarization")),
        "mean_speech_activity_f1": mean([row.get("speech_activity_f1") for row in rows]),
        "mean_official_der": mean([row.get("official_der") for row in rows]),
        "mean_official_jer": mean([row.get("official_jer") for row in rows]),
        "mean_speaker_count_abs_diff": mean([row.get("speaker_count_abs_diff") for row in rows]),
        "mean_source_final_speaker_count_abs_diff": mean([row.get("source_final_speaker_count_abs_diff") for row in rows]),
        "source_final_speaker_count_exact_match_rate": mean([int(row["source_final_speaker_count_exact_match"]) for row in rows if row.get("source_final_speaker_count_exact_match") is not None]),
        "mean_source_final_turn_count_ratio": mean([row.get("source_final_turn_count_ratio") for row in rows]),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Diarization and Speaker-Structure Metrics", "", "**Protocol:** official pyannote.metrics DER/JER on reference-annotated source audio, plus same-backend source/final structure.", "", "## Aggregate", ""]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | dataset | ref/source/final speakers | activity F1 | DER | JER |", "|---|---|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        lines.append(f"| {row.get('audio_id')} | {row.get('source_dataset_id')} | {row.get('reference_speaker_count')}/{row.get('source_detected_speaker_count')}/{row.get('final_detected_speaker_count')} | {_fmt(row.get('speech_activity_f1'))} | {_fmt(row.get('official_der'))} | {_fmt(row.get('official_jer'))} |")
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--frame-sec", type=float, default=0.25)
    parser.add_argument("--collar-sec", type=float, default=0.0)
    parser.add_argument("--skip-overlap", action="store_true")
    parser.add_argument("--require-official", action="store_true")
    args = parser.parse_args()
    cfg = config_inputs(args.config)
    rows = [row_metrics(row, cfg, args.frame_sec, collar=args.collar_sec, skip_overlap=args.skip_overlap, require_official=args.require_official) for row in load_summary_rows(args.run_summary)]
    payload = {"schema_version": "aegisvox.eval.diarization_metrics.v2", "run_summary": str(args.run_summary), "config": str(args.config), "aggregate": aggregate(rows), "rows": rows, "official_protocol": {"backend": "pyannote.metrics", "collar_sec": args.collar_sec, "skip_overlap": args.skip_overlap, "required": args.require_official}}
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
