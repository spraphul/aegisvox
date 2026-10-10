#!/usr/bin/env python3
"""Compute directly measured duration-retention statistics for AegisVox runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import row_completed, final_audio_path, frame_activity, input_audio_path, load_summary_rows, mean, predicted_diarization_segments, run_dir_from_summary_row, speech_duration, wav_info, write_csv, write_json


def gate_duration_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "missing"
    if ratio < 0.90 or ratio > 1.15:
        return "fail"
    if ratio < 0.95 or ratio > 1.08:
        return "warn"
    return "pass"


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    run_dir = run_dir_from_summary_row(row)
    inp = input_audio_path(run_dir)
    fin = final_audio_path(run_dir)
    input_info = wav_info(inp)
    final_info = wav_info(fin)
    input_dur = input_info["duration_sec"]
    final_dur = final_info["duration_sec"]
    ratio = (final_dur / input_dur) if isinstance(input_dur, (int, float)) and input_dur and isinstance(final_dur, (int, float)) else None
    pred = predicted_diarization_segments(run_dir)
    predicted_speech_sec = speech_duration(pred) if pred else None
    speech_duration_ratio = (predicted_speech_sec / input_dur) if isinstance(input_dur, (int, float)) and input_dur and isinstance(predicted_speech_sec, (int, float)) else None
    return {
        "audio_id": row.get("audio_id"),
        "run_id": row.get("run_id"),
        "completed": row_completed(row),
        "input_audio_path": str(inp),
        "final_audio_path": str(fin),
        "input_duration_sec": input_dur,
        "final_duration_sec": final_dur,
        "absolute_duration_delta_sec": (final_dur - input_dur) if isinstance(input_dur, (int, float)) and isinstance(final_dur, (int, float)) else None,
        "duration_ratio_final_over_input": ratio,
        "duration_gate": gate_duration_ratio(ratio),
        "predicted_speech_sec_input": predicted_speech_sec,
        "predicted_speech_ratio_input": speech_duration_ratio,
        "omitted_speech_seconds": None,
        "inserted_speech_seconds": None,
        "repetition_seconds": None,
        "claim_boundary": "Direct file-duration retention only. Content completeness is evaluated separately with matched-ASR and human listener judgments.",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_completed": sum(1 for row in rows if row.get("completed")),
        "mean_duration_ratio": mean([row.get("duration_ratio_final_over_input") for row in rows]),
        "num_duration_pass": sum(1 for row in rows if row.get("duration_gate") == "pass"),
        "num_duration_warn": sum(1 for row in rows if row.get("duration_gate") == "warn"),
        "num_duration_fail": sum(1 for row in rows if row.get("duration_gate") == "fail"),
        "num_duration_missing": sum(1 for row in rows if row.get("duration_gate") == "missing"),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Duration-Retention Metrics",
        "",
        "**Claim boundary:** directly measured input/final duration ratios only; duration equality is not treated as content completeness.",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | input sec | final sec | ratio | gate |", "|---|---:|---:|---:|---|"]
    for row in payload["rows"]:
        lines.append(f"| {row.get('audio_id')} | {_fmt(row.get('input_duration_sec'))} | {_fmt(row.get('final_duration_sec'))} | {_fmt(row.get('duration_ratio_final_over_input'))} | {row.get('duration_gate')} |")
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    rows = [row_metrics(row) for row in load_summary_rows(args.run_summary)]
    payload = {"schema_version": "aegisvox.eval.duration_completeness.v1", "run_summary": str(args.run_summary), "aggregate": aggregate(rows), "rows": rows}
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
