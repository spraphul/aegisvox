#!/usr/bin/env python3
"""Build the frozen per-conversation strata registry for W4 equity analyses.

This script creates metadata strata only.  It does not infer outcomes or use a
surrogate clinical score.  Every value is copied from frozen timeline/edit
evidence or deterministically binned under the clinical-study protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{lineno}: expected object")
        rows.append(value)
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_by(rows: list[dict[str, Any]], key: str, *, source: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"{source}: missing {key}")
        if value in out:
            raise ValueError(f"{source}: duplicate {key}={value}")
        out[value] = row
    return out


def collapse_paired_transcripts(
    rows: list[dict[str, Any]], *, source: Path
) -> dict[str, dict[str, Any]]:
    """Collapse the frozen reference/hypothesis transcript pair per audio.

    ``matched_asr_paired_transcripts.jsonl`` intentionally has two rows per
    conversation: one reference transcript and one hypothesis transcript.  The
    strata registry needs only their shared immutable run/dataset/language
    metadata.  We therefore require an exact reference+hypothesis pair and fail
    if any shared field disagrees rather than accepting an arbitrary duplicate.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        audio_id = str(row.get("audio_id") or "")
        if not audio_id:
            raise ValueError(f"{source}: missing audio_id")
        grouped.setdefault(audio_id, []).append(row)

    out: dict[str, dict[str, Any]] = {}
    shared_fields = ("run_id", "source_dataset_id", "source_language")
    for audio_id, group in grouped.items():
        labels = [str(row.get("source") or "") for row in group]
        if len(group) != 2 or sorted(labels) != ["hypothesis", "reference"]:
            raise ValueError(
                f"{source}: {audio_id} must have exactly one reference and one "
                f"hypothesis row; got labels={labels}"
            )
        canonical = dict(group[0])
        for field in shared_fields:
            values = {str(row.get(field) or "") for row in group}
            if len(values) != 1 or not next(iter(values)):
                raise ValueError(
                    f"{source}: {audio_id} inconsistent or missing {field}: {sorted(values)}"
                )
            canonical[field] = next(iter(values))
        out[audio_id] = canonical
    return out


def quantile_type7(values: list[float], probability: float) -> float:
    values = sorted(float(x) for x in values)
    if not values:
        raise ValueError("cannot compute a quantile of an empty collection")
    if len(values) == 1:
        return values[0]
    h = (len(values) - 1) * probability
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        return values[lo]
    return values[lo] + (h - lo) * (values[hi] - values[lo])


def duration_quartile(value: float, boundaries: list[float]) -> str:
    if value <= boundaries[0]:
        return "Q1_shortest"
    if value <= boundaries[1]:
        return "Q2"
    if value <= boundaries[2]:
        return "Q3"
    return "Q4_longest"


def speaker_count_bin(value: int) -> str:
    return str(value) if value in {1, 2, 3} else "4+"


def edit_count_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    return "4+"


def edit_duration_bin(value: float) -> str:
    if value < 1.0:
        return "<1s"
    if value < 2.0:
        return "1-2s"
    if value < 5.0:
        return "2-5s"
    if value < 10.0:
        return "5-10s"
    return ">=10s"


def resolve_deid(artifact_root: Path, run_id: str) -> Path:
    path = artifact_root / run_id / "deid_text.json"
    if not path.is_file():
        raise ValueError(f"missing frozen edit plan: {path}")
    return path


def build_registry(
    *,
    timelines_path: Path,
    paired_path: Path,
    artifact_root: Path,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timeline_rows = read_jsonl(timelines_path)
    paired_rows = read_jsonl(paired_path)
    timelines = unique_by(timeline_rows, "audio_id", source=timelines_path)
    pairs = collapse_paired_transcripts(paired_rows, source=paired_path)
    if len(timelines) != expected_rows or len(pairs) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} unique rows; timelines={len(timelines)}, pairs={len(pairs)}"
        )
    if set(timelines) != set(pairs):
        raise ValueError(
            "timeline/paired audio IDs are not bijective: "
            f"timeline_only={sorted(set(timelines)-set(pairs))[:5]}, "
            f"paired_only={sorted(set(pairs)-set(timelines))[:5]}"
        )

    durations = [float(row["duration_sec"]) for row in timelines.values()]
    boundaries = [quantile_type7(durations, p) for p in (0.25, 0.5, 0.75)]
    output: list[dict[str, Any]] = []
    edit_duration_counts: Counter[str] = Counter()
    phi_type_counts: Counter[str] = Counter()

    for audio_id in sorted(timelines):
        timeline = timelines[audio_id]
        pair = pairs[audio_id]
        run_id = str(pair.get("run_id") or "")
        if not run_id:
            raise ValueError(f"{audio_id}: paired row missing run_id")
        deid_path = resolve_deid(artifact_root, run_id)
        deid = json.loads(deid_path.read_text(encoding="utf-8"))
        edits = deid.get("edits") or []
        if not isinstance(edits, list):
            raise ValueError(f"{deid_path}: edits must be a list")
        edit_records = []
        local_phi: Counter[str] = Counter()
        for edit in edits:
            start = float(edit["start_sec"])
            end = float(edit["end_sec"])
            if end < start:
                raise ValueError(f"{deid_path}: edit end before start")
            duration = end - start
            duration_bin = edit_duration_bin(duration)
            phi_type = str(edit.get("phi_type") or "unknown")
            edit_duration_counts[duration_bin] += 1
            phi_type_counts[phi_type] += 1
            local_phi[phi_type] += 1
            edit_records.append(
                {
                    "edit_id": str(edit.get("edit_id") or ""),
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": duration,
                    "duration_bin": duration_bin,
                    "phi_type": phi_type,
                    "source_language": str(edit.get("source_language") or timeline.get("language") or "unknown"),
                    "replacement_language": str(edit.get("replacement_language") or "unknown"),
                }
            )
        speakers = sorted(
            {
                str(segment.get("speaker_id"))
                for segment in (timeline.get("diarization_segments") or [])
                if segment.get("speaker_id") not in (None, "")
            }
        )
        duration = float(timeline["duration_sec"])
        output.append(
            {
                "audio_id": audio_id,
                "run_id": run_id,
                "dataset_id": str(timeline.get("dataset_id") or pair.get("source_dataset_id") or "unknown"),
                "language": str(timeline.get("language") or pair.get("source_language") or "unknown"),
                "duration_sec": duration,
                "duration_quartile": duration_quartile(duration, boundaries),
                "speaker_count": len(speakers),
                "speaker_count_bin": speaker_count_bin(len(speakers)),
                "speaker_ids": speakers,
                "edit_count": len(edit_records),
                "edit_count_bin": edit_count_bin(len(edit_records)),
                "total_edit_duration_sec": sum(edit["duration_sec"] for edit in edit_records),
                "max_edit_duration_sec": max((edit["duration_sec"] for edit in edit_records), default=0.0),
                "phi_type_counts": dict(sorted(local_phi.items())),
                "edits": edit_records,
                "source_asr_backend": str(timeline.get("asr_backend") or "unknown"),
                "source_diarization_backend": str(timeline.get("diarization_backend") or "unknown"),
                "source_timeline_sha256": str(timeline.get("transcript_sha256") or ""),
                "deid_plan_sha256": sha256(deid_path),
            }
        )

    summary = {
        "schema_version": "aegisvox.clinical_strata_registry.v1",
        "status": "complete",
        "num_rows": len(output),
        "num_unique_audio_ids": len({row["audio_id"] for row in output}),
        "duration_quartile_boundaries_sec_type7": boundaries,
        "dataset_counts": dict(sorted(Counter(row["dataset_id"] for row in output).items())),
        "language_counts": dict(sorted(Counter(row["language"] for row in output).items())),
        "duration_quartile_counts": dict(sorted(Counter(row["duration_quartile"] for row in output).items())),
        "speaker_count_bin_counts": dict(sorted(Counter(row["speaker_count_bin"] for row in output).items())),
        "edit_count_bin_counts": dict(sorted(Counter(row["edit_count_bin"] for row in output).items())),
        "edit_duration_bin_counts": dict(sorted(edit_duration_counts.items())),
        "phi_type_counts": dict(sorted(phi_type_counts.items())),
        "num_edits": sum(row["edit_count"] for row in output),
        "inputs": {
            "timelines": {"path": str(timelines_path), "sha256": sha256(timelines_path)},
            "paired_transcripts": {"path": str(paired_path), "sha256": sha256(paired_path)},
            "artifact_root": str(artifact_root),
        },
        "claim_boundary": "Deterministic frozen metadata strata only; contains no clinical outcome or proxy label.",
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-timelines", type=Path, required=True)
    parser.add_argument("--paired-transcripts", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=390)
    args = parser.parse_args()
    rows, summary = build_registry(
        timelines_path=args.source_timelines,
        paired_path=args.paired_transcripts,
        artifact_root=args.artifact_root,
        expected_rows=args.expected_rows,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary["output_jsonl"] = str(args.output_jsonl)
    summary["output_sha256"] = sha256(args.output_jsonl)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "num_rows": len(rows), "num_edits": summary["num_edits"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
