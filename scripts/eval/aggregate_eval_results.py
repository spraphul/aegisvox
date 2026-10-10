#!/usr/bin/env python3
"""Aggregate modular AegisVox eval metric JSON files into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def metric_name(payload: dict[str, Any], path: Path) -> str:
    schema = str(payload.get("schema_version") or path.stem)
    if schema.startswith("aegisvox.eval."):
        # Metric schema versions evolve independently; aggregate keys must stay
        # stable or a v2 metric silently disappears from readiness checks and
        # rows_by_audio consumers.
        return re.sub(r"\.v[0-9]+$", "", schema.removeprefix("aegisvox.eval."))
    return path.stem


def aggregate(files: list[Path]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    rows_by_audio: dict[str, dict[str, Any]] = {}
    for path in files:
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"metric file must be JSON object: {path}")
        name = metric_name(payload, path)
        metrics[name] = {"path": str(path), "aggregate": payload.get("aggregate", {})}
        for row in payload.get("rows", []) or []:
            if not isinstance(row, dict) or not row.get("audio_id"):
                continue
            audio_id = str(row["audio_id"])
            rows_by_audio.setdefault(audio_id, {"audio_id": audio_id})[name] = row
    return {
        "schema_version": "aegisvox.eval.aggregate.v1",
        "metric_files": [str(path) for path in files],
        "metrics": metrics,
        "rows_by_audio": rows_by_audio,
        "claim_boundary": "Aggregated metric packet. Individual metric claim boundaries still apply; not paper-ready unless coverage tracker rows are paper_ready.",
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# AegisVox Evaluation Aggregate", "", "**Claim boundary:** aggregated protocol metrics unless the coverage tracker marks each metric paper-ready.", "", "## Metric aggregates", ""]
    for name, metric in payload["metrics"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- `path`: {metric['path']}")
        for key, value in (metric.get("aggregate") or {}).items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines.append("## Audio coverage")
    lines.append("")
    lines.append("| audio_id | metric groups |")
    lines.append("|---|---|")
    for audio_id, row in sorted(payload["rows_by_audio"].items()):
        groups = sorted(k for k in row if k != "audio_id")
        lines.append(f"| {audio_id} | {', '.join(groups)} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate(args.metric_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), "num_metric_files": len(args.metric_json), "num_audio_rows": len(payload["rows_by_audio"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
