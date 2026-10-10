#!/usr/bin/env python3
"""Compute bootstrap CIs and paired comparison summaries for metric JSON files.

The script is intentionally generic: each metric JSON should expose a `rows`
array with an `audio_id` and numeric row-level fields. It produces paper-support
statistics without inventing metrics that are absent from the input files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from common import source_dataset_id, source_language, write_json

SKIP_NUMERIC_KEYS = {
    "audio_id",
    "run_id",
    "completed",
    "schema_version",
}


def load_metric(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise SystemExit(f"metric JSON missing rows[]: {path}")
    schema = str(payload.get("schema_version") or "")
    match = re.fullmatch(r"aegisvox\.eval\.(.+?)\.v\d+", schema)
    metric_name = match.group(1) if match else path.stem
    return {"path": str(path), "stem": metric_name, "payload": payload, "rows": rows}


def numeric_fields(rows: list[dict[str, Any]], *, min_count: int) -> list[str]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if key in SKIP_NUMERIC_KEYS:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                counts[key] = counts.get(key, 0) + 1
    return sorted(k for k, c in counts.items() if c >= min_count)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def bootstrap_mean_ci(values: list[float], *, iterations: int, seed: int, alpha: float) -> dict[str, Any]:
    values = [float(v) for v in values if math.isfinite(float(v))]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "method": "bootstrap_mean"}
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(max(0, iterations)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    return {
        "n": n,
        "mean": mean(values),
        "median": percentile(values, 0.5),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "ci_low": percentile(boots, alpha / 2.0) if boots else None,
        "ci_high": percentile(boots, 1.0 - alpha / 2.0) if boots else None,
        "iterations": iterations,
        "alpha": alpha,
        "method": "nonparametric_row_bootstrap_mean",
    }


def metric_ci_summary(metric: dict[str, Any], *, iterations: int, seed: int, alpha: float, min_count: int) -> dict[str, Any]:
    rows = metric["rows"]
    out: dict[str, Any] = {
        "metric": metric["stem"],
        "source_json": metric["path"],
        "num_rows": len(rows),
        "fields": {},
    }
    for idx, field in enumerate(numeric_fields(rows, min_count=min_count)):
        vals = [float(row[field]) for row in rows if isinstance(row, dict) and isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool) and math.isfinite(float(row[field]))]
        out["fields"][field] = bootstrap_mean_ci(vals, iterations=iterations, seed=seed + idx * 1009, alpha=alpha)
    return out


def stratified_metric_ci_summary(
    metric: dict[str, Any],
    *,
    group_name: str,
    group_fn,
    iterations: int,
    seed: int,
    alpha: float,
    min_count: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in metric["rows"]:
        if not isinstance(row, dict):
            continue
        group = str(group_fn(row) or "unknown")
        groups.setdefault(group, []).append(row)
    return {
        "metric": metric["stem"],
        "grouping": group_name,
        "groups": {
            group: metric_ci_summary(
                {**metric, "rows": rows},
                iterations=iterations,
                seed=seed + idx * 2029,
                alpha=alpha,
                min_count=min_count,
            )
            for idx, (group, rows) in enumerate(sorted(groups.items()))
        },
    }


def row_map(metric: dict[str, Any], field: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in metric["rows"]:
        if not isinstance(row, dict):
            continue
        audio_id = row.get("audio_id")
        value = row.get(field)
        if audio_id is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        v = float(value)
        if math.isfinite(v):
            out[str(audio_id)] = v
    return out


def paired_comparisons(
    *,
    full_metric: dict[str, Any],
    baseline_metric: dict[str, Any],
    iterations: int,
    seed: int,
    alpha: float,
    min_count: int,
) -> dict[str, Any]:
    full_fields = set(numeric_fields(full_metric["rows"], min_count=min_count))
    base_fields = set(numeric_fields(baseline_metric["rows"], min_count=min_count))
    fields = sorted(full_fields & base_fields)
    out: dict[str, Any] = {
        "full_metric": full_metric["stem"],
        "baseline_metric": baseline_metric["stem"],
        "full_json": full_metric["path"],
        "baseline_json": baseline_metric["path"],
        "fields": {},
    }
    for idx, field in enumerate(fields):
        f = row_map(full_metric, field)
        b = row_map(baseline_metric, field)
        ids = sorted(set(f) & set(b))
        diffs = [f[audio_id] - b[audio_id] for audio_id in ids]
        if len(diffs) < min_count:
            continue
        out["fields"][field] = {
            **bootstrap_mean_ci(diffs, iterations=iterations, seed=seed + idx * 1613, alpha=alpha),
            "paired_n": len(diffs),
            "mean_full": mean([f[i] for i in ids]),
            "mean_baseline": mean([b[i] for i in ids]),
            "mean_difference_full_minus_baseline": mean(diffs),
            "num_full_better_if_lower": sum(1 for d in diffs if d < 0),
            "num_full_better_if_higher": sum(1 for d in diffs if d > 0),
            "claim_boundary": "Directionality depends on metric semantics; inspect field-specific meaning before claiming improvement.",
        }
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Bootstrap and Paired Statistical Report",
        "",
        "**Claim boundary:** generic row-level nonparametric bootstrap. Field directionality is not inferred; paired differences are full-minus-baseline.",
        "",
        "## Bootstrap CIs",
        "",
    ]
    for metric in payload["bootstrap_ci"]:
        lines.append(f"### {metric['metric']}")
        lines.append("")
        lines.append("| field | n | mean | median | ci_low | ci_high |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for field, stats in metric["fields"].items():
            lines.append(
                f"| {field} | {stats.get('n')} | {_fmt(stats.get('mean'))} | {_fmt(stats.get('median'))} | "
                f"{_fmt(stats.get('ci_low'))} | {_fmt(stats.get('ci_high'))} |"
            )
        lines.append("")
    for grouping, summaries in (payload.get("stratified_bootstrap_ci") or {}).items():
        lines += [f"## Stratified bootstrap: {grouping}", ""]
        for metric in summaries:
            lines.append(f"### {metric['metric']}")
            lines.append("")
            lines.append("| group | rows | numeric fields |")
            lines.append("|---|---:|---:|")
            for group, group_summary in metric.get("groups", {}).items():
                lines.append(f"| {group} | {group_summary.get('num_rows')} | {len(group_summary.get('fields') or {})} |")
            lines.append("")
            lines.append("| group | field | n | mean | median | ci_low | ci_high |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for group, group_summary in metric.get("groups", {}).items():
                for field, stats in (group_summary.get("fields") or {}).items():
                    lines.append(
                        f"| {group} | {field} | {stats.get('n')} | {_fmt(stats.get('mean'))} | "
                        f"{_fmt(stats.get('median'))} | {_fmt(stats.get('ci_low'))} | {_fmt(stats.get('ci_high'))} |"
                    )
            lines.append("")
    if payload.get("paired_comparisons"):
        lines += ["## Paired Comparisons", ""]
        for comp in payload["paired_comparisons"]:
            lines.append(f"### {comp['full_metric']} vs {comp['baseline_metric']}")
            lines.append("")
            lines.append("| field | paired_n | mean_full | mean_baseline | mean_diff | ci_low | ci_high |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for field, stats in comp["fields"].items():
                lines.append(
                    f"| {field} | {stats.get('paired_n')} | {_fmt(stats.get('mean_full'))} | {_fmt(stats.get('mean_baseline'))} | "
                    f"{_fmt(stats.get('mean_difference_full_minus_baseline'))} | {_fmt(stats.get('ci_low'))} | {_fmt(stats.get('ci_high'))} |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-json", type=Path, action="append", required=True, help="Metric JSON with rows[] for CI computation.")
    parser.add_argument("--paired", nargs=2, action="append", metavar=("FULL_JSON", "BASELINE_JSON"), help="Paired comparison metric JSONs with shared audio_id rows.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()

    metrics = [load_metric(p) for p in args.metric_json]
    payload: dict[str, Any] = {
        "schema_version": "aegisvox.eval.statistical_report.v2",
        "bootstrap_ci": [metric_ci_summary(m, iterations=args.iterations, seed=args.seed, alpha=args.alpha, min_count=args.min_count) for m in metrics],
        "stratified_bootstrap_ci": {
            "by_source_dataset": [
                stratified_metric_ci_summary(
                    m,
                    group_name="source_dataset_id",
                    group_fn=source_dataset_id,
                    iterations=args.iterations,
                    seed=args.seed + idx * 4001,
                    alpha=args.alpha,
                    min_count=args.min_count,
                )
                for idx, m in enumerate(metrics)
            ],
            "by_source_language": [
                stratified_metric_ci_summary(
                    m,
                    group_name="source_language",
                    group_fn=source_language,
                    iterations=args.iterations,
                    seed=args.seed + idx * 5003,
                    alpha=args.alpha,
                    min_count=args.min_count,
                )
                for idx, m in enumerate(metrics)
            ],
        },
        "paired_comparisons": [],
        "claim_boundary": "Generic statistical helper. It does not infer whether higher or lower is better for a field.",
    }
    for idx, pair in enumerate(args.paired or []):
        payload["paired_comparisons"].append(
            paired_comparisons(
                full_metric=load_metric(Path(pair[0])),
                baseline_metric=load_metric(Path(pair[1])),
                iterations=args.iterations,
                seed=args.seed + idx * 7919,
                alpha=args.alpha,
                min_count=args.min_count,
            )
        )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), "num_metrics": len(metrics), "num_paired": len(payload["paired_comparisons"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
