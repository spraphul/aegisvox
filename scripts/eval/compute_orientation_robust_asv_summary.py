#!/usr/bin/env python3
"""Conservatively correct fixed-direction ASV EER for score inversion.

If same-source converted scores are systematically *lower* than impostor
scores, a knowledgeable attacker can invert the score. Conventional privacy
reporting must therefore not interpret raw EER > 0.5 as stronger-than-random
privacy. This report preserves the original fixed-direction values and adds the
attacker-optimal orientation-corrected EER min(EER, 1-EER), including a valid
transformation of the reported bootstrap interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transformed_interval(low: float, high: float) -> tuple[float, float]:
    if not (0 <= low <= high <= 1):
        raise ValueError("EER interval must lie in [0,1]")
    transform = lambda value: min(value, 1.0 - value)
    lower = min(transform(low), transform(high))
    upper = 0.5 if low <= 0.5 <= high else max(transform(low), transform(high))
    return lower, upper


def summarize(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    aggregate = value.get("aggregate") or {}
    raw = aggregate.get("asv_eer"); low = aggregate.get("asv_eer_ci95_low"); high = aggregate.get("asv_eer_ci95_high")
    if not all(isinstance(v, (int, float)) for v in (raw, low, high)):
        raise ValueError(f"missing EER/CI in {path}")
    if aggregate.get("num_completed") != 390 or aggregate.get("num_computed_speakers") != 809:
        raise ValueError(f"ASV population mismatch in {path}")
    corrected_low, corrected_high = transformed_interval(float(low), float(high))
    direction = "lower_score_indicates_same_source" if float(raw) > 0.5 else "higher_score_indicates_same_source"
    return {
        "backend": value.get("backend"), "backend_version": value.get("backend_version"),
        "num_source_recordings": 390, "num_source_speakers": 809,
        "num_positive_trials": aggregate.get("num_positive_trials"),
        "num_negative_trials": aggregate.get("num_negative_trials"),
        "raw_fixed_direction_eer": float(raw),
        "raw_fixed_direction_ci95": [float(low), float(high)],
        "attacker_optimal_orientation": direction,
        "attacker_optimal_eer": min(float(raw), 1.0 - float(raw)),
        "attacker_optimal_eer_ci95": [corrected_low, corrected_high],
        "fixed_high_score_far1_linkability": aggregate.get("source_linkability_at_far1"),
        "fixed_high_score_far1_linkability_ci95": [
            aggregate.get("source_linkability_at_far1_ci95_low"), aggregate.get("source_linkability_at_far1_ci95_high")
        ],
        "input": {"path": str(path), "sha256": sha256(path), "schema_version": value.get("schema_version")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", action="append", required=True, help="system/backend=metric.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    systems: dict[str, dict[str, Any]] = {}
    for raw in args.metric:
        key, value = raw.split("=", 1)
        if not key or key in systems:
            raise ValueError(f"blank/duplicate metric key: {key!r}")
        systems[key] = summarize(Path(value))
    report = {
        "schema_version": "aegisvox.orientation_robust_asv_summary.v1", "status": "complete",
        "systems": systems,
        "headline_metric": "attacker_optimal_eer",
        "orientation_policy": "An attacker may choose cosine score orientation; headline EER is min(raw EER, 1-raw EER).",
        "far1_boundary": "FAR1 values remain fixed-high-score auxiliary evidence because lower-tail trial scores were not retained in the compact artifact; they are not orientation-robust headline evidence.",
        "claim_boundary": "Measured project ASV protocols, not official VoicePrivacy challenge evaluation and not an anonymity guarantee.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status":"complete","systems":len(systems)},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
