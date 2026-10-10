#!/usr/bin/env python3
"""Fail-closed final statistical hierarchy for the comprehensive study.

The report combines the four frozen primary estimands only after genuine human
results and converged crossed mixed models exist. Secondary studies remain in
their predeclared within-study multiplicity families and are referenced by
content hash rather than pooled post hoc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing artifact: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered); running = 0.0; result = {}
    for rank, (name, p_value) in enumerate(ordered, 1):
        running = max(running, (m - rank + 1) * p_value)
        result[name] = min(1.0, running)
    return result


def human_primary(mixed: dict[str, Any]) -> dict[str, Any]:
    rows = mixed.get("primary_results")
    if isinstance(rows, dict):
        # jsonlite may encode a one-row/data-frame object as parallel vectors.
        keys = list(rows)
        length = max((len(v) for v in rows.values() if isinstance(v, list)), default=1)
        rows = [{key: (value[i] if isinstance(value, list) else value) for key, value in rows.items()} for i in range(length)]
    if not isinstance(rows, list):
        raise ValueError("mixed model primary_results missing")
    matches = [row for row in rows if str(row.get("endpoint")) == "clinical.meaning"]
    if len(matches) != 1:
        raise ValueError("exactly one clinical.meaning mixed-model primary is required")
    row = matches[0]
    p_value = row.get("p_value"); estimate = row.get("estimate")
    if not isinstance(p_value, (int, float)) or not 0 <= float(p_value) <= 1:
        raise ValueError("invalid human clinical primary p-value")
    if not isinstance(estimate, (int, float)):
        raise ValueError("invalid human clinical primary estimate")
    return {
        "endpoint": "human_clinical_preservation",
        "model": row.get("model"), "estimand": row.get("estimand"),
        "estimate": float(estimate), "std_error": row.get("std_error"),
        "ci95_low": row.get("ci95_low"), "ci95_high": row.get("ci95_high"),
        "raw_p_two_sided": float(p_value), "direction": "positive favors AegisVox-F5",
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    model = load(args.primary_model_estimands)
    if model.get("schema_version") != "aegisvox.primary_model_estimands.v1" or model.get("status") != "complete":
        raise ValueError("primary model estimands are incomplete")
    if model.get("unresolved_primary_estimands") != ["human_clinical_preservation"]:
        raise ValueError("model primary report must have only the human primary unresolved")

    human = load(args.human_results)
    if human.get("schema_version") != "aegisvox.human_study_results.v1" or human.get("status") != "complete":
        raise ValueError("human study results are incomplete")
    if human.get("num_valid_ratings") != 4500 or human.get("panel_counts") != {"clinical": 750, "perceptual": 1250, "privacy": 2500}:
        raise ValueError("human rating denominators are not exact")
    provenance = human.get("provenance") or {}
    if provenance.get("synthetic") is not False or provenance.get("human_attestation_required") is not True:
        raise ValueError("human results lack genuine attested provenance")

    mixed = load(args.human_mixed_models)
    if mixed.get("schema_version") != "aegisvox.clinical_human_mixed_models.v1" or mixed.get("status") != "complete":
        raise ValueError("human mixed models are incomplete")
    if mixed.get("counts") != {"clinical": 750, "perceptual": 1250, "privacy": 2500}:
        raise ValueError("mixed-model denominators are not exact")
    convergence = mixed.get("convergence") or {}
    if convergence.get("all_gradients_finite") is not True or convergence.get("all_ordinal_gradients_below_5e_minus_3") is not True:
        raise ValueError("mixed-model convergence gate failed")

    required_support = {
        "equity": args.equity,
        "frontend": args.frontend,
        "robustness": args.robustness,
    }
    support = {}
    for label, path in required_support.items():
        value = load(path)
        if value.get("status") != "complete":
            raise ValueError(f"supporting statistics incomplete: {label}")
        support[label] = {"path": str(path), "sha256": sha256(path), "schema_version": value.get("schema_version")}

    cap = model["clinical_no_major_error"]["primary"]
    relex = model["replacement_realization"]["primary"]
    human_endpoint = human_primary(mixed)
    scalar = {
        "clinical_no_major_error": float(cap["raw_p_two_sided"]),
        "replacement_realization": float(relex["raw_p_two_sided"]),
        "human_clinical_preservation": float(human_endpoint["raw_p_two_sided"]),
    }
    adjusted = holm(scalar)
    primaries = {
        "clinical_no_major_error": {**cap, "holm_adjusted_p": adjusted["clinical_no_major_error"]},
        "replacement_realization": {**relex, "holm_adjusted_p": adjusted["replacement_realization"]},
        "human_clinical_preservation": {**human_endpoint, "holm_adjusted_p": adjusted["human_clinical_preservation"]},
        "privacy_utility_tradeoff": model["privacy_utility_tradeoff"],
    }
    return {
        "schema_version": "aegisvox.final_hierarchical_statistics.v1", "status": "complete",
        "primary_estimands": primaries,
        "primary_family": {
            "frozen_size": 4,
            "scalar_hypothesis_tests": 3,
            "multivariate_descriptive_estimands": 1,
            "scalar_multiplicity_method": "Holm family-wise error control across the three scalar primary hypotheses",
            "privacy_utility_policy": "predeclared effect vector; no post-hoc scalarization and no equivalence claim from nonsignificance",
        },
        "missingness_policy": (
            "Intention-to-transform is primary. Generation, ASR, schema, judge, and operational failures remain in denominators. "
            "Uncertain model judgments are failures in the primary analysis and successes in the optimistic bound. "
            "Uncertain human ratings remain in provenance/distributions and are excluded only from likelihoods as frozen."
        ),
        "multiplicity_policy": (
            "Holm controls the scalar primary family. Secondary endpoints retain their predeclared within-study BH/FDR or family-wise procedures; "
            "subgroups and interactions are exploratory unless independently powered."
        ),
        "secondary_family_artifacts": support,
        "inputs": {
            "primary_model_estimands": {"path": str(args.primary_model_estimands), "sha256": sha256(args.primary_model_estimands)},
            "human_results": {"path": str(args.human_results), "sha256": sha256(args.human_results)},
            "human_mixed_models": {"path": str(args.human_mixed_models), "sha256": sha256(args.human_mixed_models)},
        },
        "deterministic_clinical_proxy_used": False,
        "claim_boundary": "Final statistical hierarchy for the frozen public390 and genuine attested human study; it does not itself authorize paper writing or imply deployment safety/compliance.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary-model-estimands", type=Path, required=True)
    ap.add_argument("--human-results", type=Path, required=True)
    ap.add_argument("--human-mixed-models", type=Path, required=True)
    ap.add_argument("--equity", type=Path, required=True)
    ap.add_argument("--frontend", type=Path, required=True)
    ap.add_argument("--robustness", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = finalize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
