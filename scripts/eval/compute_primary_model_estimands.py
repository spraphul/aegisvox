#!/usr/bin/env python3
"""Compute the two frozen non-human primary AegisVox estimands.

Estimand 1 is the public390 conversation-level F5-minus-stock difference in
no-major-clinical-error under the qualified source-anchored CAP judge.
Estimand 2 is the 601-edit F5-minus-stock difference in replacement realization,
with conversation-clustered bootstrap and sign-flip inference. Intention-to-
transform denominators and optimistic/pessimistic uncertainty bounds are
explicit. Human and privacy-utility primaries are intentionally not imputed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


REALIZED = {"realized", "acceptable_variant"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = key(row)
        if not value or value in result:
            raise ValueError(f"{label}: blank/duplicate ID {value!r}")
        result[value] = row
    return result


def paired_cluster_inference(
    records: list[tuple[str, float]], *, bootstrap_iterations: int,
    permutation_iterations: int, seed: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("paired inference requires at least one record")
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for cluster, difference in records:
        by_cluster[cluster].append(float(difference))
    clusters = sorted(by_cluster)
    values = np.asarray([d for _, d in records], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.empty(bootstrap_iterations, dtype=np.float64)
    for i in range(bootstrap_iterations):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        draw = [d for index in sampled for d in by_cluster[clusters[int(index)]]]
        boots[i] = float(np.mean(draw))
    observed = abs(float(values.mean()))
    extreme = 0
    for _ in range(permutation_iterations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(clusters))
        permuted = [float(signs[i]) * d for i, cluster in enumerate(clusters) for d in by_cluster[cluster]]
        extreme += int(abs(float(np.mean(permuted))) >= observed - 1e-15)
    return {
        "n_itt": len(records), "n_clusters": len(clusters),
        "f5_rate": None, "stock_rate": None,
        "risk_difference_f5_minus_stock": float(values.mean()),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "raw_p_two_sided": (extreme + 1) / (permutation_iterations + 1),
        "bootstrap_iterations": bootstrap_iterations,
        "permutation_iterations": permutation_iterations,
        "resampling_unit": "source_conversation",
    }


def cap_value(row: dict[str, Any], *, optimistic: bool) -> int:
    result = row.get("result") or {}
    assessment = str(result.get("conversation_assessment") or "")
    allowed = {"no_major_error", "major_omission", "major_contradiction", "major_unsupported_addition", "uncertain"}
    if assessment not in allowed:
        raise ValueError(f"invalid CAP conversation assessment: {assessment!r}")
    return int(assessment == "no_major_error" or (optimistic and assessment == "uncertain"))


def cap_estimand(f5_path: Path, stock_path: Path, *, expected_rows: int, iterations: int, seed: int) -> dict[str, Any]:
    f5 = unique(read_jsonl(f5_path), lambda r: str(r.get("audio_id") or ""), "F5 CAP")
    stock = unique(read_jsonl(stock_path), lambda r: str(r.get("audio_id") or ""), "stock CAP")
    if len(f5) != expected_rows or set(f5) != set(stock):
        raise ValueError("CAP population is not an exact paired public390 matrix")
    reports = {}
    for index, optimistic in enumerate((False, True)):
        label = "pessimistic_uncertain_failure" if not optimistic else "optimistic_uncertain_success"
        f5_values = {key: cap_value(row, optimistic=optimistic) for key, row in f5.items()}
        stock_values = {key: cap_value(row, optimistic=optimistic) for key, row in stock.items()}
        report = paired_cluster_inference(
            [(key, f5_values[key] - stock_values[key]) for key in sorted(f5)],
            bootstrap_iterations=iterations, permutation_iterations=iterations * 2, seed=seed + index,
        )
        report["f5_rate"] = sum(f5_values.values()) / expected_rows
        report["stock_rate"] = sum(stock_values.values()) / expected_rows
        report["f5_successes"] = sum(f5_values.values())
        report["stock_successes"] = sum(stock_values.values())
        reports[label] = report
    uncertain = {
        "f5": sum(str((row.get("result") or {}).get("conversation_assessment")) == "uncertain" for row in f5.values()),
        "stock": sum(str((row.get("result") or {}).get("conversation_assessment")) == "uncertain" for row in stock.values()),
    }
    return {
        "estimand": "F5 minus stock VoiceCraft conversation-level no-major-clinical-error rate",
        "unit": "source_conversation", "expected_itt": expected_rows,
        "primary": reports["pessimistic_uncertain_failure"],
        "uncertainty_sensitivity": reports, "uncertain_counts": uncertain,
        "input_hashes": {"f5": sha256(f5_path), "stock": sha256(stock_path)},
    }


def judgment_value(row: dict[str, Any], *, optimistic: bool) -> int:
    status = str(row.get("replacement_status") or "")
    if status not in REALIZED | {"not_realized", "uncertain"}:
        raise ValueError(f"invalid replacement status: {status!r}")
    return int(status in REALIZED or (optimistic and status == "uncertain"))


def relex_estimand(f5_path: Path, stock_path: Path, *, expected_edits: int, iterations: int, seed: int) -> dict[str, Any]:
    f5 = unique(read_jsonl(f5_path), lambda r: str(r.get("adjudication_id") or ""), "F5 relex")
    stock = unique(read_jsonl(stock_path), lambda r: str(r.get("adjudication_id") or ""), "stock relex")
    if len(f5) != expected_edits or set(f5) != set(stock):
        raise ValueError("relex population is not an exact paired edit matrix")
    reports = {}
    for index, optimistic in enumerate((False, True)):
        label = "pessimistic_uncertain_failure" if not optimistic else "optimistic_uncertain_success"
        f5_values = {key: judgment_value(row, optimistic=optimistic) for key, row in f5.items()}
        stock_values = {key: judgment_value(row, optimistic=optimistic) for key, row in stock.items()}
        records = [(key.split("::edit", 1)[0], f5_values[key] - stock_values[key]) for key in sorted(f5)]
        report = paired_cluster_inference(
            records, bootstrap_iterations=iterations, permutation_iterations=iterations * 2,
            seed=seed + index,
        )
        report["f5_rate"] = sum(f5_values.values()) / expected_edits
        report["stock_rate"] = sum(stock_values.values()) / expected_edits
        report["f5_successes"] = sum(f5_values.values())
        report["stock_successes"] = sum(stock_values.values())
        reports[label] = report
    uncertain = {
        "f5": sum(str(row.get("replacement_status")) == "uncertain" for row in f5.values()),
        "stock": sum(str(row.get("replacement_status")) == "uncertain" for row in stock.values()),
    }
    return {
        "estimand": "F5 minus stock VoiceCraft replacement realization rate",
        "unit": "edit_intention_clustered_by_source_conversation", "expected_itt": expected_edits,
        "primary": reports["pessimistic_uncertain_failure"],
        "uncertainty_sensitivity": reports, "uncertain_counts": uncertain,
        "input_hashes": {"f5": sha256(f5_path), "stock": sha256(stock_path)},
    }


def metric_rows(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{label}: missing rows[]")
    return unique(rows, lambda r: str(r.get("audio_id") or ""), label)


def numeric_field(row: dict[str, Any], field: str, label: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValueError(f"{label}: missing/nonfinite {field}")
    return float(value)


def privacy_utility_estimand(
    *, f5_cap_path: Path, stock_cap_path: Path,
    f5_ecapa_path: Path, stock_ecapa_path: Path,
    f5_xvector_path: Path, stock_xvector_path: Path,
    expected_rows: int, iterations: int, seed: int,
) -> dict[str, Any]:
    f5_cap = unique(read_jsonl(f5_cap_path), lambda r: str(r.get("audio_id") or ""), "F5 CAP")
    stock_cap = unique(read_jsonl(stock_cap_path), lambda r: str(r.get("audio_id") or ""), "stock CAP")
    f5_ecapa = metric_rows(f5_ecapa_path, label="F5 ECAPA")
    stock_ecapa = metric_rows(stock_ecapa_path, label="stock ECAPA")
    f5_xvector = metric_rows(f5_xvector_path, label="F5 x-vector")
    stock_xvector = metric_rows(stock_xvector_path, label="stock x-vector")
    maps = (f5_cap, stock_cap, f5_ecapa, stock_ecapa, f5_xvector, stock_xvector)
    ids = set(f5_cap)
    if len(ids) != expected_rows or any(set(value) != ids for value in maps):
        raise ValueError("privacy-utility inputs are not an exact paired public390 population")

    endpoint_specs = {
        "ecapa_max_source_converted_similarity": (f5_ecapa, stock_ecapa, "max_source_converted_similarity", False),
        "xvector_max_source_converted_similarity": (f5_xvector, stock_xvector, "max_source_converted_similarity", False),
    }
    endpoints: dict[str, Any] = {}
    for index, (name, (f5_map, stock_map, field, higher_is_better)) in enumerate(endpoint_specs.items()):
        f5_values = {key: numeric_field(f5_map[key], field, f"F5/{name}") for key in ids}
        stock_values = {key: numeric_field(stock_map[key], field, f"stock/{name}") for key in ids}
        report = paired_cluster_inference(
            [(key, f5_values[key] - stock_values[key]) for key in sorted(ids)],
            bootstrap_iterations=iterations, permutation_iterations=iterations * 2, seed=seed + index,
        )
        report["f5_mean"] = float(np.mean(list(f5_values.values())))
        report["stock_mean"] = float(np.mean(list(stock_values.values())))
        report["higher_is_better"] = higher_is_better
        report["interpretation"] = "negative F5-minus-stock difference favors F5 unlinkability"
        endpoints[name] = report

    f5_utility = {key: numeric_field(f5_cap[key].get("metrics") or {}, "strict_preservation_rate", "F5 CAP utility") for key in ids}
    stock_utility = {key: numeric_field(stock_cap[key].get("metrics") or {}, "strict_preservation_rate", "stock CAP utility") for key in ids}
    utility = paired_cluster_inference(
        [(key, f5_utility[key] - stock_utility[key]) for key in sorted(ids)],
        bootstrap_iterations=iterations, permutation_iterations=iterations * 2, seed=seed + 2,
    )
    utility["f5_mean"] = float(np.mean(list(f5_utility.values())))
    utility["stock_mean"] = float(np.mean(list(stock_utility.values())))
    utility["higher_is_better"] = True
    utility["interpretation"] = "positive F5-minus-stock difference favors F5 CAP preservation"
    return {
        "estimand": "ASV source unlinkability versus clinical CAP preservation for F5 and stock VoiceCraft",
        "unit": "source_conversation", "expected_itt": expected_rows,
        "privacy_endpoints": endpoints,
        "clinical_utility_endpoint": {"strict_cap_preservation_rate": utility},
        "joint_interpretation_policy": (
            "Report the privacy and utility effect vector with separate paired confidence intervals. "
            "Do not infer equivalence from a nonsignificant privacy difference and do not collapse the vector into an unvalidated scalar."
        ),
        "input_hashes": {
            "f5_cap": sha256(f5_cap_path), "stock_cap": sha256(stock_cap_path),
            "f5_ecapa": sha256(f5_ecapa_path), "stock_ecapa": sha256(stock_ecapa_path),
            "f5_xvector": sha256(f5_xvector_path), "stock_xvector": sha256(stock_xvector_path),
        },
    }


def orientation_robust_asv_estimand(path: Path) -> dict[str, Any]:
    """Load frozen attacker-oriented EER evidence without reinterpreting FAR1."""
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "aegisvox.orientation_robust_asv_summary.v1" or payload.get("status") != "complete":
        raise ValueError("orientation-robust ASV summary is incomplete or has the wrong schema")
    systems = payload.get("systems")
    required = {
        "aegisvox_f5/ecapa", "aegisvox_f5/xvector",
        "stock_voicecraft/ecapa", "stock_voicecraft/xvector",
    }
    if not isinstance(systems, dict) or set(systems) != required:
        raise ValueError("orientation-robust ASV summary must contain the exact four frozen system/backend cells")
    for key, row in systems.items():
        if row.get("num_source_recordings") != 390 or row.get("num_source_speakers") != 809:
            raise ValueError(f"orientation-robust ASV population mismatch for {key}")
        value = row.get("attacker_optimal_eer")
        ci = row.get("attacker_optimal_eer_ci95")
        if not isinstance(value, (int, float)) or not isinstance(ci, list) or len(ci) != 2:
            raise ValueError(f"missing attacker-optimal EER evidence for {key}")
        if not (0 <= float(ci[0]) <= float(value) <= float(ci[1]) <= 0.5):
            raise ValueError(f"invalid attacker-optimal EER/CI for {key}")
    return {
        "estimand": "attacker-optimal ASV EER under score-orientation choice",
        "systems": systems,
        "headline_metric": payload.get("headline_metric"),
        "orientation_policy": payload.get("orientation_policy"),
        "far1_boundary": payload.get("far1_boundary"),
        "claim_boundary": payload.get("claim_boundary"),
        "input": {"path": str(path), "sha256": sha256(path)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f5-cap", type=Path, required=True)
    ap.add_argument("--stock-cap", type=Path, required=True)
    ap.add_argument("--f5-relex", type=Path, required=True)
    ap.add_argument("--stock-relex", type=Path, required=True)
    ap.add_argument("--f5-ecapa", type=Path, required=True)
    ap.add_argument("--stock-ecapa", type=Path, required=True)
    ap.add_argument("--f5-xvector", type=Path, required=True)
    ap.add_argument("--stock-xvector", type=Path, required=True)
    ap.add_argument("--orientation-robust-asv", type=Path, required=True)
    ap.add_argument("--expected-conversations", type=int, default=390)
    ap.add_argument("--expected-edits", type=int, default=601)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.iterations < 1000:
        raise ValueError("at least 1,000 resamples are required")
    report = {
        "schema_version": "aegisvox.primary_model_estimands.v1", "status": "complete",
        "clinical_no_major_error": cap_estimand(
            args.f5_cap, args.stock_cap, expected_rows=args.expected_conversations,
            iterations=args.iterations, seed=args.seed,
        ),
        "replacement_realization": relex_estimand(
            args.f5_relex, args.stock_relex, expected_edits=args.expected_edits,
            iterations=args.iterations, seed=args.seed + 100003,
        ),
        "privacy_utility_tradeoff": privacy_utility_estimand(
            f5_cap_path=args.f5_cap, stock_cap_path=args.stock_cap,
            f5_ecapa_path=args.f5_ecapa, stock_ecapa_path=args.stock_ecapa,
            f5_xvector_path=args.f5_xvector, stock_xvector_path=args.stock_xvector,
            expected_rows=args.expected_conversations, iterations=args.iterations,
            seed=args.seed + 200003,
        ),
        "absolute_asv_privacy": orientation_robust_asv_estimand(args.orientation_robust_asv),
        "primary_family_status": "three_of_four_nonhuman_estimands_complete",
        "unresolved_primary_estimands": ["human_clinical_preservation"],
        "missingness_policy": "Intention-to-transform; uncertain judgments are failure in the primary analysis and success in the optimistic sensitivity bound; no row is silently dropped.",
        "multiplicity_policy": "Raw p-values only until all four frozen primary estimands are available; final family-wise correction is deferred to the hierarchical report. The privacy-utility vector is not collapsed into one p-value.",
        "deterministic_clinical_proxy_used": False,
        "claim_boundary": "Qualified GPT-5.4 model-adjudicated clinical endpoints plus measured ASV evidence; not human ground truth, not ASV equivalence evidence, and not a completed four-estimand primary family.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "complete",
        "cap_risk_difference": report["clinical_no_major_error"]["primary"]["risk_difference_f5_minus_stock"],
        "relex_risk_difference": report["replacement_realization"]["primary"]["risk_difference_f5_minus_stock"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
