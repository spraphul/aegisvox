#!/usr/bin/env python3
"""Aggregate frozen bidirectional Qwen3 CAP judgments against GPT-5.4 evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def judge_result(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("payload") or row)["result"]


def binary_status(value: str) -> str:
    if value == "preserved":
        return "preserved"
    if value == "uncertain":
        return "uncertain"
    return "not_preserved"


def weighted_rate(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float | None:
    denominator = sum(float(row["inverse_probability_weight"]) for row in rows)
    if not denominator:
        return None
    return sum(float(row["inverse_probability_weight"]) * bool(predicate(row)) for row in rows) / denominator


def kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if not labels_a:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    ca, cb = Counter(labels_a), Counter(labels_b)
    expected = sum((ca[k] / len(labels_a)) * (cb[k] / len(labels_b)) for k in set(ca) | set(cb))
    return (observed - expected) / (1 - expected) if expected < 1 else None


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    index = (len(values) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def exact_binomial_two_sided(a: int, b: int) -> float:
    n = a + b
    if not n:
        return 1.0
    smaller = min(a, b)
    probability = sum(math.comb(n, value) for value in range(smaller + 1)) / (2**n)
    return min(1.0, 2 * probability)


def cluster_bootstrap_rate(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], *, seed: int, draws: int,
) -> list[float]:
    by_audio: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_audio[row["audio_id"]].append(row)
    clusters = sorted(by_audio)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(draws):
        sampled = [row for audio in rng.choices(clusters, k=len(clusters)) for row in by_audio[audio]]
        value = weighted_rate(sampled, predicate)
        if value is not None:
            values.append(value)
    return values


def rate_with_ci(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], *, seed: int, draws: int,
) -> dict[str, Any]:
    value = weighted_rate(rows, predicate)
    boots = cluster_bootstrap_rate(rows, predicate, seed=seed, draws=draws)
    return {
        "numerator_unweighted": sum(bool(predicate(row)) for row in rows),
        "denominator_unweighted": len(rows),
        "estimate": value,
        "ci95": [percentile(boots, 0.025), percentile(boots, 0.975)] if boots else None,
    }


def summarize_condition(rows: list[dict[str, Any]], *, seed: int, draws: int) -> dict[str, Any]:
    covered = [row for row in rows if row["qwen_consensus_outcome"] != "uncertain"]
    gpt_binary = [binary_status(row["gpt_status"]) for row in covered]
    qwen_binary = [binary_status(row["qwen_consensus_outcome"]) for row in covered]
    confusion = Counter((row["gpt_status"], row["qwen_consensus_outcome"]) for row in covered)
    sf_pres_tf_missing = sum(
        row["qwen_source_first_outcome"] == "preserved" and row["qwen_target_first_outcome"] == "missing"
        for row in rows if row["source_first_valid"] and row["target_first_valid"]
    )
    sf_missing_tf_pres = sum(
        row["qwen_source_first_outcome"] == "missing" and row["qwen_target_first_outcome"] == "preserved"
        for row in rows if row["source_first_valid"] and row["target_first_valid"]
    )
    return {
        "num_cases": len(rows),
        "source_first_valid": sum(row["source_first_valid"] for row in rows),
        "target_first_valid": sum(row["target_first_valid"] for row in rows),
        "both_directions_valid": sum(row["source_first_valid"] and row["target_first_valid"] for row in rows),
        "num_bidirectional_consensus": len(covered),
        "bidirectional_consensus_coverage": len(covered) / len(rows) if rows else None,
        "outcome_counts_gpt54": dict(Counter(row["gpt_status"] for row in rows)),
        "outcome_counts_qwen_consensus": dict(Counter(row["qwen_consensus_outcome"] for row in rows)),
        "strict_preservation_qwen_conservative": rate_with_ci(
            rows, lambda row: row["qwen_consensus_outcome"] == "preserved", seed=seed, draws=draws,
        ),
        "strict_preservation_gpt54": rate_with_ci(
            rows, lambda row: row["gpt_status"] == "preserved", seed=seed + 1, draws=draws,
        ),
        "binary_agreement_covered": rate_with_ci(
            covered,
            lambda row: binary_status(row["gpt_status"]) == binary_status(row["qwen_consensus_outcome"]),
            seed=seed + 2,
            draws=draws,
        ),
        "exact_multiclass_agreement_covered": rate_with_ci(
            covered,
            lambda row: row["gpt_status"] == row["qwen_consensus_outcome"],
            seed=seed + 3,
            draws=draws,
        ),
        "binary_cohens_kappa_covered": kappa(gpt_binary, qwen_binary),
        "clinical_impact_exact_agreement_covered": (
            sum(row["gpt_clinical_impact"] == row["qwen_consensus_clinical_impact"] for row in covered) / len(covered)
            if covered else None
        ),
        "multiclass_confusion_covered": {
            f"gpt={left}|qwen={right}": count for (left, right), count in sorted(confusion.items())
        },
        "order_asymmetry": {
            "source_first_preserved_target_first_missing": sf_pres_tf_missing,
            "source_first_missing_target_first_preserved": sf_missing_tf_pres,
            "exact_two_sided_p": exact_binomial_two_sided(sf_pres_tf_missing, sf_missing_tf_pres),
        },
    }


def paired_contrast(
    rows: list[dict[str, Any]], *, left: str, right: str, seed: int, draws: int, importance: str | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if importance is None or row["clinical_importance"] == importance]
    pairs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        pairs[(row["audio_id"], row["cap_id"])][row["condition"]] = row
    complete = {key: value for key, value in pairs.items() if left in value and right in value}

    def estimate(keys: list[tuple[str, str]]) -> float | None:
        numerator = denominator = 0.0
        for key in keys:
            lrow, rrow = complete[key][left], complete[key][right]
            weight = float(lrow["inverse_probability_weight"])
            if not math.isclose(weight, float(rrow["inverse_probability_weight"]), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"pair weight mismatch: {key}")
            numerator += weight * (
                float(rrow["qwen_consensus_outcome"] == "preserved")
                - float(lrow["qwen_consensus_outcome"] == "preserved")
            )
            denominator += weight
        return numerator / denominator if denominator else None

    keys = sorted(complete)
    by_audio: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in keys:
        by_audio[key[0]].append(key)
    audios = sorted(by_audio)
    rng = random.Random(seed)
    boots = []
    for _ in range(draws):
        sampled = [key for audio in rng.choices(audios, k=len(audios)) for key in by_audio[audio]]
        value = estimate(sampled)
        if value is not None:
            boots.append(value)
    return {
        "left_condition": left,
        "right_condition": right,
        "clinical_importance": importance or "all",
        "num_matched_caps": len(keys),
        "estimate_right_minus_left": estimate(keys),
        "ci95": [percentile(boots, 0.025), percentile(boots, 0.975)] if boots else None,
        "uncertain_or_invalid_counted_as_not_preserved": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-first", action="append", type=Path, required=True)
    parser.add_argument("--target-first", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    registry_rows = read_jsonl(args.registry)
    registry = {row["case_id"]: row for row in registry_rows}
    if len(registry) != 1800:
        raise SystemExit(f"expected 1800 registry cases, found {len(registry)}")
    source = {row["case_id"]: judge_result(row) for path in args.source_first for row in read_jsonl(path)}
    target = {row["case_id"]: judge_result(row) for path in args.target_first for row in read_jsonl(path)}
    unknown = (set(source) | set(target)) - set(registry)
    if unknown:
        raise SystemExit(f"judgments contain {len(unknown)} unknown case IDs")

    item_rows = []
    for case_id, base in registry.items():
        sf, tf = source.get(case_id), target.get(case_id)
        outcome_agreement = sf is not None and tf is not None and sf["outcome"] == tf["outcome"]
        impact_agreement = sf is not None and tf is not None and sf["clinical_impact"] == tf["clinical_impact"]
        item_rows.append({
            **base,
            "qwen_source_first_outcome": sf["outcome"] if sf else "uncertain",
            "qwen_target_first_outcome": tf["outcome"] if tf else "uncertain",
            "qwen_consensus_outcome": sf["outcome"] if outcome_agreement else "uncertain",
            "qwen_source_first_clinical_impact": sf["clinical_impact"] if sf else "uncertain",
            "qwen_target_first_clinical_impact": tf["clinical_impact"] if tf else "uncertain",
            "qwen_consensus_clinical_impact": sf["clinical_impact"] if outcome_agreement and impact_agreement else "uncertain",
            "source_first_valid": sf is not None,
            "target_first_valid": tf is not None,
            "bidirectional_outcome_agreement": outcome_agreement,
            "bidirectional_impact_agreement": impact_agreement,
            "source_first_evidence_ids": {"source": sf["source_evidence_ids"], "target": sf["target_evidence_ids"]} if sf else None,
            "target_first_evidence_ids": {"source": tf["source_evidence_ids"], "target": tf["target_evidence_ids"]} if tf else None,
        })

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in item_rows:
        by_condition[row["condition"]].append(row)
    condition_reports = {
        condition: summarize_condition(rows, seed=args.seed + index * 10, draws=args.bootstrap_draws)
        for index, (condition, rows) in enumerate(sorted(by_condition.items()))
    }
    contrasts = {
        "all": paired_contrast(
            item_rows, left="stock_voicecraft", right="aegisvox_f5", seed=args.seed + 100, draws=args.bootstrap_draws,
        ),
        **{
            importance: paired_contrast(
                item_rows,
                left="stock_voicecraft",
                right="aegisvox_f5",
                importance=importance,
                seed=args.seed + 101 + index,
                draws=args.bootstrap_draws,
            )
            for index, importance in enumerate(("critical", "major", "minor"))
        },
    }
    report = {
        "schema_version": "aegisvox.qwen3_32b_cap_cross_model_sensitivity.v1",
        "status": "complete",
        "model": args.model,
        "num_unique_source_caps": len({(row["audio_id"], row["cap_id"]) for row in item_rows}),
        "num_condition_cases": len(item_rows),
        "num_source_first_valid": len(source),
        "num_target_first_valid": len(target),
        "num_source_first_fail_closed": len(registry) - len(source),
        "num_target_first_fail_closed": len(registry) - len(target),
        "conditions": condition_reports,
        "paired_f5_minus_stock": contrasts,
        "decision_rule": (
            "Qwen outcome is accepted only when source-first and target-first outcomes agree; disagreement or "
            "invalid/missing output is uncertain and counts as not preserved in conservative preservation estimands."
        ),
        "sampling_design": "frozen equal allocation by public dataset and CAP importance with inverse-stratum weighting",
        "uncertainty": f"95% percentile intervals from {args.bootstrap_draws} audio-cluster bootstrap draws",
        "model_role": "independent open-weight corroboration/sensitivity judge; not human ground truth",
        "claim_boundary": (
            "GPT-5.4 and Qwen3-32B are model judges. Agreement supports cross-model robustness; disagreement remains "
            "visible and neither model is treated as ground truth."
        ),
        "evidence_sha256": {
            "registry": sha256(args.registry),
            **{f"source_first_{index}": sha256(path) for index, path in enumerate(args.source_first)},
            **{f"target_first_{index}": sha256(path) for index, path in enumerate(args.target_first)},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_path = args.output_dir / "item_level_results.jsonl"
    items_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in item_rows))
    report["item_level_results_sha256"] = sha256(items_path)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Qwen3-32B Independent CAP Corroboration",
        "",
        f"Frozen source CAPs: {report['num_unique_source_caps']}; condition cases: {len(item_rows)}.",
        "Qwen3-32B is an independent open-weight sensitivity judge, not ground truth.",
        "",
        "| Condition | SF valid | TF valid | Consensus | GPT strict | Qwen strict | Binary agreement | Kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, value in sorted(condition_reports.items()):
        lines.append(
            f"| {condition} | {value['source_first_valid']}/600 | {value['target_first_valid']}/600 | "
            f"{value['num_bidirectional_consensus']}/600 | "
            f"{value['strict_preservation_gpt54']['estimate']:.3f} | "
            f"{value['strict_preservation_qwen_conservative']['estimate']:.3f} | "
            f"{value['binary_agreement_covered']['estimate']:.3f} | "
            f"{value['binary_cohens_kappa_covered'] if value['binary_cohens_kappa_covered'] is not None else 'N/A'} |"
        )
    lines.extend(["", "## Paired AegisVox-F5 minus stock VoiceCraft contrast", ""])
    for stratum, value in contrasts.items():
        ci = value["ci95"]
        lines.append(
            f"- {stratum}: {value['estimate_right_minus_left']:.4f} "
            f"(95% CI {ci[0]:.4f} to {ci[1]:.4f}; {value['num_matched_caps']} matched CAPs)."
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    checksum_paths = [items_path, report_path, args.output_dir / "report.md"]
    (args.output_dir / "SHA256SUMS.portable").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
