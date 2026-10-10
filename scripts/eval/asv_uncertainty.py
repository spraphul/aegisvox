"""Uncertainty helpers for AegisVox ASV trial protocols.

The inferential unit is the source-speaker trial block, not an individual
source/impostor score.  Each bootstrap draw resamples source speakers and then
draws a bounded impostor set within each selected block.  This avoids treating
the quadratic cross-speaker score matrix as independent observations while
keeping full390 evaluation tractable.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(x) for x in values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def bootstrap_asv_uncertainty(
    blocks: list[dict[str, Any]],
    *,
    compute_eer: Callable[[list[float], list[float]], dict[str, Any]],
    threshold_at_far: Callable[[list[float], float], float | None],
    far: float = 0.01,
    iterations: int = 1000,
    seed: int = 20260806,
    impostors_per_source: int = 20,
) -> dict[str, Any]:
    usable = [
        block for block in blocks
        if isinstance(block.get("positive_score"), (int, float))
        and isinstance(block.get("negative_scores"), list)
        and block.get("negative_scores")
    ]
    if not usable or iterations <= 0:
        return {
            "method": "source_speaker_cluster_bootstrap",
            "num_source_speaker_units": len(usable),
            "iterations": int(iterations),
            "seed": int(seed),
            "impostors_per_source_per_draw": int(impostors_per_source),
            "asv_eer_ci95_low": None,
            "asv_eer_ci95_high": None,
            "source_linkability_at_far1_ci95_low": None,
            "source_linkability_at_far1_ci95_high": None,
        }
    rng = random.Random(seed)
    n = len(usable)
    eer_draws: list[float] = []
    link_draws: list[float] = []
    for _ in range(iterations):
        selected = [usable[rng.randrange(n)] for _ in range(n)]
        positives = [float(block["positive_score"]) for block in selected]
        negatives: list[float] = []
        for block in selected:
            candidates = [float(x) for x in block["negative_scores"]]
            k = min(len(candidates), impostors_per_source)
            negatives.extend(candidates if k == len(candidates) else rng.sample(candidates, k))
        eer = compute_eer(positives, negatives).get("eer")
        threshold = threshold_at_far(negatives, far)
        link = (
            sum(score >= threshold for score in positives) / len(positives)
            if threshold is not None and positives
            else None
        )
        if isinstance(eer, (int, float)):
            eer_draws.append(float(eer))
        if isinstance(link, (int, float)):
            link_draws.append(float(link))
    return {
        "method": "source_speaker_cluster_bootstrap",
        "num_source_speaker_units": n,
        "iterations": int(iterations),
        "seed": int(seed),
        "impostors_per_source_per_draw": int(impostors_per_source),
        "far": float(far),
        "asv_eer_ci95_low": quantile(eer_draws, 0.025),
        "asv_eer_ci95_high": quantile(eer_draws, 0.975),
        "source_linkability_at_far1_ci95_low": quantile(link_draws, 0.025),
        "source_linkability_at_far1_ci95_high": quantile(link_draws, 0.975),
        "claim_boundary": (
            "Cluster bootstrap over source-speaker units with a bounded random impostor set per source; "
            "the interval applies to this stated ASV trial protocol, not the official VoicePrivacy protocol."
        ),
    }
