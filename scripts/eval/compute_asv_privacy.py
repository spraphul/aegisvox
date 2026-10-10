#!/usr/bin/env python3
"""Compute ASV/privacy/linkability metrics for AegisVox eval runs.

This script aggregates per-run `speaker_privacy_metrics.json` artifacts and can
optionally compute missing artifacts with the existing AegisVox
SpeakerPrivacyMetricsService. It then builds same-source and impostor trials from
source/converted embeddings to report protocol EER and calibrated linkability.

Claim boundary: this is a stated ECAPA ASV trial protocol, not the official
VoicePrivacy challenge protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

from common import distribution_summary, row_completed, load_json, load_summary_rows, mean, write_csv, write_json
from asv_uncertainty import bootstrap_asv_uncertainty

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aegisvox.storage import LocalArtifactStore  # noqa: E402
from aegisvox.voice_pool import cosine_similarity  # noqa: E402


def uri(run_id: str, name: str) -> str:
    return f"s3://aegisvox/runs/{run_id}/{name}"


def compute_missing_privacy_metrics(row: dict[str, Any], artifact_root: Path) -> dict[str, Any] | None:
    # Import model/audio dependencies only when embeddings must actually be
    # computed. This keeps report aggregation usable in lightweight Python
    # environments (including Python 3.13, where stdlib audioop was removed).
    from aegisvox.privacy_metrics import SpeakerPrivacyMetricsService
    from aegisvox.speaker_profile import SpeechBrainECAPAEmbeddingBackend

    run_id = str(row.get("run_id") or "")
    if not run_id:
        return None
    store = LocalArtifactStore(artifact_root)
    output_uri = uri(run_id, "speaker_privacy_metrics.json")
    output_path = store.local_path(output_uri)
    if output_path.exists():
        payload = load_json(output_path, {})
        return payload if isinstance(payload, dict) else None
    converted_uri = uri(run_id, "segments.converted.direct_input_v1.json")
    if not store.exists(converted_uri):
        converted_uri = uri(run_id, "segments.converted.json")
    try:
        return SpeakerPrivacyMetricsService(
            store,
            embedding_backend=SpeechBrainECAPAEmbeddingBackend(),
        ).compute_to_artifact(
            speaker_profiles_uri=uri(run_id, "speaker_profiles.json"),
            assignments_uri=uri(run_id, "speaker_target_map.json"),
            converted_segments_uri=converted_uri,
            output_metrics_uri=output_uri,
            output_prefix_uri=uri(run_id, "privacy_metrics"),
        )
    except Exception as exc:  # noqa: BLE001 - per-row metric failures become row status, not whole eval failure
        return {
            "service": "speaker_privacy_metrics",
            "run_id": run_id,
            "status": "compute_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "speaker_metrics": [],
            "summary": {},
        }


def load_privacy_metrics(row: dict[str, Any], artifact_root: Path, compute_missing: bool) -> dict[str, Any] | None:
    run_id = str(row.get("run_id") or "")
    if not run_id:
        return None
    store = LocalArtifactStore(artifact_root)
    path = store.local_path(uri(run_id, "speaker_privacy_metrics.json"))
    if path.exists():
        payload = load_json(path, {})
        return payload if isinstance(payload, dict) else None
    if compute_missing:
        return compute_missing_privacy_metrics(row, artifact_root)
    return None


def load_embedding_from_uri(artifact_root: Path, emb_uri: str | None) -> list[float] | None:
    if not emb_uri:
        return None
    store = LocalArtifactStore(artifact_root)
    try:
        payload = store.read_json(str(emb_uri))
    except Exception:
        return None
    emb = payload.get("embedding") if isinstance(payload, dict) else None
    if not isinstance(emb, list) or not emb:
        return None
    try:
        return [float(x) for x in emb]
    except Exception:
        return None


def compute_eer(positive_scores: list[float], negative_scores: list[float]) -> dict[str, Any]:
    if not positive_scores or not negative_scores:
        return {"eer": None, "eer_threshold": None, "num_positive_trials": len(positive_scores), "num_negative_trials": len(negative_scores)}
    import bisect

    pos = sorted(float(x) for x in positive_scores)
    neg = sorted(float(x) for x in negative_scores)
    thresholds = sorted(set(pos + neg))
    candidates = [thresholds[0] - 1e-6, *thresholds, thresholds[-1] + 1e-6]
    best = None
    npos = len(pos)
    nneg = len(neg)
    for th in candidates:
        # Positive accepts score >= threshold. FNR is positives below threshold.
        fn = bisect.bisect_left(pos, th)
        # Negative accepts score >= threshold. FPR is negatives at/above threshold.
        fp = nneg - bisect.bisect_left(neg, th)
        fnr = fn / npos
        fpr = fp / nneg
        diff = abs(fnr - fpr)
        eer = (fnr + fpr) / 2.0
        item = (diff, eer, th, fnr, fpr)
        if best is None or item < best:
            best = item
    assert best is not None
    _, eer, th, fnr, fpr = best
    return {
        "eer": eer,
        "eer_threshold": th,
        "eer_fnr_at_threshold": fnr,
        "eer_fpr_at_threshold": fpr,
        "num_positive_trials": len(positive_scores),
        "num_negative_trials": len(negative_scores),
    }


def threshold_at_far(negative_scores: list[float], far: float) -> float | None:
    if not negative_scores:
        return None
    scores = sorted(negative_scores)
    # Small-n conservative threshold: choose score such that empirical FAR is <= target where possible.
    allowed_fp = max(0, math.floor(far * len(scores)))
    if allowed_fp <= 0:
        return max(scores) + 1e-6
    return scores[-allowed_fp]


def row_metrics(summary_row: dict[str, Any], artifact_root: Path, compute_missing: bool) -> dict[str, Any]:
    privacy = load_privacy_metrics(summary_row, artifact_root, compute_missing=compute_missing)
    run_id = str(summary_row.get("run_id") or "")
    if not privacy:
        return {
            "audio_id": summary_row.get("audio_id"),
            "run_id": run_id,
            "completed": row_completed(summary_row),
            "privacy_metrics_available": False,
            "status": "missing_privacy_metrics",
            "claim_boundary": "ASV privacy metrics missing for this row.",
        }
    speaker_rows = [r for r in privacy.get("speaker_metrics") or [] if isinstance(r, dict)]
    computed = [r for r in speaker_rows if r.get("status") == "computed"]
    # Assignment collision/distinctiveness from assignment map.
    store = LocalArtifactStore(artifact_root)
    assignments_payload = {}
    try:
        assignments_payload = store.read_json(uri(run_id, "speaker_target_map.json"))
    except Exception:
        assignments_payload = {}
    assignments = assignments_payload.get("assignments") if isinstance(assignments_payload, dict) else {}
    target_ids = []
    source_target_distances = []
    if isinstance(assignments, dict):
        for assignment in assignments.values():
            if not isinstance(assignment, dict):
                continue
            target_ids.append(str(assignment.get("target_voice_sample_id") or assignment.get("target_voice_id") or assignment.get("target_reference_audio_uri") or ""))
            if isinstance(assignment.get("source_target_distance"), (int, float)):
                source_target_distances.append(float(assignment["source_target_distance"]))
    duplicates = len([x for x in target_ids if x]) - len(set(x for x in target_ids if x))
    return {
        "audio_id": summary_row.get("audio_id"),
        "run_id": run_id,
        "completed": row_completed(summary_row),
        "privacy_metrics_available": True,
        "status": privacy.get("status") or "available",
        "backend": privacy.get("backend"),
        "num_speakers": len(speaker_rows),
        "num_computed_speakers": len(computed),
        "num_missing_converted_audio": (privacy.get("summary") or {}).get("num_missing_converted_audio"),
        "max_source_converted_similarity": (privacy.get("summary") or {}).get("max_source_converted_similarity"),
        "min_target_converted_similarity": (privacy.get("summary") or {}).get("min_target_converted_similarity"),
        "num_source_linkability_flags": (privacy.get("summary") or {}).get("num_source_linkability_flags"),
        "num_target_similarity_flags": (privacy.get("summary") or {}).get("num_target_similarity_flags"),
        "source_converted_similarities": [r.get("source_converted_similarity") for r in computed],
        "target_converted_similarities": [r.get("target_converted_similarity") for r in computed],
        "source_embedding_uris": {r.get("speaker_id"): r.get("source_embedding_uri") for r in computed},
        "converted_embedding_uris": {r.get("speaker_id"): r.get("converted_embedding_uri") for r in computed},
        "duplicate_target_assignment_count": duplicates,
        "duplicate_target_assignment_rate": (duplicates / len(target_ids)) if target_ids else None,
        "min_source_target_assignment_distance": min(source_target_distances) if source_target_distances else None,
        "claim_boundary": "Measured ECAPA ASV trial result; not an official VoicePrivacy challenge result or anonymity guarantee.",
    }


def build_trials(rows: list[dict[str, Any]], artifact_root: Path) -> dict[str, Any]:
    source_keys: list[tuple[str, str]] = []
    source_vecs: list[list[float]] = []
    converted_keys: list[tuple[str, str]] = []
    converted_vecs: list[list[float]] = []
    for row in rows:
        run_id = str(row.get("run_id") or "")
        for spk, emb_uri in (row.get("source_embedding_uris") or {}).items():
            emb = load_embedding_from_uri(artifact_root, emb_uri)
            if emb is not None:
                source_keys.append((run_id, str(spk)))
                source_vecs.append(emb)
        for spk, emb_uri in (row.get("converted_embedding_uris") or {}).items():
            emb = load_embedding_from_uri(artifact_root, emb_uri)
            if emb is not None:
                converted_keys.append((run_id, str(spk)))
                converted_vecs.append(emb)
    if not source_vecs or not converted_vecs:
        eer = compute_eer([], [])
        return {
            "positive_scores": [],
            "negative_scores": [],
            "trial_rows": [],
            "source_trial_blocks": [],
            **eer,
            "far1_threshold": None,
            "same_source_accept_rate_at_far1": None,
        }

    try:
        import numpy as np

        src = np.asarray(source_vecs, dtype=np.float32)
        conv = np.asarray(converted_vecs, dtype=np.float32)
        src_norm = np.linalg.norm(src, axis=1, keepdims=True)
        conv_norm = np.linalg.norm(conv, axis=1, keepdims=True)
        src = src / np.maximum(src_norm, 1e-12)
        conv = conv / np.maximum(conv_norm, 1e-12)
        scores = src @ conv.T
        positive_scores: list[float] = []
        negative_scores: list[float] = []
        source_trial_blocks: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        conv_index = {key: idx for idx, key in enumerate(converted_keys)}
        for i, skey in enumerate(source_keys):
            pos_j = conv_index.get(skey)
            if pos_j is not None:
                positive_score = float(scores[i, pos_j])
                positive_scores.append(positive_score)
            # Keep all impostor scores for EER/FAR; this is paper-scale but still
            # small enough for full390 once vectorized.
            if pos_j is None:
                neg = scores[i, :]
            else:
                neg = np.concatenate([scores[i, :pos_j], scores[i, pos_j + 1 :]])
            negative_scores.extend(float(x) for x in neg.tolist())
            if pos_j is not None:
                source_trial_blocks.append({
                    "source_run_id": skey[0],
                    "source_speaker_id": skey[1],
                    "positive_score": positive_score,
                    "negative_scores": [float(x) for x in neg.tolist()],
                })
        # Trial CSV can be large but useful for audit; emit it after vectorized
        # scoring without recomputing cosine in Python.
        for i, skey in enumerate(source_keys):
            for j, ckey in enumerate(converted_keys):
                label = 1 if skey == ckey else 0
                trial_rows.append({
                    "source_run_id": skey[0],
                    "source_speaker_id": skey[1],
                    "converted_run_id": ckey[0],
                    "converted_speaker_id": ckey[1],
                    "same_source_speaker_label": label,
                    "similarity": float(scores[i, j]),
                })
    except Exception:
        positive_scores = []
        negative_scores = []
        trial_rows = []
        source_trial_blocks_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for source_key, source_emb in zip(source_keys, source_vecs):
            for conv_key, conv_emb in zip(converted_keys, converted_vecs):
                score = cosine_similarity(source_emb, conv_emb)
                label = 1 if source_key == conv_key else 0
                if label:
                    positive_scores.append(score)
                    block = source_trial_blocks_by_key.setdefault(source_key, {
                        "source_run_id": source_key[0],
                        "source_speaker_id": source_key[1],
                        "positive_score": None,
                        "negative_scores": [],
                    })
                    block["positive_score"] = score
                else:
                    negative_scores.append(score)
                    source_trial_blocks_by_key.setdefault(source_key, {
                        "source_run_id": source_key[0],
                        "source_speaker_id": source_key[1],
                        "positive_score": None,
                        "negative_scores": [],
                    })["negative_scores"].append(score)
                trial_rows.append({
                    "source_run_id": source_key[0],
                    "source_speaker_id": source_key[1],
                    "converted_run_id": conv_key[0],
                    "converted_speaker_id": conv_key[1],
                    "same_source_speaker_label": label,
                    "similarity": score,
                })
        source_trial_blocks = list(source_trial_blocks_by_key.values())
    eer = compute_eer(positive_scores, negative_scores)
    far1_threshold = threshold_at_far(negative_scores, 0.01)
    linkability = (sum(1 for s in positive_scores if far1_threshold is not None and s >= far1_threshold) / len(positive_scores)) if positive_scores else None
    uncertainty = bootstrap_asv_uncertainty(
        source_trial_blocks,
        compute_eer=compute_eer,
        threshold_at_far=threshold_at_far,
    )
    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "trial_rows": trial_rows,
        "source_trial_blocks": source_trial_blocks,
        **eer,
        "far1_threshold": far1_threshold,
        "same_source_accept_rate_at_far1": linkability,
        "uncertainty": uncertainty,
    }


def aggregate(rows: list[dict[str, Any]], trials: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_completed": sum(1 for r in rows if r.get("completed")),
        "num_rows_with_privacy_metrics": sum(1 for r in rows if r.get("privacy_metrics_available")),
        "num_computed_speakers": sum(int(r.get("num_computed_speakers") or 0) for r in rows),
        "mean_max_source_converted_similarity": mean([r.get("max_source_converted_similarity") for r in rows]),
        "mean_min_target_converted_similarity": mean([r.get("min_target_converted_similarity") for r in rows]),
        "total_source_linkability_flags": sum(int(r.get("num_source_linkability_flags") or 0) for r in rows),
        "total_target_similarity_flags": sum(int(r.get("num_target_similarity_flags") or 0) for r in rows),
        "total_duplicate_target_assignments": sum(int(r.get("duplicate_target_assignment_count") or 0) for r in rows),
        "mean_duplicate_target_assignment_rate": mean([r.get("duplicate_target_assignment_rate") for r in rows]),
        "asv_eer": trials.get("eer"),
        "asv_eer_threshold": trials.get("eer_threshold"),
        "source_linkability_at_far1": trials.get("same_source_accept_rate_at_far1"),
        "far1_threshold": trials.get("far1_threshold"),
        "num_positive_trials": trials.get("num_positive_trials"),
        "num_negative_trials": trials.get("num_negative_trials"),
        "asv_eer_ci95_low": (trials.get("uncertainty") or {}).get("asv_eer_ci95_low"),
        "asv_eer_ci95_high": (trials.get("uncertainty") or {}).get("asv_eer_ci95_high"),
        "source_linkability_at_far1_ci95_low": (trials.get("uncertainty") or {}).get("source_linkability_at_far1_ci95_low"),
        "source_linkability_at_far1_ci95_high": (trials.get("uncertainty") or {}).get("source_linkability_at_far1_ci95_high"),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# ASV Privacy / Linkability Metrics", "", "**Protocol boundary:** ECAPA same-source versus cross-speaker trials; not the official VoicePrivacy challenge protocol.", "", "## Aggregate", ""]
    for key, value in payload["aggregate"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Rows", "", "| audio_id | speakers | max src-conv sim | min tgt-conv sim | source flags | target flags | duplicate targets |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        lines.append(f"| {row.get('audio_id')} | {row.get('num_computed_speakers')} | {_fmt(row.get('max_source_converted_similarity'))} | {_fmt(row.get('min_target_converted_similarity'))} | {row.get('num_source_linkability_flags')} | {row.get('num_target_similarity_flags')} | {row.get('duplicate_target_assignment_count')} |")
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.3f}" if isinstance(value, (int, float)) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-trials-csv", type=Path, required=True)
    parser.add_argument("--compute-missing", action="store_true")
    args = parser.parse_args()
    rows = [row_metrics(row, args.artifact_root, compute_missing=args.compute_missing) for row in load_summary_rows(args.run_summary)]
    trials = build_trials(rows, args.artifact_root)
    backend_values = sorted({str(row.get("backend")) for row in rows if row.get("backend")})
    payload = {
        "schema_version": "aegisvox.eval.asv_privacy.v2",
        "run_summary": str(args.run_summary),
        "artifact_root": str(args.artifact_root),
        "compute_missing": args.compute_missing,
        "backend": backend_values[0] if len(backend_values) == 1 else backend_values,
        "aggregate": aggregate(rows, trials),
        "rows": rows,
        "trials_summary": {k: v for k, v in trials.items() if k not in {"trial_rows", "positive_scores", "negative_scores", "source_trial_blocks"}},
        "score_distributions": {
            "source_to_output_same_speaker": distribution_summary(trials.get("positive_scores") or []),
            "source_to_output_impostor": distribution_summary(trials.get("negative_scores") or []),
            "target_to_output_assigned_target": distribution_summary([
                value for row in rows for value in (row.get("target_converted_similarities") or [])
            ]),
        },
        "protocol": "ECAPA same-source source-to-converted positives and cross-speaker source-to-converted impostors; cosine scoring; empirical EER and source linkability at FAR=1%.",
        "claim_boundary": "Measured ASV performance under the stated ECAPA trial protocol; not an official VoicePrivacy challenge result or a guarantee of speaker anonymity.",
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, [{k: v for k, v in row.items() if k not in {"source_embedding_uris", "converted_embedding_uris", "source_converted_similarities", "target_converted_similarities"}} for row in rows])
    write_csv(args.output_trials_csv, trials.get("trial_rows") or [])
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
