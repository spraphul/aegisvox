#!/usr/bin/env python3
"""Compute independent ASV/linkability metrics from audio, not cached ECAPA embeddings.

This is the M6 strengthening evaluator: it recomputes source, converted, and
pseudo-target speaker embeddings with a caller-selected backend so the paper is
not relying on a single ECAPA metric stack.

Supported backends:
- speechbrain-xvector: SpeechBrain x-vector VoxCeleb model.
- speechbrain-ecapa: SpeechBrain ECAPA VoxCeleb model, useful for sanity reruns.
- hf-wavlm-style: WavLM pooled hidden-state embedding. This is an independent
  embedding-linkability stress test, not a canonical ASV model.

Claim boundary: a stated independent ASV trial protocol rather than the official
VoicePrivacy challenge protocol. It is independent of the primary cached ECAPA
scorer when using speechbrain-xvector or hf-wavlm-style.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Protocol

from common import distribution_summary, load_summary_rows, mean, row_completed, write_csv, write_json
from asv_uncertainty import bootstrap_asv_uncertainty

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aegisvox.schema import AegisVoxContractError, validate_artifact_uri  # noqa: E402
from aegisvox.speaker_profile import SpeechBrainECAPAEmbeddingBackend, concatenate_wav_segments  # noqa: E402
from aegisvox.storage import LocalArtifactStore  # noqa: E402
from aegisvox.voice_pool import cosine_similarity  # noqa: E402
from aegisvox.voice_quality_profile import HFWavLMStyleEmbeddingBackend  # noqa: E402


class EmbeddingBackend(Protocol):
    backend_name: str
    backend_version: str

    def embed_wav(self, wav_path: Path) -> list[float]: ...


def uri(run_id: str, name: str) -> str:
    return f"s3://aegisvox/runs/{run_id}/{name}"


def safe_slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "backend"


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_json_uri(store: LocalArtifactStore, artifact_uri: str) -> dict[str, Any]:
    validate_artifact_uri(artifact_uri)
    payload = store.read_json(artifact_uri)
    if not isinstance(payload, dict):
        raise AegisVoxContractError(f"expected JSON object at {artifact_uri}")
    return payload


def converted_rows_by_speaker(converted_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = converted_payload.get("segments")
    if not isinstance(rows, list):
        raise AegisVoxContractError("converted segments artifact must contain segments list")
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("conversion_status") != "converted":
            continue
        speaker_id = row.get("speaker_id")
        audio_uri = row.get("converted_audio_uri")
        if not speaker_id or not audio_uri:
            raise AegisVoxContractError("converted row missing speaker_id/converted_audio_uri")
        by_speaker.setdefault(str(speaker_id), []).append(dict(row))
    return by_speaker


def make_backend(args: argparse.Namespace) -> EmbeddingBackend:
    if args.backend == "speechbrain-xvector":
        backend = SpeechBrainECAPAEmbeddingBackend(
            model_id=args.speechbrain_model_id or "speechbrain/spkrec-xvect-voxceleb",
            savedir=args.speechbrain_savedir,
            device=args.device,
        )
        backend.backend_name = "speechbrain-xvector"  # type: ignore[attr-defined]
        backend.backend_version = f"{backend.model_id};device={backend._configured_device() or 'default'}"  # type: ignore[attr-defined]
        return backend
    if args.backend == "speechbrain-ecapa":
        return SpeechBrainECAPAEmbeddingBackend(
            model_id=args.speechbrain_model_id or "speechbrain/spkrec-ecapa-voxceleb",
            savedir=args.speechbrain_savedir,
            device=args.device,
        )
    if args.backend == "hf-wavlm-style":
        return HFWavLMStyleEmbeddingBackend(
            model_id=args.hf_model_id or "microsoft/wavlm-large",
            device=args.device or "auto",
            max_seconds=args.max_seconds,
        )
    raise ValueError(f"unsupported backend: {args.backend}")


def ensure_embedded(store: LocalArtifactStore, backend: EmbeddingBackend, audio_uri: str, out_uri: str) -> tuple[list[float] | None, str | None]:
    out_path = store.local_path(out_uri)
    if out_path.exists():
        payload = load_json(out_path, {})
        emb = payload.get("embedding") if isinstance(payload, dict) else None
        if isinstance(emb, list) and emb:
            return [float(x) for x in emb], None
    audio_path = store.local_path(audio_uri)
    if not audio_path.exists():
        return None, f"audio_missing:{audio_path}"
    try:
        emb = backend.embed_wav(audio_path)
    except Exception as exc:  # noqa: BLE001
        return None, f"embed_failed:{type(exc).__name__}:{exc}"
    store.write_json(out_uri, {"embedding": emb, "backend": backend.backend_name, "backend_version": backend.backend_version, "audio_uri": audio_uri})
    return emb, None


def row_metric(row: dict[str, Any], artifact_root: Path, backend: EmbeddingBackend, metric_slug: str) -> dict[str, Any]:
    run_id = str(row.get("run_id") or "")
    store = LocalArtifactStore(artifact_root)
    base = f"privacy_metrics_{metric_slug}"
    result: dict[str, Any] = {
        "audio_id": row.get("audio_id"),
        "run_id": run_id,
        "completed": row_completed(row),
        "backend": backend.backend_name,
        "backend_version": backend.backend_version,
        "status": "missing_run_id" if not run_id else "computed",
        "speaker_metrics": [],
        "source_embedding_uris": {},
        "converted_embedding_uris": {},
    }
    if not run_id:
        return result
    try:
        profiles_payload = load_json_uri(store, uri(run_id, "speaker_profiles.json"))
        assignments_payload = load_json_uri(store, uri(run_id, "speaker_target_map.json"))
        converted_uri = uri(run_id, "segments.converted.direct_input_v1.json")
        if not store.exists(converted_uri):
            converted_uri = uri(run_id, "segments.converted.json")
        converted_payload = load_json_uri(store, converted_uri)
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "artifact_load_failed", "error_type": type(exc).__name__, "error": str(exc)})
        return result

    profiles = profiles_payload.get("speakers") if isinstance(profiles_payload, dict) else None
    assignments = assignments_payload.get("assignments") if isinstance(assignments_payload, dict) else None
    if not isinstance(profiles, dict) or not isinstance(assignments, dict):
        result.update({"status": "bad_artifact_schema"})
        return result
    by_speaker = converted_rows_by_speaker(converted_payload)
    source_sims: list[float] = []
    target_sims: list[float] = []
    for speaker_id, profile in sorted(profiles.items()):
        if not isinstance(profile, dict):
            continue
        speaker_row: dict[str, Any] = {"speaker_id": speaker_id, "status": "computed"}
        assignment = assignments.get(speaker_id) if isinstance(assignments, dict) else None
        source_audio_uri = str(profile.get("source_audio_uri") or "")
        target_audio_uri = str((assignment or {}).get("target_reference_audio_uri") or "") if isinstance(assignment, dict) else ""
        conv_rows = by_speaker.get(str(speaker_id), [])
        if not source_audio_uri or not target_audio_uri or not conv_rows:
            speaker_row.update({"status": "missing_required_audio_uri", "num_converted_segments": len(conv_rows)})
            result["speaker_metrics"].append(speaker_row)
            continue
        converted_audio_uri = uri(run_id, f"{base}/speaker_converted_audio/{speaker_id}.wav")
        converted_path = store.local_path(converted_audio_uri)
        try:
            if not converted_path.exists():
                converted_paths = [store.local_path(str(r["converted_audio_uri"])) for r in conv_rows]
                missing = [str(p) for p in converted_paths if not p.exists()]
                if missing:
                    raise AegisVoxContractError(f"missing converted segments: {missing[:3]}")
                duration = concatenate_wav_segments(converted_paths, converted_path)
            else:
                duration = None
        except Exception as exc:  # noqa: BLE001
            speaker_row.update({"status": "converted_concat_failed", "error_type": type(exc).__name__, "error": str(exc)})
            result["speaker_metrics"].append(speaker_row)
            continue

        source_emb_uri = uri(run_id, f"{base}/source_embeddings/{speaker_id}.json")
        converted_emb_uri = uri(run_id, f"{base}/converted_embeddings/{speaker_id}.json")
        target_emb_uri = uri(run_id, f"{base}/target_embeddings/{speaker_id}.json")
        source_emb, source_err = ensure_embedded(store, backend, source_audio_uri, source_emb_uri)
        converted_emb, converted_err = ensure_embedded(store, backend, converted_audio_uri, converted_emb_uri)
        target_emb, target_err = ensure_embedded(store, backend, target_audio_uri, target_emb_uri)
        if source_emb is None or converted_emb is None:
            speaker_row.update({"status": "embedding_failed", "source_error": source_err, "converted_error": converted_err, "target_error": target_err})
            result["speaker_metrics"].append(speaker_row)
            continue
        src_conv = cosine_similarity(source_emb, converted_emb)
        tgt_conv = cosine_similarity(target_emb, converted_emb) if target_emb is not None else None
        source_sims.append(src_conv)
        if tgt_conv is not None:
            target_sims.append(tgt_conv)
        speaker_row.update(
            {
                "num_converted_segments": len(conv_rows),
                "converted_duration_sec": duration,
                "source_audio_uri": source_audio_uri,
                "converted_audio_uri": converted_audio_uri,
                "target_reference_audio_uri": target_audio_uri,
                "source_embedding_uri": source_emb_uri,
                "converted_embedding_uri": converted_emb_uri,
                "target_embedding_uri": target_emb_uri if target_emb is not None else None,
                "source_converted_similarity": round(src_conv, 6),
                "target_converted_similarity": round(tgt_conv, 6) if tgt_conv is not None else None,
            }
        )
        result["source_embedding_uris"][speaker_id] = source_emb_uri
        result["converted_embedding_uris"][speaker_id] = converted_emb_uri
        result["speaker_metrics"].append(speaker_row)
    computed = [r for r in result["speaker_metrics"] if r.get("status") == "computed"]
    result.update(
        {
            "num_speakers": len(result["speaker_metrics"]),
            "num_computed_speakers": len(computed),
            "max_source_converted_similarity": max(source_sims) if source_sims else None,
            "min_target_converted_similarity": min(target_sims) if target_sims else None,
            "source_converted_similarities": source_sims,
            "target_converted_similarities": target_sims,
            "privacy_metrics_available": bool(computed),
        }
    )
    return result


def compute_eer(pos_scores: list[float], neg_scores: list[float]) -> dict[str, Any]:
    if not pos_scores or not neg_scores:
        return {"eer": None, "eer_threshold": None, "num_positive_trials": len(pos_scores), "num_negative_trials": len(neg_scores)}
    import bisect

    pos = sorted(pos_scores)
    neg = sorted(neg_scores)
    thresholds = sorted(set(pos + neg))
    candidates = [thresholds[0] - 1e-6, *thresholds, thresholds[-1] + 1e-6]
    best = None
    for th in candidates:
        fnr = bisect.bisect_left(pos, th) / len(pos)
        fpr = (len(neg) - bisect.bisect_left(neg, th)) / len(neg)
        item = (abs(fnr - fpr), (fnr + fpr) / 2.0, th, fnr, fpr)
        if best is None or item < best:
            best = item
    assert best is not None
    _, eer, th, fnr, fpr = best
    return {"eer": eer, "eer_threshold": th, "eer_fnr_at_threshold": fnr, "eer_fpr_at_threshold": fpr, "num_positive_trials": len(pos_scores), "num_negative_trials": len(neg_scores)}


def threshold_at_far(negative_scores: list[float], far: float) -> float | None:
    if not negative_scores:
        return None
    scores = sorted(float(x) for x in negative_scores)
    allowed_fp = max(0, math.floor(far * len(scores)))
    if allowed_fp <= 0:
        return max(scores) + 1e-6
    return scores[-allowed_fp]


def build_trials(rows: list[dict[str, Any]], artifact_root: Path) -> dict[str, Any]:
    store = LocalArtifactStore(artifact_root)
    source_keys: list[tuple[str, str]] = []
    source_vecs: list[list[float]] = []
    converted_keys: list[tuple[str, str]] = []
    converted_vecs: list[list[float]] = []
    for row in rows:
        run_id = str(row.get("run_id") or "")
        for spk, emb_uri in (row.get("source_embedding_uris") or {}).items():
            payload = load_json(store.local_path(str(emb_uri)), {})
            emb = payload.get("embedding") if isinstance(payload, dict) else None
            if isinstance(emb, list) and emb:
                source_keys.append((run_id, str(spk)))
                source_vecs.append([float(x) for x in emb])
        for spk, emb_uri in (row.get("converted_embedding_uris") or {}).items():
            payload = load_json(store.local_path(str(emb_uri)), {})
            emb = payload.get("embedding") if isinstance(payload, dict) else None
            if isinstance(emb, list) and emb:
                converted_keys.append((run_id, str(spk)))
                converted_vecs.append([float(x) for x in emb])
    if not source_vecs or not converted_vecs:
        return {"trial_rows": [], "positive_scores": [], "negative_scores": [], "source_trial_blocks": [], **compute_eer([], []), "far1_threshold": None, "same_source_accept_rate_at_far1": None, "uncertainty": bootstrap_asv_uncertainty([], compute_eer=compute_eer, threshold_at_far=threshold_at_far)}
    import numpy as np

    src = np.asarray(source_vecs, dtype=np.float32)
    conv = np.asarray(converted_vecs, dtype=np.float32)
    src = src / np.maximum(np.linalg.norm(src, axis=1, keepdims=True), 1e-12)
    conv = conv / np.maximum(np.linalg.norm(conv, axis=1, keepdims=True), 1e-12)
    scores = src @ conv.T
    conv_index = {key: idx for idx, key in enumerate(converted_keys)}
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    source_trial_blocks: list[dict[str, Any]] = []
    for i, skey in enumerate(source_keys):
        pos_j = conv_index.get(skey)
        if pos_j is not None:
            positive_score = float(scores[i, pos_j])
            positive_scores.append(positive_score)
            neg = np.concatenate([scores[i, :pos_j], scores[i, pos_j + 1 :]])
        else:
            neg = scores[i, :]
        negative_scores.extend(float(x) for x in neg.tolist())
        if pos_j is not None:
            source_trial_blocks.append({
                "source_run_id": skey[0],
                "source_speaker_id": skey[1],
                "positive_score": positive_score,
                "negative_scores": [float(x) for x in neg.tolist()],
            })
    trial_rows = []
    for i, skey in enumerate(source_keys):
        for j, ckey in enumerate(converted_keys):
            trial_rows.append(
                {
                    "source_run_id": skey[0],
                    "source_speaker_id": skey[1],
                    "converted_run_id": ckey[0],
                    "converted_speaker_id": ckey[1],
                    "same_source_speaker_label": 1 if skey == ckey else 0,
                    "similarity": float(scores[i, j]),
                }
            )
    eer = compute_eer(positive_scores, negative_scores)
    far1 = threshold_at_far(negative_scores, 0.01)
    link = (sum(1 for s in positive_scores if far1 is not None and s >= far1) / len(positive_scores)) if positive_scores else None
    uncertainty = bootstrap_asv_uncertainty(source_trial_blocks, compute_eer=compute_eer, threshold_at_far=threshold_at_far)
    return {"trial_rows": trial_rows, "positive_scores": positive_scores, "negative_scores": negative_scores, "source_trial_blocks": source_trial_blocks, **eer, "far1_threshold": far1, "same_source_accept_rate_at_far1": link, "uncertainty": uncertainty}


def aggregate(rows: list[dict[str, Any]], trials: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_completed": sum(1 for r in rows if r.get("completed")),
        "num_rows_with_privacy_metrics": sum(1 for r in rows if r.get("privacy_metrics_available")),
        "num_computed_speakers": sum(int(r.get("num_computed_speakers") or 0) for r in rows),
        "mean_max_source_converted_similarity": mean([r.get("max_source_converted_similarity") for r in rows]),
        "mean_min_target_converted_similarity": mean([r.get("min_target_converted_similarity") for r in rows]),
        "asv_eer": trials.get("eer"),
        "asv_eer_threshold": trials.get("eer_threshold"),
        "far1_threshold": trials.get("far1_threshold"),
        "source_linkability_at_far1": trials.get("same_source_accept_rate_at_far1"),
        "num_positive_trials": trials.get("num_positive_trials"),
        "num_negative_trials": trials.get("num_negative_trials"),
        "asv_eer_ci95_low": (trials.get("uncertainty") or {}).get("asv_eer_ci95_low"),
        "asv_eer_ci95_high": (trials.get("uncertainty") or {}).get("asv_eer_ci95_high"),
        "source_linkability_at_far1_ci95_low": (trials.get("uncertainty") or {}).get("source_linkability_at_far1_ci95_low"),
        "source_linkability_at_far1_ci95_high": (trials.get("uncertainty") or {}).get("source_linkability_at_far1_ci95_high"),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Independent ASV / Linkability Backend Metrics",
        "",
        f"Backend: `{payload['backend']}`",
        "",
        "**Protocol boundary:** independent same-source versus cross-speaker ASV trials; not the official VoicePrivacy challenge protocol.",
        "",
        "## Aggregate",
        "",
    ]
    for k, v in payload["aggregate"].items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Rows", "", "| audio_id | speakers | max source-converted sim | min target-converted sim | status |", "|---|---:|---:|---:|---|"]
    for row in payload["rows"]:
        def fmt(x: Any) -> str:
            return "" if x is None else f"{float(x):.3f}" if isinstance(x, (int, float)) else str(x)
        lines.append(f"| {row.get('audio_id')} | {row.get('num_computed_speakers')} | {fmt(row.get('max_source_converted_similarity'))} | {fmt(row.get('min_target_converted_similarity'))} | {row.get('status')} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-summary", type=Path, required=True)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--output-trials-csv", type=Path, required=True)
    ap.add_argument("--backend", choices=["speechbrain-xvector", "speechbrain-ecapa", "hf-wavlm-style"], default="speechbrain-xvector")
    ap.add_argument("--speechbrain-model-id", default=None)
    ap.add_argument("--speechbrain-savedir", default=None)
    ap.add_argument("--hf-model-id", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    args = ap.parse_args()
    backend = make_backend(args)
    slug = safe_slug(f"{backend.backend_name}_{backend.backend_version}")[:96]
    rows = [row_metric(row, args.artifact_root, backend, slug) for row in load_summary_rows(args.run_summary)]
    trials = build_trials(rows, args.artifact_root)
    payload = {
        "schema_version": "aegisvox.eval.asv_privacy_independent.v2",
        "run_summary": str(args.run_summary),
        "artifact_root": str(args.artifact_root),
        "backend": backend.backend_name,
        "backend_version": backend.backend_version,
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
        "protocol": "Independent same-source source-to-converted positives and cross-speaker source-to-converted impostors; cosine scoring; empirical EER and source linkability at FAR=1%.",
        "claim_boundary": "Measured ASV performance under the stated independent backend; not an official VoicePrivacy challenge result or a guarantee of speaker anonymity.",
    }
    write_json(args.output_json, payload)
    csv_rows = []
    for row in rows:
        csv_rows.append({k: v for k, v in row.items() if k not in {"speaker_metrics", "source_embedding_uris", "converted_embedding_uris", "source_converted_similarities", "target_converted_similarities"}})
    write_csv(args.output_csv, csv_rows)
    write_csv(args.output_trials_csv, trials["trial_rows"])
    write_markdown(args.output_md, payload)
    print(json.dumps({"output_json": str(args.output_json), "backend": backend.backend_name, **payload["aggregate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
