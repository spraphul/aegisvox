#!/usr/bin/env python3
"""Build matched-ASR paired transcripts from AegisVox run artifacts.

The main utility evaluation should compare ASR(original audio) with
ASR(final transformed audio) using the same recognizer/settings. This avoids
mixing transformation errors with public-reference transcript conventions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RUN_ID_RE = re.compile(r"^public_eval_full390_.+?_\d{4}_(?P<audio_id>(?:primock57_mixed|fareez|simsamu)_\d{4})(?:_|$)")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def infer_audio_id(run_dir: Path, reference: dict[str, Any], hypothesis: dict[str, Any]) -> str:
    for payload in [reference, hypothesis]:
        for key in ["audio_id", "source_audio_id"]:
            if payload.get(key):
                return str(payload[key])
    m = RUN_ID_RE.match(run_dir.name)
    if m:
        return m.group("audio_id")
    parts = run_dir.name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[-4:-1]) if parts[-1].startswith("nooverlap") else run_dir.name
    return run_dir.name


def build_rows(
    *,
    runs_root: Path,
    run_glob: str,
    run_summary: Path | None = None,
    reference_asr_name: str,
    hypothesis_asr_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if run_summary is not None:
        payload = load_json(run_summary)
        run_ids = [str(row.get("run_id") or "") for row in payload.get("rows", []) if isinstance(row, dict)]
        run_dirs = [runs_root / run_id for run_id in run_ids if run_id]
    else:
        run_dirs = sorted(p for p in runs_root.glob(run_glob) if p.is_dir())
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            issues.append({"run_id": run_dir.name, "missing_run_dir": True})
            continue
        ref_path = run_dir / reference_asr_name
        hyp_path = run_dir / hypothesis_asr_name
        if not ref_path.exists() or not hyp_path.exists():
            issues.append(
                {
                    "run_id": run_dir.name,
                    "missing_reference_asr": not ref_path.exists(),
                    "missing_hypothesis_asr": not hyp_path.exists(),
                }
            )
            continue
        reference = load_json(ref_path)
        hypothesis = load_json(hyp_path)
        ref_text = str(reference.get("text") or "").strip()
        hyp_text = str(hypothesis.get("text") or "").strip()
        audio_id = infer_audio_id(run_dir, reference, hypothesis)
        if not ref_text or not hyp_text:
            issues.append(
                {
                    "run_id": run_dir.name,
                    "audio_id": audio_id,
                    "empty_reference_text": not bool(ref_text),
                    "empty_hypothesis_text": not bool(hyp_text),
                }
            )
            continue
        ref_backend = str(reference.get("backend") or "")
        hyp_backend = str(hypothesis.get("backend") or "")
        ref_version = str(reference.get("backend_version") or "")
        hyp_version = str(hypothesis.get("backend_version") or "")
        if not ref_backend or ref_backend != hyp_backend or not ref_version or ref_version != hyp_version:
            issues.append(
                {
                    "run_id": run_dir.name,
                    "audio_id": audio_id,
                    "backend_mismatch": True,
                    "reference_backend": ref_backend,
                    "reference_backend_version": ref_version,
                    "hypothesis_backend": hyp_backend,
                    "hypothesis_backend_version": hyp_version,
                }
            )
            continue
        base = {
            "audio_id": audio_id,
            "run_id": run_dir.name,
            "source_dataset_id": (
                "primock57_mixed"
                if audio_id.startswith("primock57")
                else "fareez_respiratory_interviews"
                if audio_id.startswith("fareez")
                else "simsamu"
                if audio_id.startswith("simsamu")
                else "unknown"
            ),
            "source_language": "fr" if audio_id.startswith("simsamu") else "en" if audio_id.startswith(("primock57", "fareez")) else "unknown",
            "reference_asr_path": str(ref_path),
            "hypothesis_asr_path": str(hyp_path),
            "reference_backend": ref_backend,
            "reference_backend_version": ref_version,
            "hypothesis_backend": hyp_backend,
            "hypothesis_backend_version": hyp_version,
            "claim_boundary": "Matched-ASR utility pair: reference is original-audio ASR, hypothesis is final-audio ASR.",
        }
        rows.append({**base, "source": "reference", "text": ref_text})
        rows.append({**base, "source": "hypothesis", "text": hyp_text})
    return rows, issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--run-glob", default="public_eval_full390_*nooverlap_exclusive_20260722T2039Z")
    ap.add_argument("--run-summary", type=Path, help="Optional batch_summary.json. If provided, run IDs are taken from rows[] instead of run-glob.")
    ap.add_argument("--reference-asr-name", default="asr.input_direct_input_v1.json")
    ap.add_argument("--hypothesis-asr-name", default="asr.final_fresh_phi_audit.json")
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--summary-json", type=Path, required=True)
    args = ap.parse_args()

    rows, issues = build_rows(
        runs_root=args.runs_root,
        run_glob=args.run_glob,
        run_summary=args.run_summary,
        reference_asr_name=args.reference_asr_name,
        hypothesis_asr_name=args.hypothesis_asr_name,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    audio_ids = sorted({row["audio_id"] for row in rows})
    summary = {
        "schema_version": "aegisvox.eval.matched_asr_pairs.v1",
        "runs_root": str(args.runs_root),
        "run_glob": args.run_glob,
        "run_summary": str(args.run_summary) if args.run_summary else None,
        "reference_asr_name": args.reference_asr_name,
        "hypothesis_asr_name": args.hypothesis_asr_name,
        "num_audio_pairs": len(audio_ids),
        "num_rows": len(rows),
        "num_issues": len(issues),
        "issues": issues[:200],
        "output_jsonl": str(args.output_jsonl),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
