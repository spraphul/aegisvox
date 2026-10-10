"""Open Swara source-manifest export helpers.

Open Swara on Hugging Face is organized as audio files under
`voices/{language}/{gender}/*.wav`. This module exports those files into the
AegisVox voice-pool source-manifest contract consumed by `build_voice_pool.py`.
It does not fabricate audio or metadata; missing Hugging Face dependencies or
missing local files fail loudly.
"""

from __future__ import annotations

import importlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .schema import AegisVoxContractError


class OpenSwaraExportError(RuntimeError):
    """Raised when Open Swara source-manifest export cannot proceed."""


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
DEFAULT_LICENSE = "cc-by-sa-4.0"
DEFAULT_DATASET_ID = "jaymunshi/open-swara"


def _slug(value: str) -> str:
    value = value.strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


@dataclass(frozen=True)
class OpenSwaraFileRecord:
    source_path: str
    repo_file: str | None
    language: str
    gender: str
    sample_id: str
    voice_id: str
    license: str = DEFAULT_LICENSE
    is_english_accent: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_source_manifest_row(self, *, use_path: str | None = None) -> dict[str, Any]:
        path = use_path or self.source_path
        row = {
            "voice_id": self.voice_id,
            "sample_id": self.sample_id,
            "path": str(path),
            "language": self.language,
            "gender_presentation": self.gender,
            "license": self.license,
            "provenance": dict(self.provenance),
            "metadata": {
                "source_dataset": "open_swara",
                "repo_file": self.repo_file,
                "is_english_accent": self.is_english_accent,
                **dict(self.metadata),
            },
        }
        return row


def parse_openswara_repo_file(repo_file: str, *, dataset_id: str = DEFAULT_DATASET_ID, license: str = DEFAULT_LICENSE) -> OpenSwaraFileRecord | None:
    path = Path(repo_file)
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        return None
    parts = repo_file.split("/")
    if len(parts) < 4 or parts[0] != "voices":
        return None
    language = _slug(parts[1])
    gender = _slug(parts[2])
    stem = _slug(path.stem)
    is_english_accent = "_english_open_swara_" in stem or stem.endswith("_english")
    sample_id = stem
    voice_id = stem
    return OpenSwaraFileRecord(
        source_path=repo_file,
        repo_file=repo_file,
        language=language,
        gender=gender,
        sample_id=sample_id,
        voice_id=voice_id,
        license=license,
        is_english_accent=is_english_accent,
        provenance={"source": "huggingface_dataset", "dataset_id": dataset_id, "repo_file": repo_file},
    )


def records_from_repo_files(
    repo_files: Iterable[str],
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    languages: set[str] | None = None,
    genders: set[str] | None = None,
    license: str = DEFAULT_LICENSE,
    max_samples: int | None = None,
) -> list[OpenSwaraFileRecord]:
    languages = {_slug(x) for x in languages} if languages else None
    genders = {_slug(x) for x in genders} if genders else None
    records = []
    for repo_file in sorted(repo_files):
        record = parse_openswara_repo_file(repo_file, dataset_id=dataset_id, license=license)
        if record is None:
            continue
        if languages and record.language not in languages:
            continue
        if genders and record.gender not in genders:
            continue
        records.append(record)
        if max_samples is not None and len(records) >= max_samples:
            break
    if not records:
        raise OpenSwaraExportError("no Open Swara audio files matched the requested filters")
    return records


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _load_existing_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id")
        if sample_id:
            rows[str(sample_id)] = row
    return rows


def export_local_openswara_source_manifest(
    *,
    local_root: str | Path,
    output_jsonl: str | Path,
    languages: set[str] | None = None,
    genders: set[str] | None = None,
    license: str = DEFAULT_LICENSE,
    max_samples: int | None = None,
) -> dict[str, Any]:
    root = Path(local_root).expanduser().resolve()
    if not root.exists():
        raise OpenSwaraExportError(f"local Open Swara root not found: {root}")
    repo_files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            if rel.startswith("voices/"):
                repo_files.append(rel)
            else:
                # Also accept roots that directly contain language/gender/audio.
                repo_files.append(f"voices/{rel}")
    records = records_from_repo_files(repo_files, languages=languages, genders=genders, license=license, max_samples=max_samples)
    rows = []
    for record in records:
        rel = record.repo_file.removeprefix("voices/") if record.repo_file else record.source_path
        source_path = root / "voices" / rel
        if not source_path.exists():
            source_path = root / rel
        if not source_path.exists():
            raise OpenSwaraExportError(f"matched Open Swara file is missing locally: {source_path}")
        rows.append(record.to_source_manifest_row(use_path=str(source_path)))
    _write_jsonl(output_jsonl, rows)
    return {
        "ok": True,
        "mode": "local",
        "output_jsonl": str(output_jsonl),
        "num_rows": len(rows),
        "languages": sorted({row["language"] for row in rows}),
        "genders": sorted({row["gender_presentation"] for row in rows}),
    }


def _load_huggingface_hub() -> Any:
    try:
        return importlib.import_module("huggingface_hub")
    except Exception as exc:
        raise OpenSwaraExportError(
            "Open Swara HF export requires huggingface_hub. Install voice-pool dependencies first."
        ) from exc


def export_hf_openswara_source_manifest(
    *,
    output_jsonl: str | Path,
    audio_output_dir: str | Path,
    dataset_id: str = DEFAULT_DATASET_ID,
    revision: str | None = None,
    languages: set[str] | None = None,
    genders: set[str] | None = None,
    license: str = DEFAULT_LICENSE,
    max_samples: int | None = None,
    symlink_from_cache: bool = False,
) -> dict[str, Any]:
    hub = _load_huggingface_hub()
    try:
        repo_files = hub.list_repo_files(repo_id=dataset_id, repo_type="dataset", revision=revision)
    except Exception as exc:
        raise OpenSwaraExportError(f"failed to list Hugging Face dataset files for {dataset_id}: {exc}") from exc
    records = records_from_repo_files(
        repo_files,
        dataset_id=dataset_id,
        languages=languages,
        genders=genders,
        license=license,
        max_samples=max_samples,
    )
    audio_output_dir = Path(audio_output_dir).expanduser().resolve()
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(output_jsonl)
    partial_jsonl = output_jsonl.with_suffix(output_jsonl.suffix + ".partial")
    rows_by_sample = _load_existing_rows(partial_jsonl)
    rows = list(rows_by_sample.values())
    for record in records:
        if record.sample_id in rows_by_sample:
            continue
        assert record.repo_file is not None
        target = audio_output_dir / record.repo_file
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            # Resume-safe path: if a previous interrupted exporter already
            # staged the file, do not re-enter hf_hub_download. This avoids
            # paying download cost again and avoids getting stuck on a stale
            # hub/xet connection before manifest finalization.
            pass
        else:
            try:
                downloaded = Path(
                    hub.hf_hub_download(
                        repo_id=dataset_id,
                        filename=record.repo_file,
                        repo_type="dataset",
                        revision=revision,
                    )
                )
            except Exception as exc:
                raise OpenSwaraExportError(f"failed to download {record.repo_file} from {dataset_id}: {exc}") from exc
            if symlink_from_cache:
                target.symlink_to(downloaded)
            else:
                shutil.copyfile(downloaded, target)
        row = record.to_source_manifest_row(use_path=str(target))
        rows_by_sample[record.sample_id] = row
        rows.append(row)
        _append_jsonl(partial_jsonl, row)
    rows = [rows_by_sample[r.sample_id] for r in records if r.sample_id in rows_by_sample]
    _write_jsonl(output_jsonl, rows)
    if partial_jsonl.exists():
        partial_jsonl.unlink()
    return {
        "ok": True,
        "mode": "hf",
        "dataset_id": dataset_id,
        "revision": revision,
        "output_jsonl": str(output_jsonl),
        "audio_output_dir": str(audio_output_dir),
        "num_rows": len(rows),
        "languages": sorted({row["language"] for row in rows}),
        "genders": sorted({row["gender_presentation"] for row in rows}),
    }
