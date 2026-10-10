"""Configuration helpers for local/ORD runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AegisVoxPaths:
    project_root: Path
    sds_root: Path
    artifact_root: Path
    model_root: Path
    dataset_root: Path
    log_root: Path

    @classmethod
    def from_env(cls) -> "AegisVoxPaths":
        sds = Path(os.environ.get("AEGISVOX_DATA_ROOT", "./artifacts")).expanduser()
        project = Path(os.environ.get("AEGISVOX_PROJECT_ROOT", sds / "aegisvox")).expanduser()
        return cls(
            project_root=project,
            sds_root=sds,
            artifact_root=Path(os.environ.get("AEGISVOX_ARTIFACT_ROOT", project / "artifacts")).expanduser(),
            model_root=Path(os.environ.get("AEGISVOX_MODEL_ROOT", project / "models")).expanduser(),
            dataset_root=Path(os.environ.get("AEGISVOX_DATASET_ROOT", project / "datasets")).expanduser(),
            log_root=Path(os.environ.get("AEGISVOX_LOG_ROOT", project / "logs")).expanduser(),
        )
