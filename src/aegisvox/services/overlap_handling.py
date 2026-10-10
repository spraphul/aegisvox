"""Overlap handling service wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aegisvox.overlap_handling import OverlapHandlingService as _OverlapHandlingService
from aegisvox.schema import OverlapMode
from aegisvox.storage import LocalArtifactStore


@dataclass
class OverlapHandlingService:
    artifact_store: LocalArtifactStore

    def handle_to_artifact(
        self,
        *,
        segments_uri: str,
        output_overlap_uri: str,
        mode: OverlapMode | str = OverlapMode.SKIP_AND_FLAG,
        output_prefix_uri: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _OverlapHandlingService(self.artifact_store).handle_to_artifact(
            segments_uri=segments_uri,
            output_overlap_uri=output_overlap_uri,
            mode=mode,
            output_prefix_uri=output_prefix_uri,
            options=options,
        )
