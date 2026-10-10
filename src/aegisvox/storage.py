"""Artifact storage clients.

Phase 0 provides a local filesystem-backed store that understands `s3://bucket/key`
URIs by mapping them under a local artifact root. Later MinIO/S3 clients should
keep the same interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from .schema import AegisVoxContractError, validate_artifact_uri


@dataclass(frozen=True)
class LocalArtifactStore:
    root: Path

    def __init__(self, root: str | Path):
        object.__setattr__(self, "root", Path(root).expanduser().resolve())
        self.root.mkdir(parents=True, exist_ok=True)

    def local_path(self, uri: str) -> Path:
        validate_artifact_uri(uri)
        parsed = urlparse(uri)
        if parsed.scheme == "file":
            return Path(parsed.path)
        if parsed.scheme == "s3":
            if not parsed.netloc or not parsed.path:
                raise AegisVoxContractError(f"invalid s3 artifact URI: {uri}")
            return self.root / parsed.netloc / parsed.path.lstrip("/")
        raise AegisVoxContractError(f"unsupported URI: {uri}")

    def write_bytes(self, uri: str, data: bytes) -> Path:
        path = self.local_path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def read_bytes(self, uri: str) -> bytes:
        return self.local_path(uri).read_bytes()

    def write_json(self, uri: str, payload: Any) -> Path:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        return self.write_bytes(uri, data)

    def read_json(self, uri: str) -> Any:
        return json.loads(self.read_bytes(uri).decode("utf-8"))

    def exists(self, uri: str) -> bool:
        return self.local_path(uri).exists()
