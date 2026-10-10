"""Metadata backend factory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from .metadata import SQLiteMetadataStore
from .metadata_postgres import PostgresMetadataSetupError, PostgresMetadataStore

MetadataBackend = Literal["sqlite", "postgres"]


class MetadataBackendConfigError(RuntimeError):
    """Raised when the requested metadata backend is not configured."""


def create_metadata_store(
    *,
    backend: str | None = None,
    sqlite_path: str | Path | None = None,
    postgres_dsn: str | None = None,
):
    selected = (backend or os.environ.get("AEGISVOX_METADATA_BACKEND") or "sqlite").strip().lower()
    if selected == "sqlite":
        path = sqlite_path or os.environ.get("AEGISVOX_METADATA_DB")
        if path is None:
            raise MetadataBackendConfigError("SQLite metadata backend requires a db path")
        return SQLiteMetadataStore(path)
    if selected == "postgres":
        dsn = postgres_dsn or os.environ.get("AEGISVOX_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
        if not dsn:
            raise MetadataBackendConfigError("Postgres metadata backend requires AEGISVOX_POSTGRES_DSN or DATABASE_URL")
        try:
            return PostgresMetadataStore(dsn)
        except PostgresMetadataSetupError as exc:
            raise MetadataBackendConfigError(str(exc)) from exc
    raise MetadataBackendConfigError(f"unsupported metadata backend {selected!r}; expected 'sqlite' or 'postgres'")
