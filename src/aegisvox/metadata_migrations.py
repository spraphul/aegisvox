"""Metadata schema migration helpers for local SQLite and future Postgres.

This module is intentionally small and dependency-light. SQLite is the Phase 1
runtime path. Postgres support is a fail-loud adapter that requires an explicit
psycopg installation before production-style deployments can use it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .metadata import SQLiteMetadataStore


class MetadataMigrationError(RuntimeError):
    """Raised when a metadata migration cannot be applied."""


@dataclass(frozen=True)
class MigrationResult:
    backend: str
    target: str
    schema_path: str
    statements_applied: int
    tables: list[str]
    job_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "target": self.target,
            "schema_path": self.schema_path,
            "statements_applied": self.statements_applied,
            "tables": list(self.tables),
            "job_columns": list(self.job_columns),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def default_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / "metadata_schema.sql"


def split_sql_statements(sql: str) -> list[str]:
    """Split a simple SQL migration file into executable statements.

    The AegisVox schema file is deliberately simple: no procedural SQL and no
    semicolon-containing string literals. This splitter strips full-line
    comments and returns non-empty semicolon-delimited statements.
    """

    body_lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        body_lines.append(line)
    body = "\n".join(body_lines)
    return [statement.strip() for statement in body.split(";") if statement.strip()]


def _sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def migrate_sqlite(db_path: str | Path, schema_path: str | Path | None = None) -> MigrationResult:
    """Apply the metadata schema to a SQLite DB and run in-place compatibility migrations."""

    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = Path(schema_path or default_schema_path()).expanduser().resolve()
    if not schema.exists():
        raise MetadataMigrationError(f"metadata schema file not found: {schema}")
    statements = split_sql_statements(schema.read_text())
    with sqlite3.connect(db_path) as conn:
        for statement in statements:
            conn.execute(statement)
    # Reuse the production SQLite store initializer so column additions stay in
    # one place and older ORD/local DBs are migrated safely.
    SQLiteMetadataStore(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = _sqlite_tables(conn)
        job_columns = _sqlite_columns(conn, "jobs") if "jobs" in tables else []
    return MigrationResult(
        backend="sqlite",
        target=str(db_path),
        schema_path=str(schema),
        statements_applied=len(statements),
        tables=tables,
        job_columns=job_columns,
    )


def migrate_postgres(dsn: str, schema_path: str | Path | None = None) -> MigrationResult:
    """Apply the metadata schema to Postgres using psycopg when installed.

    This is intentionally fail-loud: if psycopg is missing, or the DSN is not
    reachable, callers get an actionable setup error instead of silently falling
    back to SQLite.
    """

    schema = Path(schema_path or default_schema_path()).expanduser().resolve()
    if not schema.exists():
        raise MetadataMigrationError(f"metadata schema file not found: {schema}")
    statements = split_sql_statements(schema.read_text())
    try:
        import psycopg  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exact import failure is environment-dependent
        raise MetadataMigrationError(
            "Postgres migration requires psycopg. Install with `python -m pip install 'psycopg[binary]>=3.2'` "
            "or install the aegisvox postgres optional dependency when it is enabled."
        ) from exc

    try:
        with psycopg.connect(dsn) as conn:  # type: ignore[name-defined]
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' ORDER BY table_name"
                )
                tables = [str(row[0]) for row in cur.fetchall()]
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'jobs' ORDER BY ordinal_position"
                )
                job_columns = [str(row[0]) for row in cur.fetchall()]
    except Exception as exc:
        raise MetadataMigrationError(f"failed to apply Postgres metadata schema: {exc}") from exc
    return MigrationResult(
        backend="postgres",
        target=dsn,
        schema_path=str(schema),
        statements_applied=len(statements),
        tables=tables,
        job_columns=job_columns,
    )


def migrate_metadata(
    *,
    backend: Literal["sqlite", "postgres"],
    target: str | Path,
    schema_path: str | Path | None = None,
) -> MigrationResult:
    if backend == "sqlite":
        return migrate_sqlite(target, schema_path=schema_path)
    if backend == "postgres":
        return migrate_postgres(str(target), schema_path=schema_path)
    raise MetadataMigrationError(f"unsupported metadata migration backend {backend!r}")
