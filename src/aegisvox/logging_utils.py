"""Structured audit-safe JSON logging utilities.

AegisVox services often need durable operational logs for ORD/tmux runs, but
those logs must not leak runtime credentials or obvious PHI. This module writes
one sanitized JSON object per line so logs are easy to grep, ingest, and attach
to audits without persisting raw secrets.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import TextIO
from datetime import datetime, timezone
from typing import Any

from .audit import sanitize_for_audit

_VALID_LEVELS = {"debug", "info", "warning", "error", "critical"}


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSONL audit logs."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_audit_log_record(
    *,
    event_type: str,
    service: str,
    level: str = "info",
    run_id: str | None = None,
    job_id: str | None = None,
    message: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Build a sanitized structured log record.

    `fields` may include arbitrary backend exception data or command metadata;
    it is recursively sanitized before the record is returned. Values that are
    `None` are omitted to keep logs compact.
    """

    normalized_level = str(level).lower()
    if normalized_level not in _VALID_LEVELS:
        normalized_level = "info"

    record: dict[str, Any] = {
        "timestamp": utc_timestamp(),
        "level": normalized_level,
        "event_type": str(event_type),
        "service": str(service),
    }
    if run_id is not None:
        record["run_id"] = str(run_id)
    if job_id is not None:
        record["job_id"] = str(job_id)
    if message is not None:
        record["message"] = str(message)
    for key, value in fields.items():
        if value is not None:
            record[str(key)] = value
    clean = sanitize_for_audit(record)
    if not isinstance(clean, Mapping):  # defensive; sanitize_for_audit preserves mappings
        raise TypeError("sanitized audit log record is not a mapping")
    return dict(clean)


def write_audit_log_event(
    stream: TextIO,
    *,
    event_type: str,
    service: str,
    level: str = "info",
    run_id: str | None = None,
    job_id: str | None = None,
    message: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Write one sanitized JSONL audit record to `stream` and return it."""

    record = build_audit_log_record(
        event_type=event_type,
        service=service,
        level=level,
        run_id=run_id,
        job_id=job_id,
        message=message,
        **fields,
    )
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()
    return record


def open_audit_log_stream(path: str | None) -> TextIO | None:
    """Open an append-only JSONL log stream, or return stderr for '-' / None for empty."""

    if not path:
        return None
    if path == "-":
        return sys.stderr
    return open(path, "a", encoding="utf-8")
