"""Shared CLI helpers for audit-safe script output.

AegisVox command-line tools are often run inside tmux/ORD logs where stderr is
persistent. These helpers ensure setup/runtime errors are JSON shaped and passed
through the same redaction layer used by API, worker, and readiness paths.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .audit import redact_text, sanitize_for_audit


def safe_error_payload(exc: Exception | str, *, reason: str | None = None, max_length: int = 1000) -> dict[str, Any]:
    """Return an audit-safe `{"ok": false, ...}` payload for CLI errors."""

    payload: dict[str, Any] = {"ok": False, "error": redact_text(str(exc), max_length=max_length)}
    if reason:
        payload["reason"] = redact_text(reason, max_length=200)
    return payload


def print_json(payload: Any, *, stream: TextIO | None = None, indent: int = 2) -> None:
    """Print sanitized JSON payload to a stream."""

    target = stream or sys.stdout
    print(json.dumps(sanitize_for_audit(payload), indent=indent, sort_keys=True), file=target)


def print_json_error(exc: Exception | str, *, reason: str | None = None, stream: TextIO | None = None) -> None:
    """Print a sanitized JSON error payload to stderr by default."""

    print_json(safe_error_payload(exc, reason=reason), stream=stream or sys.stderr)
