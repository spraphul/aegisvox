"""Audit-safe redaction helpers.

Runtime services may receive credentials, PHI-bearing transcripts, filesystem
paths, and command output. Failure metadata should be useful for debugging but
must not persist secrets or obvious PHI values. These helpers provide a compact,
dependency-free sanitizer for logs/events/job error JSON.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"

SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|openai[_-]?key|hf[_-]?token|huggingface[_-]?token|pyannote[_-]?auth[_-]?token|"
    r"authorization|auth[_-]?token|access[_-]?token|refresh[_-]?token|secret|password|passwd|credential)",
    re.IGNORECASE,
)

SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/\-=]{12,}\b", re.IGNORECASE),
    re.compile(r"(?i)(OPENAI_API_KEY|OPENAI_API_KEY|HF_TOKEN|HUGGINGFACE_TOKEN|PYANNOTE_AUTH_TOKEN)\s*=\s*[^\s'\"]+"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,'\"}]+"),
]

PHI_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b(?:MRN|medical record(?: number)?|record number)[:#\s-]*[A-Za-z0-9-]{4,}\b", re.IGNORECASE),
    re.compile(r"\b(?:member ID|patient ID|account number)[:#\s-]*[A-Za-z0-9-]{4,}\b", re.IGNORECASE),
]


def redact_text(value: str, *, max_length: int | None = 4000) -> str:
    """Redact obvious secrets/PHI from a string and optionally bound length."""

    text = str(value)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    for pattern in PHI_PATTERNS:
        text = pattern.sub(REDACTED, text)
    if max_length is not None and len(text) > max_length:
        omitted = len(text) - max_length
        text = text[:max_length] + f"...[truncated {omitted} chars]"
    return text


def sanitize_for_audit(value: Any, *, max_string_length: int | None = 4000) -> Any:
    """Recursively sanitize values before writing audit/event/error metadata."""

    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if SENSITIVE_KEY_RE.search(key_str):
                clean[key_str] = REDACTED
            else:
                clean[key_str] = sanitize_for_audit(item, max_string_length=max_string_length)
        return clean
    if isinstance(value, str):
        return redact_text(value, max_length=max_string_length)
    if isinstance(value, bytes):
        return redact_text(value.decode("utf-8", errors="replace"), max_length=max_string_length)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_for_audit(item, max_string_length=max_string_length) for item in value]
    return value
