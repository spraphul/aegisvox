"""Optional API authentication for AegisVox services.

The local prototype defaults to unauthenticated loopback-friendly operation, but
ORD/prod-facing deployments can enable fail-closed API auth with environment
variables or explicit `create_app(...)` arguments. Secrets are never returned in
status payloads.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass, field
from typing import Any


class APIAuthConfigError(RuntimeError):
    """Raised when API auth is required but not safely configured."""


_TRUE_VALUES = {"1", "true", "yes", "on", "required", "enabled"}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _split_secret_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class APIAuthConfig:
    required: bool = False
    api_keys: tuple[str, ...] = field(default_factory=tuple)
    bearer_tokens: tuple[str, ...] = field(default_factory=tuple)
    protected_prefixes: tuple[str, ...] = ("/v1",)

    @classmethod
    def from_env(
        cls,
        *,
        required: bool | None = None,
        api_keys: list[str] | tuple[str, ...] | None = None,
        bearer_tokens: list[str] | tuple[str, ...] | None = None,
    ) -> "APIAuthConfig":
        resolved_required = env_flag("AEGISVOX_API_AUTH_REQUIRED", False) if required is None else bool(required)
        resolved_api_keys = tuple(api_keys) if api_keys is not None else _split_secret_list(os.environ.get("AEGISVOX_API_KEYS"))
        env_bearers = _split_secret_list(os.environ.get("AEGISVOX_API_BEARER_TOKENS"))
        single_bearer = _split_secret_list(os.environ.get("AEGISVOX_API_BEARER_TOKEN"))
        resolved_bearers = tuple(bearer_tokens) if bearer_tokens is not None else env_bearers + single_bearer
        config = cls(
            required=resolved_required,
            api_keys=tuple(token for token in resolved_api_keys if token),
            bearer_tokens=tuple(token for token in resolved_bearers if token),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.required and not self.api_keys and not self.bearer_tokens:
            raise APIAuthConfigError(
                "AegisVox API auth is required but no credentials are configured. "
                "Set AEGISVOX_API_KEYS or AEGISVOX_API_BEARER_TOKEN(S) at runtime."
            )

    def protects(self, path: str) -> bool:
        return self.required and any(path == prefix or path.startswith(prefix + "/") for prefix in self.protected_prefixes)

    def status(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "protected_prefixes": list(self.protected_prefixes),
            "num_api_keys": len(self.api_keys),
            "num_bearer_tokens": len(self.bearer_tokens),
        }

    def authorize(self, *, authorization_header: str | None = None, api_key_header: str | None = None) -> bool:
        if not self.required:
            return True
        supplied_bearer = None
        if authorization_header:
            parts = authorization_header.strip().split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                supplied_bearer = parts[1].strip()
        if supplied_bearer and any(hmac.compare_digest(supplied_bearer, token) for token in self.bearer_tokens):
            return True
        supplied_key = (api_key_header or "").strip()
        if supplied_key and any(hmac.compare_digest(supplied_key, token) for token in self.api_keys):
            return True
        return False
