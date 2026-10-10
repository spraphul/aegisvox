"""Model residency contracts for production-style AegisVox backends.

AegisVox should not run GPU models as one subprocess per segment/span in
production quality runs. This module makes that expectation explicit and gives
service adapters a single fail-loud guard controlled by
``AEGISVOX_REQUIRE_PERSISTENT_MODEL_SERVICES=1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProcessMode(str, Enum):
    SUBPROCESS_PER_ITEM = "subprocess_per_item"
    SUBPROCESS_PER_AUDIO = "subprocess_per_audio"
    PERSISTENT_WORKER = "persistent_worker"
    REMOTE_API = "remote_api"
    IN_PROCESS_CACHED = "in_process_cached"
    DETERMINISTIC = "deterministic"


@dataclass(frozen=True)
class ModelResidency:
    service: str
    backend: str
    process_mode: ProcessMode
    singleton: bool
    model_load_scope: str
    endpoint: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "backend": self.backend,
            "process_mode": self.process_mode.value,
            "singleton": self.singleton,
            "model_load_scope": self.model_load_scope,
            "endpoint": self.endpoint,
            "notes": self.notes,
        }


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def require_persistent_model_services() -> bool:
    return env_bool("AEGISVOX_REQUIRE_PERSISTENT_MODEL_SERVICES", default=False)


def assert_persistent_allowed(residency: ModelResidency) -> None:
    """Raise when a heavy model backend violates production residency policy."""

    if not require_persistent_model_services():
        return
    if residency.process_mode in {
        ProcessMode.PERSISTENT_WORKER,
        ProcessMode.REMOTE_API,
        ProcessMode.IN_PROCESS_CACHED,
        ProcessMode.DETERMINISTIC,
    }:
        return
    raise RuntimeError(
        f"{residency.service} backend {residency.backend!r} is {residency.process_mode.value}; "
        "production/full-quality runs require persistent singleton model services. "
        "Configure the service HTTP endpoint or unset AEGISVOX_REQUIRE_PERSISTENT_MODEL_SERVICES for a debug-only run."
    )
