"""Minimal event sink for Phase 0/1."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import EventEnvelope


@dataclass
class InMemoryEventBus:
    events: list[EventEnvelope] = field(default_factory=list)

    def emit(self, event: EventEnvelope) -> None:
        self.events.append(event)

    def by_run(self, run_id: str) -> list[EventEnvelope]:
        return [event for event in self.events if event.run_id == run_id]
