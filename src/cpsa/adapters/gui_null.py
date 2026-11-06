"""GUI that renders nothing."""

from __future__ import annotations

from cpsa.domain.model import DecisionRecord


class NullGui:
    """Satisfies ``cpsa.ports.Gui`` without displaying anything."""

    def start(self) -> None:
        pass

    def publish(self, record: DecisionRecord) -> None:
        pass

    def stop(self) -> None:
        pass
