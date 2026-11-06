"""Cooldown and verdict gate rules for actuation."""

from __future__ import annotations

from cpsa.domain.model import Tag, Verdict


def should_evaluate(
    tag: Tag, last_actuation_ts: float | None, now: float, cooldown_s: float
) -> bool:
    """Return whether a detection should be verified and possibly actuated.

    Only stereotypy tags are evaluated, and only when no actuation happened for the
    current tag yet or the cooldown has elapsed since the last one.
    """
    if not tag.is_stereotypy:
        return False
    return last_actuation_ts is None or now - last_actuation_ts >= cooldown_s


def allows_actuation(verdict: Verdict) -> bool:
    """Fail-open gate: only an explicit rejection prevents actuation."""
    return verdict is not Verdict.REJECTED
