"""Cooldown and verdict gate rules for actuation."""

from __future__ import annotations

from cpsa.domain.model import Tag, Verdict, VisionAssessment


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


def as_assessment(result: object) -> VisionAssessment | None:
    """Normalize a verifier result to an assessment, or ``None`` when it is unusable.

    The verifier port allows a bare :class:`Verdict`, which counts as an assessment
    with no severity. Any other type is a contract violation; the caller fails open.
    """
    if isinstance(result, VisionAssessment):
        return result
    if isinstance(result, Verdict):
        return VisionAssessment(result)
    return None
