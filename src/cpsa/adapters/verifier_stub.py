"""Placeholder vision verifier."""

from __future__ import annotations

from cpsa.domain.model import Detection, Verdict, VisionAssessment


class StubVerifier:
    """Always returns an ``UNDECIDED`` assessment, so actuation follows the IMU tag alone.

    It deliberately does not implement ``cpsa.ports.DetectionObserver``: with no
    vision episode there is nothing to extend, so the IMU-only path is unchanged.

    Replace it with a real verifier by implementing ``cpsa.ports.Verifier`` and
    changing its construction in ``cpsa.board.__main__``.
    """

    def verify(self, detection: Detection) -> VisionAssessment:
        return VisionAssessment(Verdict.UNDECIDED)
