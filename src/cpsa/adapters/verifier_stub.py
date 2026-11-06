"""Placeholder vision verifier."""

from __future__ import annotations

from cpsa.domain.model import Detection, Verdict


class StubVerifier:
    """Always returns ``UNDECIDED``, so actuation depends on the IMU tag alone.

    Replace it with a real verifier by implementing ``cpsa.ports.Verifier`` and
    changing its construction in ``cpsa.board.__main__``.
    """

    def verify(self, detection: Detection) -> Verdict:
        return Verdict.UNDECIDED
