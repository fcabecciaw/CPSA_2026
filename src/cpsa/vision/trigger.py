"""Camera-only episode trigger (vision RASD FR-028, FR-029; plan F7).

Pure logic: it consumes contact starts and an oscillation flag per analyzed
frame, and decides when vision alone starts an episode, emits follow-ups, and
clears the tag when an episode ends. No I/O; numpy-free.

Frozen interface (``docs/plans/camera-only-trigger.md``, F7): the constructor,
``contact``/``oscillation``/``poll``/``episode_ended``, ``TriggerConfig`` and
``TriggerFire`` are fixed; their semantics must not change.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from cpsa.domain.model import Tag

__all__ = ["TriggerConfig", "TriggerFire", "VisionTrigger"]

_EPS = 1e-9

# FR-016 grades only; the rank makes the maximum of a contact window cheap.
_RANK: dict[Tag, int] = {Tag.NON_DANGEROUS: 1, Tag.DANGEROUS: 2}


@dataclass(frozen=True)
class TriggerConfig:
    """Thresholds of the vision trigger (FR-028, FR-029)."""

    min_contacts: int = 2  # N: contacts needed within ``contact_window_s``
    contact_window_s: float = 10.0  # T
    oscillation_min_s: float = 3.0  # D: continuous rocking/flapping needed
    refractory_s: float = 5.0  # no new start this long after an episode ends


@dataclass(frozen=True)
class TriggerFire:
    """One trigger decision, turned into a ``Detection`` by the verifier."""

    t: float  # monotonic time of the frame that fired
    # FR-016 contact severity; NON_DANGEROUS for oscillation; NON_STEREOTIPY for clear
    severity: Tag
    reason: Literal["contacts", "oscillation", "clear"]


@dataclass
class _Contact:
    """One contact start and its current grade (deduplicated by ``t_start``)."""

    t_start: float
    severity: Tag

    def raise_grade(self, severity: Tag) -> None:
        """Keep the higher of the recorded and the supplied grade."""
        if _RANK[severity] > _RANK[self.severity]:
            self.severity = severity


class VisionTrigger:
    """Decide episode starts and follow-ups from vision events.

    Semantics:
      * start (``episode_active=False``, not refractory): fire ``contacts`` when
        at least ``min_contacts`` contact starts that have not fired yet fall
        within the last ``contact_window_s`` (severity: the maximum of those
        contacts); else fire ``oscillation`` when ``oscillation`` has been
        continuously active for at least ``oscillation_min_s``;
      * follow-up (``episode_active=True``): fire on every new contact (its own
        severity), or once per ``oscillation_min_s`` of continued oscillation;
      * refractory: no start fire within ``refractory_s`` after ``episode_ended``;
      * every contact counts once.
    """

    def __init__(self, cfg: TriggerConfig) -> None:
        self._cfg = cfg
        # Contact starts still inside the window, oldest first. A contact that
        # has already produced a fire is counted, and every newer one is not.
        self._contacts: deque[_Contact] = deque()
        self._counted = 0
        # Continuous oscillation: when the current run began, the frame of the
        # latest sample, and when the next fire is due.
        self._osc_start: float | None = None
        self._osc_last: float | None = None
        self._osc_due: float | None = None
        self._refractory_until: float | None = None

    @property
    def config(self) -> TriggerConfig:
        """The thresholds this trigger was built with."""
        return self._cfg

    def contact(self, t_start: float, severity: Tag) -> None:
        """Record one new contact event (call once per contact)."""
        t = float(t_start)
        if not isfinite(t):
            raise ValueError("contact start must be finite")
        if severity not in _RANK:
            raise ValueError(f"contact severity must be a stereotypy grade, got {severity!r}")
        self._prune(t)
        for event in self._contacts:
            if event.t_start == t:
                # The same run is graded again over its recoil tail; keep the
                # final grade but do not count it twice.
                event.raise_grade(severity)
                return
        self._contacts.append(_Contact(t, severity))

    def oscillation(self, t: float, active: bool) -> None:
        """Record whether rocking or flapping is active at analyzed frame ``t``."""
        now = float(t)
        if not isfinite(now):
            raise ValueError("oscillation time must be finite")
        if not active:
            self._osc_start = None
            self._osc_last = now
            self._osc_due = None
            return
        last = self._osc_last
        if (
            self._osc_start is None
            or last is None
            or now < last
            or now - last > self._cfg.oscillation_min_s
        ):
            # No sample for longer than the required run breaks continuity.
            self._osc_start = now
            self._osc_due = now + self._cfg.oscillation_min_s
        self._osc_last = now

    def poll(self, t: float, episode_active: bool) -> TriggerFire | None:
        """Return a fire when a start or a follow-up is due at ``t``, else ``None``."""
        now = float(t)
        if not isfinite(now):
            raise ValueError("poll time must be finite")
        self._prune(now)
        if episode_active:
            return self._follow_up(now)
        return self._start(now)

    def episode_ended(self, t: float) -> TriggerFire:
        """Start the refractory period and return the ``clear`` fire (NON_STEREOTIPY)."""
        now = float(t)
        if not isfinite(now):
            raise ValueError("episode end must be finite")
        # A new episode needs N new contacts: the ones of the episode that just
        # ended must not combine with the first new one.
        self._contacts.clear()
        self._counted = 0
        self._osc_start = None
        self._osc_last = None
        self._osc_due = None
        self._refractory_until = now + self._cfg.refractory_s
        return TriggerFire(now, Tag.NON_STEREOTIPY, "clear")

    # -- internals --------------------------------------------------------

    def _prune(self, now: float) -> None:
        """Drop contact starts older than the window, keeping the count in step."""
        horizon = now - self._cfg.contact_window_s
        while self._contacts and self._contacts[0].t_start < horizon - _EPS:
            self._contacts.popleft()
            if self._counted > 0:
                self._counted -= 1

    def _oscillation_active(self, now: float) -> bool:
        """Whether the last oscillation sample is fresh enough to still hold."""
        last = self._osc_last
        return last is not None and 0.0 <= now - last <= self._cfg.oscillation_min_s + _EPS

    def _start(self, now: float) -> TriggerFire | None:
        """Return a start fire when the window or the oscillation run qualifies."""
        if self._refractory_until is not None:
            if now < self._refractory_until - _EPS:
                return None
            self._refractory_until = None
        uncounted = len(self._contacts) - self._counted
        if uncounted >= self._cfg.min_contacts:
            # Only contacts that have not fired yet start an episode: a start the
            # verifier did not turn into an episode must not re-fire every frame.
            fresh = list(self._contacts)[self._counted :]
            severity = max(fresh, key=lambda event: _RANK[event.severity]).severity
            self._counted = len(self._contacts)
            return TriggerFire(now, severity, "contacts")
        return self._oscillation_fire(now)

    def _follow_up(self, now: float) -> TriggerFire | None:
        """Return a follow-up fire for a new contact or a sustained oscillation."""
        if self._counted < len(self._contacts):
            event = self._contacts[self._counted]
            self._counted += 1
            return TriggerFire(now, event.severity, "contacts")
        return self._oscillation_fire(now)

    def _oscillation_fire(self, now: float) -> TriggerFire | None:
        """Fire while oscillation has been continuously active for ``D``."""
        if not self._oscillation_active(now) or self._osc_start is None:
            return None
        due = self._osc_due
        if now - self._osc_start < self._cfg.oscillation_min_s - _EPS:
            return None
        if due is not None and now < due - _EPS:
            return None
        self._osc_due = now + self._cfg.oscillation_min_s
        return TriggerFire(now, Tag.NON_DANGEROUS, "oscillation")
