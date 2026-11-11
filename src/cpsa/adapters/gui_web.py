"""Web dashboard adapter: publishes the medical topics through the ``Publisher`` port.

``WebGui`` is both the runtime's ``Gui`` (IF-704) and the vision verifier's
``VisionObserver`` (F2, F14). It translates domain and vision objects into protocol
payloads with :mod:`cpsa.protocol.medical` and never blocks the caller: the
publisher queues or replaces messages, and every failure is logged and swallowed
so an observer can never change a verdict. The RGB image of each analyzed frame is
forwarded unmodified to a publisher that offers ``/video.mjpeg``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cpsa.domain.model import DecisionRecord, Tag
from cpsa.ports import Clock, Publisher
from cpsa.protocol.medical import ImuStatus, Trigger, decision, episode, hands, state
from cpsa.vision.types import EpisodeReport, KeypointLayout, TrackedFrame

logger = logging.getLogger(__name__)

#: IF-454: at most 15 ``medical.hands`` states per second.
HANDS_MAX_HZ = 15.0
_MIN_HANDS_INTERVAL_S = 1.0 / HANDS_MAX_HZ
_EPS = 1e-9


class WebGui:
    """Publish ``medical.decision``/``episode``/``hands`` and keep the latest state.

    Args:
        publisher: server-side publisher; threaded and never blocking.
        layout: keypoint layout of the pose model used by the verifier.
        frame_w, frame_h: camera frame size in pixels, for ``medical.hands``.
        clock: monotonic clock shared with the runtime, for the 15 Hz throttle.
        trigger: episode trigger source reported in ``medical.state`` (F11).
    """

    def __init__(
        self,
        publisher: Publisher,
        layout: KeypointLayout,
        frame_w: int,
        frame_h: int,
        clock: Clock,
        *,
        trigger: Trigger = "imu",
    ) -> None:
        self._publisher = publisher
        self._layout = layout
        self._frame_w = int(frame_w)
        self._frame_h = int(frame_h)
        self._clock = clock
        self._trigger = trigger
        self._lock = threading.Lock()
        self._tag: Tag | None = None
        self._last_decision: dict[str, Any] | None = None
        self._episode: dict[str, Any] | None = None
        self._last_hands_t: float | None = None

    # -- Gui ---------------------------------------------------------------

    def start(self) -> None:
        """Nothing to open: the publisher is owned by the server."""

    def publish(self, record: DecisionRecord) -> None:
        """Send one ``medical.decision`` event and remember the current tag."""
        payload = decision(record)
        with self._lock:
            self._last_decision = payload
            self._tag = record.tag
        self._event("medical.decision", payload)

    def stop(self) -> None:
        """Nothing to close: the publisher is owned by the server."""

    # -- VisionObserver ----------------------------------------------------

    def on_episode(self, episode_id: str, t_start: float, active: bool) -> None:
        """Track the episode and announce its start.

        The report event (IF-453) is sent by :meth:`on_report` once the verifier
        has finished the report, so an end does not clear the dashboard early.
        """
        with self._lock:
            self._episode = {
                "episode_id": episode_id,
                "t_start": float(t_start),
                "active": bool(active),
            }
        if active:
            self._event(
                "medical.episode",
                {
                    "episode_id": episode_id,
                    "active": True,
                    "contact_count": 0,
                    "repetition_hz": None,
                    "peak_severity": None,
                    "labels": [],
                    "contacts": [],
                },
            )

    def on_tracked(self, tracked: TrackedFrame) -> None:
        """Send a ``medical.hands`` state, throttled to at most 15 Hz."""
        now = self._clock.monotonic()
        with self._lock:
            last = self._last_hands_t
            if last is not None and now - last < _MIN_HANDS_INTERVAL_S - _EPS:
                return
            self._last_hands_t = now
        self._state("medical.hands", hands(tracked, self._layout, self._frame_w, self._frame_h))

    def on_frame_image(self, seq: int, t: float, rgb: NDArray[np.uint8]) -> None:
        """Hand one analyzed RGB frame to the video stream, without copying it (F14).

        The server adapter owns the encoder and the frame pacing; a publisher without
        live video (the fake mode, a test double) simply has no ``offer_video_frame``.
        """
        offer = getattr(self._publisher, "offer_video_frame", None)
        if not callable(offer):
            return
        try:
            offer(seq, rgb)
        except Exception:
            logger.exception("Publisher rejected a video frame")

    def on_report(self, report: EpisodeReport) -> None:
        """Send the full ``medical.episode`` event for a finished report."""
        payload = episode(report)
        with self._lock:
            self._episode = {
                "episode_id": report.episode_id,
                "t_start": float(report.t_start),
                "active": bool(report.active),
            }
        self._event("medical.episode", payload)

    # -- state for the 1 Hz publisher -------------------------------------

    @property
    def tag(self) -> Tag | None:
        """Tag of the most recent decision record, or ``None`` before the first."""
        with self._lock:
            return self._tag

    def snapshot_state(self, imu: ImuStatus) -> dict[str, Any]:
        """Build the current ``medical.state`` payload for the mode's publisher."""
        with self._lock:
            tag = self._tag
            episode_state = None if self._episode is None else dict(self._episode)
            last_decision = None if self._last_decision is None else dict(self._last_decision)
        return state(imu, tag, episode_state, last_decision, trigger=self._trigger)

    # -- publisher calls ---------------------------------------------------

    def _state(self, message_type: str, payload: Mapping[str, Any]) -> None:
        try:
            self._publisher.state(message_type, payload)
        except Exception:
            logger.exception("Publisher rejected state %s", message_type)

    def _event(self, message_type: str, payload: Mapping[str, Any]) -> None:
        try:
            self._publisher.event(message_type, payload)
        except Exception:
            logger.exception("Publisher rejected event %s", message_type)
