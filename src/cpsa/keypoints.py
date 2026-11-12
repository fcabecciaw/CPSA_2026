"""Keypoint stabilization between MoveNet and the overlay (R2-G).

MoveNet runs at about 22 Hz on the board while the skeleton room is redrawn at
the camera's rate: the raw keypoints land on a different pixel every frame even
when the body stands still, a joint whose score dips under the overlay
threshold blinks off, and one bad heatmap peak sends a wrist to the corner of
the frame for a single frame. :class:`KeypointStabilizer` fixes all three the
way :class:`cpsa.tracker.Tracker` fixes the YOLOX boxes:

* every keypoint runs its own One Euro filter on ``x, y`` -- a low cutoff at
  rest, raised by ``beta`` times the measured speed so a real swing is followed
  within a frame;
* the published position leaves a deadband only when the filtered position has
  moved more than ``deadband_px`` (or 1 % of the shoulder width), so a still
  body publishes pixel-identical coordinates frame after frame;
* a keypoint appears after ``show_frames`` frames at ``show_score`` and
  disappears only after ``hide_frames`` frames under ``hide_score``, holding
  its last position meanwhile, so a score dip never blinks a joint;
* a jump past a quarter of the frame diagonal is an outlier: it is held for one
  frame and accepted on the next, so a heatmap peak cannot teleport a joint;
* the filters restart on a model swap and when the person has been gone for
  more than a second.

A visible keypoint publishes at least ``show_score`` and an invisible one
publishes ``0.0``, so a consumer that draws at the show threshold -- the
overlay and the dashboard do -- draws exactly the visible keypoints, and a bone
is only ever drawn between two visible joints.

Everything is plain numpy over the 17 ``(x, y)`` pairs: one frame costs a few
tens of microseconds, far inside the 0.3 ms the skeleton path budgets.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

from cpsa.config import SmoothingConfig
from cpsa.movenet import KEYPOINTS, MovenetPose

__all__ = [
    "OUTLIER_FRACTION",
    "RESET_AFTER_S",
    "SHOULDER_FRACTION",
    "KeypointStabilizer",
]

RESET_AFTER_S = 1.0
"""With the person gone this long, the filters restart from the new pose."""

OUTLIER_FRACTION = 0.25
"""A one-frame jump past this fraction of the frame diagonal is an outlier."""

SHOULDER_FRACTION = 0.01
"""The deadband widens to this fraction of the shoulder width when that is wider."""

_MIN_DT_S = 1e-3
"""Floor of the sample interval: two poses may share a timestamp after a swap."""

_LEFT_SHOULDER = KEYPOINTS.index("left_shoulder")
_RIGHT_SHOULDER = KEYPOINTS.index("right_shoulder")
_COUNT = len(KEYPOINTS)


class KeypointStabilizer:
    """Stabilize MoveNet keypoints with One Euro filters, a deadband and hysteresis.

    Args:
        config: The validated ``movenet.smoothing`` section. ``enabled: false``
            returns every pose unchanged.
    """

    def __init__(self, config: SmoothingConfig) -> None:
        self._config = config
        self._blank()

    def _blank(self) -> None:
        """Forget every filter and every hysteresis count."""
        self._filtered: NDArray[np.float64] = np.zeros((_COUNT, 2), dtype=np.float64)
        """One Euro output of every keypoint, ``(x, y)`` in source pixels."""
        self._speed: NDArray[np.float64] = np.zeros((_COUNT, 2), dtype=np.float64)
        """Filtered ``(dx, dy)`` of every keypoint, in pixels per second."""
        self._ready: NDArray[np.bool_] = np.zeros(_COUNT, dtype=bool)
        """Which filters carry a value and may be advanced."""
        self._displayed: NDArray[np.float32] = np.zeros((_COUNT, 2), dtype=np.float32)
        """The published ``(x, y)``: pixel-identical until it leaves the deadband."""
        self._placed: NDArray[np.bool_] = np.zeros(_COUNT, dtype=bool)
        """Which keypoints have a displayed position at all."""
        self._visible: NDArray[np.bool_] = np.zeros(_COUNT, dtype=bool)
        """Which keypoints are shown, whatever their current score."""
        self._shown: NDArray[np.int64] = np.zeros(_COUNT, dtype=np.int64)
        """Consecutive frames at or above ``show_score``."""
        self._hidden: NDArray[np.int64] = np.zeros(_COUNT, dtype=np.int64)
        """Consecutive frames below ``hide_score``."""
        self._outlier: NDArray[np.bool_] = np.zeros(_COUNT, dtype=bool)
        """Keypoints whose last jump is waiting for a second frame."""
        self._skeleton_shown = 0
        self._skeleton_hidden = 0
        self._present = False
        self._last_t: float | None = None
        self._last_present: float | None = None

    def reset(self) -> None:
        """Forget every filter, for example on a model swap or a camera reopen."""
        self._blank()

    @property
    def present(self) -> bool:
        """Whether the whole skeleton is shown."""
        return self._present

    @property
    def visible(self) -> NDArray[np.bool_]:
        """A copy of the shown mask, in :data:`cpsa.movenet.KEYPOINTS` order."""
        return cast(NDArray[np.bool_], self._visible.copy())

    def update(
        self,
        pose: MovenetPose,
        timestamp: float,
        *,
        size: tuple[int, int],
    ) -> MovenetPose:
        """Stabilize one decoded pose and return what to draw and to publish.

        Args:
            pose: The raw pose of this frame.
            timestamp: The frame's monotonic capture time, in seconds.
            size: ``(width, height)`` of the source frame, in pixels; its
                diagonal sets the outlier limit.

        Returns:
            The pose to overlay and to publish: ``x, y`` are the shown pixels
            (held while the keypoint is hidden), a visible keypoint carries at
            least ``show_score`` and an invisible one ``0.0``, and ``score`` is
            the mean of the published keypoint scores -- ``0.0`` while the
            skeleton is not present.

        Raises:
            ValueError: If ``pose.keypoints`` is not ``(17, 3)``.
        """

        config = self._config
        keypoints = np.asarray(pose.keypoints, dtype=np.float64)
        if not config.enabled:
            return pose
        if keypoints.shape != (_COUNT, 3):
            raise ValueError(f"keypoints must be ({_COUNT}, 3), got {keypoints.shape}")

        if self._last_present is not None and timestamp - self._last_present > RESET_AFTER_S:
            self.reset()
            self._last_present = timestamp
        dt = _MIN_DT_S if self._last_t is None else max(timestamp - self._last_t, _MIN_DT_S)
        self._last_t = timestamp

        scores = keypoints[:, 2]
        self._advance_presence(float(pose.score))
        if self._present:
            self._last_present = timestamp
        self._advance_keypoints(scores)
        if not self._present:
            # The skeleton is hidden: so is every joint, whatever its own score.
            self._visible[:] = False

        self._filter(
            keypoints[:, :2],
            self._present & (scores >= config.hide_score),
            dt,
            math.hypot(*size),
        )
        self._move()

        published = np.empty((_COUNT, 3), dtype=np.float32)
        published[:, :2] = self._displayed
        published[:, 2] = np.where(self._visible, np.maximum(scores, config.show_score), 0.0)
        return MovenetPose(keypoints=published, score=float(np.mean(published[:, 2])))

    # -- presence ----------------------------------------------------------

    def _advance_presence(self, score: float) -> None:
        """Run the whole-skeleton hysteresis on the pose score."""
        config = self._config
        self._skeleton_shown = self._skeleton_shown + 1 if score >= config.show_score else 0
        self._skeleton_hidden = self._skeleton_hidden + 1 if score < config.hide_score else 0
        if not self._present:
            if self._skeleton_shown >= config.show_frames:
                self._present = True
                self._skeleton_hidden = 0
        elif self._skeleton_hidden >= config.hide_frames:
            self._present = False
            self._skeleton_shown = 0

    def _advance_keypoints(self, scores: NDArray[np.float64]) -> None:
        """Run the per-keypoint hysteresis on this frame's scores."""
        config = self._config
        self._shown = np.where(scores >= config.show_score, self._shown + 1, 0)
        self._hidden = np.where(scores < config.hide_score, self._hidden + 1, 0)
        appeared = ~self._visible & (self._shown >= config.show_frames)
        vanished = self._visible & (self._hidden >= config.hide_frames)
        self._visible = (self._visible | appeared) & ~vanished
        self._hidden[appeared] = 0
        self._shown[vanished] = 0

    # -- position ----------------------------------------------------------

    def _filter(
        self,
        raw: NDArray[np.float64],
        measured: NDArray[np.bool_],
        dt: float,
        diagonal: float,
    ) -> None:
        """Advance the One Euro filter of every measured keypoint by ``dt``."""
        config = self._config
        distance = np.hypot(
            raw[:, 0] - self._filtered[:, 0],
            raw[:, 1] - self._filtered[:, 1],
        )
        jump = self._ready & measured & (distance > OUTLIER_FRACTION * diagonal)
        held = jump & ~self._outlier
        """A jump whose second frame has not arrived yet."""
        snapped = jump & self._outlier
        """A jump confirmed by a second frame: accepted at once, not smeared."""
        accept = measured & ~held
        self._outlier = np.where(measured, jump, False)
        # A confirmed jump restarts the filter at the new position: easing into
        # it over the One Euro time constant would lag the body for half a second.
        self._ready &= ~snapped

        fresh = accept & ~self._ready
        self._filtered[fresh] = raw[fresh]
        self._speed[fresh] = 0.0
        self._ready |= fresh
        advance = accept & ~fresh
        if advance.any():
            tau = 1.0 / (2.0 * math.pi * config.d_cutoff)
            speed_alpha = 1.0 / (1.0 + tau / dt)
            delta = (raw - self._filtered) / dt
            self._speed[advance] += speed_alpha * (delta[advance] - self._speed[advance])
            cutoff = config.min_cutoff + config.beta * np.abs(self._speed)
            alpha = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
            self._filtered[advance] += alpha[advance] * (raw[advance] - self._filtered[advance])

    def _move(self) -> None:
        """Move the displayed position of the keypoints that left the deadband."""
        threshold = self._deadband()
        moved = np.hypot(
            self._filtered[:, 0] - self._displayed[:, 0],
            self._filtered[:, 1] - self._displayed[:, 1],
        )
        move = self._ready & (~self._visible | ~self._placed | (moved > threshold))
        self._displayed[move] = self._filtered[move]
        self._placed |= move

    def _deadband(self) -> float:
        """The position deadband: ``deadband_px`` or 1 % of the shoulder width."""
        config = self._config
        if not (self._present and self._visible[_LEFT_SHOULDER] and self._visible[_RIGHT_SHOULDER]):
            return config.deadband_px
        width = math.hypot(
            float(self._filtered[_RIGHT_SHOULDER, 0] - self._filtered[_LEFT_SHOULDER, 0]),
            float(self._filtered[_RIGHT_SHOULDER, 1] - self._filtered[_LEFT_SHOULDER, 1]),
        )
        return max(config.deadband_px, SHOULDER_FRACTION * width)
