"""The cheap motion check that keeps a still scene off the DPU.

The detector is the expensive part of a frame, but a booth often stands still
for minutes: the same objects, the same boxes, the same classes. Comparing the
frame with the last one that was actually inferred answers the only question
that matters -- did anything move? -- with two resizes and one small absolute
difference, a fraction of the preprocessing and decode of a real frame.

The score is measured on an 80x60 grey thumbnail. It averages sixteen source
pixels per cell, which is what keeps ordinary sensor noise below the motion
threshold, but it does so in two steps: the cheap bilinear resize decimates the
frame first and only the 320x240 result is area-averaged. A single full-frame
``INTER_AREA`` pass reads the whole 921 KB frame with the expensive box filter
and costs several times more than the rest of the gate together.

:class:`StillGate` holds the answer's state: the last inferred grey frame, the
count of still frames, and when the last real detection ran. It is deliberately
pessimistic: three still frames in a row before the first skip, and a forced
detection every ``force_every_s`` seconds, so an object that appears without
moving the frame cannot go unnoticed for longer than that.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ["MOTION_SIZE", "STILL_FRAMES", "StillGate"]

MOTION_SIZE = (80, 60)
"""``(width, height)`` the grey frame shrinks to before the difference."""

MOTION_HALF_SIZE = (320, 240)
"""``(width, height)`` of the bilinear decimation step before the area average."""

STILL_FRAMES = 3
"""Consecutive still frames before the loop may skip the DPU."""


def small_gray(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return ``rgb`` as the 80x60 grey image the difference is measured on.

    The frame is decimated to :data:`MOTION_HALF_SIZE` with the cheap bilinear
    filter first, then the 320x240 grey image is area-averaged down to 80x60.
    The bilinear step samples four pixels per cell, the area step averages the
    sixteen pixels of the 4x4 block, so a still frame's sensor noise is divided
    by four while a real move still shifts every cell.
    """

    half = cv2.resize(
        rgb,
        (min(MOTION_HALF_SIZE[0], rgb.shape[1]), min(MOTION_HALF_SIZE[1], rgb.shape[0])),
        interpolation=cv2.INTER_LINEAR,
    )
    sized = cv2.resize(
        cv2.cvtColor(half, cv2.COLOR_RGB2GRAY), MOTION_SIZE, interpolation=cv2.INTER_AREA
    )
    return np.asarray(sized, dtype=np.uint8)


class StillGate:
    """Decide, frame by frame, whether the scene has stopped moving.

    The motion score of a frame is the mean absolute difference between its
    80x60 grey image and the one of the last inferred frame, in grey levels:
    sensor noise lands around 0.5, a person moving across the booth above 1.

    The scene's stillness is shared: one reference frame and one still counter
    for every frame the loop hands over. Only the forced detection runs per
    *lane* -- the alternating YOLOX schedule runs two kinds of pass, so each
    needs its own deadline or one pass would refresh only every other time.

    Args:
        motion_thr: Score below which the scene counts as still.
        force_every_s: Longest time without a full detection, per lane.
    """

    def __init__(self, motion_thr: float, force_every_s: float) -> None:
        self._motion_thr = float(motion_thr)
        self._force_every_s = float(force_every_s)
        self._reference: NDArray[np.uint8] | None = None
        self._still = 0
        self._last_infer: dict[int, float] = {}
        self._score = math.inf

    @property
    def score(self) -> float:
        """Motion score of the last frame, in grey levels (``inf`` before the first)."""
        return self._score

    @property
    def still_frames(self) -> int:
        """Consecutive still frames seen so far."""
        return self._still

    def reset(self) -> None:
        """Forget the reference frame, for example when the camera is reopened."""
        self._reference = None
        self._still = 0
        self._last_infer.clear()
        self._score = math.inf

    def update(self, rgb: NDArray[np.uint8], now: float, *, lane: int = 0) -> bool:
        """Record one frame and return ``True`` when it must go to the detector.

        Args:
            rgb: The frame the loop decoded, ``(height, width, 3)`` uint8.
            now: ``time.monotonic()`` of the frame, for the refresh timer.
            lane: Which pass of the loop this frame belongs to. Each lane has
                its own forced-refresh deadline; ``0`` is the only lane when
                the loop runs one kind of pass.

        Returns:
            ``True`` to preprocess, infer and decode the frame, ``False`` to
            draw the previous boxes again.
        """

        small = small_gray(rgb)
        reference = self._reference
        self._score = math.inf if reference is None else float(cv2.absdiff(small, reference).mean())
        still = self._score < self._motion_thr
        self._still = self._still + 1 if still else 0
        forced = now - self._last_infer.get(lane, 0.0) >= self._force_every_s
        if not still or self._still < STILL_FRAMES or forced:
            self._reference = small
            self._last_infer[lane] = now
            return True
        return False
