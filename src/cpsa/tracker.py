"""Box stabilization between the detector and the overlay.

A raw YOLOX box jitters by a few pixels from frame to frame even when nothing
moves, and a one-frame miss makes it blink. :class:`Tracker` keeps a track per
object: detections of the same class are joined to tracks by IoU, the track
filters its box with an adaptive EMA and only moves the *displayed* box when
the filter has really left a deadband around it. The displayed box is therefore
pixel-identical frame after frame while the object stands still, follows a real
move within a frame or two and survives a short dropout.

The tracker is pure numpy on little ``(4,)`` boxes, so 20 boxes cost a fraction
of the 16.7 ms of a 60 fps frame.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cpsa.config import TrackerConfig
from cpsa.yolox import Detection

__all__ = ["BIRTH_SCORE", "FAST_IOU", "Tracker", "box_iou"]

BIRTH_SCORE = 0.6
"""A new track at or above this score is shown at once instead of after ``min_hits``."""

FAST_IOU = 0.5
"""Below this IoU with the filtered box the track uses ``fast_alpha``."""

MAX_SHIFT_FRACTION = 0.04
"""Centre shift, as a fraction of the displayed box diagonal, that leaves the deadband."""

MAX_SIZE_FRACTION = 0.08
"""Relative width or height change that leaves the deadband."""

Box = tuple[float, float, float, float]
BoxLike = Sequence[float] | NDArray[np.float64]


def box_iou(a: BoxLike, b: BoxLike) -> float:
    """Intersection over union of two ``x1, y1, x2, y2`` boxes."""

    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    width = min(ax2, bx2) - max(ax1, bx1)
    height = min(ay2, by2) - max(ay1, by1)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _box(box: BoxLike) -> Box:
    """Return a detection box as four floats."""

    x1, y1, x2, y2 = (float(value) for value in box)
    return (x1, y1, x2, y2)


def _iou_matrix(
    detections: NDArray[np.float64], tracks: NDArray[np.float64]
) -> NDArray[np.float64]:
    """IoU of every detection box against every track box."""

    ax1, ay1, ax2, ay2 = (
        detections[:, 0][:, None],
        detections[:, 1][:, None],
        detections[:, 2][:, None],
        detections[:, 3][:, None],
    )
    bx1, by1, bx2, by2 = (
        tracks[:, 0][None, :],
        tracks[:, 1][None, :],
        tracks[:, 2][None, :],
        tracks[:, 3][None, :],
    )
    width = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    height = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    intersection = width * height
    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)
    union = area_a + area_b - intersection
    scores = np.zeros(intersection.shape, dtype=np.float64)
    np.divide(intersection, union, out=scores, where=union > 0.0)
    return scores


def _diagonal_iou(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """IoU of row ``i`` of ``a`` against row ``i`` of ``b``, for every row."""

    width = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0.0, None)
    height = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0.0, None)
    intersection = width * height
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a + area_b - intersection
    scores = np.zeros(intersection.shape, dtype=np.float64)
    np.divide(intersection, union, out=scores, where=union > 0.0)
    return scores


def _left_deadband(
    displayed: BoxLike, filtered: BoxLike, overlap: float, deadband_iou: float
) -> bool:
    """Whether ``filtered`` differs from ``displayed`` enough to move the box."""

    if overlap < deadband_iou:
        return True
    width = displayed[2] - displayed[0]
    height = displayed[3] - displayed[1]
    diagonal = math.hypot(width, height)
    centre_x = (displayed[0] + displayed[2]) / 2.0
    centre_y = (displayed[1] + displayed[3]) / 2.0
    filtered_x = (filtered[0] + filtered[2]) / 2.0
    filtered_y = (filtered[1] + filtered[3]) / 2.0
    if diagonal > 0.0 and math.hypot(filtered_x - centre_x, filtered_y - centre_y) > (
        MAX_SHIFT_FRACTION * diagonal
    ):
        return True
    filtered_width = filtered[2] - filtered[0]
    filtered_height = filtered[3] - filtered[1]
    if width > 0.0 and abs(filtered_width - width) > MAX_SIZE_FRACTION * width:
        return True
    return height > 0.0 and abs(filtered_height - height) > MAX_SIZE_FRACTION * height


@dataclass
class _Track:
    """One object followed across frames."""

    class_id: int
    filtered: NDArray[np.float64]
    displayed: Box
    score: float
    hits: int
    shown: bool
    misses: int = 0


class Tracker:
    """Stabilize detector boxes with per-class IoU tracks and a deadband.

    Args:
        config: The validated ``tracker`` section. ``enabled: false`` returns
            the raw detections unchanged.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracks: list[_Track] = []
        self._delta = np.zeros(4, dtype=np.float64)

    @property
    def tracks(self) -> int:
        """Number of live tracks, shown or still being confirmed."""
        return len(self._tracks)

    def reset(self) -> None:
        """Forget every track, for example when the camera is reopened."""
        self._tracks.clear()

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        """Match ``detections`` to the tracks and return the boxes to display."""
        config = self._config
        if not config.enabled:
            return list(detections)
        old_count = len(self._tracks)
        boxes = np.array([detection.box for detection in detections], dtype=np.float64).reshape(
            -1, 4
        )
        scores = self._overlaps(boxes, old_count)
        matched = self._match(scores, detections, old_count)
        changed: list[tuple[_Track, bool]] = []
        for index, detection in enumerate(detections):
            track_index = matched.get(index)
            if track_index is None:
                self._tracks.append(self._birth(detection))
                continue
            track = self._tracks[track_index]
            was_shown = track.shown
            self._filter(track, boxes[index], scores[index, track_index], float(detection.score))
            changed.append((track, was_shown))
        matched_tracks = set(matched.values())
        for track_index in range(old_count):
            if track_index not in matched_tracks:
                track = self._tracks[track_index]
                track.misses += 1
                track.hits = 0
        self._tracks = [track for track in self._tracks if track.misses <= config.max_misses]
        self._move_displayed(changed)
        return self._published()

    def _overlaps(self, boxes: NDArray[np.float64], track_count: int) -> NDArray[np.float64]:
        """IoU of every detection against every live track."""
        if track_count == 0:
            return np.empty((0, 0))
        filtered = np.stack([track.filtered for track in self._tracks])
        return _iou_matrix(boxes, filtered)

    def _match(
        self,
        scores: NDArray[np.float64],
        detections: Sequence[Detection],
        track_count: int,
    ) -> dict[int, int]:
        """Greedy same-class IoU matching: detection index to track index."""
        config = self._config
        if not detections or track_count == 0:
            return {}
        classes = np.fromiter(
            (track.class_id for track in self._tracks), dtype=np.int64, count=track_count
        )
        detection_classes = np.fromiter(
            (int(detection.class_id) for detection in detections), dtype=np.int64
        )
        scores[detection_classes[:, None] != classes[None, :]] = -1.0
        matched: dict[int, int] = {}
        used: set[int] = set()
        for flat in np.argsort(-scores, axis=None, kind="stable"):
            detection_index, track_index = divmod(int(flat), track_count)
            if scores[detection_index, track_index] < config.iou_match:
                break
            if detection_index in matched or track_index in used:
                continue
            matched[detection_index] = track_index
            used.add(track_index)
        return matched

    def _birth(self, detection: Detection) -> _Track:
        """Start a track on an unmatched detection."""
        config = self._config
        box = _box(detection.box)
        score = float(detection.score)
        high = score >= BIRTH_SCORE
        return _Track(
            class_id=int(detection.class_id),
            filtered=np.asarray(box, dtype=np.float64),
            displayed=box,
            score=score,
            hits=1,
            shown=high or config.min_hits <= 1,
        )

    def _filter(
        self,
        track: _Track,
        box: NDArray[np.float64],
        overlap: float,
        score: float,
    ) -> None:
        """Filter one matched track; the displayed box is moved later, together."""
        config = self._config
        alpha = config.fast_alpha if overlap < FAST_IOU else config.alpha
        np.subtract(box, track.filtered, out=self._delta)
        self._delta *= alpha
        track.filtered += self._delta
        track.score += config.alpha * (score - track.score)
        track.hits += 1
        track.misses = 0
        track.shown = track.shown or track.hits >= config.min_hits

    def _move_displayed(self, changed: Sequence[tuple[_Track, bool]]) -> None:
        """Move the displayed boxes of the tracks that left their deadband."""
        shown = [(track, was_shown) for track, was_shown in changed if track.shown]
        if not shown:
            return
        filtered = np.stack([track.filtered for track, _ in shown])
        displayed = np.asarray([track.displayed for track, _ in shown], dtype=np.float64)
        overlaps = _diagonal_iou(displayed, filtered)
        for (track, was_shown), overlap in zip(shown, overlaps, strict=True):
            if not was_shown or _left_deadband(
                track.displayed, track.filtered, float(overlap), self._config.deadband_iou
            ):
                track.displayed = _as_box(track.filtered)

    def _published(self) -> list[Detection]:
        """The detections to overlay: one per shown track, in track order."""
        return [
            Detection(box=track.displayed, score=track.score, class_id=track.class_id)
            for track in self._tracks
            if track.shown
        ]


def _as_box(box: BoxLike) -> Box:
    """Return a numpy box as the tuple :class:`Detection` carries."""

    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
