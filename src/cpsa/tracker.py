"""Box stabilization between the detector and the overlay.

A raw YOLOX box jitters by a few pixels from frame to frame even when nothing
moves, and a one-frame miss makes it blink. :class:`Tracker` keeps a track per
object: detections are joined to tracks by IoU, the track filters its box with
an adaptive EMA and only moves the *displayed* box when the filter has really
left a deadband around it. The displayed box is therefore pixel-identical frame
after frame while the object stands still, follows a real move within a frame or
two and survives a short dropout.

Two detector habits are handled here too. A class that flickers between ``cup``
and ``bowl`` would otherwise kill the track and create a new one every few
frames, so a track keeps the majority class of its last 15 matched frames and
only relabels itself when the new class holds 70% of them for half a second.
And a track whose displayed box has held still for a second is *settled*: it
gets a wider deadband, a 5%-quantized score and thirty missed frames of life,
which is what keeps a table of still objects rock steady.

The tracker is pure numpy on little ``(4,)`` boxes, so 20 boxes cost a fraction
of the 16.7 ms of a 60 fps frame.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from cpsa.config import TrackerConfig
from cpsa.yolox import Detection

__all__ = [
    "BIRTH_SCORE",
    "CLASS_SWITCH_S",
    "FAST_IOU",
    "MATCH_IOU",
    "SCORE_STEP",
    "SETTLED_MAX_MISSES",
    "STILL_S",
    "Tracker",
    "VOTE_FRAMES",
    "box_iou",
]

BIRTH_SCORE = 0.6
"""A new track at or above this score is shown at once instead of after ``min_hits``."""

FAST_IOU = 0.5
"""Below this IoU with the filtered box the track uses ``fast_alpha``."""

MATCH_IOU = 0.5
"""IoU at which a detection of another class may still join a track."""

CLASS_PREFERENCE = 1.0
"""Added to a same-class pair: an agreeing class wins over any IoU advantage."""

VOTE_FRAMES = 15
"""Matched frames the class vote of a track looks at."""

VOTE_SHARE = 0.70
"""Share of the vote window a new class needs before it can relabel the track."""

CLASS_SWITCH_S = 0.5
"""How long a new class must keep :data:`VOTE_SHARE` before the label changes."""

STILL_S = 1.0
"""A track whose displayed box has held this long becomes *settled*."""

SETTLED_DEADBAND_IOU = 0.80
"""Deadband of a settled track: wider, so an unmoved box never twitches."""

SETTLED_SHIFT_FRACTION = 0.08
"""Centre shift, as a fraction of the diagonal, that unsettles a settled track."""

SETTLED_SIZE_FRACTION = 0.12
"""Relative side change that unsettles a settled track."""

SETTLED_ESCAPE_FRAMES = 3
"""Consecutive frames outside the settled deadband before the box follows."""

SETTLED_MAX_MISSES = 30
"""Missed frames a settled track survives: half a second of a silent detector."""

SETTLED_SCORE_ALPHA = 0.05
"""Score EMA weight of a settled track: the label of a still box barely moves."""

SCORE_STEP = 0.05
"""The published score of a settled track is rounded to this step."""

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
    displayed: BoxLike,
    filtered: BoxLike,
    overlap: float,
    deadband_iou: float,
    shift_fraction: float = MAX_SHIFT_FRACTION,
    size_fraction: float = MAX_SIZE_FRACTION,
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
        shift_fraction * diagonal
    ):
        return True
    filtered_width = filtered[2] - filtered[0]
    filtered_height = filtered[3] - filtered[1]
    if width > 0.0 and abs(filtered_width - width) > size_fraction * width:
        return True
    return height > 0.0 and abs(filtered_height - height) > size_fraction * height


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
    votes: deque[int] = field(default_factory=lambda: deque(maxlen=VOTE_FRAMES))
    pending_class: int | None = None
    pending_since: float = 0.0
    displayed_since: float = 0.0
    settled: bool = False
    change_frames: int = 0


class Tracker:
    """Stabilize detector boxes with IoU tracks, a class vote and a deadband.

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

    def update(self, detections: Sequence[Detection], now: float | None = None) -> list[Detection]:
        """Match ``detections`` to the tracks and return the boxes to display.

        Args:
            detections: The detector output of one frame.
            now: ``time.monotonic()`` of the frame; the default reads the clock.
                The class-switch and settle timers run on it, so a frame rate
                that changes (the still gate skips frames) cannot stretch them.
        """

        config = self._config
        if not config.enabled:
            return list(detections)
        when = time.monotonic() if now is None else now
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
                self._tracks.append(self._birth(detection, when))
                continue
            track = self._tracks[track_index]
            was_shown = track.shown
            self._filter(
                track,
                boxes[index],
                scores[index, track_index],
                float(detection.score),
                int(detection.class_id),
                when,
            )
            changed.append((track, was_shown))
        matched_tracks = set(matched.values())
        for track_index in range(old_count):
            if track_index not in matched_tracks:
                track = self._tracks[track_index]
                track.misses += 1
                track.hits = 0
        self._tracks = [track for track in self._tracks if track.misses <= self._miss_limit(track)]
        self._move_displayed(changed, when)
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
        """Greedy IoU matching across classes, an agreeing class first.

        A detection joins the track it overlaps most, whatever its class, as
        long as the IoU is at least :data:`MATCH_IOU` (``iou_match`` when the
        class agrees): the label is the track's business, not the matcher's, so
        a detector that calls a cup a bowl keeps the track instead of replacing
        it. ``scores`` is left untouched; the caller needs the raw IoU.
        """
        config = self._config
        if not detections or track_count == 0:
            return {}
        classes = np.fromiter(
            (track.class_id for track in self._tracks), dtype=np.int64, count=track_count
        )
        detection_classes = np.fromiter(
            (int(detection.class_id) for detection in detections), dtype=np.int64
        )
        same_class = detection_classes[:, None] == classes[None, :]
        allowed = np.where(same_class, scores >= config.iou_match, scores >= MATCH_IOU)
        preference = np.where(same_class, scores + CLASS_PREFERENCE, scores)
        preference[~allowed] = -1.0
        matched: dict[int, int] = {}
        used: set[int] = set()
        for flat in np.argsort(-preference, axis=None, kind="stable"):
            detection_index, track_index = divmod(int(flat), track_count)
            if preference[detection_index, track_index] < 0.0:
                break
            if detection_index in matched or track_index in used:
                continue
            matched[detection_index] = track_index
            used.add(track_index)
        return matched

    def _birth(self, detection: Detection, now: float) -> _Track:
        """Start a track on an unmatched detection."""
        config = self._config
        box = _box(detection.box)
        score = float(detection.score)
        class_id = int(detection.class_id)
        high = score >= BIRTH_SCORE
        return _Track(
            class_id=class_id,
            filtered=np.asarray(box, dtype=np.float64),
            displayed=box,
            score=score,
            hits=1,
            shown=high or config.min_hits <= 1,
            votes=deque([class_id], maxlen=VOTE_FRAMES),
            displayed_since=now,
        )

    def _filter(
        self,
        track: _Track,
        box: NDArray[np.float64],
        overlap: float,
        score: float,
        class_id: int,
        now: float,
    ) -> None:
        """Filter one matched track; the displayed box is moved later, together."""
        config = self._config
        alpha = config.fast_alpha if overlap < FAST_IOU else config.alpha
        np.subtract(box, track.filtered, out=self._delta)
        self._delta *= alpha
        track.filtered += self._delta
        score_alpha = SETTLED_SCORE_ALPHA if track.settled else config.alpha
        track.score += score_alpha * (score - track.score)
        track.hits += 1
        track.misses = 0
        track.shown = track.shown or track.hits >= config.min_hits
        self._vote(track, class_id, now)

    def _vote(self, track: _Track, class_id: int, now: float) -> None:
        """Relabel a track whose detections changed class, without killing it."""
        track.votes.append(class_id)
        leader, count = Counter(track.votes).most_common(1)[0]
        if leader == track.class_id or count < VOTE_SHARE * len(track.votes):
            track.pending_class = None
            return
        if track.pending_class != leader:
            track.pending_class = leader
            track.pending_since = now
        if now - track.pending_since >= CLASS_SWITCH_S:
            track.class_id = leader
            track.pending_class = None

    def _miss_limit(self, track: _Track) -> int:
        """How many missed frames a track survives; settled tracks live longer."""
        if track.settled:
            return max(SETTLED_MAX_MISSES, self._config.max_misses)
        return self._config.max_misses

    def _move_displayed(self, changed: Sequence[tuple[_Track, bool]], now: float) -> None:
        """Move the displayed boxes of the tracks that left their deadband."""
        shown = [(track, was_shown) for track, was_shown in changed if track.shown]
        if not shown:
            return
        filtered = np.stack([track.filtered for track, _ in shown])
        displayed = np.asarray([track.displayed for track, _ in shown], dtype=np.float64)
        overlaps = _diagonal_iou(displayed, filtered)
        for (track, was_shown), overlap in zip(shown, overlaps, strict=True):
            if not was_shown:
                self._display(track, now)
                continue
            settled = track.settled
            left = _left_deadband(
                track.displayed,
                track.filtered,
                float(overlap),
                SETTLED_DEADBAND_IOU if settled else self._config.deadband_iou,
                SETTLED_SHIFT_FRACTION if settled else MAX_SHIFT_FRACTION,
                SETTLED_SIZE_FRACTION if settled else MAX_SIZE_FRACTION,
            )
            if not left:
                track.change_frames = 0
                self._settle(track, now)
                continue
            if settled and track.change_frames + 1 < SETTLED_ESCAPE_FRAMES:
                # A settled box only follows a change that lasts: one frame of
                # detector noise on a still object must not move it.
                track.change_frames += 1
                continue
            if settled:
                track.settled = False
            self._display(track, now)

    def _display(self, track: _Track, now: float) -> None:
        """Snap the displayed box to the filtered one and restart its still timer."""
        track.displayed = _as_box(track.filtered)
        track.displayed_since = now
        track.change_frames = 0

    def _settle(self, track: _Track, now: float) -> None:
        """Mark a track settled once its displayed box has held for :data:`STILL_S`."""
        if not track.settled and now - track.displayed_since >= STILL_S:
            track.settled = True
            track.change_frames = 0

    def _published(self) -> list[Detection]:
        """The detections to overlay: one per shown track, in track order."""
        return [
            Detection(box=track.displayed, score=_shown_score(track), class_id=track.class_id)
            for track in self._tracks
            if track.shown
        ]


def _shown_score(track: _Track) -> float:
    """The score to draw: the plain EMA, or a settled track's in 5% steps."""
    if not track.settled:
        return track.score
    return round(track.score / SCORE_STEP) * SCORE_STEP


def _as_box(box: BoxLike) -> Box:
    """Return a numpy box as the tuple :class:`Detection` carries."""

    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
