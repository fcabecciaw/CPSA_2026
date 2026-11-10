"""Subject selection, face width and framing state (FR-005, FR-006, FR-012).

The pipeline analyzes one seated subject. Selection follows a configured rule
(``largest`` or ``central``). The face width in pixels is measured from the
ears, the eyes or the jaw and stabilized with a median over recent frames; it
is the scale that turns pixels into face-widths for every downstream feature.

Framing never uses lower-body keypoints. A half-body frame therefore keeps the
upper-body confidences unchanged, and a full-body frame is recognized from the
person-box height alone (FR-005).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cpsa.vision.types import FramingState, KeypointLayout, PersonPose

__all__ = ["SubjectConfig", "framing_state", "select_subject"]

_FACE_ANCHOR_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
_EYE_NAMES = ("left_eye", "right_eye")
_EAR_NAMES = ("left_ear", "right_ear")
_JAW_NAMES = ("face_jaw_left", "face_jaw_right")
_WRIST_NAMES = ("left_wrist", "right_wrist")
_SUBJECT_RULES = frozenset({"largest", "central"})
_FACE_WIDTH_SOURCES = frozenset({"ears", "eyes", "jaw", "auto"})
_TINY = 1e-9


@dataclass(frozen=True)
class SubjectConfig:
    """Subject-selection, framing and face-width settings.

    ``rule`` picks the analyzed person when several are detected: ``largest``
    takes the box with the largest area, ``central`` the box whose center is
    closest to the image center (ties broken by area). ``full_body_ratio`` is
    the person-box height divided by the face width at or above which a frame
    counts as ``FULL_BODY``. ``face_anchor_conf`` accepts a face anchor as
    visible and ``wrist_conf`` a wrist as visible. ``face_width_source`` is one
    of ``ears``, ``eyes``, ``jaw`` or ``auto`` (jaw, then ears, then eyes).
    ``eye_to_face_ratio`` converts inter-ocular distance to face width, and
    ``face_width_window`` is the median window in frames.
    """

    rule: str = "largest"
    full_body_ratio: float = 9.0
    face_anchor_conf: float = 0.3
    wrist_conf: float = 0.3
    face_width_source: str = "auto"
    eye_to_face_ratio: float = 2.2
    face_width_window: int = 15

    def __post_init__(self) -> None:
        if self.rule not in _SUBJECT_RULES:
            raise ValueError(
                f"unknown subject rule {self.rule!r}; expected one of {sorted(_SUBJECT_RULES)}"
            )
        if not self.full_body_ratio > 0.0:
            raise ValueError("full_body_ratio must be positive")
        if not 0.0 < self.face_anchor_conf <= 1.0:
            raise ValueError("face_anchor_conf must be in (0, 1]")
        if not 0.0 < self.wrist_conf <= 1.0:
            raise ValueError("wrist_conf must be in (0, 1]")
        if self.face_width_source not in _FACE_WIDTH_SOURCES:
            raise ValueError(
                f"unknown face width source {self.face_width_source!r}; "
                f"expected one of {sorted(_FACE_WIDTH_SOURCES)}"
            )
        if not self.eye_to_face_ratio > 0.0:
            raise ValueError("eye_to_face_ratio must be positive")
        if self.face_width_window < 1:
            raise ValueError("face_width_window must be at least 1")


def _pair(layout: KeypointLayout, names: tuple[str, str]) -> tuple[int, int] | None:
    """Return the two layout indices of a named keypoint pair, if both exist."""
    first, second = names
    if not (layout.has(first) and layout.has(second)):
        return None
    return layout.index(first), layout.index(second)


def _confident(conf: NDArray[np.float64], index: int, threshold: float) -> bool:
    value = conf[index]
    return bool(np.isfinite(value)) and bool(value >= threshold)


def _distance(xy: NDArray[np.float64], first: int, second: int) -> float | None:
    """Euclidean distance between two keypoints, or ``None`` when not finite."""
    a = xy[first]
    b = xy[second]
    if not (bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b)))):
        return None
    distance = float(np.linalg.norm(a - b))
    return distance if distance > _TINY else None


def _person_arrays(
    person: PersonPose, layout: KeypointLayout
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return validated ``(xy, conf)`` arrays of one person."""
    xy = np.asarray(person.xy, dtype=np.float64)
    conf = np.asarray(person.conf, dtype=np.float64)
    size = len(layout)
    if xy.shape != (size, 2):
        raise ValueError(f"person xy must have shape ({size}, 2), got {xy.shape}")
    if conf.shape != (size,):
        raise ValueError(f"person conf must have shape ({size},), got {conf.shape}")
    return xy, conf


def _valid_box(box: NDArray[np.float64]) -> tuple[float, float, float, float] | None:
    """Return ``(x1, y1, x2, y2)`` for a finite, non-empty box, else ``None``."""
    corners = np.asarray(box, dtype=np.float64)
    if corners.shape != (4,) or not bool(np.all(np.isfinite(corners))):
        return None
    x1, y1, x2, y2 = (float(value) for value in corners)
    if abs(x2 - x1) <= _TINY or abs(y2 - y1) <= _TINY:
        return None
    return x1, y1, x2, y2


def _face_width_measurement(
    person: PersonPose,
    layout: KeypointLayout,
    cfg: SubjectConfig,
) -> float | None:
    """Measure the face width in pixels with the configured source.

    ``ears`` is the ear-to-ear distance, ``eyes`` the inter-ocular distance
    scaled by ``eye_to_face_ratio`` and ``jaw`` the jaw-end distance (half-body
    layouts only). ``auto`` tries jaw, ears and eyes in that order and returns
    the first available measurement.
    """
    xy, conf = _person_arrays(person, layout)
    candidates: tuple[tuple[str, float], ...]
    if cfg.face_width_source == "ears":
        candidates = (("ears", 1.0),)
    elif cfg.face_width_source == "eyes":
        candidates = (("eyes", cfg.eye_to_face_ratio),)
    elif cfg.face_width_source == "jaw":
        candidates = (("jaw", 1.0),)
    else:
        candidates = (("jaw", 1.0), ("ears", 1.0), ("eyes", cfg.eye_to_face_ratio))

    for source, scale in candidates:
        if source == "ears":
            pair = _pair(layout, _EAR_NAMES)
        elif source == "eyes":
            pair = _pair(layout, _EYE_NAMES)
        else:
            pair = _pair(layout, _JAW_NAMES)
        if pair is None:
            continue
        first, second = pair
        if not (
            _confident(conf, first, cfg.face_anchor_conf)
            and _confident(conf, second, cfg.face_anchor_conf)
        ):
            continue
        distance = _distance(xy, first, second)
        if distance is not None:
            return distance * scale
    return None


class _FaceWidthTracker:
    """Median face width over the last ``face_width_window`` measurements.

    A frame without a usable measurement keeps the current median, so a short
    face dropout does not change the feature scale. ``reset`` clears the
    window; the tracker calls it between episodes.
    """

    def __init__(self, cfg: SubjectConfig) -> None:
        self._cfg = cfg
        self._samples: deque[float] = deque(maxlen=cfg.face_width_window)

    @property
    def value(self) -> float | None:
        if not self._samples:
            return None
        return float(np.median(np.asarray(self._samples, dtype=np.float64)))

    def update(self, person: PersonPose, layout: KeypointLayout) -> float | None:
        measurement = _face_width_measurement(person, layout, self._cfg)
        if measurement is not None and np.isfinite(measurement):
            self._samples.append(float(measurement))
        return self.value

    def reset(self) -> None:
        self._samples.clear()


def select_subject(
    persons: Sequence[PersonPose],
    width: int,
    height: int,
    cfg: SubjectConfig,
) -> PersonPose | None:
    """Choose the single analyzed person, or ``None`` when no box is valid.

    ``largest`` maximizes the box area (ties broken by detector score).
    ``central`` minimizes the normalized distance from the box center to the
    image center (ties broken by area).
    """
    image_width = max(float(width), 1.0)
    image_height = max(float(height), 1.0)
    diagonal = math.hypot(image_width, image_height)

    best: PersonPose | None = None
    best_key: tuple[float, float] | None = None
    for person in persons:
        box = _valid_box(person.box)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if cfg.rule == "largest":
            key = (area, float(person.score))
        else:
            distance = math.hypot(
                (x1 + x2) / 2.0 - image_width / 2.0, (y1 + y2) / 2.0 - image_height / 2.0
            )
            key = (-distance / diagonal, area)
        if best_key is None or key > best_key:
            best = person
            best_key = key
    return best


def framing_state(
    person: PersonPose | None,
    layout: KeypointLayout,
    face_width_px: float | None,
    cfg: SubjectConfig,
) -> FramingState:
    """Classify the framing of the selected subject (FR-005).

    Face anchors are the eyes, nose and ears. A frame is ``FACE_NOT_VISIBLE``
    when fewer than two of them are confident, ``HANDS_NOT_VISIBLE`` when
    neither wrist is confident, and otherwise ``FULL_BODY`` when the box height
    is at least ``full_body_ratio`` face widths. When the face width is unknown
    the frame is conservatively ``HALF_BODY``. Lower-body keypoints are never
    consulted.
    """
    if person is None:
        return FramingState.NO_PERSON

    _, conf = _person_arrays(person, layout)
    anchors = [layout.index(name) for name in _FACE_ANCHOR_NAMES if layout.has(name)]
    visible_anchors = sum(1 for index in anchors if _confident(conf, index, cfg.face_anchor_conf))
    if visible_anchors < 2:
        return FramingState.FACE_NOT_VISIBLE

    wrists = [layout.index(name) for name in _WRIST_NAMES if layout.has(name)]
    if not any(_confident(conf, index, cfg.wrist_conf) for index in wrists):
        return FramingState.HANDS_NOT_VISIBLE

    box = _valid_box(person.box)
    if (
        box is not None
        and face_width_px is not None
        and np.isfinite(face_width_px)
        and float(face_width_px) > _TINY
    ):
        box_height = box[3] - box[1]
        if box_height / float(face_width_px) >= cfg.full_body_ratio:
            return FramingState.FULL_BODY
    return FramingState.HALF_BODY
