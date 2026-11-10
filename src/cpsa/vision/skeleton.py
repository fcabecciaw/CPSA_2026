"""Session-level bone-length references and the plausibility check (FR-009..FR-011).

Reference lengths are kept in face widths, so they do not change when the
subject moves closer to or farther from the camera. They are collected from
confident frames of the current session, frozen once enough samples exist and
then reused for every following episode (FR-009).

A measurement is excluded before the Kalman update when a bone through it is
longer than ``max_ratio`` times its reference (FR-010). Only the upper bound is
checked: foreshortening shortens a bone but never lengthens it. When a broken
bone makes both of its joints suspect, the lower-confidence joint is rejected
so one plausible endpoint survives. The observed forearm length divided by its
reference is the foreshortening ratio (FR-011).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cpsa.domain.model import Side
from cpsa.vision.types import KeypointLayout

__all__ = ["SkeletonConfig", "SkeletonModel"]

# (reference name, first joint, second joint)
_BONES: tuple[tuple[str, str, str], ...] = (
    ("shoulder_width", "left_shoulder", "right_shoulder"),
    ("left_upper_arm", "left_shoulder", "left_elbow"),
    ("left_forearm", "left_elbow", "left_wrist"),
    ("right_upper_arm", "right_shoulder", "right_elbow"),
    ("right_forearm", "right_elbow", "right_wrist"),
)

_FOREARM_JOINTS: Mapping[Side, tuple[str, str]] = {
    Side.LEFT: ("left_elbow", "left_wrist"),
    Side.RIGHT: ("right_elbow", "right_wrist"),
}
_TINY = 1e-12


@dataclass(frozen=True)
class SkeletonConfig:
    """Reference collection and plausibility limits.

    ``max_ratio`` is the largest accepted bone length as a multiple of its
    reference (default 1.4). ``ref_conf`` is the joint confidence a frame needs
    to contribute a reference sample. ``ref_frames`` bounds the sample buffer,
    ``ref_min_frames`` is the number of samples after which the reference is
    frozen, and ``check_conf`` is the joint confidence needed for a
    plausibility check.
    """

    max_ratio: float = 1.4
    ref_conf: float = 0.5
    ref_frames: int = 60
    ref_min_frames: int = 30
    check_conf: float = 0.3

    def __post_init__(self) -> None:
        if not self.max_ratio > 1.0:
            raise ValueError("max_ratio must be greater than 1")
        if not 0.0 < self.ref_conf <= 1.0:
            raise ValueError("ref_conf must be in (0, 1]")
        if not 0.0 < self.check_conf <= 1.0:
            raise ValueError("check_conf must be in (0, 1]")
        if self.ref_frames < 1:
            raise ValueError("ref_frames must be at least 1")
        if self.ref_min_frames < 1:
            raise ValueError("ref_min_frames must be at least 1")
        if self.ref_min_frames > self.ref_frames:
            raise ValueError("ref_min_frames must not exceed ref_frames")


class SkeletonModel:
    """Reference bone lengths in face widths plus the plausibility mask."""

    def __init__(self, cfg: SkeletonConfig, layout: KeypointLayout) -> None:
        self._cfg = cfg
        self._layout = layout
        self._bones: tuple[tuple[str, int, int], ...] = tuple(
            (name, layout.index(first), layout.index(second))
            for name, first, second in _BONES
            if layout.has(first) and layout.has(second)
        )
        self._samples: dict[str, deque[float]] = {
            name: deque(maxlen=cfg.ref_frames) for name, _, _ in self._bones
        }
        self._frozen: dict[str, bool] = {name: False for name, _, _ in self._bones}
        self._forearms: dict[Side, tuple[int, int] | None] = {
            side: self._joint_pair(names) for side, names in _FOREARM_JOINTS.items()
        }
        self._face_width_px: float | None = None

    def _joint_pair(self, names: tuple[str, str]) -> tuple[int, int] | None:
        first, second = names
        if not (self._layout.has(first) and self._layout.has(second)):
            return None
        return self._layout.index(first), self._layout.index(second)

    @property
    def reference_fw(self) -> Mapping[str, float]:
        """Reference lengths in face widths, keyed by bone name.

        The value is the median of the samples collected so far; it stops
        changing once the bone reference is frozen. A bone is absent until its
        first sample. The mapping is a copy; mutating it does not change the
        model.
        """
        references: dict[str, float] = {}
        for name, samples in self._samples.items():
            if samples:
                references[name] = float(np.median(np.asarray(samples, dtype=np.float64)))
        return references

    def reset(self) -> None:
        """Forget the references; the next session recollects them."""
        for name in self._samples:
            self._samples[name].clear()
            self._frozen[name] = False
        self._face_width_px = None

    def state(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of the session-level references.

        Audit replay restores this snapshot at episode start, so the first
        episode published after it sees the same bone references as the live
        session did.
        """
        return {
            "samples": {
                name: [float(value) for value in samples] for name, samples in self._samples.items()
            },
            "frozen": dict(self._frozen),
            "face_width_px": self._face_width_px,
        }

    def restore(self, state: Mapping[str, object]) -> None:
        """Restore a snapshot produced by :meth:`state`."""
        samples = state.get("samples")
        frozen = state.get("frozen")
        if not isinstance(samples, Mapping) or not isinstance(frozen, Mapping):
            raise ValueError("skeleton state needs 'samples' and 'frozen' mappings")
        for name, values in samples.items():
            if not isinstance(name, str) or name not in self._samples:
                continue
            if not isinstance(values, list) or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
            ):
                raise ValueError(f"invalid skeleton samples for {name}")
            buffer = self._samples[name]
            buffer.clear()
            buffer.extend(float(value) for value in values)
        for name, value in frozen.items():
            if not isinstance(name, str) or name not in self._frozen:
                continue
            if not isinstance(value, bool):
                raise ValueError(f"invalid frozen flag for {name}")
            self._frozen[name] = value
        width = state.get("face_width_px")
        if width is not None and (not isinstance(width, (int, float)) or isinstance(width, bool)):
            raise ValueError("face_width_px must be a number or null")
        self._face_width_px = None if width is None else float(width)

    def _points_and_conf(
        self, xy: ArrayLike, conf: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        points = np.asarray(xy, dtype=np.float64)
        confidences = np.asarray(conf, dtype=np.float64)
        size = len(self._layout)
        if points.shape != (size, 2):
            raise ValueError(f"xy must have shape ({size}, 2), got {points.shape}")
        if confidences.shape != (size,):
            raise ValueError(f"conf must have shape ({size},), got {confidences.shape}")
        return points, confidences

    @staticmethod
    def _length_fw(
        points: NDArray[np.float64], first: int, second: int, face_width_px: float
    ) -> float:
        """Return the joint distance in face widths."""
        return float(np.linalg.norm(points[first] - points[second])) / face_width_px

    def _collect(
        self,
        points: NDArray[np.float64],
        confidences: NDArray[np.float64],
        face_width_px: float,
    ) -> None:
        cfg = self._cfg
        for name, first, second in self._bones:
            if self._frozen[name]:
                continue
            if not (confidences[first] >= cfg.ref_conf and confidences[second] >= cfg.ref_conf):
                continue
            if not (
                bool(np.all(np.isfinite(points[first])))
                and bool(np.all(np.isfinite(points[second])))
            ):
                continue
            length = self._length_fw(points, first, second, face_width_px)
            if not np.isfinite(length):
                continue
            self._samples[name].append(length)
            if len(self._samples[name]) >= cfg.ref_min_frames:
                self._frozen[name] = True

    def observe(
        self,
        xy: ArrayLike,
        conf: ArrayLike,
        face_width_px: float | None,
    ) -> NDArray[np.bool_]:
        """Update the references and return the mask of accepted keypoints.

        ``True`` marks a measurement that may be used by the tracker. Once a
        bone reference is frozen, a joint is rejected when a bone through it
        exceeds ``max_ratio`` times that reference; the lower-confidence
        endpoint of a broken bone is the one rejected. Nothing is rejected while
        no reference has been frozen or the face width is unknown.
        """
        points, confidences = self._points_and_conf(xy, conf)
        accepted = (
            np.all(np.isfinite(points), axis=1) & np.isfinite(confidences) & (confidences > 0.0)
        )
        face_width = None
        if face_width_px is not None:
            value = float(face_width_px)
            if np.isfinite(value) and value > _TINY:
                face_width = value
        if face_width is None:
            return accepted

        self._face_width_px = face_width
        self._collect(points, confidences, face_width)
        references = self.reference_fw
        if not references:
            return accepted

        rejected = np.zeros(len(self._layout), dtype=np.bool_)
        for name, first, second in self._bones:
            reference = references.get(name)
            if reference is None or reference <= _TINY:
                continue
            if not self._frozen[name]:
                # Use the running median for reports, but reject only against a
                # reference that has enough samples to be trustworthy.
                continue
            if not (
                confidences[first] >= self._cfg.check_conf
                and confidences[second] >= self._cfg.check_conf
            ):
                continue
            length = self._length_fw(points, first, second, face_width)
            if not np.isfinite(length):
                continue
            if length > self._cfg.max_ratio * reference:
                # Both joints are suspect; keep the more confident one.
                if confidences[first] <= confidences[second]:
                    rejected[first] = True
                else:
                    rejected[second] = True
        accepted &= ~rejected
        return accepted

    def foreshortening(self, xy: ArrayLike) -> Mapping[Side, float | None]:
        """Observed forearm length divided by its reference, per side.

        The observed length is converted to face widths with the face width of
        the last :meth:`observe` call. A side is ``None`` until its bone has a
        reference sample, when no face width has been observed yet, or when the
        elbow or wrist position is not finite.
        """
        points = np.asarray(xy, dtype=np.float64)
        size = len(self._layout)
        if points.shape != (size, 2):
            raise ValueError(f"xy must have shape ({size}, 2), got {points.shape}")
        ratios: dict[Side, float | None] = {Side.LEFT: None, Side.RIGHT: None}
        face_width = self._face_width_px
        if face_width is None or face_width <= _TINY:
            return ratios
        references = self.reference_fw
        for side, pair in self._forearms.items():
            if pair is None:
                continue
            reference = references.get(f"{side.value}_forearm")
            if reference is None or reference <= _TINY:
                continue
            first, second = pair
            if not (
                bool(np.all(np.isfinite(points[first])))
                and bool(np.all(np.isfinite(points[second])))
            ):
                continue
            observed_fw = self._length_fw(points, first, second, face_width)
            ratios[side] = observed_fw / reference
        return ratios
