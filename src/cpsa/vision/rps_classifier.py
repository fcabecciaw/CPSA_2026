"""Rock/paper/scissors classifier over 21 hand keypoints (F-D).

The runtime counterpart of ``cpsa_ml.gestures``: pure NumPy, no framework. The
features are the same version 1 hand normalisation the training code uses
(wrist-centred, rotated so the middle-finger knuckle points up, scaled to that
distance, flattened), so the parity vector stored in the ``.npz`` proves both
sides see the same input.

``.npz`` keys: ``w1 b1 w2 b2 w3 b3 feature_version classes test_input
test_proba``. The network is ``42 -> 64 -> 64 -> 4`` with ReLU and a softmax
over the last layer; :class:`RpsClassifier` accepts any hidden width that the
stored matrices agree on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["FEATURE_VERSION", "RPS_CLASSES", "RpsClassifier", "rps_features"]

RPS_CLASSES: tuple[str, ...] = ("rock", "paper", "scissors", "none")
"""Classifier output order; ``none`` means "not a gesture"."""

FEATURE_VERSION = "v1"
"""Version of :func:`rps_features`; stored in the ``.npz`` as ``feature_version``."""

_KEYPOINTS = 21
_FEATURES = 2 * _KEYPOINTS
_MIDDLE_KNUCKLE = 9
_EPS = 1e-6
_WEIGHT_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3")


def rps_features(kpts2d: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalise ``(21, 3)`` hand keypoints into the version 1 ``(42,)`` vector.

    The wrist (keypoint 0) is subtracted from every point, the hand is rotated
    so that the wrist -> middle-finger knuckle (keypoint 9) axis points up in
    image coordinates (negative ``y``), the result is divided by that axis
    length, and the 21 ``(x, y)`` pairs are flattened in keypoint order. The
    confidence column is ignored.

    A degenerate hand whose wrist and middle knuckle coincide, or whose
    coordinates are not finite, yields the zero vector rather than a division
    by zero, so the network still runs and reports its ``none``-leaning
    probabilities.

    Args:
        kpts2d: ``(21, 3)`` array of ``u, v, conf`` keypoints.

    Returns:
        The ``(42,)`` float32 feature vector.

    Raises:
        ValueError: If ``kpts2d`` does not have shape ``(21, 3)``.
    """

    keypoints = np.asarray(kpts2d, dtype=np.float32)
    if keypoints.shape != (_KEYPOINTS, 3):
        raise ValueError(f"kpts2d must be (21, 3), got shape {keypoints.shape}")
    if not np.isfinite(keypoints[:, :2]).all():
        return np.zeros(_FEATURES, dtype=np.float32)
    points = keypoints[:, :2].astype(np.float64)
    offset = points - points[0]
    axis = offset[_MIDDLE_KNUCKLE]
    length = float(math.hypot(float(axis[0]), float(axis[1])))
    if not math.isfinite(length) or length < _EPS:
        return np.zeros(_FEATURES, dtype=np.float32)
    # Rotate by -90 deg - atan2(axis) so the axis lands on (0, -1): "up" in
    # image coordinates, where y grows downwards.
    angle = -0.5 * math.pi - math.atan2(float(axis[1]), float(axis[0]))
    cos, sin = math.cos(angle), math.sin(angle)
    x = offset[:, 0]
    y = offset[:, 1]
    rotated = np.column_stack((cos * x - sin * y, sin * x + cos * y))
    return (rotated / length).astype(np.float32).reshape(_FEATURES)


class RpsClassifier:
    """A loaded rock/paper/scissors MLP, ready for per-frame inference."""

    def __init__(
        self,
        weights: Sequence[NDArray[np.float64]],
        classes: Sequence[str],
        feature_version: str | int = FEATURE_VERSION,
    ) -> None:
        version = _feature_version(feature_version)
        if version != FEATURE_VERSION:
            raise ValueError(
                f"unsupported rps feature version {version}, expected {FEATURE_VERSION}"
            )
        tuple_classes = tuple(str(name) for name in classes)
        if tuple_classes != RPS_CLASSES:
            raise ValueError(f"rps classes must be {RPS_CLASSES}, got {tuple_classes}")
        w1, b1, w2, b2, w3, b3 = (np.asarray(weight, dtype=np.float64) for weight in weights)
        _check_layer("w1", w1, (_FEATURES, None))
        _check_layer("b1", b1, (w1.shape[1],))
        _check_layer("w2", w2, (w1.shape[1], None))
        _check_layer("b2", b2, (w2.shape[1],))
        _check_layer("w3", w3, (w2.shape[1], len(tuple_classes)))
        _check_layer("b3", b3, (len(tuple_classes),))
        self._w1, self._b1 = w1, b1
        self._w2, self._b2 = w2, b2
        self._w3, self._b3 = w3, b3
        self._classes = tuple_classes
        self._feature_version = version

    @property
    def classes(self) -> tuple[str, ...]:
        """The output class names, in probability order."""

        return self._classes

    @property
    def feature_version(self) -> str:
        """The feature version this classifier expects."""

        return self._feature_version

    @classmethod
    def load(cls, path: Path) -> RpsClassifier:
        """Load the weights and metadata from a ``.npz`` file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If a key or a matrix shape is missing or wrong, or the
                feature version is not :data:`FEATURE_VERSION`.
        """

        with np.load(Path(path), allow_pickle=False) as data:
            missing = [
                key for key in (*_WEIGHT_KEYS, "feature_version", "classes") if key not in data
            ]
            if missing:
                raise ValueError(f"{path} is missing the rps keys: {', '.join(missing)}")
            weights = [np.asarray(data[key], dtype=np.float64) for key in _WEIGHT_KEYS]
            version = _feature_version(data["feature_version"])
            classes = tuple(str(name) for name in np.asarray(data["classes"]).reshape(-1))
        return cls(weights, classes, version)

    def predict(self, kpts2d: NDArray[np.float32]) -> tuple[str, float]:
        """Return the most likely class and its probability for one hand."""

        probabilities = self.predict_proba(np.asarray(kpts2d, dtype=np.float32)[np.newaxis, :, :])
        index = int(np.argmax(probabilities[0]))
        return self._classes[index], float(probabilities[0, index])

    def predict_proba(self, batch: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return the ``(N, 4)`` class probabilities for ``(N, 21, 3)`` keypoints.

        Raises:
            ValueError: If ``batch`` is not ``(N, 21, 3)``.
        """

        hands = np.asarray(batch, dtype=np.float32)
        if hands.ndim != 3 or hands.shape[1:] != (_KEYPOINTS, 3):
            raise ValueError(f"batch must be (N, 21, 3), got shape {hands.shape}")
        features = np.stack([rps_features(hand) for hand in hands]).astype(np.float64)
        hidden = np.maximum(features @ self._w1 + self._b1, 0.0)
        hidden = np.maximum(hidden @ self._w2 + self._b2, 0.0)
        logits = hidden @ self._w3 + self._b3
        return _softmax(logits)


def _feature_version(value: object) -> str:
    """Return the stored feature version as ``"v1"``-style text.

    The trainer stores ``np.array("v1")``; older files and tests may store the
    integer ``1`` instead, so both spellings are accepted.
    """

    array = np.asarray(value)
    if array.dtype.kind == "U":
        return str(array.reshape(()))
    if array.dtype.kind == "S":
        return bytes(array.reshape(())).decode("utf-8")
    number = int(array.reshape(()))
    return f"v{number}"


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _check_layer(name: str, value: NDArray[Any], shape: tuple[int | None, ...]) -> None:
    if value.ndim != len(shape):
        raise ValueError(f"{name} must be {len(shape)}-D, got shape {value.shape}")
    for actual, expected in zip(value.shape, shape, strict=True):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
