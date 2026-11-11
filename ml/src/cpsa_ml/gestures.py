"""Train the rock/paper/scissors gesture MLP from HaGRIDv2 landmark annotations.

The classifier is a 42 -> 64 -> 64 -> 4 multilayer perceptron over the ``v1``
hand features.  It is trained on the HaGRIDv2 ``annotations_with_landmarks``
JSON files only (no images), on the CPU, and exported to a small ``.npz`` that
the pure-NumPy runtime twin ``cpsa.vision.rps_classifier`` loads.

Example::

    python -m cpsa_ml.gestures \\
        --annotations build/data/hagrid_annotations/annotations \\
        --out ../models/rps_mlp.npz \\
        --report build/models/rps_mlp_report.json

Classes (``RPS_CLASSES``, index order): ``rock``, ``paper``, ``scissors``,
``none``.  HaGRID gesture labels map as in ``HAGRID_LABELS``: ``fist`` is
``rock``; ``palm``/``stop``/``stop_inverted`` are ``paper``;
``peace``/``peace_inverted`` are ``scissors``; every other label is ``none``.
HaGRID's own ``rock`` label is the horns sign, so it maps to ``none``.

Features (``rps_features``, version ``v1``)
-------------------------------------------
Input is a ``(21, 3)`` landmark array in **normalized HaGRID image
coordinates**: columns 0 and 1 are ``x`` and ``y`` in ``[0, 1]`` with ``y``
pointing down (image convention), and column 2 is the confidence, which the
features ignore.  MediaPipe hand order is assumed: wrist 0, then thumb, index,
middle, ring and pinky with four joints each; joint 9 is the middle-finger
knuckle.  The features are computed as follows:

1. subtract the wrist (joint 0) from every landmark;
2. rotate in the image plane so that the wrist -> joint 9 vector points up,
   that is towards ``-y``;
3. divide by the wrist -> joint-9 distance;
4. flatten the 21 ``(x, y)`` pairs into 42 float32 values.

The result is translation, rotation and scale invariant.  A degenerate hand
(wrist and joint 9 closer than ``1e-6``) yields 42 zeros.

Viewpoint augmentation
----------------------
The camera that feeds the runtime is not perfectly top-down, so training also
jitters the raw 21 points before :func:`rps_features` (the feature function
itself is unchanged): a random-axis foreshortening by ``cos(theta)`` with
``theta`` uniform in ``[0, 50]`` degrees, a random shear of ``+-0.15`` and a
mild perspective homography whose box corners move by about 10% of the hand
box.  The existing mirror/rotation/scale/noise augmentation in feature space
stays.  The report carries a tilt-robustness table: test accuracy under
synthetic foreshortening at 0, 20, 40 and 60 degrees with a fixed random axis,
for both the previously exported model and the newly trained one.

Runtime contract (parity)
-------------------------The exported ``.npz`` holds ``w1 b1 w2 b2 w3 b3 feature_version classes
test_input test_proba``.  All weights are float32 and use the row-vector
layout ``h = x @ w + b``: ``w1`` is ``(42, 64)``, ``w2`` is ``(64, 64)``,
``w3`` is ``(64, 4)``.  The forward pass is exactly::

    h1 = relu(x @ w1 + b1)
    h2 = relu(h1 @ w2 + b2)
    logits = h2 @ w3 + b3
    proba = softmax(logits)          # subtract the row max before exp

``test_input`` is ``(16, 21, 3)`` raw landmark samples from the HaGRID test
split and ``test_proba`` is the ``(16, 4)`` output of the pass above on
``rps_features(test_input)``.  The runtime test asserts that its own
``predict_proba(test_input)`` matches ``test_proba`` within ``atol=1e-5``,
which proves that both sides compute the same features and the same network.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FEATURE_DIM",
    "FEATURE_VERSION",
    "HAGRID_LABELS",
    "MAX_SHEAR",
    "MAX_TILT_DEG",
    "NUM_KEYPOINTS",
    "PARITY_SAMPLES",
    "PERSPECTIVE",
    "RPS_CLASSES",
    "TILT_ANGLES",
    "SplitData",
    "TrainingResult",
    "build_model",
    "classify_ratios",
    "export_npz",
    "features_from_points",
    "finger_ratios",
    "foreshorten_points",
    "forward_proba",
    "label_to_class",
    "load_parity",
    "load_split",
    "load_weights",
    "main",
    "rps_features",
    "rule_baseline",
    "tilt_robustness",
    "train_mlp",
    "viewpoint_augment",
    "write_report",
]

LOGGER = logging.getLogger(__name__)

NUM_KEYPOINTS = 21
FEATURE_DIM = 42
FEATURE_VERSION = "v1"
HIDDEN = 64
PARITY_SAMPLES = 16
NONE_RATIO = 2.0

#: Out-of-plane viewpoint augmentation bounds, applied to the raw 2D points.
MAX_TILT_DEG = 50.0
MAX_SHEAR = 0.15
#: Perspective coefficients in half-box units; box corners move by ~10%.
PERSPECTIVE = 0.10
#: Tilt angles (degrees) of the report's robustness table.
TILT_ANGLES: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0)

RPS_CLASSES: tuple[str, ...] = ("rock", "paper", "scissors", "none")
ROCK, PAPER, SCISSORS, NONE = 0, 1, 2, 3

#: HaGRID gesture label -> RPS class.  Everything not listed is ``none``.
HAGRID_LABELS: dict[str, tuple[str, ...]] = {
    "rock": ("fist",),
    "paper": ("palm", "stop", "stop_inverted"),
    "scissors": ("peace", "peace_inverted"),
}

#: MediaPipe fingertip and knuckle (MCP) indices, index to pinky.
FINGER_TIPS: tuple[int, ...] = (8, 12, 16, 20)
FINGER_MCPS: tuple[int, ...] = (5, 9, 13, 17)

_LABEL_TO_CLASS: dict[str, int] = {
    label: RPS_CLASSES.index(name) for name, labels in HAGRID_LABELS.items() for label in labels
}
_KNOWN_LABELS = frozenset(_LABEL_TO_CLASS)
_EPS = 1e-6


def label_to_class(label: str) -> int:
    """Return the :data:`RPS_CLASSES` index of one HaGRID gesture label."""

    return _LABEL_TO_CLASS.get(label.strip().lower(), NONE)


def rps_features(kpts2d: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the 42 ``v1`` features of one hand.

    Args:
        kpts2d: ``(21, 3)`` landmarks as normalized image coordinates ``x, y``
            in ``[0, 1]`` (``y`` down) plus a confidence column that is ignored.

    Returns:
        ``(42,)`` float32 features: wrist-centred, rotated so the middle-finger
        knuckle (joint 9) points up, and scaled by the wrist -> joint-9
        distance.  Degenerate hands yield zeros.

    Raises:
        ValueError: If ``kpts2d`` is not shaped ``(21, 3)``.
    """

    points = np.asarray(kpts2d, dtype=np.float32)
    if points.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected ({NUM_KEYPOINTS}, 3) keypoints, got {points.shape}")
    if not np.isfinite(points[:, :2]).all():
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    xy = points[:, :2] - points[0, :2]
    middle = xy[9]
    scale = float(np.hypot(float(middle[0]), float(middle[1])))
    if not math.isfinite(scale) or scale < _EPS:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    # Rotate by theta so that joint 9 ends up pointing towards -y ("up" in
    # image coordinates): atan2(middle) + theta == -pi/2.
    theta = -0.5 * math.pi - math.atan2(float(middle[1]), float(middle[0]))
    cos, sin = math.cos(theta), math.sin(theta)
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    rotated = xy @ rotation.T / np.float32(scale)
    return np.ascontiguousarray(rotated.reshape(FEATURE_DIM), dtype=np.float32)


def _box_frame(
    xy: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return box-normalized points, box centre and box size of a hand batch."""

    low = xy.min(axis=1)
    high = xy.max(axis=1)
    center = 0.5 * (low + high)
    size = np.maximum((high - low).max(axis=1), 1e-6)
    normalized = (xy - center[:, None, :]) / size[:, None, None]
    return normalized, center, size


def _as_point_batch(
    kpts2d: NDArray[np.float32],
) -> tuple[NDArray[np.float32], bool]:
    """Return ``((N, 21, 3) points, was_single_sample)`` and validate the shape."""

    points = np.asarray(kpts2d, dtype=np.float32)
    single = points.ndim == 2
    if single:
        points = points[None]
    if points.ndim != 3 or points.shape[1:] != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected (..., {NUM_KEYPOINTS}, 3) keypoints, got {points.shape}")
    return points, single


def foreshorten_points(
    kpts2d: NDArray[np.float32],
    theta_deg: float | NDArray[np.float64],
    axis_deg: float | NDArray[np.float64],
) -> NDArray[np.float32]:
    """Compress the hand along ``axis_deg`` by ``cos(theta_deg)``.

    This is the evaluation transform behind the report's tilt-robustness
    table; it uses one axis and one angle per sample (scalars broadcast) and
    leaves the confidence column untouched.  ``theta_deg=0`` is the identity.
    """

    points, single = _as_point_batch(kpts2d)
    batch = points.shape[0]
    theta = np.broadcast_to(np.asarray(theta_deg, dtype=np.float64), (batch,))
    axis = np.broadcast_to(np.asarray(axis_deg, dtype=np.float64), (batch,))
    theta = np.deg2rad(theta)
    axis = np.deg2rad(axis)

    xy = points[..., :2].astype(np.float64)
    normalized, center, size = _box_frame(xy)
    compress = np.cos(theta)
    one_minus = 1.0 - compress
    ux, uy = np.cos(axis), np.sin(axis)
    f11 = 1.0 - one_minus * ux * ux
    f12 = -one_minus * ux * uy
    f21 = f12
    f22 = 1.0 - one_minus * uy * uy
    u, v = normalized[..., 0], normalized[..., 1]
    warped = np.stack(
        (f11[:, None] * u + f12[:, None] * v, f21[:, None] * u + f22[:, None] * v), axis=-1
    )
    result = points.copy()
    result[..., :2] = (center[:, None, :] + size[:, None, None] * warped).astype(np.float32)
    return result[0] if single else result


def viewpoint_augment(
    kpts2d: NDArray[np.float32],
    rng: np.random.Generator,
    *,
    max_tilt_deg: float = MAX_TILT_DEG,
    max_shear: float = MAX_SHEAR,
    perspective: float = PERSPECTIVE,
    probability: float = 1.0,
) -> NDArray[np.float32]:
    """Apply random out-of-plane viewpoint changes to raw hand landmarks.

    Each sample gets, in this order, a foreshortening (compress along a uniform
    random axis by ``cos(theta)``, ``theta`` uniform in ``[0, max_tilt_deg]``),
    a shear of up to ``+-max_shear`` along a random image axis, and a mild
    perspective homography whose box corners move by about 10% of the hand box.
    The transforms act on the 2D points in normalized image coordinates; the
    confidence column is copied unchanged.  ``probability`` is the share of
    samples that keep the transform (the rest stay clean).  All randomness
    comes from ``rng``, so a fixed seed gives a fixed result.  Accepts
    ``(21, 3)`` or ``(N, 21, 3)`` and returns the same shape and dtype.
    """

    points, single = _as_point_batch(kpts2d)
    batch = points.shape[0]
    xy = points[..., :2].astype(np.float64)
    normalized, center, size = _box_frame(xy)

    theta = np.deg2rad(rng.uniform(0.0, max_tilt_deg, size=batch))
    axis = rng.uniform(0.0, np.pi, size=batch)
    compress = np.cos(theta)
    one_minus = 1.0 - compress
    ux, uy = np.cos(axis), np.sin(axis)
    f11 = 1.0 - one_minus * ux * ux
    f12 = -one_minus * ux * uy
    f22 = 1.0 - one_minus * uy * uy

    shear = rng.uniform(-max_shear, max_shear, size=batch)
    along_x = rng.random(batch) < 0.5
    s11 = np.where(along_x, 1.0, shear)
    s12 = np.where(along_x, shear, 0.0)
    s21 = np.where(along_x, 0.0, shear)
    s22 = 1.0
    a11 = s11 * f11 + s12 * f12
    a12 = s11 * f12 + s12 * f22
    a21 = s21 * f11 + s22 * f12
    a22 = s21 * f12 + s22 * f22

    # Perspective: project in box-normalized coordinates.  The coefficients
    # are bounded so a box corner moves by roughly `perspective` of the box.
    p0 = rng.uniform(-perspective, perspective, size=batch)
    p1 = rng.uniform(-perspective, perspective, size=batch)
    u, v = normalized[..., 0], normalized[..., 1]
    denominator = 1.0 + p0[:, None] * u + p1[:, None] * v
    denominator = np.where(np.abs(denominator) < 1e-3, 1.0, denominator)
    warped = np.stack(
        (
            (a11[:, None] * u + a12[:, None] * v) / denominator,
            (a21[:, None] * u + a22[:, None] * v) / denominator,
        ),
        axis=-1,
    )
    result = points.copy()
    result[..., :2] = (center[:, None, :] + size[:, None, None] * warped).astype(np.float32)
    if probability < 1.0:
        applied = rng.random(batch) < probability
        result[~applied] = points[~applied]
    return result[0] if single else result


def features_from_points(points: NDArray[np.float32]) -> NDArray[np.float32]:
    """Batched :func:`rps_features` for augmented ``(N, 21, 3)`` landmarks.

    The math is identical to :func:`rps_features` (same rotation convention,
    same float32 arithmetic); it only exists so a training batch of raw
    augmented points can be featurized in one vectorised call.  Degenerate or
    non-finite hands yield zero rows.
    """

    batch = np.asarray(points, dtype=np.float32)
    if batch.ndim != 3 or batch.shape[1:] != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected (N, {NUM_KEYPOINTS}, 3) keypoints, got {batch.shape}")
    xy = batch[..., :2]
    centered = xy - xy[:, :1, :]
    middle = centered[:, 9, :]
    scale = np.hypot(middle[:, 0], middle[:, 1]).astype(np.float32)
    theta = (-0.5 * math.pi - np.arctan2(middle[:, 1], middle[:, 0])).astype(np.float32)
    cos, sin = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    x, y = centered[..., 0], centered[..., 1]
    rotated = np.stack(
        (cos[:, None] * x - sin[:, None] * y, sin[:, None] * x + cos[:, None] * y), axis=-1
    )
    flat = (rotated / np.maximum(scale, _EPS)[:, None, None]).reshape(-1, FEATURE_DIM)
    flat = flat.astype(np.float32)
    flat[~np.isfinite(flat).all(axis=1)] = 0.0
    flat[scale < _EPS] = 0.0
    return flat


def finger_ratios(kpts2d: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the four fingertip/knuckle distances from the wrist (rule baseline)."""

    points = np.asarray(kpts2d, dtype=np.float32)
    if points.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected ({NUM_KEYPOINTS}, 3) keypoints, got {points.shape}")
    wrist = points[0, :2]
    ratios = np.zeros(len(FINGER_TIPS), dtype=np.float32)
    for index, (tip, mcp) in enumerate(zip(FINGER_TIPS, FINGER_MCPS, strict=True)):
        base = float(np.hypot(*(points[mcp, :2] - wrist)))
        reach = float(np.hypot(*(points[tip, :2] - wrist)))
        ratios[index] = reach / base if base > _EPS else 0.0
    return ratios


def classify_ratios(ratios: NDArray[np.float32], threshold: float) -> NDArray[np.int64]:
    """Map per-finger extension ratios to RPS classes with a fixed pattern table.

    A finger is *extended* when its ratio exceeds ``threshold``.  The four
    patterns ``0000``, ``1111`` and ``1100`` (index, middle, ring, pinky) are
    ``rock``, ``paper`` and ``scissors``; every other pattern is ``none``.
    """

    extended = np.asarray(ratios, dtype=np.float32) > np.float32(threshold)
    patterns = {
        (False, False, False, False): ROCK,
        (True, True, True, True): PAPER,
        (True, True, False, False): SCISSORS,
    }
    labels = np.full(extended.shape[0], NONE, dtype=np.int64)
    for pattern, label in patterns.items():
        labels[np.all(extended == np.asarray(pattern), axis=1)] = label
    return labels


def rule_baseline(
    train_ratios: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    eval_ratios: NDArray[np.float32],
    eval_labels: NDArray[np.int64],
) -> dict[str, Any]:
    """Fit the threshold of the rule classifier on train and score it on ``eval``.

    The single extension threshold is chosen by a grid search maximising the
    four-class train accuracy; the report quotes that train accuracy and the
    test accuracy of the chosen threshold.
    """

    candidates = np.unique(np.quantile(train_ratios, np.linspace(0.05, 0.95, 37)))
    best_threshold = float(candidates[0]) if candidates.size else 1.0
    best_train = -1.0
    for threshold in candidates:
        accuracy = float(np.mean(classify_ratios(train_ratios, float(threshold)) == train_labels))
        if accuracy > best_train:
            best_threshold, best_train = float(threshold), accuracy
    predictions = classify_ratios(eval_ratios, best_threshold)
    return {
        "name": "finger_extension_rules",
        "description": (
            "Fingertip-to-wrist distance over knuckle-to-wrist distance per finger; "
            "0000 rock, 1111 paper, 1100 scissors (index, middle, ring, pinky), "
            "every other pattern none."
        ),
        "threshold": round(best_threshold, 5),
        "train_accuracy": round(best_train, 5),
        "test_accuracy": round(float(np.mean(predictions == eval_labels)), 5),
        "confusion_matrix": confusion_matrix(eval_labels, predictions),
    }


def confusion_matrix(labels: NDArray[np.int64], predictions: NDArray[np.int64]) -> list[list[int]]:
    """Return the integer confusion matrix with rows = truth, columns = prediction."""

    matrix = np.zeros((len(RPS_CLASSES), len(RPS_CLASSES)), dtype=np.int64)
    for truth, predicted in zip(labels, predictions, strict=True):
        matrix[int(truth), int(predicted)] += 1
    return matrix.tolist()


def per_class_accuracy(
    labels: NDArray[np.int64], predictions: NDArray[np.int64]
) -> dict[str, float | None]:
    """Return accuracy per class, ``None`` for classes absent from ``labels``."""

    scores: dict[str, float | None] = {}
    for index, name in enumerate(RPS_CLASSES):
        mask = labels == index
        scores[name] = (
            round(float(np.mean(predictions[mask] == labels[mask])), 5) if mask.any() else None
        )
    return scores


@dataclass(frozen=True)
class SplitData:
    """Features, labels and baseline ratios of one annotation split.

    Args:
        features: ``(N, 42)`` v1 features.
        labels: ``(N,)`` class indices into :data:`RPS_CLASSES`.
        ratios: ``(N, 4)`` finger extension ratios for the rule baseline.
        kpts2d: Optional ``(N, 21, 3)`` raw landmarks (kept for parity samples).
        gestures: Counts of the source HaGRID gesture label per class.
    """

    features: NDArray[np.float32]
    labels: NDArray[np.int64]
    ratios: NDArray[np.float32]
    kpts2d: NDArray[np.float32] | None = None
    gestures: Mapping[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def counts(self) -> dict[str, int]:
        """Return the number of samples per RPS class."""

        return {
            name: int(np.count_nonzero(self.labels == index))
            for index, name in enumerate(RPS_CLASSES)
        }


def _annotation_files(annotations: Path, split: str) -> list[Path]:
    if annotations.is_file():
        return [annotations]
    split_dir = annotations / split
    if not split_dir.is_dir():
        return []
    return sorted(
        path
        for path in split_dir.glob("*.json")
        if path.is_file() and not path.name.startswith(".")
    )


def _iter_annotation_hands(path: Path) -> Iterator[tuple[str, NDArray[np.float32]]]:
    """Yield ``(gesture label, (21, 2) landmarks)`` for every usable hand of a file."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read HaGRID annotations {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"HaGRID annotations {path} must be a JSON object")
    gesture = path.stem
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        hands = entry.get("hand_landmarks") or []
        labels = entry.get("labels") or []
        for index, hand in enumerate(hands):
            points = np.asarray(hand, dtype=np.float32)
            if points.shape != (NUM_KEYPOINTS, 2) or not np.isfinite(points).all():
                continue
            label = str(labels[index]) if index < len(labels) else gesture
            yield label.strip().lower() or gesture, points


@dataclass
class _Bucket:
    """Accumulator for one class while reading annotation files."""

    features: list[NDArray[np.float32]] = field(default_factory=list)
    ratios: list[NDArray[np.float32]] = field(default_factory=list)
    kpts: list[NDArray[np.float32]] = field(default_factory=list)

    def add(self, points: NDArray[np.float32], keep_kpts: bool) -> None:
        kpts2d = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        kpts2d[:, :2] = points
        kpts2d[:, 2] = 1.0
        self.features.append(rps_features(kpts2d))
        self.ratios.append(finger_ratios(kpts2d))
        if keep_kpts:
            self.kpts.append(kpts2d)

    def __len__(self) -> int:
        return len(self.features)


def load_split(
    annotations: Path,
    split: str,
    *,
    seed: int = 0,
    none_ratio: float = NONE_RATIO,
    max_per_class: int | None = None,
    keep_kpts: bool = False,
) -> SplitData:
    """Load one HaGRID annotation split into features and labels.

    Gesture files whose names map to ``rock``/``paper``/``scissors`` are read
    first to count the minority classes; the remaining ``none`` gestures are
    then subsampled to at most ``none_ratio`` times the largest of those
    classes.  ``max_per_class`` additionally caps every class (used by tests).
    """

    annotations = Path(annotations)
    files = _annotation_files(annotations, split)
    if not files:
        raise ValueError(f"no HaGRID annotation JSON files for split {split!r} in {annotations}")

    relevant = [path for path in files if path.stem in _KNOWN_LABELS]
    if not relevant:
        relevant = files
    relevant_set = set(relevant)
    other = [path for path in files if path not in relevant_set]
    if annotations.is_file():
        other = []

    buckets = {index: _Bucket() for index in range(len(RPS_CLASSES))}
    gesture_counts: Counter[str] = Counter()
    for path in relevant:
        for label, points in _iter_annotation_hands(path):
            gesture_counts[label] += 1
            buckets[label_to_class(label)].add(points, keep_kpts)

    minority = max((len(buckets[index]) for index in (ROCK, PAPER, SCISSORS)), default=0)
    none_cap = int(round(none_ratio * minority)) if minority else None
    for path in other:
        for label, points in _iter_annotation_hands(path):
            gesture_counts[label] += 1
            buckets[NONE].add(points, keep_kpts)

    rng = np.random.default_rng(seed)
    for index, bucket in buckets.items():
        limit = none_cap if index == NONE else None
        if max_per_class is not None:
            limit = max_per_class if limit is None else min(limit, max_per_class)
        if limit is not None and len(bucket) > limit:
            keep = np.sort(rng.choice(len(bucket), size=limit, replace=False))
            bucket.features = [bucket.features[int(i)] for i in keep]
            bucket.ratios = [bucket.ratios[int(i)] for i in keep]
            if bucket.kpts:
                bucket.kpts = [bucket.kpts[int(i)] for i in keep]

    features = np.concatenate(
        [np.stack(buckets[index].features) for index in range(4) if buckets[index].features]
    )
    ratios = np.concatenate(
        [np.stack(buckets[index].ratios) for index in range(4) if buckets[index].ratios]
    )
    labels = np.concatenate(
        [
            np.full(len(buckets[index]), index, dtype=np.int64)
            for index in range(4)
            if buckets[index].features
        ]
    )
    kpts = None
    if keep_kpts:
        kpts = np.concatenate(
            [np.stack(buckets[index].kpts) for index in range(4) if buckets[index].kpts]
        )
    return SplitData(
        features=np.ascontiguousarray(features, dtype=np.float32),
        labels=labels,
        ratios=np.ascontiguousarray(ratios, dtype=np.float32),
        kpts2d=None if kpts is None else np.ascontiguousarray(kpts, dtype=np.float32),
        gestures=dict(gesture_counts),
    )


def load_parity(
    test: SplitData, *, seed: int = 0
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return ``(test_input, features)`` parity samples for the artifact.

    Four samples per class are taken from the test split (less when a class is
    smaller) and cycled up to :data:`PARITY_SAMPLES`; the caller computes
    ``test_proba`` from ``features`` with :func:`forward_proba`.
    """

    if test.kpts2d is None:
        raise ValueError("test split was loaded without keypoints; pass keep_kpts=True")
    rng = np.random.default_rng(seed + 1)
    picks: list[int] = []
    per_class = PARITY_SAMPLES // len(RPS_CLASSES)
    for index in range(len(RPS_CLASSES)):
        available = np.flatnonzero(test.labels == index)
        if available.size == 0:
            continue
        take = min(per_class, available.size)
        picks.extend(int(i) for i in rng.choice(available, size=take, replace=False))
    if not picks:
        raise ValueError("test split has no samples for parity")
    while len(picks) < PARITY_SAMPLES:
        picks.extend(picks[: PARITY_SAMPLES - len(picks)])
    test_input = np.ascontiguousarray(
        test.kpts2d[np.asarray(picks[:PARITY_SAMPLES])], dtype=np.float32
    )
    features = np.stack([rps_features(sample) for sample in test_input])
    return test_input, features


def forward_proba(
    features: NDArray[np.float32], weights: Mapping[str, NDArray[np.float32]]
) -> NDArray[np.float32]:
    """Run the exported network in NumPy: ReLU, ReLU, softmax over ``x @ w + b``."""

    hidden = np.maximum(np.asarray(features, dtype=np.float32) @ weights["w1"] + weights["b1"], 0.0)
    hidden = np.maximum(hidden @ weights["w2"] + weights["b2"], 0.0)
    logits = hidden @ weights["w3"] + weights["b3"]
    logits = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits)
    return np.ascontiguousarray(exponent / exponent.sum(axis=1, keepdims=True), dtype=np.float32)


WEIGHT_KEYS: tuple[str, ...] = ("w1", "b1", "w2", "b2", "w3", "b3")


def load_weights(path: Path) -> dict[str, NDArray[np.float32]]:
    """Load the six weight arrays of an F-D ``.npz`` artifact."""

    with np.load(Path(path)) as data:
        missing = [name for name in WEIGHT_KEYS if name not in data.files]
        if missing:
            raise ValueError(f"{path} is missing weight arrays: {', '.join(missing)}")
        return {name: np.ascontiguousarray(data[name], dtype=np.float32) for name in WEIGHT_KEYS}


def tilt_robustness(
    weights: Mapping[str, NDArray[np.float32]],
    test: SplitData,
    *,
    angles: Sequence[float] = TILT_ANGLES,
    seed: int = 0,
) -> dict[str, Any]:
    """Score ``weights`` on the test split under synthetic foreshortening.

    Each sample gets a random compression axis fixed by ``seed``; the table
    reports accuracy, per-class accuracy and the confusion matrix per angle.
    ``theta=0`` is the clean test split.
    """

    if test.kpts2d is None:
        raise ValueError("test split was loaded without keypoints; pass keep_kpts=True")
    axes = np.random.default_rng(seed).uniform(0.0, 180.0, size=len(test))
    detail: dict[str, Any] = {}
    for angle in angles:
        points = foreshorten_points(test.kpts2d, float(angle), axes)
        predictions = forward_proba(features_from_points(points), weights).argmax(axis=1)
        predictions = predictions.astype(np.int64)
        detail[f"{angle:g}"] = {
            "accuracy": round(float(np.mean(predictions == test.labels)), 5),
            "per_class_accuracy": per_class_accuracy(test.labels, predictions),
            "confusion_matrix": confusion_matrix(test.labels, predictions),
        }
    return {
        "angles_deg": [float(angle) for angle in angles],
        "axis_seed": seed,
        "detail": detail,
        "accuracy": {name: value["accuracy"] for name, value in detail.items()},
    }


def build_model() -> Any:
    """Return the torch ``42 -> 64 -> 64 -> 4`` MLP (CPU, ReLU activations)."""

    from torch import nn

    return nn.Sequential(
        nn.Linear(FEATURE_DIM, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, len(RPS_CLASSES)),
    )


def _augment(features: Any, generator: Any, torch: Any) -> Any:
    """Mirror, rotate ±20°, scale ±10% and jitter σ=0.02 the feature batch."""

    batch = int(features.shape[0])
    points = features.reshape(batch, NUM_KEYPOINTS, 2)
    mirror = torch.rand(batch, generator=generator) < 0.5
    flip = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    points = torch.where(mirror[:, None, None], points * flip, points)
    angle = (torch.rand(batch, generator=generator) - 0.5) * (40.0 * math.pi / 180.0)
    scale = 0.9 + 0.2 * torch.rand(batch, generator=generator)
    cos, sin = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
    x, y = points[..., 0], points[..., 1]
    rotated = torch.stack((cos * x - sin * y, sin * x + cos * y), dim=-1) * scale[:, None, None]
    noise = torch.randn(rotated.shape, generator=generator) * 0.02
    return (rotated + noise).reshape(batch, FEATURE_DIM)


def _predict_torch(model: Any, features: Any, torch: Any, batch_size: int = 8192) -> Any:
    model.eval()
    outputs: list[Any] = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), batch_size):
            outputs.append(model(features[start : start + batch_size]).argmax(dim=1))
    return torch.cat(outputs) if outputs else torch.empty(0, dtype=torch.long)


@dataclass
class TrainingResult:
    """Best weights and training history of one MLP run."""

    weights: dict[str, NDArray[np.float32]]
    best_epoch: int
    best_val_accuracy: float
    epochs_run: int
    train_accuracy: float
    history: list[dict[str, float]] = field(default_factory=list)


def train_mlp(
    train: SplitData,
    val: SplitData,
    *,
    seed: int = 0,
    epochs: int = 40,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    augment: bool = True,
    viewpoint: bool = True,
    viewpoint_probability: float = 1.0,
) -> TrainingResult:
    """Train the MLP on CPU and return the best (by validation accuracy) weights.

    When ``train.kpts2d`` is available and ``viewpoint`` is set, every batch is
    first passed through :func:`viewpoint_augment` and then featurized with
    :func:`features_from_points`; the feature-space augmentation in
    :func:`_augment` still follows.  Validation always uses the unaugmented
    features, so early stopping scores the clean split.
    """

    import copy

    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()) or 1)
    generator = torch.Generator().manual_seed(seed)
    viewpoint_rng = np.random.default_rng(seed + 1009)
    raw_points = None
    if viewpoint and train.kpts2d is not None:
        raw_points = np.ascontiguousarray(train.kpts2d, dtype=np.float32)

    device = torch.device("cpu")
    model = build_model().to(device)
    features = torch.from_numpy(train.features)
    labels = torch.from_numpy(train.labels)
    val_features = torch.from_numpy(val.features)
    val_labels = torch.from_numpy(val.labels)

    counts = np.bincount(train.labels, minlength=len(RPS_CLASSES)).astype(np.float32)
    weights = counts.sum() / (len(RPS_CLASSES) * np.maximum(counts, 1.0))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    best_epoch = 0
    history: list[dict[str, float]] = []
    since_best = 0
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(int(features.shape[0]), generator=generator)
        total_loss = 0.0
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size]
            if raw_points is not None:
                augmented = viewpoint_augment(
                    raw_points[batch.numpy()], viewpoint_rng, probability=viewpoint_probability
                )
                inputs = torch.from_numpy(features_from_points(augmented))
            else:
                inputs = features[batch]
            if augment:
                inputs = _augment(inputs, generator, torch)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels[batch])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch.numel())
        val_predictions = _predict_torch(model, val_features, torch)
        val_accuracy = float((val_predictions == val_labels).float().mean().item())
        entry = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(int(features.shape[0]), 1), 5),
            "val_accuracy": round(val_accuracy, 5),
        }
        history.append(entry)
        LOGGER.info("epoch %d: loss %.4f val %.4f", epoch, entry["train_loss"], val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    train_predictions = _predict_torch(model, features, torch)
    train_accuracy = float((train_predictions == labels).float().mean().item())
    state = model.state_dict()
    exported = {
        "w1": state["0.weight"].detach().numpy().T.copy(),
        "b1": state["0.bias"].detach().numpy().copy(),
        "w2": state["2.weight"].detach().numpy().T.copy(),
        "b2": state["2.bias"].detach().numpy().copy(),
        "w3": state["4.weight"].detach().numpy().T.copy(),
        "b3": state["4.bias"].detach().numpy().copy(),
    }
    return TrainingResult(
        weights={
            name: np.ascontiguousarray(value, dtype=np.float32) for name, value in exported.items()
        },
        best_epoch=best_epoch,
        best_val_accuracy=round(best_accuracy, 5),
        epochs_run=epoch,
        train_accuracy=round(train_accuracy, 5),
        history=history,
    )


def export_npz(
    path: Path,
    weights: Mapping[str, NDArray[np.float32]],
    test_input: NDArray[np.float32],
    test_proba: NDArray[np.float32],
) -> int:
    """Write the F-D artifact and return its size in bytes."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        w1=weights["w1"],
        b1=weights["b1"],
        w2=weights["w2"],
        b2=weights["b2"],
        w3=weights["w3"],
        b3=weights["b3"],
        feature_version=np.array(FEATURE_VERSION),
        classes=np.array(RPS_CLASSES),
        test_input=np.ascontiguousarray(test_input, dtype=np.float32),
        test_proba=np.ascontiguousarray(test_proba, dtype=np.float32),
    )
    return path.stat().st_size


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write the JSON report to ``path``, creating its parent directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _split_summary(data: SplitData) -> dict[str, Any]:
    return {
        "samples": len(data),
        "class_counts": data.counts(),
        "gesture_counts": dict(sorted(data.gestures.items())),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""

    parser = argparse.ArgumentParser(
        prog="python -m cpsa_ml.gestures",
        description="Train the rock/paper/scissors MLP from HaGRIDv2 landmark annotations.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="annotations directory with train/val/test sub-directories of gesture JSON files",
    )
    parser.add_argument("--out", type=Path, default=Path("models/rps_mlp.npz"), help="output .npz")
    parser.add_argument("--report", type=Path, default=Path("build/rps_mlp_report.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--none-ratio", type=float, default=NONE_RATIO)
    parser.add_argument(
        "--viewpoint-probability",
        type=float,
        default=1.0,
        help="share of training samples that receive the out-of-plane viewpoint augmentation",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="cap every class (fast smoke runs; the committed model uses every sample)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Train, evaluate, export the artifact and write the report."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )
    started = time.perf_counter()
    annotations = Path(args.annotations)
    out_path = Path(args.out)
    previous = load_weights(out_path) if out_path.is_file() else None
    if previous is not None:
        LOGGER.info("loaded previous artifact %s for the tilt comparison", out_path)

    train = load_split(
        annotations,
        "train",
        seed=args.seed,
        none_ratio=args.none_ratio,
        max_per_class=args.max_per_class,
        keep_kpts=True,
    )
    try:
        val = load_split(
            annotations,
            "val",
            seed=args.seed + 1000,
            none_ratio=args.none_ratio,
            max_per_class=args.max_per_class,
        )
    except ValueError:
        LOGGER.warning("no val split; holding out 10%% of train")
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(train))
        cut = max(1, len(train) // 10)
        val = SplitData(
            features=train.features[order[:cut]],
            labels=train.labels[order[:cut]],
            ratios=train.ratios[order[:cut]],
            kpts2d=None if train.kpts2d is None else train.kpts2d[order[:cut]],
        )
        keep = order[cut:]
        train = SplitData(
            features=train.features[keep],
            labels=train.labels[keep],
            ratios=train.ratios[keep],
            kpts2d=None if train.kpts2d is None else train.kpts2d[keep],
            gestures=train.gestures,
        )
    test = load_split(
        annotations,
        "test",
        seed=args.seed + 2000,
        none_ratio=args.none_ratio,
        max_per_class=args.max_per_class,
        keep_kpts=True,
    )
    LOGGER.info("samples: train %d, val %d, test %d", len(train), len(val), len(test))

    result = train_mlp(
        train,
        val,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        viewpoint_probability=args.viewpoint_probability,
    )
    test_proba = forward_proba(test.features, result.weights)
    test_predictions = test_proba.argmax(axis=1).astype(np.int64)
    test_accuracy = float(np.mean(test_predictions == test.labels))
    val_proba = forward_proba(val.features, result.weights)
    val_accuracy = float(np.mean(val_proba.argmax(axis=1) == val.labels))
    baseline = rule_baseline(train.ratios, train.labels, test.ratios, test.labels)
    train_proba = forward_proba(train.features, result.weights)
    train_accuracy = float(np.mean(train_proba.argmax(axis=1) == train.labels))

    new_tilt = tilt_robustness(result.weights, test, angles=TILT_ANGLES, seed=args.seed)
    tilt_report: dict[str, Any] = {
        "metric": (
            "test accuracy under synthetic foreshortening (random axis per sample, fixed seed); "
            "0 deg is the clean split"
        ),
        "angles_deg": [float(angle) for angle in TILT_ANGLES],
        "seed": args.seed,
        "new": new_tilt,
    }
    keep_new = True
    if previous is not None:
        previous_tilt = tilt_robustness(previous, test, angles=TILT_ANGLES, seed=args.seed)
        tilt_report["previous"] = {"model": "bfdaa14", "path": str(out_path), **previous_tilt}
        new_accuracy = new_tilt["accuracy"]
        old_accuracy = previous_tilt["accuracy"]
        theta0_ok = bool(new_accuracy["0"] >= 0.95)
        improved_40 = bool(new_accuracy["40"] > old_accuracy["40"])
        improved_60 = bool(new_accuracy["60"] > old_accuracy["60"])
        keep_new = theta0_ok and improved_40 and improved_60
        tilt_report["delta"] = {
            name: round(float(new_accuracy[name] - old_accuracy[name]), 5) for name in new_accuracy
        }
        tilt_report["decision"] = {
            "rule": (
                "keep the new model iff theta=0 accuracy is at least 0.95 and both the "
                "40 deg and the 60 deg accuracy improve"
            ),
            "theta0_ok": theta0_ok,
            "improved_40": improved_40,
            "improved_60": improved_60,
            "kept": "new" if keep_new else "previous",
        }
    else:
        tilt_report["decision"] = {"rule": "no previous artifact at --out", "kept": "new"}

    test_input, parity_features = load_parity(test, seed=args.seed)
    if keep_new:
        test_proba_parity = forward_proba(parity_features, result.weights)
        size_bytes = export_npz(out_path, result.weights, test_input, test_proba_parity)
        artifact = {
            "path": str(out_path),
            "bytes": size_bytes,
            "limit_bytes": 100 * 1024,
            "under_limit": size_bytes < 100 * 1024,
            "parity_samples": int(test_input.shape[0]),
            "model": "new",
        }
    else:
        size_bytes = out_path.stat().st_size
        artifact = {
            "path": str(out_path),
            "bytes": size_bytes,
            "limit_bytes": 100 * 1024,
            "under_limit": size_bytes < 100 * 1024,
            "parity_samples": int(test_input.shape[0]),
            "model": "previous (bfdaa14)",
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acceptance": "AC-05",
        "dataset": {
            "annotations": str(annotations),
            "splits": {
                "train": _split_summary(train),
                "val": _split_summary(val),
                "test": _split_summary(test),
            },
        },
        "features": {"version": FEATURE_VERSION, "dim": FEATURE_DIM},
        "classes": list(RPS_CLASSES),
        "hagrid_labels": {name: list(labels) for name, labels in HAGRID_LABELS.items()},
        "model": {
            "architecture": [FEATURE_DIM, HIDDEN, HIDDEN, len(RPS_CLASSES)],
            "activation": "relu, relu, softmax",
            "parameters": sum(int(value.size) for value in result.weights.values()),
            "seed": args.seed,
            "epochs_run": result.epochs_run,
            "best_epoch": result.best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "augmentation": "mirror 0.5, rotate +-20 deg, scale +-10%, noise sigma 0.02",
            "viewpoint_augmentation": (
                "foreshortening cos(theta), theta uniform 0-50 deg along a uniform random axis; "
                "shear +-0.15; mild perspective homography with ~10% hand-box corner jitter; "
                "applied to the raw 2D points before rps_features; "
                f"probability {args.viewpoint_probability:g}"
            ),
            "history": result.history,
        },
        "metrics": {
            "train_accuracy": round(train_accuracy, 5),
            "val_accuracy": round(val_accuracy, 5),
            "test_accuracy": round(test_accuracy, 5),
            "target": 0.95,
            "target_met": round(test_accuracy, 5) >= 0.95,
            "confusion_matrix": confusion_matrix(test.labels, test_predictions),
            "confusion_matrix_order": list(RPS_CLASSES),
            "per_class_accuracy": per_class_accuracy(test.labels, test_predictions),
        },
        "tilt_robustness": tilt_report,
        "baseline": baseline,
        "artifact": artifact,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_report(Path(args.report), report)

    LOGGER.info(
        "test accuracy %.4f (target 0.95), baseline %.4f, npz %.1f KB -> %s",
        test_accuracy,
        baseline["test_accuracy"],
        size_bytes / 1024.0,
        out_path,
    )
    LOGGER.info(
        "tilt accuracy new %s kept=%s",
        new_tilt["accuracy"],
        tilt_report["decision"]["kept"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
