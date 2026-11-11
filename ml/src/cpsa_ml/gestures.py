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

Runtime contract (parity)
-------------------------
The exported ``.npz`` holds ``w1 b1 w2 b2 w3 b3 feature_version classes
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
    "NUM_KEYPOINTS",
    "PARITY_SAMPLES",
    "RPS_CLASSES",
    "SplitData",
    "TrainingResult",
    "build_model",
    "classify_ratios",
    "export_npz",
    "finger_ratios",
    "forward_proba",
    "label_to_class",
    "load_parity",
    "load_split",
    "main",
    "rps_features",
    "rule_baseline",
    "train_mlp",
    "write_report",
]

LOGGER = logging.getLogger(__name__)

NUM_KEYPOINTS = 21
FEATURE_DIM = 42
FEATURE_VERSION = "v1"
HIDDEN = 64
PARITY_SAMPLES = 16
NONE_RATIO = 2.0

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
) -> TrainingResult:
    """Train the MLP on CPU and return the best (by validation accuracy) weights."""

    import copy

    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()) or 1)
    generator = torch.Generator().manual_seed(seed)

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

    train = load_split(
        annotations,
        "train",
        seed=args.seed,
        none_ratio=args.none_ratio,
        max_per_class=args.max_per_class,
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
        )
        keep = order[cut:]
        train = SplitData(
            features=train.features[keep],
            labels=train.labels[keep],
            ratios=train.ratios[keep],
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
    )
    test_proba = forward_proba(test.features, result.weights)
    test_predictions = test_proba.argmax(axis=1).astype(np.int64)
    test_accuracy = float(np.mean(test_predictions == test.labels))
    val_proba = forward_proba(val.features, result.weights)
    val_accuracy = float(np.mean(val_proba.argmax(axis=1) == val.labels))
    baseline = rule_baseline(train.ratios, train.labels, test.ratios, test.labels)
    train_proba = forward_proba(train.features, result.weights)
    train_accuracy = float(np.mean(train_proba.argmax(axis=1) == train.labels))

    test_input, parity_features = load_parity(test, seed=args.seed)
    test_proba_parity = forward_proba(parity_features, result.weights)
    size_bytes = export_npz(Path(args.out), result.weights, test_input, test_proba_parity)

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
        "baseline": baseline,
        "artifact": {
            "path": str(args.out),
            "bytes": size_bytes,
            "limit_bytes": 100 * 1024,
            "under_limit": size_bytes < 100 * 1024,
            "parity_samples": int(test_input.shape[0]),
        },
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_report(Path(args.report), report)

    LOGGER.info(
        "test accuracy %.4f (target 0.95), baseline %.4f, npz %.1f KB -> %s",
        test_accuracy,
        baseline["test_accuracy"],
        size_bytes / 1024.0,
        args.out,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
