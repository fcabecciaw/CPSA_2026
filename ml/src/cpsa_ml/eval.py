"""Accuracy metrics for the ``HALFBUST_62`` pose variants (pure numpy).

The same module scores the FP32 model (Ultralytics predictions or the deploy
graph) and the INT8 model (the ``cpsa`` decoder on the quantized head
outputs), so the variant study compares like with like.  It only needs numpy
and the half-bust label reader: predictions and ground truth are already
decoded and in pixels, so no torch, no PIL and no decoders live here.

Reported metrics
----------------
detection
    Person detection AP50: per image, predictions are matched greedily by
    descending score to targets with box IoU ``>= 0.5`` (class-aware,
    one-to-one); AP is the all-point interpolated area under the
    precision-recall curve (Ultralytics' ``compute_ap``).

pose
    Pose mAP50-95 by OKS, the same matching rule at OKS thresholds
    0.50:0.05:0.95, AP per threshold, mean over thresholds.  It repeats
    Ultralytics' :func:`ultralytics.utils.metrics.kpt_iou`::

        d  = squared keypoint distance
        e  = d / ((2 * sigma) ** 2 * area * 2)
        OKS = mean(exp(-e) over visible ground-truth keypoints)

    with ``area = box_width * box_height * 0.53`` -- the ``0.53`` factor is
    the xtcocoapi convention that Ultralytics uses in
    ``PoseValidator._process_batch``.

task metrics (face-width units)
    The runtime (``cpsa.vision``) measures every geometric threshold in face
    widths, so keypoint errors are divided by the face width of the same
    ground-truth instance.  The width uses the runtime's ``auto`` rule
    (``cpsa.vision.subject._face_width_measurement``): jaw-end distance, else
    ear-to-ear, else inter-ocular distance times ``eye_to_face_ratio`` (2.2).
    An instance without a usable face width is excluded from the normalized
    metrics and counted in ``counts.instances_without_face_width``.

Per keypoint group (face anchors, shoulders, elbows, wrists, left hand, right
hand) the report contains:

``pck``
    PCK@0.1 and PCK@0.2 face widths over the *evaluated* keypoints: ground
    truth visible, a detected person associated by box IoU, a face width, and
    a predicted keypoint confidence ``>= kpt_conf_thr``.
``pck_all``
    The same count divided by *all* ground-truth visible keypoints, so an
    undetected person or a rejected keypoint counts as an error.  This is the
    task-level number the variant decision rule uses.
``missing_rate``
    Missing keypoints / all ground-truth visible keypoints, where missing
    means no associated prediction or a predicted confidence below
    ``kpt_conf_thr`` (or a non-finite coordinate).

Centroid errors are the distance between the ground-truth centroid (visible
points) and the predicted centroid (confident points) of the 14 runtime face
anchors and of each hand, divided by the instance face width.

framing
    The runtime decides ``full_body`` vs ``half_body`` from the person-box
    height divided by the face width (``cpsa.vision.subject.framing_state``).
    The report classifies every ground-truth person twice --- from its own COCO
    box and from the box of the prediction matched to it --- and reports the
    fraction in each class.  A full-frame val split must read mostly
    ``full_body`` and a half-body val split mostly ``half_body``; otherwise the
    model's box does not describe the true person extent.  The threshold is
    read from ``SubjectConfig().full_body_ratio``; it is never copied here.

OKS sigmas
----------
``HALFBUST62_OKS_SIGMAS`` has 62 entries in Ultralytics' convention, i.e. the
values a dataset YAML passes as ``kpt_oks_sigmas``:

* body 0-10: COCO-17 sigmas of the matching joints
  ``(0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
  0.062)`` -- Ultralytics' ``OKS_SIGMA`` divided by 10;
* face anchors 11-19 and both hands (20-40, 41-61): ``0.026``, the smallest
  COCO value (nose).  Face and hand points are small rigid parts whose
  localization noise relative to the person box is at most that of the nose,
  and COCO-WholeBody evaluates them with the same annotation-scale constants.
  Hard-coding the value keeps the score independent of the dataset YAML that
  happens to be attached, which is what makes the nine variants comparable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "HALFBUST62_FACE_ANCHOR_INDICES",
    "HALFBUST62_GROUPS",
    "HALFBUST62_N_KPT",
    "HALFBUST62_OKS_SIGMAS",
    "EvalConfig",
    "ImageSample",
    "PersonPrediction",
    "PersonTarget",
    "average_precision",
    "evaluate",
    "face_width_px",
    "oks_matrix",
    "read_label_file",
    "read_yolo_pose_labels",
    "runtime_full_body_ratio",
]

HALFBUST62_N_KPT = 62

#: Keypoint groups scored independently, in schema order.  ``HALFBUST_62`` is
#: body 0-10, face 11-19, left hand 20-40 and right hand 41-61
#: (``cpsa_ml.schema.HALFBUST_62``).
HALFBUST62_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("face", tuple(range(11, 20))),
    ("shoulders", (5, 6)),
    ("elbows", (7, 8)),
    ("wrists", (9, 10)),
    ("left_hand", tuple(range(20, 41))),
    ("right_hand", tuple(range(41, 62))),
)

#: The runtime's face anchors: nose, eyes, ears and the nine schema face points
#: (``cpsa.vision.layouts.HALFBUST_62.face``).  Used for the face centroid.
HALFBUST62_FACE_ANCHOR_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4) + tuple(range(11, 20))

_BODY_SIGMAS = (0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062)
_FACE_HAND_SIGMA = 0.026

HALFBUST62_OKS_SIGMAS: tuple[float, ...] = (
    _BODY_SIGMAS + (_FACE_HAND_SIGMA,) * 9 + (_FACE_HAND_SIGMA,) * 21 + (_FACE_HAND_SIGMA,) * 21
)

_JAW_PAIR = (11, 12)
_EAR_PAIR = (3, 4)
_EYE_PAIR = (1, 2)
_LEFT_HAND = tuple(range(20, 41))
_RIGHT_HAND = tuple(range(41, 62))
_FACE_WIDTH_SOURCES = frozenset({"auto", "jaw", "ears", "eyes"})
_EPS = 1e-7

#: xtcocoapi/Ultralytics scale applied to the ground-truth box area in OKS.
OKS_AREA_SCALE = 0.53


def runtime_full_body_ratio() -> float:
    """Return ``cpsa.vision.subject.SubjectConfig().full_body_ratio``.

    The runtime owns the framing rule; the accuracy report reads the same
    threshold so a metric and the deployed decision cannot drift apart.

    Raises:
        ImportError: If the ``cpsa`` runtime package is not importable (an
            isolated ``cpsa-ml`` install, for example).
    """

    from cpsa.vision.subject import SubjectConfig

    return float(SubjectConfig().full_body_ratio)


@dataclass(frozen=True, slots=True, eq=False)
class PersonPrediction:
    """One decoded person prediction, in image pixels.

    ``box`` is ``(x1, y1, x2, y2)``, ``keypoints`` is ``(K, 3)`` with
    ``x, y`` in pixels and ``conf`` in ``[0, 1]``.  Equality is identity
    based, because the numpy arrays make field-wise ``==`` ambiguous; the code
    uses identity to track which prediction was already matched.
    """

    box: NDArray[np.float64]
    score: float
    keypoints: NDArray[np.float64]
    class_id: int = 0

    def __post_init__(self) -> None:
        box = np.asarray(self.box, dtype=np.float64)
        keypoints = np.asarray(self.keypoints, dtype=np.float64)
        if box.shape != (4,):
            raise ValueError(f"box must be (4,), got {box.shape}")
        if keypoints.ndim != 2 or keypoints.shape[1] != 3:
            raise ValueError(f"keypoints must be (K, 3), got {keypoints.shape}")
        if not np.isfinite(box).all() or not np.isfinite(float(self.score)):
            raise ValueError("prediction box and score must be finite")
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "keypoints", keypoints)


@dataclass(frozen=True, slots=True, eq=False)
class PersonTarget:
    """One ground-truth person, in image pixels.

    ``keypoints`` is ``(K, 3)`` with ``x, y`` in pixels and the label
    visibility in the third column (``> 0`` means annotated).
    """

    box: NDArray[np.float64]
    keypoints: NDArray[np.float64]
    class_id: int = 0

    def __post_init__(self) -> None:
        box = np.asarray(self.box, dtype=np.float64)
        keypoints = np.asarray(self.keypoints, dtype=np.float64)
        if box.shape != (4,):
            raise ValueError(f"box must be (4,), got {box.shape}")
        if keypoints.ndim != 2 or keypoints.shape[1] != 3:
            raise ValueError(f"keypoints must be (K, 3), got {keypoints.shape}")
        if not np.isfinite(box).all():
            raise ValueError("target box must be finite")
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "keypoints", keypoints)


@dataclass(frozen=True, slots=True)
class ImageSample:
    """Predictions and ground truth of one image, in the same pixel space."""

    name: str
    predictions: tuple[PersonPrediction, ...] = ()
    targets: tuple[PersonTarget, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Metric thresholds and the OKS/face-width conventions.

    ``kpt_conf_thr`` is the confidence at or above which a predicted keypoint
    counts as detected (the runtime's ``face_anchor_conf``/``wrist_conf``
    default).  ``match_iou`` associates a prediction with a target for the
    per-keypoint metrics; detection and pose AP use their own thresholds.
    ``score_thr`` filters predictions a second time (the decoders already
    apply their confidence threshold, so it defaults to 0).  ``full_body_ratio``
    is the runtime framing threshold (person-box height / face width); when it
    is left as ``None`` the value is read from the runtime and, if the runtime
    is not installed, framing is reported as ``None``.
    """

    n_kpt: int = HALFBUST62_N_KPT
    sigmas: tuple[float, ...] = HALFBUST62_OKS_SIGMAS
    kpt_conf_thr: float = 0.3
    match_iou: float = 0.5
    score_thr: float = 0.0
    pck_thresholds: tuple[float, ...] = (0.1, 0.2)
    oks_thresholds: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
    iou_threshold: float = 0.5
    face_width_source: str = "auto"
    eye_to_face_ratio: float = 2.2
    area_scale: float = OKS_AREA_SCALE
    full_body_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.n_kpt <= 0:
            raise ValueError(f"n_kpt must be positive, got {self.n_kpt}")
        if len(self.sigmas) != self.n_kpt:
            raise ValueError(f"sigmas must have {self.n_kpt} values, got {len(self.sigmas)}")
        if any(not sigma > 0.0 for sigma in self.sigmas):
            raise ValueError("sigmas must be positive")
        if not 0.0 <= self.score_thr <= 1.0:
            raise ValueError(f"score_thr must be in [0, 1], got {self.score_thr}")
        if not 0.0 <= self.kpt_conf_thr <= 1.0:
            raise ValueError(f"kpt_conf_thr must be in [0, 1], got {self.kpt_conf_thr}")
        if not 0.0 < self.match_iou <= 1.0:
            raise ValueError(f"match_iou must be in (0, 1], got {self.match_iou}")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {self.iou_threshold}")
        if self.face_width_source not in _FACE_WIDTH_SOURCES:
            raise ValueError(
                f"face_width_source must be one of {sorted(_FACE_WIDTH_SOURCES)}, "
                f"got {self.face_width_source!r}"
            )
        if not self.eye_to_face_ratio > 0.0:
            raise ValueError("eye_to_face_ratio must be positive")
        if any(not value > 0.0 for value in self.pck_thresholds):
            raise ValueError("pck_thresholds must be positive")
        if any(not 0.0 < value <= 1.0 for value in self.oks_thresholds):
            raise ValueError("oks_thresholds must be in (0, 1]")
        if self.full_body_ratio is None:
            try:
                ratio = runtime_full_body_ratio()
            except ImportError:
                pass
            else:
                object.__setattr__(self, "full_body_ratio", ratio)
        elif not self.full_body_ratio > 0.0:
            raise ValueError(f"full_body_ratio must be positive, got {self.full_body_ratio}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready copy of the configuration (recorded in every report)."""

        return {
            "n_kpt": int(self.n_kpt),
            "sigmas": [float(value) for value in self.sigmas],
            "kpt_conf_thr": float(self.kpt_conf_thr),
            "match_iou": float(self.match_iou),
            "score_thr": float(self.score_thr),
            "pck_thresholds": [float(value) for value in self.pck_thresholds],
            "oks_thresholds": [float(value) for value in self.oks_thresholds],
            "iou_threshold": float(self.iou_threshold),
            "face_width_source": self.face_width_source,
            "eye_to_face_ratio": float(self.eye_to_face_ratio),
            "area_scale": float(self.area_scale),
            "full_body_ratio": (
                None if self.full_body_ratio is None else float(self.full_body_ratio)
            ),
            "groups": {name: list(indices) for name, indices in HALFBUST62_GROUPS},
            "face_anchor_indices": list(HALFBUST62_FACE_ANCHOR_INDICES),
        }


# ---------------------------------------------------------------------------
# Label reading
# ---------------------------------------------------------------------------


def read_yolo_pose_labels(
    text: str,
    width: int,
    height: int,
    n_kpt: int = HALFBUST62_N_KPT,
) -> tuple[PersonTarget, ...]:
    """Parse Ultralytics YOLO-pose label text into pixel-space targets.

    Each line is ``class cx cy w h`` followed by ``n_kpt`` triples
    ``x y visibility``, all normalized to the image.  ``width``/``height``
    scale them back to pixels; the half-bust builder writes one full-crop
    person per file (``cx cy w h = 0.5 0.5 1 1``) but any box is accepted.

    Args:
        text: Contents of the ``.txt`` label file.
        width: Image width in pixels.
        height: Image height in pixels.
        n_kpt: Number of keypoints per person.

    Returns:
        One :class:`PersonTarget` per non-empty line.

    Raises:
        ValueError: If the image size, the keypoint count or a line is invalid.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    if n_kpt <= 0:
        raise ValueError(f"n_kpt must be positive, got {n_kpt}")
    expected = 5 + 3 * n_kpt
    targets: list[PersonTarget] = []
    for number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != expected:
            raise ValueError(
                f"label line {number} has {len(fields)} values; {expected} are needed for "
                f"{n_kpt} keypoints"
            )
        values = np.asarray([float(value) for value in fields], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"label line {number} contains a non-finite value")
        class_id = int(values[0])
        center_x, center_y, box_width, box_height = values[1:5]
        box = np.array(
            [
                (center_x - box_width / 2.0) * width,
                (center_y - box_height / 2.0) * height,
                (center_x + box_width / 2.0) * width,
                (center_y + box_height / 2.0) * height,
            ],
            dtype=np.float64,
        )
        keypoints = values[5:].reshape(n_kpt, 3).copy()
        keypoints[:, 0] *= width
        keypoints[:, 1] *= height
        targets.append(PersonTarget(box=box, keypoints=keypoints, class_id=class_id))
    return tuple(targets)


def read_label_file(
    path: str | Path,
    width: int,
    height: int,
    n_kpt: int = HALFBUST62_N_KPT,
) -> tuple[PersonTarget, ...]:
    """Read one YOLO-pose ``.txt`` file; see :func:`read_yolo_pose_labels`."""

    return read_yolo_pose_labels(Path(path).read_text(encoding="utf-8"), width, height, n_kpt)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _visible(keypoints: NDArray[np.float64]) -> NDArray[np.bool_]:
    return keypoints[:, 2] > 0.0


def _pair_distance(
    keypoints: NDArray[np.float64],
    first: int,
    second: int,
) -> float | None:
    if first >= len(keypoints) or second >= len(keypoints):
        return None
    if keypoints[first, 2] <= 0.0 or keypoints[second, 2] <= 0.0:
        return None
    first_point = keypoints[first, :2]
    second_point = keypoints[second, :2]
    if not np.isfinite(first_point).all() or not np.isfinite(second_point).all():
        return None
    distance = float(np.linalg.norm(first_point - second_point))
    return distance if distance > _EPS else None


def face_width_px(
    keypoints: ArrayLike,
    *,
    source: str = "auto",
    eye_to_face_ratio: float = 2.2,
) -> float | None:
    """Return the face width of one person in pixels, or ``None``.

    Mirrors ``cpsa.vision.subject._face_width_measurement``: ``jaw`` is the
    distance between the two jaw ends (schema indices 11 and 12), ``ears`` the
    ear-to-ear distance (3 and 4) and ``eyes`` the inter-ocular distance
    times ``eye_to_face_ratio`` (1 and 2).  ``auto`` tries jaw, ears, then
    eyes.  A pair counts only when both points are visible (third column
    ``> 0``) and finite.

    Args:
        keypoints: ``(K, 3)`` ground-truth keypoints, ``x``/``y`` in pixels and
            a positive visibility in the third column.
        source: ``auto``, ``jaw``, ``ears`` or ``eyes``.
        eye_to_face_ratio: Inter-ocular distance to face-width scale factor.
    """

    array = np.asarray(keypoints, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"keypoints must be (K, 3), got {array.shape}")
    if source not in _FACE_WIDTH_SOURCES:
        raise ValueError(f"source must be one of {sorted(_FACE_WIDTH_SOURCES)}, got {source!r}")
    if source == "auto":
        candidates: tuple[tuple[tuple[int, int], float], ...] = (
            (_JAW_PAIR, 1.0),
            (_EAR_PAIR, 1.0),
            (_EYE_PAIR, float(eye_to_face_ratio)),
        )
    else:
        pairs = {"jaw": (_JAW_PAIR, 1.0), "ears": (_EAR_PAIR, 1.0)}
        if source == "eyes":
            candidates = ((_EYE_PAIR, float(eye_to_face_ratio)),)
        else:
            candidates = (pairs[source],)
    for (first, second), scale in candidates:
        distance = _pair_distance(array, first, second)
        if distance is not None:
            return distance * scale
    return None


def _xyxy_area(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    width = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
    height = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    areas: NDArray[np.float64] = width * height
    return areas


def _iou_matrix(
    targets: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``(N_targets, M_predictions)`` IoU."""

    if targets.size == 0 or predictions.size == 0:
        return np.zeros((len(targets), len(predictions)), dtype=np.float64)
    x1 = np.maximum(targets[:, None, 0], predictions[None, :, 0])
    y1 = np.maximum(targets[:, None, 1], predictions[None, :, 1])
    x2 = np.minimum(targets[:, None, 2], predictions[None, :, 2])
    y2 = np.minimum(targets[:, None, 3], predictions[None, :, 3])
    intersection = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    target_areas = _xyxy_area(targets)[:, None]
    prediction_areas = _xyxy_area(predictions)[None, :]
    union = target_areas + prediction_areas - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def oks_matrix(
    target_keypoints: NDArray[np.float64],
    target_boxes: NDArray[np.float64],
    prediction_keypoints: NDArray[np.float64],
    sigmas: Sequence[float],
    area_scale: float = OKS_AREA_SCALE,
) -> NDArray[np.float64]:
    """``(N_targets, M_predictions)`` OKS, Ultralytics-compatible.

    Args:
        target_keypoints: ``(N, K, 3)`` with ``x, y, visibility``.
        target_boxes: ``(N, 4)`` xyxy boxes; OKS normalizes by their area.
        prediction_keypoints: ``(M, K, 3)`` with ``x, y, conf``.
        sigmas: One positive scale per keypoint.
        area_scale: Multiplier on the target box area (Ultralytics: 0.53).
    """

    truth = np.asarray(target_keypoints, dtype=np.float64)
    boxes = np.asarray(target_boxes, dtype=np.float64)
    predicted = np.asarray(prediction_keypoints, dtype=np.float64)
    if truth.ndim != 3 or truth.shape[2] != 3:
        raise ValueError(f"target_keypoints must be (N, K, 3), got {truth.shape}")
    if boxes.shape != (truth.shape[0], 4):
        raise ValueError(f"target_boxes must be (N, 4), got {boxes.shape}")
    if predicted.ndim != 3 or predicted.shape[2] != 3:
        raise ValueError(f"prediction_keypoints must be (M, K, 3), got {predicted.shape}")
    if truth.shape[1] != predicted.shape[1]:
        raise ValueError(
            f"targets have {truth.shape[1]} keypoints but predictions have {predicted.shape[1]}"
        )
    sigma = np.asarray(sigmas, dtype=np.float64)
    if sigma.shape != (truth.shape[1],):
        raise ValueError(f"sigmas must have {truth.shape[1]} values, got {sigma.shape}")
    if truth.shape[0] == 0 or predicted.shape[0] == 0:
        return np.zeros((truth.shape[0], predicted.shape[0]), dtype=np.float64)
    squared = np.sum(
        (truth[:, None, :, :2] - predicted[None, :, :, :2]) ** 2,
        axis=-1,
    )
    areas = _xyxy_area(boxes) * float(area_scale)
    e = squared / ((2.0 * sigma) ** 2 * (areas[:, None, None] + _EPS) * 2.0)
    visible = truth[:, None, :, 2] > 0.0
    numerator = (np.exp(-e) * visible).sum(axis=-1)
    denominator = visible.sum(axis=-1)[:, 0]
    oks: NDArray[np.float64] = numerator / (denominator[:, None] + _EPS)
    return oks


# ---------------------------------------------------------------------------
# AP and matching
# ---------------------------------------------------------------------------


def average_precision(tp: ArrayLike, n_targets: int) -> float:
    """All-point interpolated AP (Ultralytics ``compute_ap``).

    Args:
        tp: Boolean flags of the detections in descending score order.
        n_targets: Number of ground-truth targets.

    Returns:
        ``nan`` when there are no targets (AP is undefined), otherwise the
        area under the precision-recall envelope in ``[0, 1]``.
    """

    flags = np.asarray(tp, dtype=np.bool_)
    if n_targets <= 0:
        return float("nan")
    if flags.size == 0:
        return 0.0
    tp_cumulative = np.cumsum(flags, dtype=np.float64)
    fp_cumulative = np.cumsum(~flags, dtype=np.float64)
    recall = tp_cumulative / float(n_targets)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1.0)
    mrecall = np.concatenate(([0.0], recall, [1.0]))
    mprecision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mprecision.size - 1, 0, -1):
        mprecision[index - 1] = max(mprecision[index - 1], mprecision[index])
    changes = np.where(mrecall[1:] != mrecall[:-1])[0]
    area = float(np.sum((mrecall[changes + 1] - mrecall[changes]) * mprecision[changes + 1]))
    return min(max(area, 0.0), 1.0)


def _match(
    similarity: NDArray[np.float64],
    scores: NDArray[np.float64],
    target_classes: NDArray[np.intp],
    prediction_classes: NDArray[np.intp],
    threshold: float,
) -> NDArray[np.bool_]:
    """Greedy one-to-one matching by descending score (Ultralytics rule)."""

    n_targets = similarity.shape[0]
    tp = np.zeros(len(scores), dtype=np.bool_)
    if n_targets == 0 or len(scores) == 0:
        return tp
    claimed = np.zeros(n_targets, dtype=np.bool_)
    for index in np.argsort(-scores, kind="stable"):
        candidates = similarity[:, index].copy()
        candidates[claimed] = -1.0
        candidates[target_classes != prediction_classes[index]] = -1.0
        best = int(np.argmax(candidates))
        if candidates[best] >= threshold:
            claimed[best] = True
            tp[index] = True
    return tp


def _stack_predictions(
    predictions: Sequence[PersonPrediction],
    n_kpt: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    if not predictions:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, n_kpt, 3), dtype=np.float64),
            np.zeros(0, dtype=np.intp),
        )
    boxes = np.stack([person.box for person in predictions])
    scores = np.asarray([person.score for person in predictions], dtype=np.float64)
    keypoints = np.stack([person.keypoints for person in predictions])
    classes = np.asarray([person.class_id for person in predictions], dtype=np.intp)
    if keypoints.shape[1] != n_kpt:
        raise ValueError(f"predictions have {keypoints.shape[1]} keypoints, expected {n_kpt}")
    return boxes, scores, keypoints, classes


def _stack_targets(
    targets: Sequence[PersonTarget],
    n_kpt: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    if not targets:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.zeros((0, n_kpt, 3), dtype=np.float64),
            np.zeros(0, dtype=np.intp),
        )
    boxes = np.stack([target.box for target in targets])
    keypoints = np.stack([target.keypoints for target in targets])
    classes = np.asarray([target.class_id for target in targets], dtype=np.intp)
    return boxes, keypoints, classes


def _associate(
    target_box: NDArray[np.float64],
    target_class: int,
    predictions: Sequence[PersonPrediction],
    used: set[int],
    config: EvalConfig,
) -> tuple[int, PersonPrediction] | None:
    """Highest-IoU unused prediction above ``match_iou`` (ties: higher score)."""

    best: tuple[int, PersonPrediction] | None = None
    best_key: tuple[float, float] | None = None
    for index, person in enumerate(predictions):
        if index in used or person.class_id != target_class:
            continue
        overlap = float(_iou_matrix(target_box[None, :], person.box[None, :])[0, 0])
        if overlap < config.match_iou:
            continue
        key = (overlap, float(person.score))
        if best_key is None or key > best_key:
            best = (index, person)
            best_key = key
    return best


def _finite_keypoints(keypoints: NDArray[np.float64]) -> NDArray[np.bool_]:
    finite: NDArray[np.bool_] = np.isfinite(keypoints).all(axis=-1)
    return finite


def _centroid(points: NDArray[np.float64]) -> NDArray[np.float64] | None:
    if len(points) == 0:
        return None
    return np.mean(points[:, :2], axis=0)


def _centroid_error(
    target: NDArray[np.float64],
    prediction: NDArray[np.float64],
    indices: Sequence[int],
    face_width: float,
    config: EvalConfig,
) -> float | None:
    """Face-width-normalized centroid distance over one keypoint group."""

    truth_indices = [index for index in indices if target[index, 2] > 0.0]
    prediction_indices = [
        index
        for index in indices
        if prediction[index, 2] >= config.kpt_conf_thr
        and bool(np.isfinite(prediction[index, :2]).all())
    ]
    if not truth_indices or not prediction_indices:
        return None
    truth_centroid = _centroid(target[truth_indices])
    prediction_centroid = _centroid(prediction[prediction_indices])
    if truth_centroid is None or prediction_centroid is None:
        return None
    return float(np.linalg.norm(truth_centroid - prediction_centroid) / face_width)


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_fw": None, "median_fw": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_fw": float(np.mean(array)),
        "median_fw": float(np.median(array)),
    }


def _framing_bucket(box: ArrayLike, face_width: float | None, ratio: float) -> str:
    """Classify one person as ``full_body``, ``half_body`` or ``unclassified``.

    Uses the box test of ``cpsa.vision.subject.framing_state``: a person counts
    as full body when the box height is at least ``ratio`` face widths.  The
    face/hand visibility gates of the runtime are deliberately ignored here:
    this metric asks whether the *box* describes the person extent, not whether
    the keypoints are confident.  A person without a box or with an unknown
    face width is unclassified, so the fractions still cover every ground-truth
    person.
    """

    if face_width is None or not math.isfinite(face_width) or face_width <= _EPS:
        return "unclassified"
    bounds = np.asarray(box, dtype=np.float64).reshape(-1)
    if bounds.size != 4:
        return "unclassified"
    height = float(bounds[3] - bounds[1])
    if not math.isfinite(height) or height <= 0.0:
        return "unclassified"
    return "full_body" if height / float(face_width) >= ratio else "half_body"


def _framing_summary(counts: dict[str, int], total: int) -> dict[str, Any]:
    """Counts and fractions over all ground-truth persons for one box source."""

    summary: dict[str, Any] = {name: int(value) for name, value in counts.items()}
    for name, value in counts.items():
        summary[f"{name}_fraction"] = value / total if total else float("nan")
    return summary


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    samples: Sequence[ImageSample],
    config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Score decoded predictions against ground truth over a list of images.

    Args:
        samples: One :class:`ImageSample` per image, predictions and targets
            already in the same pixel space.
        config: Thresholds and conventions; see :class:`EvalConfig`.

    Returns:
        A JSON-ready report with ``config``, ``counts``, ``detection``,
        ``pose``, ``groups``, ``centroids`` and ``framing`` (``None`` when no
        framing threshold is available).
    """

    settings = config or EvalConfig()
    n_kpt = settings.n_kpt
    sigma = np.asarray(settings.sigmas, dtype=np.float64)

    detection_scores: list[NDArray[np.float64]] = []
    detection_tp: list[NDArray[np.bool_]] = []
    pose_scores: list[NDArray[np.float64]] = []
    pose_tp: dict[float, list[NDArray[np.bool_]]] = {
        float(threshold): [] for threshold in settings.oks_thresholds
    }

    group_names = [name for name, _ in HALFBUST62_GROUPS]
    group_counts: dict[str, dict[str, int]] = {
        name: {
            "gt_keypoints": 0,
            "missing": 0,
            "unscaled": 0,
            **{f"correct_{threshold:.2f}": 0 for threshold in settings.pck_thresholds},
        }
        for name in group_names
    }
    centroid_errors: dict[str, list[float]] = {
        "left_hand": [],
        "right_hand": [],
        "face": [],
    }
    framing_counts: dict[str, dict[str, int]] = {
        "gt_box": {"full_body": 0, "half_body": 0, "unclassified": 0},
        "predicted_box": {"full_body": 0, "half_body": 0, "unclassified": 0},
    }

    n_targets = 0
    n_predictions = 0
    n_instances = 0
    n_person_missing = 0
    n_with_scale = 0
    n_without_scale = 0

    for sample in samples:
        predictions = [person for person in sample.predictions if person.score > settings.score_thr]
        boxes, scores, keypoints, classes = _stack_predictions(predictions, n_kpt)
        target_boxes, target_keypoints, target_classes = _stack_targets(sample.targets, n_kpt)
        if target_keypoints.shape[1] != n_kpt:
            raise ValueError(
                f"{sample.name}: targets have {target_keypoints.shape[1]} keypoints, "
                f"expected {n_kpt}"
            )
        n_predictions += len(predictions)
        n_targets += len(sample.targets)

        order = np.argsort(-scores, kind="stable")
        if len(scores):
            detection_scores.append(scores[order])
            detection_tp.append(
                _match(
                    _iou_matrix(target_boxes, boxes),
                    scores,
                    target_classes,
                    classes,
                    settings.iou_threshold,
                )[order]
            )

        if len(scores):
            oks = oks_matrix(target_keypoints, target_boxes, keypoints, sigma, settings.area_scale)
            pose_scores.append(scores[order])
            for threshold, flags in pose_tp.items():
                flags.append(_match(oks, scores, target_classes, classes, threshold)[order])

        # Per-instance task metrics.
        used: set[int] = set()
        for target in sample.targets:
            n_instances += 1
            truth = np.asarray(target.keypoints, dtype=np.float64)
            visible = _visible(truth)
            face_width = face_width_px(
                truth,
                source=settings.face_width_source,
                eye_to_face_ratio=settings.eye_to_face_ratio,
            )
            if face_width is None:
                n_without_scale += 1
            else:
                n_with_scale += 1
            association = _associate(target.box, target.class_id, predictions, used, settings)
            if association is None:
                n_person_missing += 1
                person = None
            else:
                matched_index, person = association
                used.add(matched_index)
            predicted = (
                np.asarray(person.keypoints, dtype=np.float64)
                if person is not None
                else np.zeros((n_kpt, 3), dtype=np.float64)
            )
            predicted_finite = _finite_keypoints(predicted)
            detected = (
                predicted_finite & (predicted[:, 2] >= settings.kpt_conf_thr)
                if person is not None
                else np.zeros(n_kpt, dtype=np.bool_)
            )

            if settings.full_body_ratio is not None:
                ratio = settings.full_body_ratio
                framing_counts["gt_box"][_framing_bucket(target.box, face_width, ratio)] += 1
                predicted_bucket = (
                    "unclassified"
                    if person is None
                    else _framing_bucket(person.box, face_width, ratio)
                )
                framing_counts["predicted_box"][predicted_bucket] += 1

            for name, indices in HALFBUST62_GROUPS:
                index = np.asarray(indices, dtype=np.intp)
                group_visible = visible[index]
                counts = group_counts[name]
                counts["gt_keypoints"] += int(group_visible.sum())
                if person is None:
                    counts["missing"] += int(group_visible.sum())
                    continue
                group_missing = group_visible & ~detected[index]
                counts["missing"] += int(group_missing.sum())
                if face_width is None:
                    counts["unscaled"] += int((group_visible & ~group_missing).sum())
                    continue
                errors = np.linalg.norm(predicted[index, :2] - truth[index, :2], axis=1)
                for threshold in settings.pck_thresholds:
                    correct = group_visible & detected[index] & (errors <= threshold * face_width)
                    counts[f"correct_{threshold:.2f}"] += int(correct.sum())

            if face_width is None or person is None:
                continue
            hand_error = _centroid_error(truth, predicted, _LEFT_HAND, face_width, settings)
            if hand_error is not None:
                centroid_errors["left_hand"].append(hand_error)
            hand_error = _centroid_error(truth, predicted, _RIGHT_HAND, face_width, settings)
            if hand_error is not None:
                centroid_errors["right_hand"].append(hand_error)
            face_error = _centroid_error(
                truth, predicted, HALFBUST62_FACE_ANCHOR_INDICES, face_width, settings
            )
            if face_error is not None:
                centroid_errors["face"].append(face_error)

    detection_flags = np.concatenate(detection_tp) if detection_tp else np.zeros(0, dtype=np.bool_)
    detection_sorted = np.concatenate(detection_scores) if detection_scores else np.zeros(0)
    order = np.argsort(-detection_sorted, kind="stable")
    ap50 = average_precision(detection_flags[order], n_targets)

    ap_by_oks: dict[str, float] = {}
    if pose_scores:
        pose_flags_all = np.concatenate(pose_scores)
        order = np.argsort(-pose_flags_all, kind="stable")
        for threshold, flags in pose_tp.items():
            stacked = np.concatenate(flags) if flags else np.zeros(0, dtype=np.bool_)
            ap_by_oks[f"{threshold:.2f}"] = average_precision(stacked[order], n_targets)
    else:
        empty_ap = float("nan") if n_targets == 0 else 0.0
        for threshold in settings.oks_thresholds:
            ap_by_oks[f"{threshold:.2f}"] = empty_ap
    finite_ap = [value for value in ap_by_oks.values() if math.isfinite(value)]
    map50_95 = float(np.mean(finite_ap)) if finite_ap else float("nan")

    groups: dict[str, Any] = {}
    for name in group_names:
        counts = group_counts[name]
        gt_keypoints = counts["gt_keypoints"]
        evaluated = gt_keypoints - counts["missing"] - counts["unscaled"]
        missing_rate = counts["missing"] / gt_keypoints if gt_keypoints else float("nan")
        pck: dict[str, float | None] = {}
        pck_all: dict[str, float | None] = {}
        for threshold in settings.pck_thresholds:
            key = f"{threshold:.2f}"
            correct = counts[f"correct_{threshold:.2f}"]
            pck[key] = correct / evaluated if evaluated > 0 else None
            pck_all[key] = correct / gt_keypoints if gt_keypoints else None
        groups[name] = {
            "gt_keypoints": gt_keypoints,
            "missing": counts["missing"],
            "missing_rate": missing_rate,
            "evaluated": evaluated,
            "pck": pck,
            "pck_all": pck_all,
        }

    hands = centroid_errors["left_hand"] + centroid_errors["right_hand"]
    centroids = {
        "hand": {
            "left": _summarize(centroid_errors["left_hand"]),
            "right": _summarize(centroid_errors["right_hand"]),
            "all": _summarize(hands),
        },
        "face": _summarize(centroid_errors["face"]),
    }

    return {
        "config": settings.to_dict(),
        "counts": {
            "images": len(samples),
            "targets": int(n_targets),
            "predictions": int(n_predictions),
            "instances": int(n_instances),
            "person_missing": int(n_person_missing),
            "person_missing_rate": (
                n_person_missing / n_instances if n_instances else float("nan")
            ),
            "instances_with_face_width": int(n_with_scale),
            "instances_without_face_width": int(n_without_scale),
        },
        "detection": {"ap50": ap50},
        "pose": {
            "map50_95": map50_95,
            "map50": ap_by_oks.get("0.50", float("nan")),
            "ap_by_oks": ap_by_oks,
        },
        "groups": groups,
        "centroids": centroids,
        "framing": (
            None
            if settings.full_body_ratio is None
            else {
                "full_body_ratio": float(settings.full_body_ratio),
                "instances": int(n_instances),
                **{
                    source: _framing_summary(counts, n_instances)
                    for source, counts in framing_counts.items()
                },
            }
        ),
    }
