"""Teacher auto-labeler for the booth recordings (stage-1 body, stage-2 hands).

The demo has no usable ground truth for the real booth: the user records a few
thousand frames of seated people with :mod:`scripts.record_frames`, and this
module turns them into two training sets with one teacher pass:

``<out>/halfbust``
    Full-frame YOLO-pose labels on the ``HALFBUST_62`` schema for the stage-1
    body model.  The rtmlib person detector (YOLOX) gives the person box, the
    whole-body teacher (RTMW, 133 keypoints) gives the keypoints, and
    :meth:`cpsa_ml.schema.KeypointSchema.extract` maps them through
    ``HALFBUST_62.wholebody_index`` -- the indices are never repeated here.
    Teacher keypoints below their group threshold or outside the frame get
    ``v = 0``; the rest get ``v = 2`` so a teacher label is never mistaken for
    a human annotation.

``<out>/hand21``
    Hand crops for the stage-2 hand model.  Each accepted hand block becomes a
    :class:`cpsa_ml.datasets.hands.HandSample` and goes through the same crop
    builder as HaGRID, so the deployment framing is identical.  Here the
    handedness *is* known (left block 91-111, right block 112-132) and is
    recorded in ``stats.json``.  A hand block is skipped when fewer than
    ``min_hand_keypoints`` points reach ``min_hand_point_score`` (default 2.0,
    stricter than the 1.0 group threshold), when its extent exceeds
    ``max_hand_fraction`` (default 0.6) of the person box side, or when its
    landmarks fail the palm/middle-finger ratio (:data:`PALM_RATIO_RANGE`).
    When a hand is not visible the teacher often fires a second block over the
    torso or a blurry background; those crops would only teach the hand model
    noise.

Frames and videos are both accepted (``--frames DIR`` and/or ``--video FILE``).
Every frame is assigned to ``train``/``val`` by a seeded hash of its relative
path, so the split is stable across resumable runs.  An interrupted run is
restarted by rerunning the same command: frames whose label already exists are
skipped and ``stats.json`` is rewritten atomically after every chunk.

Teacher
    ``rtmlib`` 0.0.16 with ONNX Runtime.  The default teacher is RTMW-x
    (``rtmw-dw-x-l``, COCO-WholeBody 133 keypoints, 288x384 input); RTMW-l is
    the lighter alternative.  Both are Apache-2.0 via MMPose.  The download
    URLs, input sizes and licenses live in :data:`TEACHER_MODELS` and are
    copied into ``stats.json`` so a run can be reproduced.

The booth recordings contain faces and voices: keep them local, record only
people who agreed, and do not publish the frames.

Example::

    python -m cpsa_ml.pseudolabel \\
        --frames data/booth/P01_2026-06-01T10-00-00Z \\
        --out data/labeled/P01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cpsa_ml.datasets.halfbust import write_dataset_yaml
from cpsa_ml.datasets.hands import (
    HandBuildConfig,
    HandSample,
    SplitStats,
    write_hand_crops,
    write_hand_dataset_yaml,
)
from cpsa_ml.schema import (
    HALFBUST_62,
    HALFBUST_BODY_INDICES,
    HALFBUST_FACE_INDICES,
    HALFBUST_LEFT_HAND_INDICES,
    HALFBUST_RIGHT_HAND_INDICES,
    HAND_21,
    KeypointSchema,
)

__all__ = [
    "DEFAULT_TEACHER",
    "PERSON_DETECTOR",
    "TEACHER_MODELS",
    "AutoLabelConfig",
    "DetectorModel",
    "GroupThresholds",
    "PersonDetector",
    "TeacherModel",
    "WholeBodyTeacher",
    "auto_label",
    "label_from_box",
    "load_bgr",
    "main",
    "split_for",
]

LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
#: Visibility written for a teacher-labeled keypoint.  Ultralytics treats any
#: ``v > 0`` as labelled; ``2`` marks "teacher-labeled" for downstream audits.
FILLED_VISIBILITY = 2.0
#: Smallest accepted hand box in pixels; smaller teacher hands are noise.
MIN_HAND_BOX = 8.0
#: Largest accepted hand extent as a fraction of the larger person box side.
#: A visible hand stays well below it; a hallucinated second hand straddles the
#: torso and exceeds it (measured 0.85-1.07 on real frames).
DEFAULT_MAX_HAND_FRACTION = 0.6
#: Accepted ``d(wrist, middle MCP) / d(middle MCP, middle tip)`` ratio.  In a
#: real hand the middle finger is comparable to the palm when extended and is
#: never shorter than about a third of it even in a fist, so this catches a
#: teacher block whose landmarks are scattered over the background.
PALM_RATIO_RANGE = (0.4, 3.5)
#: Minimum per-landmark teacher score for a hand crop.  The group threshold
#: (1.0) admits weak responses; on the local check every real hand scored >= 2.5
#: on all 21 points while the hallucinated blocks had at most 7 above 2.0.
DEFAULT_MIN_HAND_POINT_SCORE = 2.0


@dataclass(frozen=True, slots=True)
class TeacherModel:
    """A pretrained whole-body teacher checkpoint."""

    name: str
    url: str
    input_size: tuple[int, int]
    license: str
    license_url: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "input_size": list(self.input_size),
            "license": self.license,
            "license_url": self.license_url,
            "notes": self.notes,
        }


TEACHER_MODELS: dict[str, TeacherModel] = {
    "rtmw-x": TeacherModel(
        name="rtmw-dw-x-l_simcc-cocktail14_270e-384x288",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
        ),
        input_size=(288, 384),
        license="Apache-2.0",
        license_url="https://github.com/open-mmlab/mmpose/blob/main/LICENSE",
        notes=(
            "RTMW-x whole-body (133 keypoints), DWPose-style distillation from "
            "Cocktail14; rtmlib 'performance' mode pose model."
        ),
    ),
    "rtmw-l": TeacherModel(
        name="rtmw-dw-l-m_simcc-cocktail14_270e-256x192",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip"
        ),
        input_size=(192, 256),
        license="Apache-2.0",
        license_url="https://github.com/open-mmlab/mmpose/blob/main/LICENSE",
        notes=(
            "RTMW-l whole-body (133 keypoints), distilled from the RTMW-m "
            "teacher; lighter and faster than RTMW-x."
        ),
    ),
}

#: Default teacher: the strongest RTMW checkpoint available through rtmlib.
DEFAULT_TEACHER = "rtmw-x"


@dataclass(frozen=True, slots=True)
class DetectorModel:
    """The person detector used to get full-frame person boxes."""

    name: str
    url: str
    input_size: tuple[int, int]
    license: str
    license_url: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "input_size": list(self.input_size),
            "license": self.license,
            "license_url": self.license_url,
        }


PERSON_DETECTOR = DetectorModel(
    name="yolox-m_8xb8-300e_humanart",
    url=(
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
    ),
    input_size=(640, 640),
    license="Apache-2.0",
    license_url="https://github.com/open-mmlab/mmpose/blob/main/LICENSE",
)


# ---------------------------------------------------------------------------
# HALFBUST_62 group positions (derived from the schema, never hard-coded)
# ---------------------------------------------------------------------------


def _schema_positions(wholebody_indices: Sequence[int]) -> tuple[int, ...]:
    """Return the ``HALFBUST_62`` positions of the given WholeBody indices."""

    wanted = set(wholebody_indices)
    return tuple(i for i, wb in enumerate(HALFBUST_62.wholebody_index) if wb in wanted)


BODY_POSITIONS = _schema_positions(HALFBUST_BODY_INDICES)
FACE_POSITIONS = _schema_positions(HALFBUST_FACE_INDICES)
LEFT_HAND_POSITIONS = _schema_positions(HALFBUST_LEFT_HAND_INDICES)
RIGHT_HAND_POSITIONS = _schema_positions(HALFBUST_RIGHT_HAND_INDICES)

#: Statistics/threshold groups.  Both hands share one threshold.
THRESHOLD_GROUPS = ("body", "face", "hands")
_GROUP_THRESHOLD_POSITIONS: dict[str, tuple[int, ...]] = {
    "body": BODY_POSITIONS,
    "face": FACE_POSITIONS,
    "hands": tuple(sorted(LEFT_HAND_POSITIONS + RIGHT_HAND_POSITIONS)),
}

#: Hand schema positions of the WholeBody hand blocks.
_LEFT_HAND_BLOCK = tuple(range(91, 112))
_RIGHT_HAND_BLOCK = tuple(range(112, 133))


def _check_schema_layout() -> None:
    """Fail fast if ``HALFBUST_62`` or ``HAND_21`` ever change shape."""

    expected = {"body": 11, "face": 9, "hands": 42}
    actual = {group: len(positions) for group, positions in _GROUP_THRESHOLD_POSITIONS.items()}
    if actual != expected:
        raise RuntimeError(f"HALFBUST_62 group layout changed: {actual} != {expected}")
    if HAND_21.wholebody_index != _LEFT_HAND_BLOCK:
        raise RuntimeError("HAND_21 must be the left-hand WholeBody block")


_check_schema_layout()


@dataclass(frozen=True, slots=True)
class GroupThresholds:
    """Teacher score thresholds, per keypoint group.

    The teacher's SimCC scores are raw response maxima, not probabilities, so
    the defaults are calibrated against the recorded frames instead of being
    read as confidences.
    """

    body: float = 0.5
    face: float = 1.0
    hands: float = 1.0

    def __post_init__(self) -> None:
        for name in THRESHOLD_GROUPS:
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} threshold must be finite")

    def for_group(self, group: str) -> float:
        try:
            return float(getattr(self, group))
        except AttributeError as exc:
            raise ValueError(f"unknown threshold group {group!r}") from exc

    def per_position(self) -> NDArray[np.float32]:
        """Return one threshold per ``HALFBUST_62`` position."""

        values = np.full(HALFBUST_62.num_keypoints, np.inf, dtype=np.float32)
        for group, positions in _GROUP_THRESHOLD_POSITIONS.items():
            values[list(positions)] = self.for_group(group)
        return values

    def to_dict(self) -> dict[str, float]:
        return {name: self.for_group(name) for name in THRESHOLD_GROUPS}


# ---------------------------------------------------------------------------
# Images and teachers
# ---------------------------------------------------------------------------


def load_bgr(path: Path) -> NDArray[np.uint8] | None:
    """Read an image as a BGR ``uint8`` array (the convention rtmlib expects)."""

    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        LOGGER.warning("cannot read image %s: %s", path, exc)
        return None
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _import_rtmlib() -> Any:
    try:
        import rtmlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "rtmlib is not installed; install the 'pseudolabel' extra "
            "(`uv sync --extra pseudolabel`, or `pip install rtmlib onnxruntime-gpu`)"
        ) from exc
    return rtmlib


class WholeBodyTeacher:
    """Top-down COCO-WholeBody teacher (one 133-keypoint result per box)."""

    def __init__(
        self,
        model: str | TeacherModel = DEFAULT_TEACHER,
        *,
        device: str = "cpu",
        backend: str = "onnxruntime",
    ) -> None:
        self.model = TEACHER_MODELS[model] if isinstance(model, str) else model
        rtmlib = _import_rtmlib()
        self._estimator = rtmlib.RTMPose(
            self.model.url,
            model_input_size=self.model.input_size,
            backend=backend,
            device=device,
        )

    def __call__(
        self,
        image: NDArray[np.uint8],
        boxes: Sequence[Sequence[float]],
    ) -> list[NDArray[np.float32]]:
        """Estimate keypoints for each ``(x1, y1, x2, y2)`` box in ``image``."""

        if len(boxes) == 0:
            return []
        keypoints, scores = self._estimator(image, bboxes=[list(box) for box in boxes])
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(len(boxes), 133, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(len(boxes), 133)
        return [
            np.concatenate((kpts, score[:, None]), axis=1)
            for kpts, score in zip(keypoints, scores, strict=True)
        ]


class PersonDetector:
    """YOLOX person detector used to get full-frame person boxes."""

    def __init__(
        self,
        *,
        score_thr: float = 0.5,
        device: str = "cpu",
        backend: str = "onnxruntime",
    ) -> None:
        self.score_thr = score_thr
        rtmlib = _import_rtmlib()
        self._detector = rtmlib.YOLOX(
            PERSON_DETECTOR.url,
            model_input_size=PERSON_DETECTOR.input_size,
            score_thr=score_thr,
            backend=backend,
            device=device,
        )

    def __call__(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return ``(N, 4)`` person boxes in pixels.

        ``rtmlib`` applies its ``score_thr`` inside YOLOX and does not return
        the per-box scores, so the threshold is applied there.
        """

        boxes = np.asarray(self._detector(image), dtype=np.float32)
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        boxes = boxes.reshape(-1, 4)
        height, width = image.shape[:2]
        clipped = np.stack(
            [
                np.clip(boxes[:, 0], 0.0, width),
                np.clip(boxes[:, 1], 0.0, height),
                np.clip(boxes[:, 2], 0.0, width),
                np.clip(boxes[:, 3], 0.0, height),
            ],
            axis=1,
        )
        keep = (clipped[:, 2] - clipped[:, 0] >= 2.0) & (clipped[:, 3] - clipped[:, 1] >= 2.0)
        return clipped[keep]


def _module_version(name: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
        return "unknown"


def _teacher_meta(teacher: WholeBodyTeacher) -> dict[str, object]:
    rtmlib = _import_rtmlib()
    return {
        "backend": "rtmlib",
        "rtmlib_version": str(getattr(rtmlib, "version", None) or _module_version("rtmlib")),
        "model": teacher.model.to_dict(),
        "library_license": "Apache-2.0",
        "library_license_url": "https://github.com/Tau-J/rtmlib/blob/main/LICENSE",
    }


# ---------------------------------------------------------------------------
# Label lines
# ---------------------------------------------------------------------------


def label_from_box(
    class_id: int,
    box: Sequence[float],
    keypoints: NDArray[np.float32],
) -> str:
    """Format one YOLO-pose line; ``box`` is normalized ``x1 y1 x2 y2``."""

    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    tokens = [
        str(class_id),
        f"{(x1 + x2) / 2.0:.6f}",
        f"{(y1 + y2) / 2.0:.6f}",
        f"{width:.6f}",
        f"{height:.6f}",
    ]
    for x, y, visibility in keypoints:
        if visibility > 0:
            tokens.extend((f"{float(x):.6f}", f"{float(y):.6f}", str(int(visibility))))
        else:
            tokens.extend(("0.0", "0.0", "0"))
    return " ".join(tokens)


def _accepted_mask(
    wholebody_keypoints: NDArray[np.float32],
    image_size: tuple[int, int],
    thresholds: GroupThresholds,
) -> NDArray[np.bool_]:
    """Return the keypoints the teacher scored above threshold inside the frame."""

    width, height = image_size
    inside = (wholebody_keypoints[:, 0] >= 0.0) & (wholebody_keypoints[:, 0] <= float(width))
    inside &= (wholebody_keypoints[:, 1] >= 0.0) & (wholebody_keypoints[:, 1] <= float(height))
    return inside & (wholebody_keypoints[:, 2] >= thresholds.per_position())


def _person_label(
    wholebody_keypoints: NDArray[np.float32],
    box_pixels: Sequence[float],
    image_size: tuple[int, int],
    thresholds: GroupThresholds,
    schema: KeypointSchema = HALFBUST_62,
) -> str:
    """Build one full-frame YOLO-pose line from ``schema``-ordered keypoints."""

    width, height = image_size
    x = wholebody_keypoints[:, 0] / float(width)
    y = wholebody_keypoints[:, 1] / float(height)
    keep = _accepted_mask(wholebody_keypoints, image_size, thresholds)
    keypoints = np.zeros((schema.num_keypoints, 3), dtype=np.float32)
    keypoints[keep, 0] = np.clip(x[keep], 0.0, 1.0)
    keypoints[keep, 1] = np.clip(y[keep], 0.0, 1.0)
    keypoints[keep, 2] = FILLED_VISIBILITY
    x1, y1, x2, y2 = (float(value) for value in box_pixels)
    return label_from_box(
        0,
        (x1 / width, y1 / height, x2 / width, y2 / height),
        keypoints,
    )


def _hand_sample(
    teacher_keypoints: NDArray[np.float32],
    block: tuple[int, ...],
    handedness: str,
    *,
    frame_path: Path,
    image_size: tuple[int, int],
    person_box: Sequence[float],
    thresholds: GroupThresholds,
    min_keypoints: int,
    expand: float,
    max_fraction: float,
    palm_ratio: tuple[float, float],
    min_score: float,
) -> tuple[HandSample | None, str]:
    """Turn one WholeBody hand block into a crop sample.

    Returns ``(sample, reason)``; ``reason`` is ``"ok"`` on success and
    ``"few_keypoints"``, ``"low_score"``, ``"too_large"``, ``"bad_topology"``
    or ``"tiny_box"`` on rejection.

    ``min_score`` is a stricter per-landmark cutoff than the group threshold:
    it keeps only confident hand points, so a crop label never mixes a real
    hand with a weak response.  ``max_fraction`` bounds the hand extent as a
    fraction of the larger person box side.  When a hand is not visible the
    teacher still fires a second hand block whose keypoints can straddle the
    whole torso; those crops teach the hand model nothing and are rejected
    here.  ``palm_ratio`` adds an anatomical check on the accepted landmarks
    (see :data:`PALM_RATIO_RANGE`), because a diffuse response over a blurry
    background can stay small yet still produce an impossible hand skeleton.
    """

    points = teacher_keypoints[list(block), :3]
    width, height = image_size
    finite = np.isfinite(points).all(axis=1)
    inside = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= float(height))
    )
    usable = finite & inside
    accept = usable & (points[:, 2] >= max(thresholds.hands, min_score))
    if int(np.count_nonzero(accept)) < min_keypoints:
        loose = usable & (points[:, 2] >= thresholds.hands)
        reason = "low_score" if int(np.count_nonzero(loose)) >= min_keypoints else "few_keypoints"
        return None, reason
    accepted = points[accept]

    x1, y1 = float(accepted[:, 0].min()), float(accepted[:, 1].min())
    x2, y2 = float(accepted[:, 0].max()), float(accepted[:, 1].max())
    person_x1, person_y1, person_x2, person_y2 = (float(value) for value in person_box)
    person_span = max(person_x2 - person_x1, person_y2 - person_y1, MIN_HAND_BOX)
    if max(x2 - x1, y2 - y1) > max_fraction * person_span:
        return None, "too_large"

    wrist, middle_mcp, middle_tip = points[0, :2], points[9, :2], points[12, :2]
    palm = float(np.linalg.norm(wrist - middle_mcp))
    finger = float(np.linalg.norm(middle_mcp - middle_tip))
    low, high = palm_ratio
    if not (accept[0] and accept[9] and accept[12]) or finger <= 1e-3:
        return None, "bad_topology"
    if not low <= palm / finger <= high:
        return None, "bad_topology"

    # Expand around the landmark extent so the box covers the visible fingers
    # plus a small halo, then clip to the frame.
    centre_x, centre_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    half_width = max(0.5 * (x2 - x1) * expand, MIN_HAND_BOX / 2.0)
    half_height = max(0.5 * (y2 - y1) * expand, MIN_HAND_BOX / 2.0)
    box = (
        max(0.0, centre_x - half_width),
        max(0.0, centre_y - half_height),
        min(float(width), centre_x + half_width),
        min(float(height), centre_y + half_height),
    )
    if box[2] - box[0] < MIN_HAND_BOX or box[3] - box[1] < MIN_HAND_BOX:
        return None, "tiny_box"

    keypoints = points.copy()
    keypoints[:, 2] = accept.astype(np.float32)
    sample = HandSample(
        image_path=frame_path,
        image_size=image_size,
        box=box,
        keypoints=keypoints,
        source="recorded",
        origin_split="train",
        handedness=handedness,
    )
    return sample, "ok"


# ---------------------------------------------------------------------------
# Configuration and run statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AutoLabelConfig:
    """Inputs, teacher settings and output geometry for one auto-label run.

    Attributes:
        out: Output root; ``halfbust/``, ``hand21/`` and ``stats.json`` live here.
        frames: Directory of recorded frames (searched recursively).
        video: Video file; frames are sampled every ``video_stride`` frames.
        video_stride: Sample every Nth video frame.
        teacher: Teacher checkpoint name in :data:`TEACHER_MODELS`.
        thresholds: Per-group teacher score thresholds.
        person_score: YOLOX person box score threshold.
        device: ONNX Runtime device, e.g. ``cpu`` or ``cuda:0``.
        val_fraction: Share of frames held out for validation (seeded hash).
        seed: Split and crop-jitter seed.
        chunk_size: Frames per ``stats.json`` flush and progress log.
        max_frames: Stop after this many processed frames.
        min_hand_keypoints: Minimum teacher-labeled hand points for a crop.
        min_hand_point_score: Minimum per-landmark score of those points; the
            group threshold alone admits weak responses.
        hand_expand: Expansion of the landmark box before the crop builder.
        max_hand_fraction: Largest hand extent, as a fraction of the larger
            person box side; larger blocks are teacher hallucinations of a
            hand that is not visible and are skipped.
        palm_ratio: Accepted palm/middle-finger ratio of a hand block; see
            :data:`PALM_RATIO_RANGE`.
        halfbust: Write the full-frame ``HALFBUST_62`` dataset.
        hands: Write the ``hand21`` crop dataset.
        imgsz: Hand crop size.
        crop_expand: Hand crop enlargement range (see ``HandBuildConfig``).
        crop_shift: Hand crop centre jitter.
        crop_scale_jitter: Hand crop scale jitter.
        min_visible: Landmarks required inside a hand crop.
        jpeg_quality: JPEG quality of copied frames and crops.
    """

    out: Path
    frames: Path | None = None
    video: Path | None = None
    video_stride: int = 5
    teacher: str = DEFAULT_TEACHER
    thresholds: GroupThresholds = field(default_factory=GroupThresholds)
    person_score: float = 0.5
    device: str = "cpu"
    val_fraction: float = 0.1
    seed: int = 0
    chunk_size: int = 256
    max_frames: int | None = None
    min_hand_keypoints: int = 10
    min_hand_point_score: float = DEFAULT_MIN_HAND_POINT_SCORE
    hand_expand: float = 1.2
    max_hand_fraction: float = DEFAULT_MAX_HAND_FRACTION
    palm_ratio: tuple[float, float] = PALM_RATIO_RANGE
    halfbust: bool = True
    hands: bool = True
    imgsz: int = 192
    crop_expand: tuple[float, float] = (1.6, 2.2)
    crop_shift: float = 0.06
    crop_scale_jitter: float = 0.10
    min_visible: int = 5
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        object.__setattr__(self, "out", Path(self.out))
        if self.frames is None and self.video is None:
            raise ValueError("give frames and/or video")
        if self.frames is not None:
            object.__setattr__(self, "frames", Path(self.frames))
        if self.video is not None:
            object.__setattr__(self, "video", Path(self.video))
        if self.teacher not in TEACHER_MODELS:
            raise ValueError(
                f"teacher must be one of {sorted(TEACHER_MODELS)}, got {self.teacher!r}"
            )
        if not self.halfbust and not self.hands:
            raise ValueError("enable halfbust and/or hands; nothing would be written")
        if self.video_stride <= 0:
            raise ValueError("video_stride must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.min_hand_keypoints < 1 or self.min_hand_keypoints > 21:
            raise ValueError("min_hand_keypoints must be in [1, 21]")
        if not np.isfinite(self.min_hand_point_score):
            raise ValueError("min_hand_point_score must be finite")
        if self.hand_expand < 1.0:
            raise ValueError("hand_expand must be >= 1.0")
        if not 0.0 < self.max_hand_fraction <= 1.0:
            raise ValueError("max_hand_fraction must be in (0, 1]")
        low, high = self.palm_ratio
        if low <= 0.0 or high <= low:
            raise ValueError("palm_ratio must be an increasing positive range")

    def hand_build_config(self) -> HandBuildConfig:
        """Return the hand-crop geometry shared with the public datasets."""

        return HandBuildConfig(
            out=self.out / "hand21",
            imgsz=self.imgsz,
            expand=self.crop_expand,
            shift=self.crop_shift,
            scale_jitter=self.crop_scale_jitter,
            min_visible=self.min_visible,
            jpeg_quality=self.jpeg_quality,
            seed=self.seed,
        )

    def meta(self) -> dict[str, Any]:
        """Settings that must match to resume an existing output tree."""

        return {
            "teacher": self.teacher,
            "thresholds": self.thresholds.to_dict(),
            "person_score": self.person_score,
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "min_hand_keypoints": self.min_hand_keypoints,
            "min_hand_point_score": self.min_hand_point_score,
            "hand_expand": self.hand_expand,
            "max_hand_fraction": self.max_hand_fraction,
            "palm_ratio": list(self.palm_ratio),
            "halfbust": self.halfbust,
            "hands": self.hands,
            "hand_crop": {
                "imgsz": self.imgsz,
                "expand": list(self.crop_expand),
                "shift": self.crop_shift,
                "scale_jitter": self.crop_scale_jitter,
                "min_visible": self.min_visible,
            },
        }


def split_for(relative_path: str, *, seed: int, val_fraction: float) -> str:
    """Assign a stable train/val split from a seeded hash of the path."""

    digest = hashlib.sha256(f"{seed}:{relative_path}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2.0**32
    return "val" if bucket < val_fraction else "train"


#: Frame counters, pre-seeded so ``stats.json`` keeps zero rows readable.
FRAME_COUNTERS = (
    "seen",
    "processed",
    "labeled",
    "no_person",
    "unlabeled",
    "existing",
    "unreadable",
    "teacher_failed",
)


@dataclass
class RunStats:
    """Resumable counters written to ``<out>/stats.json``."""

    meta: dict[str, Any]
    frames: Counter[str] = field(default_factory=lambda: Counter(dict.fromkeys(FRAME_COUNTERS, 0)))
    prompts: Counter[str] = field(default_factory=Counter)
    splits: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "train": {"frames": 0, "persons": 0},
            "val": {"frames": 0, "persons": 0},
        }
    )
    persons: Counter[str] = field(default_factory=Counter)
    keypoints: Counter[str] = field(default_factory=Counter)
    hand_crops: dict[str, SplitStats] = field(
        default_factory=lambda: {"train": SplitStats(), "val": SplitStats()}
    )
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "frames": dict(sorted(self.frames.items())),
            "prompts": dict(sorted(self.prompts.items())),
            "splits": self.splits,
            "persons": dict(sorted(self.persons.items())),
            "keypoints": dict(sorted(self.keypoints.items())),
            "hand_crops": {split: stats.to_dict() for split, stats in self.hand_crops.items()},
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunStats:
        stats = cls(meta=dict(payload.get("meta", {})))
        stats.frames = Counter(payload.get("frames", {}))
        stats.prompts = Counter(payload.get("prompts", {}))
        stats.persons = Counter(payload.get("persons", {}))
        stats.keypoints = Counter(payload.get("keypoints", {}))
        stats.splits = {
            split: {
                "frames": int(values.get("frames", 0)),
                "persons": int(values.get("persons", 0)),
            }
            for split, values in payload.get("splits", {}).items()
        }
        stats.hand_crops = {
            split: SplitStats.from_dict(values)
            for split, values in payload.get("hand_crops", {}).items()
        }
        stats.elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
        return stats


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` atomically so an interrupted run never truncates it."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_stats(path: Path, config: AutoLabelConfig, teacher_meta: dict[str, Any]) -> RunStats:
    """Resume the counters of a previous run, rejecting different settings."""

    meta = config.meta()
    meta["teacher"] = teacher_meta
    if not path.is_file():
        return RunStats(meta=meta)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.warning("ignoring unreadable %s: %s", path, exc)
        return RunStats(meta=meta)
    if payload.get("meta") != meta:
        raise ValueError(
            f"{path} was written with different settings; use a fresh output directory"
        )
    stats = RunStats.from_dict(payload)
    stats.meta = meta
    return stats


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


def _list_frames(root: Path) -> list[Path]:
    """List the recorded frames under ``root`` in a stable order."""

    frames = [
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(frames)


def _iter_video_frames(path: Path, stride: int) -> Iterator[tuple[str, NDArray[np.uint8]]]:
    """Yield every ``stride``-th BGR frame of ``path`` with a stable name."""

    import cv2  # noqa: PLC0415 - optional dependency of the pseudolabel extra

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % stride == 0:
                yield f"frame_{index:06d}", np.ascontiguousarray(frame)
            index += 1
    finally:
        capture.release()


def _frame_name(path: Path, root: Path) -> str:
    """Return a unique, filesystem-safe stem for a recorded frame."""

    relative = path.relative_to(root).with_suffix("").as_posix()
    return relative.replace("/", "__").replace(" ", "_")


def _write_frame(
    destination: Path,
    image: NDArray[np.uint8] | None,
    source_path: Path | None,
    jpeg_quality: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path is not None:
        shutil.copy2(source_path, destination)
        return
    if image is None:  # pragma: no cover - defensive
        raise ValueError("either the image or the source path is required")
    Image.fromarray(image[:, :, ::-1]).save(
        destination, format="JPEG", quality=jpeg_quality, subsampling=0
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _label_frame(
    image: NDArray[np.uint8],
    boxes: NDArray[np.float32],
    *,
    frame_path: Path,
    label_path: Path,
    split: str,
    config: AutoLabelConfig,
    teacher: Callable[[NDArray[np.uint8], Sequence[Sequence[float]]], list[NDArray[np.float32]]],
    hand_config: HandBuildConfig,
    stats: RunStats,
) -> bool:
    """Label one already-written frame; return whether a label was written."""

    height, width = image.shape[:2]
    lines: list[str] = []
    hand_samples: list[HandSample] = []
    for box in boxes:
        try:
            result = teacher(image, [box])[0]
        except Exception as exc:  # noqa: BLE001 - one bad person must not stop a run
            LOGGER.warning("teacher failed on %s: %s", frame_path, exc)
            stats.frames["teacher_failed"] += 1
            continue
        wholebody = HALFBUST_62.extract(result)
        lines.append(_person_label(wholebody, box, (width, height), config.thresholds))
        stats.persons["total"] += 1
        stats.keypoints["visible"] += int(
            np.count_nonzero(_accepted_mask(wholebody, (width, height), config.thresholds))
        )
        if not config.hands:
            continue
        for block, handedness in ((_LEFT_HAND_BLOCK, "left"), (_RIGHT_HAND_BLOCK, "right")):
            sample, reason = _hand_sample(
                result,
                block,
                handedness,
                frame_path=frame_path,
                image_size=(width, height),
                person_box=box,
                thresholds=config.thresholds,
                min_keypoints=config.min_hand_keypoints,
                min_score=config.min_hand_point_score,
                expand=config.hand_expand,
                max_fraction=config.max_hand_fraction,
                palm_ratio=config.palm_ratio,
            )
            if sample is None:
                stats.persons[f"no_{handedness}_hand"] += 1
                stats.persons[f"{reason}_{handedness}_hand"] += 1
                continue
            hand_samples.append(sample)

    if not lines:
        return False

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    stats.frames["labeled"] += 1
    stats.splits.setdefault(split, {"frames": 0, "persons": 0})
    stats.splits[split]["frames"] += 1
    stats.splits[split]["persons"] += len(lines)

    if hand_samples:
        counters = stats.hand_crops.setdefault(split, SplitStats())
        counters.source_images += 1
        write_hand_crops(
            hand_config,
            split,
            hand_samples,
            stats=counters,
        )
    return True


def auto_label(
    config: AutoLabelConfig,
    *,
    detector: Callable[[NDArray[np.uint8]], NDArray[np.float32]] | None = None,
    teacher: Callable[[NDArray[np.uint8], Sequence[Sequence[float]]], list[NDArray[np.float32]]]
    | None = None,
) -> dict[str, Any]:
    """Auto-label recorded frames and return the ``stats.json`` payload.

    Args:
        config: Inputs, thresholds and output geometry.
        detector: Optional injected ``image -> (N, 4)`` person detector.
        teacher: Optional injected ``(image, boxes) -> keypoints`` teacher.

    Returns:
        The payload written to ``<out>/stats.json``.

    Raises:
        ValueError: If a source is missing or the output tree was written with
            different settings.
    """

    stats_path = config.out / "stats.json"
    hand_config = config.hand_build_config()
    if teacher is None:
        loaded_teacher = WholeBodyTeacher(config.teacher, device=config.device)
        teacher_meta = _teacher_meta(loaded_teacher)
        run_teacher: Any = loaded_teacher
    else:
        teacher_meta = {"backend": "injected", "model": {"name": type(teacher).__name__}}
        run_teacher = teacher
    run_detector: Any = (
        detector
        if detector is not None
        else PersonDetector(score_thr=config.person_score, device=config.device)
    )

    stats = _load_stats(stats_path, config, teacher_meta)
    frame_root = config.out / ("halfbust" if config.halfbust else "frames")
    for directory in (
        frame_root / "images" / "train",
        frame_root / "images" / "val",
        frame_root / "labels" / "train",
        frame_root / "labels" / "val",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if config.hands:
        for split in ("train", "val"):
            (hand_config.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (hand_config.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    pending = 0
    processed = 0
    chunk_start = started
    for name, image, source_path in _iter_sources(config):
        split = split_for(name, seed=config.seed, val_fraction=config.val_fraction)
        prompt = source_path.parent.name if source_path is not None else Path(name).stem
        stats.frames["seen"] += 1
        stats.prompts[prompt] += 1

        destination = frame_root / "images" / split / f"{name}.jpg"
        label_path = frame_root / "labels" / split / f"{name}.txt"
        if label_path.is_file() and destination.is_file():
            stats.frames["existing"] += 1
            continue

        if image is None and source_path is not None:
            image = load_bgr(source_path)
        if image is None:
            stats.frames["unreadable"] += 1
            continue

        boxes = run_detector(image)
        if len(boxes) == 0:
            stats.frames["no_person"] += 1
        else:
            _write_frame(destination, image, source_path, config.jpeg_quality)
            if not _label_frame(
                image,
                boxes,
                frame_path=destination,
                label_path=label_path,
                split=split,
                config=config,
                teacher=run_teacher,
                hand_config=hand_config,
                stats=stats,
            ):
                stats.frames["unlabeled"] += 1
        processed += 1
        pending += 1
        stats.frames["processed"] += 1

        if pending >= config.chunk_size:
            stats.elapsed_seconds += time.monotonic() - chunk_start
            chunk_start = time.monotonic()
            _write_json_atomic(stats_path, stats.to_dict())
            LOGGER.info("auto-labeled %d frames", stats.frames["processed"])
            pending = 0
        if config.max_frames is not None and processed >= config.max_frames:
            break

    stats.elapsed_seconds += time.monotonic() - chunk_start
    if config.halfbust:
        write_dataset_yaml(config.out / "halfbust", HALFBUST_62, split="train")
        (config.out / "halfbust" / "schema.json").write_text(
            json.dumps(HALFBUST_62.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if config.hands:
        write_hand_dataset_yaml(config.out / "hand21", HAND_21)
        (config.out / "hand21" / "schema.json").write_text(
            json.dumps(HAND_21.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    payload = stats.to_dict()
    payload["elapsed_seconds"] = round(stats.elapsed_seconds + time.monotonic() - started, 3)
    _write_json_atomic(stats_path, payload)
    LOGGER.info(
        "done: %d frames processed, %d labeled, %d hand crops",
        payload["frames"].get("processed", 0),
        payload["frames"].get("labeled", 0),
        sum(value.images_written for value in stats.hand_crops.values()),
    )
    return payload


def _iter_sources(
    config: AutoLabelConfig,
) -> Iterator[tuple[str, NDArray[np.uint8] | None, Path | None]]:
    """Yield ``(name, image, source_path)`` for every configured source."""

    if config.frames is not None:
        root = config.frames
        if not root.is_dir():
            raise ValueError(f"frames directory not found: {root}")
        for path in _list_frames(root):
            yield _frame_name(path, root), None, path
    if config.video is not None:
        if not config.video.is_file():
            raise ValueError(f"video not found: {config.video}")
        for name, frame in _iter_video_frames(config.video, config.video_stride):
            yield f"{config.video.stem}__{name}", frame, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, default=None, help="recorded frames (recursive)")
    parser.add_argument("--video", type=Path, default=None, help="video file to sample")
    parser.add_argument("--stride", type=int, default=5, help="keep every Nth video frame")
    parser.add_argument("--out", type=Path, required=True, help="output root")
    parser.add_argument("--teacher", choices=sorted(TEACHER_MODELS), default=DEFAULT_TEACHER)
    parser.add_argument("--device", default="cpu", help="onnxruntime device, e.g. cpu or cuda:0")
    parser.add_argument("--person-score", type=float, default=0.5)
    parser.add_argument("--body-score", type=float, default=GroupThresholds().body)
    parser.add_argument("--face-score", type=float, default=GroupThresholds().face)
    parser.add_argument("--hand-score", type=float, default=GroupThresholds().hands)
    parser.add_argument("--min-hand-keypoints", type=int, default=10)
    parser.add_argument(
        "--hand-point-score",
        type=float,
        default=DEFAULT_MIN_HAND_POINT_SCORE,
        help="minimum teacher score per accepted hand landmark",
    )
    parser.add_argument("--hand-expand", type=float, default=1.2)
    parser.add_argument(
        "--max-hand-fraction",
        type=float,
        default=DEFAULT_MAX_HAND_FRACTION,
        help="largest hand extent as a fraction of the person box side",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=192)
    parser.add_argument("--expand", default="1.6,2.2")
    parser.add_argument("--shift", type=float, default=0.06)
    parser.add_argument("--scale-jitter", type=float, default=0.10)
    parser.add_argument("--min-visible", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--no-halfbust", action="store_true", help="skip the full-frame dataset")
    parser.add_argument("--no-hands", action="store_true", help="skip the hand crops")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point of the auto-labeler."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.frames is None and args.video is None:
        raise SystemExit("give --frames DIR and/or --video FILE")
    low, _, high = args.expand.partition(",")
    try:
        payload = auto_label(
            AutoLabelConfig(
                out=args.out,
                frames=args.frames,
                video=args.video,
                video_stride=args.stride,
                teacher=args.teacher,
                thresholds=GroupThresholds(
                    body=args.body_score, face=args.face_score, hands=args.hand_score
                ),
                person_score=args.person_score,
                device=args.device,
                val_fraction=args.val_fraction,
                seed=args.seed,
                chunk_size=args.chunk_size,
                max_frames=args.max_frames,
                min_hand_keypoints=args.min_hand_keypoints,
                min_hand_point_score=args.hand_point_score,
                hand_expand=args.hand_expand,
                max_hand_fraction=args.max_hand_fraction,
                halfbust=not args.no_halfbust,
                hands=not args.no_hands,
                imgsz=args.imgsz,
                crop_expand=(float(low), float(high or low)),
                crop_shift=args.shift,
                crop_scale_jitter=args.scale_jitter,
                min_visible=args.min_visible,
                jpeg_quality=args.jpeg_quality,
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
