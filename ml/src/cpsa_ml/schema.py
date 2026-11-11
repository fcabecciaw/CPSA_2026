"""Keypoint schemas shared by the CPSA vision models.

The canonical source is COCO-WholeBody (133 keypoints).  A schema selects an
ordered subset of those keypoints and stores the naming and horizontal-flip
information that Ultralytics needs in a dataset YAML.

COCO-WholeBody index layout (https://github.com/jin-s13/COCO-WholeBody):

======================  =================
Range                   Group
======================  =================
0-16                    body (COCO 17)
17-22                   feet (6)
23-90                   face (iBUG 68, index = 23 + iBUG)
91-111                  left hand (21)
112-132                 right hand (21)
======================  =================

The hand blocks use the MediaPipe 21-landmark order, so ``HAND_21`` (the
left-hand block) is also the schema of the standalone hand model fed by
MediaPipe-style labels (HaGRID, Ultralytics hand-keypoints).

Each schema also derives the Ultralytics ``kpt_oks_sigmas`` of its points
(:attr:`KeypointSchema.oks_sigmas`) from the COCO-WholeBody annotation-scale
constants, so every generated ``data.yaml`` carries a usable keypoint loss
weighting instead of the ``1 / nkpt`` fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "COCO17",
    "COCO_WHOLEBODY_SIGMAS",
    "HALFBUST_62",
    "HAND_21",
    "OKS_SIGMA_FLOOR",
    "SCHEMAS",
    "KeypointSchema",
    "get_schema",
]

# ---------------------------------------------------------------------------
# COCO-17
# ---------------------------------------------------------------------------

COCO17_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# ---------------------------------------------------------------------------
# OKS sigmas
# ---------------------------------------------------------------------------

#: Per-keypoint standard deviations of COCO-WholeBody (133 entries), ordered
#: like the WholeBody index layout above.  Copied verbatim from the
#: authoritative annotation-scale constants of mmpose,
#: ``configs/_base_/datasets/coco_wholebody.py`` (``sigmas=[...]``), which
#: cites COCO-WholeBody ``evaluation/myeval_wholebody.py``.  Neither mmpose nor
#: xtcocotools is installed in ``ml/.venv``, so the values are hard-coded here
#: and ``tests/test_oks_sigmas.py`` pins their length plus representative
#: body, face and hand entries.  The first 17 values are the COCO-17 sigmas
#: Ultralytics' ``OKS_SIGMA`` uses (already divided by 10); WholeBody continues
#: them for the feet, the iBUG-68 face and both hands.
COCO_WHOLEBODY_SIGMAS: tuple[float, ...] = (
    # body (COCO-17)
    (0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079)
    + (0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089)
    # feet (6)
    + (0.068, 0.066, 0.066, 0.092, 0.094, 0.094)
    # face (iBUG-68)
    + (0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031, 0.025, 0.020, 0.023)
    + (0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011)
    + (0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007)
    + (0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011)
    + (0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008, 0.008)
    + (0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009, 0.009, 0.007)
    + (0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.010, 0.008)
    # left hand (21)
    + (0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018, 0.024)
    + (0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.020, 0.019, 0.022, 0.031)
    # right hand (21)
    + (0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018, 0.024)
    + (0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.020, 0.019, 0.022, 0.031)
)

#: Lower bound applied to every schema sigma.  ``0.025`` is the smallest
#: COCO-17 sigma (eyes), which is what standard YOLO-pose training uses for
#: those points.  The iBUG face landmarks and the hand joints have much smaller
#: annotation-scale constants (down to 0.007), and Ultralytics' keypoint loss
#: ``1 - exp(-e / sigma)`` saturates at 1 for a joint whose error ``e`` exceeds
#: a few times sigma; its gradient ``exp(-e / sigma) / sigma`` then vanishes and
#: a joint that starts far from its target never converges.  Flooring at the
#: eye sigma keeps such points learnable.
OKS_SIGMA_FLOOR = 0.025

# ---------------------------------------------------------------------------
# HALFBUST-62
# ---------------------------------------------------------------------------

# Body 0-10 is the COCO body subset that remains visible in a head-to-hip crop.
HALFBUST_BODY_INDICES = tuple(range(11))
HALFBUST_BODY_NAMES = COCO17_NAMES[:11]

# Face points use WholeBody = 23 + iBUG68.  iBUG 0 and 16 are the right and
# left jaw ends, 19 and 24 the eyebrow centres, 48 and 54 the mouth corners,
# 51 and 57 the upper/lower lip centres, and 8 the chin.
HALFBUST_FACE_INDICES = (23, 39, 31, 42, 47, 71, 77, 74, 80)
HALFBUST_FACE_NAMES = (
    "face_jaw_right",  # iBUG 0
    "face_jaw_left",  # iBUG 16
    "face_chin",  # iBUG 8
    "face_brow_right",  # iBUG 19
    "face_brow_left",  # iBUG 24
    "face_mouth_right",  # iBUG 48
    "face_mouth_left",  # iBUG 54
    "face_lip_upper",  # iBUG 51
    "face_lip_lower",  # iBUG 57
)

HALFBUST_LEFT_HAND_INDICES = tuple(range(91, 112))
HALFBUST_RIGHT_HAND_INDICES = tuple(range(112, 133))
HALFBUST_LEFT_HAND_NAMES = tuple(f"left_hand_{i}" for i in range(21))
HALFBUST_RIGHT_HAND_NAMES = tuple(f"right_hand_{i}" for i in range(21))

HALFBUST_62_INDICES = (
    HALFBUST_BODY_INDICES
    + HALFBUST_FACE_INDICES
    + HALFBUST_LEFT_HAND_INDICES
    + HALFBUST_RIGHT_HAND_INDICES
)
HALFBUST_62_NAMES = (
    HALFBUST_BODY_NAMES + HALFBUST_FACE_NAMES + HALFBUST_LEFT_HAND_NAMES + HALFBUST_RIGHT_HAND_NAMES
)


def _counterpart(name: str) -> str:
    """Return the horizontally mirrored name for a keypoint.

    Names encode left/right explicitly.  Mid-line points (nose, chin, lips)
    and other self-mapped points are returned unchanged.
    """

    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    if "_left" in name:
        return name.replace("_left", "_right", 1)
    if "_right" in name:
        return name.replace("_right", "_left", 1)
    return name


def _derive_flip_idx(names: tuple[str, ...]) -> tuple[int, ...]:
    """Derive ``flip_idx`` by swapping explicit left/right names."""

    index = {name: i for i, name in enumerate(names)}
    flip: list[int] = []
    for name in names:
        mirrored = _counterpart(name)
        try:
            flip.append(index[mirrored])
        except KeyError as exc:  # pragma: no cover - defensive, checked by tests
            raise ValueError(f"no mirror partner for keypoint {name!r}") from exc
    return tuple(flip)


@dataclass(frozen=True, slots=True)
class KeypointSchema:
    """An ordered keypoint subset of COCO-WholeBody."""

    name: str
    wholebody_index: tuple[int, ...]
    names: tuple[str, ...]
    flip_idx: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.wholebody_index) != len(self.names):
            raise ValueError(f"{self.name}: index/name length mismatch")
        if len(self.flip_idx) != len(self.names):
            raise ValueError(f"{self.name}: flip_idx length mismatch")
        if len(set(self.wholebody_index)) != len(self.wholebody_index):
            raise ValueError(f"{self.name}: duplicate COCO-WholeBody indices")
        if any(i < 0 or i >= 133 for i in self.wholebody_index):
            raise ValueError(f"{self.name}: COCO-WholeBody index out of range")
        for i, j in enumerate(self.flip_idx):
            if j < 0 or j >= len(self.names) or self.flip_idx[j] != i:
                raise ValueError(f"{self.name}: flip_idx is not an involution at {i}")

    @property
    def num_keypoints(self) -> int:
        return len(self.names)

    @property
    def kpt_shape(self) -> tuple[int, int]:
        """Ultralytics ``kpt_shape``: ``(num_keypoints, 3)``."""

        return (len(self.names), 3)

    @property
    def oks_sigmas(self) -> tuple[float, ...]:
        """Ultralytics ``kpt_oks_sigmas`` for this schema, in schema order.

        Written into every generated ``data.yaml``: Ultralytics' pose trainer
        stores the dataset's ``kpt_oks_sigmas`` on the model and
        :class:`ultralytics.utils.loss.v8PoseLoss` otherwise falls back to
        ``ones(nkpt) / nkpt`` for any ``kpt_shape`` other than ``[17, 3]``,
        which saturates the keypoint loss for far-away joints.

        Each value is the COCO-WholeBody sigma of the matching
        :attr:`wholebody_index` entry, floored at :data:`OKS_SIGMA_FLOOR`, and
        mirrored pairs are unified by taking the larger value so a horizontal
        flip cannot change the loss (WholeBody's two jaw-end sigmas differ:
        0.042 vs 0.045).
        """

        sigmas = [
            max(OKS_SIGMA_FLOOR, COCO_WHOLEBODY_SIGMAS[index]) for index in self.wholebody_index
        ]
        for index, mirror in enumerate(self.flip_idx):
            paired = max(sigmas[index], sigmas[mirror])
            sigmas[index] = paired
            sigmas[mirror] = paired
        return tuple(sigmas)

    def extract(self, wholebody: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float32]:
        """Select this schema's keypoints from a ``(133, 3)`` array.

        Args:
            wholebody: COCO-WholeBody keypoints ordered as ``(133, 3)`` with
                ``(x, y, visibility)`` rows.

        Returns:
            A ``(K, 3)`` float32 copy in schema order.
        """

        array = np.asarray(wholebody, dtype=np.float32)
        if array.shape != (133, 3):
            raise ValueError(f"expected COCO-WholeBody keypoints (133, 3), got {array.shape}")
        return array[list(self.wholebody_index), :].copy()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kpt_shape": list(self.kpt_shape),
            "wholebody_index": list(self.wholebody_index),
            "names": list(self.names),
            "flip_idx": list(self.flip_idx),
        }


COCO17 = KeypointSchema(
    name="coco17",
    wholebody_index=tuple(range(17)),
    names=COCO17_NAMES,
    flip_idx=_derive_flip_idx(COCO17_NAMES),
)

HALFBUST_62 = KeypointSchema(
    name="halfbust62",
    wholebody_index=HALFBUST_62_INDICES,
    names=HALFBUST_62_NAMES,
    flip_idx=_derive_flip_idx(HALFBUST_62_NAMES),
)

# The standalone hand model sees exactly one hand per crop, so the 21 keypoints
# are the left-hand WholeBody block with MediaPipe ordering (wrist, then
# thumb/index/middle/ring/pinky chains of 4).  A crop is mirrored as a whole,
# which keeps each landmark on the same finger joint, so ``flip_idx`` is the
# identity rather than a left/right swap.  Handedness is not a model output; it
# is recorded in the dataset statistics only.
HAND_21_NAMES = tuple(f"hand_{i}" for i in range(21))

HAND_21 = KeypointSchema(
    name="hand21",
    wholebody_index=HALFBUST_LEFT_HAND_INDICES,
    names=HAND_21_NAMES,
    flip_idx=tuple(range(21)),
)

SCHEMAS: dict[str, KeypointSchema] = {
    "halfbust62": HALFBUST_62,
    "halfbust_62": HALFBUST_62,
    "coco17": COCO17,
    "coco17_3d": COCO17,
    "hand21": HAND_21,
    "hand_21": HAND_21,
}


def get_schema(schema: str | KeypointSchema) -> KeypointSchema:
    """Resolve a schema name (case-insensitive) or pass a schema through."""

    if isinstance(schema, KeypointSchema):
        return schema
    try:
        return SCHEMAS[schema.lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown keypoint schema {schema!r}; expected one of {sorted(SCHEMAS)}"
        ) from exc
