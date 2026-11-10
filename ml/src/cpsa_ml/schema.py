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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "COCO17",
    "HALFBUST_62",
    "HAND_21",
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
