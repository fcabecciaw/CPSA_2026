"""Keypoint layouts used by the 2D vision pipeline (FR-004).

Two layouts are provided:

* :data:`COCO17` -- the standard COCO body keypoints, used as a stand-in until
  the half-body model is available.  It has no hand keypoints, so hand shape is
  always ``unknown`` for this layout.
* :data:`HALFBUST_62` -- the target layout: 11 upper-body points, 9 face
  points and 2 x 21 hand points.  The names match ``ml/src/cpsa_ml/schema.py``
  exactly; the ML package is not imported at runtime.

Face anchors are mapped to a :class:`~cpsa.domain.model.FaceRegion`:

* eyes and brows -> ``EYES``;
* nose and jaw ends -> ``NOSE_CHEEK``;
* mouth corners, lips and chin -> ``MOUTH``;
* ears -> ``EAR``.

``edges`` is the skeleton used for drawing (body, face outline and finger
chains).  It contains no lower-body edges because the half-body model has no
lower-body keypoints.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from cpsa.domain.model import FaceRegion
from cpsa.vision.types import KeypointLayout

__all__ = ["COCO17", "HALFBUST_62", "LAYOUTS", "get_layout"]

# ---------------------------------------------------------------------------
# COCO-17
# ---------------------------------------------------------------------------

COCO17_NAMES: tuple[str, ...] = (
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

_COCO17_FACE = (0, 1, 2, 3, 4)
_COCO17_FACE_REGIONS: tuple[tuple[int, FaceRegion], ...] = (
    (0, FaceRegion.NOSE_CHEEK),
    (1, FaceRegion.EYES),
    (2, FaceRegion.EYES),
    (3, FaceRegion.EAR),
    (4, FaceRegion.EAR),
)

_COCO17_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 0),
    (6, 0),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# ---------------------------------------------------------------------------
# HALFBUST-62
# ---------------------------------------------------------------------------

_HALFBUST_BODY_NAMES: tuple[str, ...] = COCO17_NAMES[:11]

_HALFBUST_FACE_NAMES: tuple[str, ...] = (
    "face_jaw_right",
    "face_jaw_left",
    "face_chin",
    "face_brow_right",
    "face_brow_left",
    "face_mouth_right",
    "face_mouth_left",
    "face_lip_upper",
    "face_lip_lower",
)

_HALFBUST_LEFT_HAND_NAMES: tuple[str, ...] = tuple(f"left_hand_{i}" for i in range(21))
_HALFBUST_RIGHT_HAND_NAMES: tuple[str, ...] = tuple(f"right_hand_{i}" for i in range(21))

HALFBUST_62_NAMES: tuple[str, ...] = (
    _HALFBUST_BODY_NAMES
    + _HALFBUST_FACE_NAMES
    + _HALFBUST_LEFT_HAND_NAMES
    + _HALFBUST_RIGHT_HAND_NAMES
)

_HALFBUST_FACE = (0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19)
_HALFBUST_FACE_REGIONS: tuple[tuple[int, FaceRegion], ...] = (
    (0, FaceRegion.NOSE_CHEEK),  # nose
    (1, FaceRegion.EYES),  # left eye
    (2, FaceRegion.EYES),  # right eye
    (3, FaceRegion.EAR),  # left ear
    (4, FaceRegion.EAR),  # right ear
    (11, FaceRegion.NOSE_CHEEK),  # jaw right
    (12, FaceRegion.NOSE_CHEEK),  # jaw left
    (13, FaceRegion.MOUTH),  # chin
    (14, FaceRegion.EYES),  # brow right
    (15, FaceRegion.EYES),  # brow left
    (16, FaceRegion.MOUTH),  # mouth corner right
    (17, FaceRegion.MOUTH),  # mouth corner left
    (18, FaceRegion.MOUTH),  # upper lip
    (19, FaceRegion.MOUTH),  # lower lip
)

_HALFBUST_LEFT_HAND: tuple[int, ...] = tuple(range(20, 41))
_HALFBUST_RIGHT_HAND: tuple[int, ...] = tuple(range(41, 62))

_HAND_EDGES: tuple[tuple[int, int], ...] = (
    # Thumb.
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger.
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger.
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger.
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Little finger.
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    # Palm chain.
    (5, 9),
    (9, 13),
    (13, 17),
)


def _hand_edges(offset: int) -> tuple[tuple[int, int], ...]:
    return tuple((offset + a, offset + b) for a, b in _HAND_EDGES)


_HALFBUST_BODY_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 0),
    (6, 0),
)

_HALFBUST_FACE_EDGES: tuple[tuple[int, int], ...] = (
    # Jaw outline and chin.
    (11, 13),
    (13, 12),
    # Mouth.
    (11, 16),
    (12, 17),
    (16, 17),
    (16, 18),
    (18, 17),
    (16, 19),
    (19, 17),
    (13, 19),
    # Brows and their relation to the eyes.
    (14, 15),
    (14, 1),
    (15, 2),
    # Jaw ends to the ears.
    (11, 3),
    (12, 4),
)

HALFBUST_62 = KeypointLayout(
    name="halfbust62",
    names=HALFBUST_62_NAMES,
    face=_HALFBUST_FACE,
    face_regions=_HALFBUST_FACE_REGIONS,
    left_hand=_HALFBUST_LEFT_HAND,
    right_hand=_HALFBUST_RIGHT_HAND,
    edges=(
        _HALFBUST_BODY_EDGES
        + _HALFBUST_FACE_EDGES
        + _hand_edges(_HALFBUST_LEFT_HAND[0])
        + _hand_edges(_HALFBUST_RIGHT_HAND[0])
    ),
)

COCO17 = KeypointLayout(
    name="coco17",
    names=COCO17_NAMES,
    face=_COCO17_FACE,
    face_regions=_COCO17_FACE_REGIONS,
    left_hand=(),
    right_hand=(),
    edges=_COCO17_EDGES,
)

LAYOUTS: Mapping[str, KeypointLayout] = MappingProxyType(
    {
        "coco17": COCO17,
        "coco17_3d": COCO17,
        "halfbust62": HALFBUST_62,
        "halfbust_62": HALFBUST_62,
    }
)


def get_layout(name: str) -> KeypointLayout:
    """Resolve a configured layout name (case-insensitive).

    Accepts the aliases used by ``ml/src/cpsa_ml/schema.py`` (``coco17``,
    ``coco17_3d``, ``halfbust62``, ``halfbust_62``).
    """

    if not isinstance(name, str):
        raise TypeError(f"layout name must be a string, got {type(name).__name__}")
    try:
        return LAYOUTS[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown keypoint layout {name!r}; expected one of {sorted(LAYOUTS)}"
        ) from exc
