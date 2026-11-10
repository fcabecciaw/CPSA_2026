"""Keypoint layouts: names, face regions, hands, edges and ML schema sync."""

from __future__ import annotations

import pytest

from cpsa.domain.model import FaceRegion, Side
from cpsa.vision.layouts import COCO17, HALFBUST_62, get_layout


def test_coco17_has_body_keypoints_without_hands() -> None:
    assert len(COCO17) == 17
    assert COCO17.name == "coco17"
    assert COCO17.left_hand == ()
    assert COCO17.right_hand == ()
    assert COCO17.has("nose")
    assert COCO17.index("left_wrist") == 9
    assert not COCO17.has("left_hand_0")


def test_halfbust_has_62_keypoints_in_ml_order() -> None:
    assert len(HALFBUST_62) == 62
    assert HALFBUST_62.name == "halfbust62"
    assert HALFBUST_62.names[:11] == (
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
    )
    assert HALFBUST_62.names[11:20] == (
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
    assert HALFBUST_62.names[20:] == (
        tuple(f"left_hand_{i}" for i in range(21)) + tuple(f"right_hand_{i}" for i in range(21))
    )


def test_names_are_unique_and_indexable() -> None:
    for layout in (COCO17, HALFBUST_62):
        assert len(set(layout.names)) == len(layout)
        for index, name in enumerate(layout.names):
            assert layout.index(name) == index
        with pytest.raises(KeyError):
            layout.index("not_a_keypoint")


def test_hands_are_21_point_chains_per_side() -> None:
    assert len(HALFBUST_62.left_hand) == 21
    assert len(HALFBUST_62.right_hand) == 21
    assert HALFBUST_62.hand(Side.LEFT) == HALFBUST_62.left_hand
    assert HALFBUST_62.hand(Side.RIGHT) == HALFBUST_62.right_hand
    assert HALFBUST_62.names[HALFBUST_62.left_hand[0]] == "left_hand_0"
    assert HALFBUST_62.names[HALFBUST_62.right_hand[20]] == "right_hand_20"


def test_face_regions_follow_the_configured_rule() -> None:
    expected_tokens = {
        FaceRegion.EYES: ("eye", "brow"),
        FaceRegion.NOSE_CHEEK: ("nose", "jaw"),
        FaceRegion.MOUTH: ("mouth", "lip", "chin"),
        FaceRegion.EAR: ("ear",),
    }
    for layout in (COCO17, HALFBUST_62):
        regions = dict(layout.face_regions)
        assert set(regions) == set(layout.face)
        assert len(regions) == len(layout.face_regions)
        for region, tokens in expected_tokens.items():
            for index in layout.face:
                keypoint = layout.names[index]
                if any(token in keypoint for token in tokens):
                    assert regions[index] is region, keypoint


def test_lower_body_keypoints_are_not_required_by_halfbust() -> None:
    for name in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle"):
        assert not HALFBUST_62.has(name)
    assert all("hip" not in name and "knee" not in name for name in HALFBUST_62.names)


def test_edges_reference_valid_keypoints() -> None:
    for layout in (COCO17, HALFBUST_62):
        assert layout.edges
        for first, second in layout.edges:
            assert 0 <= first < len(layout)
            assert 0 <= second < len(layout)
            assert first != second
        assert len(set(layout.edges)) == len(layout.edges)


def test_edges_cover_body_and_finger_chains() -> None:
    assert (0, 1) in COCO17.edges
    assert (5, 6) in COCO17.edges
    assert (7, 9) in COCO17.edges
    hand_edges = ((0, 1), (3, 4), (0, 5), (7, 8), (0, 9), (11, 12), (19, 20))
    for offset in (HALFBUST_62.left_hand[0], HALFBUST_62.right_hand[0]):
        for first, second in hand_edges:
            assert (offset + first, offset + second) in HALFBUST_62.edges


def test_get_layout_resolves_aliases() -> None:
    assert get_layout("coco17") is COCO17
    assert get_layout("COCO17") is COCO17
    assert get_layout(" halfbust62 ") is HALFBUST_62
    assert get_layout("halfbust_62") is HALFBUST_62
    with pytest.raises(ValueError, match="unknown keypoint layout"):
        get_layout("rtmpose")
    with pytest.raises(TypeError):
        get_layout(None)  # type: ignore[arg-type]


def test_layouts_match_the_ml_schema() -> None:
    schema = pytest.importorskip(
        "cpsa_ml.schema", reason="cpsa_ml is not importable in this environment"
    )
    assert COCO17.names == tuple(schema.COCO17.names)
    assert HALFBUST_62.names == tuple(schema.HALFBUST_62.names)
