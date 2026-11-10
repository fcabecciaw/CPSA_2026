"""Schema and flip-pair tests for HALFBUST_62 and COCO17."""

from __future__ import annotations

import numpy as np
import pytest

from cpsa_ml.schema import COCO17, HALFBUST_62, get_schema


def test_halfbust_62_has_expected_indices() -> None:
    expected = (
        tuple(range(11))
        + (23, 39, 31, 42, 47, 71, 77, 74, 80)
        + tuple(range(91, 112))
        + tuple(range(112, 133))
    )
    assert HALFBUST_62.wholebody_index == expected
    assert len(HALFBUST_62.names) == 62
    assert HALFBUST_62.kpt_shape == (62, 3)


def test_flip_idx_is_an_involution() -> None:
    for schema in (HALFBUST_62, COCO17):
        assert len(schema.flip_idx) == len(schema.names)
        for i, j in enumerate(schema.flip_idx):
            assert schema.flip_idx[j] == i
            assert j == schema.flip_idx[i]


def test_left_right_names_pair_correctly() -> None:
    for schema in (HALFBUST_62, COCO17):
        names = schema.names
        for i, name in enumerate(names):
            mirrored = names[schema.flip_idx[i]]
            if "left" in name:
                assert "right" in mirrored, (name, mirrored)
            elif "right" in name:
                assert "left" in mirrored, (name, mirrored)
            else:
                assert mirrored == name, (name, mirrored)


def test_halfbust_face_pairs_and_self_maps() -> None:
    names = HALFBUST_62.names
    # Positions 11-19 are the nine face points.
    assert names[11] == "face_jaw_right"
    assert HALFBUST_62.flip_idx[11] == 12
    assert HALFBUST_62.flip_idx[12] == 11
    assert HALFBUST_62.flip_idx[13] == 13  # chin
    assert HALFBUST_62.flip_idx[14] == 15  # brows
    assert HALFBUST_62.flip_idx[16] == 17  # mouth corners
    assert HALFBUST_62.flip_idx[18] == 18  # upper lip
    assert HALFBUST_62.flip_idx[19] == 19  # lower lip


def test_hands_pair_same_finger_point() -> None:
    for index in range(21):
        left = 20 + index
        right = 41 + index
        assert HALFBUST_62.flip_idx[left] == right
        assert HALFBUST_62.flip_idx[right] == left


def test_coco17_flip_pairs() -> None:
    assert COCO17.flip_idx == (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


def test_extract_uses_schema_order() -> None:
    wholebody = np.zeros((133, 3), dtype=np.float32)
    for index in range(133):
        wholebody[index] = (index, index + 0.5, 2.0)
    extracted = HALFBUST_62.extract(wholebody)
    assert extracted.shape == (62, 3)
    assert extracted[0, 0] == 0
    assert extracted[11, 0] == 23
    assert extracted[20, 0] == 91
    assert extracted[-1, 0] == 132


def test_extract_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="133"):
        HALFBUST_62.extract(np.zeros((17, 3), dtype=np.float32))


def test_get_schema_aliases() -> None:
    assert get_schema("HALFBUST_62") is HALFBUST_62
    assert get_schema("halfbust62") is HALFBUST_62
    assert get_schema("COCO17") is COCO17
    assert get_schema(COCO17) is COCO17
    with pytest.raises(ValueError, match="unknown keypoint schema"):
        get_schema("bogus")
