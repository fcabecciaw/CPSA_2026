"""COCO-80 label tables of the showcase detector (F-S1)."""

from __future__ import annotations

import pytest

from cpsa.vision.coco import COCO80_EN, COCO80_IT

KNOWN = {
    0: ("person", "persona"),
    1: ("bicycle", "bicicletta"),
    2: ("car", "auto"),
    16: ("dog", "cane"),
    39: ("bottle", "bottiglia"),
    41: ("cup", "tazza"),
    67: ("cell phone", "cellulare"),
    76: ("scissors", "forbici"),
    77: ("teddy bear", "orsacchiotto"),
    79: ("toothbrush", "spazzolino"),
}


def test_both_tables_have_the_eighty_coco_classes() -> None:
    assert len(COCO80_EN) == 80
    assert len(COCO80_IT) == 80


def test_index_zero_is_person_in_both_languages() -> None:
    assert COCO80_EN[0] == "person"
    assert COCO80_IT[0] == "persona"


@pytest.mark.parametrize("index", sorted(KNOWN))
def test_known_class_indices(index: int) -> None:
    english, italian = KNOWN[index]
    assert COCO80_EN[index] == english
    assert COCO80_IT[index] == italian


def test_italian_labels_are_ascii_for_the_hershey_fonts() -> None:
    for label in COCO80_IT:
        assert label.isascii(), label


def test_labels_are_non_empty_and_unique_per_language() -> None:
    for labels in (COCO80_EN, COCO80_IT):
        assert all(label.strip() for label in labels)
        assert len(set(labels)) == len(labels)
