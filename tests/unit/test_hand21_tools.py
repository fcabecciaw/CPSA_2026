"""Hand21 helpers: deterministic data prep and the IR overlay sheet."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "vitis"
for _path in (TOOLS, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import overlay_hand21 as ov  # noqa: E402
import prepare_hand21_data as prep  # noqa: E402

from cpsa.vision.types import PersonPose  # noqa: E402


def test_select_paths_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index:02d}.png" for index in range(12)]

    first = prep.select_paths(paths, 5, 4, seed=0)
    again = prep.select_paths(paths, 5, 4, seed=0)
    other = prep.select_paths(paths, 5, 4, seed=1)

    assert first == again
    assert not set(first[0]) & set(first[1])
    assert len(first[0]) == 5 and len(first[1]) == 4
    assert first[0] != other[0]


def test_select_paths_rejects_impossible_counts(tmp_path: Path) -> None:
    paths = [tmp_path / "a.png", tmp_path / "b.png"]

    with np.testing.assert_raises(ValueError):
        prep.select_paths(paths, 2, 1)
    with np.testing.assert_raises(ValueError):
        prep.select_paths(paths, 0, 1)


def test_write_ir_frames_replicates_the_runtime_input(tmp_path: Path) -> None:
    frames = np.arange(3 * 16 * 16, dtype=np.uint8).reshape(3, 16, 16)
    names = prep.write_ir_frames(frames, 2, tmp_path)

    assert names == ["ir_0000.png", "ir_0001.png"]
    for name in names:
        with Image.open(tmp_path / name) as image:
            array = np.asarray(image)
        assert array.shape == (16, 16, 3)
        np.testing.assert_array_equal(array[..., 0], array[..., 1])
        np.testing.assert_array_equal(array[..., 1], array[..., 2])


def _person(box: tuple[float, float, float, float] = (10.0, 10.0, 150.0, 110.0)) -> PersonPose:
    x1, y1, x2, y2 = box
    xy = np.stack([np.linspace(x1, x2, 21), np.linspace(y1, y2, 21)], axis=1)
    return PersonPose(
        box=np.asarray(box),
        score=0.8,
        xy=xy,
        conf=np.ones(21),
    )


def test_draw_persons_draws_bones_and_joints() -> None:
    image = np.full((120, 160, 3), 30, dtype=np.uint8)

    canvas = np.asarray(ov.draw_persons(image, [_person()]))

    assert canvas.shape == image.shape
    assert np.count_nonzero(canvas != image) > 100
    assert np.count_nonzero(np.all(canvas == ov.BOX_COLOUR, axis=-1)) > 0
    assert np.count_nonzero(np.all(canvas == ov.WRIST_COLOUR, axis=-1)) > 0


def test_draw_persons_skips_low_confidence_keypoints() -> None:
    image = np.full((120, 160, 3), 30, dtype=np.uint8)
    person = _person()
    person = PersonPose(box=person.box, score=person.score, xy=person.xy, conf=np.zeros(21))

    canvas = np.asarray(ov.draw_persons(image, [person]))

    assert np.count_nonzero(np.all(canvas == ov.BOX_COLOUR, axis=-1)) > 0
    assert np.count_nonzero(np.all(canvas == ov.BONE_COLOUR, axis=-1)) == 0
    assert np.count_nonzero(np.all(canvas == ov.JOINT_COLOUR, axis=-1)) == 0


def test_contact_sheet_lays_out_a_grid() -> None:
    tiles = [(Image.new("RGB", (40, 30), (10, 20, 30)), f"tile {index}") for index in range(5)]

    sheet = ov.contact_sheet(tiles, columns=3, tile_width=40, caption_height=10)

    assert sheet.size == (120, 2 * (30 + 10))
