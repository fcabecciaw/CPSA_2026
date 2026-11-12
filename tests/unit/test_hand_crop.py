"""F-G crop convention: forward map, angle, inverse map and argument checks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.vision.hand_crop import (
    DEFAULT_CROP_SIDE,
    angle_from_up,
    crop_matrix,
    from_crop,
    to_crop,
)

CROP = 192
SIDE = 240.0
CENTRE = (300.0, 200.0)


def test_crop_matrix_maps_the_centre_to_the_crop_centre() -> None:
    matrix = crop_matrix(*CENTRE, SIDE, 0.7)

    mapped = matrix @ np.array([CENTRE[0], CENTRE[1], 1.0])

    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(mapped, [CROP / 2, CROP / 2], atol=1e-9)


def test_crop_matrix_maps_image_up_to_crop_up_for_an_upright_hand() -> None:
    # An upright hand (up = image up) needs no rotation, so the crop keeps y.
    below = to_crop([CENTRE[0], CENTRE[1] + 10.0], *CENTRE, SIDE, 0.0)

    np.testing.assert_allclose(below, [CROP / 2, CROP / 2 + 10.0 * CROP / SIDE], atol=1e-9)


@pytest.mark.parametrize(
    "up",
    [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.6, 0.8), (-0.6, -0.8)],
)
def test_angle_from_up_maps_the_hand_up_direction_onto_image_up(up: tuple[float, float]) -> None:
    angle = angle_from_up(up)
    point = to_crop(
        [CENTRE[0] + up[0], CENTRE[1] + up[1]],
        *CENTRE,
        SIDE,
        angle,
    )

    assert point[0] == pytest.approx(CROP / 2, abs=1e-9)
    assert point[1] < CROP / 2


def test_angle_from_up_does_not_depend_on_the_vector_length() -> None:
    assert angle_from_up((3.0, 4.0)) == pytest.approx(angle_from_up((0.3, 0.4)), abs=1e-12)


def test_to_crop_then_from_crop_round_trips_full_frame_points() -> None:
    generator = np.random.default_rng(7)
    points = generator.uniform([0.0, 0.0], [640.0, 480.0], size=(50, 2))

    crop_points = to_crop(points, *CENTRE, SIDE, -0.9)
    back = from_crop(crop_points, *CENTRE, SIDE, -0.9)

    np.testing.assert_allclose(back, points, atol=1e-9)


def test_crop_matrix_and_to_crop_agree() -> None:
    matrix = crop_matrix(*CENTRE, SIDE, 0.4)
    points = np.array([[0.0, 0.0], [640.0, 480.0], [310.0, 190.0]])

    expected = points @ matrix[:, :2].T + matrix[:, 2]

    np.testing.assert_allclose(to_crop(points, *CENTRE, SIDE, 0.4), expected, atol=1e-9)


def test_to_crop_keeps_the_shape_of_one_point_and_a_batch() -> None:
    assert to_crop([1.0, 2.0], *CENTRE, SIDE, 0.0).shape == (2,)
    assert to_crop([[1.0, 2.0], [3.0, 4.0]], *CENTRE, SIDE, 0.0).shape == (2, 2)
    assert from_crop(np.zeros((3, 2)), *CENTRE, SIDE, 0.0).shape == (3, 2)


def test_default_crop_side_is_the_f_g_model_input() -> None:
    assert DEFAULT_CROP_SIDE == 192
    assert crop_matrix(*CENTRE, SIDE, 0.0)[0, 0] == pytest.approx(DEFAULT_CROP_SIDE / SIDE)


def test_to_crop_puts_the_hand_above_the_centre_in_the_crop() -> None:
    # In the image the fingers are below the wrist (down along +y); the crop
    # must turn the hand so the fingers point up, toward smaller crop rows.
    angle = angle_from_up((0.0, 1.0))
    wrist = to_crop([CENTRE[0], CENTRE[1] - 30.0], *CENTRE, SIDE, angle)
    fingers = to_crop([CENTRE[0], CENTRE[1] + 30.0], *CENTRE, SIDE, angle)

    assert fingers[1] < wrist[1]
    assert fingers[1] < CROP / 2
    assert wrist[1] > CROP / 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"side": 0.0},
        {"side": -4.0},
        {"side": SIDE, "n": 0},
        {"cx": math.nan},
        {"cy": math.inf},
    ],
)
def test_crop_matrix_rejects_invalid_arguments(kwargs: dict[str, float]) -> None:
    arguments = {"cx": CENTRE[0], "cy": CENTRE[1], "side": SIDE, "angle": 0.0}
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        crop_matrix(**arguments)


@pytest.mark.parametrize("points", [[1.0, 2.0, 3.0], np.zeros((4, 3)), np.zeros((0,))])
def test_to_crop_rejects_points_without_two_coordinates(points: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError):
        to_crop(points, *CENTRE, SIDE, 0.0)


def test_angle_from_up_rejects_a_zero_vector() -> None:
    with pytest.raises(ValueError):
        angle_from_up((0.0, 0.0))
