"""Depth hand localizer (F-H) on synthetic depth: plane, arm, hand blob."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("cv2")

from cpsa.adapters.pose.hand_locator import (  # noqa: E402
    HAND_FILL,
    LocatorConfig,
    TableModel,
    calibrate_table,
    foreground_mask,
    gate_detections,
    locate_hands,
)
from cpsa.vision.calibration import Intrinsics  # noqa: E402
from cpsa.vision.hand_crop import to_crop  # noqa: E402

SCALE_M = 3.125e-5
HEIGHT = 480
WIDTH = 640
INTRINSICS = Intrinsics(fx=430.0, fy=430.0, cx=320.0, cy=240.0, width=WIDTH, height=HEIGHT)


def table_depth() -> NDArray[np.float32]:
    """A tilted table: 0.75 m at the far (top) edge, 0.42 m at the near one."""

    rows = np.arange(HEIGHT, dtype=np.float32)[:, np.newaxis]
    return np.broadcast_to(0.75 - 0.0007 * rows, (HEIGHT, WIDTH)).copy()


def disc(centre: tuple[int, int], radius: int) -> NDArray[np.bool_]:
    """Boolean mask of one filled disc."""

    rows, columns = np.ogrid[:HEIGHT, :WIDTH]
    return (columns - centre[0]) ** 2 + (rows - centre[1]) ** 2 <= radius**2


def empty_frame() -> NDArray[np.uint16]:
    """One raw Z16 frame of the empty table."""

    return np.round(table_depth() / SCALE_M).astype(np.uint16)


def as_raw(depth_m: NDArray[np.float32]) -> NDArray[np.uint16]:
    """Quantise a metric depth image to raw Z16."""

    return np.round(depth_m / SCALE_M).astype(np.uint16)


def hand_depth(
    *,
    blob_centre: tuple[int, int] = (320, 300),
    blob_radius: int = 35,
    arm_columns: tuple[int, int] = (300, 340),
    arm_rows: tuple[int, int] = (300, HEIGHT - 1),
    blob_height_m: float = 0.12,
    arm_height_m: float = 0.10,
) -> NDArray[np.float32]:
    """A plane plus an arm box entering from the near edge and a hand blob."""

    depth = table_depth()
    blob = disc(blob_centre, blob_radius)
    arm = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    arm[arm_rows[0] : arm_rows[1] + 1, arm_columns[0] : arm_columns[1] + 1] = True
    depth[arm & ~blob] -= arm_height_m
    depth[blob] -= blob_height_m
    return depth


def two_hand_depth() -> NDArray[np.float32]:
    """A large hand at x=320 and a smaller one at x=500, both over the table."""

    depth = hand_depth()
    small = disc((500, 340), 28)
    arm = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    arm[330 : HEIGHT - 1, 480:520] = True
    depth[arm & ~small] -= 0.10
    depth[small] -= 0.12
    return depth


def tall_object_depth() -> NDArray[np.float32]:
    """An object 0.5 m above the table, above the configured band."""

    depth = table_depth()
    depth[200:350, 100:200] -= 0.5
    return depth


@pytest.fixture(scope="module")
def table() -> TableModel:
    return calibrate_table(
        [empty_frame() for _ in range(4)], scale_m=SCALE_M, intrinsics=INTRINSICS
    )


def _locate(
    depth: NDArray[np.uint16], table: TableModel, config: LocatorConfig | None = None
) -> list:
    return locate_hands(
        depth,
        table,
        scale_m=SCALE_M,
        intrinsics=INTRINSICS,
        config=LocatorConfig() if config is None else config,
    )


def test_calibration_recovers_the_table_plane_and_zone(table: TableModel) -> None:
    n_x, n_y, n_z, d = table.plane

    assert n_z < 0.0  # the normal points from the table toward the camera
    assert d > 0.0
    assert math.hypot(n_x, n_y, n_z) == pytest.approx(1.0)
    assert table.background_m.shape == (HEIGHT, WIDTH)
    assert np.isfinite(table.background_m).all()
    assert table.zone.dtype == np.bool_
    # The background is the table itself: inside the zone and on the plane.
    assert bool(table.zone[240, 320])
    assert bool(table.zone[120, 100])
    assert table.background_m[240, 320] == pytest.approx(table_depth()[240, 320], abs=2e-3)


def test_calibration_keeps_never_measured_pixels_as_nan_and_fills_the_zone() -> None:
    frames = [empty_frame() for _ in range(3)]
    for frame in frames:
        frame[430:470, 40:90] = 0
    table = calibrate_table(frames, scale_m=SCALE_M, intrinsics=INTRINSICS)

    # No background there, so the plane predicts the height; the zone still
    # covers the spot, which is why the fallback exists.
    assert np.isnan(table.background_m[450, 60])
    assert bool(table.zone[450, 60])


def test_empty_table_returns_no_hand(table: TableModel) -> None:
    assert _locate(empty_frame(), table) == []


def test_one_hand_gives_one_upright_crop(table: TableModel) -> None:
    rois = _locate(as_raw(hand_depth()), table)

    assert len(rois) == 1
    roi = rois[0]
    # Entry where the arm crosses the near image edge, tip at the far blob end.
    assert roi.entry == pytest.approx((320.0, HEIGHT - 1), abs=4.0)
    assert roi.tip[1] == pytest.approx(265.0, abs=4.0)
    assert roi.tip[0] == pytest.approx(320.0, abs=12.0)
    # Palm at the widest point of the hand part, not on the narrower arm.
    assert roi.cx == pytest.approx(320.0, abs=6.0)
    assert roi.cy == pytest.approx(300.0, abs=8.0)
    # Wrist to fingers is straight up in the image, so no rotation.
    assert roi.angle == pytest.approx(0.0, abs=0.1)
    assert roi.palm_depth_m == pytest.approx(0.75 - 0.0007 * 300 - 0.12, abs=0.01)
    assert 0.002 < roi.area_m2 < 0.06


def test_the_crop_holds_the_hand_with_the_fingers_up(table: TableModel) -> None:
    roi = _locate(as_raw(hand_depth()), table)[0]

    tip = to_crop(roi.tip, roi.cx, roi.cy, roi.side, roi.angle)
    palm = to_crop((roi.cx, roi.cy), roi.cx, roi.cy, roi.side, roi.angle)

    # The crop holds the hand with the fingers above the palm; the forearm may
    # leave the crop, the entry is on the arm, not on the hand.
    assert 0.0 <= tip[0] <= 192.0
    assert 0.0 <= tip[1] <= 192.0
    assert palm == pytest.approx((96.0, 96.0), abs=1e-9)
    assert tip[1] < palm[1]


def test_crop_side_follows_the_hand_extent(table: TableModel) -> None:
    # A short hand part makes the extent the reach from the tip, so the side is
    # that hand size over the F-G fill fraction.
    config = LocatorConfig(hand_length_m=0.09, min_height_m=0.02)
    roi = _locate(as_raw(hand_depth()), table, config)[0]
    reach_px = 0.09 * INTRINSICS.fx / roi.palm_depth_m

    assert roi.side == pytest.approx((reach_px + 2.0) / HAND_FILL, rel=0.2)


def test_hand_entering_from_a_side_rotates_the_crop(table: TableModel) -> None:
    # Same arm, entering from the image's right edge and pointing left.
    depth = table_depth()
    blob = disc((400, 240), 35)
    arm = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    arm[220:260, 400:WIDTH] = True
    depth[arm & ~blob] -= 0.10
    depth[blob] -= 0.12

    rois = _locate(as_raw(depth), table)

    assert len(rois) == 1
    roi = rois[0]
    assert roi.entry[0] > roi.cx > roi.tip[0]
    assert roi.angle == pytest.approx(math.pi / 2, abs=0.15)
    wrist = to_crop(roi.entry, roi.cx, roi.cy, roi.side, roi.angle)
    tip = to_crop(roi.tip, roi.cx, roi.cy, roi.side, roi.angle)
    assert tip[1] < wrist[1]


def test_a_blob_that_touches_no_border_uses_its_near_edge_point(table: TableModel) -> None:
    # A hand held fully over the table: the entry is the blob point closest to
    # the near image edge, so wrist and tip still span the blob vertically.
    depth = table_depth()
    depth[disc((320, 260), 30)] -= 0.12

    rois = _locate(as_raw(depth), table)

    assert len(rois) == 1
    assert rois[0].entry == pytest.approx((320.0, 290.0), abs=4.0)
    assert rois[0].tip[1] == pytest.approx(230.0, abs=4.0)
    assert rois[0].tip[0] == pytest.approx(320.0, abs=12.0)
    assert rois[0].angle == pytest.approx(0.0, abs=0.25)


def test_two_hands_are_returned_largest_first(table: TableModel) -> None:
    rois = _locate(as_raw(two_hand_depth()), table)

    assert len(rois) == 2
    assert rois[0].area_m2 > rois[1].area_m2
    assert rois[0].cx == pytest.approx(320.0, abs=8.0)
    assert rois[1].cx == pytest.approx(500.0, abs=8.0)


def test_two_hands_are_capped_by_max_hands(table: TableModel) -> None:
    rois = _locate(as_raw(two_hand_depth()), table, LocatorConfig(max_hands=1))

    assert len(rois) == 1


def test_a_tall_object_outside_the_height_band_is_ignored(table: TableModel) -> None:
    assert _locate(as_raw(tall_object_depth()), table) == []


def test_a_tall_object_does_not_hide_a_hand(table: TableModel) -> None:
    depth = hand_depth()
    depth[200:350, 100:200] = tall_object_depth()[200:350, 100:200]

    rois = _locate(as_raw(depth), table)

    assert len(rois) == 1
    assert rois[0].cx == pytest.approx(320.0, abs=8.0)


def test_foreground_mask_marks_the_hand_and_not_the_table(table: TableModel) -> None:
    mask = foreground_mask(
        as_raw(hand_depth()),
        table,
        scale_m=SCALE_M,
        intrinsics=INTRINSICS,
        config=LocatorConfig(),
    )

    assert mask.shape == (HEIGHT, WIDTH)
    assert bool(mask[300, 320])
    assert bool(mask[400, 320])
    assert not bool(mask[240, 100])


def test_locate_hands_rejects_a_frame_that_does_not_match_the_model(table: TableModel) -> None:
    with pytest.raises(ValueError):
        _locate(empty_frame()[:400, :400], table)


@pytest.mark.parametrize(
    "config",
    [
        LocatorConfig(downsample=0),
        LocatorConfig(min_height_m=0.2, max_height_m=0.1),
    ],
)
def test_locate_hands_rejects_an_invalid_config(table: TableModel, config: LocatorConfig) -> None:
    with pytest.raises(ValueError):
        _locate(empty_frame(), table, config)


def test_calibrate_table_rejects_empty_and_mismatched_frames() -> None:
    with pytest.raises(ValueError):
        calibrate_table([], scale_m=SCALE_M, intrinsics=INTRINSICS)
    with pytest.raises(ValueError):
        calibrate_table(
            [empty_frame(), empty_frame()[:100, :100]],
            scale_m=SCALE_M,
            intrinsics=INTRINSICS,
        )
    with pytest.raises(ValueError):
        calibrate_table([empty_frame()], scale_m=0.0, intrinsics=INTRINSICS)


ON_HAND = np.array([[280.0, 250.0, 360.0, 340.0]])
"""A box around the synthetic hand blob and the top of its arm."""


def _gate(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    depth: NDArray[np.uint16],
    table: TableModel,
    config: LocatorConfig | None = None,
) -> NDArray[np.bool_]:
    return gate_detections(
        boxes,
        scores,
        depth,
        table,
        scale_m=SCALE_M,
        intrinsics=INTRINSICS,
        config=LocatorConfig() if config is None else config,
    )


def test_gate_keeps_a_box_on_the_hand_and_rejects_off_hand_boxes(table: TableModel) -> None:
    depth = as_raw(hand_depth())
    boxes = np.array(
        [
            [280.0, 250.0, 360.0, 340.0],  # on the hand
            [20.0, 20.0, 120.0, 140.0],  # empty table
            [280.0, 250.0, 620.0, 470.0],  # hand plus a lot of table: 7% overlap
            [280.0, 250.0, 360.0, 340.0],  # on the hand again
        ]
    )
    scores = np.array([0.9, 0.9, 0.9, 0.2])

    keep = _gate(boxes, scores, depth, table)

    assert keep.tolist() == [True, False, False, True]


def test_gate_reads_min_overlap_from_the_config(table: TableModel) -> None:
    depth = as_raw(hand_depth())

    loose = _gate(ON_HAND, np.array([0.9]), depth, table, LocatorConfig(min_overlap=0.2))
    strict = _gate(ON_HAND, np.array([0.9]), depth, table, LocatorConfig(min_overlap=0.99))

    assert loose.tolist() == [True]
    assert strict.tolist() == [False]


def test_gate_rejects_a_box_too_large_to_be_a_hand(table: TableModel) -> None:
    depth = as_raw(hand_depth())

    keep = _gate(
        ON_HAND,
        np.array([0.9]),
        depth,
        table,
        LocatorConfig(max_hand_size_m=0.05),
    )

    assert keep.tolist() == [False]


def test_gate_applies_the_score_threshold(table: TableModel) -> None:
    depth = as_raw(hand_depth())
    boxes = np.repeat(ON_HAND, 2, axis=0)
    scores = np.array([0.3, 0.9])

    keep = _gate(boxes, scores, depth, table, LocatorConfig(min_score=0.5))

    assert keep.tolist() == [False, True]


def test_gate_keeps_a_box_without_depth_out_of_the_table(table: TableModel) -> None:
    depth = as_raw(hand_depth())
    depth[:] = 0

    keep = _gate(ON_HAND, np.array([0.9]), depth, table)

    assert keep.tolist() == [False]


def test_gate_returns_an_empty_mask_for_no_boxes(table: TableModel) -> None:
    keep = _gate(np.zeros((0, 4)), np.zeros(0), as_raw(hand_depth()), table)

    assert keep.shape == (0,)
    assert keep.dtype == np.bool_


def test_gate_rejects_mismatched_boxes_and_scores(table: TableModel) -> None:
    depth = as_raw(hand_depth())
    with pytest.raises(ValueError):
        _gate(np.repeat(ON_HAND, 2, axis=0), np.array([0.9]), depth, table)
    with pytest.raises(ValueError):
        _gate(np.array([1.0, 2.0, 3.0]), np.array([0.9]), depth, table)
    with pytest.raises(ValueError):
        _gate(ON_HAND, np.array([0.9]), depth, table, LocatorConfig(min_overlap=1.5))
