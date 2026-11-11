"""Pose decoders: decoded output, raw heads, letterbox, DFL and fix_point (P-10)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.adapters.pose.dpu import (
    dequantize_fix_point,
    input_quantization_lut,
    order_outputs_by_stride,
    quantize_fix_point,
    quantize_rgb_lut,
)
from cpsa.vision.pose_decode import (
    QuantizedRawHeadDecoder,
    decode_raw_heads,
    decode_ultralytics,
    letterbox,
    unletterbox,
)
from cpsa.vision.types import PersonPose

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "pose" / "yolov8_pose_raw_golden.npz"


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _decoded_tensor(
    candidates: list[tuple[tuple[float, float, float, float], float, NDArray[np.float64], float]],
    n_anchors: int,
    n_kpt: int,
    n_class: int = 1,
    class_index: int = 0,
) -> NDArray[np.float64]:
    """Encode persons into an Ultralytics decoded ``(1, 4+nc+3K, N)`` tensor.

    ``candidates`` holds ``(box_xyxy, score, xy, conf)`` per person.
    """

    output = np.zeros((1, 4 + n_class + 3 * n_kpt, n_anchors), dtype=np.float64)
    for anchor, (box, score, xy, conf) in enumerate(candidates):
        x1, y1, x2, y2 = box
        output[0, 0:4, anchor] = (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            x2 - x1,
            y2 - y1,
        )
        output[0, 4 + class_index, anchor] = score
        conf_values = np.broadcast_to(np.asarray(conf, dtype=np.float64), (xy.shape[0],))
        keypoints = np.stack([xy[:, 0], xy[:, 1], conf_values], axis=1).reshape(-1)
        output[0, 4 + n_class :, anchor] = keypoints
    return output


def _encode_raw_candidate(
    height: int,
    width: int,
    stride: int,
    n_kpt: int,
    grid: tuple[int, int],
    distances: tuple[int, int, int, int],
    score: float,
    keypoints_xy: NDArray[np.float64],
    keypoints_conf: float,
    reg_max: int = 16,
    n_class: int = 1,
    class_index: int = 0,
) -> NDArray[np.float64]:
    """Build a raw NHWC head tensor with one exact synthetic candidate."""

    raw = np.zeros((1, height, width, 4 * reg_max + n_class + 3 * n_kpt), dtype=np.float64)
    grid_x, grid_y = grid
    for side, distance in enumerate(distances):
        raw[0, grid_y, grid_x, side * reg_max + distance] = 50.0
    raw[0, grid_y, grid_x, 4 * reg_max + class_index] = _logit(score)
    for keypoint, (x, y) in enumerate(keypoints_xy):
        base = 4 * reg_max + n_class + 3 * keypoint
        raw[0, grid_y, grid_x, base] = (x / stride - grid_x) / 2.0
        raw[0, grid_y, grid_x, base + 1] = (y / stride - grid_y) / 2.0
        raw[0, grid_y, grid_x, base + 2] = _logit(keypoints_conf)
    return raw


def _expected_raw_box(
    stride: int,
    grid: tuple[int, int],
    distances: tuple[int, int, int, int],
) -> NDArray[np.float64]:
    left, top, right, bottom = (float(value) * stride for value in distances)
    anchor_x = (grid[0] + 0.5) * stride
    anchor_y = (grid[1] + 0.5) * stride
    return np.array(
        [anchor_x - left, anchor_y - top, anchor_x + right, anchor_y + bottom],
        dtype=np.float64,
    )


def test_decode_ultralytics_round_trip_within_one_pixel() -> None:
    n_kpt = 21
    keypoints = np.random.default_rng(20261109).uniform(100.0, 180.0, size=(n_kpt, 2))
    box = (100.0, 120.0, 180.0, 240.0)
    output = _decoded_tensor([(box, 0.9, keypoints, 0.8)], 4, n_kpt)

    persons = decode_ultralytics(output, n_kpt, conf_thr=0.5, iou_thr=0.5)

    assert len(persons) == 1
    person = persons[0]
    assert np.allclose(person.box, box, atol=1e-6)
    assert np.allclose(person.xy, keypoints, atol=1e-6)
    assert np.allclose(person.conf, 0.8)
    assert person.score == pytest.approx(0.9)


def test_decode_ultralytics_orders_by_score_and_thresholds() -> None:
    n_kpt = 2
    keypoints = np.zeros((n_kpt, 2))
    low = ((0.0, 0.0, 10.0, 10.0), 0.4, keypoints, 0.5)
    high = ((100.0, 100.0, 120.0, 130.0), 0.8, keypoints, 0.5)
    output = _decoded_tensor([low, high], 4, n_kpt)

    persons = decode_ultralytics(output, n_kpt, conf_thr=0.5, iou_thr=0.5)
    assert [person.score for person in persons] == pytest.approx([0.8])

    persons = decode_ultralytics(output, n_kpt, conf_thr=0.1, iou_thr=0.5)
    assert [person.score for person in persons] == pytest.approx([0.8, 0.4])


def test_decode_ultralytics_nms_is_class_aware() -> None:
    n_kpt = 2
    keypoints = np.zeros((n_kpt, 2))
    first = ((0.0, 0.0, 20.0, 20.0), 0.9, keypoints, 0.5)
    overlapping = ((2.0, 2.0, 22.0, 22.0), 0.8, keypoints, 0.5)
    tensor = _decoded_tensor([first, overlapping], 4, n_kpt, n_class=2, class_index=0)

    assert len(decode_ultralytics(tensor, n_kpt, 0.25, 0.5)) == 1

    tensor[0, 5, 1] = tensor[0, 4, 1]
    tensor[0, 4, 1] = 0.0
    assert len(decode_ultralytics(tensor, n_kpt, 0.25, 0.5)) == 2


@pytest.mark.parametrize("n_kpt", [17, 62])
def test_raw_heads_round_trip_within_one_pixel(n_kpt: int) -> None:
    stride = 16
    height, width = 6, 6
    grid = (2, 3)
    distances = (2, 1, 3, 4)
    rng = np.random.default_rng(20261109)
    keypoints = rng.uniform(10.0, 90.0, size=(n_kpt, 2))
    raw = _encode_raw_candidate(
        height,
        width,
        stride,
        n_kpt,
        grid,
        distances,
        score=0.9,
        keypoints_xy=keypoints,
        keypoints_conf=0.7,
    )

    persons = decode_raw_heads([raw], [stride], n_kpt, conf_thr=0.5, iou_thr=0.5)

    assert len(persons) == 1
    person = persons[0]
    assert np.allclose(person.box, _expected_raw_box(stride, grid, distances), atol=1e-6)
    assert np.allclose(person.xy, keypoints, atol=1e-6)
    assert np.allclose(person.conf, 0.7, atol=1e-6)
    assert person.score == pytest.approx(0.9)


def test_raw_heads_use_every_scale_and_stride_order() -> None:
    n_kpt = 3
    keypoints = np.array([[10.0, 12.0], [20.0, 22.0], [30.0, 32.0]])
    first = np.zeros((1, 12, 12, 4 * 16 + 1 + 3 * n_kpt))
    second = _encode_raw_candidate(6, 6, 16, n_kpt, (1, 2), (1, 1, 1, 1), 0.8, keypoints, 0.9)
    third = _encode_raw_candidate(3, 3, 32, n_kpt, (0, 1), (1, 2, 1, 2), 0.7, keypoints, 0.9)

    persons = decode_raw_heads(
        [third, first, second], [32, 8, 16], n_kpt, conf_thr=0.5, iou_thr=0.5
    )

    assert len(persons) == 2
    assert [person.score for person in persons] == pytest.approx([0.8, 0.7])
    assert np.allclose(persons[0].box, _expected_raw_box(16, (1, 2), (1, 1, 1, 1)))


def test_golden_fixture_pins_the_upstream_version() -> None:
    fixture = np.load(GOLDEN)
    assert str(fixture["version"]) == "8.4.203"
    assert int(fixture["seed"]) == 20261109


@pytest.mark.parametrize("n_kpt", [17, 62])
def test_raw_heads_match_the_ultralytics_golden(n_kpt: int) -> None:
    fixture = np.load(GOLDEN)
    prefix = f"kpt{n_kpt}_"
    strides = [int(value) for value in fixture[f"{prefix}strides"]]
    raw = [fixture[f"{prefix}raw_{index}"] for index in range(len(strides))]
    reg_max = int(fixture[f"{prefix}reg_max"])
    expected = fixture[f"{prefix}decoded"]
    top = int(fixture[f"{prefix}top"])
    conf_thr = float(fixture[f"{prefix}conf_thr"])
    iou_thr = float(fixture[f"{prefix}iou_thr"])

    persons = decode_raw_heads(raw, strides, n_kpt, conf_thr, iou_thr, reg_max)

    assert len(persons) == 1
    _assert_matches_golden(persons[0], expected, top, n_kpt)


@pytest.mark.parametrize("n_kpt", [17, 62])
def test_decode_ultralytics_matches_the_upstream_golden(n_kpt: int) -> None:
    fixture = np.load(GOLDEN)
    prefix = f"kpt{n_kpt}_"
    decoded = fixture[f"{prefix}decoded"]
    top = int(fixture[f"{prefix}top"])
    conf_thr = float(fixture[f"{prefix}conf_thr"])
    iou_thr = float(fixture[f"{prefix}iou_thr"])

    persons = decode_ultralytics(decoded, n_kpt, conf_thr, iou_thr)

    assert len(persons) == 1
    _assert_matches_golden(persons[0], decoded, top, n_kpt)


def _assert_matches_golden(
    person: PersonPose,
    decoded: NDArray[np.float32],
    anchor: int,
    n_kpt: int,
) -> None:
    """Compare against the upstream decoded column (xywh boxes, x/y/conf keypoints)."""

    expected = decoded[0, :, anchor].astype(np.float64)
    n_class = expected.size - 4 - 3 * n_kpt
    cx, cy, width, height = expected[:4]
    expected_box = np.array(
        [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]
    )
    expected_kpts = expected[4 + n_class :].reshape(n_kpt, 3)
    assert np.allclose(person.box, expected_box, atol=1e-3)
    assert np.allclose(person.xy, expected_kpts[:, :2], atol=1e-3)
    assert np.allclose(person.conf, expected_kpts[:, 2], atol=1e-3)
    assert person.score == pytest.approx(float(expected[4 : 4 + n_class].max()), abs=1e-6)


def test_raw_heads_reject_bad_shapes() -> None:
    valid = np.zeros((1, 4, 4, 4 * 16 + 1 + 3 * 2))
    with pytest.raises(ValueError, match="strides"):
        decode_raw_heads([valid], [8, 16], 2, 0.5, 0.5)
    with pytest.raises(ValueError, match="must not be empty"):
        decode_raw_heads([], [], 2, 0.5, 0.5)
    with pytest.raises(ValueError, match="strides"):
        decode_raw_heads([valid], [0], 2, 0.5, 0.5)
    with pytest.raises(ValueError, match="channels"):
        decode_raw_heads([valid], [8], 3, 0.5, 0.5)
    with pytest.raises(ValueError, match=r"\(1, H, W, C\)"):
        decode_raw_heads([np.zeros((4, 4))], [8], 2, 0.5, 0.5)
    with pytest.raises(ValueError, match="batch size"):
        decode_raw_heads([np.zeros((2, 4, 4, 100))], [8], 2, 0.5, 0.5)
    with pytest.raises(ValueError, match="n_kpt"):
        decode_raw_heads([valid], [8], 0, 0.5, 0.5)
    with pytest.raises(ValueError, match="reg_max"):
        decode_raw_heads([valid], [8], 2, 0.5, 0.5, reg_max=0)


def test_decoded_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="channels"):
        decode_ultralytics(np.zeros((1, 10, 4)), 2, 0.5, 0.5)
    with pytest.raises(ValueError, match="batch size"):
        decode_ultralytics(np.zeros((2, 11, 4)), 2, 0.5, 0.5)
    with pytest.raises(ValueError, match="n_kpt"):
        decode_ultralytics(np.zeros((1, 11, 4)), 0, 0.5, 0.5)


def test_letterbox_pads_and_preserves_aspect() -> None:
    image = np.full((90, 160, 3), 80, dtype=np.uint8)

    letterboxed, scale, pad = letterbox(image, (240, 320))

    assert letterboxed.shape == (240, 320, 3)
    assert letterboxed.dtype == np.uint8
    assert scale == pytest.approx(2.0)
    assert pad == pytest.approx((0.0, 30.0))
    assert np.all(letterboxed[:30] == 114)
    assert np.all(letterboxed[30:210] == 80)
    assert np.all(letterboxed[210:] == 114)


def test_letterbox_pads_both_axes() -> None:
    image = np.full((100, 50, 3), 200, dtype=np.uint8)

    letterboxed, scale, pad = letterbox(image, (100, 200))

    assert letterboxed.shape == (100, 200, 3)
    assert scale == pytest.approx(1.0)
    assert pad == pytest.approx((75.0, 0.0))
    assert np.all(letterboxed[:, :75] == 114)
    assert np.all(letterboxed[:, 75:125] == 200)
    assert np.all(letterboxed[:, 125:] == 114)


def test_letterbox_bilinear_interpolation() -> None:
    row = [[0, 0, 0], [255, 255, 255]]
    image = np.array([row, row], dtype=np.uint8)

    letterboxed, scale, pad = letterbox(image, (3, 3))

    assert scale == pytest.approx(1.5)
    assert pad == pytest.approx((0.0, 0.0))
    assert letterboxed.shape == (3, 3, 3)
    for line in letterboxed:
        assert line[:, 0].tolist() == [0, 128, 255]


def test_letterbox_unletterbox_round_trip() -> None:
    image = np.zeros((90, 160, 3), dtype=np.uint8)
    letterboxed, scale, pad = letterbox(image, (240, 320))
    assert letterboxed.shape == (240, 320, 3)
    box = np.array([10.0, 20.0, 100.0, 140.0])
    xy = np.array([[15.0, 25.0], [90.0, 130.0]])
    conf = np.array([0.9, 0.8])
    letterboxed_person = PersonPose(
        box=np.array(
            [
                box[0] * scale + pad[0],
                box[1] * scale + pad[1],
                box[2] * scale + pad[0],
                box[3] * scale + pad[1],
            ]
        ),
        score=0.75,
        xy=xy * scale + np.asarray(pad),
        conf=conf.copy(),
    )

    restored = unletterbox([letterboxed_person], scale, pad)

    assert len(restored) == 1
    assert np.allclose(restored[0].box, box)
    assert np.allclose(restored[0].xy, xy)
    assert np.allclose(restored[0].conf, conf)
    assert restored[0].score == pytest.approx(0.75)
    # The originals are not aliased or mutated.
    assert not np.shares_memory(restored[0].box, letterboxed_person.box)


def test_letterbox_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        letterbox(np.zeros((4, 4)), (8, 8))
    with pytest.raises(ValueError, match="positive"):
        letterbox(np.zeros((4, 4, 3), dtype=np.uint8), (0, 8))
    with pytest.raises(ValueError, match="pad_value"):
        letterbox(np.zeros((4, 4, 3), dtype=np.uint8), (8, 8), pad_value=300)
    with pytest.raises(ValueError, match="scale"):
        unletterbox([], 0.0, (0.0, 0.0))


def test_letterbox_uses_the_injected_resize() -> None:
    image = np.full((90, 160, 3), 80, dtype=np.uint8)
    calls: list[tuple[tuple[int, ...], int, int]] = []

    def resize(rgb: NDArray[np.uint8], out_height: int, out_width: int) -> NDArray[np.uint8]:
        calls.append((rgb.shape, out_height, out_width))
        return np.full((out_height, out_width, 3), 7, dtype=np.uint8)

    letterboxed, scale, pad = letterbox(image, (240, 320), resize=resize)

    assert calls == [((90, 160, 3), 180, 320)]
    assert scale == pytest.approx(2.0)
    assert pad == pytest.approx((0.0, 30.0))
    assert np.all(letterboxed[:30] == 114)
    assert np.all(letterboxed[30:210] == 7)
    assert np.all(letterboxed[210:] == 114)


def test_numpy_and_opencv_resize_agree_within_one_lsb() -> None:
    pytest.importorskip("cv2")
    from cpsa.adapters.pose.resize import LETTERBOX_RESIZE

    assert LETTERBOX_RESIZE is not None
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    with_cv2, cv2_scale, cv2_pad = letterbox(image, (384, 512), resize=LETTERBOX_RESIZE)
    with_numpy, numpy_scale, numpy_pad = letterbox(image, (384, 512))

    assert with_cv2.shape == with_numpy.shape == (384, 512, 3)
    assert (cv2_scale, cv2_pad) == (numpy_scale, numpy_pad)
    difference = np.abs(with_cv2.astype(np.int16) - with_numpy.astype(np.int16))
    assert int(difference.max()) <= 1


def test_quantize_fix_point_scales_and_saturates() -> None:
    values = np.array([0.0, 1.0, -1.0, 0.5, -0.5])

    assert quantize_fix_point(values, 7).tolist() == [0, 127, -128, 64, -64]
    assert quantize_fix_point(np.array([0.4, -0.4, 1.6]), 0).tolist() == [0, 0, 2]
    assert quantize_fix_point(np.array([0.25, -0.0625]), 4).tolist() == [4, -1]
    assert quantize_fix_point(np.array([1e9, -1e9]), 0).tolist() == [127, -128]
    with pytest.raises(ValueError, match="non-finite"):
        quantize_fix_point(np.array([np.nan]), 4)


def test_dequantize_fix_point_and_round_trip() -> None:
    quantized = np.array([-128, -1, 0, 1, 127], dtype=np.int8)

    restored = dequantize_fix_point(quantized, 7)

    assert np.allclose(restored, quantized.astype(np.float64) / 128.0)
    assert quantize_fix_point(restored, 7).tolist() == quantized.tolist()
    full = np.arange(-128, 128, dtype=np.int8)
    assert quantize_fix_point(dequantize_fix_point(full, 4), 4).tolist() == full.tolist()


def test_order_outputs_by_stride() -> None:
    fine = np.full((1, 12, 12, 4), 8.0)
    medium = np.full((1, 6, 6, 4), 16.0)
    coarse = np.full((1, 3, 3, 4), 32.0)

    ordered = order_outputs_by_stride([coarse, fine, medium], [32, 8, 16])

    assert [stride for stride, _ in ordered] == [8, 16, 32]
    assert [int(output[0, 0, 0, 0]) for _, output in ordered] == [8, 16, 32]
    with pytest.raises(ValueError, match="strides"):
        order_outputs_by_stride([fine], [8, 16])


def _raw_shapes(n_kpt: int, n_class: int) -> list[tuple[int, int, int]]:
    channels = 4 * 16 + n_class + 3 * n_kpt
    return [(12, 16, channels), (6, 8, channels), (3, 4, channels)]


def _reference_persons(
    outputs: list[NDArray[np.int8]],
    strides: list[int],
    fix_points: list[int],
    n_kpt: int,
    conf_thr: float,
    iou_thr: float,
) -> list[PersonPose]:
    """Run the reference path: dequantize every tensor, then decode."""

    dequantized = [
        dequantize_fix_point(output, fix_point)
        for output, fix_point in zip(outputs, fix_points, strict=True)
    ]
    return decode_raw_heads(dequantized, strides, n_kpt, conf_thr, iou_thr)


def _assert_same_persons(fast: list[PersonPose], reference: list[PersonPose]) -> None:
    assert len(fast) == len(reference)
    for actual, expected in zip(fast, reference, strict=True):
        assert np.allclose(actual.box, expected.box, atol=1e-4, rtol=0.0)
        assert actual.score == pytest.approx(expected.score, abs=1e-6)
        assert np.allclose(actual.xy, expected.xy, atol=1e-4, rtol=0.0)
        assert np.allclose(actual.conf, expected.conf, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("fix_point", [-3, 0, 4, 7])
@pytest.mark.parametrize("n_class", [1, 3])
def test_quantized_decoder_matches_the_reference_on_random_outputs(
    fix_point: int,
    n_class: int,
) -> None:
    rng = np.random.default_rng(20261110)
    n_kpt = 4
    strides = [8, 16, 32]
    shapes = _raw_shapes(n_kpt, n_class)
    outputs = [
        rng.integers(-128, 127, size=shape, endpoint=True, dtype=np.int8) for shape in shapes
    ]
    fix_points = [fix_point] * len(shapes)
    decoder = QuantizedRawHeadDecoder(shapes, strides, fix_points, n_kpt)

    fast = decoder.decode(outputs, conf_thr=0.25, iou_thr=0.45)
    reference = _reference_persons(outputs, strides, fix_points, n_kpt, 0.25, 0.45)

    assert len(reference) > 0
    assert decoder.last_candidate_count >= len(reference)
    _assert_same_persons(fast, reference)
    _assert_same_persons(
        decoder.decode([output[np.newaxis] for output in outputs], 0.25, 0.45),
        reference,
    )


@pytest.mark.parametrize("fix_point", [-3, 0, 4, 7])
def test_quantized_decoder_matches_when_every_candidate_is_below_threshold(
    fix_point: int,
) -> None:
    rng = np.random.default_rng(20261110)
    n_kpt = 3
    n_class = 1
    strides = [8, 16, 32]
    shapes = _raw_shapes(n_kpt, n_class)
    outputs = [
        rng.integers(-128, 127, size=shape, endpoint=True, dtype=np.int8) for shape in shapes
    ]
    class_start = 4 * 16
    for output in outputs:
        output[..., class_start : class_start + n_class] = -128
    fix_points = [fix_point] * len(shapes)
    decoder = QuantizedRawHeadDecoder(shapes, strides, fix_points, n_kpt)

    fast = decoder.decode(outputs, conf_thr=0.5, iou_thr=0.45)
    reference = _reference_persons(outputs, strides, fix_points, n_kpt, 0.5, 0.45)

    assert fast == []
    assert reference == []
    assert decoder.last_candidate_count == 0


@pytest.mark.parametrize("n_kpt", [17, 62])
@pytest.mark.parametrize("fix_point", [0, 4])
def test_quantized_decoder_matches_the_golden_with_many_survivors(
    n_kpt: int,
    fix_point: int,
) -> None:
    fixture = np.load(GOLDEN)
    prefix = f"kpt{n_kpt}_"
    strides = [int(value) for value in fixture[f"{prefix}strides"]]
    reg_max = int(fixture[f"{prefix}reg_max"])
    conf_thr = float(fixture[f"{prefix}conf_thr"])
    iou_thr = float(fixture[f"{prefix}iou_thr"])
    outputs = [
        quantize_fix_point(fixture[f"{prefix}raw_{index}"], fix_point)[0]
        for index in range(len(strides))
    ]
    for output in outputs:
        output[..., 4 * reg_max] = 127
    shapes = [
        (int(output.shape[0]), int(output.shape[1]), int(output.shape[2])) for output in outputs
    ]
    fix_points = [fix_point] * len(strides)
    decoder = QuantizedRawHeadDecoder(shapes, strides, fix_points, n_kpt, reg_max)

    fast = decoder.decode(outputs, conf_thr, iou_thr)
    reference = _reference_persons(outputs, strides, fix_points, n_kpt, conf_thr, iou_thr)

    assert decoder.last_candidate_count == sum(shape[0] * shape[1] for shape in shapes)
    assert len(reference) > 0
    _assert_same_persons(fast, reference)


@pytest.mark.parametrize("n_kpt", [17, 62])
@pytest.mark.parametrize("fix_point", [0, 4])
def test_quantized_decoder_matches_the_golden_threshold_path(
    n_kpt: int,
    fix_point: int,
) -> None:
    fixture = np.load(GOLDEN)
    prefix = f"kpt{n_kpt}_"
    strides = [int(value) for value in fixture[f"{prefix}strides"]]
    reg_max = int(fixture[f"{prefix}reg_max"])
    conf_thr = float(fixture[f"{prefix}conf_thr"])
    iou_thr = float(fixture[f"{prefix}iou_thr"])
    outputs = [
        quantize_fix_point(fixture[f"{prefix}raw_{index}"], fix_point)[0]
        for index in range(len(strides))
    ]
    shapes = [
        (int(output.shape[0]), int(output.shape[1]), int(output.shape[2])) for output in outputs
    ]
    fix_points = [fix_point] * len(strides)
    decoder = QuantizedRawHeadDecoder(shapes, strides, fix_points, n_kpt, reg_max)

    fast = decoder.decode(outputs, conf_thr, iou_thr)
    reference = _reference_persons(outputs, strides, fix_points, n_kpt, conf_thr, iou_thr)

    _assert_same_persons(fast, reference)


def test_quantized_decoder_handles_a_threshold_on_a_quantization_step() -> None:
    n_kpt = 2
    n_class = 1
    channel_count = 4 * 16 + n_class + 3 * n_kpt
    shape = (4, 4, channel_count)
    raw = np.full(shape, -128, dtype=np.int8)
    raw[..., 4 * 16 + n_class :] = -1
    raw.reshape(-1, channel_count)[5, 4 * 16] = -19
    raw.reshape(-1, channel_count)[5, : 4 * 16] = 0
    outputs = [raw]
    fix_points = [4]
    decoder = QuantizedRawHeadDecoder([shape], [8], fix_points, n_kpt)

    step_score = float(1.0 / (1.0 + np.exp(19.0 / 16.0)))
    just_below = float(np.nextafter(step_score, 0.0))

    assert _reference_persons(outputs, [8], fix_points, n_kpt, step_score, 0.45) == []
    assert len(_reference_persons(outputs, [8], fix_points, n_kpt, just_below, 0.45)) == 1
    for conf_thr in (step_score, just_below):
        fast = decoder.decode(outputs, conf_thr, 0.45)
        reference = _reference_persons(outputs, [8], fix_points, n_kpt, conf_thr, 0.45)
        _assert_same_persons(fast, reference)


def test_input_quantization_lut_reproduces_the_reference_formula() -> None:
    image = np.random.default_rng(20261110).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)

    for fix_point in (-3, 0, 4, 7):
        lut = input_quantization_lut(fix_point)
        assert lut.shape == (256,)
        assert lut.dtype == np.int8
        expected = quantize_fix_point(image.astype(np.float32) / 255.0, fix_point)
        assert np.array_equal(lut[image], expected)
        every_value = np.arange(256, dtype=np.uint8)
        assert np.array_equal(
            lut[every_value],
            quantize_fix_point(every_value.astype(np.float32) / 255.0, fix_point),
        )


def test_quantize_rgb_lut_matches_the_reference_formula() -> None:
    image = np.random.default_rng(20261110).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    buffer = np.empty(image.size, dtype=np.int8)

    for fix_point in (-3, 0, 4, 7):
        lut = input_quantization_lut(fix_point)
        quantized = quantize_rgb_lut(image, lut, buffer)
        assert quantized.shape == image.shape
        assert np.shares_memory(quantized, buffer)
        assert np.array_equal(
            quantized,
            quantize_fix_point(image.astype(np.float32) / 255.0, fix_point),
        )

    with pytest.raises(ValueError, match="uint8"):
        quantize_rgb_lut(image.astype(np.float32), lut, buffer)
    with pytest.raises(ValueError, match="256-element"):
        quantize_rgb_lut(image, np.zeros(255, dtype=np.int8), buffer)
    with pytest.raises(ValueError, match="out must hold"):
        quantize_rgb_lut(image, lut, np.empty(image.size - 1, dtype=np.int8))


def test_letterbox_is_a_no_op_for_the_model_input_size() -> None:
    image = np.random.default_rng(20261110).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)

    letterboxed, scale, pad = letterbox(image, (480, 640))

    assert letterboxed is image
    assert scale == 1.0
    assert pad == (0.0, 0.0)


def test_quantized_decoder_rejects_bad_configuration_and_inputs() -> None:
    shape = (4, 4, 4 * 16 + 1 + 3 * 2)
    with pytest.raises(ValueError, match="n_kpt"):
        QuantizedRawHeadDecoder([shape], [8], [0], 0)
    with pytest.raises(ValueError, match="reg_max"):
        QuantizedRawHeadDecoder([shape], [8], [0], 2, reg_max=0)
    with pytest.raises(ValueError, match="must not be empty"):
        QuantizedRawHeadDecoder([], [], [], 2)
    with pytest.raises(ValueError, match="fix points"):
        QuantizedRawHeadDecoder([shape], [8], [0, 0], 2)
    with pytest.raises(ValueError, match=r"\(H, W, C\)"):
        QuantizedRawHeadDecoder([(4, 4)], [8], [0], 2)
    with pytest.raises(ValueError, match="positive"):
        QuantizedRawHeadDecoder([(0, 4, shape[2])], [8], [0], 2)
    with pytest.raises(ValueError, match="strides"):
        QuantizedRawHeadDecoder([shape], [0], [0], 2)
    with pytest.raises(ValueError, match="channels"):
        QuantizedRawHeadDecoder([(4, 4, 4 * 16 + 3 * 2)], [8], [0], 2)

    decoder = QuantizedRawHeadDecoder([shape], [8], [4], 2)
    valid = np.zeros(shape, dtype=np.int8)
    with pytest.raises(ValueError, match="expects"):
        decoder.decode([valid, valid], 0.25, 0.45)
    with pytest.raises(ValueError, match="int8"):
        decoder.decode([valid.astype(np.float64)], 0.25, 0.45)
    with pytest.raises(ValueError, match="must be"):
        decoder.decode([np.zeros((8, 8, shape[2]), dtype=np.int8)], 0.25, 0.45)
