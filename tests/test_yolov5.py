"""YOLOv5-nano (Y5): decode, int8 prefilter, preprocessing and the DPU detector."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.dpu import input_quantization_lut_scaled
from cpsa.yolov5 import (
    MAX_CLASSES_PER_CELL,
    YOLOV5_ANCHORS,
    YOLOV5_INPUT_SCALE,
    YOLOV5_INPUT_SIZE,
    YOLOV5_PAD_VALUE,
    QuantizedYolov5Decoder,
    Yolov5Detector,
    class_thresholds,
    decode_yolov5,
    detections_from_heads,
    prepare_input,
    prepare_input_into,
)
from cpsa.yolox import _SLOT_CANDIDATE_CAP

STRIDES = (8, 16, 32)
BACKGROUND = -1000.0
#: The booth scene: a person and the bottle held in front of it.
PERSON = 0
BOTTLE = 39

#: Shapes and per-head fix points of the compiled YOLOv5-nano graph.
QUANTIZED_SHAPES = ((80, 80, 255), (40, 40, 255), (20, 20, 255))
QUANTIZED_FIX_POINTS = (3, 4, 4)

#: One head slot: ``[tx, ty, tw, th, objectness, 80 classes]``.
SLOT = 85
OBJECTNESS = 4
CLASSES = 5


def _heads(
    sizes: tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (2, 2)),
    classes: int = 80,
    background: float = BACKGROUND,
) -> list[NDArray[np.float64]]:
    """Background heads: objectness and class values far below any threshold."""

    heads = []
    for height, width in sizes:
        head = np.zeros((height, width, 3 * (5 + classes)), dtype=np.float64)
        for anchor in range(3):
            head[..., anchor * SLOT + OBJECTNESS] = background
            head[..., anchor * SLOT + CLASSES : (anchor + 1) * SLOT] = background
        heads.append(head)
    return heads


def _set(
    head: NDArray[np.float64],
    row: int,
    column: int,
    *,
    anchor: int = 0,
    tx: float = 0.0,
    ty: float = 0.0,
    log_w: float = 0.0,
    log_h: float = 0.0,
    objectness: float = 20.0,
    class_id: int = 0,
    class_logit: float = 20.0,
    background: float = BACKGROUND,
) -> None:
    """Write one candidate into one anchor slot of a cell (slots are anchor-major)."""

    base = anchor * SLOT
    head[row, column, base] = tx
    head[row, column, base + 1] = ty
    head[row, column, base + 2] = log_w
    head[row, column, base + 3] = log_h
    head[row, column, base + OBJECTNESS] = objectness
    head[row, column, base + CLASSES : (anchor + 1) * SLOT] = background
    head[row, column, base + CLASSES + class_id] = class_logit


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


# -- the decode (the zoo post-processing, ``type: YOLOV5``) ------------------- #


def test_grid_offset_exp_and_anchor_decode() -> None:
    heads = _heads()
    # All logits zero: sigmoid = 0.5, so centre = (0.5 * 2 - 0.5 + cell) * stride
    # and the size is exactly the anchor (``(2 * 0.5) ** 2 == 1``).
    _set(heads[0], 0, 0, anchor=0)
    _set(heads[0], 0, 1, anchor=1)
    _set(heads[2], 0, 0, anchor=2)

    boxes, scores, classes = decode_yolov5(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)

    assert len(scores) == 3
    assert scores == pytest.approx([1.0, 1.0, 1.0])
    assert classes.tolist() == [0, 0, 0]
    # stride 8, cell (0, 0), anchor (10, 13): centre (4, 4).
    assert boxes[0] == pytest.approx([-1.0, -2.5, 9.0, 10.5])
    # stride 8, cell (0, 1), anchor (16, 30): centre (12, 4).
    assert boxes[1] == pytest.approx([4.0, -11.0, 20.0, 19.0])
    # stride 32, cell (0, 0), anchor (373, 326): centre (16, 16).
    assert boxes[2] == pytest.approx([-170.5, -147.0, 202.5, 179.0])


def test_slots_are_anchor_major_not_channel_planes() -> None:
    """Slot ``a`` starts at channel ``85 * a``: a head is not three 85-plane stacks."""

    heads = _heads()
    _set(heads[0], 1, 1, anchor=2, tx=0.0, ty=0.0)

    boxes, _scores, _classes = decode_yolov5(
        heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True
    )

    # Anchor slot 2 of stride 8 is (33, 23); the box is that anchor at cell (1, 1).
    assert boxes[0] == pytest.approx([12.0 - 16.5, 12.0 - 11.5, 12.0 + 16.5, 12.0 + 11.5])


def test_score_is_objectness_times_class() -> None:
    heads = _heads()
    _set(
        heads[0],
        0,
        0,
        anchor=1,
        objectness=_logit(0.8),
        class_id=3,
        class_logit=_logit(0.9),
    )
    heads[0][0, 0, SLOT + CLASSES] = _logit(0.5)

    _boxes, scores, classes = decode_yolov5(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)

    assert scores == pytest.approx([0.8 * 0.9])
    assert classes.tolist() == [3]


def test_logits_false_uses_probabilities_as_is() -> None:
    heads = _heads(background=0.0)
    _set(
        heads[0],
        0,
        0,
        anchor=0,
        tx=0.5,
        ty=0.5,
        log_w=0.5,
        log_h=0.5,
        objectness=0.8,
        class_id=2,
        class_logit=0.9,
        background=0.0,
    )

    _boxes, scores, classes = decode_yolov5(
        heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=False
    )

    assert scores == pytest.approx([0.8 * 0.9])
    assert classes.tolist() == [2]


def test_the_conf_threshold_is_strict() -> None:
    heads = _heads()
    _set(heads[0], 0, 0, objectness=_logit(0.5), class_id=0, class_logit=20.0)

    _boxes, scores, _classes = decode_yolov5(
        heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True
    )

    assert scores.size == 0


def test_class_aware_nms_suppresses_duplicates_of_one_class_only() -> None:
    heads = _heads()
    # Stride 32's cells are 32 px apart and its first anchor is 116x90, so these
    # two person boxes overlap well above the 0.45 IoU threshold.
    _set(heads[2], 0, 0, class_id=PERSON)
    _set(heads[2], 0, 1, class_id=PERSON)
    _set(heads[2], 1, 1, class_id=BOTTLE)

    _boxes, scores, classes = decode_yolov5(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)

    assert scores.tolist() == pytest.approx([1.0, 1.0])
    assert sorted(classes.tolist()) == [PERSON, BOTTLE]


def test_a_bottle_sharing_a_persons_slot_is_decoded_with_the_person() -> None:
    heads = _heads()
    _set(heads[2], 0, 0, anchor=0, class_id=PERSON)
    heads[2][0, 0, CLASSES + BOTTLE] = 20.0

    _boxes, _scores, classes = decode_yolov5(
        heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True
    )

    assert sorted(classes.tolist()) == [PERSON, BOTTLE]


def test_per_class_thresholds_keep_a_weak_bottle_in_the_persons_slot() -> None:
    heads = _heads()
    _set(heads[0], 0, 0, objectness=_logit(0.8), class_id=BOTTLE, class_logit=_logit(0.5))

    _boxes, scores, classes = decode_yolov5(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)
    assert scores.size == 0

    overrides = class_thresholds(0.5, class_conf={"bottle": 0.25})
    _boxes, scores, classes = decode_yolov5(
        heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True, class_thr=overrides
    )

    assert classes.tolist() == [BOTTLE]
    assert scores == pytest.approx([0.4])


def test_a_slot_yields_at_most_the_three_best_classes() -> None:
    heads = _heads()
    _set(heads[0], 0, 0, class_id=PERSON)
    for class_id in (BOTTLE, 41, 2, 3):
        heads[0][0, 0, CLASSES + class_id] = 20.0

    _boxes, _scores, classes = decode_yolov5(
        heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True
    )

    assert classes.size == MAX_CLASSES_PER_CELL


def test_three_slots_of_one_cell_are_decoded_independently() -> None:
    heads = _heads()
    for anchor in range(3):
        _set(heads[0], 0, 0, anchor=anchor, class_id=anchor)

    boxes, _scores, _classes = decode_yolov5(heads, STRIDES, conf_thr=0.5, iou_thr=0.0, logits=True)

    widths = sorted(box[2] - box[0] for box in boxes)
    assert widths == pytest.approx(sorted(YOLOV5_ANCHORS[8][anchor][0] for anchor in range(3)))


@pytest.mark.parametrize(
    ("heads", "strides", "message"),
    [
        (_heads(), (8, 16), "strides"),
        (_heads(), (8, 16, 24), "anchors"),
        ([_heads()[0][..., :20], _heads()[1], _heads()[2]], STRIDES, "anchor slots"),
    ],
)
def test_invalid_inputs_raise(
    heads: list[NDArray[np.float64]], strides: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_yolov5(heads, strides, conf_thr=0.5, iou_thr=0.45, logits=True)


def test_bad_thresholds_are_rejected() -> None:
    heads = _heads()
    with pytest.raises(ValueError, match="conf_thr"):
        decode_yolov5(heads, STRIDES, conf_thr=1.5, iou_thr=0.45, logits=True)
    with pytest.raises(ValueError, match="iou_thr"):
        decode_yolov5(heads, STRIDES, conf_thr=0.5, iou_thr=-0.1, logits=True)


# -- the int8, threshold-first decoder ---------------------------------------- #


def _quantized_heads(
    shapes: tuple[tuple[int, int, int], ...] = ((8, 8, 255), (4, 4, 255), (2, 2, 255)),
    seed: int = 7,
) -> list[NDArray[np.int8]]:
    """Uniform random int8 heads: every slot is a possible candidate."""

    generator = np.random.default_rng(seed)
    return [generator.integers(-128, 128, size=shape, dtype=np.int8) for shape in shapes]


def _dequantized(
    heads: list[NDArray[np.int8]], fix_points: tuple[int, ...]
) -> list[NDArray[np.float64]]:
    """The float heads :func:`decode_yolov5` sees after the DPU dequantizes."""

    return [
        head.astype(np.float64) * 2.0**-fix_point
        for head, fix_point in zip(heads, fix_points, strict=True)
    ]


def _planted_heads(
    shapes: tuple[tuple[int, int, int], ...] = ((8, 8, 255), (4, 4, 255), (2, 2, 255)),
    planted: int = 4,
    seed: int = 11,
) -> list[NDArray[np.int8]]:
    """Background at -128 plus a few planted slots, as a real frame looks."""

    generator = np.random.default_rng(seed)
    heads = []
    for height, width, channels in shapes:
        head = np.full((height, width, channels), -128, dtype=np.int8)
        for anchor in range(3):
            base = anchor * SLOT
            head[..., base : base + 4] = generator.integers(
                -40, 40, size=(height, width, 4), dtype=np.int8
            )
        for index in range(planted):
            row = int(generator.integers(0, height))
            column = int(generator.integers(0, width))
            base = (index % 3) * SLOT
            head[row, column, base + OBJECTNESS] = generator.integers(60, 121)
            head[row, column, base + CLASSES : base + SLOT] = generator.integers(-100, -19, size=80)
            head[row, column, base + CLASSES + PERSON] = generator.integers(70, 128)
            if index % 2 == 0:
                head[row, column, base + CLASSES + BOTTLE] = generator.integers(30, 90)
        heads.append(head)
    return heads


def _crowded_heads(planted: int = 150, seed: int = 7) -> list[NDArray[np.int8]]:
    """The compiled graph's heads with ``planted`` object slots in loose clusters.

    Eight slots share neighbouring cells, so boxes of one class overlap the way
    a crowd's do and the class-aware NMS has real work; half of the slots carry
    a bottle and every fifth a cup, the multi-label path.
    """

    generator = np.random.default_rng(seed)
    heads = [np.full(shape, -128, dtype=np.int8) for shape in QUANTIZED_SHAPES]
    for index in range(planted):
        # Two thirds of the crowd on the fine 80x80 head, like a real frame.
        which = 0 if index < 100 else (1 if index < 135 else 2)
        height, width, _ = QUANTIZED_SHAPES[which]
        head = heads[which]
        cluster = index // 8
        row = min((cluster * 37) % max(height - 4, 1) + int(generator.integers(0, 3)), height - 1)
        column = min((cluster * 53) % max(width - 4, 1) + int(generator.integers(0, 3)), width - 1)
        base = (index % 3) * SLOT
        head[row, column, base : base + 4] = generator.integers(-30, 30, size=4, dtype=np.int8)
        head[row, column, base + OBJECTNESS] = generator.integers(90, 127)
        head[row, column, base + CLASSES : base + SLOT] = generator.integers(-110, -30, size=80)
        head[row, column, base + CLASSES + PERSON] = generator.integers(90, 127)
        if index % 2 == 0:
            head[row, column, base + CLASSES + BOTTLE] = generator.integers(50, 110)
        if index % 5 == 0:
            head[row, column, base + CLASSES + 41] = generator.integers(40, 90)  # cup
    return heads


@pytest.mark.parametrize("conf_thr", [0.0, 0.05, 0.35, 0.9, 1.0])
def test_the_quantized_decoder_matches_the_float_decode(conf_thr: float) -> None:
    heads = _quantized_heads()
    decoder = QuantizedYolov5Decoder([head.shape for head in heads], STRIDES, QUANTIZED_FIX_POINTS)

    actual = decoder.decode(heads, conf_thr, 0.45)
    expected = decode_yolov5(
        _dequantized(heads, QUANTIZED_FIX_POINTS),
        STRIDES,
        conf_thr=conf_thr,
        iou_thr=0.45,
        logits=True,
    )

    for got, want in zip(actual, expected, strict=True):
        assert np.array_equal(got, want)


@pytest.mark.parametrize(
    "overrides",
    [
        class_thresholds(0.35, class_conf={"person": 0.40}, object_conf=0.25),
        {PERSON: 1.0, BOTTLE: 0.0},
        {PERSON: 0.9},
    ],
)
def test_the_quantized_decoder_matches_the_reference_with_per_class_thresholds(
    overrides: dict[int, float],
) -> None:
    # Small heads: the frame stays under the decode slot cap, so the float
    # reference sees exactly the slots the int8 path evaluates.
    heads = _planted_heads(shapes=((4, 4, 255), (2, 2, 255), (1, 1, 255)))
    decoder = QuantizedYolov5Decoder([head.shape for head in heads], STRIDES, QUANTIZED_FIX_POINTS)

    actual = decoder.decode(heads, 0.35, 0.45, overrides)
    expected = decode_yolov5(
        _dequantized(heads, QUANTIZED_FIX_POINTS),
        STRIDES,
        conf_thr=0.35,
        iou_thr=0.45,
        logits=True,
        class_thr=overrides,
    )

    for got, want in zip(actual, expected, strict=True):
        assert np.array_equal(got, want)


def test_the_int8_prefilter_drops_the_background_grid() -> None:
    heads = _planted_heads(planted=3)
    decoder = QuantizedYolov5Decoder([head.shape for head in heads], STRIDES, QUANTIZED_FIX_POINTS)

    detections = decoder.decode(heads, 0.35, 0.45)

    cells = sum(head.shape[0] * head.shape[1] for head in heads)
    assert detections[0].size > 0
    assert decoder.last_candidate_count < cells


def test_the_quantized_decoder_accepts_nhwc_outputs_with_the_batch_axis() -> None:
    heads = _planted_heads(planted=2, seed=5)
    decoder = QuantizedYolov5Decoder([head.shape for head in heads], STRIDES, QUANTIZED_FIX_POINTS)

    expected = decoder.decode(heads, 0.35, 0.45)
    actual = decoder.decode([head[None] for head in heads], 0.35, 0.45)

    for got, want in zip(actual, expected, strict=True):
        assert np.array_equal(got, want)


def test_the_quantized_decoder_rejects_float_heads() -> None:
    heads = [np.zeros(shape, dtype=np.float64) for shape in ((8, 8, 255), (4, 4, 255), (2, 2, 255))]
    decoder = QuantizedYolov5Decoder([head.shape for head in heads], STRIDES, QUANTIZED_FIX_POINTS)

    with pytest.raises(ValueError, match="int8"):
        decoder.decode(heads, 0.35, 0.45)


def test_the_quantized_decoder_rejects_a_mismatched_shape() -> None:
    decoder = QuantizedYolov5Decoder(
        [(8, 8, 255), (4, 4, 255), (2, 2, 255)], STRIDES, QUANTIZED_FIX_POINTS
    )
    heads = [np.full((8, 8, 255), -128, np.int8), np.full((4, 4, 85), -128, np.int8)]
    heads.append(np.full((2, 2, 255), -128, np.int8))

    with pytest.raises(ValueError, match="outputs\\[1\\]"):
        decoder.decode(heads, 0.35, 0.45)


# -- preprocessing (the zoo prototxt: RGB, mean 0, scale 1 / 255) ------------- #


def test_prepare_input_letterboxes_to_640_and_keeps_rgb_order() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., 0] = 10
    frame[..., 1] = 20
    frame[..., 2] = 30

    image, scale, pad = prepare_input(frame)

    assert image.shape == (*YOLOV5_INPUT_SIZE, 3)
    assert scale == pytest.approx(1.0)
    assert pad == pytest.approx((0.0, 80.0))
    # The pad band is the zoo's grey; the frame keeps its channel order, so the
    # LUT applies ``pixel / 255`` to R, G and B as they arrive.
    assert image[0, 0].tolist() == [YOLOV5_PAD_VALUE] * 3
    assert image[80, 0].tolist() == [10, 20, 30]


def test_prepare_input_into_writes_the_quantized_letterbox() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., :] = 255
    lut = input_quantization_lut_scaled(6, YOLOV5_INPUT_SCALE)
    view = np.empty((*YOLOV5_INPUT_SIZE, 3), dtype=np.int8)

    scale, pad = prepare_input_into(frame, YOLOV5_INPUT_SIZE, lut, view)

    assert scale == pytest.approx(1.0)
    assert pad == pytest.approx((0.0, 80.0))
    # 255 * (1 / 255) * 2**6 = 64, the only part of the range the graph uses.
    assert int(view[80, 0, 0]) == 64
    assert int(view[0, 0, 0]) == lut[YOLOV5_PAD_VALUE]


def test_prepare_input_into_skips_the_pad_only_when_asked() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lut = input_quantization_lut_scaled(6, YOLOV5_INPUT_SCALE)
    view = np.full((*YOLOV5_INPUT_SIZE, 3), 7, dtype=np.int8)

    prepare_input_into(frame, YOLOV5_INPUT_SIZE, lut, view, fill_pad=False)

    assert int(view[0, 0, 0]) == 7
    assert int(view[80, 0, 0]) == lut[0]


def test_a_640x480_frame_is_written_without_a_resize(monkeypatch: pytest.MonkeyPatch) -> None:
    """At scale 1.0 the source rows land in the graph input as they are.

    The letterbox of a 4:3 frame into 640x640 keeps every pixel, so resizing it
    would only copy 921 KB of memory a frame. The pad rows still change with the
    geometry, so they are quantized when the frame size changes.
    """

    import cv2

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., 0] = np.arange(640, dtype=np.uint8)[None, :]
    frame[..., 1] = np.arange(480, dtype=np.uint8)[:, None]
    lut = input_quantization_lut_scaled(6, YOLOV5_INPUT_SCALE)
    view = np.full((*YOLOV5_INPUT_SIZE, 3), 7, dtype=np.int8)
    expected = np.full((*YOLOV5_INPUT_SIZE, 3), lut[YOLOV5_PAD_VALUE], dtype=np.int8)
    cv2.LUT(frame, lut.view(np.uint8), dst=expected[80:560].view(np.uint8))

    def no_resize(*args: object, **kwargs: object) -> None:
        raise AssertionError("a 640x480 frame was resized")

    monkeypatch.setattr(cv2, "resize", no_resize)
    scale, pad = prepare_input_into(frame, YOLOV5_INPUT_SIZE, lut, view)

    assert scale == pytest.approx(1.0)
    assert pad == pytest.approx((0.0, 80.0))
    assert np.array_equal(view, expected)


def test_a_strided_frame_goes_through_the_general_letterbox() -> None:
    """A non-contiguous source cannot be quantized in place: it is letterboxed."""
    import cv2

    frame = np.zeros((480, 700, 3), dtype=np.uint8)
    frame[..., 0] = np.arange(700, dtype=np.uint8)[None, :]
    frame[..., 2] = 200
    source = frame[:, :640]  # a view: the row stride no longer matches the width
    assert not source.flags["C_CONTIGUOUS"]
    lut = input_quantization_lut_scaled(6, YOLOV5_INPUT_SCALE)
    view = np.empty((*YOLOV5_INPUT_SIZE, 3), dtype=np.int8)
    reference = np.empty((*YOLOV5_INPUT_SIZE, 3), dtype=np.int8)

    prepare_input_into(source, YOLOV5_INPUT_SIZE, lut, view)
    cv2.LUT(np.ascontiguousarray(source), lut.view(np.uint8), dst=reference[80:560].view(np.uint8))
    reference[:80] = lut[YOLOV5_PAD_VALUE]
    reference[560:] = lut[YOLOV5_PAD_VALUE]

    assert np.array_equal(view, reference)


def test_prepare_input_into_rejects_a_mismatched_view() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lut = input_quantization_lut_scaled(6, YOLOV5_INPUT_SCALE)

    with pytest.raises(ValueError, match="out must be"):
        prepare_input_into(frame, YOLOV5_INPUT_SIZE, lut, np.empty((640, 640), np.int8))


# -- source pixels and the float bridge -------------------------------------- #


def test_detections_from_heads_maps_to_source_pixels_and_round_trips_int8() -> None:
    heads = _planted_heads()
    quantized = [
        np.clip(np.rint(head.astype(np.float64) * 2.0**fix_point), -128, 127).astype(np.int8)
        for head, fix_point in zip(heads, QUANTIZED_FIX_POINTS, strict=True)
    ]

    staged = detections_from_heads(
        heads,
        STRIDES,
        conf_thr=0.35,
        iou_thr=0.45,
        scale=1.0,
        pad=(0.0, 80.0),
        source_hw=(480, 640),
        fix_points=QUANTIZED_FIX_POINTS,
    )
    direct = detections_from_heads(
        _dequantized(quantized, QUANTIZED_FIX_POINTS),
        STRIDES,
        conf_thr=0.35,
        iou_thr=0.45,
        scale=1.0,
        pad=(0.0, 80.0),
        source_hw=(480, 640),
    )

    assert staged == direct
    assert all(0.0 <= detection.box[0] <= detection.box[2] <= 640.0 for detection in staged)


# --------------------------------------------------------------------------- #
# Fake xir/vart, enough of the runtime surface for the detector.
# --------------------------------------------------------------------------- #


INPUT_DIMS = [1, 640, 640, 3]
OUTPUT_DIMS = {
    8: [1, 80, 80, 255],
    16: [1, 40, 40, 255],
    32: [1, 20, 20, 255],
}


class _Tensor:
    def __init__(self, name: str, dims: list[int], fix_point: int | None = None) -> None:
        self.name = name
        self.dims = dims
        self._attributes = {} if fix_point is None else {"fix_point": fix_point}

    def has_attr(self, name: str) -> bool:
        return name in self._attributes

    def get_attr(self, name: str) -> object:
        return self._attributes[name]


class _Subgraph:
    def __init__(
        self,
        name: str,
        device: str,
        inputs: list[_Tensor],
        outputs: list[_Tensor],
    ) -> None:
        self._name = name
        self._attributes: dict[str, object] = {"device": device}
        self._inputs = inputs
        self._outputs = outputs
        self.payload: list[NDArray[np.int8]] = []
        #: Optional per-execution payloads, to prove the detector's double buffer.
        self.payloads_by_execution: list[list[NDArray[np.int8]]] | None = None

    def get_name(self) -> str:
        return self._name

    def has_attr(self, name: str) -> bool:
        return name in self._attributes

    def get_attr(self, name: str) -> object:
        return self._attributes[name]

    def get_input_tensors(self) -> list[_Tensor]:
        return list(self._inputs)

    def get_output_tensors(self) -> list[_Tensor]:
        return list(self._outputs)


class _Root:
    def __init__(self, subgraphs: list[_Subgraph]) -> None:
        self._subgraphs = subgraphs

    def toposort_child_subgraph(self) -> list[_Subgraph]:
        return list(self._subgraphs)


class _Graph:
    pending: list[_Subgraph] = []

    def __init__(self, subgraphs: list[_Subgraph]) -> None:
        self._subgraphs = subgraphs

    @classmethod
    def deserialize(cls, path: str) -> _Graph:
        return cls(cls.pending)

    def get_root_subgraph(self) -> _Root:
        return _Root(self._subgraphs)


class _Runner:
    def __init__(self, subgraph: _Subgraph) -> None:
        self._subgraph = subgraph
        self.executions = 0

    def get_input_tensors(self) -> list[_Tensor]:
        return self._subgraph.get_input_tensors()

    def get_output_tensors(self) -> list[_Tensor]:
        return self._subgraph.get_output_tensors()

    def execute_async(
        self, inputs: list[NDArray[np.int8]], outputs: list[NDArray[np.int8]]
    ) -> tuple[int, int]:
        self.executions += 1
        sequence = self._subgraph.payloads_by_execution
        payload = (
            self._subgraph.payload
            if sequence is None
            else sequence[(self.executions - 1) % len(sequence)]
        )
        assert len(outputs) == len(payload)
        for buffer, data in zip(outputs, payload, strict=True):
            np.copyto(buffer, data)
        # Like the VART Python binding: a (job_id, status) tuple, passed back to wait().
        return (self.executions, 0)

    def wait(self, job: tuple[int, int]) -> None:
        assert isinstance(job, tuple), "VART hands back the (job_id, status) tuple"
        job_id, _status = job
        assert 0 < job_id <= self.executions


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    subgraphs: list[_Subgraph],
) -> None:
    xir = ModuleType("xir")
    xir.Graph = _Graph  # type: ignore[attr-defined]
    vart = ModuleType("vart")
    vart.Runner = SimpleNamespace(create_runner=lambda subgraph, mode: _Runner(subgraph))
    _Graph.pending = subgraphs
    monkeypatch.setitem(sys.modules, "xir", xir)
    monkeypatch.setitem(sys.modules, "vart", vart)


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "yolov5_nano.xmodel"
    path.write_bytes(b"fake xmodel")
    return path


def _head(
    fix_point: int, cells: list[tuple[int, int, int, dict[str, float]]], stride: int
) -> NDArray[np.int8]:
    """Int8 NHWC head with the given ``(row, column, anchor, values)`` slots set."""

    dims = OUTPUT_DIMS[stride]
    dense = np.zeros((dims[1], dims[2], dims[3]), dtype=np.float64)
    for anchor in range(3):
        dense[..., anchor * SLOT + OBJECTNESS] = BACKGROUND
        dense[..., anchor * SLOT + CLASSES : (anchor + 1) * SLOT] = BACKGROUND
    for row, column, anchor, kwargs in cells:
        _set(dense, row, column, anchor=anchor, **kwargs)
    return np.clip(np.rint(dense * (2.0**fix_point)), -128.0, 127.0).astype(np.int8)


def _detector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    payloads: dict[int, NDArray[np.int8]] | None = None,
    output_order: tuple[int, ...] = (8, 16, 32),
    **kwargs: float,
) -> Yolov5Detector:
    fix_points = dict(zip(STRIDES, QUANTIZED_FIX_POINTS, strict=True))
    outputs = [
        _Tensor(f"out_{stride}", OUTPUT_DIMS[stride], fix_point=fix_points[stride])
        for stride in output_order
    ]
    subgraph = _Subgraph(
        "subgraph_dpu", "DPU", [_Tensor("input", INPUT_DIMS, fix_point=6)], outputs
    )
    if payloads is not None:
        subgraph.payload = [payloads[stride] for stride in output_order]
    _install_fake_runtime(monkeypatch, [subgraph])
    return Yolov5Detector(_model(tmp_path), **kwargs)


# -- the detector ------------------------------------------------------------ #


def test_the_detector_reads_the_graph_and_orders_outputs_by_stride(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = _detector(
        monkeypatch,
        tmp_path,
        output_order=(32, 8, 16),
        payloads={
            32: _head(4, [(0, 0, 0, {"objectness": 10.0, "class_id": 2})], 32),
            8: _head(3, [], 8),
            16: _head(4, [], 16),
        },
    )

    assert detector.input_size == YOLOV5_INPUT_SIZE
    assert detector.strides == STRIDES
    assert detector.fix_points == QUANTIZED_FIX_POINTS
    assert set(detector.stage_ms) == {"preprocess", "dpu", "decode"}


def test_the_detector_returns_detections_in_source_pixels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = _detector(
        monkeypatch,
        tmp_path,
        payloads={
            8: _head(3, [(12, 0, 0, {"log_w": math.log(2.0)})], 8),
            16: _head(4, [], 16),
            32: _head(4, [], 32),
        },
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_id == PERSON
    assert detection.score == pytest.approx(1.0, abs=1e-3)
    x1, y1, x2, y2 = detection.box
    assert 0.0 <= x1 < x2 <= 640.0
    assert 0.0 <= y1 < y2 <= 480.0
    assert all(value >= 0.0 for value in detector.stage_ms.values())


def test_the_detector_requires_the_model_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runtime(monkeypatch, [])

    with pytest.raises(FileNotFoundError):
        Yolov5Detector(tmp_path / "missing.xmodel")


def test_the_detector_requires_a_dpu_subgraph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subgraph = _Subgraph("cpu", "USER", [], [])
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match="DPU subgraph"):
        Yolov5Detector(_model(tmp_path))


@pytest.mark.parametrize(
    ("input_dims", "message"),
    [
        ([1, 640, 3, 640], "NHWC"),
        ([640, 640, 3], "4-D"),
        ([1, 640, 640, 4], "NHWC"),
    ],
)
def test_the_detector_rejects_unexpected_input_layouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_dims: list[int],
    message: str,
) -> None:
    outputs = [
        _Tensor(f"out_{stride}", OUTPUT_DIMS[stride], fix_point=fix_point)
        for stride, fix_point in zip(STRIDES, QUANTIZED_FIX_POINTS, strict=True)
    ]
    subgraph = _Subgraph("dpu", "DPU", [_Tensor("input", input_dims, fix_point=6)], outputs)
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match=message):
        Yolov5Detector(_model(tmp_path))


def test_the_detector_rejects_non_yolov5_output_strides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = [_Tensor("out_4", [1, 160, 160, 255], fix_point=3)]
    subgraph = _Subgraph("dpu", "DPU", [_Tensor("input", INPUT_DIMS, fix_point=6)], outputs)
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match="strides"):
        Yolov5Detector(_model(tmp_path))


@pytest.mark.parametrize("thresholds", [{"conf_thr": 1.5}, {"iou_thr": -0.1}])
def test_the_detector_rejects_bad_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, thresholds: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        _detector(monkeypatch, tmp_path, **thresholds)


# -- pipelined submit/wait --------------------------------------------------- #


def _payload(stride: int, fix_point: int, class_id: int) -> NDArray[np.int8]:
    """One detection inside the letterboxed part of a 480x640 frame."""

    return _head(
        fix_point,
        [(12, 0, 0, {"tx": 2.5, "ty": 10.0, "objectness": 10.0, "class_id": class_id})],
        stride,
    )


def _pipelined_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Yolov5Detector, _Subgraph]:
    detector = _detector(monkeypatch, tmp_path)
    _Graph.pending[0].payloads_by_execution = [
        [
            _payload(8, 3, 2),
            np.full(OUTPUT_DIMS[16], -128, np.int8),
            np.full(OUTPUT_DIMS[32], -128, np.int8),
        ],
        [
            _payload(8, 3, 5),
            np.full(OUTPUT_DIMS[16], -128, np.int8),
            np.full(OUTPUT_DIMS[32], -128, np.int8),
        ],
    ]
    return detector, _Graph.pending[0]


def test_the_detector_keeps_two_frames_in_flight_without_overwriting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector, _subgraph = _pipelined_detector(monkeypatch, tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    first = detector.submit(frame)
    assert detector.pending_jobs == 1
    second = detector.submit(frame)
    assert detector.pending_jobs == 2

    first_detections = detector.wait(first)
    second_detections = detector.wait(second)

    assert [detection.class_id for detection in first_detections] == [2]
    assert [detection.class_id for detection in second_detections] == [5]


def test_a_third_submit_requires_a_wait(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    detector, _subgraph = _pipelined_detector(monkeypatch, tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.submit(frame)
    detector.submit(frame)

    with pytest.raises(RuntimeError, match="busy"):
        detector.submit(frame)


def test_wait_rejects_an_unknown_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    detector, _subgraph = _pipelined_detector(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="token"):
        detector.wait(1234)


def test_detect_is_submit_plus_wait(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    detector, _subgraph = _pipelined_detector(monkeypatch, tmp_path)

    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    assert detector.pending_jobs == 0
    assert [detection.class_id for detection in detections] == [2]
    assert set(detector.stage_ms) == {"preprocess", "dpu", "decode"}


def test_the_detector_decodes_the_same_detections_as_the_float_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = {
        8: _head(3, [(0, 0, 0, {"log_w": math.log(2.0)})], 8),
        16: _head(4, [], 16),
        32: _head(4, [(0, 0, 1, {"objectness": 8.0, "class_id": 41})], 32),
    }
    detector = _detector(monkeypatch, tmp_path, payloads=outputs)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    _image, scale, pad = prepare_input(frame)
    expected = detections_from_heads(
        _dequantized([outputs[stride] for stride in STRIDES], QUANTIZED_FIX_POINTS),
        STRIDES,
        conf_thr=0.35,
        iou_thr=0.45,
        scale=scale,
        pad=pad,
        source_hw=(480, 640),
    )

    assert detections == expected
    assert detector.last_candidate_count == 2


# -- host micro-benchmark ---------------------------------------------------- #


@pytest.mark.benchmark
def test_a_multi_label_frame_decodes_well_inside_the_decode_budget() -> None:
    """Host micro-benchmark of the full 640x640 int8 decode.

    The heads are the compiled graph's (80x80/40x40/20x20, 255 channels) with
    24 planted slots, half of which hold the person and the bottle the showcase
    decodes multi-label; the background is at -128, so the int8 prefilter still
    has to scan the whole grid.
    """

    heads = _planted_heads(shapes=QUANTIZED_SHAPES, planted=24)
    decoder = QuantizedYolov5Decoder(QUANTIZED_SHAPES, STRIDES, QUANTIZED_FIX_POINTS)
    overrides = class_thresholds(0.35, class_conf={"person": 0.40}, object_conf=0.25)
    for _ in range(30):
        decoder.decode(heads, 0.35, 0.45, overrides)

    # The host is shared with the other booth processes: the fastest frame is
    # the honest estimate of the work one decode costs.
    samples = []
    for _ in range(200):
        started = time.perf_counter()
        decoder.decode(heads, 0.35, 0.45, overrides)
        samples.append((time.perf_counter() - started) * 1000.0)
    per_frame_ms = min(samples)

    assert per_frame_ms < 2.5, f"{per_frame_ms:.3f} ms per frame for the whole decode"


def test_live_class_thresholds_apply_to_the_next_decode_without_a_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payloads = {
        8: _payload(8, 3, 0),
        16: np.full(OUTPUT_DIMS[16], -128, np.int8),
        32: np.full(OUTPUT_DIMS[32], -128, np.int8),
    }
    detector = _detector(monkeypatch, tmp_path, payloads=payloads, class_thr={0: 1.0})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    runner = detector._runner
    assert detector.detect(frame) == []

    detector.set_class_thresholds({0: 0.1})
    assert [item.class_id for item in detector.detect(frame)] == [0]
    assert detector._runner is runner

    with pytest.raises(ValueError):
        detector.set_class_thresholds({0: 1.5})
    assert detector.class_thresholds == {0: 0.1}


def test_a_crowded_frame_decodes_like_the_float_path() -> None:
    """150 planted objects in clusters, more than the decode evaluates.

    The decoder scores only the objectness-best slots of the frame, so the float
    reference is run on the same heads with the cap's rejects silenced: the two
    paths must still agree box for box, which is what pins the selection rule.
    """
    heads = _crowded_heads()
    decoder = QuantizedYolov5Decoder(QUANTIZED_SHAPES, STRIDES, QUANTIZED_FIX_POINTS)
    overrides = class_thresholds(0.35, class_conf={"person": 0.40}, object_conf=0.25)

    actual = decoder.decode(heads, 0.35, 0.45, overrides)

    expected = decode_yolov5(
        _frame_the_cap_decodes(heads),
        STRIDES,
        conf_thr=0.35,
        iou_thr=0.45,
        logits=True,
        class_thr=overrides,
    )
    for got, want in zip(actual, expected, strict=True):
        assert np.array_equal(got, want)
    # The crowd really fills the frame: more slots survive the objectness gate
    # than the cap evaluates, and the detection list is full.
    assert _survivors(heads) > _SLOT_CANDIDATE_CAP
    assert actual[1].size == 100


def _survivors(heads: list[NDArray[np.int8]]) -> int:
    """Slots of ``heads`` above the background sentinel of the crowded frame."""
    return sum(
        int(np.count_nonzero(head.reshape(-1, SLOT)[:, OBJECTNESS] > np.iinfo(np.int8).min))
        for head in heads
    )


def _frame_the_cap_decodes(heads: list[NDArray[np.int8]]) -> list[NDArray[np.float64]]:
    """Dequantized ``heads`` with every slot beyond the decode cap silenced.

    :class:`QuantizedYolov5Decoder` evaluates the class logits of the
    :data:`_SLOT_CANDIDATE_CAP` objectness-best survivors only (ties to the lower
    slot). Every other slot could never be a detection, so dropping it is the
    same as leaving it below the background: this returns the heads the float
    path can be compared against. Background slots are exactly ``-128`` in the
    test frames, below the loosest gate, so the survivors are the slots above it.
    """
    survivors = [
        np.flatnonzero(head.reshape(-1, SLOT)[:, OBJECTNESS] > np.iinfo(np.int8).min)
        for head in heads
    ]
    count = sum(index.size for index in survivors)
    kept = np.zeros(count, dtype=np.bool_)
    if count > _SLOT_CANDIDATE_CAP:
        objectness = np.concatenate(
            [
                head.reshape(-1, SLOT)[index, OBJECTNESS]
                for head, index in zip(heads, survivors, strict=True)
            ]
        ).astype(np.int16)
        kept[np.argsort(-objectness, kind="stable")[:_SLOT_CANDIDATE_CAP]] = True
    else:
        kept[:] = True

    dequantized = [
        head.astype(np.float64) * 2.0**-fix_point
        for head, fix_point in zip(heads, QUANTIZED_FIX_POINTS, strict=True)
    ]
    start = 0
    for head, index in zip(dequantized, survivors, strict=True):
        stop = start + index.size
        head.reshape(-1, SLOT)[index[~kept[start:stop]], OBJECTNESS] = BACKGROUND
        start = stop
    return dequantized


@pytest.mark.benchmark
def test_a_crowded_frame_decodes_well_inside_the_decode_budget() -> None:
    """Host micro-benchmark of the crowded decode, the busiest the NMS gets.

    The board estimate is this host number times the board's slower cores; the
    profile script reports the same heads on the board with ``--busy``.
    """

    heads = _crowded_heads()
    decoder = QuantizedYolov5Decoder(QUANTIZED_SHAPES, STRIDES, QUANTIZED_FIX_POINTS)
    overrides = class_thresholds(0.35, class_conf={"person": 0.40}, object_conf=0.25)
    for _ in range(30):
        decoder.decode(heads, 0.35, 0.45, overrides)

    # The host is shared with the other booth processes: the fastest frame is
    # the honest estimate of the work one decode costs.
    samples = []
    for _ in range(100):
        started = time.perf_counter()
        decoder.decode(heads, 0.35, 0.45, overrides)
        samples.append((time.perf_counter() - started) * 1000.0)
    per_frame_ms = min(samples)

    assert per_frame_ms < 2.0, f"{per_frame_ms:.3f} ms per frame for the crowded decode"


def test_a_crowded_frame_hands_the_tracker_only_post_nms_detections() -> None:
    """The tracker and the still gate see the capped survivors, never candidates.

    ``Yolov5Detector.wait`` returns exactly what :meth:`QuantizedYolov5Decoder.decode`
    returns, so this is the list the loop tracks: at most
    :data:`~cpsa.yolox.MAX_DETECTIONS` boxes, none of two of one class above the
    NMS threshold.
    """
    from cpsa.yolox import MAX_DETECTIONS, _iou

    heads = _crowded_heads()
    decoder = QuantizedYolov5Decoder(QUANTIZED_SHAPES, STRIDES, QUANTIZED_FIX_POINTS)
    overrides = class_thresholds(0.35, class_conf={"person": 0.40}, object_conf=0.25)

    boxes, scores, classes = decoder.decode(heads, 0.35, 0.45, overrides)

    assert decoder.last_candidate_count > boxes.shape[0]
    assert boxes.shape[0] == MAX_DETECTIONS
    for first in range(boxes.shape[0]):
        for second in range(first + 1, boxes.shape[0]):
            if classes[first] != classes[second]:
                continue
            assert _iou(boxes[first], boxes[second][None, :])[0] <= 0.45
