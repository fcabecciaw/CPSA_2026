"""YOLOX decode, the int8 prefilter, preprocessing and the COCO labels (F-S1)."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa import yolox
from cpsa.dpu import input_quantization_lut_scaled, quantize_fix_point
from cpsa.yolox import (
    COCO80_EN,
    COCO80_IT,
    MAX_CLASSES_PER_CELL,
    MAX_DETECTIONS,
    QuantizedYoloxDecoder,
    boxes_to_source,
    class_thresholds,
    decode_yolox,
    label_for,
    labels_for,
    prepare_input,
    prepare_input_into,
)

STRIDES = (8, 16, 32)
BACKGROUND = -1000.0
#: The booth scene: a person and the bottle held in front of it.
PERSON = 0
BOTTLE = 39

#: Shapes and fix point of the compiled YOLOX-nano graph (``inspect_xmodel.py``).
QUANTIZED_SHAPES = ((52, 52, 85), (26, 26, 85), (13, 13, 85))
QUANTIZED_FIX_POINT = 3


def _heads(
    sizes: tuple[tuple[int, int], ...] = ((2, 2), (1, 1), (1, 1)),
    classes: int = 80,
    background: float = BACKGROUND,
) -> list[NDArray[np.float64]]:
    """Background heads: objectness and class values far below any threshold."""

    heads = []
    for height, width in sizes:
        head = np.zeros((height, width, 5 + classes), dtype=np.float64)
        head[..., 4] = background
        head[..., 5:] = background
        heads.append(head)
    return heads


def _set(
    head: NDArray[np.float64],
    row: int,
    column: int,
    *,
    tx: float = 0.0,
    ty: float = 0.0,
    log_w: float = 0.0,
    log_h: float = 0.0,
    objectness: float = 20.0,
    class_id: int = 0,
    class_logit: float = 20.0,
    background: float = BACKGROUND,
) -> None:
    """Write one candidate cell: grid offsets, log size, objectness and class logits."""

    head[row, column, 0] = tx
    head[row, column, 1] = ty
    head[row, column, 2] = log_w
    head[row, column, 3] = log_h
    head[row, column, 4] = objectness
    head[row, column, 5:] = background
    head[row, column, 5 + class_id] = class_logit


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def test_grid_offset_exp_and_stride_decode() -> None:
    heads = _heads()
    _set(heads[0], 0, 0, tx=0.5, ty=0.25, log_w=math.log(2.0), log_h=math.log(2.0))
    _set(heads[0], 0, 1, tx=0.0, ty=0.0, log_w=math.log(0.5), log_h=math.log(0.5))
    _set(heads[2], 0, 0, tx=0.0, ty=0.0, log_w=0.0, log_h=0.0)

    boxes, scores, classes = decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)

    assert len(scores) == 3
    assert scores == pytest.approx([1.0, 1.0, 1.0])
    assert classes.tolist() == [0, 0, 0]
    # stride 8, cell (0, 0): centre (4, 2), 16x16 box.
    assert boxes[0] == pytest.approx([-4.0, -6.0, 12.0, 10.0])
    # stride 8, cell (0, 1): centre (8, 0), 4x4 box.
    assert boxes[1] == pytest.approx([6.0, -2.0, 10.0, 2.0])
    # stride 32, single cell: centre (0, 0), 32x32 box.
    assert boxes[2] == pytest.approx([-16.0, -16.0, 16.0, 16.0])


def test_score_is_objectness_times_best_class_in_the_logit_domain() -> None:
    heads = _heads()
    _set(heads[0], 0, 0, objectness=_logit(0.8), class_id=3, class_logit=_logit(0.9))
    heads[0][0, 0, 5] = _logit(0.5)

    _boxes, scores, classes = decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)

    assert scores == pytest.approx([0.8 * 0.9])
    assert classes.tolist() == [3]


def test_logits_false_uses_probabilities_as_is() -> None:
    heads = _heads(background=0.0)
    _set(heads[0], 0, 0, objectness=0.8, class_id=2, class_logit=0.9, background=0.0)

    _boxes, scores, classes = decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=False)

    assert scores == pytest.approx([0.72])
    assert classes.tolist() == [2]


def test_the_conf_threshold_is_strict() -> None:
    heads = _heads(background=0.0)
    _set(heads[0], 0, 0, objectness=0.5, class_logit=0.5, background=0.0)

    assert decode_yolox(heads, STRIDES, conf_thr=0.25, iou_thr=0.45, logits=False)[1].size == 0
    kept = decode_yolox(heads, STRIDES, conf_thr=0.2499, iou_thr=0.45, logits=False)[1]
    assert kept == pytest.approx([0.25])


def test_class_aware_nms_suppresses_duplicates_of_one_class_only() -> None:
    heads = _heads()
    _set(heads[0], 0, 0, tx=0.5, ty=0.5, log_w=math.log(2.0), log_h=math.log(2.0), objectness=10.0)
    _set(heads[0], 0, 1, tx=-0.4, ty=0.5, log_w=math.log(2.0), log_h=math.log(2.0))

    assert len(decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)[1]) == 1

    heads = _heads()
    _set(heads[0], 0, 0, tx=0.5, ty=0.5, log_w=math.log(2.0), log_h=math.log(2.0), objectness=10.0)
    _set(
        heads[0],
        0,
        1,
        tx=-0.4,
        ty=0.5,
        log_w=math.log(2.0),
        log_h=math.log(2.0),
        class_id=7,
    )
    _boxes, scores, classes = decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)
    assert len(scores) == 2
    assert sorted(classes.tolist()) == [0, 7]


# -- multi-label decode: the bottle in front of the person --------------------


def test_a_bottle_sharing_a_persons_cell_is_decoded_with_the_person() -> None:
    heads = _heads()
    _set(heads[0], 0, 0, objectness=_logit(0.9), class_id=PERSON, class_logit=_logit(0.9))
    heads[0][0, 0, 5 + BOTTLE] = _logit(0.8)

    boxes, scores, classes = decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)

    # One cell, two objects: the person no longer swallows the bottle it holds.
    assert classes.tolist() == [PERSON, BOTTLE]
    assert scores == pytest.approx([0.9 * 0.9, 0.9 * 0.8])
    assert boxes[0].tolist() == pytest.approx(boxes[1].tolist())


def test_per_class_thresholds_keep_a_weak_bottle_in_the_persons_cell() -> None:
    heads = _heads()
    _set(heads[0], 0, 0, objectness=_logit(0.9), class_id=PERSON, class_logit=_logit(0.9))
    heads[0][0, 0, 5 + BOTTLE] = _logit(0.4)

    global_only = decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)
    per_class = decode_yolox(
        heads,
        STRIDES,
        conf_thr=0.5,
        iou_thr=0.45,
        logits=True,
        class_thr={PERSON: 0.6, BOTTLE: 0.25},
    )
    person_dropped = decode_yolox(
        heads,
        STRIDES,
        conf_thr=0.5,
        iou_thr=0.45,
        logits=True,
        class_thr={PERSON: 0.9, BOTTLE: 0.25},
    )

    assert global_only[2].tolist() == [PERSON]
    assert per_class[2].tolist() == [PERSON, BOTTLE]
    # The bottle's own threshold is what keeps it: 0.9 * 0.4 = 0.36.
    assert per_class[1] == pytest.approx([0.81, 0.36])
    # ... and a person threshold above 0.81 keeps the bottle alone.
    assert person_dropped[2].tolist() == [BOTTLE]


def test_a_cell_yields_at_most_the_three_best_classes() -> None:
    heads = _heads(background=0.0)
    _set(heads[0], 0, 0, objectness=1.0, class_id=PERSON, class_logit=0.9, background=0.0)
    for class_id, probability in ((3, 0.8), (7, 0.7), (11, 0.6), (13, 0.5)):
        heads[0][0, 0, 5 + class_id] = probability

    _boxes, scores, classes = decode_yolox(heads, STRIDES, conf_thr=0.1, iou_thr=0.45, logits=False)

    assert classes.tolist() == [PERSON, 3, 7]
    assert scores == pytest.approx([0.9, 0.8, 0.7])
    assert scores.size == MAX_CLASSES_PER_CELL == 3


def test_class_thresholds_resolve_names_over_object_and_global_defaults() -> None:
    thresholds = class_thresholds(0.35, class_conf={"person": 0.4})

    assert len(thresholds) == len(COCO80_EN)
    assert thresholds[PERSON] == 0.4
    assert thresholds[BOTTLE] == 0.35

    loose = class_thresholds(0.35, class_conf={"person": 0.4}, object_conf=0.25)

    assert loose[PERSON] == 0.4
    assert loose[BOTTLE] == loose[COCO80_EN.index("cup")] == 0.25


def test_bad_class_thresholds_are_rejected() -> None:
    heads = _heads()

    with pytest.raises(ValueError, match="class threshold"):
        decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True, class_thr={80: 0.3})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True, class_thr={0: 1.5})
    with pytest.raises(ValueError, match="unknown COCO class"):
        class_thresholds(0.35, class_conf={"unicorn": 0.4})
    with pytest.raises(ValueError, match="object_conf"):
        class_thresholds(0.35, object_conf=2.0)


def _reference_nms(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    classes: NDArray[np.intp],
    iou_thr: float,
    cap: int | None = None,
) -> list[int]:
    """The single-pass class-aware greedy NMS the native call must reproduce."""

    if cap is not None and scores.size > cap:
        best = np.argsort(-scores, kind="stable")[:cap]
        boxes, scores, classes = boxes[best], scores[best], classes[best]
    remaining = np.ones(scores.size, dtype=np.bool_)
    keep: list[int] = []
    for position in np.argsort(-scores, kind="stable"):
        if not remaining[position]:
            continue
        keep.append(int(position))
        remaining[position] = False
        rest = np.flatnonzero(remaining)
        if rest.size == 0:
            break
        overlap = yolox._iou(boxes[position], boxes[rest])
        same_class = classes[rest] == classes[position]
        remaining[rest[same_class & (overlap > iou_thr)]] = False
    return sorted(keep)


def test_the_native_nms_keeps_what_the_greedy_pass_keeps() -> None:
    """``cv2.dnn.NMSBoxesBatched`` reproduces the frame-wide class-aware greedy pass.

    Half of the sampled boxes reach outside the frame (their origin is negative),
    which the native call only handles after the decoder translates them.
    """
    generator = np.random.default_rng(20261112)
    for _ in range(40):
        count = int(generator.integers(2, 90))
        centres = generator.uniform(-60.0, 380.0, size=(count, 2))
        sides = generator.uniform(4.0, 140.0, size=(count, 2))
        boxes = np.column_stack([centres - sides / 2.0, centres + sides / 2.0])
        scores = generator.uniform(0.01, 1.0, size=count)
        classes = generator.integers(0, 5, size=count).astype(np.intp)

        actual = yolox._class_aware_nms(boxes, scores, classes, 0.45)

        assert sorted(actual.tolist()) == _reference_nms(boxes, scores, classes, 0.45)


def test_the_native_nms_keeps_a_crowd_of_clusters() -> None:
    """A busy frame: clusters of same-class duplicates, a few other classes around."""
    generator = np.random.default_rng(7)
    centres = generator.uniform(0.0, 400.0, size=(200, 2))
    centres = np.repeat(centres[:40], 5, axis=0) + generator.normal(0.0, 3.0, (200, 2))
    sides = generator.uniform(20.0, 120.0, size=(200, 2))
    boxes = np.column_stack([centres - sides / 2.0, centres + sides / 2.0])
    scores = generator.uniform(0.05, 1.0, size=200)
    classes = generator.integers(0, 4, size=200).astype(np.intp)

    actual = yolox._class_aware_nms(boxes, scores, classes, 0.45)

    assert sorted(actual.tolist()) == _reference_nms(boxes, scores, classes, 0.45)
    assert actual.size < 200  # the clusters really did collapse


def test_the_native_nms_suppresses_a_box_outside_the_frame() -> None:
    """A negative origin is geometry, not an exemption: the duplicate still dies.

    ``cv2.dnn.NMSBoxesBatched`` silently drops rects with a negative origin when
    they are handed over untranslated, which would keep a duplicate and lose a
    real detection, so the translation is part of the contract.
    """
    boxes = np.array(
        [[-120.0, -80.0, -20.0, 10.0], [-118.0, -78.0, -22.0, 8.0], [-200.0, 300.0, -100.0, 400.0]]
    )
    scores = np.array([0.9, 0.8, 0.7])
    classes = np.array([0, 0, 0], dtype=np.intp)

    actual = yolox._class_aware_nms(boxes, scores, classes, 0.45)

    assert actual.tolist() == [0, 2]


def test_the_native_nms_breaks_exact_score_ties_by_index_within_a_class() -> None:
    """Equal scores: the lower index is visited first, as the greedy pass did."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 10.0, 10.0]])
    scores = np.array([0.75, 0.75])
    classes = np.array([3, 3], dtype=np.intp)

    actual = yolox._class_aware_nms(boxes, scores, classes, 0.45)

    assert actual.tolist() == [0]


def test_a_score_tie_across_classes_keeps_both_in_native_order() -> None:
    """Cross-class ties keep both boxes; the native call orders them by input index.

    The previous per-class pass walked the classes in ``np.unique`` order, so the
    *order* of two exactly equal scores could differ; the kept set never did, and
    ``_finalize`` sorts the survivors by score afterwards.
    """
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
    scores = np.array([0.5, 0.5])
    classes = np.array([2, 0], dtype=np.intp)

    actual = yolox._class_aware_nms(boxes, scores, classes, 0.45)

    assert sorted(actual.tolist()) == [0, 1]


def test_the_candidate_cap_keeps_the_best_scores_before_nms() -> None:
    """A pathological frame is capped before the native pass sees it."""
    count = 400
    columns = np.arange(count, dtype=np.float64)
    boxes = np.column_stack(
        [columns * 10.0, np.zeros(count), columns * 10.0 + 5.0, np.full(count, 5.0)]
    )
    scores = (columns + 1.0) / count  # every box is its own object: nothing overlaps
    classes = np.zeros(count, dtype=np.intp)

    _boxes, kept_scores, _classes = yolox._finalize([boxes], [scores], [classes], 0.45)

    assert kept_scores.size == MAX_DETECTIONS
    assert np.array_equal(np.sort(kept_scores), np.sort(scores)[-MAX_DETECTIONS:])
    # The cap sits above the output size: the last kept detection comes from the
    # top candidates, never from the ones the cap dropped.
    assert kept_scores.min() >= np.sort(scores)[-yolox._NMS_CANDIDATE_CAP]


def test_detections_are_score_descending_and_capped() -> None:
    heads = _heads(sizes=((11, 11), (1, 1), (1, 1)), background=0.0)
    for row in range(11):
        for column in range(11):
            _set(
                heads[0],
                row,
                column,
                tx=0.5,
                ty=0.5,
                log_w=math.log(0.1),
                log_h=math.log(0.1),
                objectness=float(row * 11 + column) / 121.0,
                class_logit=1.0,
                background=0.0,
            )

    _boxes, scores, _classes = decode_yolox(
        heads, STRIDES, conf_thr=0.0, iou_thr=0.45, logits=False
    )

    assert scores.size == MAX_DETECTIONS == 100
    assert np.all(np.diff(scores) <= 0.0)


def test_empty_result_has_typed_empty_arrays() -> None:
    boxes, scores, classes = decode_yolox(
        _heads(), STRIDES, conf_thr=0.35, iou_thr=0.45, logits=True
    )

    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert classes.shape == (0,)
    assert boxes.dtype == np.float64
    assert classes.dtype == np.intp


def test_non_finite_candidates_are_dropped() -> None:
    heads = _heads()
    _set(heads[0], 0, 0, log_w=math.inf)
    _set(heads[0], 0, 1, tx=math.nan)
    _set(heads[0], 1, 0, ty=math.inf)

    boxes, scores, _classes = decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)

    assert boxes.shape == (0, 4)
    assert scores.size == 0


@pytest.mark.parametrize(
    ("heads", "strides", "conf_thr", "iou_thr"),
    [
        (_heads(), STRIDES[:2], 0.5, 0.45),
        ((), (), 0.5, 0.45),
        ([np.zeros((2, 2), dtype=np.float64)], (8,), 0.5, 0.45),
        ([np.zeros((2, 2, 4), dtype=np.float64)], (8,), 0.5, 0.45),
        (_heads()[:2], (8, 16), 1.5, 0.45),
        (_heads()[:2], (8, 16), 0.5, -0.1),
        (_heads(), (8, 16, 0), 0.5, 0.45),
    ],
)
def test_invalid_inputs_raise(
    heads: object, strides: tuple[int, ...], conf_thr: float, iou_thr: float
) -> None:
    with pytest.raises(ValueError):
        decode_yolox(heads, strides, conf_thr=conf_thr, iou_thr=iou_thr, logits=True)


def test_heads_with_different_channel_counts_are_rejected() -> None:
    heads = _heads()
    mixed = _heads()
    mixed[1] = np.zeros((1, 1, 5 + 79), dtype=np.float64)

    with pytest.raises(ValueError, match="channels"):
        decode_yolox(
            [heads[0], mixed[1], heads[2]], STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True
        )


# -- the int8, threshold-first decoder (S5, 60 fps) ---------------------------


def _quantized_heads(
    shapes: tuple[tuple[int, int, int], ...] = QUANTIZED_SHAPES,
    seed: int = 7,
) -> list[NDArray[np.int8]]:
    """Uniform random int8 heads: every cell is a possible candidate."""

    generator = np.random.default_rng(seed)
    return [generator.integers(-128, 128, size=shape, dtype=np.int8) for shape in shapes]


def _dequantized(heads: list[NDArray[np.int8]], fix_point: int) -> list[NDArray[np.float64]]:
    """The float heads :func:`decode_yolox` would see after the DPU dequantizes."""

    return [head.astype(np.float64) * 2.0**-fix_point for head in heads]


def _realistic_heads(
    shapes: tuple[tuple[int, int, int], ...] = QUANTIZED_SHAPES,
    planted: int = 8,
    seed: int = 11,
) -> list[NDArray[np.int8]]:
    """Background at -128 plus a few planted object cells, as a real frame looks."""

    generator = np.random.default_rng(seed)
    heads = []
    for height, width, channels in shapes:
        head = np.full((height, width, channels), -128, dtype=np.int8)
        head[..., :4] = generator.integers(-40, 40, size=(height, width, 4), dtype=np.int8)
        for _ in range(planted):
            row = int(generator.integers(0, height))
            column = int(generator.integers(0, width))
            head[row, column, 4] = generator.integers(60, 121)
            head[row, column, 5:] = generator.integers(-100, -19, size=channels - 5)
            head[row, column, 5 + int(generator.integers(0, channels - 5))] = generator.integers(
                70, 128
            )
        heads.append(head)
    return heads


def _overlapping_heads(
    shapes: tuple[tuple[int, int, int], ...] = ((13, 13, 85),),
    planted: int = 6,
    seed: int = 13,
) -> list[NDArray[np.int8]]:
    """The booth frame: person cells that also hold a bottle and a cup.

    Every planted cell has three classes above the showcase thresholds, the
    overlap the multi-label decode exists for; the background is at -128, as a
    real frame's, so the int8 prefilter still drops the whole grid.
    """

    generator = np.random.default_rng(seed)
    heads = []
    for height, width, channels in shapes:
        head = np.full((height, width, channels), -128, dtype=np.int8)
        head[..., :4] = generator.integers(-40, 40, size=(height, width, 4), dtype=np.int8)
        for _ in range(planted):
            row = int(generator.integers(0, height))
            column = int(generator.integers(0, width))
            head[row, column, 4] = generator.integers(70, 121)
            head[row, column, 5:] = generator.integers(-100, -19, size=channels - 5)
            head[row, column, 5 + PERSON] = generator.integers(80, 128)
            head[row, column, 5 + BOTTLE] = generator.integers(40, 100)
            head[row, column, 5 + 41] = generator.integers(30, 90)  # cup
        heads.append(head)
    return heads


@pytest.mark.parametrize("conf_thr", [0.0, 0.05, 0.35, 0.9, 1.0])
def test_the_quantized_decoder_matches_decode_yolox_on_random_heads(conf_thr: float) -> None:
    heads = _quantized_heads()
    decoder = QuantizedYoloxDecoder(QUANTIZED_SHAPES, STRIDES, (QUANTIZED_FIX_POINT,) * 3)

    actual = decoder.decode(heads, conf_thr, 0.45)
    expected = decode_yolox(
        _dequantized(heads, QUANTIZED_FIX_POINT),
        STRIDES,
        conf_thr=conf_thr,
        iou_thr=0.45,
        logits=True,
    )

    for got, want in zip(actual, expected, strict=True):
        assert np.array_equal(got, want)


@pytest.mark.parametrize("conf_thr", [0.0, 0.2, 0.35, 0.6, 0.95, 1.0])
def test_the_quantized_decoder_matches_decode_yolox_on_realistic_heads(conf_thr: float) -> None:
    heads = _realistic_heads()
    decoder = QuantizedYoloxDecoder(QUANTIZED_SHAPES, STRIDES, (QUANTIZED_FIX_POINT,) * 3)

    actual = decoder.decode(heads, conf_thr, 0.45)
    expected = decode_yolox(
        _dequantized(heads, QUANTIZED_FIX_POINT),
        STRIDES,
        conf_thr=conf_thr,
        iou_thr=0.45,
        logits=True,
    )

    for got, want in zip(actual, expected, strict=True):
        assert np.array_equal(got, want)
    if conf_thr > 0.0:
        assert decoder.last_candidate_count < sum(head.shape[0] * head.shape[1] for head in heads)


def test_the_quantized_decoder_matches_decode_yolox_on_a_threshold_sweep() -> None:
    shapes = ((8, 8, 10),)
    heads = _realistic_heads(shapes=shapes, planted=6, seed=3)
    decoder = QuantizedYoloxDecoder(shapes, (8,), (3,))

    for step in range(-40, 41, 5):
        conf_thr = 1.0 / (1.0 + math.exp(-step / 8.0))
        actual = decoder.decode(heads, conf_thr, 0.45)
        expected = decode_yolox(
            _dequantized(heads, 3), (8,), conf_thr=conf_thr, iou_thr=0.45, logits=True
        )
        for got, want in zip(actual, expected, strict=True):
            assert np.array_equal(got, want), conf_thr


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
    shapes = ((13, 13, 85),)
    heads = _overlapping_heads(shapes=shapes, planted=6)
    decoder = QuantizedYoloxDecoder(shapes, (8,), (3,))
    conf_thr = 0.35

    actual = decoder.decode(heads, conf_thr, 0.45, overrides)
    expected = decode_yolox(
        _dequantized(heads, 3),
        (8,),
        conf_thr=conf_thr,
        iou_thr=0.45,
        logits=True,
        class_thr=overrides,
    )

    for got, want in zip(actual, expected, strict=True):
        assert np.array_equal(got, want)
    # The cell overlap is real: some boxes are decoded as more than one class.
    assert len(actual[1]) > len({tuple(box) for box in actual[0].tolist()})


@pytest.mark.benchmark
def test_a_multi_label_frame_decodes_well_inside_the_decode_budget() -> None:
    """Host micro-benchmark of the full 416x416 int8 decode.

    The heads carry 24 object cells, each with the person, a bottle and a cup in
    it: three times the candidates of the old single-label decode and the
    worst case the multi-label change can add to a frame.
    """

    heads = _overlapping_heads(shapes=QUANTIZED_SHAPES, planted=8)
    decoder = QuantizedYoloxDecoder(QUANTIZED_SHAPES, STRIDES, (QUANTIZED_FIX_POINT,) * 3)
    overrides = class_thresholds(0.35, class_conf={"person": 0.40}, object_conf=0.25)
    for _ in range(50):
        decoder.decode(heads, 0.35, 0.45, overrides)

    # The host is shared with the other booth processes: the fastest frame is
    # the honest estimate of the work one decode costs.
    samples = []
    for _ in range(300):
        started = time.perf_counter()
        decoder.decode(heads, 0.35, 0.45, overrides)
        samples.append((time.perf_counter() - started) * 1000.0)
    per_frame_ms = min(samples)

    assert per_frame_ms < 1.0, f"{per_frame_ms:.3f} ms per frame for the whole decode"


def test_the_quantized_decoder_accepts_nhwc_outputs_with_the_batch_axis() -> None:
    heads = _realistic_heads(shapes=((4, 4, 10),), planted=2)
    decoder = QuantizedYoloxDecoder(((4, 4, 10),), (8,), (3,))
    batched = [head.reshape(1, 4, 4, 10) for head in heads]

    assert np.array_equal(
        decoder.decode(batched, 0.35, 0.45)[0], decoder.decode(heads, 0.35, 0.45)[0]
    )


def test_the_quantized_decoder_uses_each_head_fix_point() -> None:
    shapes = ((4, 4, 10), (2, 2, 10), (1, 1, 10))
    heads = _realistic_heads(shapes=shapes, planted=2, seed=5)
    decoder = QuantizedYoloxDecoder(shapes, STRIDES, (3, 4, 6))

    actual = decoder.decode(heads, 0.35, 0.45)
    expected = decode_yolox(
        [head.astype(np.float64) * 2.0**-fix for head, fix in zip(heads, (3, 4, 6), strict=True)],
        STRIDES,
        conf_thr=0.35,
        iou_thr=0.45,
        logits=True,
    )

    for got, want in zip(actual, expected, strict=True):
        assert np.array_equal(got, want)


def test_the_quantized_decoder_reports_the_prefiltered_candidate_count() -> None:
    heads = _realistic_heads(shapes=((8, 8, 10),), planted=3)
    decoder = QuantizedYoloxDecoder(((8, 8, 10),), (8,), (3,))
    assert decoder.last_candidate_count == 0

    boxes, _scores, _classes = decoder.decode(heads, 0.35, 0.45)
    assert len(boxes) <= decoder.last_candidate_count <= 64

    assert decoder.decode(heads, 1.0, 0.45)[0].shape == (0, 4)
    assert decoder.last_candidate_count == 0


@pytest.mark.parametrize(
    ("shapes", "strides", "fix_points"),
    [
        ((), (), ()),
        (((4, 4, 10),), (8, 16), (3, 3)),
        (((4, 4, 10),), (8,), (3, 3)),
        (((4, 4),), (8,), (3,)),
        (((0, 4, 10),), (8,), (3,)),
        (((4, 4, 5),), (8,), (3,)),
        (((4, 4, 10),), (0,), (3,)),
    ],
)
def test_the_quantized_decoder_rejects_bad_geometry(
    shapes: object, strides: object, fix_points: object
) -> None:
    with pytest.raises(ValueError):
        QuantizedYoloxDecoder(shapes, strides, fix_points)  # type: ignore[arg-type]


def test_the_quantized_decoder_rejects_bad_outputs() -> None:
    heads = _realistic_heads(shapes=((4, 4, 10),), planted=1)
    decoder = QuantizedYoloxDecoder(((4, 4, 10),), (8,), (3,))

    with pytest.raises(ValueError, match="outputs"):
        decoder.decode([], 0.35, 0.45)
    with pytest.raises(ValueError, match="int8"):
        decoder.decode([heads[0].astype(np.float64)], 0.35, 0.45)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must be"):
        decoder.decode([heads[0][:2]], 0.35, 0.45)
    with pytest.raises(ValueError, match="conf_thr"):
        decoder.decode(heads, 1.5, 0.45)
    with pytest.raises(ValueError, match="iou_thr"):
        decoder.decode(heads, 0.35, -0.1)


# -- preprocessing ------------------------------------------------------------


def test_prepare_input_converts_to_bgr_and_letterboxes_with_grey_padding() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # red only: BGR input must carry it in channel 2

    bgr, scale, pad = prepare_input(rgb)

    assert bgr.shape == (416, 416, 3)
    assert bgr.flags["C_CONTIGUOUS"]
    assert scale == pytest.approx(416 / 200)
    assert pad == (0.0, pytest.approx((416 - round(100 * 416 / 200)) / 2))
    centre = bgr[208, 208]
    assert centre[2] == 200 and centre[0] == 0 and centre[1] == 0
    assert bgr[0, 0].tolist() == [114, 114, 114]


def test_prepare_input_rejects_bad_frames() -> None:
    with pytest.raises(ValueError):
        prepare_input(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        prepare_input(np.zeros((10, 10, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        prepare_input(np.zeros((0, 10, 3), dtype=np.uint8))


def test_boxes_to_source_inverts_the_letterbox_and_clips() -> None:
    boxes = np.asarray([[-10.0, -20.0, 100.0, 500.0]], dtype=np.float64)

    mapped = boxes_to_source(boxes, 0.5, (10.0, 20.0), (400, 300))

    assert mapped.tolist() == [[0.0, 0.0, 180.0, 400.0]]


def test_the_pass_classes_split_persons_from_every_other_class() -> None:
    assert yolox.PERSON_CLASS == 0
    assert frozenset({0}) == yolox.PERSON_CLASSES
    assert frozenset(range(1, len(COCO80_EN))) == yolox.OBJECT_CLASSES
    assert yolox.PERSON_CLASSES.isdisjoint(yolox.OBJECT_CLASSES)


def test_a_pass_knows_whether_it_reports_only_persons() -> None:
    assert yolox.Pass(1.0, yolox.PERSON_CLASSES).persons_only is True
    assert yolox.Pass(1.5, yolox.OBJECT_CLASSES).persons_only is False
    assert yolox.Pass(1.5).persons_only is False


def test_boxes_to_source_rejects_a_bad_scale() -> None:
    with pytest.raises(ValueError):
        boxes_to_source(np.zeros((0, 4)), 0.0, (0.0, 0.0), (10, 10))


# -- the centre zoom (model.zoom) ---------------------------------------------


def test_the_zoom_of_one_is_the_whole_frame() -> None:
    crop = yolox.zoom_crop(640, 480, 1.0)

    assert crop == yolox.ZoomCrop(left=0, top=0, width=640, height=480)
    assert crop.origin == (0.0, 0.0)
    assert crop.size == (480, 640)


@pytest.mark.parametrize("zoom", [1.0, 1.3, 1.5])
def test_a_zoom_crop_round_trips_the_boxes_it_saw(zoom: float) -> None:
    """A box projected into the crop comes back to the same full-frame pixels."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = yolox.zoom_crop(640, 480, zoom)
    lut = input_quantization_lut_scaled(-2, 1.0)
    out = np.empty((416, 416, 3), dtype=np.int8)
    view = frame[crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]

    scale, pad = prepare_input_into(view, (416, 416), lut, out)
    boxes = np.asarray(
        [
            [crop.left + 20.0, crop.top + 30.0, crop.left + 140.0, crop.top + 260.0],
            [crop.left + 2.0, crop.top + 2.0, crop.left + 60.0, crop.top + 90.0],
        ]
    )
    local = boxes - [crop.left, crop.top, crop.left, crop.top]
    input_boxes = local * scale + [pad[0], pad[1], pad[0], pad[1]]

    mapped = boxes_to_source(input_boxes, scale, pad, crop.size, crop.origin)

    assert mapped == pytest.approx(boxes, abs=1e-9)


def test_a_box_that_runs_into_the_pad_stops_at_the_crop_edge() -> None:
    crop = yolox.zoom_crop(640, 480, 1.5)
    # Far outside the resized region of the input: the pad must not leak.
    input_boxes = np.asarray([[-100.0, -100.0, 900.0, 900.0]])

    mapped = boxes_to_source(input_boxes, 0.8, (10.0, 20.0), crop.size, crop.origin)

    assert mapped.tolist() == [
        [
            float(crop.left),
            float(crop.top),
            float(crop.left + crop.width),
            float(crop.top + crop.height),
        ]
    ]


def test_an_off_centre_zoom_moves_the_crop_and_stays_inside_the_frame() -> None:
    crop = yolox.zoom_crop(640, 480, 1.3, (0.4, 0.6))

    assert crop.left + crop.width / 2 == pytest.approx(0.4 * 640, abs=0.5)
    assert crop.top + crop.height / 2 == pytest.approx(0.6 * 480, abs=0.5)

    edge = yolox.zoom_crop(640, 480, 1.5, (0.0, 1.0))
    assert edge.left == 0
    assert edge.top == 480 - edge.height


@pytest.mark.parametrize(
    ("zoom", "centre"),
    [(0.9, (0.5, 0.5)), (0.0, (0.5, 0.5)), (1.3, (1.5, 0.5)), (1.3, (0.5, -0.1))],
)
def test_a_bad_zoom_is_rejected(zoom: float, centre: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        yolox.zoom_crop(640, 480, zoom, centre)


def test_the_quantization_lut_maps_raw_pixels() -> None:
    lut = input_quantization_lut_scaled(-2, 1.0)

    assert lut.shape == (256,)
    assert lut.dtype == np.int8
    assert lut[255] == 64
    assert lut.tolist() == quantize_fix_point(np.arange(256, dtype=np.float32), -2).tolist()


def test_prepare_input_into_matches_the_two_step_path() -> None:
    rgb = np.random.default_rng(4).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    lut = input_quantization_lut_scaled(-2, 1.0)
    out = np.empty((416, 416, 3), dtype=np.int8)

    scale, pad = prepare_input_into(rgb, (416, 416), lut, out)

    bgr, expected_scale, expected_pad = prepare_input(rgb)
    assert scale == pytest.approx(expected_scale)
    assert pad == pytest.approx(expected_pad)
    assert np.array_equal(out.view(np.uint8), lut.view(np.uint8)[bgr])


def test_prepare_input_into_keeps_the_pad_when_asked() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    lut = input_quantization_lut_scaled(-2, 1.0)
    out = np.empty((416, 416, 3), dtype=np.int8)
    prepare_input_into(rgb, (416, 416), lut, out)
    before = out.copy()

    prepare_input_into(rgb, (416, 416), lut, out, fill_pad=False)

    assert np.array_equal(out, before)


def test_prepare_input_into_rejects_a_bad_buffer() -> None:
    lut = input_quantization_lut_scaled(-2, 1.0)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        prepare_input_into(rgb, (416, 416), lut, np.empty((416, 416, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        prepare_input_into(rgb, (416, 416), lut, np.empty((320, 320, 3), dtype=np.int8))


# -- labels -------------------------------------------------------------------


def test_both_label_tables_hold_eighty_ascii_names() -> None:
    assert len(COCO80_EN) == len(COCO80_IT) == 80
    assert COCO80_EN[0] == "person" and COCO80_IT[0] == "persona"
    assert all(name.isascii() for name in COCO80_EN + COCO80_IT)


def test_labels_for_language() -> None:
    assert labels_for("it") == COCO80_IT
    assert labels_for("en") == COCO80_EN
    with pytest.raises(ValueError):
        labels_for("de")


def test_an_unknown_class_keeps_its_number() -> None:
    assert label_for(0, COCO80_IT) == "persona"
    assert label_for(80, COCO80_IT) == "class 80"
    assert label_for(-1, COCO80_IT) == "class -1"
