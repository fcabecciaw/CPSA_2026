"""YOLOX head decode: grid, stride, exp, sigmoid switch, NMS and limits (F-S1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.vision.yolox_decode import MAX_DETECTIONS, decode_yolox

STRIDES = (8, 16, 32)
BACKGROUND = -1000.0


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
    # stride 8, cell (0, 1): centre ((0 + 1) * 8, 0), 4x4 box.
    assert boxes[1] == pytest.approx([6.0, -2.0, 10.0, 2.0])
    # stride 32, single cell: centre (0, 0), 32x32 box.
    assert boxes[2] == pytest.approx([-16.0, -16.0, 16.0, 16.0])


def test_score_is_objectness_times_best_class_in_logit_domain() -> None:
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


def test_conf_threshold_is_strict() -> None:
    heads = _heads(background=0.0)
    _set(heads[0], 0, 0, objectness=0.5, class_logit=0.5, background=0.0)

    assert decode_yolox(heads, STRIDES, conf_thr=0.25, iou_thr=0.45, logits=False)[1].size == 0
    kept = decode_yolox(heads, STRIDES, conf_thr=0.2499, iou_thr=0.45, logits=False)[1]
    assert kept == pytest.approx([0.25])


def test_class_aware_nms_suppresses_duplicates_of_one_class_only() -> None:
    heads = _heads()
    # Two nearly identical stride-8 boxes; the second has the higher score.
    _set(heads[0], 0, 0, tx=0.5, ty=0.5, log_w=math.log(2.0), log_h=math.log(2.0), objectness=10.0)
    _set(heads[0], 0, 1, tx=-0.4, ty=0.5, log_w=math.log(2.0), log_h=math.log(2.0))

    _boxes, scores, _classes = decode_yolox(heads, STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True)
    assert len(scores) == 1

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
    assert scores[0] == pytest.approx(120 / 121)


def test_empty_result_has_typed_empty_arrays() -> None:
    heads = _heads()

    boxes, scores, classes = decode_yolox(heads, STRIDES, conf_thr=0.35, iou_thr=0.45, logits=True)

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
        (_heads(sizes=((2, 2), (1, 1), (1, 1))), (8, 16, 0), 0.5, 0.45),
    ],
)
def test_invalid_inputs_raise(
    heads: object, strides: tuple[int, ...], conf_thr: float, iou_thr: float
) -> None:
    with pytest.raises(ValueError):
        decode_yolox(heads, strides, conf_thr=conf_thr, iou_thr=iou_thr, logits=True)


def test_heads_with_different_channel_counts_are_rejected() -> None:
    heads = _heads()
    mixed = _heads(classes=80)
    mixed[1] = np.zeros((1, 1, 5 + 79), dtype=np.float64)

    with pytest.raises(ValueError, match="channels"):
        decode_yolox(
            [heads[0], mixed[1], heads[2]], STRIDES, conf_thr=0.5, iou_thr=0.45, logits=True
        )
