"""YOLOX DPU adapter: preprocessing, LUT, head decode and runner wiring (F-S1)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.adapters.detect import yolox_dpu
from cpsa.adapters.pose.dpu import (
    input_quantization_lut,
    input_quantization_lut_scaled,
    quantize_fix_point,
)
from cpsa.vision.detection import Detection

INPUT_DIMS = [1, 416, 416, 3]
OUTPUT_DIMS = {
    8: [1, 52, 52, 85],
    16: [1, 26, 26, 85],
    32: [1, 13, 13, 85],
}


# --------------------------------------------------------------------------- #
# Fake xir/vart, enough of the runtime surface for the adapter.
# --------------------------------------------------------------------------- #


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

    def execute_async(self, inputs: list[NDArray[np.int8]], outputs: list[NDArray[np.int8]]) -> int:
        self.executions += 1
        assert len(outputs) == len(self._subgraph.payload)
        for buffer, payload in zip(outputs, self._subgraph.payload, strict=True):
            np.copyto(buffer, payload)
        return self.executions

    def wait(self, job_id: int) -> None:
        assert job_id == self.executions


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
    path = tmp_path / "yolox_nano.xmodel"
    path.write_bytes(b"fake xmodel")
    return path


def _cell(
    head: NDArray[np.float64],
    row: int,
    column: int,
    *,
    tx: float = 0.5,
    ty: float = 0.5,
    log_w: float = 0.0,
    log_h: float = 0.0,
    objectness: float = 20.0,
    class_id: int = 0,
) -> None:
    head[row, column, 0] = tx
    head[row, column, 1] = ty
    head[row, column, 2] = log_w
    head[row, column, 3] = log_h
    head[row, column, 4] = objectness
    head[row, column, 5:] = -1000.0
    head[row, column, 5 + class_id] = 20.0


def _head(
    fix_point: int, cells: list[tuple[int, int, dict[str, float]]], stride: int
) -> np.ndarray:
    """Int8 NHWC head with the given cells set, as the DPU returns it."""

    dims = OUTPUT_DIMS[stride]
    dense = np.full((dims[1], dims[2], dims[3]), -1000.0, dtype=np.float64)
    dense[..., :4] = 0.0
    for row, column, kwargs in cells:
        _cell(dense, row, column, **kwargs)
    quantized = np.clip(np.rint(dense * (2.0**fix_point)), -128.0, 127.0).astype(np.int8)
    return quantized


def _detector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    payloads: dict[int, NDArray[np.int8]] | None = None,
    output_order: tuple[int, ...] = (8, 16, 32),
    **kwargs: float,
) -> yolox_dpu.DpuYoloxDetector:
    outputs = [
        _Tensor(f"out_{stride}", OUTPUT_DIMS[stride], fix_point=3) for stride in output_order
    ]
    subgraph = _Subgraph(
        "subgraph_dpu", "DPU", [_Tensor("input", INPUT_DIMS, fix_point=-2)], outputs
    )
    if payloads is not None:
        subgraph.payload = [payloads[stride] for stride in output_order]
    _install_fake_runtime(monkeypatch, [subgraph])
    return yolox_dpu.DpuYoloxDetector(_model(tmp_path), **kwargs)


# --------------------------------------------------------------------------- #
# Preprocessing and geometry
# --------------------------------------------------------------------------- #


def test_prepare_input_converts_to_bgr_and_letterboxes_with_grey_padding() -> None:
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[..., 0] = 200  # red only: BGR input must carry it in channel 2

    bgr, scale, pad = yolox_dpu.prepare_input(rgb)

    assert bgr.shape == (416, 416, 3)
    assert bgr.flags["C_CONTIGUOUS"]
    assert scale == pytest.approx(416 / 200)
    assert pad == (0.0, pytest.approx((416 - round(100 * 416 / 200)) / 2))
    centre = bgr[208, 208]
    assert centre[2] == 200 and centre[0] == 0 and centre[1] == 0
    assert bgr[0, 0].tolist() == [yolox_dpu.YOLOX_PAD_VALUE] * 3


def test_prepare_input_rejects_bad_frames() -> None:
    with pytest.raises(ValueError):
        yolox_dpu.prepare_input(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        yolox_dpu.prepare_input(np.zeros((10, 10, 3), dtype=np.float32))


def test_boxes_to_source_inverts_the_letterbox_and_clips() -> None:
    boxes = np.asarray([[-10.0, -20.0, 100.0, 500.0]], dtype=np.float64)

    mapped = yolox_dpu.boxes_to_source(boxes, 0.5, (10.0, 20.0), (400, 300))

    assert mapped.tolist() == [[0.0, 0.0, 180.0, 400.0]]


def test_boxes_to_source_rejects_a_bad_scale() -> None:
    with pytest.raises(ValueError):
        yolox_dpu.boxes_to_source(np.zeros((0, 4)), 0.0, (0.0, 0.0), (10, 10))


# --------------------------------------------------------------------------- #
# Input quantization
# --------------------------------------------------------------------------- #


def test_scaled_lut_quantizes_raw_pixels() -> None:
    lut = input_quantization_lut_scaled(-2, 1.0)

    assert lut.shape == (256,)
    assert lut.dtype == np.int8
    assert lut[255] == 64
    assert lut.tolist() == quantize_fix_point(np.arange(256, dtype=np.float32), -2).tolist()


def test_scaled_lut_differs_from_the_normalized_pose_lut() -> None:
    assert input_quantization_lut_scaled(-2, 1.0)[255] == 64
    assert input_quantization_lut(-2)[255] == 0


def test_scaled_lut_rejects_a_bad_scale() -> None:
    with pytest.raises(ValueError):
        input_quantization_lut_scaled(-2, -1.0)
    with pytest.raises(ValueError):
        input_quantization_lut_scaled(-2, math.inf)


# --------------------------------------------------------------------------- #
# Head decode without the runner
# --------------------------------------------------------------------------- #


def test_detections_from_heads_maps_to_source_pixels_and_sorts() -> None:
    heads = [
        _head(3, [(0, 0, {"tx": 2.5, "ty": 2.5, "objectness": 5.0})], 8),
        _head(3, [], 16),
        _head(3, [(0, 0, {"tx": 0.5, "objectness": 10.0, "class_id": 41})], 32),
    ]

    detections = yolox_dpu.detections_from_heads(
        [head.astype(np.float64) / 8.0 for head in heads],
        (8, 16, 32),
        conf_thr=0.35,
        iou_thr=0.45,
        scale=0.5,
        pad=(8.0, 8.0),
        source_hw=(400, 400),
    )

    assert [detection.class_id for detection in detections] == [41, 0]
    assert detections[0].score > detections[1].score
    assert detections[0].box[0] == pytest.approx(0.0)
    assert detections[0].box[2] > detections[0].box[0]


def test_detections_from_heads_drops_boxes_without_area() -> None:
    head = np.zeros((2, 2, 85), dtype=np.float64)
    head[..., 4] = 20.0
    head[..., 5:] = -1000.0
    head[0, 0, 5] = 20.0
    head[0, 0, 2] = -1e6
    head[0, 0, 3] = -1e6

    detections = yolox_dpu.detections_from_heads(
        [head],
        (8,),
        conf_thr=0.35,
        iou_thr=0.45,
        scale=1.0,
        pad=(0.0, 0.0),
        source_hw=(100, 100),
    )

    assert detections == []


def test_detections_from_heads_simulates_the_int8_output_precision() -> None:
    dense = np.zeros((2, 2, 85), dtype=np.float64)
    dense[..., 4] = 20.0
    dense[..., 5:] = -1000.0
    dense[0, 0, 5] = 20.0
    head = np.rint(dense * 8.0) / 8.0

    detections = yolox_dpu.detections_from_heads(
        [head],
        (8,),
        conf_thr=0.35,
        iou_thr=0.45,
        scale=1.0,
        pad=(0.0, 0.0),
        source_hw=(100, 100),
        fix_points=(3,),
    )

    assert len(detections) == 1
    assert detections[0].score == pytest.approx(1.0)


def test_detections_from_heads_rejects_mismatched_fix_points() -> None:
    with pytest.raises(ValueError):
        yolox_dpu.detections_from_heads(
            [np.zeros((1, 1, 85))],
            (8,),
            conf_thr=0.35,
            iou_thr=0.45,
            scale=1.0,
            pad=(0.0, 0.0),
            source_hw=(10, 10),
            fix_points=(3, 3),
        )


# --------------------------------------------------------------------------- #
# Runner wiring
# --------------------------------------------------------------------------- #


def test_detector_reads_the_graph_and_orders_outputs_by_stride(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = _detector(
        monkeypatch,
        tmp_path,
        output_order=(32, 8, 16),
        payloads={
            32: _head(3, [(0, 0, {"tx": 0.0, "objectness": 10.0, "class_id": 2})], 32),
            8: _head(3, [], 8),
            16: _head(3, [], 16),
        },
    )

    assert detector.input_size == (416, 416)
    assert detector.strides == (8, 16, 32)
    assert set(detector.stage_ms) == {"preprocess", "dpu", "decode"}


def test_detector_returns_detections_in_source_pixels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = _detector(
        monkeypatch,
        tmp_path,
        payloads={
            8: _head(3, [(0, 0, {"tx": 2.5, "ty": 10.0, "log_w": math.log(2.0)})], 8),
            16: _head(3, [], 16),
            32: _head(3, [], 32),
        },
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    detection = detections[0]
    assert isinstance(detection, Detection)
    assert detection.class_id == 0
    assert detection.score == pytest.approx(1.0, abs=1e-3)
    x1, y1, x2, y2 = detection.box
    assert 0.0 <= x1 < x2 <= 640.0
    assert 0.0 <= y1 < y2 <= 480.0
    assert all(value >= 0.0 for value in detector.stage_ms.values())
    assert detector.stage_ms["dpu"] >= 0.0


def test_detector_requires_the_model_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_runtime(monkeypatch, [])

    with pytest.raises(FileNotFoundError):
        yolox_dpu.DpuYoloxDetector(tmp_path / "missing.xmodel")


def test_detector_requires_a_dpu_subgraph(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    subgraph = _Subgraph("cpu", "USER", [], [])
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match="DPU subgraph"):
        yolox_dpu.DpuYoloxDetector(_model(tmp_path))


@pytest.mark.parametrize(
    ("input_dims", "message"),
    [
        ([1, 416, 3, 416], "NHWC"),
        ([416, 416, 3], "4-D"),
        ([1, 416, 416, 4], "NHWC"),
    ],
)
def test_detector_rejects_unexpected_input_layouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_dims: list[int],
    message: str,
) -> None:
    outputs = [_Tensor(f"out_{stride}", OUTPUT_DIMS[stride], fix_point=3) for stride in (8, 16, 32)]
    subgraph = _Subgraph("dpu", "DPU", [_Tensor("input", input_dims, fix_point=-2)], outputs)
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match=message):
        yolox_dpu.DpuYoloxDetector(_model(tmp_path))


def test_detector_rejects_non_yolox_output_strides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = [_Tensor("out_4", [1, 104, 104, 85], fix_point=3)]
    subgraph = _Subgraph("dpu", "DPU", [_Tensor("input", INPUT_DIMS, fix_point=-2)], outputs)
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match="strides"):
        yolox_dpu.DpuYoloxDetector(_model(tmp_path))


@pytest.mark.parametrize("thresholds", [{"conf_thr": 1.5}, {"iou_thr": -0.1}])
def test_detector_rejects_bad_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, thresholds: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        _detector(monkeypatch, tmp_path, **thresholds)
