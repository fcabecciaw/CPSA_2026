"""The DPU detector: graph wiring, source-pixel decode and the two slots (F-S1)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa import yolox
from cpsa.yolox import Detection

INPUT_DIMS = [1, 416, 416, 3]
OUTPUT_DIMS = {
    8: [1, 52, 52, 85],
    16: [1, 26, 26, 85],
    32: [1, 13, 13, 85],
}


# --------------------------------------------------------------------------- #
# Fake xir/vart, enough of the runtime surface for the detector.
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
        if sequence is None:
            payload = self._subgraph.payload
        else:
            # A real DPU writes into the caller's buffers when the job runs, so a
            # second job overwrites a shared output buffer: that is what the
            # double buffer has to prevent.
            payload = sequence[(self.executions - 1) % len(sequence)]
        assert len(outputs) == len(payload)
        for buffer, data in zip(outputs, payload, strict=True):
            np.copyto(buffer, data)
        # Like the VART Python binding: a (job_id, status) tuple, passed back to wait().
        return (self.executions, 0)

    def wait(self, job: tuple[int, int]) -> None:
        # Jobs complete independently: the detector may wait for N-1 after N ran.
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
    return np.clip(np.rint(dense * (2.0**fix_point)), -128.0, 127.0).astype(np.int8)


def _detector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    payloads: dict[int, NDArray[np.int8]] | None = None,
    output_order: tuple[int, ...] = (8, 16, 32),
    **kwargs: float,
) -> yolox.YoloxDetector:
    outputs = [
        _Tensor(f"out_{stride}", OUTPUT_DIMS[stride], fix_point=3) for stride in output_order
    ]
    subgraph = _Subgraph(
        "subgraph_dpu", "DPU", [_Tensor("input", INPUT_DIMS, fix_point=-2)], outputs
    )
    if payloads is not None:
        subgraph.payload = [payloads[stride] for stride in output_order]
    _install_fake_runtime(monkeypatch, [subgraph])
    return yolox.YoloxDetector(_model(tmp_path), **kwargs)


def test_the_detector_reads_the_graph_and_orders_outputs_by_stride(
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


def test_the_detector_returns_detections_in_source_pixels(
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


def test_the_detector_requires_the_model_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runtime(monkeypatch, [])

    with pytest.raises(FileNotFoundError):
        yolox.YoloxDetector(tmp_path / "missing.xmodel")


def test_the_detector_requires_a_dpu_subgraph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subgraph = _Subgraph("cpu", "USER", [], [])
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match="DPU subgraph"):
        yolox.YoloxDetector(_model(tmp_path))


@pytest.mark.parametrize(
    ("input_dims", "message"),
    [
        ([1, 416, 3, 416], "NHWC"),
        ([416, 416, 3], "4-D"),
        ([1, 416, 416, 4], "NHWC"),
    ],
)
def test_the_detector_rejects_unexpected_input_layouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_dims: list[int],
    message: str,
) -> None:
    outputs = [_Tensor(f"out_{stride}", OUTPUT_DIMS[stride], fix_point=3) for stride in (8, 16, 32)]
    subgraph = _Subgraph("dpu", "DPU", [_Tensor("input", input_dims, fix_point=-2)], outputs)
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match=message):
        yolox.YoloxDetector(_model(tmp_path))


def test_the_detector_rejects_non_yolox_output_strides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = [_Tensor("out_4", [1, 104, 104, 85], fix_point=3)]
    subgraph = _Subgraph("dpu", "DPU", [_Tensor("input", INPUT_DIMS, fix_point=-2)], outputs)
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match="strides"):
        yolox.YoloxDetector(_model(tmp_path))


@pytest.mark.parametrize("thresholds", [{"conf_thr": 1.5}, {"iou_thr": -0.1}])
def test_the_detector_rejects_bad_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, thresholds: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        _detector(monkeypatch, tmp_path, **thresholds)


# --------------------------------------------------------------------------- #
# Pipelined submit/wait (S5, 60 fps)
# --------------------------------------------------------------------------- #


def _payload(stride: int, class_id: int) -> NDArray[np.int8]:
    """One detection inside the letterboxed part of a 480x640 frame."""

    return _head(
        3,
        [(0, 0, {"tx": 2.5, "ty": 10.0, "objectness": 10.0, "class_id": class_id})],
        stride,
    )


def _pipelined_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[yolox.YoloxDetector, _Subgraph]:
    detector = _detector(monkeypatch, tmp_path)
    _Graph.pending[0].payloads_by_execution = [
        [
            _payload(8, 2),
            np.full(OUTPUT_DIMS[16], -128, np.int8),
            np.full(OUTPUT_DIMS[32], -128, np.int8),
        ],
        [
            _payload(8, 5),
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

    assert detector.pending_jobs == 0
    first = detector.submit(frame)
    assert detector.pending_jobs == 1
    second = detector.submit(frame)
    assert detector.pending_jobs == 2

    # Decode frame 1 after frame 2 has run: a single shared buffer would return
    # frame 2's class here.
    first_detections = detector.wait(first)
    assert detector.pending_jobs == 1
    second_detections = detector.wait(second)
    assert detector.pending_jobs == 0

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
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert detector.pending_jobs == 0
    assert [detection.class_id for detection in detections] == [2]
    assert set(detector.stage_ms) == {"preprocess", "dpu", "decode"}
    assert all(value >= 0.0 for value in detector.stage_ms.values())


def test_the_detector_decodes_the_same_detections_as_the_float_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = {
        8: _head(3, [(0, 0, {"tx": 2.5, "ty": 10.0, "log_w": math.log(2.0)})], 8),
        16: _head(3, [], 16),
        32: _head(3, [(0, 0, {"tx": 0.0, "objectness": 8.0, "class_id": 41})], 32),
    }
    detector = _detector(monkeypatch, tmp_path, payloads=outputs)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    _image, scale, pad = yolox.prepare_input(frame)
    expected = yolox.detections_from_heads(
        [outputs[stride].astype(np.float64) / 8.0 for stride in (8, 16, 32)],
        (8, 16, 32),
        conf_thr=0.35,
        iou_thr=0.45,
        scale=scale,
        pad=pad,
        source_hw=(480, 640),
    )
    assert detections == expected
    assert detector.last_candidate_count == 2


def test_a_zoomed_detector_maps_its_boxes_to_the_full_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = {
        8: _head(3, [(0, 0, {"tx": 2.5, "ty": 10.0, "log_w": math.log(2.0)})], 8),
        16: _head(3, [], 16),
        32: _head(3, [], 32),
    }
    detector = _detector(monkeypatch, tmp_path, payloads=outputs, zoom=1.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    crop = yolox.zoom_crop(640, 480, 1.5)
    view = frame[crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
    _image, scale, pad = yolox.prepare_input(view)
    expected = yolox.detections_from_heads(
        [outputs[stride].astype(np.float64) / 8.0 for stride in (8, 16, 32)],
        (8, 16, 32),
        conf_thr=0.35,
        iou_thr=0.45,
        scale=scale,
        pad=pad,
        source_hw=crop.size,
    )
    shifted = [
        Detection(
            box=(
                item.box[0] + crop.left,
                item.box[1] + crop.top,
                item.box[2] + crop.left,
                item.box[3] + crop.top,
            ),
            score=item.score,
            class_id=item.class_id,
        )
        for item in expected
    ]

    assert detections == shifted
    assert all(
        crop.left <= item.box[0] < item.box[2] <= crop.left + crop.width for item in detections
    )
    assert all(
        crop.top <= item.box[1] < item.box[3] <= crop.top + crop.height for item in detections
    )


@pytest.mark.parametrize("zoom", [0.9, 0.0, float("nan")])
def test_the_detector_rejects_a_zoom_below_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, zoom: float
) -> None:
    with pytest.raises(ValueError, match="zoom"):
        _detector(monkeypatch, tmp_path, zoom=zoom)
