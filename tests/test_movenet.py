"""MoveNet runtime: preprocessing, the zoo decode and the two DPU slots (F-E2)."""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa import movenet
from cpsa.dpu import OutputTensor, input_quantization_lut_scaled
from cpsa.movenet import (
    KEYPOINTS,
    MOVENET_INPUT_SCALE,
    MOVENET_INPUT_SIZE,
    MovenetHeads,
    MovenetPose,
)

INPUT_DIMS = [1, 192, 192, 3]
HEAD_DIMS = {
    "heatmaps": [1, 48, 48, 17],
    "centers": [1, 48, 48, 1],
    "regs": [1, 48, 48, 34],
    "offsets": [1, 48, 48, 34],
}
FIX_POINTS = {"heatmaps": 3, "centers": 3, "regs": 2, "offsets": 7}
#: Output order of the compiled graph (``inspect_xmodel.py``).
OUTPUT_ORDER = ("offsets", "regs", "centers", "heatmaps")


def _lut() -> NDArray[np.int8]:
    return input_quantization_lut_scaled(6, MOVENET_INPUT_SCALE)


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
        self.payloads: list[list[NDArray[np.int8]]] = []
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

    def execute_async(self, inputs: list[NDArray[np.int8]], outputs: list[NDArray[np.int8]]) -> int:
        self.executions += 1
        sequence = self._subgraph.payloads_by_execution
        if sequence is not None:
            # A real DPU writes into the caller's buffers when the job runs, so a
            # second job overwrites a shared output buffer: that is what the
            # double buffer has to prevent.
            payload = sequence[(self.executions - 1) % len(sequence)]
        else:
            payload = self._subgraph.payloads
        assert len(outputs) == len(payload)
        for buffer, data in zip(outputs, payload, strict=True):
            np.copyto(buffer, data)
        return self.executions

    def wait(self, job_id: int) -> None:
        assert 0 < job_id <= self.executions


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, subgraphs: list[_Subgraph]) -> None:
    xir = ModuleType("xir")
    xir.Graph = _Graph  # type: ignore[attr-defined]
    vart = ModuleType("vart")
    vart.Runner = SimpleNamespace(create_runner=lambda subgraph, mode: _Runner(subgraph))
    _Graph.pending = subgraphs
    monkeypatch.setitem(sys.modules, "xir", xir)
    monkeypatch.setitem(sys.modules, "vart", vart)


def _model(tmp_path: Path) -> Path:
    path = tmp_path / "movenet.xmodel"
    path.write_bytes(b"fake xmodel")
    return path


# --------------------------------------------------------------------------- #
# Synthetic heads
# --------------------------------------------------------------------------- #


def _logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def _heads(
    *,
    center: tuple[int, int] = (24, 24),
    center_probability: float = 0.9,
    centers: dict[tuple[int, int], float] | None = None,
    peaks: dict[int, tuple[int, int, float]] | None = None,
    regs: dict[int, tuple[float, float]] | None = None,
    offsets: dict[int, tuple[float, float]] | None = None,
) -> MovenetHeads:
    """Float heads in the model domain; probabilities are pre-sigmoid logits.

    Without ``peaks`` every keypoint peaks on the person-center cell, which is
    what the zoo decode does for an upright subject. Passing ``peaks`` replaces
    that default, so a cell that is not listed stays below the threshold.
    ``centers`` replaces the single center cell with a map of them, which is how
    a two-person scene is built.
    """

    feature = movenet.MOVENET_FEATURE_SIZE
    if peaks is None:
        peaks = {
            index: (center[1], center[0], center_probability) for index in range(len(KEYPOINTS))
        }
    if centers is None:
        centers = {center: center_probability}
    heatmaps = np.full((17, feature, feature), _logit(0.001))
    for index, (row, column, probability) in peaks.items():
        heatmaps[index, row, column] = _logit(probability)
    center_head = np.full((1, feature, feature), _logit(0.001))
    for (column, row), probability in centers.items():
        center_head[0, row, column] = _logit(probability)
    regression = np.zeros((34, feature, feature))
    for index, (dx, dy) in (regs or {}).items():
        regression[index * 2, center[1], center[0]] = dx
        regression[index * 2 + 1, center[1], center[0]] = dy
    offset = np.zeros((34, feature, feature))
    for index, (dx, dy) in (offsets or {}).items():
        offset[index * 2, :, :] = dx
        offset[index * 2 + 1, :, :] = dy
    return MovenetHeads(
        heatmaps=heatmaps,
        centers=center_head,
        regs=regression,
        offsets=offset,
    )


def _buffers(
    peaks: dict[int, tuple[int, int, float]] | None = None,
    *,
    centers: dict[tuple[int, int], float] | None = None,
) -> dict[str, np.ndarray]:
    """Int8 DPU buffers, one per head, with a center at cell (24, 24).

    As in :func:`_heads`, ``peaks=None`` puts every keypoint on the center
    cell and an explicit mapping replaces that default. ``centers`` gives the
    center head one peak per person, keyed by ``(column, row)``.
    """

    feature = movenet.MOVENET_FEATURE_SIZE
    if peaks is None:
        peaks = {index: (24, 24, 0.9) for index in range(len(KEYPOINTS))}
    dense = {
        "heatmaps": np.full((feature, feature, 17), _logit(0.001)),
        "centers": np.full((feature, feature, 1), _logit(0.001)),
        "regs": np.zeros((feature, feature, 34)),
        "offsets": np.zeros((feature, feature, 34)),
    }
    for (column, row), probability in (centers or {(24, 24): 0.9}).items():
        dense["centers"][row, column, 0] = _logit(probability)
    for index, (row, column, probability) in peaks.items():
        dense["heatmaps"][row, column, index] = _logit(probability)
    return _quantize_dense(dense)


def _quantize_dense(dense: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Quantize NHWC float heads at their fix points, as the DPU buffers hold them."""
    return {
        name: np.clip(np.rint(head * (2.0 ** FIX_POINTS[name])), -128.0, 127.0).astype(np.int8)[
            None
        ]
        for name, head in dense.items()
    }


def _buffers_of(heads: MovenetHeads) -> dict[str, np.ndarray]:
    """The int8 buffers a DPU would write for these float heads."""
    return _quantize_dense(
        {
            "heatmaps": np.moveaxis(heads.heatmaps, 0, -1),
            "centers": np.moveaxis(heads.centers, 0, -1),
            "regs": np.moveaxis(heads.regs, 0, -1),
            "offsets": np.moveaxis(heads.offsets, 0, -1),
        }
    )


def _head_tensors() -> list[OutputTensor]:
    return [
        OutputTensor(
            name=f"head_{name}",
            shape=(48, 48, HEAD_DIMS[name][3]),
            fix_point=FIX_POINTS[name],
            stride=4,
        )
        for name in OUTPUT_ORDER
    ]


def _timed_ms(function: Callable[[], object]) -> float:
    """Return how long one call of ``function`` took, in milliseconds."""

    started = time.perf_counter()
    function()
    return (time.perf_counter() - started) * 1000.0


def _interleaved_ms(
    fast: Callable[[], object], reference: Callable[[], object], runs: int = 40
) -> tuple[float, float]:
    """Time two functions alternately, after a warm-up, and return both medians.

    Alternating the calls is what makes the comparison fair: both see the same
    CPU frequency and allocator state, so a slow machine cannot pass or fail the
    ratio by itself. The warm-up matters on a laptop, where the first calls run
    at a lower clock.
    """

    for _ in range(25):
        fast()
        reference()
    fast_ms = [_timed_ms(fast) for _ in range(runs)]
    reference_ms = [_timed_ms(reference) for _ in range(runs)]
    return statistics.median(fast_ms), statistics.median(reference_ms)


def _payload(peaks: dict[int, tuple[int, int, float]] | None = None) -> list[NDArray[np.int8]]:
    buffers = _buffers(peaks)
    return [buffers[name] for name in OUTPUT_ORDER]


def _detector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> movenet.DpuMovenet:
    outputs = [
        _Tensor(f"subgraph_head_{name}_fix", HEAD_DIMS[name], fix_point=FIX_POINTS[name])
        for name in OUTPUT_ORDER
    ]
    subgraph = _Subgraph(
        "subgraph_dpu", "DPU", [_Tensor("input_0_fix", INPUT_DIMS, fix_point=6)], outputs
    )
    subgraph.payloads = _payload()
    _install_fake_runtime(monkeypatch, [subgraph])
    return movenet.DpuMovenet(_model(tmp_path))


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_preprocess_resizes_to_the_model_input_and_keeps_rgb_order() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., 0] = 200

    model_input = movenet.preprocess(frame)

    assert model_input.shape == (*MOVENET_INPUT_SIZE, 3)
    assert model_input.dtype == np.uint8
    assert model_input.flags["C_CONTIGUOUS"]
    # RGB in, RGB out: the zoo graph does not swap channels.
    assert int(model_input[..., 0].min()) == 200
    assert int(model_input[..., 1].max()) == 0


def test_preprocess_accepts_a_frame_that_is_already_the_model_size() -> None:
    frame = np.zeros((192, 192, 3), dtype=np.uint8)

    model_input = movenet.preprocess(frame)

    assert model_input.shape == (192, 192, 3)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (np.zeros((192, 192), dtype=np.uint8), "H, W, 3"),
        (np.zeros((192, 192, 4), dtype=np.uint8), "H, W, 3"),
        (np.zeros((192, 192, 3), dtype=np.float32), "uint8"),
        (np.zeros((0, 192, 3), dtype=np.uint8), "empty"),
    ],
)
def test_preprocess_rejects_unusable_frames(frame: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        movenet.preprocess(frame)


def test_preprocess_into_quantizes_with_the_model_input_scale() -> None:
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    out = np.zeros((*MOVENET_INPUT_SIZE, 3), dtype=np.int8)

    movenet.preprocess_into(frame, _lut(), out)

    resized = movenet.preprocess(frame)
    # The xint8 grid covers [-2, 2), so the host feeds pixel / 128: int8 = pixel / 2.
    expected = np.clip(np.rint(resized.astype(np.float64) / 2.0), -128.0, 127.0)
    np.testing.assert_array_equal(out, expected.astype(np.int8))
    assert int(out.min()) >= 0

    movenet.preprocess_into(np.full((192, 192, 3), 255, dtype=np.uint8), _lut(), out)
    # 255 / 128 rounds to 2.0, the last cell of the grid: the int8 maximum.
    assert int(out.min()) == 127


@pytest.mark.parametrize(
    ("out", "message"),
    [
        (np.zeros((192, 192, 3), dtype=np.uint8), "int8"),
        (np.zeros((192, 192, 1), dtype=np.int8), "int8 array"),
    ],
)
def test_preprocess_into_rejects_a_bad_destination(out: np.ndarray, message: str) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        movenet.preprocess_into(frame, _lut(), out)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def test_decode_places_a_keypoint_on_the_center_cell() -> None:
    pose = movenet.decode(_heads())

    assert isinstance(pose, MovenetPose)
    assert pose.keypoints.shape == (len(KEYPOINTS), 3)
    assert pose.keypoints.dtype == np.float32
    # Cell (24, 24) at stride 4 is pixel (96, 96) on the 192 input.
    np.testing.assert_allclose(pose.keypoints[:, :2], [[96.0, 96.0]] * len(KEYPOINTS))
    assert pose.score == pytest.approx(0.9, abs=1e-3)


def test_decode_maps_normalised_keypoints_onto_the_source_frame() -> None:
    pose = movenet.decode(_heads(), size=(480, 640))

    np.testing.assert_allclose(pose.keypoints[:, 0], [320.0] * len(KEYPOINTS), rtol=1e-6)
    np.testing.assert_allclose(pose.keypoints[:, 1], [240.0] * len(KEYPOINTS), rtol=1e-6)


def test_decode_applies_the_offset_field() -> None:
    pose = movenet.decode(_heads(offsets={0: (0.5, -0.25)}))

    assert pose.keypoints[0, 0] == pytest.approx(98.0)
    assert pose.keypoints[0, 1] == pytest.approx(95.0)


def test_decode_follows_the_regression_field_to_the_keypoint_peak() -> None:
    heads = _heads(
        peaks={0: (10, 30, 0.8), 1: (24, 24, 0.7)},
        regs={0: (6.0, -14.0)},
    )

    pose = movenet.decode(heads)

    # Keypoint 0 regresses from cell (24, 24) to (30, 10), where its own peak is.
    assert pose.keypoints[0, 0] == pytest.approx(30 * 4.0)
    assert pose.keypoints[0, 1] == pytest.approx(10 * 4.0)
    assert pose.keypoints[0, 2] == pytest.approx(0.8, abs=1e-3)
    # Keypoint 1 regresses nowhere and finds the heatmap peak on its own cell.
    assert pose.keypoints[1, 2] == pytest.approx(0.7, abs=1e-3)


def test_decode_drops_keypoints_below_the_heatmap_threshold() -> None:
    pose = movenet.decode(_heads(peaks={0: (30, 10, 0.05), 1: (30, 10, 0.6)}, regs={}))

    assert pose.keypoints[0, 2] == 0.0
    assert pose.keypoints[1, 2] == pytest.approx(0.6, abs=1e-3)
    assert pose.score == pytest.approx(0.6 / len(KEYPOINTS), abs=1e-4)


def test_decode_weights_the_center_head_by_the_center_prior() -> None:
    # A corner peak is far more confident but the zoo's prior keeps the person
    # in the middle of the frame, so the center cell wins.
    heads = _heads(center=(24, 24), center_probability=0.5, peaks={0: (24, 24, 0.9)})
    heads.centers[0, 0, 0] = _logit(0.99)  # top-left corner

    pose = movenet.decode(heads)

    assert pose.keypoints[0, 0] == pytest.approx(96.0)
    assert pose.keypoints[0, 1] == pytest.approx(96.0)


def test_the_center_prior_is_the_package_weight_table() -> None:
    prior = movenet.center_prior()

    assert prior.shape == (48, 48)
    assert prior[24, 24] == pytest.approx(1.0 / 1.8)
    assert prior[0, 0] == pytest.approx(1.0 / (np.sqrt(2.0) * 24.0 + 1.8))
    assert prior[24, 47] == pytest.approx(1.0 / (23.0 + 1.8))


@pytest.mark.parametrize(
    ("heads", "message"),
    [
        ({"centers": np.zeros((2, 48, 48))}, "centers"),
        ({"regs": np.zeros((33, 48, 48))}, "regs"),
        ({"heatmaps": np.zeros((16, 48, 48))}, "heatmaps"),
        ({"heatmaps": np.zeros((17, 24, 24))}, "48x48"),
    ],
)
def test_movenet_heads_reject_bad_shapes(
    heads: dict[str, NDArray[np.float64]], message: str
) -> None:
    full = {
        "heatmaps": np.zeros((17, 48, 48)),
        "centers": np.zeros((1, 48, 48)),
        "regs": np.zeros((34, 48, 48)),
        "offsets": np.zeros((34, 48, 48)),
    }
    full.update(heads)

    with pytest.raises(ValueError, match=message):
        MovenetHeads(**full)


def test_decode_rejects_a_bad_source_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        movenet.decode(_heads(), size=(0, 640))


def test_dequantize_heads_applies_the_head_fix_points() -> None:
    buffers = _buffers()
    tensors = [
        OutputTensor(
            name=f"head_{name}",
            shape=(48, 48, HEAD_DIMS[name][3]),
            fix_point=FIX_POINTS[name],
            stride=4,
        )
        for name in OUTPUT_ORDER
    ]

    ordered = [buffers[name] for name in OUTPUT_ORDER]
    heads = movenet.dequantize_heads(ordered, tensors, OUTPUT_ORDER)

    assert heads.heatmaps.shape == (17, 48, 48)
    assert heads.centers.shape == (1, 48, 48)
    assert heads.regs.shape == (34, 48, 48)
    assert heads.offsets.shape == (34, 48, 48)
    expected = float(buffers["centers"][0, 24, 24, 0]) / 2.0 ** FIX_POINTS["centers"]
    assert heads.centers[0, 24, 24] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("order", "message"),
    [
        (("heatmaps", "centers", "regs", "heatmaps"), "permutation"),
        (("heatmaps", "centers", "regs"), "buffers"),
    ],
)
def test_dequantize_heads_rejects_a_bad_order(order: tuple[str, ...], message: str) -> None:
    buffers = _buffers()
    tensors = [
        OutputTensor(
            name=f"head_{name}",
            shape=(48, 48, HEAD_DIMS[name][3]),
            fix_point=FIX_POINTS[name],
            stride=4,
        )
        for name in OUTPUT_ORDER
    ]

    with pytest.raises(ValueError, match=message):
        movenet.dequantize_heads([buffers[name] for name in OUTPUT_ORDER], tensors, order=order)


# --------------------------------------------------------------------------- #
# The int8 decoder (the A53 path of R2-G); the float64 ``decode`` is the oracle
# --------------------------------------------------------------------------- #


def test_the_int8_decode_matches_the_reference_decoder() -> None:
    heads = _heads(
        center=(20, 18),
        peaks={
            0: (10, 30, 0.8),
            1: (24, 24, 0.7),
            2: (30, 10, 0.6),
            3: (12, 12, 0.55),
            4: (36, 40, 0.95),
        },
        regs={0: (6.0, -14.0), 2: (-2.0, 4.0), 4: (1.0, 1.0)},
        offsets={0: (0.5, -0.25), 3: (0.125, 0.375), 4: (-0.75, 0.625), 9: (0.25, -0.5)},
    )
    # A second candidate near keypoint 0's regression target, and a weak cell
    # above the threshold away from it: the weighted peak search must agree.
    heads.heatmaps[0, 21, 18] = _logit(0.4)
    heads.heatmaps[0, 40, 5] = _logit(0.15)
    buffers = _buffers_of(heads)
    tensors = _head_tensors()
    ordered = [buffers[name] for name in OUTPUT_ORDER]

    reference = movenet.decode(
        movenet.dequantize_heads(ordered, tensors, OUTPUT_ORDER), size=(480, 640)
    )
    fast = movenet.decode_buffers(ordered, tensors, size=(480, 640), order=OUTPUT_ORDER)

    np.testing.assert_allclose(fast.keypoints[:, :2], reference.keypoints[:, :2], atol=0.5)
    np.testing.assert_allclose(fast.keypoints[:, 2], reference.keypoints[:, 2], atol=1e-3)
    assert fast.score == pytest.approx(reference.score, abs=1e-3)
    # Keypoint 9 is below the threshold, so both decoders score it zero and
    # read the offset of cell (0, 0): the coordinates match, the score is 0.
    assert fast.keypoints[9, 2] == 0.0
    np.testing.assert_allclose(fast.keypoints[9, :2], reference.keypoints[9, :2], atol=0.5)


def test_the_int8_decode_keeps_the_reference_threshold_and_prior() -> None:
    heads = _heads(peaks={0: (40, 40, 0.09), 1: (24, 24, 0.9)})
    buffers = _buffers_of(heads)
    tensors = _head_tensors()
    ordered = [buffers[name] for name in OUTPUT_ORDER]

    pose = movenet.decode_buffers(ordered, tensors, size=(480, 640), order=OUTPUT_ORDER)

    assert pose.keypoints[0, 2] == 0.0
    assert pose.keypoints[1, 2] == pytest.approx(0.9, abs=0.01)
    assert pose.keypoints[1, 0] == pytest.approx(320.0, abs=0.5)


@pytest.mark.benchmark
def test_the_int8_decode_is_at_least_ten_times_faster_than_the_reference() -> None:
    """R2-G: the A53 decode is what buys the MoveNet view its frame rate.

    Host micro-benchmark on the same int8 buffers: the reference is the path
    the board ran before (``dequantize_heads`` to ``float64`` plus the zoo
    ``decode``, 14.5 ms on the A53) and the fast path is
    :func:`~cpsa.movenet.decode_quantized`, the int8 decode the runner calls
    with the head views of its DPU slot. Interleaved medians, so a slower host
    or one scheduler hiccup cannot fail the ratio by itself; the board numbers
    are what ``scripts/profile_showcase.py`` prints.
    """

    buffers = _buffers(
        peaks={
            0: (12, 30, 0.80),
            1: (24, 24, 0.70),
            2: (30, 10, 0.60),
            5: (20, 18, 0.95),
            6: (22, 17, 0.75),
        }
    )
    tensors = _head_tensors()
    ordered = [buffers[name] for name in OUTPUT_ORDER]

    def reference() -> MovenetPose:
        return movenet.decode(
            movenet.dequantize_heads(ordered, tensors, OUTPUT_ORDER), size=(480, 640)
        )

    # The pipeline decodes the same four buffers every frame, so the runner
    # builds the int8 views of one slot once (:class:`~cpsa.movenet.DpuMovenet`)
    # and the frame pays the decode alone.
    heads = movenet.quantized_heads(ordered, tensors, OUTPUT_ORDER)

    def fast() -> MovenetPose:
        return movenet.decode_quantized(heads, size=(480, 640))

    fast_ms, reference_ms = _interleaved_ms(fast, reference)

    assert fast_ms > 0.0
    assert reference_ms / fast_ms >= 10.0, (
        f"the int8 decode is {reference_ms / fast_ms:.1f}x the reference "
        f"({fast_ms:.3f} ms vs {reference_ms:.3f} ms)"
    )


def test_quantized_heads_rejects_a_bad_input() -> None:
    buffers = _buffers()
    tensors = _head_tensors()
    ordered = [buffers[name] for name in OUTPUT_ORDER]
    with pytest.raises(ValueError, match="permutation"):
        movenet.quantized_heads(ordered, tensors, ("heatmaps", "centers", "regs", "heatmaps"))
    with pytest.raises(ValueError, match="buffers"):
        movenet.quantized_heads(ordered[:3], tensors[:3], OUTPUT_ORDER)
    with pytest.raises(ValueError, match="int8"):
        movenet.quantized_heads(
            [buffer.astype(np.float32) for buffer in ordered], tensors, OUTPUT_ORDER
        )
    with pytest.raises(ValueError, match="shape"):
        movenet.quantized_heads(
            [ordered[0], ordered[1][..., 0], ordered[2], ordered[3]],
            tensors,
            OUTPUT_ORDER,
        )


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #


def test_the_detector_reads_the_graph_and_orders_its_heads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = _detector(monkeypatch, tmp_path)

    assert detector.input_size == MOVENET_INPUT_SIZE
    assert set(detector.stage_ms) == {"preprocess", "dpu", "decode", "lock"}
    assert detector.pending_jobs == 0


def test_the_detector_returns_a_pose_in_source_pixels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = _detector(monkeypatch, tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    pose = detector.detect(frame)

    assert isinstance(pose, MovenetPose)
    assert pose.keypoints.shape == (17, 3)
    assert pose.keypoints[0, 0] == pytest.approx(320.0)
    assert pose.keypoints[0, 1] == pytest.approx(240.0)
    assert 0.0 <= pose.score <= 1.0
    assert all(value >= 0.0 for value in detector.stage_ms.values())


def test_the_detector_keeps_two_frames_in_flight_without_overwriting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = _detector(monkeypatch, tmp_path)
    subgraph = _Graph.pending[0]
    # Frame 1 has the nose on the center cell, frame 2 moves it to another cell.
    subgraph.payloads_by_execution = [
        _payload({0: (24, 24, 0.9)}),
        _payload({0: (10, 30, 0.9)}),
    ]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    first = detector.submit(frame)
    second = detector.submit(frame)
    assert detector.pending_jobs == 2

    first_pose = detector.wait(first)
    second_pose = detector.wait(second)

    assert detector.pending_jobs == 0
    assert first_pose.keypoints[0, 0] == pytest.approx(320.0)
    assert first_pose.keypoints[0, 1] == pytest.approx(240.0)
    assert second_pose.keypoints[0, 0] == pytest.approx(400.0)
    assert second_pose.keypoints[0, 1] == pytest.approx(100.0)
    # The int8 head stores the logit at fix_point 3, so the score is close, not exact.
    assert second_pose.keypoints[0, 2] == pytest.approx(0.9, abs=0.01)


def test_a_third_submit_requires_a_wait(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    detector = _detector(monkeypatch, tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector.submit(frame)
    detector.submit(frame)

    with pytest.raises(RuntimeError, match="busy"):
        detector.submit(frame)


def test_wait_rejects_an_unknown_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    detector = _detector(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="token"):
        detector.wait(1234)


def test_close_releases_the_graph_and_the_buffers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = _detector(monkeypatch, tmp_path)
    detector.close()

    with pytest.raises(RuntimeError, match="closed"):
        detector.submit(np.zeros((480, 640, 3), dtype=np.uint8))
    detector.close()  # releasing twice is harmless: the loop may race a stop


def test_the_detector_requires_the_model_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runtime(monkeypatch, [])

    with pytest.raises(FileNotFoundError):
        movenet.DpuMovenet(tmp_path / "missing.xmodel")


def test_the_detector_rejects_a_graph_that_is_not_movenet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = [_Tensor("out", [1, 52, 52, 85], fix_point=3)]
    subgraph = _Subgraph("dpu", "DPU", [_Tensor("input", [1, 416, 416, 3], fix_point=-2)], outputs)
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match="192"):
        movenet.DpuMovenet(_model(tmp_path))


def test_the_detector_rejects_unknown_head_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = [
        _Tensor(f"subgraph_out_{index}", HEAD_DIMS[name], fix_point=FIX_POINTS[name])
        for index, name in enumerate(OUTPUT_ORDER)
    ]
    subgraph = _Subgraph("dpu", "DPU", [_Tensor("input", INPUT_DIMS, fix_point=6)], outputs)
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match="which MoveNet head"):
        movenet.DpuMovenet(_model(tmp_path))


def test_the_detector_rejects_a_head_with_the_wrong_channel_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dims = dict(HEAD_DIMS)
    dims["centers"] = [1, 48, 48, 2]
    outputs = [
        _Tensor(f"head_{name}", dims[name], fix_point=FIX_POINTS[name]) for name in OUTPUT_ORDER
    ]
    subgraph = _Subgraph("dpu", "DPU", [_Tensor("input", INPUT_DIMS, fix_point=6)], outputs)
    _install_fake_runtime(monkeypatch, [subgraph])

    with pytest.raises(ValueError, match="centers head must have 1 channels"):
        movenet.DpuMovenet(_model(tmp_path))


# --------------------------------------------------------------------------- #
# The person lock: the square crop and the centre prior that keep the
# single-person graph on the body it is already tracking.
# --------------------------------------------------------------------------- #

#: Seventeen offsets, so a synthetic pose always has the 17 COCO keypoints.
_POSE_OFFSETS = (
    (-1.0, -1.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (-1.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (-0.5, -0.5),
    (0.5, -0.5),
    (-0.5, 0.5),
    (0.5, 0.5),
    (-0.25, 0.0),
    (0.25, 0.0),
    (0.0, -0.25),
    (0.0, 0.25),
)


def _pose_around(
    x: float,
    y: float,
    *,
    score: float = 0.9,
    half_extent: float = 20.0,
) -> MovenetPose:
    """A synthetic pose whose confident keypoints span a square around ``(x, y)``.

    ``half_extent`` is how far the outermost keypoint sits from the centre, so
    the body extent is ``2 * half_extent`` and :func:`cpsa.movenet.crop_region`
    takes a crop of ``3 * half_extent`` when the frame fits it.
    """

    keypoints = np.array(
        [(x + dx * half_extent, y + dy * half_extent, score) for dx, dy in _POSE_OFFSETS],
        dtype=np.float32,
    )
    return MovenetPose(keypoints=keypoints, score=score)


def _absent_pose() -> MovenetPose:
    """A pose with every keypoint under the crop score: the person is gone."""

    return MovenetPose(keypoints=np.zeros((len(KEYPOINTS), 3), dtype=np.float32), score=0.0)


def _median_ms(function: Callable[[], object], runs: int = 40) -> float:
    """Return the median wall clock of ``function``, after a warm-up."""

    for _ in range(25):
        function()
    return statistics.median([_timed_ms(function) for _ in range(runs)])


def test_crop_region_frames_the_body_with_a_margin() -> None:
    crop = movenet.crop_region(_pose_around(320.0, 240.0).keypoints[:, :2], (480, 640))

    assert crop is not None
    # 40 px of body times CROP_MARGIN, centred on the body.
    assert crop.side == int(np.ceil(40 * movenet.CROP_MARGIN))
    assert crop.x == pytest.approx(320 - crop.side / 2, abs=1)
    assert crop.y == pytest.approx(240 - crop.side / 2, abs=1)


def test_crop_region_stays_inside_the_frame() -> None:
    crop = movenet.crop_region(_pose_around(5.0, 475.0).keypoints[:, :2], (480, 640))

    assert crop is not None
    assert crop.x == 0
    assert crop.y + crop.side == 480


def test_crop_region_returns_none_for_a_degenerate_pose() -> None:
    sparse = _pose_around(320.0, 240.0).keypoints[:3, :2]
    too_large = _pose_around(320.0, 240.0, half_extent=200.0).keypoints[:, :2]
    too_small = _pose_around(320.0, 240.0, half_extent=1.0).keypoints[:, :2]

    assert movenet.crop_region(sparse, (480, 640)) is None
    assert movenet.crop_region(too_large, (480, 640)) is None
    assert movenet.crop_region(too_small, (480, 640)) is None
    assert movenet.crop_region(np.zeros((0, 2)), (480, 640)) is None


def test_preprocess_into_takes_the_crop_out_of_the_frame() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., 0] = 40
    crop = movenet.Crop(x=100, y=80, side=64)
    frame[crop.y : crop.y + crop.side, crop.x : crop.x + crop.side] = (200, 30, 10)
    lut = _lut()
    out = np.zeros((*MOVENET_INPUT_SIZE, 3), dtype=np.int8)

    movenet.preprocess_into(frame, lut, out, crop)

    roi = frame[crop.y : crop.y + crop.side, crop.x : crop.x + crop.side]
    np.testing.assert_array_equal(out, lut[movenet.preprocess(roi)])
    full = np.zeros_like(out)
    movenet.preprocess_into(frame, lut, full)
    assert not np.array_equal(out, full), "the crop must cut the rest of the frame away"


def test_preprocess_into_rejects_a_crop_outside_the_frame() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = np.zeros((*MOVENET_INPUT_SIZE, 3), dtype=np.int8)

    with pytest.raises(ValueError, match="does not fit"):
        movenet.preprocess_into(frame, _lut(), out, movenet.Crop(x=600, y=0, side=64))
    with pytest.raises(ValueError, match="does not fit"):
        movenet.preprocess_into(frame, _lut(), out, movenet.Crop(x=0, y=460, side=64))


def test_the_int8_decode_maps_a_crop_back_onto_the_frame() -> None:
    buffers = _buffers()  # every keypoint on the center cell, no offsets
    tensors = _head_tensors()
    ordered = [buffers[name] for name in OUTPUT_ORDER]
    heads = movenet.quantized_heads(ordered, tensors, OUTPUT_ORDER)
    # 2:1 x and 3:2 y: the crop's centre is the frame's centre, so a pose there
    # must come back to exactly the pixel the full frame gives.
    crop = movenet.Crop(x=160, y=80, side=320)

    cropped = movenet.decode_quantized(heads, size=(crop.side, crop.side), origin=(crop.x, crop.y))
    full = movenet.decode_quantized(heads, size=(480, 640))

    np.testing.assert_allclose(cropped.keypoints[:, :2], full.keypoints[:, :2], atol=1e-3)
    np.testing.assert_allclose(cropped.keypoints[:, 2], full.keypoints[:, 2])

    reference = movenet.decode(
        movenet.dequantize_heads(ordered, tensors, OUTPUT_ORDER),
        size=(crop.side, crop.side),
        origin=(crop.x, crop.y),
    )
    np.testing.assert_allclose(reference.keypoints[:, :2], full.keypoints[:, :2], atol=1e-4)

    # A crop that is not the frame's centre lands somewhere else: this is what
    # maps the crop's keypoints back onto the frame.
    corner = movenet.Crop(x=0, y=0, side=320)
    moved = movenet.decode_quantized(heads, size=(corner.side, corner.side))
    assert moved.keypoints[0, 0] == pytest.approx(160.0, abs=1e-3)
    assert moved.keypoints[0, 1] == pytest.approx(160.0, abs=1e-3)


def test_the_lock_prior_follows_the_tracked_person() -> None:
    lock = movenet.PersonLock()
    lock.observe(_pose_around(480.0, 300.0), 0.0, (480, 640))

    assert lock.locked
    assert lock.center == pytest.approx((480.0, 300.0), abs=1)
    # On the whole frame the tracked body sits at (36, 30) in feature cells,
    # where the image-centre prior would put (24, 24): that is the jump the
    # lock removes.
    whole = lock.prior(None, (480, 640))
    row, column = np.unravel_index(int(np.argmax(whole)), whole.shape)
    assert (column, row) == (
        pytest.approx(480 / 640 * 48, abs=1),
        pytest.approx(300 / 480 * 48, abs=1),
    )
    # The crop is centred on the body, so inside the crop the prior peaks on
    # the crop's centre cell.
    assert lock.crop is not None
    inside = lock.prior(lock.crop, (480, 640))
    row, column = np.unravel_index(int(np.argmax(inside)), inside.shape)
    assert (column, row) == (24, 24)


def test_the_lock_uses_the_image_prior_until_a_person_is_tracked() -> None:
    lock = movenet.PersonLock()

    prior = lock.prior(None, (480, 640))

    np.testing.assert_array_equal(prior, movenet.center_prior().astype(np.float32))
    assert not lock.locked
    assert lock.crop is None


def test_the_lock_keeps_the_whole_frame_for_a_body_too_large_to_crop() -> None:
    lock = movenet.PersonLock()

    lock.observe(_pose_around(133.0, 240.0, half_extent=200.0), 0.0, (480, 640))

    assert lock.locked
    assert lock.crop is None


def test_the_lock_holds_through_a_short_dropout_and_lets_go_after_leaving() -> None:
    lock = movenet.PersonLock()
    lock.observe(_pose_around(100.0, 200.0), 0.0, (480, 640))
    crop = lock.crop
    assert crop is not None

    lock.observe(_absent_pose(), 0.2, (480, 640))
    assert lock.locked and lock.crop == crop
    lock.observe(_absent_pose(), 0.2 + movenet.LEAVE_AFTER_S - 0.1, (480, 640))
    assert lock.locked and lock.crop == crop

    lock.observe(_absent_pose(), 0.2 + movenet.LEAVE_AFTER_S + 0.1, (480, 640))
    assert not lock.locked
    assert lock.crop is None


def test_the_lock_never_switches_person_frame_by_frame() -> None:
    lock = movenet.PersonLock()
    pose = _pose_around(100.0, 200.0)
    lock.observe(pose, 0.0, (480, 640))
    crop = lock.crop

    # Alternating frames are exactly the jump the hysteresis must absorb.
    for frame in range(12):
        lock.observe(pose, 0.1 * (frame + 1), (480, 640), competitor=frame % 2 == 0)
        assert lock.locked and lock.crop == crop

    # A competitor that holds for less than the window is still not enough.
    lock.observe(pose, 2.0, (480, 640), competitor=True)
    lock.observe(pose, 2.0 + movenet.SWITCH_AFTER_S - 0.1, (480, 640), competitor=True)
    assert lock.locked and lock.crop == crop
    # Losing the centre for one frame restarts the window.
    lock.observe(pose, 2.5, (480, 640), competitor=False)
    lock.observe(pose, 2.6, (480, 640), competitor=True)
    lock.observe(pose, 2.6 + movenet.SWITCH_AFTER_S + 0.1, (480, 640), competitor=True)
    assert not lock.locked
    assert lock.crop is None


def test_the_frame_center_prior_is_the_image_prior_on_the_whole_frame() -> None:
    lock = movenet.PersonLock()
    lock.observe(_pose_around(100.0, 200.0), 0.0, (480, 640))

    prior = lock.frame_center_prior(None, (480, 640))

    np.testing.assert_array_equal(prior, movenet.center_prior().astype(np.float32))


def test_the_frame_center_prior_maps_the_frame_centre_into_the_crop() -> None:
    lock = movenet.PersonLock()
    lock.observe(_pose_around(100.0, 200.0), 0.0, (480, 640))
    crop = movenet.Crop(x=100, y=80, side=300)

    prior = lock.frame_center_prior(crop, (480, 640))

    row, column = np.unravel_index(int(np.argmax(prior)), prior.shape)
    # (320, 240) is 220 px right and 160 px down of the crop's origin, and the
    # crop is 300 px wide: the frame centre is inside this input.
    assert column == pytest.approx(220 / 300 * 48, abs=1.5)
    assert row == pytest.approx(160 / 300 * 48, abs=1.5)


def test_the_detector_feeds_the_next_frame_through_the_person_crop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    detector = _detector(monkeypatch, tmp_path)
    subgraph = _Graph.pending[0]
    # Frame 1: an 8-keypoint body around feature cell (21, 15).
    spread = {index: (12 + index, 18 + index, 0.9) for index in range(8)}
    # Frame 2: the same heads the full frame would map to (400, 240).
    subgraph.payloads_by_execution = [
        _payload(spread),
        _payload({index: (24, 30, 0.9) for index in range(len(KEYPOINTS))}),
    ]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    first = detector.wait(detector.submit(frame))
    crop = detector.lock.crop
    assert crop is not None, "the first pose is healthy, so the next frame is cropped"
    second = detector.wait(detector.submit(frame))

    # The crop is what the DPU saw, so the head's cell is mapped through it.
    assert second.keypoints[0, 0] == pytest.approx((30 + 0.0) / 48 * crop.side + crop.x, abs=0.5)
    assert second.keypoints[0, 1] == pytest.approx((24 + 0.0) / 48 * crop.side + crop.y, abs=0.5)
    assert abs(second.keypoints[0, 0] - 400.0) > 50.0, "the input was not the whole frame"
    assert first.keypoints[0, 0] < 400.0


def test_the_detector_returns_to_the_full_frame_when_the_person_leaves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = SimpleNamespace(now=1000.0, monotonic=lambda: clock.now, perf_counter=time.perf_counter)
    monkeypatch.setattr(movenet, "time", clock)
    detector = _detector(monkeypatch, tmp_path)
    subgraph = _Graph.pending[0]
    spread = {index: (12 + index, 18 + index, 0.9) for index in range(8)}
    absent = {index: (0, 0, 0.01) for index in range(17)}
    subgraph.payloads_by_execution = [_payload(spread), _payload(absent), _payload(absent)]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detector.wait(detector.submit(frame))
    assert detector.lock.crop is not None

    clock.now += 0.2
    detector.wait(detector.submit(frame))
    assert detector.lock.crop is not None, "a one-frame dropout keeps the lock"

    clock.now += movenet.LEAVE_AFTER_S + 0.1
    detector.wait(detector.submit(frame))
    assert detector.lock.crop is None, "after the leave window the whole frame is back"


def test_the_detector_switches_person_only_after_the_switch_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = SimpleNamespace(now=1000.0, monotonic=lambda: clock.now, perf_counter=time.perf_counter)
    monkeypatch.setattr(movenet, "time", clock)
    detector = _detector(monkeypatch, tmp_path)
    subgraph = _Graph.pending[0]
    # The tracked body is at feature (10, 24); a more confident centre sits on
    # the image centre. The lock prior keeps the decode on the tracked one.
    peaks = {index: (24, 10, 0.9) for index in range(len(KEYPOINTS))}
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    two_people = {(10, 24): 0.4, (24, 24): 0.7}
    # One graph, two bodies: the tracked one on the left, a more confident one
    # on the image centre. The decode reads the keypoints of the body the
    # centre cell chooses.
    tracked = _buffers(peaks, centers=two_people)
    near_centre = _buffers(
        {index: (24, 24, 0.7) for index in range(len(KEYPOINTS))}, centers=two_people
    )
    tracked["heatmaps"][0, 24, 24] = near_centre["heatmaps"][0, 24, 24]
    payload = [tracked[name] for name in OUTPUT_ORDER]
    # A body too large to crop: the whole frame stays in view, so the
    # competitor near the frame centre is visible to the graph.
    detector.lock.observe(_pose_around(133.0, 240.0, half_extent=200.0), 1000.0, (480, 640))

    def decode() -> MovenetPose:
        subgraph.payloads = payload
        return detector.wait(detector.submit(frame))

    first = decode()
    assert first.keypoints[0, 0] == pytest.approx(10 / 48 * 640, abs=1.0)

    clock.now += movenet.SWITCH_AFTER_S - 0.1
    held = decode()
    assert held.keypoints[0, 0] == pytest.approx(10 / 48 * 640, abs=1.0)

    # The lock lets go during this decode, so the next input is the whole frame.
    clock.now += 0.2
    decode()
    switched = decode()
    assert switched.keypoints[0, 0] == pytest.approx(24 / 48 * 640, abs=1.0)


@pytest.mark.benchmark
def test_the_crop_preprocess_stays_within_its_budget() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[..., 0] = 200
    crop = movenet.Crop(x=170, y=90, side=300)
    lut = _lut()
    out = np.zeros((*MOVENET_INPUT_SIZE, 3), dtype=np.int8)

    def whole() -> None:
        movenet.preprocess_into(frame, lut, out)

    def square() -> None:
        movenet.preprocess_into(frame, lut, out, crop)

    whole_ms, crop_ms = _interleaved_ms(whole, square)

    assert crop_ms < 1.5, f"the crop preprocess costs {crop_ms:.3f} ms on the host"
    assert crop_ms <= whole_ms


@pytest.mark.benchmark
def test_the_person_lock_costs_less_than_a_frame() -> None:
    lock = movenet.PersonLock()
    pose = _pose_around(300.0, 240.0)
    lock.observe(pose, 0.0, (480, 640))
    crop = lock.crop

    def observe() -> None:
        lock.observe(pose, 0.0, (480, 640))

    def prior() -> None:
        lock.prior(crop, (480, 640))

    observe_ms = _median_ms(observe)
    prior_ms = _median_ms(prior)

    assert crop is not None
    assert observe_ms + prior_ms < 1.0, (
        f"the lock costs {observe_ms + prior_ms:.3f} ms on the host "
        f"({observe_ms:.3f} ms observe, {prior_ms:.3f} ms prior)"
    )
