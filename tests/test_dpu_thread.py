"""The DPU job runs on a worker thread, so the caller's CPU work overlaps it.

On the KV260 VART's Python ``execute_async`` blocks for the whole DPU job
(8.9 ms measured for YOLOv5-nano) while releasing the GIL. These tests pin the
behaviour that turns that into real overlap: :meth:`DpuRunner.execute` returns
at once, :meth:`DpuRunner.wait` delivers the outputs, errors surface on
``wait``, and :meth:`DpuRunner.close` drains the job in flight.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from cpsa.dpu import DpuRunner

JOB_S = 0.05


class _Tensor:
    def __init__(self, name: str, dims: list[int], fix_point: int) -> None:
        self.name = name
        self.dims = dims
        self._attributes = {"fix_point": fix_point}

    def has_attr(self, name: str) -> bool:
        return name in self._attributes

    def get_attr(self, name: str) -> object:
        return self._attributes[name]


class _Subgraph:
    def __init__(self) -> None:
        self._attributes = {"device": "DPU"}

    def has_attr(self, name: str) -> bool:
        return name in self._attributes

    def get_attr(self, name: str) -> object:
        return self._attributes[name]


class _Root:
    def toposort_child_subgraph(self) -> list[_Subgraph]:
        return [_Subgraph()]


class _Graph:
    @classmethod
    def deserialize(cls, path: str) -> _Graph:
        return cls()

    def get_root_subgraph(self) -> _Root:
        return _Root()


class _BlockingRunner:
    """Like VART on the board: execute_async blocks for the job, wait is instant."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.threads: list[str] = []

    def get_input_tensors(self) -> list[_Tensor]:
        return [_Tensor("input", [1, 8, 8, 3], 6)]

    def get_output_tensors(self) -> list[_Tensor]:
        return [_Tensor("output", [1, 4, 4, 8], 3)]

    def execute_async(self, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> tuple[int, int]:
        self.threads.append(threading.current_thread().name)
        time.sleep(JOB_S)
        if self.fail:
            raise RuntimeError("dpu fault")
        outputs[0].fill(7)
        return (1, 0)

    def wait(self, job: tuple[int, int]) -> None:
        assert isinstance(job, tuple)


def _runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: _BlockingRunner) -> DpuRunner:
    xir = ModuleType("xir")
    xir.Graph = _Graph  # type: ignore[attr-defined]
    vart = ModuleType("vart")
    vart.Runner = SimpleNamespace(create_runner=lambda subgraph, mode: fake)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "xir", xir)
    monkeypatch.setitem(sys.modules, "vart", vart)
    model = tmp_path / "model.xmodel"
    model.write_bytes(b"fake")
    return DpuRunner(model, min_channels=1)


def test_execute_returns_at_once_and_wait_delivers_the_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _BlockingRunner()
    runner = _runner(monkeypatch, tmp_path, fake)
    inputs = [np.zeros((1, 8, 8, 3), np.int8)]
    outputs = runner.allocate()

    started = time.perf_counter()
    job = runner.execute(inputs, outputs)
    submitted = time.perf_counter() - started
    runner.wait(job)

    assert submitted < JOB_S / 2
    assert int(outputs[0].max()) == 7
    assert fake.threads and fake.threads[0].startswith("cpsa-dpu")
    runner.close()


def test_cpu_work_overlaps_the_dpu_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner(monkeypatch, tmp_path, _BlockingRunner())
    inputs = [np.zeros((1, 8, 8, 3), np.int8)]
    outputs = runner.allocate()

    started = time.perf_counter()
    job = runner.execute(inputs, outputs)
    time.sleep(JOB_S)  # the caller's own work on another frame
    runner.wait(job)
    elapsed = time.perf_counter() - started

    # Serial would be 2 x JOB_S; overlapped is about JOB_S.
    assert elapsed < 1.6 * JOB_S
    runner.close()


def test_a_dpu_error_surfaces_on_wait(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = _runner(monkeypatch, tmp_path, _BlockingRunner(fail=True))
    job = runner.execute([np.zeros((1, 8, 8, 3), np.int8)], runner.allocate())

    with pytest.raises(RuntimeError, match="dpu fault"):
        runner.wait(job)
    runner.close()


def test_close_drains_the_job_in_flight_and_refuses_new_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _runner(monkeypatch, tmp_path, _BlockingRunner())
    outputs = runner.allocate()
    runner.execute([np.zeros((1, 8, 8, 3), np.int8)], outputs)

    runner.close()

    assert int(outputs[0].max()) == 7
    with pytest.raises(RuntimeError, match="closed"):
        runner.execute([np.zeros((1, 8, 8, 3), np.int8)], outputs)
