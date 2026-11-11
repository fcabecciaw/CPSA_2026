"""IR/depth capture script: frame writing, bounds, and the services refusal (fake camera)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import capture_ir_frames as capture  # noqa: E402

from cpsa.vision.irdepth import IrDepthFrame  # noqa: E402

HEIGHT = 480
WIDTH = 640


def make_frame(seq: int) -> IrDepthFrame:
    ir = np.arange(HEIGHT * WIDTH, dtype=np.uint8).reshape(HEIGHT, WIDTH) + np.uint8(seq * 10)
    depth = (np.arange(HEIGHT * WIDTH, dtype=np.uint16).reshape(HEIGHT, WIDTH) + seq) * 3
    return IrDepthFrame(
        seq=seq, t=100.0 + seq / 30.0, width=WIDTH, height=HEIGHT, ir=ir, depth_raw=depth
    )


class FakeSource:
    """Deliver ``frames`` synchronously, recording start/stop calls."""

    def __init__(
        self,
        frames: Sequence[IrDepthFrame] = (),
        *,
        interrupt_at: int | None = None,
        error_at: int | None = None,
    ) -> None:
        self.frames = list(frames)
        self.interrupt_at = interrupt_at
        self.error_at = error_at
        self.started = 0
        self.stopped = 0

    def start(
        self,
        on_frame: Callable[[IrDepthFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.started += 1
        for index, frame in enumerate(self.frames):
            if self.interrupt_at == index:
                raise KeyboardInterrupt
            if self.error_at == index:
                assert on_error is not None
                on_error(RuntimeError("camera gone"))
                return
            on_frame(frame)
            if getattr(on_frame, "stop", None) is not None and on_frame.stop.is_set():
                return

    def stop(self) -> None:
        self.stopped += 1


def run_capture(
    tmp_path: Path,
    source: FakeSource,
    *fflags: str,
    service_check: Callable[[], Sequence[str]] | None = None,
) -> tuple[int, FakeSource, Path]:
    out = tmp_path / "capture"
    argv = ["--out", str(out), *fflags]

    def factory(node: str, match: str) -> FakeSource:
        return source

    code = capture.main(
        argv,
        service_check=service_check if service_check is not None else (lambda: ()),
        resolver=lambda device, match: "/dev/video0",
        source_factory=factory,
    )
    return code, source, out


def load_meta(out: Path) -> dict[str, Any]:
    return json.loads((out / "meta.json").read_text(encoding="utf-8"))


def test_refuses_to_open_the_camera_while_services_are_active(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = FakeSource([make_frame(0)])
    factory_calls: list[tuple[str, str]] = []

    def factory(node: str, match: str) -> FakeSource:
        factory_calls.append((node, match))
        return source

    code = capture.main(
        ["--out", str(tmp_path / "capture"), "--count", "1"],
        service_check=lambda: ("cpsa-supervisor", "cpsa-dpu"),
        resolver=lambda device, match: "/dev/video0",
        source_factory=factory,
    )

    assert code != 0
    assert factory_calls == []
    assert source.started == 0
    stderr = capsys.readouterr().err
    assert "cpsa-supervisor, cpsa-dpu" in stderr
    assert "sudo systemctl stop cpsa-supervisor cpsa-dpu" in stderr
    assert not (tmp_path / "capture" / "meta.json").exists()


def test_allow_services_skips_the_check(tmp_path: Path) -> None:
    def failing_check() -> Sequence[str]:
        raise AssertionError("the check must not run with --allow-services")

    source = FakeSource([make_frame(0)])

    code, _, out = run_capture(
        tmp_path, source, "--count", "1", "--allow-services", service_check=failing_check
    )

    assert code == 0
    assert source.started == 1


def test_service_check_fails_closed_when_systemctl_output_is_unexpected() -> None:
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "systemctl"
        assert argv[1] == "is-active"
        return subprocess.CompletedProcess(argv, 0, stdout="active\n", stderr="")

    with pytest.raises(capture.ServiceCheckError, match="1 states for 2 units"):
        capture.active_services(run=run)


def test_service_check_fails_closed_when_systemctl_is_missing() -> None:
    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("systemctl")

    with pytest.raises(capture.ServiceCheckError, match="cannot run systemctl"):
        capture.active_services(run=run)


def test_service_check_reads_active_units_from_one_call() -> None:
    seen: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 3, stdout="inactive\nactivating\n", stderr="")

    busy = capture.active_services(run=run)

    assert busy == ("cpsa-dpu",)
    assert seen == [["systemctl", "is-active", "cpsa-supervisor", "cpsa-dpu"]]


def test_capture_writes_ir_depth_and_metadata(tmp_path: Path) -> None:
    frames = [make_frame(index) for index in range(3)]

    code, source, out = run_capture(tmp_path, FakeSource(frames), "--count", "2")

    assert code == 0
    assert source.started == 1
    assert source.stopped == 1
    ir_paths = sorted(out.glob("ir_*.png"))
    assert [path.name for path in ir_paths] == ["ir_00000.png", "ir_00001.png"]
    assert sorted(path.name for path in out.glob("depth_*.npy")) == [
        "depth_00000.npy",
        "depth_00001.npy",
    ]
    assert cv2.imread(str(ir_paths[0]), cv2.IMREAD_UNCHANGED).shape == (HEIGHT, WIDTH)
    np.testing.assert_array_equal(cv2.imread(str(ir_paths[0]), cv2.IMREAD_UNCHANGED), frames[0].ir)
    np.testing.assert_array_equal(np.load(out / "depth_00000.npy"), frames[0].depth_raw)

    meta = load_meta(out)
    assert meta["saved"] == 2
    assert meta["delivered"] == 2
    assert meta["stopped_reason"] == "count"
    assert meta["camera"] == {
        "requested": "auto",
        "match": "RealSense",
        "node": "/dev/video0",
    }
    assert meta["size"] == {"width": WIDTH, "height": HEIGHT}
    assert meta["requested_count"] == 2
    assert [frame["seq"] for frame in meta["frames"]] == [0, 1]
    assert meta["frames"][0]["ir"] == "ir_00000.png"
    assert meta["frames"][0]["depth"] == "depth_00000.npy"
    assert meta["started_at"] and meta["finished_at"]


def test_every_keeps_every_kth_delivered_frame(tmp_path: Path) -> None:
    frames = [make_frame(index) for index in range(5)]

    code, _, out = run_capture(tmp_path, FakeSource(frames), "--count", "3", "--every", "2")

    assert code == 0
    meta = load_meta(out)
    assert meta["saved"] == 3
    assert [frame["seq"] for frame in meta["frames"]] == [0, 2, 4]


def test_max_seconds_bounds_the_run(tmp_path: Path) -> None:
    code, source, out = run_capture(
        tmp_path, FakeSource([make_frame(0)]), "--count", "10", "--max-seconds", "0.01"
    )

    assert code == 0
    assert source.stopped == 1
    meta = load_meta(out)
    assert meta["saved"] == 1
    assert meta["stopped_reason"] == "max_seconds"


def test_interrupt_still_closes_the_camera(tmp_path: Path) -> None:
    frames = [make_frame(0), make_frame(1)]

    code, source, out = run_capture(tmp_path, FakeSource(frames, interrupt_at=1))

    assert code == 130
    assert source.stopped == 1
    meta = load_meta(out)
    assert meta["saved"] == 1
    assert meta["stopped_reason"] == "interrupted"


def test_camera_error_is_recorded_and_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, source, out = run_capture(
        tmp_path, FakeSource([make_frame(0), make_frame(1)], error_at=1), "--count", "5"
    )

    assert code == 1
    assert source.stopped == 1
    meta = load_meta(out)
    assert meta["error"] == "RuntimeError: camera gone"
    assert meta["stopped_reason"] == "error"
    assert "camera gone" in capsys.readouterr().err
