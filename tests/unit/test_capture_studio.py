"""RGB capture studio: services refusal, run lifecycle, files and the live preview.

Everything runs against a fake clock and hand-fed ``VideoFrame`` objects: no
camera, no board, no browser.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import capture_studio as studio_mod  # noqa: E402

from cpsa.vision.types import VideoFrame  # noqa: E402

WIDTH = 64
HEIGHT = 48
"""Frames stay small: the studio never assumes the 640x480 of the board."""


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_frame(seq: int, t: float = 0.0, value: int = 7) -> VideoFrame:
    """One 64x48 RGB frame with a known corner pixel (RGB 1,2,3)."""

    rgb = np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)
    rgb[0, 0] = (1, 2, 3)
    rgb[8:24, 8:24] = (200, 40, 20)  # a solid patch that survives JPEG coding
    rgb[1, 0] = (10, 20, 30)
    return VideoFrame(seq=seq, t=t, width=WIDTH, height=HEIGHT, decode=lambda: rgb)


def make_studio(tmp_path: Path, **kwargs: Any) -> tuple[studio_mod.Studio, FakeClock]:
    clock = FakeClock()
    camera = {
        "node": "/dev/video0",
        "requested": "auto",
        "match": "RealSense",
        "width": 640,
        "height": 480,
        "fps": 30,
        "controls": {"exposure": 300, "gain": 160, "power_line_hz": 50},
    }
    return studio_mod.Studio(tmp_path / "out", clock=clock, camera=camera, **kwargs), clock


def feed(
    studio: studio_mod.Studio,
    clock: FakeClock,
    count: int,
    *,
    dt: float = 1 / 30,
    seq_start: int = 0,
) -> None:
    """Deliver ``count`` frames at ``dt`` seconds apart on the fake clock."""

    for index in range(count):
        studio.on_frame(make_frame(seq_start + index, t=clock.t))
        clock.advance(dt)


def run_dir(tmp_path: Path) -> Path:
    directories = [path for path in (tmp_path / "out").iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class FakeSource:
    """Minimal ``VideoSource``: delivers prepared frames and counts the lifecycle."""

    def __init__(self, frames: Sequence[VideoFrame] = ()) -> None:
        self.frames = list(frames)
        self.started = 0
        self.stopped = 0

    def start(
        self,
        on_frame: Callable[[VideoFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.started += 1
        for frame in self.frames:
            on_frame(frame)

    def stop(self) -> None:
        self.stopped += 1


# -- safety ---------------------------------------------------------------


def test_refuses_to_open_the_camera_while_services_are_active(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factory_calls: list[str] = []

    def factory(node: str, args: Any) -> FakeSource:
        factory_calls.append(node)
        return FakeSource()

    code = studio_mod.main(
        ["--out", str(tmp_path / "out")],
        service_check=lambda: ("cpsa-supervisor", "cpsa-dpu"),
        resolver=lambda device, match: "/dev/video0",
        source_factory=factory,
        serve_forever=lambda studio, args: pytest.fail("the server must not start"),
    )

    assert code == 2
    assert factory_calls == []
    stderr = capsys.readouterr().err
    assert "cpsa-supervisor, cpsa-dpu" in stderr
    assert "sudo systemctl stop cpsa-supervisor cpsa-dpu" in stderr
    assert not (tmp_path / "out").exists()


def test_allow_services_skips_the_check_and_always_closes_the_camera(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_check() -> Sequence[str]:
        raise AssertionError("the check must not run with --allow-services")

    source = FakeSource()

    def serve_forever(studio: studio_mod.Studio, args: Any) -> None:
        raise KeyboardInterrupt

    code = studio_mod.main(
        ["--out", str(tmp_path / "out"), "--allow-services"],
        service_check=failing_check,
        resolver=lambda device, match: "/dev/video0",
        source_factory=lambda node, args: source,
        clock=FakeClock(),
        serve_forever=serve_forever,
    )

    assert code == 0
    assert source.started == 1
    assert source.stopped == 1
    assert "10.42.0.83" in capsys.readouterr().out


def test_camera_controls_come_from_the_cli_defaults() -> None:
    args = studio_mod.parse_args([])

    assert (args.exposure, args.gain, args.power_line_hz) == (300, 160, 50)
    assert args.port == 8080
    assert args.max_run_seconds == 120.0
    assert args.out.name.startswith("rps_capture_")
    description = studio_mod.camera_description("/dev/video2", args)
    assert description["node"] == "/dev/video2"
    assert description["controls"] == {"exposure": 300, "gain": 160, "power_line_hz": 50}


# -- validation -----------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"label": "table", "tilt": "top"},
        {"label": "rock", "tilt": "side"},
        {"label": "rock", "tilt": "top", "seconds": 0},
        {"label": "rock", "tilt": "top", "every": 0},
        {"label": "rock", "tilt": "top", "note": "x" * 201},
        {"label": "rock", "tilt": "top", "seconds": "soon"},
    ],
)
def test_parse_run_request_rejects_bad_fields(payload: dict[str, Any]) -> None:
    with pytest.raises(studio_mod.ValidationError):
        studio_mod.parse_run_request(payload)


def test_parse_run_request_fills_the_defaults() -> None:
    request = studio_mod.parse_run_request({"label": "rock", "tilt": "top"})

    assert request.seconds == studio_mod.DEFAULT_SECONDS
    assert request.every == studio_mod.DEFAULT_EVERY
    assert request.note == ""


# -- run lifecycle --------------------------------------------------------


def test_start_counts_down_then_records_and_stop_writes_files(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.5)

    run_id = studio.start(studio_mod.RunRequest(label="rock", tilt="top", seconds=10, every=1))
    feed(studio, clock, 3, dt=0.1)  # 0.0 .. 0.2 s: still counting down

    status = studio.status()
    assert status["phase"] == "countdown"
    assert status["countdown_s"] > 0
    assert status["run"]["run_id"] == run_id
    assert list(run_dir(tmp_path).glob("rgb_*.jpg")) == []

    feed(studio, clock, 10, dt=0.1, seq_start=3)  # starts recording at 0.3 s

    status = studio.status()
    assert status["phase"] == "recording"
    # the first two of the ten frames still fell inside the 0.5 s countdown
    assert status["run"]["frames"] == 8
    assert status["run"]["delivered"] == 8
    assert 9.0 <= status["remaining_s"] <= 10.0

    summary = studio.stop()

    assert summary is not None
    assert summary["run_id"] == run_id
    assert summary["stopped_reason"] == "user"
    assert summary["frames"] == 8

    directory = run_dir(tmp_path)
    assert directory.name == f"{run_id}_top_rock"
    frames = sorted(directory.glob("rgb_*.jpg"))
    assert [path.name for path in frames] == [f"rgb_{index:05d}.jpg" for index in range(8)]

    meta = load_json(directory / "meta.json")
    assert meta["run_id"] == run_id
    assert (meta["label"], meta["tilt"]) == ("rock", "top")
    assert meta["every"] == 1
    assert meta["saved"] == 8
    assert meta["delivered"] == 8
    assert meta["stopped_reason"] == "user"
    assert meta["started_at"] and meta["finished_at"]
    assert meta["camera"]["node"] == "/dev/video0"
    assert meta["camera"]["controls"] == {"exposure": 300, "gain": 160, "power_line_hz": 50}
    assert [frame["seq"] for frame in meta["frames"]] == list(range(5, 13))
    assert all(isinstance(frame["t_monotonic"], float) for frame in meta["frames"])

    index = load_json(tmp_path / "out" / "runs.json")
    assert [run["run_id"] for run in index["runs"]] == [run_id]
    assert index["runs"][0]["frames"] == 8
    assert index["updated_at"]

    # JPEG at q95, and RGB (200,40,20) comes back as BGR (20,40,200): channel order kept
    written = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    assert written is not None
    assert written.shape == (HEIGHT, WIDTH, 3)
    patch = [int(value) for value in written[16, 16]]
    assert patch == pytest.approx([20, 40, 200], abs=25)


def test_every_k_writes_only_the_kept_frames(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.0)

    studio.start(studio_mod.RunRequest(label="paper", tilt="bust", seconds=10, every=3))
    feed(studio, clock, 10, dt=0.01)
    studio.stop()

    directory = run_dir(tmp_path)
    frames = sorted(directory.glob("rgb_*.jpg"))
    assert len(frames) == 4
    meta = load_json(directory / "meta.json")
    assert meta["every"] == 3
    assert meta["delivered"] == 10
    assert [frame["seq"] for frame in meta["frames"]] == [0, 3, 6, 9]


def test_max_run_seconds_bounds_the_run(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.0, max_run_seconds=1.0)

    studio.start(studio_mod.RunRequest(label="negative", tilt="frontal", seconds=60, every=1))
    feed(studio, clock, 12, dt=0.1)  # 0.0 .. 1.1 s: the run ends at 1.0 s

    assert studio.status()["phase"] == "idle"
    meta = load_json(run_dir(tmp_path) / "meta.json")
    assert meta["stopped_reason"] == "max_seconds"
    assert meta["duration_s"] == 1.0
    assert meta["requested_seconds"] == 60
    assert meta["saved"] == 10


def test_duration_ends_the_run_and_updates_the_index(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.0)

    studio.start(studio_mod.RunRequest(label="scissors", tilt="top", seconds=1.0, every=1))
    feed(studio, clock, 15, dt=0.1)

    assert studio.status()["phase"] == "idle"
    meta = load_json(run_dir(tmp_path) / "meta.json")
    assert meta["stopped_reason"] == "duration"
    assert meta["saved"] == 10
    assert load_json(tmp_path / "out" / "runs.json")["runs"][0]["frames"] == 10


def test_poll_finalises_a_run_when_no_more_frames_arrive(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.0)

    studio.start(studio_mod.RunRequest(label="rock", tilt="top", seconds=0.5, every=1))
    studio.on_frame(make_frame(0, t=clock.t))
    clock.advance(10.0)
    studio.poll()

    assert studio.status()["phase"] == "idle"
    assert load_json(run_dir(tmp_path) / "meta.json")["stopped_reason"] == "duration"


def test_stop_during_the_countdown_leaves_no_run(tmp_path: Path) -> None:
    studio, _ = make_studio(tmp_path, countdown_s=3.0)

    studio.start(studio_mod.RunRequest(label="rock", tilt="top", seconds=10))
    assert studio.stop() is None

    assert studio.status()["phase"] == "idle"
    assert list((tmp_path / "out").iterdir()) == []
    assert not (tmp_path / "out" / "runs.json").exists()


def test_second_start_while_busy_raises(tmp_path: Path) -> None:
    studio, _ = make_studio(tmp_path, countdown_s=3.0)

    studio.start(studio_mod.RunRequest(label="rock", tilt="top", seconds=10))
    with pytest.raises(studio_mod.BusyError):
        studio.start(studio_mod.RunRequest(label="paper", tilt="top", seconds=10))


def test_shutdown_finalises_a_run_in_progress(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.0)

    studio.start(studio_mod.RunRequest(label="rock", tilt="top", seconds=60, every=1))
    feed(studio, clock, 3, dt=0.1)
    studio.shutdown()

    meta = load_json(run_dir(tmp_path) / "meta.json")
    assert meta["stopped_reason"] == "shutdown"
    assert meta["saved"] == 3


def test_camera_error_finalises_the_run_and_is_reported(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.0)

    studio.start(studio_mod.RunRequest(label="rock", tilt="top", seconds=60))
    feed(studio, clock, 2, dt=0.1)
    studio.on_camera_error(RuntimeError("camera gone"))

    assert studio.status()["phase"] == "idle"
    meta = load_json(run_dir(tmp_path) / "meta.json")
    assert meta["stopped_reason"] == "error"
    assert meta["error"] == "RuntimeError: camera gone"
    assert studio.status()["error"] == "RuntimeError: camera gone"


def test_runs_json_is_reloaded_for_the_page(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.0)
    studio.start(studio_mod.RunRequest(label="rock", tilt="top", seconds=0.2))
    feed(studio, clock, 5, dt=0.1)

    reopened, _ = make_studio(tmp_path)

    assert [run["run_id"] for run in reopened.status()["runs"]] == [
        run["run_id"] for run in studio.status()["runs"]
    ]


# -- preview --------------------------------------------------------------


def test_preview_is_rate_limited_and_keeps_only_the_latest_frame(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, preview_fps=15.0)

    assert studio.latest_preview() is None
    feed(studio, clock, 60, dt=1 / 30)  # 2 s of 30 fps frames

    latest = studio.latest_preview()
    assert latest is not None
    seq, jpeg = latest
    assert 25 <= seq <= 32
    assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
    assert not (tmp_path / "out").exists()  # idle never creates a run directory


# -- HTTP -----------------------------------------------------------------


def test_video_endpoint_streams_multipart_frames(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.0)
    studio.on_frame(make_frame(0, t=clock.t))

    async def scenario() -> None:
        client = TestClient(TestServer(studio_mod.create_app(studio)))
        await client.start_server()
        try:
            response = await client.get("/video.mjpeg")
            assert response.status == 200
            assert response.headers["Content-Type"].startswith(
                "multipart/x-mixed-replace; boundary="
            )
            head = await asyncio.wait_for(response.content.readuntil(b"\r\n\r\n"), timeout=5)
            assert head.startswith(f"--{studio_mod.BOUNDARY}".encode())
            assert b"Content-Type: image/jpeg" in head
            length = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
            jpeg = await asyncio.wait_for(response.content.readexactly(length), timeout=5)
            assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
            response.close()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_page_has_the_video_form_and_run_list(tmp_path: Path) -> None:
    studio, _ = make_studio(tmp_path)

    async def scenario() -> None:
        client = TestClient(TestServer(studio_mod.create_app(studio)))
        await client.start_server()
        try:
            response = await client.get("/")
            assert response.status == 200
            assert response.content_type == "text/html"
            page = await response.text()
            for element in (
                "/video.mjpeg",
                'id="badge"',
                'id="countdown"',
                'id="label"',
                'id="tilt"',
                'id="seconds"',
                'id="every"',
                'id="note"',
                'id="start"',
                'id="stop"',
                'id="runs"',
                "/api/start",
                "/api/stop",
                "/api/status",
            ):
                assert element in page
            for label in ("rock", "paper", "scissors", "negative"):
                assert f"<option>{label}</option>" in page
        finally:
            await client.close()

    asyncio.run(scenario())


def test_api_drives_one_run_from_start_to_stop(tmp_path: Path) -> None:
    studio, clock = make_studio(tmp_path, countdown_s=0.2)

    async def scenario() -> None:
        client = TestClient(TestServer(studio_mod.create_app(studio, tick_s=0.05)))
        await client.start_server()
        try:
            started = await client.post(
                "/api/start",
                json={
                    "label": "scissors",
                    "tilt": "frontal",
                    "seconds": 30,
                    "every": 2,
                    "note": "tilted, hand only",
                },
            )
            assert started.status == 200
            body = await started.json()
            assert body["ok"] is True
            run_id = body["run_id"]

            status = await (await client.get("/api/status")).json()
            assert status["phase"] == "countdown"
            assert status["countdown_s"] > 0
            assert status["run"]["label"] == "scissors"
            assert status["run"]["frames"] == 0

            # the countdown does not record, so no frame is written yet
            studio.on_frame(make_frame(0, t=clock.t))
            clock.advance(2.0)

            malformed = await client.post("/api/start", json={"label": "table", "tilt": "top"})
            assert malformed.status == 400
            assert "label must be one of" in (await malformed.json())["error"]

            not_json = await client.post(
                "/api/start", data=b"rock", headers={"Content-Type": "text/plain"}
            )
            assert not_json.status == 400
            assert (await not_json.json())["error"] == "expected a JSON body"

            busy = await client.post("/api/start", json={"label": "rock", "tilt": "top"})
            assert busy.status == 409

            feed(studio, clock, 10, dt=0.1, seq_start=1)

            status = await (await client.get("/api/status")).json()
            assert status["phase"] == "recording"
            assert status["run"]["frames"] == 5  # every=2 over 10 frames
            assert 0 < status["remaining_s"] <= 30
            assert status["runs"] == []

            stopped = await (await client.post("/api/stop")).json()
            assert stopped["ok"] is True
            assert stopped["run"]["run_id"] == run_id
            assert stopped["run"]["frames"] == 5

            status = await (await client.get("/api/status")).json()
            assert status["phase"] == "idle"
            assert [run["run_id"] for run in status["runs"]] == [run_id]
            assert status["runs"][0]["frames"] == 5
        finally:
            await client.close()

    asyncio.run(scenario())

    directory = run_dir(tmp_path)
    assert len(list(directory.glob("rgb_*.jpg"))) == 5
    meta = load_json(directory / "meta.json")
    assert meta["note"] == "tilted, hand only"
    assert meta["every"] == 2
    assert meta["stopped_reason"] == "user"


def test_api_status_lists_the_labels_and_tilts(tmp_path: Path) -> None:
    studio, _ = make_studio(tmp_path)

    async def scenario() -> None:
        client = TestClient(TestServer(studio_mod.create_app(studio)))
        await client.start_server()
        try:
            status = await (await client.get("/api/status")).json()
            assert status["labels"] == list(studio_mod.LABELS)
            assert status["tilts"] == list(studio_mod.TILTS)
            assert status["defaults"] == {"seconds": 60, "every": 5}
            assert status["max_run_seconds"] == 120.0
            assert status["phase"] == "idle"
        finally:
            await client.close()

    asyncio.run(scenario())
