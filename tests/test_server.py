"""The web server: static files, health, MJPEG and protocol v1 (F-K1, F-K2).

The server runs in its own thread with its own event loop, exactly as process B
does on the board, so the client side is the real ``aiohttp`` client.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import socket
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from aiohttp import ClientSession

from cpsa import protocol
from cpsa.channel import CommandChannel, MessageChannel
from cpsa.config import WebConfig
from cpsa.protocol import ACTIVE_MODELS
from cpsa.server import VIDEO_BOUNDARY, WebServer, serve
from tests.contract.protocol_schemas import validate_server_message

TIMEOUT_S = 5.0
STATUS = protocol.status(
    state="running",
    camera="ok",
    model="yolox_nano",
    fps=59.4,
    dpu_ms=4.7,
    errors=[],
    uptime_s=12.5,
)
STATE = protocol.showcase_state(
    "yolox_nano",
    8,
    4,
    [protocol.ShowcaseObject(label="persona", class_id=0, score=0.75, box=(1.0, 1.0, 5.0, 3.0))],
    fps=59.4,
    dpu_ms=4.7,
    total_ms=11.2,
)


class FakeQueue:
    """The ``multiprocessing`` queue surface the channel needs, in-process."""

    def __init__(self, maxsize: int = 64) -> None:
        self._items: deque[tuple[str, dict[str, Any]]] = deque()
        self._maxsize = maxsize
        self.closed = False

    def put_nowait(self, item: tuple[str, dict[str, Any]]) -> None:
        if len(self._items) >= self._maxsize:
            raise queue.Full
        self._items.append(item)

    def get_nowait(self) -> tuple[str, dict[str, Any]]:
        if not self._items:
            raise queue.Empty
        return self._items.popleft()

    def close(self) -> None:
        self.closed = True


class FakeFrames:
    """A shared frame channel stand-in that counts how often it was read."""

    def __init__(self, width: int = 32, height: int = 24) -> None:
        self.width = width
        self.height = height
        self.reads = 0
        self._seq = 0
        self._image = np.full((height, width, 3), 40, dtype=np.uint8)

    def write(self, value: int | None = None) -> int:
        self._seq += 1
        if value is not None:
            self._image = np.full((self.height, self.width, 3), value, dtype=np.uint8)
        return self._seq

    def read(self, last_seq: int = 0) -> tuple[int, np.ndarray] | None:
        self.reads += 1
        if self._seq <= last_seq:
            return None
        return self._seq, self._image.copy()


def web_config(tmp_path: Path, **overrides: Any) -> WebConfig:
    static = tmp_path / "dist"
    (static / "assets").mkdir(parents=True, exist_ok=True)
    (static / "index.html").write_text("<html>cpsa</html>", encoding="utf-8")
    (static / "assets" / "app.js").write_text("console.log('cpsa')", encoding="utf-8")
    settings: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 0,
        "video_fps": 60.0,
        "jpeg_quality": 70,
        "static_dir": static,
    }
    settings.update(overrides)
    return WebConfig(**settings)  # type: ignore[arg-type]


class Running:
    """A ``WebServer`` started in its own thread and stopped at the end."""

    def __init__(
        self, tmp_path: Path, *, models: Sequence[str] | None = None, **overrides: Any
    ) -> None:
        self.frames = FakeFrames()
        self.messages = MessageChannel(FakeQueue())
        self.commands = CommandChannel(FakeQueue())
        self.server = WebServer(
            web_config(tmp_path, **overrides),
            frames=self.frames,
            messages=self.messages,
            commands=self.commands,
            models=models if models is not None else ACTIVE_MODELS,
            server_version="9.9.9",
        )

    def __enter__(self) -> Running:
        self.server.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.server.stop()
        self.messages.close()
        self.commands.close()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.port}{path}"


async def receive(ws: Any, message_type: str, timeout: float = TIMEOUT_S) -> dict[str, Any]:
    """Receive messages until one of ``message_type`` arrives, then validate it."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"no {message_type} within {timeout} s"
        message = await ws.receive_json(timeout=remaining)
        validate_server_message(message)
        if message["type"] == message_type:
            return message


def wait_until(condition: Callable[[], bool], timeout: float = TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.01)


def run(scenario: Callable[[], Any]) -> Any:
    return asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Static files and health
# --------------------------------------------------------------------------- #


def test_the_index_and_the_assets_are_served(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> tuple[int, str, int, str, str | None]:
            async with ClientSession() as session:
                async with session.get(running.url("/")) as index:
                    html = await index.text()
                    cache_control = index.headers.get("Cache-Control")
                async with session.get(running.url("/assets/app.js")) as asset:
                    script = await asset.text()
                return index.status, html, asset.status, script, cache_control

        index_status, html, asset_status, script, cache_control = run(scenario)

    assert index_status == 200 and html == "<html>cpsa</html>"
    assert cache_control == "no-cache"
    assert asset_status == 200 and script == "console.log('cpsa')"


def test_vendored_files_are_served_with_the_wasm_mime_type(tmp_path: Path) -> None:
    with Running(tmp_path) as running:
        static = tmp_path / "dist"
        (static / "vendor" / "mp" / "wasm").mkdir(parents=True)
        (static / "vendor" / "mp" / "bundle.mjs").write_text("export {}", encoding="utf-8")
        (static / "vendor" / "mp" / "wasm" / "runtime.wasm").write_bytes(b"\0asm\x01\0\0\0")
        (static / "models").mkdir()
        (static / "models" / "face.task").write_bytes(b"model")

        async def scenario() -> list[tuple[int, str]]:
            results = []
            async with ClientSession() as session:
                for path in (
                    "/vendor/mp/bundle.mjs",
                    "/vendor/mp/wasm/runtime.wasm",
                    "/models/face.task",
                ):
                    async with session.get(running.url(path)) as response:
                        await response.read()
                        results.append((response.status, response.content_type))
            return results

        results = run(scenario)

    assert [status for status, _ in results] == [200, 200, 200]
    assert results[1][1] == "application/wasm"


def test_a_static_path_outside_the_root_is_not_served(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("do not serve", encoding="utf-8")
    with Running(tmp_path) as running:

        async def scenario() -> int:
            async with (
                ClientSession() as session,
                session.get(running.url("/../secret.txt")) as response,
            ):
                return response.status

        status = run(scenario)

    assert status in (403, 404)


def test_an_asset_outside_the_static_root_is_not_served(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("do not serve", encoding="utf-8")
    with Running(tmp_path) as running:

        async def scenario() -> int:
            async with (
                ClientSession() as session,
                session.get(running.url("/assets/../../secret.txt")) as response,
            ):
                return response.status

        status = run(scenario)

    assert status in (403, 404)


def test_a_missing_file_is_not_found(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> tuple[int, int]:
            async def status_of(path: str) -> int:
                async with (
                    ClientSession() as session,
                    session.get(running.url(path)) as response,
                ):
                    return response.status

            return await status_of("/assets/nope.js"), await status_of("/assets/../dist/nope")

        missing_asset, nested = run(scenario)

    assert missing_asset == 404 and nested == 404


def test_health_reports_starting_until_the_first_status(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> dict[str, Any]:
            async with (
                ClientSession() as session,
                session.get(running.url("/health")) as response,
            ):
                return await response.json()

        payload = run(scenario)

    assert payload == {"state": "starting", "fps": None, "uptime_s": 0.0}


def test_health_reports_the_latest_status(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> dict[str, Any]:
            running.messages.put("status", STATUS)
            async with ClientSession() as session:
                for _ in range(100):
                    async with session.get(running.url("/health")) as response:
                        payload = await response.json()
                    if payload["state"] == "running":
                        return payload
                    await asyncio.sleep(0.01)
                raise AssertionError("health never reported the published status")

        payload = run(scenario)

    assert payload == {"state": "running", "fps": 59.4, "uptime_s": 12.5}


# --------------------------------------------------------------------------- #
# Protocol v1
# --------------------------------------------------------------------------- #


def test_a_client_receives_hello_and_the_snapshot(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> dict[str, dict[str, Any]]:
            running.messages.put("status", STATUS)
            running.messages.put("showcase.state", STATE)
            seen: dict[str, dict[str, Any]] = {}
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                try:
                    hello = await receive(ws, "hello")
                    seen["hello"] = hello
                    # The order of two snapshots is not part of the contract.
                    while len(seen) < 3:
                        message = await ws.receive_json(timeout=TIMEOUT_S)
                        validate_server_message(message)
                        seen[message["type"]] = message
                finally:
                    await ws.close()
            return seen

        seen = run(scenario)

    assert seen["hello"]["data"] == {"server_version": "9.9.9", "protocol_version": 1}
    assert seen["status"]["data"]["state"] == "running"
    assert seen["showcase.state"]["data"]["objects"][0]["label"] == "persona"


def test_a_published_state_reaches_a_connected_client(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> dict[str, Any]:
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                try:
                    await receive(ws, "hello")
                    running.messages.put("showcase.state", STATE)
                    message = await receive(ws, "showcase.state")
                finally:
                    await ws.close()
            return dict(message["data"])

        data = run(scenario)

    assert data["counts"] == [{"label": "persona", "count": 1}]


def test_ping_is_answered_with_pong(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> dict[str, Any]:
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                try:
                    await receive(ws, "hello")
                    await ws.send_json({"v": 1, "type": "ping", "data": {"id": 7}})
                    message = await receive(ws, "pong")
                finally:
                    await ws.close()
            return dict(message["data"])

        data = run(scenario)

    assert data == {"id": 7}


def test_an_unknown_command_is_rejected_with_its_ref_seq(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> dict[str, Any]:
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                try:
                    await receive(ws, "hello")
                    await ws.send_json(
                        {"v": 1, "type": "mode.set", "seq": 3, "data": {"mode": "medical"}}
                    )
                    message = await receive(ws, "error")
                finally:
                    await ws.close()
            return dict(message["data"])

        data = run(scenario)

    assert data["code"] == "bad_command"
    assert data["ref_seq"] == 3
    assert "mode.set" in data["message"]


def test_model_set_is_routed_to_the_pipeline(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> tuple[str, dict[str, Any]]:
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                try:
                    await receive(ws, "hello")
                    await ws.send_json(
                        {"v": 1, "type": "model.set", "seq": 4, "data": {"model": "movenet"}}
                    )
                    for _ in range(200):
                        command = running.commands.get_nowait()
                        if command is not None:
                            return command
                        await asyncio.sleep(0.01)
                    raise AssertionError("the command never reached the pipeline")
                finally:
                    await ws.close()

        topic, data = run(scenario)

    assert topic == "model.set"
    assert data == {"model": "movenet"}


def test_a_disabled_model_is_a_bad_command_and_is_not_forwarded(tmp_path: Path) -> None:
    with Running(tmp_path, models=("yolox",)) as running:

        async def scenario() -> dict[str, Any]:
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                try:
                    await receive(ws, "hello")
                    await ws.send_json(
                        {"v": 1, "type": "model.set", "seq": 6, "data": {"model": "movenet"}}
                    )
                    message = await receive(ws, "error")
                finally:
                    await ws.close()
            return dict(message["data"])

        data = run(scenario)

    assert data["code"] == "bad_command"
    assert data["ref_seq"] == 6
    assert "not enabled" in data["message"]
    assert running.commands.get_nowait() is None


def test_an_unknown_model_is_a_bad_command_and_is_not_forwarded(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> dict[str, Any]:
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                try:
                    await receive(ws, "hello")
                    await ws.send_json(
                        {"v": 1, "type": "model.set", "seq": 5, "data": {"model": "medical"}}
                    )
                    message = await receive(ws, "error")
                finally:
                    await ws.close()
            return dict(message["data"])

        data = run(scenario)

    assert data["code"] == "bad_command"
    assert data["ref_seq"] == 5
    assert "medical" in data["message"]
    assert running.commands.get_nowait() is None


def test_an_invalid_message_is_a_bad_command(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> list[dict[str, Any]]:
            errors = []
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                try:
                    await receive(ws, "hello")
                    await ws.send_str("not json")
                    errors.append(dict((await receive(ws, "error"))["data"]))
                    await ws.send_json({"v": 2, "type": "ping", "data": {"id": 1}})
                    errors.append(dict((await receive(ws, "error"))["data"]))
                    await ws.send_json({"v": 1, "type": "ping", "data": {}})
                    errors.append(dict((await receive(ws, "error"))["data"]))
                    await ws.send_json({"v": 1, "type": "ping"})
                    errors.append(dict((await receive(ws, "error"))["data"]))
                    await ws.send_json({"v": 1, "type": "ping", "seq": 0, "data": {"id": 1}})
                    errors.append(dict((await receive(ws, "error"))["data"]))
                finally:
                    await ws.close()
            return errors

        errors = run(scenario)

    assert [error["code"] for error in errors] == ["bad_command"] * 5
    assert "invalid JSON" in errors[0]["message"]
    assert "data.id" in errors[2]["message"]
    assert errors[1].get("ref_seq") is None


def test_a_client_that_vanishes_is_forgotten(tmp_path: Path) -> None:
    with Running(tmp_path) as running:

        async def scenario() -> None:
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                await receive(ws, "hello")
                await ws.close()
            for _ in range(200):
                if not running.server._clients:  # type: ignore[attr-defined]
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("the disconnected client stayed registered")

        run(scenario)


# --------------------------------------------------------------------------- #
# The MJPEG stream
# --------------------------------------------------------------------------- #


def test_the_video_is_encoded_only_while_a_client_watches(tmp_path: Path) -> None:
    with Running(tmp_path) as running:
        running.frames.write(80)

        async def scenario() -> tuple[int, int]:
            async with ClientSession() as session:
                # No video client yet: the encoder must not run at all.
                await asyncio.sleep(0.2)
                before = running.frames.reads
                async with session.get(running.url("/video.mjpeg")) as response:
                    assert response.status == 200
                    body = await asyncio.wait_for(
                        _first_part(response.content.iter_any()), timeout=TIMEOUT_S
                    )
                assert b"image/jpeg" in body
                return before, running.frames.reads

        before, during = run(scenario)

        # After the client left, encoding stops: the read counter freezes.
        wait_until(lambda: _frozen(running.frames))

    assert before == 0
    assert during > 0


def test_the_video_stream_is_multipart_jpeg(tmp_path: Path) -> None:
    with Running(tmp_path) as running:
        running.frames.write(90)

        async def scenario() -> tuple[str, bytes, bytes]:
            async with (
                ClientSession() as session,
                session.get(running.url("/video.mjpeg")) as response,
            ):
                body = await asyncio.wait_for(
                    _first_part(response.content.iter_any()), timeout=TIMEOUT_S
                )
                content_type = response.headers["Content-Type"]
                assert response.headers["Cache-Control"] == "no-store"
            return content_type, body, response.headers.get("Cache-Control", "").encode()

        content_type, body, _cache = run(scenario)

    assert content_type == f"multipart/x-mixed-replace; boundary={VIDEO_BOUNDARY}"
    assert body.startswith(f"--{VIDEO_BOUNDARY}\r\n".encode())
    assert b"Content-Type: image/jpeg" in body
    assert b"X-Frame-Seq: " in body
    assert b"\xff\xd8" in body  # JPEG SOI


async def _first_part(chunks: Any) -> bytes:
    """Accumulate stream bytes until the first complete JPEG part arrived."""
    body = b""
    async for chunk in chunks:
        body += chunk
        if b"\xff\xd8" in body and b"\xff\xd9" in body:
            return body
    raise AssertionError("the video stream ended before the first frame")


def _frozen(frames: FakeFrames, window: float = 0.1) -> bool:
    first = frames.reads
    time.sleep(window)
    return frames.reads == first


# --------------------------------------------------------------------------- #
# The process-B entry point
# --------------------------------------------------------------------------- #


def test_serve_returns_one_when_the_port_is_taken(tmp_path: Path) -> None:
    socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_.bind(("127.0.0.1", 0))
    socket_.listen(1)
    try:
        config = web_config(tmp_path, port=socket_.getsockname()[1])
        code = serve(
            config,
            frames=FakeFrames(),
            messages=MessageChannel(FakeQueue()),
            stop=threading.Event(),
        )
    finally:
        socket_.close()

    assert code == 1


def test_serve_stops_when_asked(tmp_path: Path) -> None:
    stop = threading.Event()
    ready = threading.Event()
    results: list[int] = []

    def target() -> None:
        try:
            results.append(
                serve(
                    web_config(tmp_path),
                    frames=FakeFrames(),
                    messages=MessageChannel(FakeQueue()),
                    stop=stop,
                    ready=ready,
                )
            )
        finally:
            with contextlib.suppress(Exception):
                pass

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    try:
        assert ready.wait(TIMEOUT_S), "the server never reported ready"
    finally:
        stop.set()
        thread.join(TIMEOUT_S)

    assert not thread.is_alive()
    assert results == [0]


def test_the_envelope_of_every_sent_message_is_json_safe(tmp_path: Path) -> None:
    """A ``NaN`` in a payload must never reach the wire (JSON has no NaN)."""
    with Running(tmp_path) as running:

        async def scenario() -> int:
            running.messages.put("status", STATUS)
            async with ClientSession() as session:
                ws = await session.ws_connect(running.url("/ws"))
                try:
                    message = await receive(ws, "status")
                    assert message["type"] == "status"
                    json.dumps(message, allow_nan=False)
                finally:
                    await ws.close()
            return message["v"]

        version = run(scenario)

    assert version == 1


# --------------------------------------------------------------------------- #
# Live calibration (debug panel)
# --------------------------------------------------------------------------- #


async def _send_and_collect(
    running: Running, command: dict[str, Any], *, expect_error: bool
) -> tuple[dict[str, Any] | None, list[tuple[str, dict[str, Any]]]]:
    async with ClientSession() as session:
        ws = await session.ws_connect(running.url("/ws"))
        try:
            await receive(ws, "hello")
            await ws.send_json(command)
            error = dict((await receive(ws, "error"))["data"]) if expect_error else None
            forwarded: list[tuple[str, dict[str, Any]]] = []
            for _ in range(200 if not expect_error else 5):
                item = running.commands.get_nowait()
                if item is not None:
                    forwarded.append(item)
                    break
                await asyncio.sleep(0.01)
            return error, forwarded
        finally:
            await ws.close()


def test_calibration_set_is_validated_and_routed_to_the_pipeline(tmp_path: Path) -> None:
    command = {
        "v": 1,
        "type": "calibration.set",
        "seq": 3,
        "data": {"zoom": 1.5, "zoom_center": [0.5, 0.45], "gain": 320, "exposure": 200},
    }
    with Running(tmp_path) as running:
        error, forwarded = run(lambda: _send_and_collect(running, command, expect_error=False))

    assert error is None
    assert forwarded == [
        (
            "calibration.set",
            {"zoom": 1.5, "zoom_center": [0.5, 0.45], "gain": 320, "exposure": 200},
        )
    ]


def test_calibration_reset_is_routed_to_the_pipeline(tmp_path: Path) -> None:
    command = {"v": 1, "type": "calibration.reset", "seq": 4, "data": {}}
    with Running(tmp_path) as running:
        error, forwarded = run(lambda: _send_and_collect(running, command, expect_error=False))

    assert error is None
    assert forwarded == [("calibration.reset", {})]


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ({}, "at least one"),
        ({"zoom": 2.5}, "zoom"),
        ({"zoom": "1.5"}, "zoom"),
        ({"zoom_center": [0.5]}, "zoom_center"),
        ({"zoom_center": [0.5, 1.2]}, "zoom_center"),
        ({"gain": 10}, "gain"),
        ({"gain": 100.5}, "gain"),
        ({"exposure": 150}, "exposure"),
        ({"saturation": 256}, "saturation"),
        ({"brightness": 10}, "brightness"),
    ],
)
def test_a_bad_calibration_is_a_bad_command_and_is_not_forwarded(
    tmp_path: Path, data: dict[str, Any], reason: str
) -> None:
    command = {"v": 1, "type": "calibration.set", "seq": 7, "data": data}
    with Running(tmp_path) as running:
        error, forwarded = run(lambda: _send_and_collect(running, command, expect_error=True))

    assert error is not None
    assert error["code"] == "bad_command"
    assert error["ref_seq"] == 7
    assert reason in error["message"]
    assert forwarded == []
