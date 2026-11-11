"""The web adapter's lifecycle, snapshot, commands and slow-client behavior (AC-06)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket as socket_lib
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import WSCloseCode, WSMsgType

from cpsa.adapters.web import server as server_module
from cpsa.adapters.web.server import MAX_MESSAGE_BYTES, WebServer
from cpsa.application.config import ServerConfig
from tests.contract.protocol_schemas import validate_server_message

TIMEOUT_S = 3.0


def status_data(*, mode: str = "medical", uptime_s: float = 1.0) -> dict[str, Any]:
    return {
        "mode": mode,
        "mode_state": "running" if mode == "medical" else "stopped",
        "perception": {"fps": 10.0, "dpu_ms": 4.0, "camera": "ok", "model": "fake"},
        "imu": "connected" if mode == "medical" else "n/a",
        "errors": [],
        "uptime_s": uptime_s,
    }


MEDICAL_STATE = {
    "imu": "connected",
    "tag": None,
    "episode": None,
    "last_decision": None,
}
MEDICAL_HANDS = {
    "layout": "HALFBUST_62",
    "w": 640,
    "h": 480,
    "keypoints": [],
    "subject_box": [0.0, 0.0, 640.0, 480.0],
}


@contextlib.contextmanager
def running_server(
    tmp_path: Path,
    *,
    on_command: Callable[[str, dict[str, Any], int | None], None] | None = None,
    idle_timeout_s: float = 30.0,
    video: bool = False,
) -> Iterator[tuple[WebServer, list[tuple[str, dict[str, Any], int | None]]]]:
    static = tmp_path / "dist"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>cpsa ui</title>", encoding="utf-8")
    (static / "assets" / "app.js").write_text("console.log('cpsa');", encoding="utf-8")
    commands: list[tuple[str, dict[str, Any], int | None]] = []

    def handler(command: str, data: dict[str, Any], ref_seq: int | None) -> None:
        commands.append((command, data, ref_seq))
        if on_command is not None:
            on_command(command, data, ref_seq)

    config = ServerConfig(
        host="127.0.0.1",
        port=0,
        static_dir=static,
        idle_timeout_s=idle_timeout_s,
        video=video,
    )
    server = WebServer(config, handler, "test-version")
    server.start()
    try:
        yield server, commands
    finally:
        server.stop()


def publish_state(server: WebServer, topic: str, data: dict[str, Any]) -> None:
    """Publish through the loop and wait until it has been applied."""
    loop = server._loop
    assert loop is not None
    done = threading.Event()

    def apply() -> None:
        server._handle_state(topic, dict(data))
        done.set()

    loop.call_soon_threadsafe(apply)
    assert done.wait(TIMEOUT_S)


def publish_event(server: WebServer, topic: str, data: dict[str, Any]) -> None:
    """Publish an event through the loop and wait until it has been applied."""
    loop = server._loop
    assert loop is not None
    done = threading.Event()

    def apply() -> None:
        server._handle_event(topic, dict(data))
        done.set()

    loop.call_soon_threadsafe(apply)
    assert done.wait(TIMEOUT_S)


def wait_until(predicate: Callable[[], bool], timeout: float = TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not met in time")
        time.sleep(0.005)


async def receive(ws: aiohttp.ClientWebSocketResponse) -> dict[str, Any]:
    message = await asyncio.wait_for(ws.receive(), timeout=TIMEOUT_S)
    assert message.type is WSMsgType.TEXT, message
    return json.loads(message.data)


def run(scenario: Callable[[], Any]) -> Any:
    return asyncio.run(scenario())


def test_connect_receives_hello_then_the_full_snapshot(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):
        publish_state(server, "status", status_data())
        publish_state(server, "medical.state", MEDICAL_STATE)
        publish_state(server, "medical.hands", MEDICAL_HANDS)

        async def scenario() -> None:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                messages = [await receive(ws) for _ in range(4)]

            assert [message["type"] for message in messages] == [
                "hello",
                "status",
                "medical.state",
                "medical.hands",
            ]
            for message in messages:
                validate_server_message(message)
            hello = messages[0]["data"]
            assert hello["server_version"] == "test-version"
            assert hello["protocol_version"] == 1
            assert hello["modes"] == ["idle", "medical"]
            assert "status" in hello["capabilities"]
            assert messages[1]["data"]["mode"] == "medical"

        run(scenario)


def log_event(index: int) -> dict[str, Any]:
    return {
        "level": "warning" if index % 7 == 0 else "info",
        "source": f"bluecoin:CPSA_L{index % 3}",
        "message": f"connection line {index}",
    }


def test_a_new_client_gets_the_recent_log_events_after_the_snapshot(tmp_path: Path) -> None:
    """A reconnecting dashboard continues the log timeline it was showing."""
    with running_server(tmp_path) as (server, _):
        publish_state(server, "status", status_data())
        total = server_module.LOG_REPLAY_SIZE + 5
        for index in range(total):
            publish_event(server, "log", log_event(index))

        async def scenario() -> list[dict[str, Any]]:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                return [await receive(ws) for _ in range(2 + server_module.LOG_REPLAY_SIZE)]

        messages = run(scenario)
        assert [message["type"] for message in messages[:2]] == ["hello", "status"]
        logs = messages[2:]
        assert [message["type"] for message in logs] == ["log"] * server_module.LOG_REPLAY_SIZE
        assert [message["data"]["message"] for message in logs] == [
            f"connection line {index}"
            for index in range(total - server_module.LOG_REPLAY_SIZE, total)
        ]
        for message in messages:
            validate_server_message(message)


def test_a_live_log_event_is_delivered_and_valid(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):
        publish_state(server, "status", status_data())

        async def scenario() -> dict[str, Any]:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                await receive(ws)  # hello
                await receive(ws)  # status
                publish_event(server, "log", log_event(1))
                return await receive(ws)

        message = run(scenario)
        assert message["type"] == "log"
        assert message["data"] == log_event(1)
        validate_server_message(message)


def test_medical_states_are_not_snapshot_in_idle(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):
        publish_state(server, "status", status_data(mode="idle"))
        publish_state(server, "medical.state", MEDICAL_STATE)

        async def scenario() -> None:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                hello = await receive(ws)
                status = await receive(ws)
                await ws.send_json({"v": 1, "type": "ping", "data": {"id": 1}})
                pong = await receive(ws)

            assert [hello["type"], status["type"], pong["type"]] == ["hello", "status", "pong"]

        run(scenario)


def test_ping_answers_pong_with_the_same_id_and_seq_increases(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):

        async def scenario() -> None:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                messages = [await receive(ws)]
                for message_id in (7, "abc"):
                    await ws.send_json({"v": 1, "type": "ping", "data": {"id": message_id}})
                    messages.append(await receive(ws))

            assert [message["data"]["id"] for message in messages[1:]] == [7, "abc"]
            assert [message["seq"] for message in messages] == [1, 2, 3]
            for message in messages:
                validate_server_message(message)

        run(scenario)


def test_mode_set_is_forwarded_with_the_client_seq_and_errors_quote_it(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, commands):

        async def scenario() -> None:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                await receive(ws)
                await ws.send_json(
                    {"v": 1, "type": "mode.set", "seq": 3, "data": {"mode": "medical"}}
                )
                await ws.send_json({"v": 1, "type": "mode.set", "data": {"mode": "idle"}})
                await ws.send_json({"v": 1, "type": "mode.set", "seq": 4, "data": {"mode": "lab"}})
                invalid_mode = await receive(ws)
                await ws.send_json({"v": 1, "type": "preview.set", "seq": 5, "data": {"on": True}})
                unknown = await receive(ws)
                await ws.send_str("not json")
                malformed = await receive(ws)
                await ws.send_json({"v": 1, "type": "ping", "seq": "x", "data": {"id": 1}})
                bad_seq = await receive(ws)

            assert commands == [
                ("mode.set", {"mode": "medical"}, 3),
                ("mode.set", {"mode": "idle"}, None),
            ]
            assert invalid_mode["data"]["code"] == "bad_command"
            # IF-304: ref_seq identifies the client command it answers.
            assert invalid_mode["data"]["ref_seq"] == 4
            assert unknown["data"]["code"] == "bad_command"
            assert unknown["data"]["ref_seq"] == 5
            assert malformed["data"]["code"] == "bad_command"
            assert "ref_seq" not in malformed["data"]
            assert bad_seq["data"]["code"] == "bad_command"
            assert "ref_seq" not in bad_seq["data"]
            for message in (invalid_mode, unknown, malformed, bad_seq):
                validate_server_message(message)

        run(scenario)


def test_every_client_command_is_logged_with_type_data_seq_and_peer(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with running_server(tmp_path) as (server, _):

        async def scenario() -> None:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                await receive(ws)
                await ws.send_json(
                    {"v": 1, "type": "mode.set", "seq": 4, "data": {"mode": "medical"}}
                )

        with caplog.at_level(logging.INFO, logger="cpsa.web"):
            run(scenario)

    records = [
        record.getMessage() for record in caplog.records if "Client command" in record.getMessage()
    ]
    assert any(
        "type=mode.set" in message
        and "{'mode': 'medical'}" in message
        and "seq=4" in message
        and "peer=127.0.0.1:" in message
        for message in records
    ), records


def test_four_clients_are_served(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):

        async def scenario() -> None:
            async with aiohttp.ClientSession() as session:

                async def ask(message_id: int) -> Any:
                    async with session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws:
                        assert (await receive(ws))["type"] == "hello"
                        await ws.send_json({"v": 1, "type": "ping", "data": {"id": message_id}})
                        while True:
                            message = await receive(ws)
                            if message["type"] == "pong":
                                return message["data"]["id"]

                return await asyncio.gather(*(ask(index) for index in range(4)))

        assert run(scenario) == [0, 1, 2, 3]


def test_idle_connection_is_closed_after_the_timeout(tmp_path: Path) -> None:
    with running_server(tmp_path, idle_timeout_s=0.2) as (server, _):

        async def scenario() -> None:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                assert (await receive(ws))["type"] == "hello"
                message = await asyncio.wait_for(ws.receive(), timeout=TIMEOUT_S)
                assert message.type is WSMsgType.CLOSE
                assert ws.closed

        run(scenario)


def test_health_and_static_files_are_served(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):

        async def scenario() -> None:
            async with aiohttp.ClientSession() as session:
                base = f"http://127.0.0.1:{server.port}"
                health = await session.get(f"{base}/health")
                index = await session.get(f"{base}/")
                asset = await session.get(f"{base}/assets/app.js")
                missing = await session.get(f"{base}/assets/nope.js")
                return (
                    health.status,
                    await health.json(),
                    index.status,
                    await index.text(),
                    asset.status,
                    await asset.text(),
                    missing.status,
                )

        code, health, index_status, index_text, asset_status, asset_text, missing = run(scenario)
        assert (code, index_status, asset_status, missing) == (200, 200, 200, 404)
        assert health == {"mode": "idle", "mode_state": "stopped", "uptime_s": 0.0}
        assert "cpsa ui" in index_text
        assert "console.log" in asset_text


def test_health_reports_the_latest_status(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):
        publish_state(server, "status", status_data(mode="medical", uptime_s=12.5))

        async def scenario() -> Any:
            async with aiohttp.ClientSession() as session:
                response = await session.get(f"http://127.0.0.1:{server.port}/health")
                return await response.json()

        assert run(scenario) == {
            "mode": "medical",
            "mode_state": "running",
            "uptime_s": 12.5,
        }


def test_oversized_message_is_dropped(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):
        publish_state(server, "status", status_data())

        async def scenario() -> None:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                assert (await receive(ws))["type"] == "hello"
                assert (await receive(ws))["type"] == "status"
                publish_state(
                    server,
                    "status",
                    {**status_data(), "blob": "x" * MAX_MESSAGE_BYTES},
                )
                await ws.send_json({"v": 1, "type": "ping", "data": {"id": 9}})
                message = await receive(ws)

            assert message["type"] == "pong"
            assert message["data"]["id"] == 9

        run(scenario)


def padded_ping(size: int) -> str:
    """A valid ``ping`` envelope of exactly ``size`` UTF-8 bytes (IF-104)."""
    placeholder = json.dumps({"v": 1, "type": "ping", "seq": 1, "data": {"id": 1, "pad": ""}})
    payload = json.dumps(
        {
            "v": 1,
            "type": "ping",
            "seq": 1,
            "data": {"id": 1, "pad": "x" * (size - len(placeholder))},
        }
    )
    assert len(payload.encode("utf-8")) == size
    return payload


def test_a_client_message_above_the_limit_closes_the_connection(tmp_path: Path) -> None:
    """R-01: IF-104 caps every message, not only the ones the server sends."""
    with running_server(tmp_path) as (server, commands):

        async def scenario() -> tuple[dict[str, Any], Any]:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                assert (await receive(ws))["type"] == "hello"
                await ws.send_str(padded_ping(MAX_MESSAGE_BYTES))
                pong = await receive(ws)
                await ws.send_str(padded_ping(MAX_MESSAGE_BYTES + 1))
                oversized = await asyncio.wait_for(ws.receive(), timeout=TIMEOUT_S)
                return pong, oversized

        pong, oversized = run(scenario)
        assert pong["type"] == "pong"
        assert oversized.type is WSMsgType.CLOSE
        assert oversized.data == WSCloseCode.MESSAGE_TOO_BIG
        assert commands == []


def test_publications_before_the_loop_runs_are_not_lost(tmp_path: Path) -> None:
    """R-08: a state published before start() must reach the first snapshot."""
    static = tmp_path / "dist"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>cpsa ui</title>", encoding="utf-8")
    config = ServerConfig(
        host="127.0.0.1",
        port=0,
        static_dir=static,
        idle_timeout_s=30.0,
    )
    server = WebServer(config, lambda command, data, ref_seq: None, "test-version")
    server.state("status", status_data())
    server.start()
    try:

        async def scenario() -> list[dict[str, Any]]:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                return [await receive(ws) for _ in range(2)]

        messages = run(scenario)
        assert [message["type"] for message in messages] == ["hello", "status"]
        assert [message["seq"] for message in messages] == [1, 2]
        assert messages[1]["data"]["mode"] == "medical"
    finally:
        server.stop()


def test_states_of_a_mode_that_stopped_are_forgotten(tmp_path: Path) -> None:
    """R-10: a snapshot must not replay states the current mode does not serve."""
    with running_server(tmp_path) as (server, _):
        publish_state(server, "status", status_data(mode="medical"))
        publish_state(server, "medical.state", MEDICAL_STATE)

        async def scenario() -> list[str]:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                first = [await receive(ws) for _ in range(3)]
                assert [message["type"] for message in first] == [
                    "hello",
                    "status",
                    "medical.state",
                ]
                clients = list(server._clients)
                publish_state(server, "status", status_data(mode="idle"))
                idle = await receive(ws)
                assert (idle["type"], idle["data"]["mode"]) == ("status", "idle")
                wait_until(lambda: all(not c.states for c in clients))
                return [topic for topic, _ in server._snapshot()]

        assert run(scenario) == ["status"]


class AbortableTransport:
    """Stand-in for a TCP transport, to observe the abort after a stuck close."""

    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


class BlockingSocket:
    """A WebSocket stand-in whose sender blocks, like a client that never reads."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.sent: list[str] = []
        self.closed = False

    async def send_str(self, payload: str) -> None:
        await self.gate.wait()
        self.sent.append(payload)

    async def close(self) -> None:
        # The peer never reads, so closing does not release a pending send.
        self.closed = True


class HangingCloseSocket(BlockingSocket):
    """A socket whose close handshake never finishes, like a half-open peer."""

    async def close(self) -> None:
        await asyncio.Event().wait()


def attach_socket(server: WebServer, socket: BlockingSocket, transport: Any = None) -> Any:
    """Attach a stand-in socket on the server's loop and return its client."""
    loop = server._loop
    assert loop is not None
    attached = threading.Event()
    holder: dict[str, Any] = {}

    def attach() -> None:
        holder["client"] = server._attach(socket, transport)
        attached.set()

    loop.call_soon_threadsafe(attach)
    assert attached.wait(TIMEOUT_S)
    return holder["client"]


def test_dropping_a_client_cancels_its_blocked_sender(tmp_path: Path) -> None:
    """R-09: a sender stuck in send_str must not outlive the client."""
    with running_server(tmp_path) as (server, _):
        loop = server._loop
        assert loop is not None
        client = attach_socket(server, BlockingSocket())

        async def drop() -> None:
            await server._drop(client, "test")

        asyncio.run_coroutine_threadsafe(drop(), loop)
        wait_until(lambda: client.sender.done())
        assert client.sender.cancelled()
        assert client not in server._clients


def test_a_close_that_does_not_finish_aborts_the_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-09: a half-open peer must not keep a blocked close task and socket alive."""
    monkeypatch.setattr(server_module, "CLOSE_TIMEOUT_S", 0.05)
    with running_server(tmp_path) as (server, _):
        loop = server._loop
        assert loop is not None
        transport = AbortableTransport()
        client = attach_socket(server, HangingCloseSocket(), transport)

        async def drop() -> None:
            await server._drop(client, "test")

        asyncio.run_coroutine_threadsafe(drop(), loop).result(TIMEOUT_S)
        assert transport.aborted
        assert client.closed


WEBSOCKET_HANDSHAKE = (
    b"GET /ws HTTP/1.1\r\n"
    b"Host: 127.0.0.1\r\n"
    b"Upgrade: websocket\r\n"
    b"Connection: Upgrade\r\n"
    b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
    b"Sec-WebSocket-Version: 13\r\n"
    b"\r\n"
)


def test_a_stalled_peer_on_a_real_connection_is_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-09: on a real aiohttp connection, the request's transport is aborted.

    The peer completes the WebSocket handshake and then never reads, so the
    server's writes pause and even the close frame cannot be flushed: the close
    handshake runs into ``CLOSE_TIMEOUT_S``. The abort is recorded on the real
    TCP transport that the handler took from ``request.transport``; a stand-in
    that fabricates the attribute would not prove that path.
    """
    monkeypatch.setattr(server_module, "CLOSE_TIMEOUT_S", 0.05)
    with running_server(tmp_path) as (server, _):

        async def scenario() -> None:
            peer = socket_lib.socket(socket_lib.AF_INET, socket_lib.SOCK_STREAM)
            peer.setsockopt(socket_lib.SOL_SOCKET, socket_lib.SO_RCVBUF, 4096)
            await asyncio.get_running_loop().sock_connect(peer, ("127.0.0.1", server.port))
            reader, writer = await asyncio.open_connection(sock=peer)
            try:
                writer.write(WEBSOCKET_HANDSHAKE)
                await writer.drain()
                head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), TIMEOUT_S)
                assert head.split(b"\r\n", 1)[0] == b"HTTP/1.1 101 Switching Protocols"

                wait_until(lambda: len(server._clients) == 1)
                client = next(iter(server._clients))
                transport = client.transport
                assert transport is not None
                # The real aiohttp shape: the transport lives on the request.
                assert transport is client.ws._req.transport

                aborted = threading.Event()
                original_abort = transport.abort

                def abort() -> None:
                    aborted.set()
                    original_abort()

                transport.abort = abort

                loop = server._loop
                assert loop is not None

                def stall() -> None:
                    # A tiny send buffer and a peer that never reads fill the
                    # transport's buffer until the protocol pauses writing.
                    sock = transport.get_extra_info("socket")
                    sock.setsockopt(socket_lib.SOL_SOCKET, socket_lib.SO_SNDBUF, 4096)
                    for _ in range(8):
                        server.event(
                            "mode.progress",
                            {
                                "mode": "medical",
                                "step": "ready",
                                "error": None,
                                "pad": "x" * 40000,
                            },
                        )

                loop.call_soon_threadsafe(stall)
                wait_until(lambda: transport.get_write_buffer_size() > 64 * 1024)

                future = asyncio.run_coroutine_threadsafe(server._drop(client, "test"), loop)
                future.result(TIMEOUT_S)

                assert aborted.is_set()
                assert client.closed
                assert client not in server._clients
                with contextlib.suppress(ConnectionResetError, TimeoutError):
                    while await asyncio.wait_for(reader.read(4096), TIMEOUT_S):
                        pass
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

        run(scenario)


def test_slow_client_coalesces_states_and_is_disconnected_on_event_overflow(
    tmp_path: Path,
) -> None:
    with running_server(tmp_path) as (server, _):
        attached = threading.Event()
        holder: dict[str, Any] = {}
        loop = server._loop
        assert loop is not None

        def attach() -> None:
            socket = BlockingSocket()
            holder["socket"] = socket
            holder["client"] = server._attach(socket)
            attached.set()

        loop.call_soon_threadsafe(attach)
        assert attached.wait(TIMEOUT_S)
        client = holder["client"]

        for uptime in range(20):
            server.state("status", status_data(uptime_s=float(uptime)))
        wait_until(lambda: len(client.states) == 1 and client.states["status"]["uptime_s"] == 19.0)

        async def scenario() -> None:
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{server.port}/ws") as ws,
            ):
                assert (await receive(ws))["type"] == "hello"
                server.event(
                    "mode.progress",
                    {"mode": "medical", "step": "ready", "error": None},
                )

                async def drain(count: int) -> list[str]:
                    seen = []
                    while len(seen) < count:
                        message = await receive(ws)
                        if message["type"] == "mode.progress":
                            seen.append(message["data"]["step"])
                    return seen

                # Publish in chunks the reading client can absorb (at most 100
                # pending events each) so only the blocked stand-in overflows
                # its 256-event queue and the outcome does not depend on how
                # many `_handle_event` callbacks the loop runs before the real
                # client's sender gets a turn (R-23).
                assert await drain(1) == ["ready"]
                for _ in range(3):
                    for _ in range(100):
                        server.event(
                            "mode.progress",
                            {"mode": "medical", "step": "ready", "error": None},
                        )
                    assert len(await drain(100)) == 100

        run(scenario)
        wait_until(lambda: client.closed)
        assert holder["socket"].closed
        assert client not in server._clients


def test_publisher_drops_events_after_stop(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):
        pass
    # Publishing after stop must not raise or block.
    server.state("status", status_data())
    server.event("mode.progress", {"mode": "medical", "step": "ready", "error": None})


# ---------------------------------------------------------------------------
# F14: /video.mjpeg
# ---------------------------------------------------------------------------


def camera_frame(seq: int, width: int = 64, height: int = 48) -> Any:
    """A small RGB frame with a moving block, so different seqs give different JPEGs."""
    import numpy as np

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 40
    image[8:24, 4 + (seq % 8) * 4 : 24 + (seq % 8) * 4, 1] = 220
    return image


def decode_jpeg(payload: bytes) -> Any:
    import cv2
    import numpy as np

    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)


async def read_video_part(
    response: aiohttp.ClientResponse, timeout: float = TIMEOUT_S
) -> tuple[dict[str, str], bytes]:
    """Read one multipart part: its headers and its JPEG payload."""
    header_block = await asyncio.wait_for(response.content.readuntil(b"\r\n\r\n"), timeout=timeout)
    header_lines = header_block.decode("ascii").split("\r\n")
    assert header_lines[0] == f"--{server_module.VIDEO_BOUNDARY}", header_lines[0]
    headers: dict[str, str] = {}
    for line in header_lines[1:]:
        if not line:
            continue
        name, _, value = line.partition(": ")
        headers[name] = value
    length = int(headers["Content-Length"])
    payload = await asyncio.wait_for(response.content.readexactly(length), timeout=timeout)
    assert await asyncio.wait_for(response.content.readexactly(2), timeout=timeout) == b"\r\n"
    return headers, payload


def test_video_stream_sends_one_multipart_part_per_offered_frame(tmp_path: Path) -> None:
    with running_server(tmp_path, video=True) as (server, _):

        async def scenario() -> tuple[dict[str, str], bytes]:
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"http://127.0.0.1:{server.port}/video.mjpeg") as response,
            ):
                assert response.status == 200
                assert response.headers["Content-Type"] == (
                    f"multipart/x-mixed-replace; boundary={server_module.VIDEO_BOUNDARY}"
                )
                assert response.headers["Cache-Control"] == "no-store"
                wait_until(lambda: server._video_clients == 1)
                server.offer_video_frame(1, camera_frame(1))
                return await read_video_part(response)

        headers, payload = run(scenario)

    assert headers["Content-Type"] == "image/jpeg"
    assert headers["X-Frame-Seq"] == "1"
    assert int(headers["Content-Length"]) == len(payload)
    image = decode_jpeg(payload)
    assert image is not None
    assert image.shape == (48, 64, 3)


def test_video_stream_serves_a_second_frame_after_the_first(tmp_path: Path) -> None:
    with running_server(tmp_path, video=True) as (server, _):

        async def scenario() -> list[tuple[dict[str, str], bytes]]:
            parts: list[tuple[dict[str, str], bytes]] = []
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"http://127.0.0.1:{server.port}/video.mjpeg") as response,
            ):
                wait_until(lambda: server._video_clients == 1)
                for seq in (1, 2):
                    server.offer_video_frame(seq, camera_frame(seq))
                    parts.append(await read_video_part(response))
            return parts

        parts = run(scenario)

    assert [int(headers["X-Frame-Seq"]) for headers, _ in parts] == [1, 2]
    assert all(decode_jpeg(payload) is not None for _, payload in parts)


def test_video_endpoint_is_404_json_when_disabled(tmp_path: Path) -> None:
    with running_server(tmp_path) as (server, _):

        async def scenario() -> tuple[int, dict[str, Any]]:
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"http://127.0.0.1:{server.port}/video.mjpeg") as response,
            ):
                return response.status, await response.json()

        status, body = run(scenario)

    assert status == 404
    assert body == {"error": "video_disabled"}


def test_offered_frames_are_ignored_without_a_client(tmp_path: Path) -> None:
    with running_server(tmp_path, video=True) as (server, _):
        server.offer_video_frame(1, camera_frame(1))
        time.sleep(0.05)
        assert server._video_latest is None
        assert server._video_frame is None


def test_the_video_request_ends_when_the_client_disconnects(tmp_path: Path) -> None:
    with running_server(tmp_path, video=True) as (server, _):

        async def scenario() -> None:
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"http://127.0.0.1:{server.port}/video.mjpeg") as response,
            ):
                assert response.status == 200
                response.close()

        run(scenario)
        wait_until(lambda: server._video_clients == 0)
        wait_until(lambda: server._video_encoder is None)
