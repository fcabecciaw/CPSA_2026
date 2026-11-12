"""HTTP and WebSocket server for the demo UI (IF-101..205, F14).

One port serves the static frontend, ``GET /health``, ``GET /video.mjpeg`` and the
``/ws`` WebSocket, in a dedicated asyncio thread. The server is also the
implementation of the :class:`cpsa.ports.Publisher` port: producers call
:meth:`WebServer.state` and :meth:`WebServer.event` from any thread, and each client
gets its own envelope, ``seq`` counter, latest-wins state map and bounded event
queue.

The video stream is fed by :meth:`WebServer.offer_video_frame` from the vision
thread: the newest offered frame is encoded once, off the event loop, and shared by
every connected client at up to :data:`VIDEO_MAX_FPS` frames per second.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from aiohttp import WSMsgType, web
from numpy.typing import NDArray

from cpsa.protocol import PROTOCOL_VERSION, system

logger = logging.getLogger("cpsa.web")

MAX_MESSAGE_BYTES = 64 * 1024
"""A WebSocket message may not exceed this many UTF-8 bytes (IF-104)."""

MAX_INBOUND_MSG_SIZE = MAX_MESSAGE_BYTES + 1
"""Parser limit handed to aiohttp.

aiohttp rejects a frame whose declared length is *greater than or equal to*
``max_msg_size``, so the parser limit is one byte above the protocol limit: a
message of exactly :data:`MAX_MESSAGE_BYTES` is still accepted.
"""

MAX_PENDING_EVENTS = 256
"""Per-client event queue bound; a full queue disconnects the client (IF-205)."""

MAX_PENDING_PUBLISH = 256
"""Bound on publications buffered until the event loop thread is running."""

CLOSE_TIMEOUT_S = 1.0
"""How long a close handshake may take before the transport is aborted."""

LOG_REPLAY_SIZE = 50
"""How many recent ``log`` events are replayed to a newly connected client."""

VIDEO_BOUNDARY = "cpsaframe"
"""Multipart boundary of ``/video.mjpeg`` (F14)."""

VIDEO_MAX_FPS = 30.0
"""Encoded frames per second shared by every video client (F14)."""

VIDEO_JPEG_QUALITY = 70
"""JPEG quality of the streamed frames (F14)."""

VIDEO_DISCONNECT_CHECK_S = 0.5
"""How often an idle video response checks that its client is still connected."""

MODES = ("idle", "medical", "hologram", "rps", "showcase")
CAPABILITIES = (
    "status",
    "medical.state",
    "medical.decision",
    "medical.episode",
    "medical.hands",
    "hologram.hands",
    "rps.state",
    "showcase.state",
    "log",
)
DEFAULT_STATUS: Mapping[str, Any] = {
    "mode": "idle",
    "mode_state": "stopped",
    "uptime_s": 0.0,
}


class ServerSettings(Protocol):
    """The subset of ``cpsa.application.config.ServerConfig`` the adapter needs.

    The adapter layer may not import the application layer, so the settings are
    described structurally; ``ServerConfig`` satisfies this protocol.
    """

    @property
    def host(self) -> str: ...

    @property
    def port(self) -> int: ...

    @property
    def static_dir(self) -> Path: ...

    @property
    def idle_timeout_s(self) -> float: ...

    @property
    def video(self) -> bool: ...


@dataclass(eq=False)
class _Client:
    """One connected WebSocket and its outgoing queues."""

    ws: Any
    transport: Any = None
    """The TCP transport of the connection; ``None`` for a socket stand-in."""
    peer: str = "unknown"
    """``host:port`` of the connected peer, for command logging."""
    seq: int = 0
    states: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    events: deque[tuple[str, Mapping[str, Any]]] = field(default_factory=deque)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    last_seen: float = field(default_factory=time.monotonic)
    closed: bool = False
    sender: asyncio.Task[None] | None = None


class WebServer:
    """Serve the UI and the protocol on one port, and publish to every client."""

    def __init__(
        self,
        config: ServerSettings,
        on_command: Callable[[str, dict[str, Any], int | None], None],
        server_version: str,
    ) -> None:
        self._config = config
        self._on_command = on_command
        self._server_version = server_version
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._watchdog: asyncio.Task[None] | None = None
        self._clients: set[_Client] = set()
        self._states: dict[str, Mapping[str, Any]] = {}
        # Recent log lines are replayed after the state snapshot, so a dashboard
        # that reconnects shows what happened while it was away.
        self._log_replay: deque[tuple[str, Mapping[str, Any]]] = deque(maxlen=LOG_REPLAY_SIZE)
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._stopping = False
        self._port = config.port
        # Publications made before the event loop is running would otherwise be
        # lost (R-08); they wait here until _mark_ready flushes them.
        self._pending_lock = threading.Lock()
        self._pending: deque[tuple[Callable[..., None], tuple[object, ...]]] = deque(
            maxlen=MAX_PENDING_PUBLISH
        )
        self._loop_running = False
        # /video.mjpeg (F14): the newest offered frame, the encoded frame shared by
        # every client, and the encoder task that runs at most VIDEO_MAX_FPS.
        self._video_enabled = bool(config.video)
        self._video_wake: asyncio.Event | None = None
        self._video_waiters: set[asyncio.Event] = set()
        self._video_encoder: asyncio.Task[None] | None = None
        self._video_clients = 0
        self._video_latest: tuple[int, NDArray[np.uint8]] | None = None
        self._video_frame: tuple[int, bytes] | None = None
        self._video_next_encode_at = 0.0

    @property
    def port(self) -> int:
        """The bound port (useful when the configuration asked for port 0)."""
        return self._port

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the event loop thread and return once the port is bound."""
        if self._thread is not None:
            raise RuntimeError("web server is already started")
        self._ready.clear()
        self._start_error = None
        self._stopping = False
        with self._pending_lock:
            # Publications made before start() are kept: _mark_ready flushes them
            # into the running loop (R-08).
            self._loop_running = False
        self._thread = threading.Thread(target=self._run, name="cpsa-web", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("web server did not start within 10 s")
        if self._start_error is not None:
            self._thread = None
            raise RuntimeError(
                f"web server cannot listen on {self._config.host}:{self._config.port}: "
                f"{self._start_error}"
            ) from self._start_error

    def stop(self, timeout: float = 5.0) -> None:
        """Close every client and stop the event loop thread."""
        thread = self._thread
        loop = self._loop
        if thread is None or loop is None:
            return
        if loop.is_running():
            loop.call_soon_threadsafe(self._begin_stop)
        thread.join(timeout)
        if thread.is_alive():
            logger.error("Web server did not stop within %.1f s", timeout)
        self._thread = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._start_async())
        except BaseException as exc:  # reported through start()
            self._start_error = exc
            self._ready.set()
        if self._start_error is not None:
            loop.close()
            return
        # _ready is set from inside the running loop, so whoever waits for
        # start() can publish without the message being dropped.
        loop.call_soon(self._mark_ready)
        try:
            loop.run_forever()
        finally:
            loop.close()

    def _mark_ready(self) -> None:
        """Flush publications buffered before the loop ran, then release start()."""
        with self._pending_lock:
            pending = list(self._pending)
            self._pending.clear()
            self._loop_running = True
        for callback, args in pending:
            callback(*args)
        self._ready.set()

    async def _start_async(self) -> None:
        loop = asyncio.get_running_loop()
        self._video_wake = asyncio.Event()
        self._video_waiters.clear()
        self._video_latest = None
        self._video_frame = None
        self._video_clients = 0
        self._video_next_encode_at = 0.0
        self._runner = web.AppRunner(self._build_app(), access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._config.host, self._config.port)
        await self._site.start()
        self._port = self._site.port
        self._watchdog = loop.create_task(self._watchdog_loop())

    def _begin_stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        loop = asyncio.get_running_loop()
        loop.create_task(self._shutdown_async())

    async def _shutdown_async(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog
            self._watchdog = None
        # Wake the video responses so they observe _stopping and return; the runner
        # cleanup below waits for them, so none outlives this loop.
        for waiter in self._video_waiters:
            waiter.set()
        await self._stop_video_encoder()
        for client in list(self._clients):
            await self._drop(client, "server stopping")
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        asyncio.get_running_loop().stop()

    # -- Publisher port ----------------------------------------------------

    def state(self, type: str, data: Mapping[str, Any]) -> None:
        """Replace the latest state of ``type`` for every client (IF-202, IF-205)."""
        payload = dict(data)
        self._call_soon(self._handle_state, type, payload)

    def event(self, type: str, data: Mapping[str, Any]) -> None:
        """Queue an event for every client; overflow disconnects that client."""
        payload = dict(data)
        self._call_soon(self._handle_event, type, payload)

    def _call_soon(self, callback: Callable[..., None], *args: object) -> None:
        loop = self._loop
        if self._stopping or (loop is not None and loop.is_closed()):
            return
        with self._pending_lock:
            if not self._loop_running:
                self._pending.append((callback, args))
                return
        assert loop is not None
        try:
            loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            logger.debug("Dropping a message published after the server stopped")

    def _handle_state(self, type: str, data: Mapping[str, Any]) -> None:
        self._states[type] = data
        if type == "status":
            self._forget_states_outside_mode()
        for client in list(self._clients):
            if client.closed:
                continue
            client.states[type] = data
            client.wake.set()

    def _forget_states_outside_mode(self) -> None:
        """Forget states the current mode does not serve (IF-202, R-10).

        Without this, a state left behind by a mode that stopped or failed to
        start would be replayed to the next connection and kept by live clients
        until the mode publishes a replacement.
        """
        mode = str(self._states.get("status", DEFAULT_STATUS).get("mode", "idle"))
        stale = [
            topic
            for topic in self._states
            if topic != "status" and not self._applies_to(topic, mode)
        ]
        for topic in stale:
            del self._states[topic]
            for client in list(self._clients):
                client.states.pop(topic, None)

    def _handle_event(self, type: str, data: Mapping[str, Any]) -> None:
        if type == "log":
            self._log_replay.append((type, data))
        for client in list(self._clients):
            if client.closed:
                continue
            if len(client.events) >= MAX_PENDING_EVENTS:
                logger.error("Disconnecting a slow client: event queue is full")
                self._spawn_drop(client, "event queue overflow")
                continue
            client.events.append((type, data))
            client.wake.set()

    def _spawn_drop(self, client: _Client, reason: str) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.create_task(self._drop(client, reason))

    # -- HTTP routes -------------------------------------------------------

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/health", self._health)
        app.router.add_get("/video.mjpeg", self._video)
        app.router.add_get("/assets/{path:.*}", self._asset)
        app.router.add_get("/ws", self._websocket)
        return app

    async def _index(self, request: web.Request) -> web.StreamResponse:
        return await self._file(self._config.static_dir / "index.html")

    async def _asset(self, request: web.Request) -> web.StreamResponse:
        assets = self._config.static_dir / "assets"
        root = assets.resolve()
        candidate = (root / request.match_info["path"]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return web.Response(status=404, text="not found")
        return await self._file(candidate)

    async def _file(self, path: Path) -> web.StreamResponse:
        if not path.is_file():
            return web.Response(status=404, text="not found")
        return web.FileResponse(path)

    async def _health(self, request: web.Request) -> web.Response:
        status = self._states.get("status", DEFAULT_STATUS)
        return web.json_response(
            {
                "mode": status.get("mode", DEFAULT_STATUS["mode"]),
                "mode_state": status.get("mode_state", DEFAULT_STATUS["mode_state"]),
                "uptime_s": status.get("uptime_s", DEFAULT_STATUS["uptime_s"]),
            }
        )

    # -- video stream (F14) ------------------------------------------------

    def offer_video_frame(self, seq: int, rgb: NDArray[np.uint8]) -> None:
        """Offer one analyzed camera frame to ``/video.mjpeg``; never blocks (F14).

        Called from the vision verifier's thread with the decoded, unmirrored
        ``(H, W, 3)`` RGB image. The array is kept by reference: only the event loop
        encodes it, in an executor, and shares the JPEG with every client. Frames are
        ignored when the stream is disabled or no client is connected.
        """
        if not self._video_enabled:
            return
        loop = self._loop
        if loop is None or loop.is_closed() or self._stopping:
            return
        try:
            loop.call_soon_threadsafe(self._accept_video_frame, int(seq), rgb)
        except RuntimeError:
            logger.debug("Dropping a video frame offered after the server stopped")

    def _accept_video_frame(self, seq: int, rgb: NDArray[np.uint8]) -> None:
        """Keep the newest offered frame (event loop thread)."""
        if self._stopping or self._video_clients <= 0:
            return
        self._video_latest = (seq, rgb)
        wake = self._video_wake
        if wake is not None:
            wake.set()

    async def _video(self, request: web.Request) -> web.StreamResponse:
        """Stream the analyzed camera frames as ``multipart/x-mixed-replace`` (F14)."""
        if not self._video_enabled:
            return web.json_response({"error": "video_disabled"}, status=404)
        response = web.StreamResponse(
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={VIDEO_BOUNDARY}",
                "Cache-Control": "no-store",
            }
        )
        await response.prepare(request)
        self._video_clients += 1
        self._ensure_video_encoder()
        seen = 0
        try:
            while not self._stopping:
                frame = await self._next_video_frame(seen, request)
                if frame is None:
                    break
                seen = frame[0]
                await self._write_video_part(response, seen, frame[1])
        except ConnectionError:
            logger.debug("Video client disconnected", exc_info=True)
        finally:
            self._video_clients -= 1
            if self._video_clients <= 0:
                await self._stop_video_encoder()
        return response

    async def _next_video_frame(self, seen: int, request: web.Request) -> tuple[int, bytes] | None:
        """Wait for an encoded frame newer than ``seen``; ``None`` when the client must go."""
        while not self._stopping:
            if request.transport is None:
                return None
            frame = self._video_frame
            if frame is not None and frame[0] > seen:
                return frame
            waiter = asyncio.Event()
            self._video_waiters.add(waiter)
            try:
                frame = self._video_frame
                if frame is not None and frame[0] > seen:
                    return frame
                with contextlib.suppress(asyncio.TimeoutError):
                    # The timeout only re-checks a client that disconnected while no
                    # frame was available: aiohttp surfaces that on the next write.
                    await asyncio.wait_for(waiter.wait(), timeout=VIDEO_DISCONNECT_CHECK_S)
            finally:
                self._video_waiters.discard(waiter)
        return None

    @staticmethod
    async def _write_video_part(response: web.StreamResponse, seq: int, jpeg: bytes) -> None:
        """Write one multipart part and let aiohttp drain it (F14)."""
        header = (
            f"--{VIDEO_BOUNDARY}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n"
            f"X-Frame-Seq: {seq}\r\n\r\n"
        ).encode("ascii")
        # aiohttp 3.14 made write() awaitable (and drains itself); 3.9 returns None
        # and needs an explicit drain.
        written = response.write(header + jpeg + b"\r\n")
        if inspect.isawaitable(written):
            await written
        else:
            await response.drain()

    def _ensure_video_encoder(self) -> None:
        task = self._video_encoder
        if task is None or task.done():
            loop = asyncio.get_running_loop()
            self._video_encoder = loop.create_task(self._video_encoder_loop())

    async def _stop_video_encoder(self) -> None:
        task = self._video_encoder
        self._video_encoder = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _video_encoder_loop(self) -> None:
        """Encode the newest offered frame at up to ``VIDEO_MAX_FPS`` (F14).

        Runs in the event loop only to start and pace the work: the JPEG encoding
        itself happens in the default executor, so a slow encode never stalls the
        WebSocket clients.
        """
        loop = asyncio.get_running_loop()
        wake = self._video_wake
        if wake is None:
            return
        while True:
            offer = self._video_latest
            if offer is None:
                wake.clear()
                await wake.wait()
                continue
            self._video_latest = None
            seq, rgb = offer
            delay = self._video_next_encode_at - loop.time()
            if delay > 0.0:
                await asyncio.sleep(delay)
            try:
                jpeg = await loop.run_in_executor(None, _encode_jpeg, rgb)
            except Exception:
                logger.exception("Encoding a video frame failed")
                continue
            self._video_next_encode_at = loop.time() + 1.0 / VIDEO_MAX_FPS
            self._video_frame = (seq, jpeg)
            for waiter in self._video_waiters:
                waiter.set()

    # -- WebSocket ---------------------------------------------------------

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        # IF-104: the inbound parser rejects anything above the limit and closes
        # the connection with code 1009.
        ws = web.WebSocketResponse(max_msg_size=MAX_INBOUND_MSG_SIZE)
        transport = request.transport
        await ws.prepare(request)
        client = self._attach(ws, transport)
        try:
            async for message in ws:
                client.last_seen = time.monotonic()
                if message.type is WSMsgType.TEXT:
                    self._handle_command(client, message.data)
                elif message.type in (
                    WSMsgType.ERROR,
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSING,
                    WSMsgType.CLOSED,
                ):
                    break
        finally:
            await self._drop(client, "connection closed")
        return ws

    def _attach(self, ws: Any, transport: Any = None) -> _Client:
        client = _Client(ws=ws, transport=transport, peer=_peer_address(transport))
        self._clients.add(client)
        loop = asyncio.get_running_loop()
        client.sender = loop.create_task(self._sender(client))
        return client

    async def _sender(self, client: _Client) -> None:
        try:
            if not await self._send(
                client,
                "hello",
                system.hello(self._server_version, CAPABILITIES, MODES, ()),
            ):
                return
            for topic, data in self._snapshot():
                if not await self._send(client, topic, data):
                    return
            while not client.closed:
                while client.states and not client.closed:
                    topic, data = client.states.popitem()
                    if not await self._send(client, topic, data):
                        return
                while client.events and not client.closed:
                    topic, data = client.events.popleft()
                    if not await self._send(client, topic, data):
                        return
                client.wake.clear()
                await client.wake.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("WebSocket sender failed")
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                await self._drop(client, "sender finished")

    def _snapshot(self) -> list[tuple[str, Mapping[str, Any]]]:
        mode = str(self._states.get("status", DEFAULT_STATUS).get("mode", "idle"))
        states = [
            (topic, data)
            for topic, data in list(self._states.items())
            if self._applies_to(topic, mode)
        ]
        # The recent log history follows the snapshot, oldest line first, so a
        # reconnecting dashboard continues the timeline it was showing.
        return states + list(self._log_replay)

    @staticmethod
    def _applies_to(topic: str, mode: str) -> bool:
        if topic == "status":
            return True
        if topic.startswith("medical."):
            return mode == "medical"
        if topic.startswith("lab."):
            return mode == "lab"
        if topic.startswith("hologram."):
            return mode == "hologram"
        if topic.startswith("rps."):
            return mode == "rps"
        if topic.startswith("showcase."):
            return mode == "showcase"
        return True

    async def _send(self, client: _Client, topic: str, data: Mapping[str, Any]) -> bool:
        if client.closed:
            return False
        message = system.envelope(topic, data, client.seq + 1, time.monotonic(), _wall())
        try:
            payload = json.dumps(message, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            logger.exception("Cannot serialise a %s message; dropping it", topic)
            return True
        size = len(payload.encode("utf-8"))
        if size > MAX_MESSAGE_BYTES:
            logger.error(
                "Dropping a %s message of %d bytes (limit %d)", topic, size, MAX_MESSAGE_BYTES
            )
            return True
        try:
            await client.ws.send_str(payload)
        except Exception:
            logger.debug("WebSocket send failed; dropping the client", exc_info=True)
            return False
        client.seq += 1
        return True

    async def _drop(self, client: _Client, reason: str) -> None:
        if client.closed:
            return
        client.closed = True
        self._clients.discard(client)
        client.wake.set()
        if reason != "sender finished":
            logger.info("Closing a WebSocket client: %s", reason)
        try:
            # Close before cancelling the sender: cancelling a sender parked on a
            # paused drain would leave aiohttp's drain waiter cancelled and the
            # close would fail at once instead of into the timeout (R-09).
            await asyncio.wait_for(client.ws.close(), timeout=CLOSE_TIMEOUT_S)
        except asyncio.TimeoutError:
            # On Python 3.10 asyncio.TimeoutError is not the builtin TimeoutError,
            # so the alias must be spelled out (R-09).
            logger.warning("WebSocket close handshake timed out; aborting the transport")
            _abort_transport(client.transport)
        except Exception:
            logger.debug("Closing a WebSocket client failed", exc_info=True)
        finally:
            sender = client.sender
            if sender is not None and sender is not asyncio.current_task() and not sender.done():
                # A sender parked on ws.send_str against a peer that stopped
                # reading would otherwise live until the transport closes (R-09).
                sender.cancel()

    async def _watchdog_loop(self) -> None:
        timeout_s = self._config.idle_timeout_s
        interval = max(0.01, min(timeout_s / 2.0, 1.0))
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            for client in list(self._clients):
                if now - client.last_seen > timeout_s:
                    await self._drop(client, "idle timeout")

    # -- commands ----------------------------------------------------------

    def _handle_command(self, client: _Client, text: str) -> None:
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            self._bad_command(client, None, "invalid JSON")
            return
        if not isinstance(message, dict):
            self._bad_command(client, None, "invalid client envelope")
            return
        try:
            ref_seq = _command_ref_seq(message)
        except ValueError:
            self._bad_command(client, None, "invalid client envelope")
            return
        if message.get("v") != PROTOCOL_VERSION:
            self._bad_command(client, ref_seq, "invalid client envelope")
            return
        command = message.get("type")
        data = message.get("data")
        if not isinstance(command, str) or not isinstance(data, dict):
            self._bad_command(client, ref_seq, "invalid client envelope")
            return
        # Every client command is logged at INFO with its peer, so an action
        # such as a mode stop can be attributed after the fact.
        logger.info(
            "Client command type=%s data=%s seq=%s peer=%s",
            command,
            data,
            ref_seq,
            client.peer,
        )
        if command == "ping":
            if "id" not in data:
                self._bad_command(client, ref_seq, "ping requires data.id")
                return
            self._queue_event(client, "pong", system.pong(data["id"]))
        elif command == "mode.set":
            if data.get("mode") not in MODES:
                self._bad_command(client, ref_seq, "mode.set requires a known mode")
                return
            try:
                self._on_command(command, dict(data), ref_seq)
            except Exception:
                logger.exception("Command handler failed for %s", command)
                self._queue_event(
                    client,
                    "error",
                    system.error("internal", "command handler failed", ref_seq),
                )
        else:
            self._bad_command(client, ref_seq, f"unknown command {command!r}")

    def _bad_command(self, client: _Client, ref_seq: int | None, message: str) -> None:
        self._queue_event(client, "error", system.error("bad_command", message, ref_seq))

    def _queue_event(self, client: _Client, topic: str, data: Mapping[str, Any]) -> None:
        if client.closed:
            return
        if len(client.events) >= MAX_PENDING_EVENTS:
            logger.error("Disconnecting a slow client: event queue is full")
            self._spawn_drop(client, "event queue overflow")
            return
        client.events.append((topic, data))
        client.wake.set()


def _encode_jpeg(rgb: NDArray[np.uint8]) -> bytes:
    """Encode one RGB camera frame as a baseline JPEG (F14); runs in an executor."""
    import cv2

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"rgb must be an (H, W, 3) uint8 image, got {image.shape}/{image.dtype}")
    ok, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, VIDEO_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("cv2.imencode could not encode the frame")
    return bytes(buffer.tobytes())


def _command_ref_seq(message: Mapping[str, Any]) -> int | None:
    """The client's optional command counter (IF-304); ``None`` when absent."""
    seq = message.get("seq")
    if seq is None:
        return None
    if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 1:
        return seq
    raise ValueError(f"invalid client seq {seq!r}")


def _peer_address(transport: Any) -> str:
    """Return the peer's ``host:port``, or ``"unknown"`` without a transport."""
    if transport is None:
        return "unknown"
    with contextlib.suppress(Exception):
        peername = transport.get_extra_info("peername")
        if peername:
            return f"{peername[0]}:{peername[1]}"
    return "unknown"


def _abort_transport(transport: Any) -> None:
    """Abort a WebSocket whose close handshake did not finish (best effort).

    The transport is the connection's real TCP transport, taken from
    ``request.transport`` when the client connected; aiohttp's
    ``WebSocketResponse`` has no ``_response`` attribute (R-09).
    """
    if transport is None:
        return
    with contextlib.suppress(Exception):
        transport.abort()


def _wall() -> str:
    """ISO-8601 UTC wall-clock timestamp for the envelope."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
