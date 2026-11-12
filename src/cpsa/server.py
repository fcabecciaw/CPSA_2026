"""The HTTP and WebSocket face of the showcase (F-K1, F-K2).

:class:`WebServer` is everything process B does: serve ``web/dist``, publish
``/health``, stream the overlaid frames as ``/video.mjpeg`` and talk protocol v1
on ``/ws``. It never touches the camera or the DPU; it reads the newest frame
from the shared :class:`~cpsa.channel.FrameChannel` and drains the
:class:`~cpsa.channel.MessageChannel` the pipeline writes.

The JPEG encoder runs only while at least one video client is connected, so a
browser that is not watching costs the pipeline nothing. It encodes at
``web.video_fps`` in an executor and shares one JPEG with every client.

A client that reconnects gets ``hello`` and then the snapshot of the latest
``status``, ``showcase.state`` and ``pose.state``; from then on it receives each
update as it is drained. ``ping`` answers ``pong``; ``model.set`` and the debug
panel's ``calibration.set`` / ``calibration.reset`` are validated here and
queued on the command channel for process A, which owns the DPU runners and the
camera (R2-G). Anything else is ``error bad_command`` (IF-103, IF-304).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from aiohttp import WSMsgType, web
from numpy.typing import NDArray

from cpsa import protocol
from cpsa.calibration import parse_update
from cpsa.channel import CommandChannel, MessageChannel
from cpsa.config import WebConfig

logger = logging.getLogger("cpsa.server")

__all__ = ["WebServer", "serve"]

VIDEO_BOUNDARY = "cpsa-frame"
"""Multipart boundary of ``/video.mjpeg``."""

MAX_INBOUND_MSG_SIZE = 64 * 1024
"""A client message above this closes the connection (aiohttp code 1009)."""

MAX_MESSAGE_BYTES = 256 * 1024
"""A server message above this is dropped instead of sent."""

VIDEO_DISCONNECT_CHECK_S = 0.5
"""An idle video response checks this often that its client is still there."""

CLOSE_TIMEOUT_S = 1.0
"""How long a WebSocket close handshake may take before the transport is aborted."""

MESSAGE_POLL_S = 0.005
"""How often the event loop drains the message channel."""

_READY_TIMEOUT_S = 10.0


class FrameSource(Protocol):
    """The newest overlaid frame, as process B sees it."""

    def read(self, last_seq: int = 0) -> tuple[int, NDArray[np.uint8]] | None: ...


@dataclass(eq=False)
class _Client:
    """One WebSocket client and the snapshots it has not been sent yet."""

    ws: web.WebSocketResponse
    peer: str
    transport: Any = None
    seq: int = 0
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    sender: asyncio.Task[None] | None = None
    closed: bool = False


class WebServer:
    """Serve the dashboard, the health probe, the MJPEG stream and protocol v1.

    Args:
        config: The F-K2 ``web`` section.
        frames: The newest overlaid frame (shared memory on the board).
        messages: The snapshot channel the pipeline publishes into.
        commands: The control channel ``model.set`` is queued on (R2-G); ``None``
            rejects the command with ``bad_command``.
        models: The models this process accepts in ``model.set``; a disabled
            MoveNet is not in the list and the client gets ``bad_command``.
        server_version: Value of ``hello.server_version``.
        url: Optional host name the dashboard is served from; unused.
    """

    def __init__(
        self,
        config: WebConfig,
        *,
        frames: FrameSource,
        messages: MessageChannel,
        commands: CommandChannel | None = None,
        models: Sequence[str] = protocol.ACTIVE_MODELS,
        server_version: str = "0.1.0",
    ) -> None:
        self._config = config
        self._frames = frames
        self._messages = messages
        self._commands = commands
        self._models = tuple(models)
        self._server_version = server_version
        self._latest: dict[str, dict[str, Any]] = {}
        self._clients: set[_Client] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._stopping = False
        self._port = 0
        self._video_clients = 0
        self._video_encoder: asyncio.Task[None] | None = None
        self._video_frame: tuple[int, bytes] | None = None
        self._video_waiters: set[asyncio.Event] = set()
        self._last_frame_seq = 0

    @property
    def port(self) -> int:
        """The port the server listens on (useful with port ``0`` in tests)."""
        return self._port

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the server in its own thread and return once it listens.

        Raises:
            RuntimeError: If the socket cannot be bound within the timeout.
        """

        self._thread = threading.Thread(target=self._run, name="cpsa-web", daemon=True)
        self._thread.start()
        if not self._ready.wait(_READY_TIMEOUT_S):
            raise RuntimeError("the HTTP server did not start in time")
        if self._start_error is not None:
            raise RuntimeError(f"the HTTP server could not start: {self._start_error}")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_async())
            if self._start_error is None:
                self._ready.set()
                loop.run_until_complete(self._serve_until_stopped())
        except BaseException as exc:  # pragma: no cover - start failures are rare
            self._start_error = exc
            self._ready.set()
            logger.exception("the HTTP server failed")
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _start_async(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/health", self._health)
        app.router.add_get("/video.mjpeg", self._video)
        app.router.add_get("/ws", self._websocket)
        app.router.add_get("/assets/{path:.*}", self._asset)
        # Everything else the dashboard ships (vendored MediaPipe runtime and
        # model, notices) is served from web/dist, after the named routes.
        app.router.add_get("/{path:.+}", self._static)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self._config.host, self._config.port)
        try:
            await site.start()
        except OSError as exc:
            self._start_error = exc
            await runner.cleanup()
            return
        self._runner = runner
        self._stopping = False
        self._port = _bound_port(runner, self._config.port)
        logger.info(
            "serving on http://%s:%d (static %s)",
            self._config.host,
            self._port,
            self._config.static_dir,
        )

    async def _serve_until_stopped(self) -> None:
        poller = asyncio.get_running_loop().create_task(self._poll_messages())
        try:
            while not self._stopping:
                await asyncio.sleep(0.05)
        finally:
            poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poller
            await self._shutdown_async()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop accepting clients, close them and stop the loop."""
        self._stopping = True
        loop = self._loop
        if loop is not None and loop.is_running():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(lambda: None)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("the HTTP server did not stop within %.1f s", timeout)
        self._thread = None
        self._loop = None

    async def _shutdown_async(self) -> None:
        for client in list(self._clients):
            await self._drop(client, "server stopping")
        await self._stop_video_encoder()
        runner = self._runner
        self._runner = None
        if runner is not None:
            await runner.cleanup()
        logger.info("the HTTP server stopped")

    # -- states ------------------------------------------------------------

    async def _poll_messages(self) -> None:
        """Drain the pipeline's snapshots into the clients, oldest first."""
        while True:
            await asyncio.sleep(MESSAGE_POLL_S)
            for topic, data in self._messages.drain():
                self._accept(topic, data)

    def _accept(self, topic: str, data: Mapping[str, Any]) -> None:
        """Keep the latest snapshot of ``topic`` and offer it to every client."""
        payload = dict(data)
        self._latest[topic] = payload
        for client in list(self._clients):
            if client.closed:
                continue
            client.pending[topic] = payload
            client.wake.set()

    def _snapshot(self) -> list[tuple[str, Mapping[str, Any]]]:
        """Latest status, then the latest states, for a client that just connected."""
        return [
            (topic, self._latest[topic])
            for topic in ("status", "showcase.state", "pose.state")
            if topic in self._latest
        ]

    # -- HTTP --------------------------------------------------------------

    async def _index(self, request: web.Request) -> web.StreamResponse:
        response = await self._file(self._config.static_dir / "index.html")
        # The page names the hashed bundles of the current build: a cached copy
        # would keep showing an old dashboard after a deploy.
        response.headers["Cache-Control"] = "no-cache"
        return response

    async def _asset(self, request: web.Request) -> web.StreamResponse:
        return await self._contained(self._config.static_dir / "assets", request.match_info["path"])

    async def _static(self, request: web.Request) -> web.StreamResponse:
        return await self._contained(self._config.static_dir, request.match_info["path"])

    async def _contained(self, base: Path, relative: str) -> web.StreamResponse:
        """Serve ``base/relative`` only when it resolves inside ``base``."""
        root = base.resolve()
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return web.Response(status=404, text="not found")
        return await self._file(candidate)

    async def _file(self, path: Path) -> web.StreamResponse:
        if not path.is_file():
            return web.Response(status=404, text="not found")
        response = web.FileResponse(path)
        # WebAssembly.instantiateStreaming needs the exact wasm MIME type.
        if path.suffix == ".wasm":
            response.content_type = "application/wasm"
        return response

    async def _health(self, request: web.Request) -> web.Response:
        status = self._latest.get("status", {})
        return web.json_response(
            {
                "state": status.get("state", "starting"),
                "fps": status.get("fps"),
                "uptime_s": status.get("uptime_s", 0.0),
            }
        )

    # -- video stream ------------------------------------------------------

    async def _video(self, request: web.Request) -> web.StreamResponse:
        """Stream the overlaid frames as ``multipart/x-mixed-replace`` (F-K2)."""
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
                await _write_video_part(response, seen, frame[1])
        except ConnectionError:
            logger.debug("a video client disconnected", exc_info=True)
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

    def _ensure_video_encoder(self) -> None:
        task = self._video_encoder
        if task is None or task.done():
            self._video_encoder = asyncio.get_running_loop().create_task(self._video_encoder_loop())

    async def _stop_video_encoder(self) -> None:
        task = self._video_encoder
        self._video_encoder = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _video_encoder_loop(self) -> None:
        """Encode the newest frame at ``web.video_fps`` while a client watches.

        The JPEG encoding runs in the default executor, so a slow encode never
        stalls the WebSocket clients or the message drain.
        """
        loop = asyncio.get_running_loop()
        period = 1.0 / self._config.video_fps
        next_at = 0.0
        while self._video_clients > 0:
            delay = next_at - loop.time()
            if delay > 0.0:
                await asyncio.sleep(delay)
            next_at = loop.time() + period
            frame = self._frames.read(self._last_frame_seq)
            if frame is None:
                continue
            seq, rgb = frame
            self._last_frame_seq = seq
            try:
                jpeg = await loop.run_in_executor(
                    None, _encode_jpeg, rgb, self._config.jpeg_quality
                )
            except Exception:
                logger.exception("encoding a video frame failed")
                continue
            self._video_frame = (seq, jpeg)
            for waiter in self._video_waiters:
                waiter.set()

    # -- WebSocket ---------------------------------------------------------

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(max_msg_size=MAX_INBOUND_MSG_SIZE)
        await ws.prepare(request)
        client = _Client(ws=ws, peer=_peer_address(request.transport), transport=request.transport)
        self._clients.add(client)
        client.sender = asyncio.get_running_loop().create_task(self._sender(client))
        try:
            async for message in ws:
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

    async def _sender(self, client: _Client) -> None:
        try:
            if not await self._send(client, "hello", protocol.hello(self._server_version)):
                return
            for topic, data in self._snapshot():
                if not await self._send(client, topic, data):
                    return
            while not client.closed:
                while client.pending and not client.closed:
                    topic, data = client.pending.popitem()
                    if not await self._send(client, topic, data):
                        return
                client.wake.clear()
                await client.wake.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("a WebSocket sender failed")
        finally:
            with contextlib.suppress(asyncio.CancelledError):
                await self._drop(client, "sender finished")

    async def _send(self, client: _Client, topic: str, data: Mapping[str, Any]) -> bool:
        if client.closed:
            return False
        message = protocol.envelope(topic, data, client.seq + 1, time.monotonic(), _wall())
        try:
            payload = json.dumps(message, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            logger.exception("cannot serialise a %s message; dropping it", topic)
            return True
        if len(payload.encode("utf-8")) > MAX_MESSAGE_BYTES:
            logger.error("dropping an oversized %s message", topic)
            return True
        try:
            await client.ws.send_str(payload)
        except Exception:
            logger.debug("a WebSocket send failed; dropping the client", exc_info=True)
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
            logger.info("closing a WebSocket client: %s", reason)
        try:
            # Close before cancelling the sender: cancelling a sender parked on a
            # paused drain would leave aiohttp's drain waiter cancelled.
            await asyncio.wait_for(client.ws.close(), timeout=CLOSE_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning("a WebSocket close handshake timed out; aborting the transport")
            _abort_transport(client.transport)
        except Exception:
            logger.debug("closing a WebSocket client failed", exc_info=True)
        finally:
            sender = client.sender
            if sender is not None and sender is not asyncio.current_task() and not sender.done():
                sender.cancel()

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
        if message.get("v") != protocol.PROTOCOL_VERSION:
            self._bad_command(client, ref_seq, "invalid client envelope")
            return
        command = message.get("type")
        data = message.get("data")
        if not isinstance(command, str) or not isinstance(data, dict):
            self._bad_command(client, ref_seq, "invalid client envelope")
            return
        logger.info(
            "client command type=%s data=%s seq=%s peer=%s", command, data, ref_seq, client.peer
        )
        if command == "ping":
            if "id" not in data:
                self._bad_command(client, ref_seq, "ping requires data.id")
                return
            self._queue(client, "pong", protocol.pong(data["id"]))
            return
        if command == "model.set":
            self._handle_model_set(client, ref_seq, data)
            return
        if command == "calibration.set":
            self._handle_calibration_set(client, ref_seq, data)
            return
        if command == "calibration.reset":
            self._forward(client, ref_seq, "calibration.reset", {})
            return
        self._bad_command(client, ref_seq, f"unknown command {command!r}")

    def _handle_model_set(
        self,
        client: _Client,
        ref_seq: int | None,
        data: Mapping[str, Any],
    ) -> None:
        """Validate ``model.set`` and hand it to process A (R2-G).

        The swap belongs to the pipeline, which owns the DPU runner: the server
        only checks the name and queues the command, so a malformed or unknown
        model is rejected here and never reaches the board.
        """
        model = data.get("model")
        if not isinstance(model, str) or model not in protocol.ACTIVE_MODELS:
            self._bad_command(client, ref_seq, f"unknown model {model!r}")
            return
        if model not in self._models:
            self._bad_command(client, ref_seq, f"model {model!r} is not enabled")
            return
        if self._commands is None:
            self._bad_command(client, ref_seq, "model switching is not available")
            return
        self._commands.put("model.set", {"model": model})
        logger.info("client asked to load %s (seq=%s peer=%s)", model, ref_seq, client.peer)

    def _handle_calibration_set(
        self,
        client: _Client,
        ref_seq: int | None,
        data: Mapping[str, Any],
    ) -> None:
        """Validate ``calibration.set`` and hand the normalized values to process A.

        A value out of range, an unknown key or an empty payload is
        ``bad_command`` here, so the pipeline only ever drains values it can
        apply as they are.
        """
        try:
            values = parse_update(data)
        except ValueError as exc:
            self._bad_command(client, ref_seq, str(exc))
            return
        payload = {
            key: list(value) if isinstance(value, tuple) else value for key, value in values.items()
        }
        self._forward(client, ref_seq, "calibration.set", payload)

    def _forward(
        self, client: _Client, ref_seq: int | None, command: str, data: dict[str, Any]
    ) -> None:
        """Queue a validated command for process A, or refuse it without a channel."""
        if self._commands is None:
            self._bad_command(client, ref_seq, f"{command} is not available")
            return
        self._commands.put(command, data)
        logger.info("client %s %s (seq=%s peer=%s)", command, data, ref_seq, client.peer)

    def _bad_command(self, client: _Client, ref_seq: int | None, message: str) -> None:
        self._queue(client, "error", protocol.error("bad_command", message, ref_seq))

    def _queue(self, client: _Client, topic: str, data: Mapping[str, Any]) -> None:
        if client.closed:
            return
        client.pending[topic] = dict(data)
        client.wake.set()


async def _write_video_part(response: web.StreamResponse, seq: int, jpeg: bytes) -> None:
    """Write one multipart part and let aiohttp drain it."""
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


def _encode_jpeg(rgb: NDArray[np.uint8], quality: int) -> bytes:
    """Encode one RGB frame as a baseline JPEG; runs in an executor."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"rgb must be an (H, W, 3) uint8 image, got {image.shape}/{image.dtype}")
    ok, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
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
    """Abort a WebSocket whose close handshake did not finish (best effort)."""
    if transport is None:
        return
    with contextlib.suppress(Exception):
        transport.abort()


def _bound_port(runner: web.AppRunner, requested: int) -> int:
    """The port a site is actually bound to; ``requested`` when it cannot be read."""
    for site in runner.addresses:
        address = site[1] if isinstance(site, tuple) else None
        if isinstance(address, int) and address > 0:
            return address
    return requested


def _wall() -> str:
    """ISO-8601 UTC wall-clock timestamp for the envelope."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def serve(
    config: WebConfig,
    *,
    frames: FrameSource,
    messages: MessageChannel,
    commands: CommandChannel | None = None,
    stop: threading.Event,
    ready: threading.Event | None = None,
    models: Sequence[str] = protocol.ACTIVE_MODELS,
    server_version: str = "0.1.0",
) -> int:
    """Run process B until ``stop`` is set; the target of the child process.

    Args:
        config: The F-K2 ``web`` section.
        frames: The shared frame channel, attached by this process.
        messages: The snapshot channel the pipeline publishes into.
        commands: The control channel ``model.set`` goes to (process A drains it).
        models: The models ``model.set`` accepts; a disabled MoveNet is left out.
        stop: Set by process A (or a signal) to end the run.
        ready: Set once the server listens, so A can start the pipeline.
        server_version: Value of ``hello.server_version``.

    Returns:
        ``0`` after an orderly stop and ``1`` if the server could not start.
    """

    server = WebServer(
        config,
        frames=frames,
        messages=messages,
        commands=commands,
        models=models,
        server_version=server_version,
    )
    try:
        server.start()
        if ready is not None:
            ready.set()
    except Exception:
        logger.exception("the HTTP server could not start")
        return 1
    try:
        while not stop.wait(0.2):
            pass
    finally:
        server.stop()
    return 0
