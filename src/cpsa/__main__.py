"""Start the showcase: configuration, signals, the two processes, exit codes.

The parent process is the pipeline (A: camera, DPU, overlay); the child is the
server (B: HTTP, WebSocket, JPEG). The parent owns the shared frame channel and
the message queue, starts B before it opens the camera or the DPU (so the child
is forked before the Vitis AI bindings and the V4L2 device exist), and on
SIGTERM/SIGINT stops both, releases the camera and the DPU and unlinks the
shared memory.

Exit codes: ``0`` orderly stop, ``1`` unexpected failure, ``2`` startup failure
(unreadable configuration, missing model, camera unavailable) after publishing
``status`` with ``state: "error"``.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from cpsa import __version__, protocol
from cpsa.channel import CommandChannel, FrameChannel, MessageChannel
from cpsa.config import ConfigError, ShowcaseConfig, load
from cpsa.pipeline import Pipeline, StartError
from cpsa.server import serve

logger = logging.getLogger("cpsa")

__all__ = ["DEFAULT_CONFIG", "EXIT_ERROR", "EXIT_OK", "EXIT_START", "main", "run"]

DEFAULT_CONFIG = Path("config/showcase.yaml")
"""The one configuration file, relative to the repository root (the unit's cwd)."""

EXIT_OK = 0
"""Orderly stop."""

EXIT_ERROR = 1
"""Unexpected failure after a successful start."""

EXIT_START = 2
"""Startup failure: the process published ``status`` ``error`` and gave up."""

SERVER_START_TIMEOUT_S = 15.0
"""How long the pipeline waits for the server process to listen."""

ERROR_FLUSH_S = 0.5
"""Time the server gets to deliver a startup-error ``status`` before the exit."""

_SERVER_JOIN_S = 5.0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, load the configuration and run the showcase."""
    args = _parse_args(argv)
    try:
        config = load(args.config)
    except ConfigError as exc:
        _configure_logging("INFO")
        logger.error("%s", exc)
        return EXIT_START
    _configure_logging(config.log.level)
    logger.info("cpsa-showcase %s: config %s", __version__, args.config)
    return run(config)


def run(
    config: ShowcaseConfig,
    *,
    stop: threading.Event | None = None,
    channel: FrameChannel | None = None,
    messages: MessageChannel | None = None,
    commands: CommandChannel | None = None,
    detector_factory: Callable[[Any], Any] | None = None,
    movenet_factory: Callable[[Any], Any] | None = None,
    camera_factory: Callable[[Any], Any] | None = None,
    start_server: bool = True,
) -> int:
    """Run pipeline A and server B until ``stop`` is set.

    Args:
        config: The validated configuration.
        stop: External stop request; the signal handlers set the internal one.
        channel: Pre-created frame channel (tests); the caller keeps ownership.
        messages: Pre-created message channel (tests).
        commands: Pre-created control channel, the server-to-pipeline direction.
        detector_factory: Overrides the DPU detector (tests).
        movenet_factory: Overrides the MoveNet runner (tests).
        camera_factory: Overrides the UVC camera (tests).
        start_server: ``False`` runs process A alone, with an in-process server
            omitted (tests of the pipeline wiring).

    Returns:
        One of :data:`EXIT_OK`, :data:`EXIT_ERROR`, :data:`EXIT_START`.
    """

    owns_channel = channel is None
    frames = (
        channel
        if channel is not None
        else FrameChannel.create(config.camera.width, config.camera.height)
    )
    if messages is None:
        messages = MessageChannel(multiprocessing.Queue(maxsize=16))
    if commands is None:
        commands = CommandChannel(multiprocessing.Queue(maxsize=8))
    stop_event = stop if stop is not None else threading.Event()
    server_stop = multiprocessing.Event()
    server_ready = multiprocessing.Event()
    server: multiprocessing.Process | None = None
    if start_server:
        server = multiprocessing.Process(
            target=_server_process,
            args=(
                config.web,
                frames.name,
                frames.width,
                frames.height,
                messages.queue,
                commands.queue,
                server_stop,
                server_ready,
                config.log.level,
                __version__,
                config.movenet.enabled,
            ),
            name="cpsa-web",
        )
        server.start()
        if not server_ready.wait(SERVER_START_TIMEOUT_S):
            logger.error("the server process did not start within %.0f s", SERVER_START_TIMEOUT_S)
            _stop_server(server_stop, server)
            _release(frames, messages, commands, owns_channel)
            return EXIT_ERROR

    pipeline = Pipeline(
        config,
        channel=frames,
        messages=messages,
        detector_factory=detector_factory,
        movenet_factory=movenet_factory,
        camera_factory=camera_factory,
        commands=commands,
    )
    remove_signals = _install_signals(stop_event)
    try:
        try:
            pipeline.start()
        except StartError as exc:
            logger.error("cannot start the showcase: %s", exc)
            camera = "missing" if exc.code == "camera_missing" else "error"
            if exc.code in ("model_missing", "dpu_error", "internal"):
                camera = "missing"
            pipeline.publish("error", [exc.code], camera)
            time.sleep(ERROR_FLUSH_S)
            return EXIT_START
        try:
            pipeline.run(stop_event)
        except Exception:
            logger.exception("the showcase failed")
            return EXIT_ERROR
        return EXIT_OK
    finally:
        remove_signals()
        pipeline.stop()
        _stop_server(server_stop, server)
        _release(frames, messages, commands, owns_channel)


def _server_process(
    web: Any,
    channel_name: str,
    width: int,
    height: int,
    queue: Any,
    command_queue: Any,
    stop: Any,
    ready: Any,
    log_level: str,
    version: str,
    movenet_enabled: bool,
) -> None:
    """Entry point of process B: attach the shared state and serve until stopped."""
    _configure_logging(log_level)
    frames = FrameChannel.attach(channel_name, width, height)
    messages = MessageChannel(queue)
    commands = CommandChannel(command_queue)
    try:
        code = serve(
            web,
            frames=frames,
            messages=messages,
            commands=commands,
            stop=stop,
            ready=ready,
            models=_models(bool(movenet_enabled)),
            server_version=version,
        )
    finally:
        frames.close()
        messages.close()
        commands.close()
    raise SystemExit(code)


def _models(movenet_enabled: bool) -> tuple[str, ...]:
    """The models ``model.set`` may load in this configuration (R2-G)."""
    if movenet_enabled:
        return tuple(protocol.ACTIVE_MODELS)
    return (protocol.ACTIVE_MODELS[0],)


def _install_signals(stop: threading.Event) -> Callable[[], None]:
    """Route SIGTERM/SIGINT of the main thread to ``stop``; return the undo."""

    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    def handler(signum: int, frame: Any) -> None:
        logger.info("received %s; stopping", signal.Signals(signum).name)
        stop.set()

    previous = {sig: signal.signal(sig, handler) for sig in (signal.SIGTERM, signal.SIGINT)}

    def restore() -> None:
        for sig, old in previous.items():
            signal.signal(sig, old)

    return restore


def _stop_server(stop: Any, server: multiprocessing.Process | None) -> None:
    """Tell process B to stop, then make sure it is gone."""
    if server is None:
        return
    stop.set()
    server.join(_SERVER_JOIN_S)
    if server.is_alive():
        logger.warning("the server process did not stop; terminating it")
        server.terminate()
        server.join(2.0)
    logger.info("the server process exited with %s", server.exitcode)


def _release(
    frames: FrameChannel,
    messages: MessageChannel,
    commands: CommandChannel,
    owns_channel: bool,
) -> None:
    """Close the handles; the creator also unlinks the shared memory."""
    commands.close()
    messages.close()
    frames.close()
    if owns_channel:
        frames.unlink()


def _configure_logging(level: str) -> None:
    """Log to stderr; systemd's journal keeps the lines on the board."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("cpsa").setLevel(level.upper())


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cpsa-showcase",
        description="YOLOX-nano 'Cosa vede l'IA' showcase: camera, DPU, dashboard.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"configuration file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--version", action="version", version=f"cpsa-showcase {__version__}")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover - the console script does this
    raise SystemExit(main())
