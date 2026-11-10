"""``cpsa-record record``: record synchronized RGB-D and wrist-IMU scripted trials.

Exit codes: 0 after an orderly stop, 1 on an unexpected error, 2 when the
configuration, protocol, participant id, calibration, camera or BlueCoins are
invalid or unavailable.

Only the ``record`` subcommand exists. Calibration capture is deferred and is not
part of this command; pass ``--calibration`` to copy an existing file.
"""

from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO, cast

from cpsa.adapters.clock import SystemClock
from cpsa.adapters.event_log import configure_logging
from cpsa.adapters.recording.writer import (
    CameraInfo,
    ImuInfo,
    RecordingError,
    SessionCounters,
    SessionWriter,
    WriterSnapshot,
    XuValues,
)
from cpsa.application.config import BlueCoinsConfig, Config, ConfigError
from cpsa.application.config import load as load_config
from cpsa.ports import FrameSource, SensorSource
from cpsa.recorder.protocol import Protocol, ProtocolError, plan_trials, protocol_sha256
from cpsa.recorder.protocol import load as load_protocol
from cpsa.recorder.session import SessionController
from cpsa.recorder.terminal import Beep, RawKeySource, TerminalConsole

logger = logging.getLogger("cpsa.recorder")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STARTUP = 2
PARTICIPANT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def validate_participant(participant_id: str) -> str:
    """Return the participant id if it is pseudonymous and filename-safe."""
    if not PARTICIPANT_PATTERN.fullmatch(participant_id):
        raise ValueError(
            "participant id must match ^[A-Za-z0-9_-]{1,32}$ "
            f"(got {participant_id!r}); use a pseudonym such as P01"
        )
    return participant_id


def session_id_for(created: datetime, participant_id: str) -> str:
    """Return ``<YYYYmmdd-HHMMSS>_<participant_id>`` in local time."""
    return f"{created.strftime('%Y%m%d-%H%M%S')}_{participant_id}"


def camera_info(config: Config) -> CameraInfo:
    camera = config.camera
    return CameraInfo(
        rgb_device=camera.rgb_device,
        depth_device=camera.depth_device,
        width=camera.width,
        height=camera.height,
        fps=camera.fps,
        rgb_exposure=camera.rgb_exposure,
        xu=XuValues(
            laser_power=camera.xu.laser_power,
            accuracy=camera.xu.accuracy,
            filter=camera.xu.filter,
            confidence=camera.xu.confidence,
        ),
    )


def imu_info(config: Config, *, no_imu: bool) -> ImuInfo:
    return ImuInfo(
        enabled=not no_imu,
        left=None if no_imu else config.bluecoins.left,
        right=None if no_imu else config.bluecoins.right,
        max_skew_ms=config.sync.max_skew_ms,
        stale_ms=config.sync.stale_ms,
    )


def create_sensor_source(config: Config, *, no_imu: bool) -> SensorSource | None:
    """Construct the BlueCoin source, or ``None`` without ever importing it."""
    if no_imu:
        return None
    from cpsa.adapters.bluecoin.source import BlueCoinSource

    factory = cast(Callable[[str, str], SensorSource], BlueCoinSource)
    return factory(config.bluecoins.left, config.bluecoins.right)


def create_camera(config: Config) -> FrameSource:
    from cpsa.adapters.camera.f200 import F200Source, XuSettings

    camera = config.camera
    return F200Source(
        camera.rgb_device,
        camera.depth_device,
        width=camera.width,
        height=camera.height,
        fps=camera.fps,
        rgb_exposure=camera.rgb_exposure,
        xu=XuSettings(
            laser_power=camera.xu.laser_power,
            accuracy=camera.xu.accuracy,
            filter=camera.xu.filter,
            confidence=camera.xu.confidence,
        ),
    )


def discover_sensors(
    source: SensorSource, settings: BlueCoinsConfig, stop_event: threading.Event
) -> set[str] | None:
    """Scan until both BlueCoins are found or retries run out (``None`` = stopped)."""
    expected = {settings.left, settings.right}
    missing = expected - source.discover(settings.scan_timeout_s)
    for attempt in range(1, settings.discovery_retries + 1):
        if not missing:
            break
        logger.warning(
            "Waiting for BlueCoin devices %s; retry %d/%d in %s s",
            ", ".join(sorted(missing)),
            attempt,
            settings.discovery_retries,
            settings.retry_interval_s,
        )
        if stop_event.wait(settings.retry_interval_s):
            return None
        missing = expected - source.discover(settings.scan_timeout_s)
    return missing


def pairs_dropped(camera: FrameSource) -> int | None:
    """Read the optional pairing counters exposed by the F200 adapter."""
    stats = getattr(camera, "stats", None)
    dropped_rgb = getattr(stats, "dropped_rgb", None)
    dropped_depth = getattr(stats, "dropped_depth", None)
    if isinstance(dropped_rgb, int) and isinstance(dropped_depth, int):
        return dropped_rgb + dropped_depth
    return None


@dataclass(frozen=True)
class _Report:
    session_id: str
    counters: SessionCounters
    path: Path


class StatsReporter:
    """Print MB/s and drop counts every ``interval_s`` while recording."""

    def __init__(
        self,
        writer: SessionWriter,
        session_id: str,
        interval_s: float = 5.0,
        stream: TextIO = sys.stderr,
    ) -> None:
        self._writer = writer
        self._session_id = session_id
        self._interval_s = interval_s
        self._stream = stream
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="cpsa-stats", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        previous = self._writer.snapshot()
        while not self._stop.wait(self._interval_s):
            current = self._writer.snapshot()
            self._line(previous, current)
            previous = current

    def _line(self, previous: WriterSnapshot, current: WriterSnapshot) -> None:
        delta_bytes = current.bytes_written - previous.bytes_written
        megabytes = delta_bytes / (self._interval_s * 1_000_000)
        frames = current.frames_written - previous.frames_written
        line = (
            f"{self._session_id}: {megabytes:.2f} MB/s  frames {frames}  "
            f"queue drops {current.frames_dropped_queue}  "
            f"IMU samples {current.imu_samples}"
        )
        self._stream.write(line + "\n")
        self._stream.flush()


def format_summary(report: _Report) -> str:
    counters = report.counters
    lines = [
        f"Session {report.session_id} -> {report.path}",
        f"Frames written: {counters.frames_written}  "
        f"queue drops: {counters.frames_dropped_queue}  "
        f"pair drops: {counters.pairs_dropped if counters.pairs_dropped is not None else 'n/a'}",
        f"Trials: {counters.trials.get('ok', 0)} ok, {counters.trials.get('bad', 0)} bad",
    ]
    for side in sorted(counters.imu_samples):
        for kind in sorted(counters.imu_samples[side]):
            count = counters.imu_samples[side][kind]
            rate = counters.imu_rate_hz.get(side, {}).get(kind, 0.0)
            lines.append(f"IMU {side}/{kind}: {count} samples, {rate:.2f} Hz")
    return "\n".join(lines)


def record_session(
    *,
    config: Config,
    protocol: Protocol,
    protocol_path: Path,
    participant_id: str,
    out_root: Path,
    calibration: Path | None,
    no_imu: bool,
    queue_size: int,
    stop_event: threading.Event,
) -> int:
    """Record one session. Hardware is constructed here and only here."""
    try:
        sensors = create_sensor_source(config, no_imu=no_imu)
    except ImportError as exc:
        logger.error("Board dependencies are missing (install the 'board' extra): %s", exc)
        return EXIT_STARTUP
    if sensors is not None:
        missing = discover_sensors(sensors, config.bluecoins, stop_event)
        if missing is None:
            logger.info("Stop requested during BlueCoin discovery")
            return EXIT_OK
        if missing:
            logger.error(
                "BlueCoin devices not found after %d retries: %s",
                config.bluecoins.discovery_retries,
                ", ".join(sorted(missing)),
            )
            return EXIT_STARTUP

    try:
        camera = create_camera(config)
    except ImportError as exc:
        logger.error("Camera dependencies are missing (install the 'camera' extra): %s", exc)
        return EXIT_STARTUP

    clock = SystemClock()
    created = clock.now()
    session_id = session_id_for(created, participant_id)

    try:
        writer = SessionWriter(
            root=out_root,
            session_id=session_id,
            participant_id=participant_id,
            camera=camera_info(config),
            imu=imu_info(config, no_imu=no_imu),
            protocol_sha256=protocol_sha256(protocol_path),
            protocol_seed=protocol.seed,
            clock=clock,
            created=created,
            calibration_path=calibration,
            notes=protocol.notes,
            pre_s=protocol.pre_s,
            post_s=protocol.post_s,
            queue_size=queue_size,
        )
    except (OSError, RecordingError, ValueError) as exc:
        logger.error("Cannot create session %s: %s", session_id, exc)
        return EXIT_STARTUP

    trials = plan_trials(protocol)
    writer.start()
    keys = RawKeySource()
    keys.start()
    console = TerminalConsole()
    beep = Beep()
    reporter = StatsReporter(writer, session_id)
    reporter.start()
    camera_started = False
    sensors_started = False
    try:
        try:
            camera.start(writer.write_frame)
            camera_started = True
            if sensors is not None:
                sensors.start(writer.write_sample)
                sensors_started = True
        except Exception as exc:
            logger.error("Cannot start recording: %s", exc)
            return EXIT_STARTUP

        controller = SessionController(
            protocol=protocol,
            trials=trials,
            recorder=writer,
            clock=clock,
            keys=keys,
            console=console,
            beep=beep.play,
            stop_event=stop_event,
        )
        result = controller.run()
        logger.info(
            "Protocol finished: %d trials completed, aborted=%s",
            result.completed_trials,
            result.aborted,
        )
    finally:
        if camera_started:
            camera.stop()
        if sensors_started and sensors is not None:
            sensors.stop()
        keys.close()
        beep.close()
        reporter.stop()
        counters = writer.finalize(pairs_dropped=pairs_dropped(camera))

    print(format_summary(_Report(session_id=session_id, counters=counters, path=writer.path)))
    return EXIT_OK


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Set ``stop_event`` on SIGINT or SIGTERM so the session is finalized."""

    def request_stop(signum: int, _frame: object) -> None:
        logger.info("Received %s; finalizing the session", signal.Signals(signum).name)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpsa-record",
        description="Record scripted synchronized RGB-D and wrist-IMU sessions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record", help="record one session")
    record.add_argument("--config", required=True, type=Path, help="board YAML configuration")
    record.add_argument("--protocol", required=True, type=Path, help="trial protocol YAML")
    record.add_argument("--participant", required=True, help="pseudonymous participant id (P01)")
    record.add_argument("--out", required=True, type=Path, help="session root directory")
    record.add_argument(
        "--calibration", type=Path, default=None, help="calibration.json to copy into the session"
    )
    record.add_argument(
        "--no-imu", action="store_true", help="record frames only; do not import the BlueCoin stack"
    )
    record.add_argument(
        "--queue", type=int, default=90, help="bounded frame queue size (default 90)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "record":  # pragma: no cover - argparse requires a subcommand
        return EXIT_STARTUP

    try:
        validate_participant(args.participant)
    except ValueError as exc:
        print(f"cpsa-record: {exc}", file=sys.stderr)
        return EXIT_STARTUP
    if args.queue < 1:
        print("cpsa-record: --queue must be >= 1", file=sys.stderr)
        return EXIT_STARTUP
    if args.calibration is not None and not args.calibration.is_file():
        print(f"cpsa-record: calibration file not found: {args.calibration}", file=sys.stderr)
        return EXIT_STARTUP

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"cpsa-record: {exc}", file=sys.stderr)
        return EXIT_STARTUP
    try:
        protocol = load_protocol(args.protocol)
    except ProtocolError as exc:
        print(f"cpsa-record: {exc}", file=sys.stderr)
        return EXIT_STARTUP

    configure_logging(config.log.base_path, config.log.level)
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    try:
        return record_session(
            config=config,
            protocol=protocol,
            protocol_path=args.protocol,
            participant_id=args.participant,
            out_root=args.out,
            calibration=args.calibration,
            no_imu=args.no_imu,
            queue_size=args.queue,
            stop_event=stop_event,
        )
    except Exception:
        logger.exception("Unexpected error")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
