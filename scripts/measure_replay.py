#!/usr/bin/env python3
"""Sample the running showcase over its WebSocket and print an fps table.

``scripts/measure_replay.sh`` runs this on the board after starting
``cpsa-showcase --replay DIR``: the app is then fed by a loop of photos instead
of the camera, so the fps of the whole chain can be measured with nobody in
front of the booth.

For every case the program sends the view and detector commands, waits for the
app to report them active, then holds the case for ``--seconds`` while it
samples ``status`` (the loop's own fps) and the ``showcase stages over ...``
line the app logs every five seconds. The table reports the median and minimum
status fps and the last stage line of the case, which is where a slow stage
shows up.

The commands go over the protocol v1 WebSocket with ``aiohttp``, the dependency
the app already ships.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

STAGES = re.compile(
    r"showcase stages over (?P<frames>\d+) frames: "
    r"camera=(?P<camera>[\d.]+) preprocess=(?P<preprocess>[\d.]+) dpu=(?P<dpu>[\d.]+) "
    r"decode=(?P<decode>[\d.]+) motion=(?P<motion>[\d.]+) track=(?P<track>[\d.]+) "
    r"draw=(?P<draw>[\d.]+) total=(?P<total>[\d.]+) skipped=(?P<skipped>[\d.]+)% "
    r"full=(?P<full>[\d.]+)% zoom=(?P<zoom>[\d.]+)% total_fps=(?P<total_fps>[\d.]+)"
)
"""The pipeline's five-second stage line (``pipeline._record_timings``)."""

STAGE_FIELDS = (
    "camera",
    "preprocess",
    "dpu",
    "decode",
    "track",
    "draw",
    "total",
)
"""The stage-line columns the table prints, in order."""


@dataclass(frozen=True)
class Case:
    """One measurement: a view, and the detector of an object view."""

    name: str
    model: str
    detector: str | None = None


CASES = (
    Case("objects / YOLOX", "yolox", "yolox"),
    Case("objects / YOLOv5", "yolox", "yolov5"),
    Case("skeleton (MoveNet)", "movenet"),
)
"""The views of the showcase: both object detectors, then the skeleton.

The skeleton view runs MoveNet, so it does not depend on the object detector and
is measured once.
"""


@dataclass
class Samples:
    """What one case observed: the status fps and the last stage line."""

    fps: list[float] = field(default_factory=list)
    stage: dict[str, str] | None = None
    active_model: str = ""
    detector: str = ""
    switching: bool = False


class App:
    """The app's WebSocket: sends commands and keeps the newest status."""

    def __init__(self, session: aiohttp.ClientSession, url: str) -> None:
        self._session = session
        self._url = url
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._seq = 0
        self._task: asyncio.Task[None] | None = None
        self._samples = Samples()

    async def connect(self, timeout: float = 60.0) -> None:
        """Retry until the app's server accepts the WebSocket."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._ws = await self._session.ws_connect(self._url, heartbeat=30.0)
                break
            except (aiohttp.ClientError, OSError):
                if time.monotonic() > deadline:
                    raise
                await asyncio.sleep(0.5)
        self._task = asyncio.create_task(self._read())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._ws is not None:
            await self._ws.close()

    @property
    def samples(self) -> Samples:
        return self._samples

    def reset(self) -> None:
        self._samples = Samples(fps=[], stage=None)

    async def send(self, command: str, data: dict[str, object]) -> None:
        """Send one protocol v1 client command."""
        if self._ws is None:
            raise RuntimeError("the WebSocket is not connected")
        self._seq += 1
        await self._ws.send_str(
            json.dumps({"v": 1, "type": command, "seq": self._seq, "data": data})
        )

    async def select(self, case: Case, timeout: float = 60.0) -> None:
        """Switch the app to the case's view and detector, and wait for it."""
        await self.send("model.set", {"model": case.model})
        if case.detector is not None:
            await self.send("calibration.set", {"detector": case.detector})
        deadline = time.monotonic() + timeout
        while True:
            samples = self._samples
            loaded = samples.active_model == case.model and (
                case.detector is None or samples.detector == case.detector
            )
            if loaded and not samples.switching:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"the app did not switch to {case.name}: "
                    f"active_model={samples.active_model!r} detector={samples.detector!r} "
                    f"switching={samples.switching}"
                )
            await asyncio.sleep(0.2)

    async def _read(self) -> None:
        assert self._ws is not None
        async for message in self._ws:
            if message.type is not aiohttp.WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            data = payload.get("data") or {}
            if payload.get("type") == "status":
                calibration = data.get("calibration") or {}
                self._samples.active_model = str(data.get("active_model", ""))
                self._samples.detector = str(calibration.get("detector", ""))
                self._samples.switching = bool(data.get("switching"))
                fps = data.get("fps")
                if fps is not None:
                    self._samples.fps.append(float(fps))


class LogTail:
    """Reads the stage lines the app appends to its log from a byte offset."""

    def __init__(self, path: Path) -> None:
        self._path = path
        # Start at the end: only the lines of the cases being measured count, so
        # a short case can never print a stage line of an earlier view.
        try:
            self._offset = path.stat().st_size
        except FileNotFoundError:
            self._offset = 0

    def new_stage(self) -> dict[str, str] | None:
        """Return the last ``showcase stages over`` line written since the last call."""
        try:
            with self._path.open("rb") as handle:
                handle.seek(self._offset)
                text = handle.read().decode("utf-8", "replace")
                self._offset = handle.tell()
        except FileNotFoundError:
            return None
        found: dict[str, str] | None = None
        for line in text.splitlines():
            match = STAGES.search(line)
            if match is not None:
                found = match.groupdict()
        return found


def print_table(rows: list[tuple[Case, Samples]]) -> None:
    """Print the measurement: one row per case, one column per stage."""
    header = [
        "case",
        "fps median",
        "fps min",
        "fps stage",
        *(f"{name} ms" for name in STAGE_FIELDS),
    ]
    print("  ".join(f"{name:>12s}" for name in header))
    for case, samples in rows:
        fps = samples.fps
        stage = samples.stage or {}
        cells = [
            case.name,
            f"{statistics.median(fps):.1f}" if fps else "-",
            f"{min(fps):.1f}" if fps else "-",
            stage.get("total_fps", "-"),
            *(stage.get(name, "-") for name in STAGE_FIELDS),
        ]
        print("  ".join(f"{cell:>12s}" for cell in cells))
    print(
        "\nfps median/min: status.fps of the app (the loop's own rate); "
        "fps stage: total_fps of the last five-second stage line."
    )
    print(
        "stage columns are that line's means; dpu is submit-to-wait of one frame, "
        "total the whole chain of that frame (it includes the DPU wait)."
    )


async def measure(url: str, log: Path, seconds: float) -> list[tuple[Case, Samples]]:
    """Run every case, hold it for ``seconds`` and return what it observed."""
    tail = LogTail(log)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=5.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        app = App(session, url)
        await app.connect()
        try:
            rows: list[tuple[Case, Samples]] = []
            for case in CASES:
                await app.select(case)
                app.reset()
                stage = None
                deadline = time.monotonic() + seconds
                while True:
                    found = tail.new_stage()
                    if found is not None:
                        stage = found
                    if time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(0.2)
                samples = app.samples
                samples.stage = stage
                rows.append((case, samples))
                print(f"measured {case.name}", file=sys.stderr)
            return rows
        finally:
            await app.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="ws://127.0.0.1:8080/ws")
    parser.add_argument("--log", type=Path, required=True, help="the app's stderr log")
    parser.add_argument("--seconds", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seconds <= 0.0:
        print("--seconds must be positive", file=sys.stderr)
        return 2
    rows = asyncio.run(measure(args.url, args.log, args.seconds))
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
