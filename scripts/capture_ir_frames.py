#!/usr/bin/env python3
"""Capture raw IR/depth frames from the F200 on the KV260 for hand-model fine-tuning.

Runs **on the board** with the repo venv (``/home/kria/cpsa/.venv``)::

    .venv/bin/python scripts/capture_ir_frames.py \\
        --out build/ir_capture_20261111 --count 300 --every 2

Each kept frame is written as ``ir_XXXXX.png`` (raw 8-bit IR, 640x480) plus
``depth_XXXXX.npy`` (raw Z16), and ``meta.json`` records the camera node, the
timestamps and the run settings.  ``--every K`` keeps every K-th delivered
frame; the run ends at ``--count`` frames, at ``--max-seconds`` or on Ctrl-C.

Safety
------
On 2026-11-11 the board froze when an IR capture ran next to the demo services.
The script therefore refuses to open the camera while ``cpsa-supervisor`` or
``cpsa-dpu`` is active::

    sudo systemctl stop cpsa-supervisor cpsa-dpu

``--allow-services`` skips that check (only for a deliberate coexistence test).
The camera is always closed, also on Ctrl-C or an error.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from cpsa.adapters.camera.devices import resolve_device
from cpsa.adapters.camera.irdepth import INZI_FOURCC, IrDepthSource
from cpsa.vision.irdepth import IrDepthFrame

__all__ = [
    "SERVICES",
    "CaptureError",
    "FrameWriter",
    "ServiceCheckError",
    "active_services",
    "main",
    "parse_args",
]

IR_WIDTH = 640
"""Width of one INZI IR/depth frame."""

IR_HEIGHT = 480
"""Height of one INZI IR/depth frame."""

FPS = 30.0
"""Requested camera rate."""

SERVICES = ("cpsa-supervisor", "cpsa-dpu")
"""Board units that must be stopped before the camera is opened."""

DEFAULT_COUNT = 300
DEFAULT_MAX_SECONDS = 120.0

#: ``systemctl is-active`` states that mean the unit is (still) running.
_BUSY_STATES = frozenset({"active", "activating", "reloading", "deactivating"})

LOGGER = logging.getLogger("cpsa.capture_ir_frames")


class ServiceCheckError(RuntimeError):
    """The state of the demo services could not be determined."""


class CaptureError(RuntimeError):
    """A frame could not be written or the camera delivered an invalid frame."""


class _Source(Protocol):
    """The part of :class:`IrDepthSource` the capture script uses."""

    def start(
        self,
        on_frame: Callable[[IrDepthFrame], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None: ...

    def stop(self) -> None: ...


SourceFactory = Callable[[str, str], _Source]
ServiceCheck = Callable[[], Sequence[str]]
NodeResolver = Callable[[str, str], str]


def active_services(
    units: Sequence[str] = SERVICES,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, ...]:
    """Return the units of ``units`` that ``systemctl is-active`` reports busy.

    Args:
        units: systemd unit names to query in one call.
        run: Test seam; defaults to :func:`subprocess.run`.

    Raises:
        ServiceCheckError: If ``systemctl`` cannot be run or its output cannot be
            mapped to the units.  The check fails closed: an unknown state must
            not let a capture race the demo services.
    """

    try:
        completed = run(
            ["systemctl", "is-active", *units],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise ServiceCheckError(f"cannot run systemctl: {exc}") from exc
    states = completed.stdout.split()
    if len(states) != len(units):
        raise ServiceCheckError(
            f"systemctl is-active printed {len(states)} states for {len(units)} units: "
            f"{completed.stdout.strip()!r}"
        )
    return tuple(unit for unit, state in zip(units, states, strict=True) if state in _BUSY_STATES)


def _resolve_node(device: str, match: str) -> str:
    """Resolve ``device`` to the INZI node the source will open."""

    return resolve_device(device, match=match, require=INZI_FOURCC, forbid=())


def _make_source(node: str, match: str) -> IrDepthSource:
    return IrDepthSource(node, width=IR_WIDTH, height=IR_HEIGHT, fps=FPS, match=match)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FrameWriter:
    """Write every ``every``-th IR/depth frame and stop the run at ``count``.

    Frames are numbered from zero over the *kept* frames, so the file names are
    contiguous even when ``--every`` skips some delivered frames.
    """

    def __init__(self, out: Path, count: int, every: int, stop: threading.Event) -> None:
        self.out = Path(out)
        self.count = count
        self.every = every
        self.stop = stop
        self.delivered = 0
        self.saved = 0
        self.frames: list[dict[str, Any]] = []
        self.error: str | None = None
        self._digits = max(5, len(str(max(count - 1, 0))))

    def prepare(self) -> None:
        """Create the output directory before the camera is opened."""

        self.out.mkdir(parents=True, exist_ok=True)

    def __call__(self, frame: IrDepthFrame) -> None:
        if self.stop.is_set():
            return
        index = self.delivered
        self.delivered += 1
        if index % self.every:
            return
        try:
            self._write(frame)
        except Exception as exc:  # the callback runs on the capture thread
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("failed to write IR frame %d", index)
        self.saved += 0 if self.error else 1
        if self.saved >= self.count or self.error:
            self.stop.set()

    def _write(self, frame: IrDepthFrame) -> None:
        if frame.ir.shape != (frame.height, frame.width):
            raise CaptureError(
                f"IR frame is {frame.ir.shape}, expected {(frame.height, frame.width)}"
            )
        if frame.depth_raw.shape != frame.ir.shape:
            raise CaptureError(f"depth frame is {frame.depth_raw.shape}, expected {frame.ir.shape}")
        stem = f"{self.saved:0{self._digits}d}"
        ir_path = self.out / f"ir_{stem}.png"
        depth_path = self.out / f"depth_{stem}.npy"
        if not cv2.imwrite(str(ir_path), frame.ir):
            raise CaptureError(f"could not write {ir_path}")
        np.save(depth_path, frame.depth_raw)
        self.frames.append(
            {
                "ir": ir_path.relative_to(self.out).as_posix(),
                "depth": depth_path.relative_to(self.out).as_posix(),
                "seq": frame.seq,
                "t_monotonic": round(float(frame.t), 6),
                "captured_at": _now(),
            }
        )


def metadata_payload(
    args: argparse.Namespace,
    node: str,
    writer: FrameWriter,
    *,
    started_at: str,
    stopped_reason: str,
) -> dict[str, Any]:
    """Assemble ``meta.json`` for one capture run."""

    return {
        "started_at": started_at,
        "finished_at": _now(),
        "camera": {"requested": args.device, "match": args.match, "node": node},
        "size": {"width": IR_WIDTH, "height": IR_HEIGHT},
        "fps": FPS,
        "requested_count": args.count,
        "every": args.every,
        "max_seconds": args.max_seconds,
        "saved": writer.saved,
        "delivered": writer.delivered,
        "stopped_reason": stopped_reason,
        "error": writer.error,
        "frames": writer.frames,
    }


def _refusal(busy: Sequence[str]) -> str:
    names = ", ".join(busy)
    return (
        f"refusing to open the camera: {names} active. The board froze on 2026-11-11 when an IR\n"
        "capture ran alongside the demo services. Stop them first:\n"
        "    sudo systemctl stop cpsa-supervisor cpsa-dpu\n"
        "then rerun this command (--allow-services skips this check)."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT, help=f"frames to save (default {DEFAULT_COUNT})"
    )
    parser.add_argument(
        "--every", type=int, default=1, help="keep every K-th delivered frame (default 1)"
    )
    parser.add_argument(
        "--match",
        default="RealSense",
        help="case-insensitive fragment of the camera card name (default: RealSense)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="V4L2 node, or 'auto' to pick the INZI node by --match (default: auto)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help=f"upper bound on the run (default {DEFAULT_MAX_SECONDS:g} s)",
    )
    parser.add_argument(
        "--allow-services",
        action="store_true",
        help="skip the cpsa-supervisor/cpsa-dpu check (do not use during a demo)",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    service_check: ServiceCheck = active_services,
    resolver: NodeResolver = _resolve_node,
    source_factory: SourceFactory = _make_source,
) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.count <= 0:
        print("--count must be positive", file=sys.stderr)
        return 2
    if args.every <= 0:
        print("--every must be positive", file=sys.stderr)
        return 2
    if args.max_seconds <= 0:
        print("--max-seconds must be positive", file=sys.stderr)
        return 2
    if not args.match.strip():
        print("--match must not be empty", file=sys.stderr)
        return 2

    if not args.allow_services:
        try:
            busy = service_check()
        except ServiceCheckError as exc:
            print(f"{exc}; pass --allow-services to skip the check", file=sys.stderr)
            return 2
        if busy:
            print(_refusal(busy), file=sys.stderr)
            return 2

    try:
        node = resolver(args.device, args.match)
    except Exception as exc:
        print(f"cannot resolve the IR camera: {exc}", file=sys.stderr)
        return 1

    writer = FrameWriter(args.out, args.count, args.every, threading.Event())
    writer.prepare()
    source = source_factory(node, args.match)
    started_at = _now()
    reason = "error"
    interrupted = False

    def on_error(exc: Exception) -> None:
        writer.error = f"{type(exc).__name__}: {exc}"
        writer.stop.set()

    try:
        source.start(writer, on_error)
        finished = writer.stop.wait(timeout=args.max_seconds)
        if writer.error:
            reason = "error"
        elif finished:
            reason = "count"
        else:
            reason = "max_seconds"
    except KeyboardInterrupt:
        interrupted = True
        reason = "interrupted"
        print("interrupted; closing the camera", file=sys.stderr)
    finally:
        source.stop()

    payload = metadata_payload(args, node, writer, started_at=started_at, stopped_reason=reason)
    meta_path = args.out / "meta.json"
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved {writer.saved} frames from {writer.delivered} delivered to {args.out}")
    print(f"wrote {meta_path}")
    if writer.error:
        print(f"capture failed: {writer.error}", file=sys.stderr)
        return 1
    if interrupted:
        return 130
    if writer.saved == 0:
        print("no frames saved; check the camera and --match", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
