#!/usr/bin/env python3
"""Hardware check for the F200 camera adapter (AC-01).

Runs paired RGB/depth capture for 60 s and reports paired fps, dropped frames,
the depth XU read-back, and the V4L2 buffer timestamps against
``time.monotonic()``. It is a script, not a test: routine tests never import it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cpsa.adapters.camera.f200 import F200Source, XuSettings, read_xu
from cpsa.application.config import CameraConfig, ConfigError, load
from cpsa.ports import RgbdFrame

MAX_TIMESTAMP_DELAY_S = 0.050
MIN_PAIRED_FPS = 29.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/board.yaml"),
        help="board configuration with the camera section (default: config/board.yaml)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="capture duration in seconds (default: 60)",
    )
    return parser.parse_args(argv)


def make_source(camera: CameraConfig) -> F200Source:
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.duration <= 0:
        print("--duration must be positive", file=sys.stderr)
        return 2
    try:
        config = load(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    camera = config.camera
    source = make_source(camera)
    delivered = 0
    first_delivery: float | None = None
    last_delivery: float | None = None
    last_seq = -1
    sequence_gaps = 0
    rgb_delays: list[float] = []
    depth_delays: list[float] = []

    def on_frame(frame: RgbdFrame) -> None:
        nonlocal delivered, first_delivery, last_delivery, last_seq, sequence_gaps
        now = time.monotonic()
        delivered += 1
        if first_delivery is None:
            first_delivery = now
        last_delivery = now
        if last_seq >= 0 and frame.seq != last_seq + 1:
            sequence_gaps += 1
        last_seq = frame.seq
        rgb_delays.append(frame.ts_rgb - now)
        depth_delays.append(frame.ts_depth - now)

    print(
        f"camera: {camera.rgb_device} (YUYV) + {camera.depth_device} (Z16), "
        f"{camera.width}x{camera.height} @ {camera.fps:g} fps"
    )
    try:
        source.start(on_frame)
    except Exception as exc:
        print(f"start failed: {exc}", file=sys.stderr)
        return 1
    started = time.monotonic()
    interrupted = False
    try:
        print(f"capturing for {args.duration:g} s ...", flush=True)
        while time.monotonic() - started < args.duration:
            remaining = args.duration - (time.monotonic() - started)
            time.sleep(max(0.0, min(0.5, remaining)))
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted", file=sys.stderr)
    finally:
        source.stop()
    elapsed = time.monotonic() - started

    stats = source.stats
    if stats is None or delivered == 0:
        print("no paired frames delivered", file=sys.stderr)
        return 1

    if first_delivery is not None and last_delivery is not None and last_delivery > first_delivery:
        paired_fps = (delivered - 1) / (last_delivery - first_delivery)
    else:
        paired_fps = 0.0
    attempted = 2 * delivered + stats.dropped_rgb + stats.dropped_depth
    dropped = stats.dropped_rgb + stats.dropped_depth
    drop_rate = dropped / attempted if attempted else 0.0

    rgb_abs_max = max(abs(value) for value in rgb_delays) if rgb_delays else 0.0
    depth_abs_max = max(abs(value) for value in depth_delays) if depth_delays else 0.0
    rgb_mean = sum(rgb_delays) / len(rgb_delays) if rgb_delays else 0.0
    depth_mean = sum(depth_delays) / len(depth_delays) if depth_delays else 0.0

    print(f"elapsed:          {elapsed:.2f} s")
    print(f"paired frames:    {delivered} ({paired_fps:.2f} fps)")
    print(
        f"drops:            rgb={stats.dropped_rgb} depth={stats.dropped_depth} "
        f"({drop_rate * 100:.2f} %)"
    )
    print(f"sequence gaps:    {sequence_gaps}")
    print(
        "timestamp delay:  "
        f"rgb mean={rgb_mean * 1000:+.2f} ms max|.|={rgb_abs_max * 1000:.2f} ms; "
        f"depth mean={depth_mean * 1000:+.2f} ms max|.|={depth_abs_max * 1000:.2f} ms"
    )

    exit_code = 0
    if interrupted:
        print("FAIL: capture interrupted before the duration elapsed")
        exit_code = 1
    if paired_fps < MIN_PAIRED_FPS:
        print(f"FAIL: paired fps {paired_fps:.2f} < {MIN_PAIRED_FPS:g}")
        exit_code = 1
    if max(rgb_abs_max, depth_abs_max) > MAX_TIMESTAMP_DELAY_S:
        print(f"FAIL: timestamp delay exceeds +/-{MAX_TIMESTAMP_DELAY_S * 1000:.0f} ms")
        exit_code = 1
    if sequence_gaps:
        print("FAIL: paired sequence numbers are not contiguous")
        exit_code = 1

    try:
        xu = read_xu(camera.depth_device)
    except Exception as exc:
        print(f"XU read-back failed: {exc}", file=sys.stderr)
        return 1
    expected = {
        "laser_power": camera.xu.laser_power,
        "accuracy": camera.xu.accuracy,
        "filter": camera.xu.filter,
        "confidence": camera.xu.confidence,
    }
    print("XU read-back:     " + ", ".join(f"{name}={value}" for name, value in xu.items()))
    mismatched = [name for name, value in expected.items() if xu.get(name) != value]
    if mismatched:
        print("FAIL: XU mismatch for " + ", ".join(mismatched))
        exit_code = 1

    print("PASS" if exit_code == 0 else "FAIL")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
