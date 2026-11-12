#!/usr/bin/env python3
"""Per-stage profile of the ``showcase`` pipeline (AC-S6, 60 fps budget).

Runs on the KV260 with the services stopped (so the DPU is free) and needs no
camera and no model: by default every stage is measured on a synthetic 640x480
frame, which is what the coordinator re-runs after a deploy::

    sudo systemctl stop cpsa-showcase
    cd ~/cpsa && PYTHONPATH=src python3 scripts/profile_showcase.py

With ``--model models/yolox_nano.xmodel`` the real graph runs on the DPU through
:class:`~cpsa.yolox.YoloxDetector`, and with ``--movenet-model
models/movenet.xmodel`` the real MoveNet graph runs through
:class:`~cpsa.movenet.DpuMovenet`: preprocess, DPU and decode are timed by
``submit``/``wait``, exactly as the showcase loop drives them. Without them the
DPU stages report the wired-in stand-ins (``--dpu-ms`` 4.6 ms for YOLOX,
``--movenet-dpu-ms`` 3.14 ms for MoveNet, the board measurements) and the CPU
stages are measured for real.

R2-G loads one model at a time, so the report has one budget line per model: the
DPU runs frame N while the CPU decodes, draws and encodes frame N-1, and a frame
of the YOLOX view costs ``max(preprocess + dpu, camera + preprocess + decode +
draw)`` while a frame of the MoveNet view costs ``max(movenet pre + dpu, camera
+ movenet pre + movenet decode + skeleton)``; the JPEG encode runs on the web
server's own thread. The 60 fps budget is 16.7 ms, the old 25 fps target 40 ms.
The three MoveNet stages also carry their own targets (1.5 / 3.2 / 1.0 ms) and
``skeleton draw`` its 1.5 ms: a MoveNet frame only fits if they hold.

``movenet decode`` is measured on the int8 views the runner keeps for a DPU
slot (the A53 path, :func:`cpsa.movenet.decode_quantized`); the line also
reports what the old float64 reference decoder (`dequantize_heads` + `decode`)
costs on the same heads, so the speed-up is visible.

``decode`` is measured with the showcase thresholds (``person`` at 0.40, every
other class at 0.25) on int8 heads whose object cells half hold a second class
above threshold, so the line reports the multi-label decode and its class-aware
NMS -- the stage that keeps a bottle held in front of a person detectable.

``motion`` is the still gate in front of the YOLOX DPU
(:class:`~cpsa.motion.StillGate`): one grey conversion, one 80x60 area resize
and one absolute difference per frame. While the scene does not move the loop
pays ``camera + motion + draw`` per frame instead of the whole YOLOX loop, plus
a full detection every ``tracker.still.force_every_s`` seconds.

Exit status is 0 when the measurement ran; ``--target-ms`` only labels the
PASS/FAIL column.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

FRAME_SIZE = (480, 640)
"""Source frame ``(height, width)`` of the booth camera."""

DEFAULT_RUNS = 40
DEFAULT_DPU_MS = 4.6
"""Board measurement of the YOLOX DPU run (2026-11-12), used without ``--model``."""

DEFAULT_MOVENET_DPU_MS = 3.14
"""Board measurement of the MoveNet DPU run (2026-11-12), used without a model."""

MOVENET_PREPROCESS_TARGET_MS = 1.5
MOVENET_DECODE_TARGET_MS = 1.0
"""Board targets of the A53 MoveNet stages (R2-G)."""

MOVENET_CROP_PREPROCESS_TARGET_MS = 1.5
"""Board target of feeding the model a square body crop instead of the frame."""

MOVENET_LOCK_TARGET_MS = 1.0
"""Board target of the person lock: crop, centre prior and the per-frame update."""

MOVENET_STABILIZE_TARGET_MS = 0.5
"""Board target of the keypoint stabilizer (One Euro, deadband, hysteresis)."""

YOLOX_CONF_THR = 0.35
YOLOX_IOU_THR = 0.45
YOLOX_CLASS_CONF = {"person": 0.40}
YOLOX_OBJECT_CONF = 0.25
"""The thresholds of ``config/showcase.yaml``, mirrored so the profile decodes
like the showcase: ``person`` strict, every other class loose enough to keep a
bottle held in front of it."""

SKELETON_DRAW_TARGET_MS = 1.5
"""Board target of the MoveNet overlay: the dimmed frame plus the neon skeleton."""

MOTION_TARGET_MS = 0.3
"""Board target of the still gate's motion score: grey 80x60 diff of the frame."""

JPEG_QUALITY = 70


def synthetic_frame(width: int, height: int) -> NDArray[np.uint8]:
    """A deterministic, camera-like frame: gradients, plates and noise."""

    rows = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
    columns = np.linspace(0.0, 255.0, width, dtype=np.float32)[None, :]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = np.clip(rows, 0, 255).astype(np.uint8)
    frame[..., 1] = np.clip(columns, 0, 255).astype(np.uint8)
    frame[..., 2] = np.clip(0.5 * (rows + columns), 0, 255).astype(np.uint8)
    generator = np.random.default_rng(20261112)
    noise = generator.integers(-12, 13, size=(height, width, 1), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # A few "objects": flat rectangles the detector would box.
    for top, left, box_height, box_width, value in (
        (80, 120, 180, 140, 200),
        (260, 400, 120, 90, 40),
        (150, 480, 160, 120, 150),
    ):
        frame[top : top + box_height, left : left + box_width] = value
    return frame


def synthetic_yuyv(width: int, height: int) -> bytes:
    """Packed YUYV bytes of a synthetic scene, as the UVC node delivers them."""

    height = height + (height % 2)
    columns = np.arange(width, dtype=np.uint16)[None, :]
    y = np.clip(90 + (columns * 80) // max(width - 1, 1), 16, 235).astype(np.uint8)
    y = np.broadcast_to(y, (height, width))
    u = np.full((height, width // 2), 110, dtype=np.uint8)
    v = np.full((height, width // 2), 150, dtype=np.uint8)
    pairs = np.empty((height, width // 2, 4), dtype=np.uint8)
    pairs[:, :, 0] = y[:, 0::2]
    pairs[:, :, 1] = u
    pairs[:, :, 2] = y[:, 1::2]
    pairs[:, :, 3] = v
    return pairs.tobytes()


def synthetic_boxes() -> list[tuple[float, float, float, float, int, float]]:
    """A handful of boxes as the overlay draws them: ``(x1, y1, x2, y2, class, score)``."""

    return [
        (120.0, 80.0, 260.0, 260.0, 0, 0.91),
        (400.0, 260.0, 490.0, 380.0, 41, 0.72),
        (480.0, 150.0, 600.0, 310.0, 39, 0.64),
    ]


def synthetic_heads() -> list[NDArray[np.int8]]:
    """The three int8 heads of a realistic frame: background plus planted boxes.

    Shape and fix point are the compiled graph's (52x52/26x26/13x13, 85 channels,
    ``fix_point 3``); the background objectness and classes sit at -16 (sigmoid
    ~= 1e-7), so the int8 prefilter drops the whole grid and only the planted
    cells reach the float decode, as on a real scene. Half of the planted cells
    carry a second class above its threshold, the person-and-bottle overlap the
    multi-label decode is for.
    """

    shapes = ((52, 52, 85), (26, 26, 85), (13, 13, 85))
    generator = np.random.default_rng(20261112)
    heads: list[NDArray[np.int8]] = []
    for height, width, channels in shapes:
        head = np.full((height, width, channels), -128, dtype=np.int8)
        head[..., :4] = generator.integers(-40, 40, size=(height, width, 4), dtype=np.int8)
        for index in range(8):
            row = int(generator.integers(0, height))
            column = int(generator.integers(0, width))
            head[row, column, 4] = generator.integers(70, 120)
            head[row, column, 5:] = generator.integers(-100, -20, size=80)
            head[row, column, 5] = generator.integers(80, 127)  # person
            if index % 2 == 0:
                head[row, column, 5 + 39] = generator.integers(40, 100)  # bottle
        heads.append(head)
    return heads


MOVENET_ORDER = ("offsets", "regs", "centers", "heatmaps")
"""Output order of the compiled MoveNet graph (``inspect_xmodel.py``)."""

MOVENET_CHANNELS = {"heatmaps": 17, "centers": 1, "regs": 34, "offsets": 34}
MOVENET_FIX_POINTS = {"heatmaps": 3, "centers": 3, "regs": 2, "offsets": 7}


def synthetic_movenet_heads() -> dict[str, NDArray[np.int8]]:
    """The four int8 heads of a plausible frame, exactly as the DPU writes them.

    Shapes are the compiled graph's (17/1/34/34 channels at 48x48 NHWC); the
    values are the model domain of each fix point, with one person's peak on
    every heatmap and on the center head, which is the cheap path of the
    candidate scan (the background sits below the 0.1 threshold, as on a real
    scene).
    """

    generator = np.random.default_rng(20261112)
    heatmaps = generator.normal(-4.0, 0.5, size=(17, 48, 48))
    centers = generator.normal(-4.0, 0.5, size=(1, 48, 48))
    regs = generator.normal(0.0, 2.0, size=(34, 48, 48))
    offsets = generator.uniform(-0.5, 0.5, size=(34, 48, 48))
    heatmaps[:, 24, 24] = 3.0
    centers[0, 24, 24] = 3.0
    dense = {"heatmaps": heatmaps, "centers": centers, "regs": regs, "offsets": offsets}
    return {
        name: np.clip(
            np.rint(np.moveaxis(head, 0, -1) * 2.0 ** MOVENET_FIX_POINTS[name]), -128, 127
        ).astype(np.int8)[None]
        for name, head in dense.items()
    }


def synthetic_keypoints(width: int, height: int) -> NDArray[np.float32]:
    """A standing person's COCO-17 keypoints in source pixels."""

    points = np.array(
        [
            (0.50, 0.18),
            (0.47, 0.16),
            (0.53, 0.16),
            (0.45, 0.17),
            (0.55, 0.17),
            (0.40, 0.30),
            (0.60, 0.30),
            (0.35, 0.45),
            (0.65, 0.45),
            (0.33, 0.60),
            (0.67, 0.60),
            (0.43, 0.62),
            (0.57, 0.62),
            (0.42, 0.80),
            (0.58, 0.80),
            (0.41, 0.97),
            (0.59, 0.97),
        ],
        dtype=np.float32,
    )
    points[:, 0] *= float(width)
    points[:, 1] *= float(height)
    return np.column_stack([points, np.full(17, 0.8, dtype=np.float32)]).astype(np.float32)


def timed(runs: int, function: Callable[[], object]) -> list[float]:
    """Run ``function`` ``runs`` times and return the milliseconds of each run."""

    for _ in range(2):
        function()
    durations: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        function()
        durations.append((time.perf_counter() - started) * 1000.0)
    return durations


def report(
    name: str,
    durations: Sequence[float],
    target_ms: float,
    *,
    note: str = "",
) -> float:
    """Print one stage line and return its mean."""

    mean = statistics.fmean(durations)
    median = statistics.median(durations)
    p95 = statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations)
    status = "ok" if mean < target_ms else "OVER"
    suffix = f"  {note}" if note else ""
    print(
        f"{name:14s} mean={mean:6.2f} ms  median={median:6.2f} ms  "
        f"p95={p95:6.2f} ms  min={min(durations):6.2f} ms  max={max(durations):6.2f} ms  "
        f"{1000.0 / mean:5.1f} fps  {status}{suffix}"
    )
    return mean


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--model",
        default=None,
        help="compiled .xmodel to run on the DPU; without it --dpu-ms is assumed",
    )
    parser.add_argument("--dpu-ms", type=float, default=DEFAULT_DPU_MS)
    parser.add_argument(
        "--movenet-model",
        default=None,
        help="compiled MoveNet .xmodel to run on the DPU; without it --movenet-dpu-ms is assumed",
    )
    parser.add_argument("--movenet-dpu-ms", type=float, default=DEFAULT_MOVENET_DPU_MS)
    parser.add_argument("--width", type=int, default=FRAME_SIZE[1])
    parser.add_argument("--height", type=int, default=FRAME_SIZE[0])
    parser.add_argument("--target-ms", type=float, default=1000.0 / 60.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs <= 0:
        print("--runs must be positive", file=sys.stderr)
        return 2
    if args.width <= 0 or args.height <= 0 or args.width % 2:
        print("width and height must be positive and width must be even", file=sys.stderr)
        return 2

    import cv2

    from cpsa.camera import yuyv_to_rgb
    from cpsa.config import SmoothingConfig
    from cpsa.dpu import OutputTensor, input_quantization_lut_scaled
    from cpsa.keypoints import KeypointStabilizer
    from cpsa.motion import StillGate
    from cpsa.movenet import (
        MOVENET_INPUT_SCALE,
        MOVENET_INPUT_SIZE,
        Crop,
        DpuMovenet,
        MovenetPose,
        PersonLock,
        decode_quantized,
        dequantize_heads,
        quantized_heads,
    )
    from cpsa.movenet import (
        decode as decode_movenet,
    )
    from cpsa.movenet import (
        preprocess_into as preprocess_movenet_into,
    )
    from cpsa.overlay import draw_overlay, draw_skeleton
    from cpsa.protocol import ShowcaseObject
    from cpsa.yolox import (
        YOLOX_INPUT_SIZE,
        YOLOX_PAD_VALUE,
        YOLOX_STRIDES,
        QuantizedYoloxDecoder,
        YoloxDetector,
        class_thresholds,
        prepare_input_into,
    )

    yolox_thresholds = class_thresholds(
        YOLOX_CONF_THR, class_conf=YOLOX_CLASS_CONF, object_conf=YOLOX_OBJECT_CONF
    )

    frame = synthetic_frame(args.width, args.height)
    raw = synthetic_yuyv(args.width, args.height)
    print(
        f"python={sys.version.split()[0]} numpy={np.__version__} "
        f"frame={args.width}x{args.height} runs={args.runs} "
        f"target={args.target_ms:.1f} ms ({1000.0 / args.target_ms:.0f} fps)"
    )

    camera_cv2 = report(
        "yuyv->rgb cv2",
        timed(args.runs, lambda: yuyv_to_rgb(raw, args.width, args.height)),
        args.target_ms,
        note="camera converter",
    )

    lut = input_quantization_lut_scaled(-2, 1.0)
    view = np.empty((*YOLOX_INPUT_SIZE, 3), dtype=np.int8)
    view.fill(lut[YOLOX_PAD_VALUE])

    def preprocess() -> None:
        prepare_input_into(frame, YOLOX_INPUT_SIZE, lut, view, fill_pad=False)

    preprocess_ms = report("preprocess", timed(args.runs, preprocess), args.target_ms)

    labels = ("persona", "bicicletta", "auto", "bottiglia", "tazza", "libro")
    objects = [
        ShowcaseObject(
            label=labels[class_id % len(labels)],
            class_id=class_id,
            score=score,
            box=(x1, y1, x2, y2),
        )
        for x1, y1, x2, y2, class_id, score in synthetic_boxes()
    ]

    def draw() -> None:
        draw_overlay(frame, objects)

    draw_ms = report("draw", timed(args.runs, draw), args.target_ms)

    # The still gate runs on every YOLOX frame: the 640x480 RGB frame becomes an
    # 80x60 grey image and is differenced against the last inferred one. The
    # frame never changes here, so every measurement is the still path.
    gate = StillGate(1.0, 0.5)

    def motion() -> None:
        gate.update(frame, time.monotonic())

    motion_ms = report(
        "motion",
        timed(args.runs, motion),
        MOTION_TARGET_MS,
        note="still gate: grey 80x60 area diff",
    )

    overlay = draw_overlay(frame, objects)

    def encode() -> None:
        cv2.imencode(
            ".jpg",
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )

    jpeg_ms = report(
        "jpeg q70",
        timed(args.runs, encode),
        args.target_ms,
        note="web server encoder thread",
    )

    decode_ms = 0.0
    dpu_ms = args.dpu_ms
    if args.model:
        detector = YoloxDetector(
            args.model,
            conf_thr=YOLOX_CONF_THR,
            iou_thr=YOLOX_IOU_THR,
            class_thr=yolox_thresholds,
        )
        tokens: list[int] = []

        def dpu_cycle() -> None:
            tokens.append(detector.submit(frame))
            detector.wait(tokens.pop())

        for _ in range(2):
            dpu_cycle()
        dpu_times: list[float] = []
        decode_times: list[float] = []
        preprocess_times: list[float] = []
        round_trip: list[float] = []
        for _ in range(args.runs):
            started = time.perf_counter()
            dpu_cycle()
            round_trip.append((time.perf_counter() - started) * 1000.0)
            stage = detector.stage_ms
            preprocess_times.append(stage["preprocess"])
            dpu_times.append(stage["dpu"])
            decode_times.append(stage["decode"])
        preprocess_ms = report("preprocess", preprocess_times, args.target_ms)
        dpu_ms = report("dpu", dpu_times, args.target_ms, note="graph")
        decode_ms = report(
            "decode",
            decode_times,
            args.target_ms,
            note="int8 prefilter, multi-label",
        )
        report("submit+wait", round_trip, args.target_ms, note="preprocess+dpu+decode")
        print(f"decode: last frame kept {detector.last_candidate_count} int8 candidates")
    else:
        heads = synthetic_heads()
        decoder = QuantizedYoloxDecoder(
            [tuple(int(value) for value in head.shape) for head in heads],
            YOLOX_STRIDES,
            (3, 3, 3),
        )
        decode_ms = report(
            "decode",
            timed(
                args.runs,
                lambda: decoder.decode(heads, YOLOX_CONF_THR, YOLOX_IOU_THR, yolox_thresholds),
            ),
            args.target_ms,
            note=f"int8 prefilter, {decoder.last_candidate_count} candidates",
        )
        dpu_ms = report(
            "dpu",
            [args.dpu_ms] * args.runs,
            args.target_ms,
            note="assumed, pass --model to measure",
        )

    # -- MoveNet: the model that replaces YOLOX on the DPU (R2-G) -------------
    movenet_lut = input_quantization_lut_scaled(6, MOVENET_INPUT_SCALE)
    movenet_view = np.empty((*MOVENET_INPUT_SIZE, 3), dtype=np.int8)

    def movenet_preprocess() -> None:
        preprocess_movenet_into(frame, movenet_lut, movenet_view)

    movenet_preprocess_ms = report(
        "movenet pre",
        timed(args.runs, movenet_preprocess),
        MOVENET_PREPROCESS_TARGET_MS,
    )

    # The person lock feeds the graph a square crop of the previous pose, so the
    # real loop pays this preprocess instead of the whole-frame one above.
    movenet_probe_crop = Crop(
        x=args.width // 4, y=args.height // 4, side=min(300, args.height // 2)
    )

    def movenet_crop_preprocess() -> None:
        preprocess_movenet_into(frame, movenet_lut, movenet_view, movenet_probe_crop)

    movenet_crop_preprocess_ms = report(
        "movenet crop",
        timed(args.runs, movenet_crop_preprocess),
        MOVENET_CROP_PREPROCESS_TARGET_MS,
        note=f"{movenet_probe_crop.side}px square body crop",
    )

    keypoints = synthetic_keypoints(args.width, args.height)

    def skeleton() -> None:
        draw_skeleton(frame, keypoints)

    skeleton_ms = report(
        "skeleton draw",
        timed(args.runs, skeleton),
        SKELETON_DRAW_TARGET_MS,
        note="dim + neon skeleton",
    )

    # A body that fits the frame's short side, so the lock really crops it.
    lock_points = (keypoints[:, :2] - np.float32([args.width / 2, args.height / 2])) * np.float32(
        0.5
    ) + np.float32([args.width / 2, args.height / 2])
    lock_pose = MovenetPose(
        keypoints=np.column_stack([lock_points, keypoints[:, 2]]).astype(np.float32),
        score=0.8,
    )
    lock = PersonLock()
    lock.observe(lock_pose, 0.0, (args.height, args.width))
    lock_crop = lock.crop

    def movenet_lock() -> None:
        lock.observe(lock_pose, 0.0, (args.height, args.width))
        lock.prior(lock_crop, (args.height, args.width))

    movenet_lock_ms = report(
        "movenet lock",
        timed(args.runs, movenet_lock),
        MOVENET_LOCK_TARGET_MS,
        note=f"crop={lock_crop.side if lock_crop else 'frame'} + prior + observe",
    )

    # The shipped ``config/showcase.yaml`` smoothing block: the same numbers the
    # board runs, so the stage measures the deployed filter and not a variant.
    stabilizer = KeypointStabilizer(
        SmoothingConfig(
            enabled=True,
            min_cutoff=1.0,
            beta=0.02,
            d_cutoff=1.0,
            deadband_px=1.5,
            show_score=0.35,
            hide_score=0.25,
            show_frames=2,
            hide_frames=4,
        )
    )
    stabilize_frames = [0]

    def movenet_stabilize() -> None:
        stabilize_frames[0] += 1
        stabilizer.update(
            lock_pose,
            stabilize_frames[0] / 30.0,
            size=(args.width, args.height),
        )

    movenet_stabilize_ms = report(
        "movenet stab",
        timed(args.runs, movenet_stabilize),
        MOVENET_STABILIZE_TARGET_MS,
        note="One Euro + deadband + hysteresis",
    )

    movenet_dpu_ms = args.movenet_dpu_ms
    movenet_decode_ms = 0.0
    if args.movenet_model:
        pose_detector = DpuMovenet(args.movenet_model)
        pose_tokens: list[int] = []

        def movenet_cycle() -> None:
            pose_tokens.append(pose_detector.submit(frame))
            pose_detector.wait(pose_tokens.pop())

        for _ in range(2):
            movenet_cycle()
        pose_dpu_times: list[float] = []
        pose_decode_times: list[float] = []
        pose_preprocess_times: list[float] = []
        pose_lock_times: list[float] = []
        for _ in range(args.runs):
            movenet_cycle()
            pose_stage = pose_detector.stage_ms
            pose_preprocess_times.append(pose_stage["preprocess"])
            pose_dpu_times.append(pose_stage["dpu"])
            pose_decode_times.append(pose_stage["decode"])
            pose_lock_times.append(pose_stage["lock"])
        movenet_preprocess_ms = report(
            "movenet pre", pose_preprocess_times, MOVENET_PREPROCESS_TARGET_MS
        )
        movenet_dpu_ms = report("movenet dpu", pose_dpu_times, args.target_ms, note="graph")
        movenet_decode_ms = report(
            "movenet decode", pose_decode_times, MOVENET_DECODE_TARGET_MS, note="int8 heads"
        )
        # The runner's own lock stage is the measured one: it sees the crop the
        # tracked pose asks for, not the probe crop above.
        movenet_lock_ms = report(
            "movenet lock", pose_lock_times, MOVENET_LOCK_TARGET_MS, note="runner stage"
        )
    else:
        buffers = synthetic_movenet_heads()
        ordered = [buffers[name] for name in MOVENET_ORDER]
        tensors = [
            OutputTensor(
                name=f"head_{name}",
                shape=buffers[name].shape[1:],
                fix_point=MOVENET_FIX_POINTS[name],
                stride=4,
            )
            for name in MOVENET_ORDER
        ]
        # The runner builds the int8 head views of a DPU slot once, so the frame
        # pays the decode alone; the reference pays its float64 conversion too.
        movenet_heads = quantized_heads(ordered, tensors, MOVENET_ORDER)
        reference_times = timed(
            args.runs,
            lambda: decode_movenet(
                dequantize_heads(ordered, tensors, MOVENET_ORDER),
                size=(args.height, args.width),
            ),
        )
        fast_times = timed(
            args.runs, lambda: decode_quantized(movenet_heads, size=(args.height, args.width))
        )
        reference_ms = statistics.median(reference_times)
        ratio = reference_ms / max(statistics.median(fast_times), 1e-6)
        movenet_decode_ms = report(
            "movenet decode",
            fast_times,
            MOVENET_DECODE_TARGET_MS,
            note=f"int8 heads, {ratio:.1f}x the float64 reference ({reference_ms:.2f} ms)",
        )
        movenet_dpu_ms = report(
            "movenet dpu",
            [args.movenet_dpu_ms] * args.runs,
            args.target_ms,
            note="assumed, pass --movenet-model to measure",
        )

    # R2-G loads one model at a time: the DPU runs frame N while the CPU decodes
    # and draws N-1, so each view has its own loop estimate.
    yolox_dpu_ms = preprocess_ms + dpu_ms
    yolox_cpu_ms = camera_cv2 + preprocess_ms + decode_ms + draw_ms
    yolox_loop_ms = max(yolox_dpu_ms, yolox_cpu_ms)
    movenet_dpu_frame_ms = movenet_preprocess_ms + movenet_dpu_ms
    movenet_cpu_ms = (
        camera_cv2
        + movenet_preprocess_ms
        + movenet_decode_ms
        + movenet_lock_ms
        + movenet_stabilize_ms
        + skeleton_ms
    )
    movenet_loop_ms = max(movenet_dpu_frame_ms, movenet_cpu_ms)
    print(
        f"\nYOLOX view: max(preprocess+dpu={yolox_dpu_ms:.2f}, "
        f"camera+preprocess+decode+draw={yolox_cpu_ms:.2f}) = {yolox_loop_ms:.2f} ms "
        f"-> {1000.0 / yolox_loop_ms:.1f} fps"
    )
    print(
        f"MoveNet view: max(movenet pre+dpu={movenet_dpu_frame_ms:.2f}, "
        f"camera+movenet pre+decode+lock+stabilize+skeleton={movenet_cpu_ms:.2f}) = "
        f"{movenet_loop_ms:.2f} ms -> {1000.0 / movenet_loop_ms:.1f} fps"
    )
    print(
        f"MoveNet person crop: pre={movenet_crop_preprocess_ms:.2f} ms "
        f"(whole frame {movenet_preprocess_ms:.2f}), lock={movenet_lock_ms:.2f}, "
        f"stabilize={movenet_stabilize_ms:.2f}"
    )
    print(
        f"still frame: camera+motion+draw="
        f"{camera_cv2 + motion_ms + draw_ms:.2f} ms per skipped frame "
        f"(full YOLOX loop {yolox_loop_ms:.2f} ms)"
    )
    print(
        f"web encoder thread: jpeg={jpeg_ms:.2f} ms "
        f"-> {1000.0 / jpeg_ms:.1f} fps (off the loop, one frame per client set)"
    )
    for name, loop_ms in (("YOLOX", yolox_loop_ms), ("MoveNet", movenet_loop_ms)):
        budget = "PASS" if loop_ms <= args.target_ms else "OVER BUDGET"
        print(
            f"budget {name:8s}: {args.target_ms:.2f} ms per frame "
            f"({1000.0 / args.target_ms:.0f} fps) {budget}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
