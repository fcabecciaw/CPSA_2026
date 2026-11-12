#!/usr/bin/env python3
"""Offline preview of the depth hand gate (F-H) on the real captures.

Host-only tool: it calibrates the empty table, runs the full-frame hand model
on every frame, and shows what the depth localizer does as a *gate*: a detection
box is kept only if it sits on a foreground blob above the table and is not too
large to be a hand. Run it with the ML environment, which carries Ultralytics::

    PYTHONPATH=src ml/.venv/bin/python scripts/hand_locator_preview.py

Outputs, under ``build/hand_locator_preview/`` unless ``--out`` says otherwise:

* ``stats.json``: per capture the frames with 0/1/2 depth ROIs, ``locate_hands``
  and gate latency, and the detections before and after the gate (and after the
  gate plus the strong confidence threshold), with one entry per frame;
* ``<capture>_frames_XX.jpg``: IR frames with the foreground outline, the depth
  ROI points and the model boxes, kept in green and rejected in red.

The captures contain faces: they are read from their absolute path, everything
is written under ``build/`` and nothing is committed.

The 2026-11-12 captures were recorded with different camera poses. A table
model calibrated on one pose does not transfer to another, so
``--depth-offset-m`` shifts the calibrated background and plane by a constant
for a capture whose camera was moved; it exists for diagnosis only (the third
capture needs about +0.165 m).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from cpsa.adapters.camera.irdepth import ir_to_input
from cpsa.adapters.pose.hand_locator import (
    HandRoi,
    LocatorConfig,
    TableModel,
    calibrate_table,
    foreground_mask,
    gate_detections,
    locate_hands,
)
from cpsa.vision.calibration import Intrinsics

DEPTH_SCALE_M = 3.125e-5
"""Raw Z16 units to metres of the F200 depth stream."""

INTRINSICS = Intrinsics(fx=430.0, fy=430.0, cx=320.0, cy=240.0, width=640, height=480)
"""Nominal IR intrinsics of the 640x480 F200 (config ``hands.intrinsics``)."""

CAPTURE_ROOT = Path("/home/fabio/orca/workspaces/CPSA_2026/snook/build")
"""Directory of the 2026-11-12 captures, read-only."""

EMPTY_CAPTURE = CAPTURE_ROOT / "ir_capture_20261112_empty"
CAPTURES = (
    CAPTURE_ROOT / "ir_capture_20261112_empty",
    CAPTURE_ROOT / "ir_capture_20261112_hand",
    CAPTURE_ROOT / "ir_capture_20261112",
)
DEFAULT_MODEL = CAPTURE_ROOT / "models" / "hand_topdown" / "hand-topdown-model" / "best.pt"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "build" / "hand_locator_preview"

MODEL_IMAGE_SIZE = 320
"""Square input side the model runs at."""

MODEL_CONFIDENCE = 0.25
"""Detector threshold of the first stage."""

STRONG_CONFIDENCE = 0.5
"""Detector threshold of the stage after the gate."""

MAX_TILES_PER_SHEET = 30
"""Contact-sheet tiles per page, five columns by six rows."""

FRAME_TILE = (320, 240)
"""Size of one frame tile in a contact sheet, pixels."""

GREEN = (0, 255, 0)
RED = (0, 0, 255)
TEAL = (200, 200, 0)
MAGENTA = (255, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--empty-capture", type=Path, default=EMPTY_CAPTURE)
    parser.add_argument(
        "--capture",
        type=Path,
        action="append",
        default=None,
        help="capture to preview; repeatable (default: the three 2026-11-12 runs)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=MODEL_IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=MODEL_CONFIDENCE)
    parser.add_argument("--conf-strong", type=float, default=STRONG_CONFIDENCE)
    parser.add_argument("--max-frames", type=int, default=0, help="frames per capture, 0 for all")
    parser.add_argument(
        "--depth-offset-m",
        type=float,
        default=0.0,
        help="constant shift of the calibrated background and plane (moved camera)",
    )
    parser.add_argument("--min-height-m", type=float, default=LocatorConfig.min_height_m)
    parser.add_argument("--min-overlap", type=float, default=LocatorConfig.min_overlap)
    parser.add_argument("--max-hand-size-m", type=float, default=LocatorConfig.max_hand_size_m)
    parser.add_argument("--no-model", action="store_true", help="depth localizer only")
    return parser.parse_args()


def capture_frames(capture: Path, limit: int) -> list[tuple[int, Path, Path]]:
    """Return ``(index, ir_path, depth_path)`` for the frames of one capture."""

    depths = sorted(capture.glob("depth_*.npy"))
    if not depths:
        raise SystemExit(f"no depth_*.npy under {capture}")
    frames = []
    for depth in depths[: limit or None]:
        index = int(depth.stem.split("_")[-1])
        ir = capture / f"ir_{index:05d}.png"
        if not ir.exists():
            raise SystemExit(f"missing {ir}")
        frames.append((index, ir, depth))
    return frames


def shifted_table(table: TableModel, offset_m: float) -> TableModel:
    """Shift the calibrated background and plane along the optical axis."""

    if offset_m == 0.0:
        return table
    n_x, n_y, n_z, d = table.plane
    return TableModel(
        background_m=(table.background_m + np.float32(offset_m)).astype(np.float32),
        plane=(n_x, n_y, n_z, d + float(offset_m)),
        zone=table.zone,
    )


def load_detector(model: Path) -> Any:
    """Load the full-frame hand model, telling the caller which venv to use."""

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - environment dependent
        raise SystemExit(
            "ultralytics is not importable; run this script with the ML environment:\n"
            "  PYTHONPATH=src ml/.venv/bin/python scripts/hand_locator_preview.py"
        ) from error
    if not model.exists():
        raise SystemExit(f"no model at {model}")
    return YOLO(str(model))


def detect(
    detector: Any, ir: NDArray[np.uint8], *, imgsz: int, conf: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the model on one preprocessed IR frame and return xyxy boxes and scores."""

    image = ir_to_input(ir)
    result = detector.predict(image, imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return (
        np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float64),
        np.asarray(boxes.conf.cpu().numpy(), dtype=np.float64),
    )


def draw_frame(
    ir: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    rois: Sequence[HandRoi],
    boxes: NDArray[np.float64],
    scores: NDArray[np.float64],
    keep: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """Draw the foreground outline, the depth ROI points and the gated boxes."""

    canvas = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, TEAL, 1)
    for roi in rois:
        cv2.circle(canvas, (round(roi.cx), round(roi.cy)), 4, MAGENTA, -1)
        cv2.circle(canvas, (round(roi.tip[0]), round(roi.tip[1])), 4, BLUE, -1)
        cv2.circle(canvas, (round(roi.entry[0]), round(roi.entry[1])), 4, YELLOW, -1)
    for box, score, kept in zip(boxes, scores, keep, strict=True):
        x1, y1, x2, y2 = (round(float(value)) for value in box)
        color = GREEN if kept else RED
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2 if kept else 1)
        cv2.putText(
            canvas,
            f"{score:.2f}",
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    return canvas


def write_pages(
    tiles: Sequence[NDArray[np.uint8]],
    out: Path,
    stem: str,
    *,
    columns: int,
    per_sheet: int,
) -> list[Path]:
    """Tile images into pages of ``columns`` x rows and write them as JPEG."""

    written: list[Path] = []
    for page, start in enumerate(range(0, len(tiles), per_sheet)):
        page_tiles = tiles[start : start + per_sheet]
        height, width = page_tiles[0].shape[:2]
        rows = math.ceil(len(page_tiles) / columns)
        canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
        for position, tile in enumerate(page_tiles):
            row, column = divmod(position, columns)
            canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = tile
        path = out / f"{stem}_{page:02d}.jpg"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(path)
    return written


def run_capture(
    capture: Path,
    table: TableModel,
    config: LocatorConfig,
    out: Path,
    *,
    detector: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the localizer and the model over one capture and write its sheets."""

    frames = capture_frames(capture, args.max_frames)
    roi_counts = {0: 0, 1: 0, 2: 0}
    extra_rois = 0
    locate_times: list[float] = []
    gate_times: list[float] = []
    detect_times: list[float] = []
    totals = {"raw": 0, "gated": 0, "gated_conf": 0}
    with_boxes = {"raw": 0, "gated": 0, "gated_conf": 0}
    per_frame: list[dict[str, Any]] = []
    tiles: list[NDArray[np.uint8]] = []
    stride = max(1, math.ceil(len(frames) / MAX_TILES_PER_SHEET))

    for position, (index, ir_path, depth_path) in enumerate(frames):
        depth = np.load(depth_path)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
        if ir is None:
            raise SystemExit(f"cannot read {ir_path}")
        started = time.perf_counter()
        rois = locate_hands(
            depth, table, scale_m=DEPTH_SCALE_M, intrinsics=INTRINSICS, config=config
        )
        locate_times.append((time.perf_counter() - started) * 1e3)
        if len(rois) in roi_counts:
            roi_counts[len(rois)] += 1
        else:
            extra_rois += 1

        if detector is None:
            boxes = np.zeros((0, 4), dtype=np.float64)
            scores = np.zeros(0, dtype=np.float64)
            keep = np.zeros(0, dtype=np.bool_)
            strong = np.zeros(0, dtype=np.bool_)
            detect_times.append(0.0)
            gate_times.append(0.0)
        else:
            started = time.perf_counter()
            boxes, scores = detect(detector, ir, imgsz=args.imgsz, conf=args.conf)
            detect_times.append((time.perf_counter() - started) * 1e3)
            started = time.perf_counter()
            keep = gate_detections(
                boxes,
                scores,
                depth,
                table,
                scale_m=DEPTH_SCALE_M,
                intrinsics=INTRINSICS,
                config=config,
            )
            gate_times.append((time.perf_counter() - started) * 1e3)
            strong = keep & (scores >= args.conf_strong)

        counts = {
            "raw": int(boxes.shape[0]),
            "gated": int(keep.sum()),
            "gated_conf": int(strong.sum()),
        }
        for stage, value in counts.items():
            totals[stage] += value
            with_boxes[stage] += int(value > 0)
        per_frame.append(
            {
                "frame": index,
                "rois": len(rois),
                "depth_rois": [
                    {
                        "cx": round(roi.cx, 1),
                        "cy": round(roi.cy, 1),
                        "side": round(roi.side, 1),
                        "angle": round(roi.angle, 4),
                        "palm_depth_m": round(roi.palm_depth_m, 4),
                        "area_m2": round(roi.area_m2, 5),
                        "entry": [round(value, 1) for value in roi.entry],
                        "tip": [round(value, 1) for value in roi.tip],
                    }
                    for roi in rois
                ],
                "detections": counts,
                "boxes": [
                    {
                        "xyxy": [round(float(value), 1) for value in box],
                        "score": round(float(score), 3),
                        "kept": bool(kept),
                        "kept_strong": bool(strong_kept),
                    }
                    for box, score, kept, strong_kept in zip(
                        boxes, scores, keep, strong, strict=True
                    )
                ],
            }
        )
        if position % stride == 0:
            mask = foreground_mask(
                depth, table, scale_m=DEPTH_SCALE_M, intrinsics=INTRINSICS, config=config
            )
            tile = draw_frame(ir, mask, rois, boxes, scores, keep)
            cv2.putText(
                tile,
                f"{index} raw={counts['raw']} keep={counts['gated']} "
                f"strong={counts['gated_conf']} rois={len(rois)}",
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                GREEN,
                1,
            )
            tiles.append(cv2.resize(tile, FRAME_TILE))

    tag = capture.name
    sheets = write_pages(tiles, out, f"{tag}_frames", columns=5, per_sheet=MAX_TILES_PER_SHEET)
    return {
        "capture": str(capture),
        "frames": len(frames),
        "roi_counts": {
            "0": roi_counts[0],
            "1": roi_counts[1],
            "2": roi_counts[2],
            "3+": extra_rois,
        },
        "timing_ms": {
            "locate_median": round(statistics.median(locate_times), 3),
            "locate_p95": round(_percentile(locate_times, 0.95), 3),
            "gate_median": round(statistics.median(gate_times), 3),
            "gate_p95": round(_percentile(gate_times, 0.95), 3),
            "detect_median": round(statistics.median(detect_times), 3),
        },
        "detections": {
            "totals": totals,
            "frames_with_boxes": with_boxes,
            "per_frame_mean": {
                stage: round(total / len(frames), 3) for stage, total in totals.items()
            },
        },
        "frame_sheets": [str(path) for path in sheets],
        "per_frame": per_frame,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of a non-empty sequence."""

    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def main() -> int:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    config = LocatorConfig(
        min_height_m=args.min_height_m,
        min_overlap=args.min_overlap,
        max_hand_size_m=args.max_hand_size_m,
    )
    empty = capture_frames(args.empty_capture, args.max_frames)
    started = time.perf_counter()
    table = calibrate_table(
        [np.load(depth) for _, _, depth in empty],
        scale_m=DEPTH_SCALE_M,
        intrinsics=INTRINSICS,
    )
    calibration_ms = (time.perf_counter() - started) * 1e3
    table = shifted_table(table, args.depth_offset_m)
    detector = None if args.no_model else load_detector(args.model)

    captures = args.capture or list(CAPTURES)
    report: dict[str, Any] = {
        "empty_capture": str(args.empty_capture),
        "empty_frames": len(empty),
        "calibration_ms": round(calibration_ms, 1),
        "table": {
            "plane": [round(value, 6) for value in table.plane],
            "zone_fraction": round(float(table.zone.mean()), 4),
            "background_valid_fraction": round(float(np.isfinite(table.background_m).mean()), 4),
        },
        "model": {
            "path": str(args.model),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "conf_strong": args.conf_strong,
            "enabled": detector is not None,
        },
        "config": {
            "min_height_m": config.min_height_m,
            "max_height_m": config.max_height_m,
            "min_area_m2": config.min_area_m2,
            "hand_length_m": config.hand_length_m,
            "max_hands": config.max_hands,
            "downsample": config.downsample,
            "median_px": config.median_px,
            "open_px": config.open_px,
            "close_px": config.close_px,
            "min_overlap": config.min_overlap,
            "max_hand_size_m": config.max_hand_size_m,
            "depth_offset_m": args.depth_offset_m,
        },
        "intrinsics": {
            "fx": INTRINSICS.fx,
            "fy": INTRINSICS.fy,
            "cx": INTRINSICS.cx,
            "cy": INTRINSICS.cy,
        },
        "captures": [],
    }
    for capture in captures:
        summary = run_capture(capture, table, config, out, detector=detector, args=args)
        report["captures"].append(summary)
        detections = summary["detections"]["totals"]
        print(
            f"{capture.name}: {summary['frames']} frames, rois {summary['roi_counts']}, "
            f"boxes raw {detections['raw']} / gate {detections['gated']} / "
            f"gate+{args.conf_strong:g} {detections['gated_conf']}, "
            f"locate {summary['timing_ms']['locate_median']} ms, "
            f"gate {summary['timing_ms']['gate_median']} ms"
        )

    stats_path = out / "stats.json"
    stats_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"table plane {report['table']['plane']} zone {report['table']['zone_fraction']}")
    print(f"wrote {stats_path}")
    for summary in report["captures"]:
        for path in summary["frame_sheets"]:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
