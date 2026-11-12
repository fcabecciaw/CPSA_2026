#!/usr/bin/env python3
"""Draw hand21 predictions on the IR frames and build a contact sheet.

The overlay comes from the same ``quant_outputs.npz`` and decode path as
``tools/vitis/eval_quant.py``: the dump's ``ir`` group is decoded with the
runtime INT8 fast path (and optionally the FP32 deploy graph) and every frame
gets its box, 21 keypoints and the ``HAND21`` bones from
:mod:`cpsa.vision.layouts`.  The sheet is written under ``build/`` and stays
untracked, because the IR capture may contain faces.

Example::

    uv run python tools/vitis/overlay_hand21.py \\
        --dump build/vitis/hand_topdown/quant/quant_outputs.npz \\
        --images build/vitis/hand_topdown/ir50 \\
        --weights build/models/hand_topdown/best.pt \\
        --out build/vitis/hand_topdown/ir50_overlay.png \\
        --out-fp32 build/vitis/hand_topdown/ir50_overlay_fp32.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import eval_quant as eq  # noqa: E402

from cpsa.vision.layouts import HAND21  # noqa: E402
from cpsa.vision.types import PersonPose  # noqa: E402

BOX_COLOUR = (0, 220, 0)
BONE_COLOUR = (0, 200, 255)
JOINT_COLOUR = (255, 220, 0)
WRIST_COLOUR = (255, 60, 60)
TEXT_COLOUR = (0, 0, 0)
TILE_BACKGROUND = (255, 255, 255)


def select_group(dump: eq.QuantDump, group: str) -> list[int]:
    """Dump indices whose ``groups`` entry is ``group``."""

    indices = [index for index, name in enumerate(dump.groups) if name == group]
    if not indices:
        raise SystemExit(f"the dump has no '{group}' frames; groups are {sorted(set(dump.groups))}")
    return indices


def draw_persons(
    image: np.ndarray,
    persons: Sequence[PersonPose],
    kpt_conf_thr: float = 0.3,
) -> Image.Image:
    """Return a copy of ``image`` with boxes, HAND21 bones and joints drawn."""

    canvas = Image.fromarray(np.asarray(image, dtype=np.uint8))
    draw = ImageDraw.Draw(canvas)
    for person in persons:
        x1, y1, x2, y2 = (float(value) for value in person.box)
        draw.rectangle((x1, y1, x2, y2), outline=BOX_COLOUR, width=2)
        draw.text((x1 + 2, max(0, y1 - 10)), f"{person.score:.2f}", fill=BOX_COLOUR)
        xy = np.asarray(person.xy, dtype=np.float64)
        conf = np.asarray(person.conf, dtype=np.float64)
        for start, end in HAND21.edges:
            if conf[start] < kpt_conf_thr or conf[end] < kpt_conf_thr:
                continue
            draw.line(
                (float(xy[start, 0]), float(xy[start, 1]), float(xy[end, 0]), float(xy[end, 1])),
                fill=BONE_COLOUR,
                width=2,
            )
        for index, (x, y) in enumerate(xy):
            if conf[index] < kpt_conf_thr:
                continue
            radius = 3 if index else 4
            colour = WRIST_COLOUR if index == 0 else JOINT_COLOUR
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)
    return canvas


def _tile(image: Image.Image, width: int) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def contact_sheet(
    tiles: Sequence[tuple[Image.Image, str]],
    *,
    columns: int = 10,
    tile_width: int = 320,
    caption_height: int = 16,
) -> Image.Image:
    """Grid of captioned tiles, one tile per frame."""

    if not tiles:
        raise SystemExit("no tiles to draw")
    if columns < 1 or tile_width < 1:
        raise SystemExit("--columns and --tile-width must be positive")
    prepared = [_tile(image, tile_width) for image, _ in tiles]
    tile_height = max(image.height for image in prepared) + caption_height
    rows = math.ceil(len(prepared) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        TILE_BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    for index, (image, caption) in enumerate(
        zip(prepared, (text for _, text in tiles), strict=True)
    ):
        column, row = index % columns, index // columns
        left, top = column * tile_width, row * tile_height
        sheet.paste(image, (left, top + caption_height))
        draw.text((left + 3, top + 2), caption, fill=TEXT_COLOUR)
    return sheet


def build_sheet(
    frames: Sequence[Sequence[PersonPose]],
    paths: Sequence[Path],
    *,
    columns: int,
    tile_width: int,
    kpt_conf_thr: float,
) -> Image.Image:
    """Load ``paths``, overlay each frame's predictions and build the grid."""

    tiles: list[tuple[Image.Image, str]] = []
    for index, (persons, path) in enumerate(zip(frames, paths, strict=True)):
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        overlay = draw_persons(image, persons, kpt_conf_thr=kpt_conf_thr)
        tiles.append((overlay, f"{index:02d} {path.stem} hands={len(persons)}"))
    return contact_sheet(tiles, columns=columns, tile_width=tile_width)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True, help="quant_outputs.npz")
    parser.add_argument("--images", type=Path, required=True, help="directory of the group images")
    parser.add_argument("--group", default="ir", help="dump group to draw")
    parser.add_argument("--out", type=Path, required=True, help="INT8 contact sheet PNG")
    parser.add_argument("--weights", type=Path, default=None, help="best.pt for an FP32 sheet")
    parser.add_argument("--out-fp32", type=Path, default=None, help="FP32 contact sheet PNG")
    parser.add_argument("--deploy-module", type=Path, default=eq.DEFAULT_DEPLOY_MODULE)
    parser.add_argument("--conf-thr", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.7)
    parser.add_argument("--kpt-conf-thr", type=float, default=0.3)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dump = eq.load_dump(args.dump)
    indices = select_group(dump, args.group)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        indices = indices[: args.limit]
    subset = dump.subset(indices)
    paths = eq.resolve_test_images(subset, args.images)

    int8_frames = eq.decode_int8(subset, eq.int8_decoder(subset), args.conf_thr, args.iou_thr)
    summary: dict[str, Any] = {
        "dump": str(dump.path),
        "group": args.group,
        "frames": len(indices),
        "int8_frames_with_detection": sum(1 for persons in int8_frames if persons),
        "int8_detections": sum(len(persons) for persons in int8_frames),
    }
    sheet = build_sheet(
        int8_frames,
        paths,
        columns=args.columns,
        tile_width=args.tile_width,
        kpt_conf_thr=args.kpt_conf_thr,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    summary["int8_sheet"] = str(args.out)

    if args.weights is not None:
        fp32_frames, timing = eq.predict_fp32(
            args.weights,
            subset,
            paths,
            deploy_module_path=args.deploy_module,
            conf_thr=args.conf_thr,
            iou_thr=args.iou_thr,
        )
        summary["fp32_frames_with_detection"] = sum(1 for persons in fp32_frames if persons)
        summary["fp32_detections"] = sum(len(persons) for persons in fp32_frames)
        summary["fp32_seconds"] = timing["seconds"]
        if args.out_fp32 is not None:
            fp32_sheet = build_sheet(
                fp32_frames,
                paths,
                columns=args.columns,
                tile_width=args.tile_width,
                kpt_conf_thr=args.kpt_conf_thr,
            )
            args.out_fp32.parent.mkdir(parents=True, exist_ok=True)
            fp32_sheet.save(args.out_fp32)
            summary["fp32_sheet"] = str(args.out_fp32)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
