#!/usr/bin/env python3
"""Build calibration frames for the Vitis AI PTQ run.

Frames are preprocessed exactly like :class:`cpsa.adapters.pose.dpu.DpuPoseEstimator`
and the ONNX adapter: RGB, letterbox to the model input, ``float32 / 255``.
Each frame is stored as ``(3, H, W)`` float32 ``.npy`` so the container needs
neither Pillow nor OpenCV.

The camera captures in ``--source`` are few, so the remaining frames are
augmentations of them (flip, brightness, contrast, crop, rotation) and seeded
synthetic patterns.  A ``manifest.json`` records every frame and its source.

Example::

    uv run python tools/vitis/make_calib.py --count 120 --out build/vitis/calib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from cpsa.vision.pose_decode import letterbox

DEFAULT_SOURCES = (Path("/home/fabio/cpsa-session-state/scratchpad/cap"),)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SIZE = (480, 640)


def load_images(sources: list[Path]) -> list[tuple[Path, np.ndarray]]:
    images: list[tuple[Path, np.ndarray]] = []
    for source in sources:
        if not source.is_dir():
            raise SystemExit(f"source directory not found: {source}")
        for path in sorted(source.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(path) as image:
                images.append((path, np.asarray(image.convert("RGB"), dtype=np.uint8)))
    if not images:
        raise SystemExit(f"no images found in {[str(source) for source in sources]}")
    return images


def augment(image: np.ndarray, rng: np.random.Generator, index: int) -> np.ndarray:
    """Deterministic augmentation menu applied to a real capture."""

    height, width = image.shape[:2]
    operation = index % 6
    if operation == 0:
        return image[:, ::-1].copy()
    if operation == 1:
        scale = 0.65 if index % 12 == 1 else 1.45
        return np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if operation == 2:
        mean = image.astype(np.float32).mean()
        return np.clip((image - mean) * 1.4 + mean + 20.0, 0, 255).astype(np.uint8)
    if operation == 3:
        crop_h = max(32, int(height * 0.75))
        crop_w = max(32, int(width * 0.75))
        top = int(rng.integers(0, height - crop_h + 1))
        left = int(rng.integers(0, width - crop_w + 1))
        return image[top : top + crop_h, left : left + crop_w].copy()
    if operation == 4:
        angle = float(rng.uniform(-12.0, 12.0))
        rotated = Image.fromarray(image).rotate(angle, resample=Image.Resampling.BILINEAR)
        return np.asarray(rotated, dtype=np.uint8)
    noise = rng.normal(0.0, 12.0, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def synthetic(rng: np.random.Generator, size: tuple[int, int], index: int) -> np.ndarray:
    """A seeded synthetic frame: gradient, blobs, checkerboard or noise."""

    height, width = size
    kind = index % 4
    if kind == 0:
        rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        columns = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        tint = rng.uniform(0.2, 1.0, size=3).astype(np.float32)
        base = (0.5 * rows + 0.5 * columns)[..., None]
        image = base * tint * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)
    if kind == 1:
        image = np.zeros((height, width, 3), dtype=np.float32)
        for _ in range(6):
            radius = int(rng.integers(40, 180))
            center_y = int(rng.integers(0, height))
            center_x = int(rng.integers(0, width))
            color = rng.uniform(40.0, 255.0, size=3)
            yy, xx = np.ogrid[:height, :width]
            mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
            image[mask] = color
        return np.clip(image, 0, 255).astype(np.uint8)
    if kind == 2:
        cell = int(rng.integers(16, 64))
        yy, xx = np.mgrid[:height, :width]
        checker = ((yy // cell) + (xx // cell)) % 2
        tint = rng.uniform(30.0, 255.0, size=3).astype(np.float32)
        image = checker[..., None] * tint
        return np.clip(image, 0, 255).astype(np.uint8)
    image = rng.normal(128.0, 40.0, size=(height, width, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=None,
        help="directory with source images (repeatable)",
    )
    parser.add_argument("--out", type=Path, default=Path("build/vitis/calib"))
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[1])
    parser.add_argument("--seed", type=int, default=20261110)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.count < 1:
        raise SystemExit("--count must be positive")
    sources = list(args.source) if args.source else list(DEFAULT_SOURCES)
    images = load_images(sources)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*.npy"):
        stale.unlink()

    size = (args.height, args.width)
    entries: list[dict[str, object]] = []
    for index in range(args.count):
        if index < len(images):
            origin, source_image = images[index]
            frame_source = "capture"
            label = str(origin)
        elif index < 2 * len(images):
            origin, source_image = images[index - len(images)]
            frame_source = "augmentation"
            label = f"{origin}#aug{index}"
        else:
            source_image = synthetic(rng, size, index)
            frame_source = "synthetic"
            label = f"synthetic#{index}"
        if frame_source == "capture":
            frame = source_image
        elif frame_source == "augmentation":
            frame = augment(source_image, rng, index)
        else:
            frame = source_image
        letterboxed, scale, pad = letterbox(frame, size)
        tensor = np.ascontiguousarray(letterboxed.transpose(2, 0, 1), dtype=np.float32) / 255.0
        path = args.out / f"frame_{index:04d}.npy"
        np.save(path, tensor)
        entries.append(
            {
                "file": path.name,
                "kind": frame_source,
                "source": label,
                "shape": list(tensor.shape),
                "scale": scale,
                "pad": list(pad),
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
            }
        )

    manifest = {
        "sources": [str(source) for source in sources],
        "source_images": len(images),
        "frames": len(entries),
        "size_hw": list(size),
        "preprocessing": "RGB -> letterbox(height, width) -> float32 / 255 -> (3, H, W)",
        "seed": args.seed,
        "entries": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "frames": len(entries),
                "captures": sum(1 for entry in entries if entry["kind"] == "capture"),
                "augmented": sum(1 for entry in entries if entry["kind"] == "augmentation"),
                "synthetic": sum(1 for entry in entries if entry["kind"] == "synthetic"),
                "mean": float(np.mean([entry["mean"] for entry in entries])),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
