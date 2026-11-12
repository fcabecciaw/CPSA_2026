#!/usr/bin/env python3
"""Prepare the host-side data a hand21 quantization run needs.

The Vitis container mounts only the repository root at ``/repo``, so the
Kaggle dataset and the IR capture have to be copied (a subset, never the whole
1.6 GB val split) under the worktree's ignored ``build/``:

* ``val_subset/{images,labels}`` -- a deterministic sample of the built val
  split, used for the INT8/FP32 accuracy comparison;
* ``calib`` -- ``--calib-val-count`` further val images plus the IR capture
  frames passed through :func:`cpsa.adapters.camera.irdepth.ir_to_input`
  (the runtime preprocessing), saved as PNG;
* ``ir50`` -- the same preprocessed IR frames, used as an unlabelled test group
  and for the overlay sheet.

The three sets are disjoint and deterministic for a seed, so a rerun produces
the same manifests and the same reported numbers.

Example::

    uv run python tools/vitis/prepare_hand21_data.py \\
        --dataset /path/to/hand-topdown --ir /path/to/ir_capture/ir.npy \\
        --out build/models/hand_topdown
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from cpsa.adapters.camera.irdepth import ir_to_input

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
DEFAULT_SEED = 0


def list_split_images(dataset: Path, split: str = "val") -> list[Path]:
    """Sorted images of ``<dataset>/images/<split>`` (raises when empty)."""

    directory = Path(dataset) / "images" / split
    if not directory.is_dir():
        raise SystemExit(f"image directory not found: {directory}")
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images found in {directory}")
    return images


def select_paths(
    paths: Sequence[Path],
    subset_count: int,
    calib_count: int,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Path], list[Path]]:
    """Split ``paths`` into ``(val_subset, calib)`` disjoint deterministic samples.

    Both samples come from one seeded permutation of the sorted input, so the
    choice does not depend on the filesystem order and a rerun with the same
    seed selects the same files.  The subset is taken first, so the calibration
    images never double as test images.

    Raises:
        ValueError: If the counts are negative or exceed the available images.
    """

    ordered = sorted(paths)
    if subset_count < 1 or calib_count < 0:
        raise ValueError(
            f"subset_count must be positive and calib_count non-negative, "
            f"got {subset_count} and {calib_count}"
        )
    if subset_count + calib_count > len(ordered):
        raise ValueError(f"asked for {subset_count + calib_count} images out of {len(ordered)}")
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    subset = [ordered[int(index)] for index in permutation[:subset_count]]
    calib = [
        ordered[int(index)] for index in permutation[subset_count : subset_count + calib_count]
    ]
    return subset, calib


def copy_split(
    images: Sequence[Path],
    labels_dir: Path,
    destination: Path,
) -> list[dict[str, str]]:
    """Copy ``images`` and their ``.txt`` labels into ``destination``.

    Raises:
        SystemExit: If an image has no label file.
    """

    image_out = destination / "images"
    label_out = destination / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for image in images:
        label = Path(labels_dir) / f"{image.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"missing label for {image}: {label}")
        shutil.copy2(image, image_out / image.name)
        shutil.copy2(label, label_out / label.name)
        copied.append({"image": str(image), "label": str(label)})
    return copied


def write_ir_frames(frames: np.ndarray, count: int, destination: Path) -> list[str]:
    """Write ``count`` IR frames as the runtime's 3-channel model input PNGs."""

    source = np.asarray(frames)
    if source.ndim == 4 and source.shape[-1] == 1:
        source = source[..., 0]
    if source.ndim != 3:
        raise SystemExit(f"IR frames must be (N, H, W), got {source.shape}")
    if count < 1 or count > source.shape[0]:
        raise SystemExit(f"requested {count} IR frames of {source.shape[0]}")
    destination.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for index in range(count):
        model_input = ir_to_input(np.asarray(source[index], dtype=np.uint8))
        name = f"ir_{index:04d}.png"
        Image.fromarray(model_input, mode="RGB").save(destination / name)
        names.append(name)
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="built dataset root")
    parser.add_argument("--ir", type=Path, required=True, help="IR capture .npy (N, H, W)")
    parser.add_argument("--out", type=Path, required=True, help="destination under build/")
    parser.add_argument("--split", default="val")
    parser.add_argument("--subset-count", type=int, default=1000)
    parser.add_argument("--calib-val-count", type=int, default=70)
    parser.add_argument("--ir-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    images = list_split_images(args.dataset, args.split)
    subset, calib_val = select_paths(
        images, args.subset_count, args.calib_val_count, seed=args.seed
    )

    subset_files = copy_split(
        subset, Path(args.dataset) / "labels" / args.split, out / "val_subset"
    )

    calib_root = out.parent.parent / "vitis" / "hand_topdown"
    calib_dir = calib_root / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    for image in calib_val:
        shutil.copy2(image, calib_dir / image.name)

    ir_frames = np.load(args.ir, mmap_mode="r")
    ir_names = write_ir_frames(np.asarray(ir_frames), args.ir_count, calib_root / "ir50")
    for name in ir_names:
        shutil.copy2(calib_root / "ir50" / name, calib_dir / name)

    manifest = {
        "dataset": str(Path(args.dataset).resolve()),
        "ir": str(Path(args.ir).resolve()),
        "split": args.split,
        "seed": int(args.seed),
        "val_subset": {
            "count": len(subset_files),
            "images": str((out / "val_subset" / "images").resolve()),
            "labels": str((out / "val_subset" / "labels").resolve()),
        },
        "calib": {
            "count": len(calib_val) + len(ir_names),
            "val_images": len(calib_val),
            "ir_images": len(ir_names),
            "dir": str(calib_dir.resolve()),
        },
        "ir50": {"count": len(ir_names), "dir": str((calib_root / "ir50").resolve())},
    }
    (out / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
