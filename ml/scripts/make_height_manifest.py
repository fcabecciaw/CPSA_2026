"""Write the gzipped staging manifest of half-bust crops at least N pixels tall.

Notebook 04 copies only a subset of the attached ``cpsa-halfbust`` dataset to
local disk: the train crops tall enough for hand and face detail, their labels,
and the whole val split. Listing the subset with a manifest means Kaggle reads
no image header over its slow network mount. Regenerate the manifest from a
local copy of the built dataset with::

    python ml/scripts/make_height_manifest.py \
        --root ~/Downloads/_output_/halfbust62 \
        --min-height 256 \
        --out ml/src/cpsa_ml/data/halfbust62_h256.txt.gz

The output is one sorted POSIX path per line, relative to the dataset root,
gzipped without a timestamp so a rerun on the same dataset is byte-identical.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from PIL import Image

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src/cpsa_ml/data/halfbust62_h256.txt.gz"


def image_height(path: Path) -> int:
    """Return the pixel height of ``path``, reading only the image header.

    Args:
        path: JPEG of the built half-bust dataset.

    Returns:
        The height in pixels.
    """

    with Image.open(path) as image:
        return int(image.height)


def build_manifest(root: Path, min_height: int) -> tuple[list[str], dict[str, int]]:
    """List the files of ``root`` that notebook 04 stages.

    The train split keeps the crops at least ``min_height`` pixels tall and
    their label files; the val split stays complete, because it is the metric
    that selects ``best.pt`` and its 4.3k crops cost little. ``data.yaml`` is
    always listed: the copy is not a usable dataset without it.

    Args:
        root: Dataset root (the folder that holds ``data.yaml``).
        min_height: Smallest train-crop height kept, in pixels.

    Returns:
        The sorted POSIX relative paths plus the counts printed to the user.

    Raises:
        FileNotFoundError: If ``root`` is not a built dataset, or a kept crop
            has no label file.
    """

    if not (root / "data.yaml").is_file():
        raise FileNotFoundError(f"no data.yaml under {root}")
    train_images = sorted((root / "images/train").iterdir())
    val_images = sorted((root / "images/val").iterdir())
    val_labels = sorted((root / "labels/val").iterdir())
    kept = [path for path in train_images if image_height(path) >= min_height]
    entries = ["data.yaml"]
    missing: list[str] = []
    for image in kept:
        label = f"labels/train/{image.stem}.txt"
        if not (root / label).is_file():
            missing.append(label)
            continue
        entries.append(f"images/train/{image.name}")
        entries.append(label)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} train crops have no label file under {root} "
            f"(first 3: {', '.join(missing[:3])})"
        )
    entries.extend(f"images/val/{path.name}" for path in val_images)
    entries.extend(f"labels/val/{path.name}" for path in val_labels)
    counts = {
        "train_images": len(train_images),
        "train_kept": len(kept),
        "val_images": len(val_images),
        "val_labels": len(val_labels),
        "manifest": len(entries),
    }
    return sorted(entries), counts


def write_manifest(entries: list[str], out: Path) -> Path:
    """Write ``entries`` to ``out`` as gzip, one POSIX path per line.

    The timestamp is pinned to 0 and no filename is embedded, so the same
    dataset always produces the same bytes.

    Args:
        entries: Relative paths, already sorted.
        out: Destination ``.txt.gz``.

    Returns:
        ``out``.
    """

    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{entry}\n" for entry in entries).encode("utf-8")
    with out.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(body)
    return out


def main(argv: list[str] | None = None) -> int:
    """Write the manifest and print the counts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="built halfbust62 dataset root")
    parser.add_argument(
        "--min-height", type=int, default=256, help="smallest train-crop height kept, in pixels"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="destination .txt.gz")
    args = parser.parse_args(argv)
    entries, counts = build_manifest(args.root, args.min_height)
    written = write_manifest(entries, args.out)
    share = 100 * counts["train_kept"] / max(counts["train_images"], 1)
    print(
        f"{args.root}: {counts['train_images']} train images, "
        f"{counts['train_kept']} at >= {args.min_height} px ({share:.1f}%)"
    )
    print(f"{args.root}: {counts['val_images']} val images, {counts['val_labels']} val labels")
    print(f"manifest: {counts['manifest']} paths -> {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
