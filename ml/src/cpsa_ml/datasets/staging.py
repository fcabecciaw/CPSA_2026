"""Copy a dataset from a slow mount to local disk before training.

Kaggle mounts attached datasets over the network (``/kaggle/input``): every
small-file read is slow (about 100 files/s observed), which stalls both the
curation scan and the per-epoch image loading. ``stage_dataset`` copies the
whole dataset tree once, in parallel, to a local directory such as ``/tmp``,
and returns the local root. A completed copy is reused (marker file).
"""

from __future__ import annotations

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__all__ = ["find_inputs", "stage_dataset"]

MARKER = ".staged_complete"


def _copy(pair: tuple[Path, Path]) -> None:
    source, target = pair
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def stage_dataset(source: Path | str, target: Path | str, workers: int = 32) -> Path:
    """Copy the dataset tree under ``source`` to ``target`` and return ``target``.

    Args:
        source: Dataset root (the folder that holds ``data.yaml``).
        target: Local destination root, created when missing.
        workers: Parallel copy threads; the work is I/O-bound.

    Returns:
        The local dataset root.

    Raises:
        FileNotFoundError: If ``source`` has no ``data.yaml``.
    """

    source = Path(source)
    target = Path(target)
    if not (source / "data.yaml").is_file():
        raise FileNotFoundError(f"no data.yaml under {source}")
    if (target / MARKER).is_file():
        print(f"staging: reusing the local copy at {target}", flush=True)
        return target
    files = [path for path in source.rglob("*") if path.is_file()]
    pairs = [(path, target / path.relative_to(source)) for path in files]
    started = time.perf_counter()
    step = max(len(pairs) // 10, 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(_copy, pairs, chunksize=64), start=1):
            if index % step == 0 or index == len(pairs):
                print(
                    f"staging: {index}/{len(pairs)} files copied to {target} "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
    (target / MARKER).write_text(f"{len(pairs)} files\n", encoding="utf-8")
    return target


#: Folders never searched by :func:`find_inputs`: they hold the per-image files.
SKIP_DIRS = frozenset({"images", "train2017", "val2017", "test2017"})


def find_inputs(root: Path | str, name: str, max_depth: int = 8) -> list[Path]:
    """Return the sorted paths named ``name`` under ``root``, searching shallowly.

    ``Path.rglob`` walks every file of every attached dataset, which takes
    minutes on Kaggle's network mount. Dataset markers (``data.yaml``,
    weights, ``labels/train``) sit a few folders deep, so this search stops
    at ``max_depth`` and never enters per-image folders (``SKIP_DIRS``) or
    the inside of a ``labels`` folder.
    """

    root = Path(root)
    found: list[Path] = []
    frontier = [(root, 0)]
    while frontier:
        folder, depth = frontier.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            path = Path(entry.path)
            if entry.name == name:
                found.append(path)
            if entry.is_dir() and depth + 1 < max_depth:
                if entry.name in SKIP_DIRS or folder.name == "labels":
                    continue
                frontier.append((path, depth + 1))
    return sorted(found)
