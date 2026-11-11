"""Copy a dataset from a slow mount to local disk before training.

Kaggle mounts attached datasets over the network (``/kaggle/input``): every
small-file read is slow (about 100 files/s observed), which stalls both the
curation scan and the per-epoch image loading. ``stage_dataset`` copies the
whole dataset tree once, in parallel, to a local directory such as ``/tmp``,
and returns the local root. A completed copy is reused (marker file).

Two staging paths keep that copy cheap:

* ``stage_dataset`` lists the source with :func:`os.walk` and never issues one
  ``stat`` per file, which over 216k files on the network mount costs hours.
  ``include`` restricts the copy to the POSIX relative paths of a shipped
  manifest (:func:`load_manifest`).
* ``stage_archive`` extracts a teacher-generated ``.tar`` once, locally.
"""

from __future__ import annotations

import gzip
import os
import shutil
import tarfile
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path

__all__ = ["find_inputs", "load_manifest", "stage_archive", "stage_dataset"]

MARKER = ".staged_complete"

#: Package folder that holds the gzipped staging manifests.
MANIFEST_PACKAGE = "cpsa_ml.data"


def _copy(pair: tuple[Path, Path]) -> None:
    source, target = pair
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _list_files(source: Path) -> list[str]:
    """Return every file under ``source`` as a sorted POSIX relative path.

    ``os.walk`` reads directory entries in bulk. Listing with ``rglob`` and
    ``is_file()`` instead costs one ``stat`` per file, and on Kaggle's network
    mount that is the difference between seconds and hours.

    Args:
        source: Dataset root to list.

    Returns:
        Sorted POSIX paths relative to ``source``, files only.
    """

    files: list[str] = []
    for folder, _, names in os.walk(source):
        relative = os.path.relpath(folder, source)
        files.extend((Path(relative) / name).as_posix() for name in names)
    files.sort()
    return files


def stage_dataset(
    source: Path | str,
    target: Path | str,
    workers: int = 64,
    include: Iterable[str] | None = None,
) -> Path:
    """Copy the dataset tree under ``source`` to ``target`` and return ``target``.

    Args:
        source: Dataset root (the folder that holds ``data.yaml``).
        target: Local destination root, created when missing.
        workers: Parallel copy threads; the work is I/O-bound.
        include: POSIX paths relative to ``source`` to copy, or ``None`` for the
            whole tree. ``data.yaml`` is always copied. The paths are matched
            against the listing, so a manifest never costs a ``stat``.

    Returns:
        The local dataset root.

    Raises:
        FileNotFoundError: If ``source`` has no ``data.yaml``, or an included
            path is not in the listing.
    """

    source = Path(source)
    target = Path(target)
    if not (source / "data.yaml").is_file():
        raise FileNotFoundError(f"no data.yaml under {source}")
    if (target / MARKER).is_file():
        print(f"staging: reusing the local copy at {target}", flush=True)
        return target
    started = time.perf_counter()
    listing = _list_files(source)
    print(
        f"staging: listed {len(listing)} files under {source} "
        f"({time.perf_counter() - started:.0f} s)",
        flush=True,
    )
    if include is not None:
        wanted = ["data.yaml", *dict.fromkeys(include)]
        available = set(listing)
        missing = [path for path in wanted if path not in available]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} included paths are missing under {source} "
                f"(first 3: {', '.join(missing[:3])})"
            )
        keep = set(wanted)
        listing = [path for path in listing if path in keep]
        print(
            f"staging: the manifest keeps {len(listing)} files under {source}",
            flush=True,
        )
    pairs = [(source / path, target / path) for path in listing]
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


def _dataset_root(target: Path) -> Path | None:
    """Return the folder that holds the extracted ``data.yaml``, if any.

    A teacher archive either holds ``data.yaml`` at its top level or wraps the
    dataset in one folder, as ``tar -cf name.tar hagrid62`` does.

    Args:
        target: Extraction folder.

    Returns:
        The folder that holds ``data.yaml``, or ``None`` when there is none.
    """

    if (target / "data.yaml").is_file():
        return target
    children = [child for child in target.iterdir() if child.is_dir()]
    if len(children) == 1 and (children[0] / "data.yaml").is_file():
        return children[0]
    return None


def stage_archive(archive: Path | str, target: Path | str) -> Path:
    """Extract an uncompressed ``.tar`` dataset to ``target`` and return its root.

    Notebook 06 writes one ``hagrid62.tar`` to ``/kaggle/working``. Copying its
    ~30k files one by one over the mount takes minutes, while extracting a
    single large file is fast; the archive stays uncompressed so the tensor
    images of the teacher labels are not re-encoded.

    Args:
        archive: The ``.tar`` file.
        target: Local folder to extract into, created when missing.

    Returns:
        The folder that holds ``data.yaml``: ``target`` or its single top-level
        folder.

    Raises:
        FileNotFoundError: If ``archive`` is missing or holds no ``data.yaml``.
    """

    archive = Path(archive)
    target = Path(target)
    if target.is_dir():
        existing = _dataset_root(target)
        if existing is not None and (existing / MARKER).is_file():
            print(f"staging: reusing the extracted dataset at {existing}", flush=True)
            return existing
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
    # Python 3.12 (and the PEP 706 backports) reject absolute paths and links
    # that escape the target folder through filter="data".
    extract_options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    step = max(len(members) // 10, 1)
    with tarfile.open(archive) as tar:
        for index, member in enumerate(members, start=1):
            tar.extract(member, target, **extract_options)
            if index % step == 0 or index == len(members):
                print(
                    f"staging: {index}/{len(members)} archive members extracted to {target} "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
    root = _dataset_root(target)
    if root is None:
        raise FileNotFoundError(f"the archive {archive} holds no data.yaml")
    (root / MARKER).write_text(f"{len(members)} members\n", encoding="utf-8")
    print(f"staging: teacher dataset ready at {root}", flush=True)
    return root


def load_manifest(name: str = "halfbust62_h256") -> list[str]:
    """Return the staging manifest shipped as ``cpsa_ml/data/<name>.txt.gz``.

    The manifest lists POSIX paths relative to the dataset root. It is read
    from package data, so a Kaggle run filters the dataset without reading a
    single image header over the network mount.

    Args:
        name: Manifest name without the ``.txt.gz`` suffix.

    Returns:
        Sorted POSIX relative paths, ``data.yaml`` included.

    Raises:
        FileNotFoundError: If the package data has no manifest with that name.
    """

    manifest = resources.files(MANIFEST_PACKAGE).joinpath(f"{name}.txt.gz")
    if not manifest.is_file():
        raise FileNotFoundError(f"no staging manifest named {name!r} in {MANIFEST_PACKAGE}")
    with resources.as_file(manifest) as path, gzip.open(path, "rt", encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    return sorted(entries)


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
