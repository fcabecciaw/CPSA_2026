"""Locate or download the COCO 2017 / COCO-WholeBody inputs of notebook 01.

The half-bust builder needs two inputs: a COCO 2017 image tree with
``train2017/`` and ``val2017/`` and the two COCO-WholeBody v1.0 annotation
files.  Kaggle attachments cover both, but the annotations are frequently
missing from public mirrors, so :func:`wholebody_annotations` falls back to the
official Google Drive ids published by
https://github.com/jin-s13/COCO-WholeBody and downloads them into a cache
outside ``/kaggle/working`` (so they are not re-published as a dataset).

The download needs the optional ``gdown`` dependency::

    pip install "cpsa-ml[sources]"

Both annotation files are large (about 1.5 GB for ``train``), so the
``info.description`` guard reads only the head of the JSON object instead of
loading the whole file into memory.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from pathlib import Path

__all__ = [
    "COCO_IMAGE_DIRS",
    "WHOLEBODY_DESCRIPTION",
    "WHOLEBODY_FILENAMES",
    "WHOLEBODY_GDRIVE_IDS",
    "find_coco_root",
    "wholebody_annotations",
]

LOGGER = logging.getLogger(__name__)

#: Official Google Drive file ids (https://github.com/jin-s13/COCO-WholeBody).
WHOLEBODY_GDRIVE_IDS: Mapping[str, str] = {
    "train": "1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf",
    "val": "1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb",
}

WHOLEBODY_FILENAMES: Mapping[str, str] = {
    "train": "coco_wholebody_train_v1.0.json",
    "val": "coco_wholebody_val_v1.0.json",
}

WHOLEBODY_DESCRIPTION = "COCO-WholeBody"
COCO_IMAGE_DIRS = ("train2017", "val2017")

#: ``info`` is the first member of the COCO JSON, so one MiB is plenty.
_INFO_PREFIX_BYTES = 1 << 20
_INFO_DESCRIPTION = re.compile(r'"info"\s*:\s*\{[^{}]*?"description"\s*:\s*"([^"]*)"', re.DOTALL)

Downloader = Callable[[str, Path], None]


def _attached_file(input_dir: Path, name: str) -> Path | None:
    """Return the first attached file called ``name`` under ``input_dir``."""

    root = Path(input_dir)
    if not root.is_dir():
        return None
    return next((path for path in sorted(root.rglob(name)) if path.is_file()), None)


def find_coco_root(input_dir: Path) -> Path:
    """Return the attached directory that holds ``train2017/`` and ``val2017/``.

    Args:
        input_dir: Directory to search, normally ``/kaggle/input``.

    Returns:
        The parent of the first ``val2017`` directory whose sibling
        ``train2017`` also exists.  Public COCO datasets nest these one or two
        levels down (for example ``awsaf49/coco-2017-dataset/coco2017``).

    Raises:
        FileNotFoundError: If no directory has both image folders.
    """

    root = Path(input_dir)
    if root.is_dir():
        for candidate in sorted(root.rglob(COCO_IMAGE_DIRS[1])):
            parent = candidate.parent
            if candidate.is_dir() and (parent / COCO_IMAGE_DIRS[0]).is_dir():
                return parent
    raise FileNotFoundError(
        f"no COCO 2017 image tree with {'/ and '.join(COCO_IMAGE_DIRS)} under {root}; "
        "attach the awsaf49/coco-2017-dataset dataset"
    )


def _info_description(path: Path) -> str | None:
    """Read ``info.description`` from a COCO JSON without loading the whole file."""

    with path.open("rb") as handle:
        prefix = handle.read(_INFO_PREFIX_BYTES)
    payload = prefix.decode("utf-8", "replace")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError:
        match = _INFO_DESCRIPTION.search(payload)
        return match.group(1) if match is not None else None
    if not isinstance(document, dict):
        return None
    info = document.get("info")
    if not isinstance(info, dict):
        return None
    description = info.get("description")
    return description if isinstance(description, str) else None


def _check_annotations(path: Path) -> None:
    """Reject a file that is not a COCO-WholeBody annotation or is truncated."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"{path} is empty")
    description = _info_description(path)
    if description != WHOLEBODY_DESCRIPTION:
        raise ValueError(
            f"{path} is not a COCO-WholeBody annotation: "
            f"info.description is {description!r}, expected {WHOLEBODY_DESCRIPTION!r}"
        )


def _download_with_gdown(file_id: str, destination: Path) -> None:
    """Download one Google Drive file with the optional ``gdown`` dependency."""

    try:
        import gdown
    except ImportError as error:  # pragma: no cover - exercised through monkeypatching
        raise RuntimeError(
            "the COCO-WholeBody annotations are not attached and gdown is not installed; "
            'run `pip install "cpsa-ml[sources]"` or attach a dataset with '
            f"{'/ and '.join(WHOLEBODY_FILENAMES.values())}"
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    result = gdown.download(id=file_id, output=str(destination), quiet=True)
    if result is None or not Path(result).is_file():
        raise RuntimeError(f"gdown did not download {file_id} to {destination}")


def wholebody_annotations(
    input_dir: Path,
    cache_dir: Path,
    *,
    download: Downloader | None = None,
) -> dict[str, Path]:
    """Return the train/val COCO-WholeBody annotation files, downloading missing ones.

    Attached files (any ``coco_wholebody_<split>_v1.0.json`` below
    ``input_dir``) take precedence, so an air-gapped session can attach them as
    a dataset.  Otherwise the file is fetched into ``cache_dir`` with ``gdown``
    and reused on later runs.

    Args:
        input_dir: Directory to search for attached annotations, normally
            ``/kaggle/input``.
        cache_dir: Download cache, normally ``/tmp/coco_wholebody`` so the
            files are not published with the output dataset.
        download: Injection point for tests; called as ``download(file_id,
            destination)`` and must write ``destination``.

    Returns:
        ``{"train": path, "val": path}``, each validated as COCO-WholeBody.

    Raises:
        ValueError: If a file is not a COCO-WholeBody annotation.
        RuntimeError: If a download is needed but fails.
        FileNotFoundError: If a downloaded file is missing.
    """

    downloader = download if download is not None else _download_with_gdown
    annotations: dict[str, Path] = {}
    for split, filename in WHOLEBODY_FILENAMES.items():
        attached = _attached_file(input_dir, filename)
        if attached is not None:
            path = attached
        else:
            path = Path(cache_dir) / filename
            if not path.is_file():
                LOGGER.info("downloading %s (%s)", filename, WHOLEBODY_GDRIVE_IDS[split])
                path.parent.mkdir(parents=True, exist_ok=True)
                downloader(WHOLEBODY_GDRIVE_IDS[split], path)
        _check_annotations(path)
        annotations[split] = path
    return annotations
