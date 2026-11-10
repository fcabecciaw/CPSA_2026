"""Sync canonical notebooks into their Kaggle push folders.

Kaggle's CLI pushes a folder containing both ``<name>.ipynb`` and
``kernel-metadata.json``.  The canonical, reviewable notebooks live directly in
``ml/notebooks/``; run this script after editing them::

    python ml/scripts/sync_notebooks.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = ("01_build_halfbust", "03_train_fp32", "05_qat")


def main() -> int:
    for stem in NOTEBOOKS:
        source = NOTEBOOK_DIR / f"{stem}.ipynb"
        if not source.is_file():
            raise FileNotFoundError(source)
        target_dir = NOTEBOOK_DIR / stem
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_dir / source.name)
        print(f"synced {source} -> {target_dir / source.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
