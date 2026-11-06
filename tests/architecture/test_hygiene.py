"""Repository hygiene: no legacy, generated, IDE or bytecode artifacts are tracked (NFR-004)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_SUFFIXES = {".c", ".h", ".mat", ".rsp", ".bat", ".xmodel", ".xclbin", ".pyc"}
FORBIDDEN_COMPONENTS = {"__pycache__", ".idea"}
FORBIDDEN_TOP_LEVEL = {"VIDEO_pipeline", "test_scripts", "IMU_pipeline"}


def _tracked_files() -> list[PurePosixPath]:
    if shutil.which("git") is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout; tracked-file hygiene cannot be inspected")
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout
    return [PurePosixPath(name) for name in output.decode().split("\0") if name]


def _is_forbidden(path: PurePosixPath) -> bool:
    return (
        path.suffix in FORBIDDEN_SUFFIXES
        or bool(FORBIDDEN_COMPONENTS.intersection(path.parts))
        or path.parts[0] in FORBIDDEN_TOP_LEVEL
    )


def test_no_forbidden_artifacts_are_tracked() -> None:
    assert [str(p) for p in _tracked_files() if _is_forbidden(p)] == []
