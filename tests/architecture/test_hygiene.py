"""Repository hygiene: no generated or legacy artifacts are tracked (NFR-004)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_SUFFIXES = {".c", ".h", ".mat", ".rsp", ".bat", ".xmodel", ".xclbin", ".pyc"}
FORBIDDEN_COMPONENTS = {"__pycache__", ".idea"}
#: The medical, pose, IMU and recorder trees the showcase replaced.
FORBIDDEN_TOP_LEVEL = {"ml", "models", "native", "VIDEO_pipeline", "IMU_pipeline", "test_scripts"}
FORBIDDEN_PATHS = {
    "src/cpsa/adapters",
    "src/cpsa/application",
    "src/cpsa/board",
    "src/cpsa/domain",
    "src/cpsa/imu",
    "src/cpsa/recorder",
    "src/cpsa/supervisor",
    "src/cpsa/vision",
    "src/cpsa/ports.py",
    "deploy/systemd/cpsa-supervisor.service",
}


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
        or any(str(path) == forbidden for forbidden in FORBIDDEN_PATHS)
        or str(path).startswith(tuple(f"{forbidden}/" for forbidden in FORBIDDEN_PATHS))
    )


def test_no_forbidden_artifacts_are_tracked() -> None:
    assert [str(path) for path in _tracked_files() if _is_forbidden(path)] == []


def test_the_package_is_the_one_flat_tree() -> None:
    modules = sorted(path.stem for path in (REPO_ROOT / "src" / "cpsa").glob("*.py"))

    assert modules == [
        "__init__",
        "__main__",
        "camera",
        "channel",
        "config",
        "dpu",
        "keypoints",
        "movenet",
        "overlay",
        "pipeline",
        "protocol",
        "server",
        "tracker",
        "yolox",
    ]
    directories = [
        path.name
        for path in (REPO_ROOT / "src" / "cpsa").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    ]
    assert directories == []


def test_the_only_console_script_is_the_showcase() -> None:
    import tomllib

    document = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert document["project"]["scripts"] == {"cpsa-showcase": "cpsa.__main__:main"}
    assert sorted(document["project"]["dependencies"]) == [
        "aiohttp>=3.9",
        "linuxpy>=0.25",
        "numpy",
        "opencv-python-headless",
        "pyyaml",
    ]


def test_the_source_has_no_debug_channel() -> None:
    """``print`` and ``pdb`` would interleave with the journal on the board."""
    offenders = []
    for path in sorted((REPO_ROOT / "src" / "cpsa").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("print(", "breakpoint(", "import pdb")):
                offenders.append(f"{path.name}:{number}: {stripped}")

    assert offenders == []


def test_every_module_documents_itself() -> None:
    missing = []
    for path in sorted((REPO_ROOT / "src" / "cpsa").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith('"""'):
            missing.append(path.name)

    assert missing == []
