"""The vendored native libraries match their provenance record (FR-005, NFR-004)."""

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from cpsa.adapters.native.classifier import CLASSIFIER_LIB, EXTRACTOR_LIB, LIB_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]


def recorded_hashes() -> dict[str, str]:
    text = (LIB_DIR / "PROVENANCE.md").read_text()
    return dict(re.findall(r"^\| `(lib\w+\.so)` \| `([0-9a-f]{64})` \|$", text, re.MULTILINE))


def test_lib_dir_holds_exactly_the_two_libraries() -> None:
    assert sorted(p.name for p in LIB_DIR.glob("*.so")) == sorted(
        [EXTRACTOR_LIB.name, CLASSIFIER_LIB.name]
    )


@pytest.mark.parametrize("library", [EXTRACTOR_LIB, CLASSIFIER_LIB], ids=lambda p: p.name)
def test_provenance_hash_matches_file(library: Path) -> None:
    assert recorded_hashes()[library.name] == hashlib.sha256(library.read_bytes()).hexdigest()


def test_repository_tracks_no_other_shared_library() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files", "*.so"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.split()
    assert sorted(tracked) == sorted(
        str(p.relative_to(REPO_ROOT)) for p in (EXTRACTOR_LIB, CLASSIFIER_LIB)
    )
