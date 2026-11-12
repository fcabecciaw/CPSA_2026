"""The dependency direction of the flat showcase package, from a static import scan.

Each module may import only the modules listed for it: the pipeline owns the
policy and the hardware, the server knows only the protocol and the shared
state, and the model loader is the single module that touches Vitis AI.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "cpsa"

ANYTHING = None  # sentinel: no restriction on external imports

STDLIB = frozenset(sys.stdlib_module_names) | {"__future__"}

#: module -> (internal modules it may import, external top-level modules it may import)
RULES: dict[str, tuple[frozenset[str], frozenset[str] | None]] = {
    "__init__": (frozenset(), frozenset()),
    "config": (frozenset(), frozenset({"yaml"})),
    "protocol": (frozenset(), frozenset()),
    "channel": (frozenset(), frozenset({"numpy"})),
    "dpu": (frozenset(), frozenset({"numpy", "vart", "xir"})),
    "camera": (frozenset({"config"}), frozenset({"cv2", "linuxpy", "numpy"})),
    "yolox": (frozenset({"dpu"}), frozenset({"cv2", "numpy"})),
    "motion": (frozenset(), frozenset({"cv2", "numpy"})),
    "movenet": (frozenset({"dpu"}), frozenset({"cv2", "numpy"})),
    "overlay": (frozenset({"protocol"}), frozenset({"cv2", "numpy"})),
    "tracker": (frozenset({"config", "yolox"}), frozenset({"numpy"})),
    "keypoints": (frozenset({"config", "movenet"}), frozenset({"numpy"})),
    "pipeline": (
        frozenset(
            {
                "protocol",
                "camera",
                "channel",
                "config",
                "keypoints",
                "motion",
                "movenet",
                "overlay",
                "tracker",
                "yolox",
            }
        ),
        frozenset({"numpy"}),
    ),
    "server": (
        frozenset({"protocol", "channel", "config"}),
        frozenset({"aiohttp", "cv2", "numpy"}),
    ),
    "__main__": (
        frozenset({"__init__", "channel", "config", "pipeline", "server"}),
        frozenset(),
    ),
}

#: Hardware and framework imports that belong to exactly one module.
SOLE_OWNERS = {
    "vart": "dpu",
    "xir": "dpu",
    "linuxpy": "camera",
    "aiohttp": "server",
    "yaml": "config",
}


def _imports(path: Path) -> tuple[set[str], set[str]]:
    """Return the internal and the external top-level names a module imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    internal: set[str] = set()
    external: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError(f"{path.name} uses a relative import (module {node.module})")
            names = [node.module or ""]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "cpsa":
                if len(parts) > 1:
                    internal.add(parts[1])
            elif parts[0]:
                external.add(parts[0])
    return internal, external


MODULES = sorted(path.stem for path in PACKAGE_ROOT.glob("*.py"))


def test_every_module_has_a_rule() -> None:
    assert sorted(RULES) == MODULES


@pytest.mark.parametrize("module", MODULES)
def test_a_module_imports_only_what_its_layer_allows(module: str) -> None:
    allowed_internal, allowed_external = RULES[module]

    internal, external = _imports(PACKAGE_ROOT / f"{module}.py")

    assert internal <= allowed_internal, f"{module} imports {sorted(internal - allowed_internal)}"
    assert module not in internal, f"{module} imports itself"
    if allowed_external is not ANYTHING:
        unexpected = external - allowed_external - STDLIB
        assert unexpected == set(), f"{module} imports {sorted(unexpected)}"


def test_vitis_ai_is_imported_by_the_model_loader_only() -> None:
    for module in MODULES:
        _internal, external = _imports(PACKAGE_ROOT / f"{module}.py")
        for name, owner in SOLE_OWNERS.items():
            if name in external:
                assert module == owner, f"{module} imports {name}, owned by {owner}"
