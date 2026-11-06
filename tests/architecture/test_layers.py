"""Enforce the package dependency direction (PLAN.md §3) with a static import scan."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = SRC_ROOT / "cpsa"

ANYTHING = None  # sentinel: no restriction on external imports

# layer -> (internal layers it may import, extra third-party top-level modules it may import)
RULES: dict[str, tuple[frozenset[str], frozenset[str] | None]] = {
    "domain": (frozenset(), frozenset()),
    "imu": (frozenset({"domain"}), frozenset({"numpy"})),
    "ports": (frozenset({"domain"}), frozenset({"numpy"})),
    "application": (frozenset({"domain", "imu", "ports"}), frozenset({"numpy", "yaml"})),
    "adapters": (frozenset({"domain", "ports"}), ANYTHING),
    "board": (frozenset({"domain", "imu", "ports", "application", "adapters"}), ANYTHING),
    "__init__": (frozenset(), frozenset()),
}

HARDWARE_MODULES = frozenset(
    {"ctypes", "blue_st_sdk", "bluepy", "mbientlab", "flux_led", "playsound", "dbus_fast"}
)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _layer_of(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else "__init__"


def imported_modules(source: str, module: str, is_package: bool) -> set[str]:
    """Return absolute names of every module imported by ``source``."""
    found: set[str] = set()
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                name = ".".join([*base, node.module] if node.module else base)
            else:
                name = node.module or ""
            found.add(name)
    return found


def violations(layer: str, imports: set[str]) -> list[str]:
    allowed_internal, allowed_external = RULES[layer]
    problems = []
    for name in sorted(imports):
        top = name.split(".")[0]
        if top == "cpsa":
            target = _layer_of(name)
            if target != layer and target not in allowed_internal:
                problems.append(f"{name}: layer '{layer}' may not import layer '{target}'")
        elif layer not in ("adapters", "board") and top in HARDWARE_MODULES:
            problems.append(f"{name}: hardware/native module outside adapters/board")
        elif (
            allowed_external is not ANYTHING
            and top not in sys.stdlib_module_names
            and top not in allowed_external
        ):
            problems.append(f"{name}: third-party module not allowed in layer '{layer}'")
    return problems


SOURCE_FILES = sorted(PACKAGE_ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.relative_to(SRC_ROOT)))
def test_module_respects_layer_rules(path: Path) -> None:
    module = _module_name(path)
    imports = imported_modules(path.read_text(), module, path.name == "__init__.py")
    assert violations(_layer_of(module), imports) == []


@pytest.mark.parametrize(
    ("layer", "imports"),
    [
        ("domain", {"numpy"}),
        ("imu", {"cpsa.application.slot"}),
        ("application", {"cpsa.adapters.clock"}),
        ("application", {"ctypes"}),
        ("adapters", {"cpsa.application.config"}),
    ],
)
def test_checker_flags_forbidden_imports(layer: str, imports: set[str]) -> None:
    assert violations(layer, imports)


def test_relative_imports_are_resolved() -> None:
    source = "from ..domain import model\nfrom . import window\n"
    found = imported_modules(source, "cpsa.imu.sync", is_package=False)
    assert found == {"cpsa.domain", "cpsa.imu"}
