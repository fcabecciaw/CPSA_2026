"""Scripted trial protocol: parsing, validation and deterministic scheduling.

The YAML file lists classes with a hand (``L``, ``R`` or ``both``), a target, a
number of repetitions and a trial duration. ``both`` expands into one ``L`` and one
``R`` trial per repetition. The protocol is run once per distance block; the trial
plan is shuffled with ``seed`` so the same seed always produces the same order.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

HANDS = ("L", "R", "both")
CATEGORIES = ("dangerous", "confounder", "stereotypy", "idle")
_ROOT_KEYS = (
    "seed",
    "distances_cm",
    "pre_s",
    "post_s",
    "rest_s",
    "cue_lead_s",
    "classes",
)
_CLASS_KEYS = ("name", "category", "hand", "target", "reps", "duration_s")


class ProtocolError(Exception):
    """The protocol is missing, unreadable or invalid at ``field``."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"invalid protocol at '{field}': {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class ProtocolClass:
    """One class definition in ``protocol.yaml``."""

    name: str
    category: str
    hand: str
    target: str
    reps: int
    duration_s: float


@dataclass(frozen=True)
class Protocol:
    """A validated protocol document."""

    seed: int
    distances_cm: tuple[int, ...]
    pre_s: float
    post_s: float
    rest_s: float
    cue_lead_s: float
    classes: tuple[ProtocolClass, ...]
    notes: str = ""


@dataclass(frozen=True)
class TrialSpec:
    """One scheduled trial, before its timestamps are known."""

    trial_id: int
    block_index: int
    distance_cm: int
    class_name: str
    category: str
    hand: str
    target: str
    duration_s: float
    rep: int


def load(path: Path) -> Protocol:
    """Read and validate the protocol at ``path``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProtocolError(str(path), f"cannot read file: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ProtocolError(str(path), f"invalid YAML: {exc}") from exc
    return parse(data)


def parse(data: object) -> Protocol:
    """Validate an already parsed protocol document."""
    root = _Section(data, "")
    classes_data = _require(root, "classes")
    if not isinstance(classes_data, Sequence) or isinstance(classes_data, (str, bytes)):
        raise ProtocolError("classes", "expected a list")
    classes = tuple(
        _parse_class(_Section(item, f"classes[{index}]"), f"classes[{index}]")
        for index, item in enumerate(classes_data)
    )
    names = [item.name for item in classes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ProtocolError("classes", f"duplicate class name(s): {', '.join(duplicates)}")

    distances = _require(root, "distances_cm")
    if not isinstance(distances, Sequence) or isinstance(distances, (str, bytes)):
        raise ProtocolError("distances_cm", "expected a non-empty list of integers")
    if not distances:
        raise ProtocolError("distances_cm", "expected a non-empty list of integers")
    parsed_distances = tuple(
        _positive_int(value, f"distances_cm[{index}]") for index, value in enumerate(distances)
    )

    protocol = Protocol(
        seed=_integer(_require(root, "seed"), "seed"),
        distances_cm=parsed_distances,
        pre_s=_non_negative(_require(root, "pre_s"), "pre_s"),
        post_s=_non_negative(_require(root, "post_s"), "post_s"),
        rest_s=_non_negative(_require(root, "rest_s"), "rest_s"),
        cue_lead_s=_non_negative(_require(root, "cue_lead_s"), "cue_lead_s"),
        classes=classes,
        notes=_optional_text(root.optional("notes"), "notes"),
    )
    root.reject_unknown()
    return protocol


def plan_trials(protocol: Protocol) -> tuple[TrialSpec, ...]:
    """Expand the protocol into distance blocks and a deterministic shuffled order."""
    rng = random.Random(protocol.seed)
    trials: list[TrialSpec] = []
    trial_id = 0
    for block_index, distance_cm in enumerate(protocol.distances_cm):
        block: list[TrialSpec] = []
        for item in protocol.classes:
            hands = ("L", "R") if item.hand == "both" else (item.hand,)
            for hand in hands:
                for rep in range(1, item.reps + 1):
                    block.append(
                        TrialSpec(
                            trial_id=0,
                            block_index=block_index,
                            distance_cm=distance_cm,
                            class_name=item.name,
                            category=item.category,
                            hand=hand,
                            target=item.target,
                            duration_s=item.duration_s,
                            rep=rep,
                        )
                    )
        rng.shuffle(block)
        for spec in block:
            trial_id += 1
            trials.append(replace(spec, trial_id=trial_id))
    return tuple(trials)


def protocol_sha256(path: Path) -> str:
    """Return the SHA-256 of the protocol bytes, as recorded in ``session.json``."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_class(section: _Section, where: str) -> ProtocolClass:
    item = ProtocolClass(
        name=section.text("name"),
        category=section.choice("category", CATEGORIES),
        hand=section.choice("hand", HANDS),
        target=section.text("target"),
        reps=section.positive_int("reps"),
        duration_s=section.positive("duration_s"),
    )
    section.reject_unknown()
    if not item.name.replace("_", "").isalnum():
        raise ProtocolError(f"{where}.name", "use letters, digits and underscores only")
    return item


class _Section:
    """A mapping in the document, tracking which keys were consumed."""

    def __init__(self, data: object, path: str) -> None:
        if not isinstance(data, Mapping):
            raise ProtocolError(path or "<root>", "expected a mapping")
        self._data = data
        self._path = path
        self._used: set[str] = set()

    def _key(self, name: str) -> str:
        return f"{self._path}.{name}" if self._path else name

    def optional(self, name: str) -> object:
        self._used.add(name)
        return self._data.get(name)

    def section(self, name: str) -> _Section:
        return _Section(_require(self, name), self._key(name))

    def text(self, name: str) -> str:
        return _string(_require(self, name), self._key(name))

    def choice(self, name: str, options: tuple[str, ...]) -> str:
        value = self.text(name)
        if value not in options:
            raise ProtocolError(self._key(name), f"expected one of {', '.join(options)}")
        return value

    def positive(self, name: str) -> float:
        return _positive(_require(self, name), self._key(name))

    def positive_int(self, name: str) -> int:
        return _positive_int(_require(self, name), self._key(name))

    def reject_unknown(self) -> None:
        unknown = sorted(str(key) for key in self._data if key not in self._used)
        if unknown:
            raise ProtocolError(self._key(unknown[0]), "unknown key")


def _require(section: _Section, name: str) -> object:
    if name not in section._data:
        raise ProtocolError(section._key(name), "missing required key")
    section._used.add(name)
    return section._data[name]


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(where, "expected a non-empty string")
    return value


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(where, "expected an integer")
    return value


def _positive_int(value: object, where: str) -> int:
    number = _integer(value, where)
    if number < 1:
        raise ProtocolError(where, "expected an integer >= 1")
    return number


def _positive(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(where, "expected a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ProtocolError(where, "expected a finite positive number")
    return number


def _non_negative(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(where, "expected a number >= 0")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ProtocolError(where, "expected a finite number >= 0")
    return number


def _optional_text(value: object, where: str) -> str:
    if value is None:
        return ""
    return _string(value, where)
