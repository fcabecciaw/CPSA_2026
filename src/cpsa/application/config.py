"""Typed, validated board configuration.

Every key is required and there are no code defaults: the YAML file is the single
source of configuration. Validation failures raise :class:`ConfigError` naming the
dotted key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(Exception):
    """The configuration is missing, unreadable or invalid at ``key``."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(f"invalid configuration at '{key}': {reason}")
        self.key = key
        self.reason = reason


@dataclass(frozen=True)
class LogConfig:
    base_path: Path
    level: str


@dataclass(frozen=True)
class BlueCoinsConfig:
    left: str
    right: str
    scan_timeout_s: float
    discovery_retries: int
    retry_interval_s: float


@dataclass(frozen=True)
class SyncConfig:
    max_skew_ms: float
    stale_ms: float


@dataclass(frozen=True)
class DispatchConfig:
    cooldown_s: float
    verification_deadline_s: float


@dataclass(frozen=True)
class RetryConfig:
    scan_timeout_s: float
    fast_attempts: int
    interval_s: float
    sleep_s: float


@dataclass(frozen=True)
class MetaMotionConfig:
    enable: bool


@dataclass(frozen=True)
class SpeakerConfig:
    enable: bool
    mac: str


@dataclass(frozen=True)
class LedStripConfig:
    enable: bool
    duration_ms: int


@dataclass(frozen=True)
class ActuatorsConfig:
    retry: RetryConfig
    metamotion: MetaMotionConfig
    speaker: SpeakerConfig
    led_strip: LedStripConfig


@dataclass(frozen=True)
class Config:
    log: LogConfig
    bluecoins: BlueCoinsConfig
    sync: SyncConfig
    dispatch: DispatchConfig
    actuators: ActuatorsConfig


def load(path: Path) -> Config:
    """Read and validate the YAML configuration at ``path``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(str(path), f"cannot read file: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(str(path), f"invalid YAML: {exc}") from exc
    return parse(data)


def parse(data: object) -> Config:
    """Validate an already parsed configuration document."""
    root = _Section(data, "")
    log = root.section("log")
    bluecoins = root.section("bluecoins")
    sync = root.section("sync")
    dispatch = root.section("dispatch")
    actuators = root.section("actuators")
    retry = actuators.section("retry")
    metamotion = actuators.section("metamotion")
    speaker = actuators.section("speaker")
    led_strip = actuators.section("led_strip")

    config = Config(
        log=LogConfig(
            base_path=Path(log.text("base_path")).expanduser(),
            level=log.choice("level", LOG_LEVELS),
        ),
        bluecoins=BlueCoinsConfig(
            left=bluecoins.text("left"),
            right=bluecoins.text("right"),
            scan_timeout_s=bluecoins.positive("scan_timeout_s"),
            discovery_retries=bluecoins.count("discovery_retries"),
            retry_interval_s=bluecoins.positive("retry_interval_s"),
        ),
        sync=SyncConfig(
            max_skew_ms=sync.positive("max_skew_ms"),
            stale_ms=sync.positive("stale_ms"),
        ),
        dispatch=DispatchConfig(
            cooldown_s=dispatch.positive("cooldown_s"),
            verification_deadline_s=dispatch.positive("verification_deadline_s"),
        ),
        actuators=ActuatorsConfig(
            retry=RetryConfig(
                scan_timeout_s=retry.positive("scan_timeout_s"),
                fast_attempts=retry.count("fast_attempts"),
                interval_s=retry.positive("interval_s"),
                sleep_s=retry.positive("sleep_s"),
            ),
            metamotion=MetaMotionConfig(enable=metamotion.flag("enable")),
            speaker=SpeakerConfig(enable=speaker.flag("enable"), mac=speaker.text("mac")),
            led_strip=LedStripConfig(
                enable=led_strip.flag("enable"),
                duration_ms=led_strip.positive_int("duration_ms"),
            ),
        ),
    )
    sections = (root, log, bluecoins, sync, dispatch, actuators, retry, metamotion, speaker)
    for section in (*sections, led_strip):
        section.reject_unknown()
    return config


class _Section:
    """A mapping in the document, tracking which keys were consumed."""

    def __init__(self, data: object, path: str) -> None:
        if not isinstance(data, Mapping):
            raise ConfigError(path or "<root>", "expected a mapping")
        self._data = data
        self._path = path
        self._used: set[str] = set()

    def _key(self, name: str) -> str:
        return f"{self._path}.{name}" if self._path else name

    def _get(self, name: str) -> object:
        if name not in self._data:
            raise ConfigError(self._key(name), "missing required key")
        self._used.add(name)
        return self._data[name]

    def section(self, name: str) -> _Section:
        return _Section(self._get(name), self._key(name))

    def text(self, name: str) -> str:
        value = self._get(name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(self._key(name), "expected a non-empty string")
        return value

    def choice(self, name: str, options: tuple[str, ...]) -> str:
        value = self.text(name)
        if value not in options:
            raise ConfigError(self._key(name), f"expected one of {', '.join(options)}")
        return value

    def flag(self, name: str) -> bool:
        value = self._get(name)
        if not isinstance(value, bool):
            raise ConfigError(self._key(name), "expected true or false")
        return value

    def positive(self, name: str) -> float:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError(self._key(name), "expected a positive number")
        return float(value)

    def positive_int(self, name: str) -> int:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(self._key(name), "expected a positive integer")
        return value

    def count(self, name: str) -> int:
        value = self._get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(self._key(name), "expected an integer >= 0")
        return value

    def reject_unknown(self) -> None:
        unknown = sorted(str(k) for k in self._data if k not in self._used)
        if unknown:
            raise ConfigError(self._key(unknown[0]), "unknown key")
