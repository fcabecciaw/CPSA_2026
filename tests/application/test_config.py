import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from cpsa.application.config import ConfigError, load, parse

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "config" / "board.yaml"


def valid() -> dict[str, Any]:
    return copy.deepcopy(yaml.safe_load(SHIPPED.read_text()))


def test_shipped_config_is_valid() -> None:
    config = load(SHIPPED)
    assert config.bluecoins.left == "CPSA_L2"
    assert config.dispatch.verification_deadline_s == 1.0
    assert config.actuators.retry.fast_attempts == 5
    assert config.actuators.led_strip.duration_ms == 2500


def test_home_is_expanded_in_log_base_path() -> None:
    assert not str(parse(valid()).log.base_path).startswith("~")


def _without(data: dict[str, Any], dotted: str) -> dict[str, Any]:
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node[part]
    del node[leaf]
    return data


def _with(data: dict[str, Any], dotted: str, value: object) -> dict[str, Any]:
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node[part]
    node[leaf] = value
    return data


@pytest.mark.parametrize(
    "key",
    ["log", "bluecoins.right", "dispatch.verification_deadline_s", "actuators.retry.sleep_s"],
)
def test_missing_key_is_named(key: str) -> None:
    with pytest.raises(ConfigError) as info:
        parse(_without(valid(), key))
    assert info.value.key == key
    assert key in str(info.value)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("sync.max_skew_ms", 0),
        ("dispatch.cooldown_s", -1),
        ("dispatch.verification_deadline_s", "fast"),
        ("bluecoins.discovery_retries", 1.5),
        ("bluecoins.discovery_retries", -1),
        ("actuators.retry.fast_attempts", True),
        ("actuators.metamotion.enable", "yes"),
        ("actuators.led_strip.duration_ms", 2.5),
        ("actuators.speaker.mac", ""),
        ("log.level", "VERBOSE"),
        ("bluecoins", ["CPSA_L2"]),
    ],
)
def test_invalid_value_is_named(key: str, value: object) -> None:
    with pytest.raises(ConfigError) as info:
        parse(_with(valid(), key, value))
    assert info.value.key == key


def test_zero_retries_are_allowed() -> None:
    assert parse(_with(valid(), "bluecoins.discovery_retries", 0)).bluecoins.discovery_retries == 0


def test_unknown_key_is_named() -> None:
    with pytest.raises(ConfigError) as info:
        parse(_with(valid(), "sync.max_skew", 30))
    assert info.value.key == "sync.max_skew"


def test_non_mapping_document_is_rejected() -> None:
    with pytest.raises(ConfigError) as info:
        parse(["not", "a", "mapping"])
    assert info.value.key == "<root>"


def test_unreadable_and_malformed_files_name_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigError, match="missing.yaml"):
        load(missing)
    broken = tmp_path / "broken.yaml"
    broken.write_text("log: [unclosed\n")
    with pytest.raises(ConfigError, match="broken.yaml"):
        load(broken)
