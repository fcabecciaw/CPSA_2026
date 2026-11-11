"""``cpsa-board`` entry point: it rejects the camera-only medical trigger (R-05).

``cpsa-board`` keeps the IMU-triggered pipeline; the camera-only trigger is run
by ``cpsa-supervisor``'s medical mode. A ``medical.trigger: vision`` config must
fail with a clear message before any adapter (and thus the BLE import/discovery)
is built, instead of surfacing a BlueCoin discovery error from the no-op source.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import cpsa.board.__main__ as board_main
from cpsa.application.config import Config, load
from tests.application.test_config import SHIPPED


class _ReachedAdapters(Exception):
    """Sentinel: ``main`` passed the trigger check and started building adapters."""


def _vision_config() -> Config:
    config = load(SHIPPED)
    return replace(config, medical=replace(config.medical, trigger="vision"))


@pytest.fixture(autouse=True)
def silent_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``main`` from touching the log directory or the real adapters."""
    monkeypatch.setattr(board_main, "configure_logging", lambda *args, **kwargs: None)


def test_main_rejects_the_camera_only_trigger(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(board_main, "load", lambda path: _vision_config())
    built: list[Config] = []

    def build_adapters(config: Config, **kwargs: object) -> None:
        built.append(config)
        raise _ReachedAdapters

    monkeypatch.setattr(board_main, "board_adapters", build_adapters)

    assert board_main.main(["--config", str(SHIPPED)]) == board_main.EXIT_STARTUP
    assert built == []  # no adapter, so no BLE import and no discovery
    err = capsys.readouterr().err
    assert "medical.trigger is 'vision'" in err
    assert "cpsa-board is IMU-only" in err
    assert "cpsa-supervisor" in err


def test_main_lets_an_imu_config_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(board_main, "load", lambda path: load(Path(path)))

    def build_adapters(config: Config, **kwargs: object) -> None:
        raise _ReachedAdapters

    monkeypatch.setattr(board_main, "board_adapters", build_adapters)

    with pytest.raises(_ReachedAdapters):
        board_main.main(["--config", str(SHIPPED)])
