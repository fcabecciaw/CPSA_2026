import ast
import logging
import os
import signal
import sys
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

import cpsa.board.__main__ as board
from cpsa.adapters.gui_null import NullGui
from cpsa.adapters.native.classifier import NativeLibraryError
from cpsa.application.config import Config, load
from cpsa.board.__main__ import BoardAdapters, install_signal_handlers, main, run
from tests.application.test_config import SHIPPED
from tests.fakes import (
    FakeClassifier,
    FakeClock,
    FakeExtractor,
    FakeSensorSource,
    MemorySink,
    RecordingActuator,
    RecordingGui,
    ScriptedVerifier,
)

BOTH = {"CPSA_L2", "CPSA_R2"}


def config(tmp_path: Path, retries: int = 2) -> Config:
    base = load(SHIPPED)
    return replace(
        base,
        log=replace(base.log, base_path=tmp_path),
        bluecoins=replace(base.bluecoins, discovery_retries=retries, retry_interval_s=0.0),
    )


class Board:
    def __init__(self, discoveries: list[set[str]], gui: object | None = None) -> None:
        self.journal: list[str] = []
        self.source = FakeSensorSource(discoveries, self.journal)
        self.actuator = RecordingActuator("meta_AA", journal=self.journal)
        self.actuators_started = False
        self.native_error: Exception | None = None
        self.adapters = BoardAdapters(
            source=self.source,
            load_classifier=self._load_classifier,
            start_actuators=self._start_actuators,
            verifier=ScriptedVerifier(),
            gui=gui if gui is not None else RecordingGui(self.journal),  # type: ignore[arg-type]
            sink=MemorySink(),
            clock=FakeClock(),
        )

    def _load_classifier(self) -> tuple[FakeExtractor, FakeClassifier]:
        if self.native_error is not None:
            raise self.native_error
        return FakeExtractor(), FakeClassifier()

    def _start_actuators(self) -> list[RecordingActuator]:
        self.actuators_started = True
        return [self.actuator]


def stopped_event() -> threading.Event:
    event = threading.Event()
    event.set()
    return event


def test_missing_bluecoin_after_retries_exits_2_naming_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    b = Board([{"CPSA_L2"}])
    with caplog.at_level(logging.ERROR):
        code = run(config(tmp_path, retries=2), b.adapters, threading.Event())

    assert code == 2
    assert b.source.discover_calls == 3  # first scan + 2 retries
    assert "CPSA_R2" in caplog.text and "CPSA_L2" not in caplog.text
    assert not b.actuators_started


def test_bluecoin_found_on_a_retry_starts_the_system(tmp_path: Path) -> None:
    b = Board([set(), {"CPSA_R2"}, BOTH])
    stop_event = threading.Event()
    start_sensors = b.source.start

    def start_then_request_stop(on_sample: object) -> None:
        start_sensors(on_sample)  # type: ignore[arg-type]
        stop_event.set()

    b.source.start = start_then_request_stop  # type: ignore[method-assign]
    assert run(config(tmp_path, retries=2), b.adapters, stop_event) == 0
    assert b.source.discover_calls == 3
    assert "sensors.start" in b.journal


def test_null_gui_run_stops_components_in_order_and_exits_0(tmp_path: Path) -> None:
    b = Board([BOTH], gui=NullGui())
    assert run(config(tmp_path), b.adapters, stopped_event()) == 0
    assert b.journal == ["sensors.start", "sensors.stop", "actuator:meta_AA.stop"]


def test_gui_is_stopped_first(tmp_path: Path) -> None:
    b = Board([BOTH])
    assert run(config(tmp_path), b.adapters, stopped_event()) == 0
    assert b.journal[2:] == ["gui.stop", "sensors.stop", "actuator:meta_AA.stop"]


def test_native_library_failure_exits_2_before_starting_actuators(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    b = Board([BOTH])
    b.native_error = NativeLibraryError("cannot load lib.so on x86_64: requires aarch64")
    with caplog.at_level(logging.ERROR):
        assert run(config(tmp_path), b.adapters, stopped_event()) == 2
    assert "requires aarch64" in caplog.text
    assert not b.actuators_started


def test_unexpected_error_exits_1(tmp_path: Path) -> None:
    b = Board([BOTH])
    b.native_error = RuntimeError("boom")
    assert run(config(tmp_path), b.adapters, stopped_event()) == 1


def test_stop_during_discovery_exits_0(tmp_path: Path) -> None:
    b = Board([set()])
    assert run(config(tmp_path, retries=5), b.adapters, stopped_event()) == 0
    assert b.source.discover_calls == 1


@pytest.mark.parametrize(
    ("contents", "key"),
    [
        (SHIPPED.read_text().replace("\n  base_path:", "\n  base_pth:", 1), "log.base_path"),
        (None, "missing.yaml"),
    ],
    ids=["missing-key", "missing-file"],
)
def test_invalid_config_exits_2_naming_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], contents: str | None, key: str
) -> None:
    path = tmp_path / "missing.yaml"
    if contents is not None:
        path.write_text(contents)
    assert main(["--config", str(path)]) == 2
    assert key in capsys.readouterr().err


def test_missing_pose_model_is_a_vision_startup_error(tmp_path: Path) -> None:
    base = load(SHIPPED)
    vision = replace(
        base.vision,
        enabled=True,
        model=replace(base.vision.model, path=tmp_path / "missing.onnx"),
        evidence=replace(base.vision.evidence, enabled=False),
    )

    with pytest.raises(board.VisionStartupError):
        board._build_verifier(replace(base, vision=vision), FakeClock())


@pytest.fixture
def restore_signal_handlers() -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.mark.usefixtures("restore_signal_handlers")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_termination_signals_request_a_stop(sig: signal.Signals) -> None:
    stop_event = threading.Event()
    install_signal_handlers(stop_event)
    os.kill(os.getpid(), sig)
    assert stop_event.wait(timeout=5)


def test_verifier_is_constructed_in_exactly_one_place() -> None:
    tree = ast.parse(Path(board.__file__).read_text())
    constructions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "verifier"
        and isinstance(node.value, ast.Call)
    ]
    assert len(constructions) == 1


@pytest.fixture
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logger")
def test_missing_board_extra_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "cpsa.adapters.bluecoin.source", None)  # import fails
    path = tmp_path / "board.yaml"
    path.write_text(SHIPPED.read_text().replace("~/Desktop/CPSA_logs", str(tmp_path / "logs")))

    assert main(["--config", str(path)]) == 2
    assert "board' extra" in caplog.text
