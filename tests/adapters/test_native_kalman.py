"""The ctypes kernel equals the NumPy reference (AC-K2) and loads safely (AC-K3).

The shared library is built from ``native/kalman2d/kalman2d.c`` into the pytest
temporary directory, exactly as an operator would build it by hand, so the test
needs ``cc`` but no prebuilt artifact; it is skipped when ``cc`` is absent.
"""

from __future__ import annotations

import ctypes
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cpsa.adapters.native.kalman import (
    ABI_VERSION,
    KALMAN_LIB,
    LIB_ENV_VAR,
    NativeKalmanKernel,
    load_native_kalman,
)
from cpsa.vision.kalman_kernel import NumpyKalmanKernel
from tests.unit import kalman_cases

LOGGER_NAME = "cpsa.adapters.native.kalman"
REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_SOURCE = REPO_ROOT / "native" / "kalman2d" / "kalman2d.c"
CC = shutil.which("cc")

STUB_ABI_MISMATCH = """
#include <stdint.h>
int32_t cpsa_kalman2d_abi_version(void) { return %d; }
int32_t cpsa_kalman2d_step(int32_t k, ...) { (void)k; return 0; }
"""

STUB_MISSING_STEP = """
#include <stdint.h>
int32_t cpsa_kalman2d_abi_version(void) { return 1; }
"""

STUB_FAILING_STEP = """
#include <stdint.h>
int32_t cpsa_kalman2d_abi_version(void) { return 1; }
int32_t cpsa_kalman2d_step(int32_t k, ...) { (void)k; return 7; }
"""


def _compile(source: Path, output: Path, extra: tuple[str, ...] = ()) -> Path:
    """Compile ``source`` into ``output``; skip the test without a C compiler."""
    if CC is None:
        pytest.skip("cc is needed to build a test library")
    result = subprocess.run(
        [CC, "-O2", "-std=c99", "-fPIC", "-shared", "-o", str(output), str(source), *extra],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()
    return output


def _compile_stub(tmp_path: Path, name: str, source_text: str) -> Path:
    """Write ``source_text`` to a C file and compile it into ``tmp_path``."""
    source = tmp_path / f"{name}.c"
    source.write_text(source_text)
    return _compile(source, tmp_path / f"{name}.so")


@pytest.fixture(scope="session")
def kalman_lib(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the kernel from the repository source into a temporary directory."""
    output = tmp_path_factory.mktemp("kalman2d") / "libcpsa_kalman2d.so"
    return _compile(KERNEL_SOURCE, output, ("-lm",))


@pytest.fixture(scope="session")
def native(kalman_lib: Path) -> NativeKalmanKernel:
    return NativeKalmanKernel(kalman_lib)


@pytest.fixture(autouse=True)
def _no_library_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's ``CPSA_KALMAN2D_LIB`` out of the loader tests."""
    monkeypatch.delenv(LIB_ENV_VAR, raising=False)


def _snapshot(case: kalman_cases.StepInput) -> dict[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in case.inputs().items()}


def _assert_matches_numpy(case: kalman_cases.StepInput, native: NativeKalmanKernel) -> None:
    before = _snapshot(case)
    expected = case.step(NumpyKalmanKernel())
    result = case.step(native)
    assert result.accepted.dtype == np.bool_
    np.testing.assert_array_equal(result.accepted, expected.accepted)
    np.testing.assert_allclose(result.x_pred, expected.x_pred, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.p_pred, expected.p_pred, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.x_filt, expected.x_filt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.p_filt, expected.p_filt, rtol=1e-12, atol=1e-12)
    for key, value in case.inputs().items():
        np.testing.assert_array_equal(value, before[key])


SCENES = kalman_cases.scenes()


@pytest.mark.parametrize(("name", "case"), SCENES, ids=[name for name, _ in SCENES])
def test_matches_numpy_on_scenes(
    name: str, case: kalman_cases.StepInput, native: NativeKalmanKernel
) -> None:
    _assert_matches_numpy(case, native)


def test_matches_numpy_on_ten_thousand_keypoint_steps(native: NativeKalmanKernel) -> None:
    steps = kalman_cases.random_keypoint_steps()
    assert sum(step.keypoints for step in steps) == 10_000
    for case in steps:
        _assert_matches_numpy(case, native)


def test_library_reports_the_frozen_abi(kalman_lib: Path) -> None:
    library = ctypes.CDLL(str(kalman_lib))
    library.cpsa_kalman2d_abi_version.restype = ctypes.c_int32
    assert ABI_VERSION == 1
    assert library.cpsa_kalman2d_abi_version() == ABI_VERSION


def test_step_raises_on_a_non_zero_return(tmp_path: Path) -> None:
    kernel = NativeKalmanKernel(_compile_stub(tmp_path, "failing", STUB_FAILING_STEP))
    case = dict(SCENES)["measured_1"]
    with pytest.raises(RuntimeError, match="returned 7"):
        case.step(kernel)


def test_loader_returns_the_kernel(kalman_lib: Path) -> None:
    assert isinstance(load_native_kalman(kalman_lib), NativeKalmanKernel)


def test_loader_uses_the_environment_override(
    kalman_lib: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(LIB_ENV_VAR, str(kalman_lib))
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        kernel = load_native_kalman()
    assert isinstance(kernel, NativeKalmanKernel)
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_loader_uses_the_explicit_path_before_the_override(
    kalman_lib: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LIB_ENV_VAR, str(Path("/nonexistent/libcpsa_kalman2d.so")))
    assert isinstance(load_native_kalman(kalman_lib), NativeKalmanKernel)


def test_loader_falls_back_to_the_override(
    kalman_lib: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LIB_ENV_VAR, str(kalman_lib))
    assert isinstance(
        load_native_kalman(Path("/nonexistent/libcpsa_kalman2d.so")), NativeKalmanKernel
    )


def _assert_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    warnings = [
        record
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert "NumPy kernel" in warnings[0].getMessage()


def test_loader_returns_none_for_a_missing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert load_native_kalman(tmp_path / "absent.so") is None
    _assert_one_warning(caplog)


def test_loader_returns_none_for_a_non_elf_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    not_elf = tmp_path / "libcpsa_kalman2d.so"
    not_elf.write_text("this is not a shared library\n")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert load_native_kalman(not_elf) is None
    _assert_one_warning(caplog)


def test_loader_returns_none_for_a_path_ctypes_rejects(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # ctypes raises ValueError rather than OSError for an embedded null byte;
    # the loader must still return None instead of raising (AC-K3).
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert load_native_kalman(Path("libcpsa\0kalman2d.so")) is None
    _assert_one_warning(caplog)


def test_loader_returns_none_for_a_missing_symbol(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    library = _compile_stub(tmp_path, "missing_step", STUB_MISSING_STEP)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert load_native_kalman(library) is None
    _assert_one_warning(caplog)


def test_loader_returns_none_for_an_abi_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    library = _compile_stub(tmp_path, "abi2", STUB_ABI_MISMATCH % (ABI_VERSION + 1))
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert load_native_kalman(library) is None
    _assert_one_warning(caplog)


def test_loader_warns_for_the_default_location(
    caplog: pytest.LogCaptureFixture,
) -> None:
    if KALMAN_LIB.exists():
        pytest.skip(f"{KALMAN_LIB} exists, so the default location is usable")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert load_native_kalman() is None
    _assert_one_warning(caplog)
