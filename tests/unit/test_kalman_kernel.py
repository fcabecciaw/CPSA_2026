"""``NumpyKalmanKernel`` equals the per-keypoint tracker math (AC-K1).

The reference below is a copy of the predict/update arithmetic of
``src/cpsa/vision/tracker2d.py`` at commit 535ceef, the commit the native kernel
plan froze as the contract: the same ``F x`` / ``F P F^T + Q`` loop,
``np.linalg.solve`` and ``np.linalg.pinv`` for the innovation, ``(I - K H) P``
for the filtered covariance and the same chi-square gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from cpsa.vision.kalman_kernel import KalmanStep, NumpyKalmanKernel
from tests.unit import kalman_cases

_STATE_DIM = 6
_IDENTITY = np.eye(_STATE_DIM, dtype=np.float64)
_MEASUREMENT_MATRIX = np.zeros((2, _STATE_DIM), dtype=np.float64)
_MEASUREMENT_MATRIX[0, 0] = 1.0
_MEASUREMENT_MATRIX[1, 1] = 1.0


def _measurement_update(
    x_pred: NDArray[np.float64],
    p_pred: NDArray[np.float64],
    position: NDArray[np.float64],
    variance: float,
    gate_chi2: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Copy of ``KeypointTracker2D._measurement_update`` at 535ceef.

    Only ``self._measurement_matrix``, ``self._cfg.gate_chi2`` and
    ``self._measurement.noise(conf)`` are substituted by their values:
    ``_MEASUREMENT_MATRIX``, ``gate_chi2`` and ``diag([variance, variance])``.
    """
    noise = np.diag(np.array([variance, variance], dtype=np.float64))
    innovation = position - _MEASUREMENT_MATRIX @ x_pred
    covariance = _MEASUREMENT_MATRIX @ p_pred @ _MEASUREMENT_MATRIX.T + noise
    try:
        # One scene feeds non-finite measurements on purpose: the rejection
        # below is the expected outcome, so silence the invalid-value warnings.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            distance = float(innovation @ np.linalg.solve(covariance, innovation))
            gain = np.asarray(
                p_pred @ _MEASUREMENT_MATRIX.T @ np.linalg.pinv(covariance),
                dtype=np.float64,
            )
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.isfinite(distance) or distance > gate_chi2:
        return None
    return gain, innovation


def reference_step(case: kalman_cases.StepInput) -> KalmanStep:
    """Run the 535ceef predict/update loop over one keypoint at a time."""
    size = case.keypoints
    x_pred = np.array(case.x, dtype=np.float64, copy=True)
    p_pred = np.array(case.P, dtype=np.float64, copy=True)
    for index in np.flatnonzero(case.predict):
        keypoint = int(index)
        x_pred[keypoint] = case.F @ case.x[keypoint]
        p_pred[keypoint] = case.F @ case.P[keypoint] @ case.F.T + case.Q

    x_filt = x_pred.copy()
    p_filt = p_pred.copy()
    accepted = np.zeros(size, dtype=np.bool_)
    for keypoint in range(size):
        if not case.update[keypoint]:
            continue
        update = _measurement_update(
            x_pred[keypoint],
            p_pred[keypoint],
            case.z[keypoint],
            float(case.r[keypoint]),
            case.gate_chi2,
        )
        if update is None:
            continue
        gain, innovation = update
        x_filt[keypoint] = x_pred[keypoint] + gain @ innovation
        p_filt[keypoint] = (_IDENTITY - gain @ _MEASUREMENT_MATRIX) @ p_pred[keypoint]
        p_filt[keypoint] = 0.5 * (p_filt[keypoint] + p_filt[keypoint].T)
        accepted[keypoint] = True
    return KalmanStep(
        x_pred=x_pred,
        p_pred=p_pred,
        x_filt=x_filt,
        p_filt=p_filt,
        accepted=accepted,
    )


SCENES = kalman_cases.scenes()


@pytest.mark.parametrize(("name", "case"), SCENES, ids=[name for name, _ in SCENES])
def test_matches_the_per_keypoint_reference(name: str, case: kalman_cases.StepInput) -> None:
    result = case.step(NumpyKalmanKernel())
    expected = reference_step(case)
    assert result.accepted.dtype == np.bool_
    np.testing.assert_array_equal(result.accepted, expected.accepted)
    np.testing.assert_allclose(result.x_pred, expected.x_pred, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.p_pred, expected.p_pred, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.x_filt, expected.x_filt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.p_filt, expected.p_filt, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("name", "case"), SCENES, ids=[name for name, _ in SCENES])
def test_inputs_are_not_modified(name: str, case: kalman_cases.StepInput) -> None:
    before = {key: np.array(value, copy=True) for key, value in case.inputs().items()}
    case.step(NumpyKalmanKernel())
    for key, value in case.inputs().items():
        np.testing.assert_array_equal(value, before[key])


def test_gated_measurements_are_rejected() -> None:
    case = dict(SCENES)["gated_outliers"]
    result = case.step(NumpyKalmanKernel())
    assert not result.accepted.any()
    np.testing.assert_array_equal(result.x_filt, result.x_pred)
    np.testing.assert_array_equal(result.p_filt, result.p_pred)


def test_singular_innovation_is_rejected() -> None:
    case = dict(SCENES)["singular_innovation"]
    assert not case.step(NumpyKalmanKernel()).accepted.any()


def test_empty_frame_needs_no_arrays() -> None:
    case = dict(SCENES)["empty"]
    result: Any = case.step(NumpyKalmanKernel())
    assert result.x_pred.shape == (0, _STATE_DIM)
    assert result.p_filt.shape == (0, _STATE_DIM, _STATE_DIM)
