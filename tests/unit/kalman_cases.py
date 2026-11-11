"""Step inputs shared by the NumPy reference test and the native kernel test.

The scenarios cover what the tracker can hand the kernel in one frame: an empty
frame, one keypoint, a full 62-keypoint layout, mixed predict/update masks,
gated outliers, non-finite measurements and a singular innovation covariance.
``random_keypoint_steps`` adds the randomized sweep both kernels are compared
on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

STATE_DIM = 6
GATE_CHI2 = 9.21
DT = 1.0 / 30.0
JERK_STD_PXPS3 = 1000.0
POSITION_STD_PX = 3.0
_KEYPOINTS = 62
# Initial one-sigma of a new keypoint, from Tracker2DConfig: 20 px, 1000 px/s,
# 10000 px/s^2. Scaling the random covariances by them keeps the conditioning
# and the magnitudes realistic.
_INITIAL_STD = np.array([20.0, 20.0, 1000.0, 1000.0, 10000.0, 10000.0])


@dataclass(frozen=True)
class StepInput:
    """The nine arguments of one ``FilterKernel.step`` call."""

    F: NDArray[np.float64]
    Q: NDArray[np.float64]
    x: NDArray[np.float64]
    P: NDArray[np.float64]
    predict: NDArray[np.bool_]
    update: NDArray[np.bool_]
    z: NDArray[np.float64]
    r: NDArray[np.float64]
    gate_chi2: float = GATE_CHI2

    @property
    def keypoints(self) -> int:
        """Number of keypoints in this step."""
        return int(self.x.shape[0])

    def inputs(self) -> dict[str, Any]:
        """Return the array arguments, for the unmodified-input checks."""
        return {
            "F": self.F,
            "Q": self.Q,
            "x": self.x,
            "P": self.P,
            "predict": self.predict,
            "update": self.update,
            "z": self.z,
            "r": self.r,
        }

    def step(self, kernel: Any) -> Any:
        """Run ``kernel.step`` on this input."""
        return kernel.step(
            self.F,
            self.Q,
            self.x,
            self.P,
            self.predict,
            self.update,
            self.z,
            self.r,
            self.gate_chi2,
        )


def transition(dt: float) -> NDArray[np.float64]:
    """The constant-acceleration state transition of ``_ConstantAcceleration``."""
    matrix = np.eye(STATE_DIM, dtype=np.float64)
    half_dt2 = 0.5 * dt * dt
    matrix[0, 2] = dt
    matrix[0, 4] = half_dt2
    matrix[1, 3] = dt
    matrix[1, 5] = half_dt2
    matrix[2, 4] = dt
    matrix[3, 5] = dt
    return matrix


def process_noise(dt: float, jerk_std_pxps3: float = JERK_STD_PXPS3) -> NDArray[np.float64]:
    """The white-jerk process noise of ``_ConstantAcceleration``."""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    block = np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0, dt2 / 2.0, dt],
        ],
        dtype=np.float64,
    )
    block *= jerk_std_pxps3 * jerk_std_pxps3
    covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            covariance[2 * row, 2 * column] = block[row, column]
            covariance[2 * row + 1, 2 * column + 1] = block[row, column]
    return covariance


def _random_states(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    states = np.empty((count, STATE_DIM), dtype=np.float64)
    states[:, 0] = rng.uniform(0.0, 1920.0, count)
    states[:, 1] = rng.uniform(0.0, 1080.0, count)
    states[:, 2] = rng.normal(0.0, 200.0, count)
    states[:, 3] = rng.normal(0.0, 200.0, count)
    states[:, 4] = rng.normal(0.0, 2000.0, count)
    states[:, 5] = rng.normal(0.0, 2000.0, count)
    return states


def _random_covariances(rng: np.random.Generator, count: int) -> NDArray[np.float64]:
    """Prior covariances shaped the way the tracker hands them to the kernel.

    A random diagonal prior spread around the initial uncertainty of a new
    keypoint, carried through one constant-acceleration predict step. Random
    SPD matrices would also exercise the contract, but their position blocks
    can be near-singular, and there ``np.linalg.pinv`` (which truncates small
    singular values) and the explicit 2x2 inverse stop agreeing to rounding;
    the tracker never produces such an ``S`` because ``R`` keeps it above
    ``meas_std_px^2 * I``.
    """
    diagonal = (_INITIAL_STD**2) * rng.uniform(0.5, 2.0, (count, STATE_DIM))
    prior = np.einsum("ki,ij->kij", diagonal, np.eye(STATE_DIM))
    transition_matrix = transition(DT)
    noise = process_noise(DT)
    return transition_matrix @ prior @ transition_matrix.T + noise


def _measured(
    rng: np.random.Generator, states: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Positions near the prediction and one measurement variance per keypoint."""
    predicted = states @ transition(DT).T
    positions = predicted[:, :2] + rng.normal(0.0, POSITION_STD_PX, (len(states), 2))
    confidences = rng.uniform(0.3, 1.0, len(states))
    return positions, POSITION_STD_PX**2 / confidences


def scenes() -> list[tuple[str, StepInput]]:
    """Return the fixed scenarios as ``(name, step input)`` pairs."""
    rng = np.random.default_rng(20261111)
    F = transition(DT)
    Q = process_noise(DT)
    cases: list[tuple[str, StepInput]] = []

    empty = np.zeros(0, dtype=np.bool_)
    cases.append(
        (
            "empty",
            StepInput(
                F=F,
                Q=Q,
                x=np.zeros((0, STATE_DIM)),
                P=np.zeros((0, STATE_DIM, STATE_DIM)),
                predict=empty,
                update=empty,
                z=np.zeros((0, 2)),
                r=np.zeros(0),
            ),
        )
    )

    for count in (1, _KEYPOINTS):
        states = _random_states(rng, count)
        covariance = _random_covariances(rng, count)
        do_predict = np.ones(count, dtype=np.bool_)
        do_update = np.ones(count, dtype=np.bool_)
        positions, variances = _measured(rng, states)
        cases.append(
            (
                f"measured_{count}",
                StepInput(F, Q, states, covariance, do_predict, do_update, positions, variances),
            )
        )
        mixed_predict = rng.random(count) < 0.6
        mixed_update = rng.random(count) < 0.5
        positions, variances = _measured(rng, states)
        cases.append(
            (
                f"mixed_masks_{count}",
                StepInput(
                    F, Q, states, covariance, mixed_predict, mixed_update, positions, variances
                ),
            )
        )

    states = _random_states(rng, _KEYPOINTS)
    covariance = _random_covariances(rng, _KEYPOINTS)
    _, variances = _measured(rng, states)
    predicted = states @ F.T
    outliers = predicted[:, :2] + rng.choice([-1.0, 1.0], (_KEYPOINTS, 2)) * 5000.0
    cases.append(
        (
            "gated_outliers",
            StepInput(
                F,
                Q,
                states,
                covariance,
                np.ones(_KEYPOINTS, dtype=np.bool_),
                np.ones(_KEYPOINTS, dtype=np.bool_),
                outliers,
                variances,
            ),
        )
    )

    broken = np.array(predicted[:, :2], dtype=np.float64, copy=True)
    broken[1::3, 0] = np.nan
    broken[2::3, 1] = np.inf
    cases.append(
        (
            "non_finite_measurement",
            StepInput(
                F,
                Q,
                states,
                covariance,
                np.ones(_KEYPOINTS, dtype=np.bool_),
                np.ones(_KEYPOINTS, dtype=np.bool_),
                broken,
                variances,
            ),
        )
    )

    singular_states = _random_states(rng, 3)
    singular_covariance = _random_covariances(rng, 3)
    singular_covariance[:, :2, :2] = 0.0
    singular_covariance[:, :2, 2:] = 0.0
    singular_covariance[:, 2:, :2] = 0.0
    cases.append(
        (
            "singular_innovation",
            StepInput(
                F,
                Q,
                singular_states,
                singular_covariance,
                np.zeros(3, dtype=np.bool_),
                np.ones(3, dtype=np.bool_),
                singular_states[:, :2] + 5.0,
                np.zeros(3),
            ),
        )
    )
    return cases


def random_keypoint_steps(count: int = 10_000, seed: int = 7) -> list[StepInput]:
    """Return random mixed-mask steps covering ``count`` keypoints in total."""
    rng = np.random.default_rng(seed)
    F = transition(DT)
    Q = process_noise(DT)
    steps: list[StepInput] = []
    remaining = count
    while remaining > 0:
        keypoints = min(100, remaining)
        remaining -= keypoints
        states = _random_states(rng, keypoints)
        covariance = _random_covariances(rng, keypoints)
        do_predict = rng.random(keypoints) < 0.7
        do_update = rng.random(keypoints) < 0.7
        positions, variances = _measured(rng, states)
        steps.append(
            StepInput(F, Q, states, covariance, do_predict, do_update, positions, variances)
        )
    return steps
