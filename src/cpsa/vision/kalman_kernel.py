"""Batched 2D keypoint Kalman step: the frozen kernel interface and its NumPy reference.

Each keypoint of a frame runs the same predict/update arithmetic, so the tracker
can hand all of them to one kernel call instead of ~600 NumPy calls per frame
(NFR-001). The contract below is frozen: the native kernel in
``native/kalman2d`` and its ``ctypes`` adapter implement exactly the same
semantics, and :class:`NumpyKalmanKernel` is both the reference the native
kernel is tested against and the default kernel of the tracker.

Per keypoint ``i``, with ``F`` and ``Q`` shared by the frame:

1. Predict. With ``predict[i]``, ``x_pred = F x`` and
   ``P_pred = F P F^T + Q``; otherwise the prior is copied unchanged.
2. Update. With ``update[i]``, the position innovation is ``y = z - x_pred[0:2]``
   and its covariance ``S = P_pred[0:2, 0:2] + r I2``. The measurement is
   accepted when ``det S`` is finite and positive and the Mahalanobis distance
   ``y^T S^-1 y`` is finite and at most ``gate_chi2``; it is then
   ``x_filt = x_pred + K y`` and ``P_filt = P_pred - K P_pred[0:2, :]``,
   symmetrized, with ``K = P_pred[:, 0:2] S^-1``.
3. Otherwise the filtered state and covariance are the predicted ones.

This is the math of ``KeypointTracker2D._measurement_update`` at commit
535ceef; a ``LinAlgError`` there is a rejection here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["FilterKernel", "KalmanStep", "NumpyKalmanKernel", "STATE_DIM"]

STATE_DIM = 6
_MEASUREMENT_DIM = 2
_IDENTITY_2 = np.eye(_MEASUREMENT_DIM, dtype=np.float64)


@dataclass(frozen=True)
class KalmanStep:
    """Predict and filter outputs of one batched kernel step.

    ``x_pred`` and ``x_filt`` are ``(K, 6)``, ``p_pred`` and ``p_filt`` are
    ``(K, 6, 6)`` and ``accepted`` is a ``(K,)`` boolean mask.
    """

    x_pred: NDArray[np.float64]
    p_pred: NDArray[np.float64]
    x_filt: NDArray[np.float64]
    p_filt: NDArray[np.float64]
    accepted: NDArray[np.bool_]


class FilterKernel(Protocol):
    """The batched predict/update step the tracker delegates to.

    ``F`` and ``Q`` are ``(6, 6)``; ``x`` is ``(K, 6)``, ``P`` is ``(K, 6, 6)``,
    ``predict`` and ``update`` are ``(K,)`` booleans, ``z`` is ``(K, 2)`` and
    ``r`` is ``(K,)``. Implementations never modify their inputs.
    """

    def step(
        self,
        F: NDArray[np.float64],
        Q: NDArray[np.float64],
        x: NDArray[np.float64],
        P: NDArray[np.float64],
        predict: NDArray[np.bool_],
        update: NDArray[np.bool_],
        z: NDArray[np.float64],
        r: NDArray[np.float64],
        gate_chi2: float,
    ) -> KalmanStep: ...


class NumpyKalmanKernel:
    """Batched NumPy implementation of the frozen kernel contract."""

    def step(
        self,
        F: NDArray[np.float64],
        Q: NDArray[np.float64],
        x: NDArray[np.float64],
        P: NDArray[np.float64],
        predict: NDArray[np.bool_],
        update: NDArray[np.bool_],
        z: NDArray[np.float64],
        r: NDArray[np.float64],
        gate_chi2: float,
    ) -> KalmanStep:
        transition = np.asarray(F, dtype=np.float64)
        noise = np.asarray(Q, dtype=np.float64)
        prior = np.asarray(x, dtype=np.float64)
        prior_covariance = np.asarray(P, dtype=np.float64)
        do_predict = np.asarray(predict, dtype=np.bool_)
        do_update = np.asarray(update, dtype=np.bool_)
        position = np.asarray(z, dtype=np.float64)
        variance = np.asarray(r, dtype=np.float64)

        x_pred = np.where(do_predict[:, None], prior @ transition.T, prior)
        p_pred = np.where(
            do_predict[:, None, None],
            transition @ prior_covariance @ transition.T + noise,
            prior_covariance,
        )

        # Rejected measurements are computed too and then discarded, so NaN and
        # infinite inputs raise no spurious floating-point warnings here.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            covariance = p_pred[:, :2, :2] + variance[:, None, None] * _IDENTITY_2
            determinant = (
                covariance[:, 0, 0] * covariance[:, 1, 1]
                - covariance[:, 0, 1] * covariance[:, 1, 0]
            )
            invertible = np.isfinite(determinant) & (determinant > 0.0)
            safe_determinant = np.where(invertible, determinant, 1.0)
            inverse = np.empty_like(covariance)
            inverse[:, 0, 0] = covariance[:, 1, 1] / safe_determinant
            inverse[:, 0, 1] = -covariance[:, 0, 1] / safe_determinant
            inverse[:, 1, 0] = -covariance[:, 1, 0] / safe_determinant
            inverse[:, 1, 1] = covariance[:, 0, 0] / safe_determinant

            innovation = position - x_pred[:, :2]
            distance = np.einsum("ki,kij,kj->k", innovation, inverse, innovation)
            accepted = do_update & invertible & np.isfinite(distance) & (distance <= gate_chi2)
            gain = p_pred[:, :, :2] @ inverse
            x_updated = x_pred + np.einsum("kij,kj->ki", gain, innovation)
            p_updated = p_pred - gain @ p_pred[:, :2, :]
            p_updated = 0.5 * (p_updated + np.swapaxes(p_updated, 1, 2))

        x_filt = np.where(accepted[:, None], x_updated, x_pred)
        p_filt = np.where(accepted[:, None, None], p_updated, p_pred)
        return KalmanStep(
            x_pred=x_pred,
            p_pred=p_pred,
            x_filt=x_filt,
            p_filt=p_filt,
            accepted=accepted,
        )
