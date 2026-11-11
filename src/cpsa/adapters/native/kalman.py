"""Batched 2D Kalman step backed by the local ``libcpsa_kalman2d`` C kernel.

The library is built from ``native/kalman2d`` by the tests on the host, by
``deploy/deploy.sh`` on the board; it is a git-ignored artifact and this module
is the only place that loads it. :func:`load_native_kalman` never raises: a
missing, unloadable or incompatible library logs one WARNING and returns
``None``, so the tracker falls back to the NumPy kernel and the demo goes on.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cpsa.vision.kalman_kernel import STATE_DIM, FilterKernel, KalmanStep

logger = logging.getLogger(__name__)

LIB_DIR = Path(__file__).resolve().parent / "lib"
KALMAN_LIB = LIB_DIR / "libcpsa_kalman2d.so"
LIB_ENV_VAR = "CPSA_KALMAN2D_LIB"
ABI_VERSION = 1

_MEASUREMENT_DIM = 2


class NativeKalmanError(RuntimeError):
    """The native Kalman library is missing, unloadable or incompatible."""


def _symbol(lib: ctypes.CDLL, name: str, path: Path) -> Any:
    try:
        return getattr(lib, name)
    except AttributeError:
        raise NativeKalmanError(f"{path} does not export {name}") from None


class NativeKalmanKernel:
    """``FilterKernel`` backed by ``cpsa_kalman2d_step`` through ``ctypes``.

    Construction loads the library once and binds the frozen F15 signature; it
    raises :class:`NativeKalmanError` when the file is missing or unloadable,
    when a symbol is absent or when the library reports another ABI version.
    """

    def __init__(self, lib_path: Path) -> None:
        path = Path(lib_path)
        try:
            lib = ctypes.CDLL(str(path))
        except (OSError, ValueError) as exc:
            raise NativeKalmanError(f"cannot load {path}: {exc}") from exc
        abi = _symbol(lib, "cpsa_kalman2d_abi_version", path)
        abi.argtypes = []
        abi.restype = ctypes.c_int32
        version = int(abi())
        if version != ABI_VERSION:
            raise NativeKalmanError(
                f"{path} reports Kalman ABI version {version}, expected {ABI_VERSION}"
            )
        self._lib = lib
        self._path = path
        self._step = _symbol(lib, "cpsa_kalman2d_step", path)
        # The frozen F15 order: k, F, Q, x, P, predict, update, z, r, gate_chi2,
        # x_pred, p_pred, x_filt, p_filt, accepted. The buffers are passed by
        # address: converting 14 ndpointer arguments costs about 13 us per call,
        # which is most of the per-frame budget, and step() has already checked
        # every dtype, shape and contiguity before it gets here.
        self._step.argtypes = [
            ctypes.c_int32,
            *([ctypes.c_void_p] * 8),
            ctypes.c_double,
            *([ctypes.c_void_p] * 4),
            ctypes.c_void_p,
        ]
        self._step.restype = ctypes.c_int32

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
        prior = _floats("x", x)
        if prior.ndim != 2 or prior.shape[1] != STATE_DIM:
            raise ValueError(f"x must have shape (k, {STATE_DIM}), got {prior.shape}")
        keypoints = prior.shape[0]
        transition = _floats("F", F)
        noise = _floats("Q", Q)
        if transition.shape != (STATE_DIM, STATE_DIM) or noise.shape != (
            STATE_DIM,
            STATE_DIM,
        ):
            raise ValueError(f"F and Q must have shape ({STATE_DIM}, {STATE_DIM})")
        prior_covariance = _floats("P", P)
        if prior_covariance.shape != (keypoints, STATE_DIM, STATE_DIM):
            raise ValueError(
                f"P must have shape ({keypoints}, {STATE_DIM}, {STATE_DIM}), "
                f"got {prior_covariance.shape}"
            )
        do_predict = _flags("predict", predict, keypoints)
        do_update = _flags("update", update, keypoints)
        position = _floats("z", z)
        if position.shape != (keypoints, _MEASUREMENT_DIM):
            raise ValueError(
                f"z must have shape ({keypoints}, {_MEASUREMENT_DIM}), got {position.shape}"
            )
        variance = _floats("r", r)
        if variance.shape != (keypoints,):
            raise ValueError(f"r must have shape ({keypoints},), got {variance.shape}")

        x_pred = np.empty((keypoints, STATE_DIM), dtype=np.float64)
        p_pred = np.empty((keypoints, STATE_DIM, STATE_DIM), dtype=np.float64)
        x_filt = np.empty((keypoints, STATE_DIM), dtype=np.float64)
        p_filt = np.empty((keypoints, STATE_DIM, STATE_DIM), dtype=np.float64)
        accepted = np.empty(keypoints, dtype=np.uint8)

        status = self._step(
            keypoints,
            transition.ctypes.data,
            noise.ctypes.data,
            prior.ctypes.data,
            prior_covariance.ctypes.data,
            do_predict.ctypes.data,
            do_update.ctypes.data,
            position.ctypes.data,
            variance.ctypes.data,
            float(gate_chi2),
            x_pred.ctypes.data,
            p_pred.ctypes.data,
            x_filt.ctypes.data,
            p_filt.ctypes.data,
            accepted.ctypes.data,
        )
        if status != 0:
            raise RuntimeError(
                f"cpsa_kalman2d_step returned {status} for {keypoints} keypoints from {self._path}"
            )
        return KalmanStep(
            x_pred=x_pred,
            p_pred=p_pred,
            x_filt=x_filt,
            p_filt=p_filt,
            accepted=accepted.astype(np.bool_),
        )


def _floats(name: str, value: Any) -> NDArray[np.float64]:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.shape == ():
        raise ValueError(f"{name} must be an array, got a scalar")
    return array


def _flags(name: str, value: Any, keypoints: int) -> NDArray[np.uint8]:
    array = np.ascontiguousarray(value, dtype=np.uint8)
    if array.shape != (keypoints,):
        raise ValueError(f"{name} must have shape ({keypoints},), got {array.shape}")
    return array


def _candidates(path: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    override = os.environ.get(LIB_ENV_VAR)
    if override:
        candidates.append(Path(override))
    candidates.append(KALMAN_LIB)
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def load_native_kalman(path: Path | None = None) -> FilterKernel | None:
    """Return the native kernel, or ``None`` with one WARNING when it is unusable.

    Tries ``path``, then ``$CPSA_KALMAN2D_LIB``, then the library next to this
    module, and checks the ABI version of the first one that loads. This never
    raises: the caller keeps the NumPy kernel and the tracker keeps running.
    """
    reasons = []
    for candidate in _candidates(path):
        try:
            kernel = NativeKalmanKernel(candidate)
        except NativeKalmanError as exc:
            reasons.append(str(exc))
            continue
        logger.info("Loaded the native Kalman kernel %s", candidate)
        return kernel
    logger.warning(
        "native Kalman kernel unavailable, using the NumPy kernel: %s", "; ".join(reasons)
    )
    return None
