"""Feature extractor and classifier backed by the vendored native libraries.

See ``lib/PROVENANCE.md`` for the origin, hashes and C signatures of the libraries.
They are built for aarch64 only and are loaded lazily, on first construction of an
adapter, so the package stays importable on any host.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import threading
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from numpy.typing import NDArray

from cpsa.domain.model import ClassificationError, Tag

logger = logging.getLogger(__name__)

LIB_DIR = Path(__file__).resolve().parent / "lib"
EXTRACTOR_LIB = LIB_DIR / "libProcessDataWristsQuat.so"
CLASSIFIER_LIB = LIB_DIR / "libPredictPericolosaWristsQuat.so"

WINDOW = 150
FEATURES = 18

# Native argument order expressed as columns of the preprocessed (150, 20) window,
# whose layout is R acc xyz, R gyr xyz, R quat xyzw, L acc xyz, L gyr xyz, L quat xyzw.
# The native ABI expects R acc/gyr, L acc/gyr, R quat, L quat.
_NATIVE_COLUMN_ORDER = (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 6, 7, 8, 9, 16, 17, 18, 19)

_Float150 = ndpointer(dtype=np.float32, shape=(WINDOW,), flags="C_CONTIGUOUS")
_Float18 = ndpointer(dtype=np.float32, shape=(FEATURES,), flags="C_CONTIGUOUS")

_load_lock = threading.Lock()
_loaded: dict[Path, ctypes.CDLL] = {}


class NativeLibraryError(RuntimeError):
    """A native library could not be loaded on this host."""


def _load(path: Path, init_symbols: tuple[str, ...]) -> ctypes.CDLL:
    """Load ``path`` once per process and call its first available init symbol."""
    with _load_lock:
        if path in _loaded:
            return _loaded[path]
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise NativeLibraryError(
                f"cannot load {path} on {platform.machine()}: the library requires aarch64 ({exc})"
            ) from exc
        for symbol in init_symbols:
            init = getattr(lib, symbol, None)
            if init is not None:
                init.restype = None
                init()
                break
        _loaded[path] = lib
        logger.info("Loaded native library %s", path.name)
        return lib


class NativeFeatureExtractor:
    """``FeatureExtractor`` backed by ``ProcessDataWristsQuat``.

    The native function keeps internal state across calls, so calls must be
    sequential and in window order.
    """

    def __init__(self) -> None:
        lib = _load(
            EXTRACTOR_LIB, ("ProcessDataWristsQuat_initialize", "ProcessDataWristsQuat_init")
        )
        self._fn = lib.ProcessDataWristsQuat
        self._fn.argtypes = [_Float150] * 20 + [_Float18]
        self._fn.restype = None

    def extract(self, window: NDArray[np.float32]) -> NDArray[np.float32]:
        if window.shape != (WINDOW, 20):
            raise ValueError(f"expected window shape ({WINDOW}, 20), got {window.shape}")
        # The C signature takes non-const inputs; pass private copies so the caller's
        # window is never mutated.
        columns = [
            np.array(window[:, c], dtype=np.float32, copy=True) for c in _NATIVE_COLUMN_ORDER
        ]
        features = np.empty(FEATURES, dtype=np.float32)
        self._fn(*columns, features)
        return features


class NativeClassifier:
    """``Classifier`` backed by ``Predict_Pericolosa_Wrists_Quat``."""

    def __init__(self) -> None:
        lib = _load(CLASSIFIER_LIB, ("c_Predict_Pericolosa_Wrists_Qua",))
        self._fn = lib.Predict_Pericolosa_Wrists_Quat
        self._fn.argtypes = [_Float18]
        self._fn.restype = ctypes.c_ubyte

    def classify(self, features: NDArray[np.float32]) -> Tag:
        vector = np.ascontiguousarray(features, dtype=np.float32).reshape(-1)
        if vector.size != FEATURES:
            raise ValueError(f"expected {FEATURES} features, got {vector.size}")
        code = int(self._fn(vector))
        try:
            return Tag(code)
        except ValueError:
            raise ClassificationError(f"classifier returned unknown code {code}") from None
