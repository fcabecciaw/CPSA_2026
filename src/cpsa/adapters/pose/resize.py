"""OpenCV letterbox resize for the pose adapters (NFR-003).

Ultralytics letterboxes its training and validation images with
``cv2.resize(..., interpolation=cv2.INTER_LINEAR)``, so running the same call at
inference time both matches training and costs a fraction of the pure-numpy
fallback: about 0.4 ms against 17.5 ms per 640x480 -> 512x384 frame on an x86
host, and the A53 gap is wider still. ``opencv-python-headless`` is part of the
``vision`` extra (``uv sync --extra vision``); without it the adapters pass
``None`` and :func:`cpsa.vision.pose_decode.letterbox` keeps its NumPy path.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from cpsa.vision.pose_decode import ResizeCallable

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the installed extra
    cv2 = None  # type: ignore[assignment]

__all__ = ["LETTERBOX_RESIZE", "opencv_resize"]


def opencv_resize(
    image: NDArray[np.uint8],
    out_height: int,
    out_width: int,
) -> NDArray[np.uint8]:
    """Bilinear-resize an ``(H, W, 3)`` uint8 image with OpenCV.

    This is the interpolation Ultralytics uses for letterboxing, so the pixels
    match the training pipeline.

    Raises:
        RuntimeError: If ``opencv-python-headless`` is not installed.
    """

    if cv2 is None:  # pragma: no cover - depends on the installed extra
        raise RuntimeError(
            "OpenCV is required for the pose preprocess fast path; "
            "install the 'vision' extra (uv sync --extra vision)"
        )
    resized = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.uint8)


LETTERBOX_RESIZE: Final[ResizeCallable | None] = None if cv2 is None else opencv_resize
"""The resize the pose adapters inject into ``letterbox``; ``None`` means NumPy."""
