"""IR/depth preprocessing for the hand demos (F-A).

One ``INZI`` frame of the F200 is 640x480x3 bytes, interleaved per pixel:
bytes 0-1 are the little-endian Z16 depth sample and byte 2 is the 8-bit IR
sample. Depth and IR are pixel-aligned, so one buffer carries both images.

The hand model runs on IR, which has low contrast on skin; :func:`ir_to_input`
stretches the frame to the full 8-bit range and applies the CLAHE that the
training pipeline also uses, then replicates it to the three channels the pose
model expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the installed extra
    cv2 = None  # type: ignore[assignment]

__all__ = [
    "CLAHE_CLIP",
    "CLAHE_TILES",
    "INZI_BYTES_PER_PIXEL",
    "IrDepthFrame",
    "decode_inzi",
    "ir_to_input",
]

INZI_BYTES_PER_PIXEL = 3
"""Bytes per pixel of one interleaved ``INZI`` frame (Z16 depth + 8-bit IR)."""

CLAHE_CLIP = 3.0
"""Clip limit of the contrast-limited adaptive equalisation."""

CLAHE_TILES = (8, 8)
"""Tile grid of the contrast-limited adaptive equalisation."""

_DEPTH_BYTES = 2
_MAX_UINT8 = 255.0


@dataclass(frozen=True)
class IrDepthFrame:
    """One decoded IR/depth frame, in the IR camera frame.

    ``ir`` is ``(H, W)`` uint8, ``depth_raw`` is ``(H, W)`` uint16 Z16 units;
    zero depth means "no measurement". ``t`` is ``CLOCK_MONOTONIC`` seconds and
    ``decode_ms`` the time the source spent decoding the INZI buffer.
    """

    seq: int
    t: float
    width: int
    height: int
    ir: NDArray[np.uint8]
    depth_raw: NDArray[np.uint16]
    decode_ms: float = 0.0


def decode_inzi(
    data: bytes, width: int, height: int
) -> tuple[NDArray[np.uint8], NDArray[np.uint16]]:
    """Split one ``INZI`` buffer into ``(ir[H, W], depth_raw[H, W])``.

    Args:
        data: The raw frame buffer, ``width * height * 3`` bytes.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Raises:
        ValueError: If a size is not positive or ``len(data)`` is not
            ``width * height * 3``.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"invalid INZI size {width}x{height}")
    expected = width * height * INZI_BYTES_PER_PIXEL
    if len(data) != expected:
        raise ValueError(f"expected {expected} INZI bytes, got {len(data)}")
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, INZI_BYTES_PER_PIXEL)
    depth_bytes = np.ascontiguousarray(pixels[:, :, :_DEPTH_BYTES])
    depth_raw = depth_bytes.view("<u2").reshape(height, width)
    ir = pixels[:, :, _DEPTH_BYTES].copy()
    return ir, depth_raw


def ir_to_input(ir: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Turn one ``(H, W)`` IR image into the ``(H, W, 3)`` model input.

    The image is min-max normalised to the full 8-bit range (a flat image maps
    to zeros), contrast-limited adaptive equalisation is applied with
    :data:`CLAHE_CLIP` and :data:`CLAHE_TILES`, and the result is replicated to
    three channels.

    Raises:
        ValueError: If ``ir`` is not a 2-D uint8 array.
        RuntimeError: If OpenCV is not installed.
    """

    image = np.asarray(ir)
    if image.ndim != 2:
        raise ValueError(f"ir must be (H, W), got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"ir must be uint8, got {image.dtype}")
    if cv2 is None:  # pragma: no cover - depends on the installed extra
        raise RuntimeError(
            "OpenCV is required for the hand IR preprocessing; "
            "install the 'vision' extra (uv sync --extra vision)"
        )

    low = float(np.min(image))
    high = float(np.max(image))
    if high > low:
        scaled = (image.astype(np.float32) - np.float32(low)) * np.float32(
            _MAX_UINT8 / (high - low)
        )
        normalised = np.rint(scaled).clip(0.0, _MAX_UINT8).astype(np.uint8)
    else:
        normalised = np.zeros_like(image)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
    enhanced = np.asarray(clahe.apply(normalised), dtype=np.uint8)
    return np.repeat(enhanced[:, :, np.newaxis], 3, axis=2)
