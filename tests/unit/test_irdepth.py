"""INZI decode and IR model input: golden buffer, constants and preprocessing (F-A)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cpsa.adapters.camera.irdepth import ir_to_input
from cpsa.vision.irdepth import (
    CLAHE_CLIP,
    CLAHE_TILES,
    INZI_BYTES_PER_PIXEL,
    decode_inzi,
    normalize_ir,
)

GOLDEN_DEPTH = np.array([[0x1234, 0x0000], [0xABCD, 0x00FF]], dtype=np.uint16)
GOLDEN_IR = np.array([[10, 20], [30, 40]], dtype=np.uint8)


def inzi_buffer(depth: np.ndarray, ir: np.ndarray) -> bytes:
    """Interleave depth and IR the way the F200 ``INZI`` node does."""

    height, width = depth.shape
    pixels = np.empty((height, width, INZI_BYTES_PER_PIXEL), dtype=np.uint8)
    pixels[:, :, 0] = (depth & 0xFF).astype(np.uint8)
    pixels[:, :, 1] = (depth >> 8).astype(np.uint8)
    pixels[:, :, 2] = ir
    return pixels.tobytes()


def test_golden_buffer_decodes_into_ir_and_little_endian_depth() -> None:
    ir, depth = decode_inzi(inzi_buffer(GOLDEN_DEPTH, GOLDEN_IR), width=2, height=2)

    assert ir.dtype == np.uint8
    assert depth.dtype == np.uint16
    assert ir.shape == (2, 2) and depth.shape == (2, 2)
    np.testing.assert_array_equal(ir, GOLDEN_IR)
    np.testing.assert_array_equal(depth, GOLDEN_DEPTH)


def test_decoded_arrays_do_not_alias_the_source_buffer() -> None:
    buffer = bytearray(inzi_buffer(GOLDEN_DEPTH, GOLDEN_IR))
    ir, depth = decode_inzi(buffer, width=2, height=2)

    buffer[0] = 0xFF
    buffer[2] = 0xFF
    assert depth[0, 0] == 0x1234
    assert ir[0, 0] == 10


def test_buffer_size_must_be_width_times_height_times_three() -> None:
    with pytest.raises(ValueError, match="expected 12 INZI bytes, got 11"):
        decode_inzi(bytes(11), width=2, height=2)
    with pytest.raises(ValueError, match="invalid INZI size"):
        decode_inzi(bytes(3), width=0, height=1)


def test_clahe_constants_are_the_training_ones() -> None:
    assert CLAHE_CLIP == 3.0
    assert CLAHE_TILES == (8, 8)


def test_ir_input_is_three_equal_channels_of_uint8() -> None:
    ir = np.arange(48, dtype=np.uint8).reshape(6, 8)

    model_input = ir_to_input(ir)

    assert model_input.shape == (6, 8, 3)
    assert model_input.dtype == np.uint8
    np.testing.assert_array_equal(model_input[:, :, 0], model_input[:, :, 1])
    np.testing.assert_array_equal(model_input[:, :, 1], model_input[:, :, 2])


def test_ir_input_matches_min_max_normalisation_then_clahe() -> None:
    ir = np.array([[100, 101, 104, 108], [110, 112, 113, 114]], dtype=np.uint8)

    expected = ir.astype(np.float32) - ir.min()
    expected *= np.float32(255.0 / (float(ir.max()) - float(ir.min())))
    normalised = np.rint(expected).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
    enhanced = clahe.apply(normalised)

    np.testing.assert_array_equal(ir_to_input(ir)[:, :, 0], enhanced)


def test_flat_ir_normalises_to_zeros_before_clahe() -> None:
    model_input = ir_to_input(np.full((4, 5), 77, dtype=np.uint8))

    assert model_input.shape == (4, 5, 3)
    # Min-max normalisation of a flat frame is all zeros; OpenCV's CLAHE maps
    # that constant frame to one constant level, so the input stays featureless.
    assert len(np.unique(model_input)) == 1


def test_ir_input_rejects_a_non_image() -> None:
    with pytest.raises(ValueError, match="ir must be"):
        ir_to_input(np.zeros((4, 5, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="ir must be uint8"):
        ir_to_input(np.zeros((4, 5), dtype=np.float32))


def test_normalize_ir_stretches_the_range_and_flattens_a_constant_frame() -> None:
    normalised = normalize_ir(np.array([[100, 110], [120, 130]], dtype=np.uint8))
    np.testing.assert_array_equal(normalised, np.array([[0, 85], [170, 255]], dtype=np.uint8))

    flat = normalize_ir(np.full((3, 4), 77, dtype=np.uint8))
    assert flat.shape == (3, 4)
    assert flat.dtype == np.uint8
    np.testing.assert_array_equal(flat, np.zeros((3, 4), dtype=np.uint8))


def test_normalize_ir_rejects_a_non_image() -> None:
    with pytest.raises(ValueError, match="ir must be"):
        normalize_ir(np.zeros((4, 5, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="ir must be uint8"):
        normalize_ir(np.zeros((4, 5), dtype=np.float32))
