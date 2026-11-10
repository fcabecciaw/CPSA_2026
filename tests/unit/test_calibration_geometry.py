"""Calibration model, projection, depth conversion and depth lifting."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cpsa.vision.calibration import (
    Calibration,
    DepthModel,
    Extrinsics,
    Intrinsics,
    _keypoint_roi_mask,
    _lift_keypoints_core,
    deproject,
    depth_to_m,
    lift_keypoints,
    load_calibration,
    project,
    project_jacobian,
)

FX = 600.0
FY = 590.0
CX = 320.0
CY = 240.0
WIDTH = 640
HEIGHT = 480

DIST = (-0.25, 0.08, 0.0012, -0.0008, 0.01)


def _intrinsics(dist: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)):
    return Intrinsics(
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        width=WIDTH,
        height=HEIGHT,
        dist=dist,
    )


def test_project_deproject_round_trip_without_distortion() -> None:
    intrinsics = _intrinsics()
    rng = np.random.default_rng(3)
    pixels = np.column_stack(
        (rng.uniform(0.0, WIDTH - 1.0, 500), rng.uniform(0.0, HEIGHT - 1.0, 500))
    )
    depths = rng.uniform(0.4, 2.0, 500)

    points = deproject(pixels, depths, intrinsics)
    reprojected = project(points, intrinsics)

    assert np.max(np.abs(reprojected - pixels)) < 1e-6


def test_project_deproject_round_trip_with_distortion() -> None:
    intrinsics = _intrinsics(DIST)
    rng = np.random.default_rng(4)
    pixels = np.column_stack((rng.uniform(40.0, 600.0, 500), rng.uniform(30.0, 450.0, 500)))
    depths = rng.uniform(0.4, 1.5, 500)

    points = deproject(pixels, depths, intrinsics)
    reprojected = project(points, intrinsics)
    residual = np.max(np.abs(reprojected - pixels))

    # Contract: below 0.05 px. Newton inversion gives a much smaller residual.
    assert residual < 0.05


def test_project_matches_pinhole_reference() -> None:
    intrinsics = _intrinsics()
    point = np.array([0.1, -0.05, 0.5])
    pixel = project(point, intrinsics)
    np.testing.assert_allclose(pixel, [CX + FX * 0.2, CY - FY * 0.1], atol=1e-12)


def test_project_jacobian_matches_finite_differences() -> None:
    intrinsics = _intrinsics(DIST)
    point = np.array([0.12, -0.08, 0.7])
    jacobian = project_jacobian(point, intrinsics)

    epsilon = 1e-6
    numeric = np.empty((2, 3))
    for axis in range(3):
        plus = point.copy()
        minus = point.copy()
        plus[axis] += epsilon
        minus[axis] -= epsilon
        numeric[:, axis] = (project(plus, intrinsics) - project(minus, intrinsics)) / (
            2.0 * epsilon
        )

    np.testing.assert_allclose(jacobian, numeric, rtol=1e-5, atol=1e-5)


def test_depth_to_m_converts_and_marks_zero_invalid() -> None:
    raw = np.array([[1000, 0, 2000]], dtype=np.uint16)
    depth = depth_to_m(raw, DepthModel(scale_m=0.00003125, offset_m=0.001))

    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(0.03225, abs=1e-7)
    assert np.isnan(depth[0, 1])
    assert depth[0, 2] == pytest.approx(0.0635, abs=1e-7)


def _schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rgb": {
            "w": WIDTH,
            "h": HEIGHT,
            "K": [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]],
            "dist": list(DIST),
        },
        "ir": {
            "w": WIDTH,
            "h": HEIGHT,
            "K": [[610.0, 0.0, 318.0], [0.0, 611.0, 242.0], [0.0, 0.0, 1.0]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "T_ir_rgb": {
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t_m": [0.06, -0.002, 0.001],
        },
        "depth": {"scale_m": 0.00003125, "offset_m": 0.0},
        "rms": {"rgb_px": 0.2, "ir_px": 0.2, "stereo_px": 0.3, "depth_mm": 2.0},
        "board": {
            "squares": [11, 8],
            "square_mm": 30.0,
            "marker_mm": 21.0,
            "dictionary": "DICT_5X5_100",
        },
    }


def _write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_calibration_reads_v1_fields(tmp_path: Path) -> None:
    calibration = load_calibration(_write(tmp_path / "calibration.json", _schema()))

    assert isinstance(calibration, Calibration)
    assert calibration.rgb.fx == FX
    assert calibration.rgb.dist == DIST
    assert calibration.ir.width == WIDTH
    np.testing.assert_allclose(calibration.T_ir_rgb.R, np.eye(3))
    np.testing.assert_allclose(calibration.T_ir_rgb.t_m, [0.06, -0.002, 0.001])
    assert calibration.depth.scale_m == 0.00003125


def _corrupted(case: str) -> dict[str, Any]:
    data = copy.deepcopy(_schema())
    if case == "missing_rgb":
        del data["rgb"]
    elif case == "missing_rgb_k":
        del data["rgb"]["K"]
    elif case == "bad_dist_length":
        data["rgb"]["dist"] = [0.0, 0.0, 0.0, 0.0]
    elif case == "bad_k_skew":
        data["rgb"]["K"][0][1] = 5.0
    elif case == "bad_scale":
        data["depth"]["scale_m"] = 0.0
    elif case == "bad_version":
        data["schema_version"] = 2
    elif case == "bool_version":
        data["schema_version"] = True
    elif case == "string_version":
        data["schema_version"] = "1"
    elif case == "float_version":
        data["schema_version"] = 1.0
    elif case == "bad_translation":
        data["T_ir_rgb"]["t_m"] = [0.0, 0.0]
    elif case == "nonfinite_translation":
        data["T_ir_rgb"]["t_m"][2] = float("inf")
    elif case == "bad_width":
        data["ir"]["w"] = "640"
    elif case == "missing_extrinsics":
        del data["T_ir_rgb"]
    elif case == "zero_fx":
        data["rgb"]["K"][0][0] = 0.0
    elif case == "negative_fy":
        data["ir"]["K"][1][1] = -611.0
    elif case == "nonfinite_cx":
        data["rgb"]["K"][0][2] = float("nan")
    elif case == "nonfinite_cy":
        data["ir"]["K"][1][2] = float("inf")
    elif case == "bad_rotation":
        data["T_ir_rgb"]["R"][0][0] = 1.2
    elif case == "reflection":
        data["T_ir_rgb"]["R"] = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    elif case == "missing_rms":
        del data["rms"]
    elif case == "missing_rms_field":
        del data["rms"]["depth_mm"]
    elif case == "bad_rms_type":
        data["rms"]["rgb_px"] = "0.2"
    elif case == "missing_board":
        del data["board"]
    elif case == "missing_board_field":
        del data["board"]["marker_mm"]
    elif case == "bad_board_squares":
        data["board"]["squares"] = [11]
    elif case == "bad_board_square_mm":
        data["board"]["square_mm"] = "30"
    elif case == "bad_board_dictionary":
        data["board"]["dictionary"] = 5
    else:  # pragma: no cover - guards the parametrisation
        raise AssertionError(case)
    return data


@pytest.mark.parametrize(
    ("case", "key"),
    [
        ("missing_rgb", "rgb"),
        ("missing_rgb_k", "rgb.K"),
        ("bad_dist_length", "rgb.dist"),
        ("bad_k_skew", "rgb.K"),
        ("bad_scale", "depth.scale_m"),
        ("bad_version", "schema_version"),
        ("bool_version", "schema_version"),
        ("string_version", "schema_version"),
        ("float_version", "schema_version"),
        ("bad_translation", "T_ir_rgb.t_m"),
        ("nonfinite_translation", "T_ir_rgb.t_m"),
        ("bad_width", "ir.w"),
        ("missing_extrinsics", "T_ir_rgb"),
        ("zero_fx", "rgb.K"),
        ("negative_fy", "ir.K"),
        ("nonfinite_cx", "rgb.K"),
        ("nonfinite_cy", "ir.K"),
        ("bad_rotation", "T_ir_rgb.R"),
        ("reflection", "T_ir_rgb.R"),
        ("missing_rms", "rms"),
        ("missing_rms_field", "rms.depth_mm"),
        ("bad_rms_type", "rms.rgb_px"),
        ("missing_board", "board"),
        ("missing_board_field", "board.marker_mm"),
        ("bad_board_squares", "board.squares"),
        ("bad_board_square_mm", "board.square_mm"),
        ("bad_board_dictionary", "board.dictionary"),
    ],
)
def test_loader_rejects_invalid_key_and_names_it(tmp_path: Path, case: str, key: str) -> None:
    path = _write(tmp_path / "calibration.json", _corrupted(case))
    with pytest.raises(ValueError, match=re.escape(key)):
        load_calibration(path)


def _lifting_calibration() -> Calibration:
    intrinsics = Intrinsics(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        width=WIDTH,
        height=HEIGHT,
    )
    return Calibration(
        rgb=intrinsics,
        ir=intrinsics,
        T_ir_rgb=Extrinsics(R=np.eye(3), t_m=np.array([0.06, 0.0, 0.0])),
        depth=DepthModel(scale_m=1.0 / 32000.0, offset_m=0.0),
    )


def test_lift_keypoints_registers_ir_depth_into_rgb_frame() -> None:
    calibration = _lifting_calibration()
    depth_raw = np.full((HEIGHT, WIDTH), 32000, dtype=np.uint16)

    # IR pixel (320, 240) is the optical axis at 1 m; after the 6 cm baseline
    # shift it projects to u = 356 in RGB.
    lifted = lift_keypoints(np.array([[356.0, 240.0]]), depth_raw, calibration)

    np.testing.assert_allclose(lifted[0], [0.06, 0.0, 1.0], atol=1e-5)


def test_lift_keypoints_uses_off_axis_ray() -> None:
    calibration = _lifting_calibration()
    depth_raw = np.full((HEIGHT, WIDTH), 32000, dtype=np.uint16)

    point_ir = deproject(np.array([[100.0, 100.0]]), np.array([1.0]), calibration.ir)[0]
    point_rgb = point_ir + calibration.T_ir_rgb.t_m
    pixel_rgb = project(point_rgb, calibration.rgb)

    lifted = lift_keypoints(pixel_rgb[None, :], depth_raw, calibration)

    np.testing.assert_allclose(lifted[0], point_rgb, atol=1e-5)


def test_lift_keypoints_returns_nan_without_valid_depth() -> None:
    calibration = _lifting_calibration()
    depth_raw = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)

    lifted = lift_keypoints(np.array([[356.0, 240.0], [100.0, 100.0]]), depth_raw, calibration)

    assert lifted.shape == (2, 3)
    assert np.all(np.isnan(lifted))


def test_roi_registration_falls_back_when_geometry_is_uncertain() -> None:
    calibration = _lifting_calibration()
    invalid = Calibration(
        rgb=calibration.rgb,
        ir=calibration.ir,
        T_ir_rgb=Extrinsics(R=2.0 * np.eye(3), t_m=calibration.T_ir_rgb.t_m),
        depth=calibration.depth,
    )

    assert _keypoint_roi_mask(np.array([[356.0, 240.0]]), invalid, 5, 0.2, 1.0) is None


def test_lift_keypoints_requires_a_positive_depth_floor() -> None:
    calibration = _lifting_calibration()
    with pytest.raises(ValueError, match="z_min"):
        lift_keypoints(
            np.array([[356.0, 240.0]]),
            np.full((HEIGHT, WIDTH), 32000, dtype=np.uint16),
            calibration,
            z_min=0.0,
        )


def test_lift_keypoints_requires_ir_depth_shape() -> None:
    calibration = _lifting_calibration()
    with pytest.raises(ValueError, match="depth image shape"):
        lift_keypoints(
            np.array([[356.0, 240.0]]),
            np.zeros((10, 10), dtype=np.uint16),
            calibration,
        )


def _rigid_lifting_calibration() -> Calibration:
    """Unequal intrinsics, non-identity rotation and a z translation."""
    dist = (-0.04, 0.01, 0.0008, -0.0005, 0.0)
    rgb = Intrinsics(
        fx=600.0,
        fy=590.0,
        cx=320.0,
        cy=240.0,
        width=WIDTH,
        height=HEIGHT,
        dist=dist,
    )
    ir = Intrinsics(
        fx=612.0,
        fy=604.0,
        cx=317.0,
        cy=243.0,
        width=WIDTH,
        height=HEIGHT,
        dist=dist,
    )
    yaw = np.deg2rad(2.0)
    pitch = np.deg2rad(-1.5)
    rotation_y = np.array(
        [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
    )
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    return Calibration(
        rgb=rgb,
        ir=ir,
        T_ir_rgb=Extrinsics(R=rotation_x @ rotation_y, t_m=np.array([0.055, -0.004, 0.025])),
        depth=DepthModel(scale_m=1.0 / 32000.0, offset_m=0.0),
    )


def _raw_depth(depth_m: float) -> int:
    return int(round(depth_m * 32000.0))


def test_lift_keypoints_sparse_scene_requires_registration() -> None:
    calibration = _rigid_lifting_calibration()
    depth_raw = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    ir_pixel = (300.0, 240.0)
    depth_ir = 0.7
    depth_raw[240, 300] = _raw_depth(depth_ir)

    point_ir = deproject(np.array([ir_pixel]), np.array([depth_ir]), calibration.ir)[0]
    point_rgb = calibration.T_ir_rgb.R @ point_ir + calibration.T_ir_rgb.t_m
    pixel_rgb = project(point_rgb, calibration.rgb)

    # Without registration the RGB keypoint would look at an empty IR pixel.
    assert np.hypot(*(pixel_rgb - np.array(ir_pixel))) > 20.0
    lifted = lift_keypoints(pixel_rgb[None, :], depth_raw, calibration, window=2)

    assert np.all(np.isfinite(lifted[0]))
    np.testing.assert_allclose(lifted[0], point_rgb, atol=1e-6)
    assert lifted[0, 2] == pytest.approx(point_rgb[2], abs=1e-6)


def test_lift_keypoints_median_window_combines_registered_depths() -> None:
    calibration = _rigid_lifting_calibration()
    depth_raw = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    keypoint = np.array([350.0, 240.0])
    registered_depths = []
    ir_pixels = []
    for depth_rgb in (0.55, 0.70, 0.85, 1.00):
        point_rgb = deproject(keypoint[None, :], np.array([depth_rgb]), calibration.rgb)[0]
        point_ir = calibration.T_ir_rgb.R.T @ (point_rgb - calibration.T_ir_rgb.t_m)
        pixel_ir = project(point_ir[None, :], calibration.ir)[0]
        u_ir, v_ir = int(round(pixel_ir[0])), int(round(pixel_ir[1]))
        depth_raw[v_ir, u_ir] = _raw_depth(depth_rgb)
        actual_ir = deproject(np.array([[u_ir, v_ir]]), np.array([depth_rgb]), calibration.ir)[0]
        actual_rgb = calibration.T_ir_rgb.R @ actual_ir + calibration.T_ir_rgb.t_m
        pixel_rgb = project(actual_rgb, calibration.rgb)
        assert np.max(np.abs(pixel_rgb - keypoint)) <= 4.0
        registered_depths.append(float(actual_rgb[2]))
        ir_pixels.append(np.array([u_ir, v_ir], dtype=np.float64))
    expected_depth = float(np.median(registered_depths))
    expected_point = deproject(keypoint[None, :], np.array([expected_depth]), calibration.rgb)[0]

    assert len({(int(pixel[0]), int(pixel[1])) for pixel in ir_pixels}) == len(ir_pixels)
    # The IR samples are more than one window away from the RGB keypoint, so
    # skipping registration would leave no depth in the window.
    assert np.min([np.hypot(*(pixel - keypoint)) for pixel in ir_pixels]) > 20.0
    lifted = lift_keypoints(keypoint[None, :], depth_raw, calibration, window=4)

    assert np.all(np.isfinite(lifted[0]))
    np.testing.assert_allclose(lifted[0], expected_point, atol=1e-6)
    assert lifted[0, 2] == pytest.approx(expected_depth, abs=1e-6)


def test_roi_registration_matches_full_registration_on_depth_discontinuities() -> None:
    calibration = _rigid_lifting_calibration()
    v, u = np.indices((HEIGHT, WIDTH))
    depth_m = np.full((HEIGHT, WIDTH), 0.9)
    depth_m[u < WIDTH // 3] = 0.22
    depth_m[(u >= WIDTH // 3) & (u < 2 * WIDTH // 3)] = 0.5
    depth_m[(u > 180) & (u < 260) & (v > 100) & (v < 180)] = 0.3
    depth_raw = np.round(depth_m * 32000.0).astype(np.uint16)
    depth_raw[::11, ::13] = 0
    rng = np.random.default_rng(23)
    keypoints = rng.uniform([200.0, 110.0], [440.0, 350.0], (62, 2))
    depth_ir = depth_to_m(depth_raw, calibration.depth)

    full = _lift_keypoints_core(keypoints, depth_ir, calibration, 5, 0.2, None)
    roi = lift_keypoints(keypoints, depth_raw, calibration)

    np.testing.assert_array_equal(np.isnan(full), np.isnan(roi))
    np.testing.assert_allclose(full, roi, rtol=0.0, atol=0.0, equal_nan=True)
    valid = np.isfinite(depth_ir) & (depth_ir >= 0.2)
    mask = _keypoint_roi_mask(
        keypoints,
        calibration,
        5,
        max(0.2, float(np.min(depth_ir[valid]))),
        float(np.max(depth_ir[valid])),
    )
    assert mask is not None
    assert float(np.mean(mask)) < 0.5
