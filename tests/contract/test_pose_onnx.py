"""ONNX pose-estimator contract test.

Set ``CPSA_POSE_ONNX`` to an exported YOLOv8-pose ``.onnx`` file to run it.
Optional environment variables:

* ``CPSA_POSE_LAYOUT`` -- layout name (default ``coco17``);
* ``CPSA_POSE_INPUT`` -- model input as ``height,width`` (default ``480,640``).

Without the model file the test skips with a reason, so the routine suite needs
no model and no accelerator.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

MODEL_ENV = "CPSA_POSE_ONNX"


def _model_path() -> Path:
    value = os.environ.get(MODEL_ENV)
    if not value:
        pytest.skip(f"set {MODEL_ENV} to an exported YOLOv8-pose .onnx file")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"{MODEL_ENV} does not point to a file: {path}")
    return path


def _input_size() -> tuple[int, int]:
    value = os.environ.get("CPSA_POSE_INPUT", "480,640")
    try:
        height, width = (int(part) for part in value.replace("x", ",").split(","))
    except ValueError as exc:
        raise AssertionError(f"CPSA_POSE_INPUT must be height,width, got {value!r}") from exc
    return height, width


def test_onnx_pose_estimator_runs_deterministically() -> None:
    pytest.importorskip("onnxruntime", reason="install the 'vision' extra to run the ONNX contract")
    from cpsa.adapters.pose.onnx import OnnxPoseEstimator
    from cpsa.vision.layouts import get_layout

    layout = get_layout(os.environ.get("CPSA_POSE_LAYOUT", "coco17"))
    estimator = OnnxPoseEstimator(
        _model_path(),
        layout,
        input_size=_input_size(),
        conf_thr=0.25,
        iou_thr=0.45,
    )
    rng = np.random.default_rng(20261109)
    image = rng.integers(0, 256, size=(*estimator.input_size, 3), dtype=np.uint8)

    first = estimator.estimate(image)
    second = estimator.estimate(image)

    assert isinstance(first, list)
    assert len(first) == len(second)
    for person, repeated in zip(first, second, strict=True):
        assert person.box.shape == (4,)
        assert person.xy.shape == (len(layout), 2)
        assert person.conf.shape == (len(layout),)
        assert np.isfinite(person.box).all()
        assert np.isfinite(person.xy).all()
        assert np.isfinite(person.conf).all()
        assert np.allclose(person.box, repeated.box, atol=1e-6)
        assert np.allclose(person.xy, repeated.xy, atol=1e-6)
