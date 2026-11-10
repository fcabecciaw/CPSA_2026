"""ONNX export and DPU operator-audit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cpsa_ml.models import load_yolo
from cpsa_ml.onnx_check import check_onnx, export_onnx, format_report


@pytest.fixture(scope="module")
def exported_models(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("onnx")
    hswish = export_onnx(
        load_yolo("coco17", variant="n", activation="hswish"),
        root / "hswish.onnx",
        imgsz=(480, 640),
    )
    silu = export_onnx(
        load_yolo("coco17", variant="n", activation="silu"),
        root / "silu.onnx",
        imgsz=(480, 640),
    )
    return hswish, silu


def test_hswish_onnx_passes_backbone_check(exported_models: tuple[Path, Path]) -> None:
    hswish, _ = exported_models
    report = check_onnx(hswish, imgsz=(480, 640), opset=17)
    assert report.passed, format_report(report)
    assert report.input_shape == (1, 3, 480, 640)
    assert report.opset == 17
    assert "Sigmoid" not in report.backbone_ops
    assert any(op in report.backbone_ops for op in ("HardSigmoid", "HardSwish"))
    assert "Conv" in report.backbone_ops
    assert report.postprocess_ops


def test_stock_silu_onnx_is_rejected(exported_models: tuple[Path, Path]) -> None:
    _, silu = exported_models
    report = check_onnx(silu, imgsz=(480, 640), opset=17)
    assert not report.passed
    assert "Sigmoid" in report.backbone_ops
    assert any("SiLU" in violation or "Sigmoid" in violation for violation in report.violations)
    assert "FAIL" in format_report(report)


def test_wrong_input_shape_is_rejected(exported_models: tuple[Path, Path]) -> None:
    hswish, _ = exported_models
    report = check_onnx(hswish, imgsz=(640, 640), opset=17)
    assert not report.passed
    assert any("input shape" in violation for violation in report.violations)
