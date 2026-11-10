"""Optional Brevitas QAT tests (skipped when the qat extra is absent)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("brevitas")

from brevitas.nn import QuantConv2d, QuantHardSwish  # noqa: E402
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint  # noqa: E402

from cpsa_ml.models import build_model  # noqa: E402
from cpsa_ml.qat import NndctProbe, brevitas_version, quantize_model, try_nndct  # noqa: E402


def test_fixed_point_quantizer_names_exist() -> None:
    assert Int8WeightPerTensorFixedPoint is not None
    assert Int8ActPerTensorFixedPoint is not None
    assert brevitas_version() is not None


def test_quantize_tiny_conv_net() -> None:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.Hardswish(),
        torch.nn.Conv2d(8, 4, 1),
    ).eval()
    original = model[0].weight.detach().clone()
    quantized = quantize_model(model)
    assert isinstance(quantized[0], QuantConv2d)
    assert isinstance(quantized[2], QuantHardSwish)
    assert isinstance(quantized[3], QuantConv2d)
    assert torch.allclose(quantized[0].weight.detach(), original)
    with torch.no_grad():
        output = quantized(torch.randn(2, 3, 16, 16))
    assert output.shape == (2, 4, 16, 16)


def _unquantized_hardswish(module: torch.nn.Module):
    for child in module.children():
        if isinstance(child, QuantHardSwish):
            continue
        if isinstance(child, torch.nn.Hardswish):
            yield child
        yield from _unquantized_hardswish(child)


def test_quantize_hswish_pose_model() -> None:
    model = build_model("halfbust62", variant="n", activation="hswish").eval()
    quantized = quantize_model(model)
    assert any(isinstance(module, QuantConv2d) for module in quantized.modules())
    assert any(isinstance(module, QuantHardSwish) for module in quantized.modules())
    assert not list(_unquantized_hardswish(quantized))
    with torch.no_grad():
        output = quantized(torch.randn(1, 3, 64, 64))
    predictions = output[0] if isinstance(output, tuple) else output
    assert predictions.shape[0] == 1


def test_try_nndct_returns_probe() -> None:
    probe = try_nndct()
    assert isinstance(probe, NndctProbe)
    assert bool(probe) == probe.available
    assert probe.reason
