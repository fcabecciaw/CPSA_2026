"""Model build and activation tests for the DPU-compatible pose graphs."""

from __future__ import annotations

import pytest
import torch
import yaml

from cpsa_ml.models import build_model, config_path, load_yolo, model_yaml_path


def test_hswish_config_uses_hardswish_and_no_attention() -> None:
    path = model_yaml_path("n", "hswish")
    config = yaml.safe_load(path.read_text())
    assert config["activation"] == "nn.Hardswish()"
    assert config["kpt_shape"] == [17, 3]
    modules = [layer[2].lower() for layer in config["backbone"] + config["head"]]
    banned = {"attention", "multiheadattention", "attn", "psa", "c2psa", "c2fpsa", "c2fattn"}
    assert not any(name in banned or name.endswith("attn") for name in modules)


def test_s_variant_config_exists() -> None:
    path = model_yaml_path("s", "hswish")
    assert path.is_file()
    assert config_path(path.name) == path
    assert yaml.safe_load(path.read_text())["activation"] == "nn.Hardswish()"


def test_build_s_variant_uses_hardswish() -> None:
    model = build_model("coco17", variant="s", activation="hswish")
    assert tuple(model.model[-1].kpt_shape) == (17, 3)
    assert not any(isinstance(module, torch.nn.SiLU) for module in model.modules())
    assert any(isinstance(module, torch.nn.Hardswish) for module in model.modules())


def test_build_coco17_hswish_forward() -> None:
    model = build_model("coco17", variant="n", activation="hswish").eval()
    assert tuple(model.model[-1].kpt_shape) == (17, 3)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 480, 640))
    predictions = output[0] if isinstance(output, tuple) else output
    assert predictions.shape[0] == 1
    assert predictions.shape[1] == 4 + 1 + 17 * 3
    assert not any(isinstance(module, torch.nn.SiLU) for module in model.modules())
    assert any(isinstance(module, torch.nn.Hardswish) for module in model.modules())


def test_build_halfbust62_hswish_forward() -> None:
    model = build_model("halfbust62", variant="n", activation="hswish").eval()
    assert tuple(model.model[-1].kpt_shape) == (62, 3)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 480, 640))
    predictions = output[0] if isinstance(output, tuple) else output
    assert predictions.shape[0] == 1
    assert predictions.shape[1] == 4 + 1 + 62 * 3
    assert not any(isinstance(module, torch.nn.SiLU) for module in model.modules())


def test_silu_control_has_silu() -> None:
    model = build_model("coco17", variant="n", activation="silu")
    assert any(isinstance(module, torch.nn.SiLU) for module in model.modules())


@pytest.mark.parametrize("variant", ["n", "s"])
def test_load_yolo_wrapper_builds_for_schema(variant: str) -> None:
    yolo = load_yolo("coco17", variant=variant, activation="hswish")
    assert yolo.task == "pose"
    assert tuple(yolo.model.model[-1].kpt_shape) == (17, 3)
