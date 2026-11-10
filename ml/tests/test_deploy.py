"""Raw-head deploy graph tests (DPU input/output contract)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cpsa_ml.deploy import (
    RawHeadDeploy,
    build_deploy_model,
    deploy_from_yolo,
    export_onnx,
    load_deploy_weights,
    main,
    output_names,
    random_input,
    write_spec,
)
from cpsa_ml.models import build_model, load_yolo

SMALL_INPUT = (160, 192)


@pytest.fixture(scope="module")
def pose_model() -> torch.nn.Module:
    return build_model("halfbust62", variant="n", activation="hswish")


@pytest.fixture(scope="module")
def deploy(pose_model: torch.nn.Module) -> RawHeadDeploy:
    model = deploy_from_yolo(pose_model)
    model.input_size = SMALL_INPUT
    return model


def test_spec_matches_halfbust_layout(deploy: RawHeadDeploy) -> None:
    spec = deploy.spec()
    assert spec["strides"] == [8, 16, 32]
    assert spec["channel_layout"] == {"box": 64, "class": 1, "keypoint": 186}
    assert spec["output_channels"] == 251
    assert spec["n_kpt"] == 62
    assert spec["reg_max"] == 16
    assert spec["output_names"] == ["head_stride8", "head_stride16", "head_stride32"]
    assert output_names([8, 16, 32]) == spec["output_names"]


def test_forward_shapes_follow_strides(deploy: RawHeadDeploy) -> None:
    with torch.no_grad():
        outputs = deploy(random_input(SMALL_INPUT))
    assert len(outputs) == 3
    for stride, output in zip(deploy.strides, outputs, strict=True):
        assert output.shape == (1, 251, SMALL_INPUT[0] // stride, SMALL_INPUT[1] // stride)


def test_outputs_are_the_head_convolution_outputs(
    deploy: RawHeadDeploy, pose_model: torch.nn.Module
) -> None:
    """The deploy graph must stop at the raw head convs, in [box, class, kpt] order."""

    head = pose_model.model[-1]
    captured: list[list[torch.Tensor]] = []
    handle = head.register_forward_pre_hook(lambda _module, args: captured.append(list(args[0])))
    try:
        with torch.no_grad():
            pose_model(random_input(SMALL_INPUT))
            deploy_outputs = deploy(random_input(SMALL_INPUT, seed=1))
    finally:
        handle.remove()

    features = captured[0]
    assert len(features) == head.nl
    with torch.no_grad():
        for level, (stride, output) in enumerate(zip(deploy.strides, deploy_outputs, strict=True)):
            expected = torch.cat(
                (
                    head.cv2[level](features[level]),
                    head.cv3[level](features[level]),
                    head.cv4[level](features[level]),
                ),
                dim=1,
            )
            assert output.shape == expected.shape
            torch.testing.assert_close(output, expected)
            assert output.shape[2] == SMALL_INPUT[0] // stride


def test_keypoint_channels_are_raw_logits(deploy: RawHeadDeploy) -> None:
    """Keypoint channels feed ``decode_raw_heads``: no activation inside the graph."""

    with torch.no_grad():
        outputs = deploy(random_input(SMALL_INPUT))
    keypoint = outputs[0][:, 65:]
    # Raw conv outputs are unbounded; a sigmoid would clamp them into [0, 1].
    assert keypoint.min() < 0.0 or keypoint.max() > 1.0


def test_yolo_wrapper_builds_equivalent_deploy_graph() -> None:
    yolo = load_yolo("halfbust62", variant="n", activation="hswish")
    wrapped = deploy_from_yolo(yolo)
    assert isinstance(wrapped, RawHeadDeploy)
    assert wrapped.n_kpt == 62
    assert wrapped.strides == (8, 16, 32)
    assert next(wrapped.parameters()).device.type == "cpu"


def test_state_dict_roundtrip_and_checkpoint_loading(tmp_path: Path) -> None:
    yolo = load_yolo("halfbust62", variant="n", activation="hswish")
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": yolo.model}, checkpoint)

    deploy = build_deploy_model("halfbust62", seed=7)
    deploy.input_size = SMALL_INPUT
    reference = deploy_from_yolo(yolo)
    reference.input_size = SMALL_INPUT
    with torch.no_grad():
        unrelated = deploy(random_input(SMALL_INPUT))
        expected = reference(random_input(SMALL_INPUT))
    assert not torch.allclose(unrelated[0], expected[0])

    load_deploy_weights(deploy, checkpoint)
    with torch.no_grad():
        loaded_outputs = deploy(random_input(SMALL_INPUT))
    for wanted, actual in zip(expected, loaded_outputs, strict=True):
        torch.testing.assert_close(wanted, actual)

    state_only = tmp_path / "state.pt"
    torch.save(deploy.state_dict(), state_only)
    other = build_deploy_model("halfbust62", seed=5)
    other.input_size = SMALL_INPUT
    load_deploy_weights(other, state_only)
    with torch.no_grad():
        torch.testing.assert_close(other(random_input(SMALL_INPUT))[0], loaded_outputs[0])


def test_write_spec_and_onnx_export(tmp_path: Path) -> None:
    deploy = build_deploy_model("halfbust62", seed=0)
    deploy.input_size = SMALL_INPUT
    spec_path = write_spec(deploy, tmp_path / "deploy_spec.json")
    assert json.loads(spec_path.read_text())["output_channels"] == 251

    import onnx

    onnx_path = export_onnx(deploy, tmp_path / "deploy.onnx", SMALL_INPUT)
    graph = onnx.load(str(onnx_path)).graph
    assert [value.name for value in graph.input] == ["images"]
    assert [value.name for value in graph.output] == output_names([8, 16, 32])
    first = graph.output[0].type.tensor_type.shape.dim
    assert tuple(dim.dim_value for dim in first) == (1, 251, 20, 24)


@pytest.mark.parametrize(
    ("activation", "activation_spec"),
    [
        ("relu", "nn.ReLU()"),
        ("lrelu", "nn.LeakyReLU(0.1015625)"),
    ],
)
def test_cli_exports_activation_config_at_requested_size(
    tmp_path: Path, activation: str, activation_spec: str
) -> None:
    """The speed-matrix configs drive the deploy CLI with ``--cfg --height --width``."""

    cfg = Path(__file__).resolve().parents[1] / "configs" / f"yolov8n-pose-{activation}.yaml"
    assert (
        main(
            [
                "--schema",
                "halfbust62",
                "--activation",
                activation,
                "--cfg",
                str(cfg),
                "--height",
                str(SMALL_INPUT[0]),
                "--width",
                str(SMALL_INPUT[1]),
                "--seed",
                "0",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )

    spec = json.loads((tmp_path / "deploy_spec.json").read_text())
    assert spec["input_size"] == list(SMALL_INPUT)
    assert spec["output_names"] == output_names([8, 16, 32])
    # The container rebuilds the graph from this YAML, so the activation has to
    # survive into the bundle.
    config = yaml.safe_load((tmp_path / "model.yaml").read_text())
    assert config["activation"] == activation_spec

    reference = np.load(tmp_path / "deploy_reference.npz")
    assert reference["images"].shape == (1, 3, *SMALL_INPUT)
    for name, shape in zip(spec["output_names"], [(20, 24), (10, 12), (5, 6)], strict=True):
        assert reference[name].shape == (1, 251, *shape)
