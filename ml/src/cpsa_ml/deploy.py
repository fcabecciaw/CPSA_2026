"""Raw-head deploy graph for the KV260 DPU.

The DPU runs the backbone, neck and head convolutions only.  The deploy graph
returns, per stride (8, 16, 32), the raw head tensors concatenated on channels::

    [ box (4 * reg_max) | class logits (nc) | keypoints (3 * K) ]

That layout is exactly what :func:`cpsa.vision.pose_decode.decode_raw_heads`
decodes on the CPU (DFL softmax, sigmoid, anchor grid, NMS).  Nothing after the
head convolutions -- no DFL, no sigmoid, no coordinate decode -- is part of the
graph, so no decode operator can fall back to the CPU.

The wrapper keeps the Ultralytics layer ``nn.Sequential`` as
:attr:`RawHeadDeploy.model`, so ``state_dict`` keys are identical to a
``YOLO(...).model`` checkpoint and a Kaggle ``best.pt`` loads unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = [
    "DEFAULT_INPUT_SIZE",
    "RawHeadDeploy",
    "build_deploy_model",
    "deploy_from_yolo",
    "export_bundle",
    "export_onnx",
    "export_torchscript",
    "load_deploy_weights",
    "main",
    "output_names",
    "random_input",
    "write_spec",
]

#: Deploy input size ``(height, width)``; same as ``ml`` ``onnx_check`` and the
#: runtime ``vision.model.input_width/height``.
DEFAULT_INPUT_SIZE = (480, 640)


def _underlying_model(model: Any) -> nn.Module:
    """Return the Ultralytics ``PoseModel`` behind a ``YOLO`` wrapper or module."""

    candidate: Any = model
    for _ in range(4):
        if not isinstance(candidate, nn.Module):
            break
        layers = getattr(candidate, "model", None)
        if isinstance(layers, nn.Sequential) and hasattr(candidate, "save"):
            return candidate
        candidate = layers
    raise TypeError(f"expected a YOLO wrapper or an Ultralytics PoseModel, got {type(model)!r}")


class RawHeadDeploy(nn.Module):
    """YOLOv8-pose truncated at the raw detection-head convolutions.

    Args:
        model: The Ultralytics ``PoseModel`` (an ``nn.Sequential`` under
            ``model.model`` whose last layer is the ``Pose`` head).
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        pose_model = _underlying_model(model)
        layers = pose_model.model
        if not isinstance(layers, nn.Sequential) or len(layers) < 2:
            raise TypeError("expected a PoseModel whose .model is a non-empty nn.Sequential")
        head = layers[-1]
        for name in ("cv2", "cv3", "cv4", "nl", "reg_max"):
            if not hasattr(head, name):
                raise TypeError(f"last layer {type(head).__name__} is not a YOLOv8-pose head")
        head_f = head.f if isinstance(head.f, (list, tuple)) else [head.f]
        if len(head_f) != head.nl:
            raise ValueError(f"head has {head.nl} levels but {len(head_f)} inputs")

        # Keys stay ``model.*`` so a trained checkpoint loads unchanged.
        self.model = pose_model.model
        self._save_ids = frozenset(int(index) for index in pose_model.save)
        self._head_f = tuple(int(index) for index in head_f)
        self.config: dict[str, Any] = dict(getattr(pose_model, "yaml", {}) or {})
        self.input_size = DEFAULT_INPUT_SIZE

    # -- metadata ---------------------------------------------------------

    @property
    def head(self) -> nn.Module:
        head = self.model[-1]
        assert isinstance(head, nn.Module)
        return head

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(int(stride) for stride in self.head.stride)

    @property
    def num_classes(self) -> int:
        return int(self.head.nc)

    @property
    def reg_max(self) -> int:
        return int(self.head.reg_max)

    @property
    def n_kpt(self) -> int:
        return int(self.head.kpt_shape[0])

    @property
    def kpt_dims(self) -> int:
        return int(self.head.kpt_shape[1])

    @property
    def output_channels(self) -> int:
        """Channels of every per-stride output: box + class + keypoint."""

        return 4 * self.reg_max + self.num_classes + self.kpt_dims * self.n_kpt

    def channel_layout(self) -> dict[str, int]:
        """Sizes of the three channel groups of an output tensor."""

        return {
            "box": 4 * self.reg_max,
            "class": self.num_classes,
            "keypoint": self.kpt_dims * self.n_kpt,
        }

    def spec(self) -> dict[str, Any]:
        """Serializable description of the deploy graph."""

        return {
            "input_size": list(self.input_size),
            "strides": list(self.strides),
            "output_channels": self.output_channels,
            "channel_layout": self.channel_layout(),
            "num_classes": self.num_classes,
            "reg_max": self.reg_max,
            "n_kpt": self.n_kpt,
            "kpt_dims": self.kpt_dims,
            "output_names": output_names(self.strides),
        }

    # -- forward ----------------------------------------------------------

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return one ``(1, C, H, W)`` raw-head tensor per stride.

        The computation mirrors ``ultralytics.nn.tasks.BaseModel._predict_once``
        over every layer but the head, then concatenates the head's box, class
        and keypoint convolutions on channels.  Ultralytics' own layer
        metadata (``f``, ``i`` and ``save``) drives the graph traversal, so no
        architecture detail is duplicated here.
        """

        layers = self.model
        head = layers[-1]
        saved: list[Any] = []
        value: Any = images
        for module in list(layers)[:-1]:
            source = module.f
            if source != -1:
                if isinstance(source, int):
                    value = saved[source]
                else:
                    value = [value if index == -1 else saved[index] for index in source]
            value = module(value)
            saved.append(value if module.i in self._save_ids else None)
        features = [saved[index] for index in self._head_f]
        return tuple(
            torch.cat(
                (
                    head.cv2[level](features[level]),
                    head.cv3[level](features[level]),
                    head.cv4[level](features[level]),
                ),
                dim=1,
            )
            for level in range(head.nl)
        )


def deploy_from_yolo(model: Any) -> RawHeadDeploy:
    """Wrap an Ultralytics ``YOLO`` wrapper or ``PoseModel`` for deployment."""

    deploy = RawHeadDeploy(_underlying_model(model))
    return deploy.eval()


def build_deploy_model(
    schema: str = "halfbust62",
    *,
    variant: str = "n",
    activation: str = "hswish",
    nc: int = 1,
    cfg: str | Path | None = None,
    seed: int | None = 0,
) -> RawHeadDeploy:
    """Build a deploy graph from the ``ml`` model config.

    Args:
        schema: Keypoint schema name (``halfbust62`` or ``coco17``).
        variant: Ultralytics scale (``n`` or ``s``).
        activation: ``hswish`` (DPU-friendly) or ``silu``; the extra DPU
            speed-matrix activations (``relu``, ``lrelu``) are selected by
            passing their config through ``cfg``.
        nc: Number of classes.
        cfg: Optional model YAML overriding the packaged config.
        seed: Seed for the random initialisation; ``None`` keeps the factory
            default weights.
    """

    if seed is not None:
        torch.manual_seed(int(seed))
    from cpsa_ml.models import build_model

    model = build_model(schema, variant=variant, activation=activation, nc=nc, cfg=cfg)
    return deploy_from_yolo(model)


def output_names(strides: Sequence[int]) -> list[str]:
    """Tensor names emitted per stride, smallest (largest map) first."""

    return [f"head_stride{int(stride)}" for stride in strides]


def random_input(
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic ``(1, 3, H, W)`` input in ``[0, 1]``."""

    generator = torch.Generator().manual_seed(int(seed))
    height, width = input_size
    return torch.rand((1, 3, height, width), generator=generator)


def load_deploy_weights(model: nn.Module, weights: str | Path) -> None:
    """Load a YOLO checkpoint or a plain ``state_dict`` into ``model``.

    Accepts the ``{"model": ...}`` layout of an Ultralytics ``.pt`` file, a
    bare ``state_dict`` and a saved ``RawHeadDeploy.state_dict()``.
    """

    payload = torch.load(str(weights), map_location="cpu", weights_only=False)
    state: Any = payload
    if isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
    if isinstance(state, nn.Module):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise TypeError(f"{weights} does not contain a state_dict or model")
    model.load_state_dict(state, strict=True)


def write_spec(model: RawHeadDeploy, path: str | Path) -> Path:
    """Write the deploy spec as JSON for the container-side pipeline."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.spec(), indent=2) + "\n", encoding="utf-8")
    return target


def export_torchscript(
    model: RawHeadDeploy,
    path: str | Path,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> Path:
    """Trace and save the deploy graph as TorchScript.

    Tracing pins the fixed ``1x3xHxW`` input.  TorchScript archives are only
    loadable by the same or a newer PyTorch, which is why the container-side
    pipeline rebuilds the module instead of relying on this file.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    example = random_input(input_size)
    with torch.no_grad():
        traced = torch.jit.trace(model.eval(), example, strict=False)
        traced = torch.jit.freeze(traced)
    traced.save(str(target))
    return target


def export_onnx(
    model: RawHeadDeploy,
    path: str | Path,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    opset: int = 17,
) -> Path:
    """Export the deploy graph to ONNX with one output per stride."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    example = random_input(input_size)
    with torch.no_grad():
        torch.onnx.export(
            model.eval(),
            (example,),
            str(target),
            input_names=["images"],
            output_names=output_names(model.strides),
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return target


def export_bundle(
    model: RawHeadDeploy,
    out_dir: str | Path,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    *,
    onnx: bool = True,
    torchscript: bool = False,
) -> dict[str, Path]:
    """Write the host artifacts the container-side PTQ pipeline consumes.

    The container PyTorch (1.13.1) cannot load TorchScript traced by the host
    PyTorch, so the bundle carries the plain ``state_dict`` plus the model
    config: the container rebuilds the same ``nn.Sequential`` with Ultralytics
    and loads the weights strictly.

    Args:
        model: Deploy graph to export.
        out_dir: Destination directory.
        input_size: Fixed ``(height, width)`` input.
        onnx: Also write an ONNX export (inspection and fallback only).
        torchscript: Also write a traced ``deploy_traced.pt`` (same-version
            PyTorch only).

    Returns:
        The written paths, keyed by artifact name.
    """

    import numpy as np
    import yaml

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.input_size = (int(input_size[0]), int(input_size[1]))

    paths: dict[str, Path] = {}
    paths["state_dict"] = root / "deploy_state_dict.pt"
    torch.save(model.state_dict(), paths["state_dict"])
    paths["spec"] = write_spec(model, root / "deploy_spec.json")
    paths["model_config"] = root / "model.yaml"
    paths["model_config"].write_text(
        yaml.safe_dump(model.config, sort_keys=False), encoding="utf-8"
    )

    images = random_input(model.input_size)
    with torch.no_grad():
        outputs = [output.cpu().numpy() for output in model(images)]
    paths["reference"] = root / "deploy_reference.npz"
    arrays = {"images": images.numpy()}
    for name, output in zip(output_names(model.strides), outputs, strict=True):
        arrays[name] = output
    np.savez_compressed(paths["reference"], **arrays)

    if torchscript:
        paths["torchscript"] = export_torchscript(
            model, root / "deploy_traced.pt", model.input_size
        )
    if onnx:
        paths["onnx"] = export_onnx(model, root / "deploy.onnx", model.input_size)
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="halfbust62")
    parser.add_argument("--variant", default="n")
    parser.add_argument("--activation", default="hswish")
    parser.add_argument("--nc", type=int, default=1)
    parser.add_argument("--cfg", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None, help="YOLO best.pt or state_dict")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=DEFAULT_INPUT_SIZE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_INPUT_SIZE[1])
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--onnx", action="store_true", help="also export ONNX")
    parser.add_argument("--torchscript", action="store_true", help="also trace TorchScript")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Export the deploy bundle (``state_dict``, spec, model config, reference)."""

    args = parse_args(argv)
    input_size = (args.height, args.width)
    model = build_deploy_model(
        args.schema,
        variant=args.variant,
        activation=args.activation,
        nc=args.nc,
        cfg=args.cfg,
        seed=None if args.weights is not None else args.seed,
    )
    if args.weights is not None:
        load_deploy_weights(model, args.weights)
    paths = export_bundle(
        model,
        args.out,
        input_size,
        onnx=args.onnx,
        torchscript=args.torchscript,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    print(json.dumps(model.spec(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
