"""YOLOv8-pose model definitions for the CPSA keypoint schemas.

The shipped configs are plain Ultralytics YAML files.  The activation variants
set the top-level ``activation`` key that Ultralytics applies to every ``Conv``
block: Hard-Swish (``hswish``), ``relu`` and ``lrelu`` are the DPU-friendly
variants of the activation study, while SiLU (``silu``) is the negative control.
There are no attention blocks, so the exported backbone and neck stay inside the
DPUCZDX8G operator set.
"""

from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path
from typing import Any

import yaml

from cpsa_ml.schema import KeypointSchema, get_schema

__all__ = [
    "build_model",
    "config_path",
    "load_yolo",
    "model_yaml_name",
    "model_yaml_path",
]

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
VARIANTS = ("n", "s")
ACTIVATIONS = ("hswish", "silu", "relu", "lrelu")


def model_yaml_name(variant: str = "n", activation: str = "hswish") -> str:
    """Return the config file name for a variant/activation pair."""

    variant = variant.lower()
    activation = activation.lower()
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if activation == "hardswish":
        activation = "hswish"
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
    return f"yolov8{variant}-pose-{activation}.yaml"


def config_path(name: str) -> Path:
    """Locate a packaged or source-tree model config."""

    packaged = Path(__file__).resolve().parent / "configs" / name
    if packaged.is_file():
        return packaged
    source = CONFIG_DIR / name
    if source.is_file():
        return source
    raise FileNotFoundError(f"model config {name!r} not found (looked in {packaged} and {source})")


def model_yaml_path(variant: str = "n", activation: str = "hswish") -> Path:
    """Return the path of the variant's YAML file."""

    return config_path(model_yaml_name(variant, activation))


def _model_config(
    schema: KeypointSchema,
    variant: str,
    activation: str,
    nc: int,
    cfg: str | Path | None,
) -> dict[str, Any]:
    variant = variant.lower()
    activation = activation.lower()
    path = Path(cfg) if cfg is not None else model_yaml_path(variant, activation)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"model config {path} did not contain a YAML mapping")
    config = copy.deepcopy(config)
    config["nc"] = int(nc)
    config["kpt_shape"] = list(schema.kpt_shape)
    config["scale"] = variant
    if activation.lower() in {"hswish", "hardswish"} and "activation" not in config:
        config["activation"] = "nn.Hardswish()"
    if activation.lower() in {"relu", "lrelu"} and "activation" not in config:
        raise ValueError(f"model config {path} must define the activation for {activation!r}")
    return config


def build_model(
    schema: str | KeypointSchema = "coco17",
    *,
    variant: str = "n",
    activation: str = "hswish",
    nc: int = 1,
    cfg: str | Path | None = None,
) -> Any:
    """Build a pose model for ``schema`` and return the underlying ``nn.Module``.

    This is the lightweight path for tests and introspection.  Use
    :func:`load_yolo` when a ``YOLO`` wrapper is needed for training/export.
    """

    from ultralytics.nn.tasks import PoseModel

    resolved = get_schema(schema)
    variant = variant.lower()
    activation = activation.lower()
    config = _model_config(resolved, variant, activation, nc, cfg)
    return PoseModel(
        cfg=config,
        ch=3,
        nc=nc,
        data_kpt_shape=resolved.kpt_shape,
        verbose=False,
    )


def _materialize_config(config: dict[str, Any], name: str) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "cpsa_ml_configs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config, sort_keys=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    path = cache_dir / f"{Path(name).stem}-{digest}.yaml"
    if not path.is_file():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return path


def load_yolo(
    schema: str | KeypointSchema = "coco17",
    *,
    variant: str = "n",
    activation: str = "hswish",
    nc: int = 1,
    cfg: str | Path | None = None,
) -> Any:
    """Build a ``ultralytics.YOLO`` wrapper for the requested schema.

    The wrapper is required by ``YOLO.train()`` and ``YOLO.export()``.  A
    schema-specific YAML is materialized in the system temp directory because
    Ultralytics reads ``kpt_shape`` when the model is constructed.
    """

    from ultralytics import YOLO

    resolved = get_schema(schema)
    variant = variant.lower()
    activation = activation.lower()
    config = _model_config(resolved, variant, activation, nc, cfg)
    name = f"yolov8{variant}-pose-{activation}-{resolved.name}-nc{nc}.yaml"
    path = _materialize_config(config, name)
    return YOLO(str(path), task="pose", verbose=False)
