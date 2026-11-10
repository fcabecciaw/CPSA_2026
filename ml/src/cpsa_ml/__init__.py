"""CPSA vision-model toolchain (separate from the ``cpsa`` wheel).

Heavy dependencies (torch/Ultralytics/Brevitas) are imported lazily so the
dataset builder and schema can be used without a GPU stack.
"""

from __future__ import annotations

from typing import Any

from cpsa_ml.schema import COCO17, HALFBUST_62, KeypointSchema, get_schema

__version__ = "0.1.0"

__all__ = [
    "COCO17",
    "HALFBUST_62",
    "BuildConfig",
    "BuildReport",
    "KeypointSchema",
    "TrainConfig",
    "brevitas_available",
    "build_halfbust_dataset",
    "build_model",
    "check_onnx",
    "config_path",
    "export_onnx",
    "get_schema",
    "load_yolo",
    "quantize_model",
    "train",
    "try_nndct",
]

_LAZY: dict[str, tuple[str, str]] = {
    "BuildConfig": ("cpsa_ml.datasets.halfbust", "BuildConfig"),
    "BuildReport": ("cpsa_ml.datasets.halfbust", "BuildReport"),
    "build_halfbust_dataset": ("cpsa_ml.datasets.halfbust", "build_halfbust_dataset"),
    "TrainConfig": ("cpsa_ml.train", "TrainConfig"),
    "train": ("cpsa_ml.train", "train"),
    "build_model": ("cpsa_ml.models", "build_model"),
    "load_yolo": ("cpsa_ml.models", "load_yolo"),
    "config_path": ("cpsa_ml.models", "config_path"),
    "check_onnx": ("cpsa_ml.onnx_check", "check_onnx"),
    "export_onnx": ("cpsa_ml.onnx_check", "export_onnx"),
    "brevitas_available": ("cpsa_ml.qat", "brevitas_available"),
    "quantize_model": ("cpsa_ml.qat", "quantize_model"),
    "try_nndct": ("cpsa_ml.qat", "try_nndct"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = __import__(module_name, fromlist=[attribute])
    value = getattr(module, attribute)
    globals()[name] = value
    return value
