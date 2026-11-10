"""Optional Brevitas fixed-point QAT path and ``pytorch_nndct`` probe.

The baseline flow is FP32 -> Vitis AI PTQ.  QAT is only used when the PTQ
accuracy drop exceeds the AC-09 gate.  This module provides a layer-wise
Brevitas wrapper that emulates the DPU's symmetric INT8, power-of-two scales:

* ``Int8WeightPerTensorFixedPoint`` for weights
* ``Int8ActPerTensorFixedPoint`` for activations
* ``QuantConv2d`` and ``QuantHardSwish`` (Brevitas 0.13.4)

The wrapper is layer-wise, not FX/graph traced.  Residual adds, batch norms and
the Ultralytics decode stay in floating point; the adapted float weights are
intended to go through the same Vitis PTQ flow afterwards.  Brevitas is never
used to export the deployment graph.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Any

__all__ = [
    "NndctProbe",
    "brevitas_available",
    "brevitas_version",
    "quantize_model",
    "try_nndct",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NndctProbe:
    """Result of probing for AMD/Xilinx ``pytorch_nndct``."""

    available: bool
    version: str | None
    reason: str

    def __bool__(self) -> bool:
        return self.available


def try_nndct() -> NndctProbe:
    """Report whether ``pytorch_nndct`` can be imported in this environment."""

    spec = importlib.util.find_spec("pytorch_nndct")
    if spec is None:
        return NndctProbe(False, None, "pytorch_nndct is not installed")
    try:
        module = importlib.import_module("pytorch_nndct")
    except Exception as exc:  # noqa: BLE001 - probe must never crash the training job
        return NndctProbe(False, None, f"import failed: {exc}")
    version = getattr(module, "__version__", None)
    return NndctProbe(True, str(version) if version else "unknown", "import succeeded")


def brevitas_available() -> bool:
    return importlib.util.find_spec("brevitas") is not None


def brevitas_version() -> str | None:
    if not brevitas_available():
        return None
    module = importlib.import_module("brevitas")
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else "unknown"


def _require_brevitas() -> None:
    if not brevitas_available():
        raise ImportError(
            'Brevitas is not installed. Install the optional extra: pip install "cpsa-ml[qat]"'
        )


def quantize_model(
    model: Any,
    *,
    inplace: bool = False,
    weight_quant: Any = None,
    act_quant: Any = None,
) -> Any:
    """Replace Conv2d and Hardswish layers with Brevitas fixed-point modules.

    Args:
        model: Any ``torch.nn.Module`` (for example an Ultralytics ``PoseModel``).
        inplace: Modify the input model instead of returning a deep copy.
        weight_quant: Override the weight quantizer class.
        act_quant: Override the activation quantizer class.

    Returns:
        The quantized module.

    Raises:
        ImportError: If Brevitas is not installed.
    """

    _require_brevitas()
    import torch
    from brevitas.nn import QuantConv2d, QuantHardSwish
    from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint

    weight_quant = weight_quant or Int8WeightPerTensorFixedPoint
    act_quant = act_quant or Int8ActPerTensorFixedPoint
    root = model if inplace else copy.deepcopy(model)
    replaced_conv = 0
    replaced_act = 0

    def replace_children(parent: Any) -> None:
        nonlocal replaced_conv, replaced_act
        for name, child in list(parent.named_children()):
            if isinstance(child, QuantConv2d):
                continue
            if isinstance(child, torch.nn.Conv2d):
                quant = QuantConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                    weight_quant=weight_quant,
                    input_quant=act_quant,
                    output_quant=None,
                )
                with torch.no_grad():
                    quant.weight.copy_(child.weight)
                    if child.bias is not None and quant.bias is not None:
                        quant.bias.copy_(child.bias)
                quant.to(device=child.weight.device, dtype=child.weight.dtype)
                quant.train(child.training)
                setattr(parent, name, quant)
                replaced_conv += 1
            elif isinstance(child, torch.nn.Hardswish):
                quant_act = QuantHardSwish(act_quant=act_quant, input_quant=act_quant)
                quant_act.train(child.training)
                setattr(parent, name, quant_act)
                replaced_act += 1
            else:
                replace_children(child)

    replace_children(root)
    if replaced_conv == 0:
        LOGGER.warning("no Conv2d layers found; did you pass the pose model itself?")
    LOGGER.info(
        "Brevitas layer-wise QAT: %d QuantConv2d, %d QuantHardSwish",
        replaced_conv,
        replaced_act,
    )
    return root
