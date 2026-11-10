"""ONNX Runtime pose estimator (CPU; NFR-004 stand-in for the DPU).

The adapter owns the full image pipeline: :func:`~cpsa.vision.pose_decode.letterbox`,
normalization to ``[0, 1]`` NCHW, one inference, then
:func:`~cpsa.vision.pose_decode.decode_ultralytics` and
:func:`~cpsa.vision.pose_decode.unletterbox`.  ``onnxruntime`` is imported
lazily so the module can be imported on hosts without the ``vision`` extra.

The model must export the decoded Ultralytics pose output, shape
``(1, 4 + nc + 3K, N)``.  Raw-head exports use
:class:`cpsa.adapters.pose.dpu.DpuPoseEstimator` instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cpsa.vision.pose_decode import decode_ultralytics, letterbox, unletterbox
from cpsa.vision.types import KeypointLayout, PersonPose

logger = logging.getLogger(__name__)

__all__ = ["OnnxPoseEstimator"]


class OnnxPoseEstimator:
    """Run one exported YOLOv8-pose ONNX model on the CPU.

    Args:
        model_path: Path to the ``.onnx`` file.
        layout: Keypoint layout the model was trained with.  ``len(layout)`` is
            the model's keypoint count.
        input_size: Model input as ``(height, width)``, matching the exported
            graph's fixed ``NCHW`` spatial dims.
        conf_thr: Minimum class score for a detection.
        iou_thr: NMS IoU threshold.
    """

    def __init__(
        self,
        model_path: str | Path,
        layout: KeypointLayout,
        input_size: tuple[int, int],
        conf_thr: float = 0.25,
        iou_thr: float = 0.45,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the installed extra
            raise RuntimeError(
                "onnxruntime is required for OnnxPoseEstimator; "
                "install the 'vision' extra (uv sync --extra vision)"
            ) from exc

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        if len(input_size) != 2 or min(input_size) <= 0:
            raise ValueError(f"input_size must be (height, width), got {input_size}")
        if not 0.0 <= conf_thr <= 1.0:
            raise ValueError(f"conf_thr must be in [0, 1], got {conf_thr}")
        if not 0.0 <= iou_thr <= 1.0:
            raise ValueError(f"iou_thr must be in [0, 1], got {iou_thr}")

        self._layout = layout
        self._input_size = (int(input_size[0]), int(input_size[1]))
        self._conf_thr = float(conf_thr)
        self._iou_thr = float(iou_thr)
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"expected one model input, got {len(inputs)}")
        self._input_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) != 4:
            raise ValueError(f"expected a 4-D NCHW input, got shape {shape}")
        fixed = shape[2:]
        if all(isinstance(value, int) for value in fixed) and tuple(fixed) != self._input_size:
            raise ValueError(
                f"model input spatial dims {tuple(fixed)} do not match input_size "
                f"{self._input_size}"
            )

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    def estimate(self, rgb: NDArray[np.uint8]) -> list[PersonPose]:
        """Run the model on an ``(H, W, 3)`` RGB uint8 image."""

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"rgb must be (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {image.dtype}")

        letterboxed, scale, pad = letterbox(image, self._input_size)
        tensor = (letterboxed.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        outputs = self._session.run(None, {self._input_name: tensor})
        if len(outputs) != 1:
            raise ValueError(
                f"expected one decoded model output, got {len(outputs)}; "
                "raw-head exports are not supported by OnnxPoseEstimator"
            )
        persons = decode_ultralytics(outputs[0], len(self._layout), self._conf_thr, self._iou_thr)
        return unletterbox(persons, scale, pad)
