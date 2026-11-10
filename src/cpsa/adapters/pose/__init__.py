"""Pose-estimation adapters: ONNX Runtime on CPU, Vitis AI DPU on the board."""

from cpsa.adapters.pose.dpu import DpuPoseEstimator
from cpsa.adapters.pose.onnx import OnnxPoseEstimator

__all__ = ["DpuPoseEstimator", "OnnxPoseEstimator"]
