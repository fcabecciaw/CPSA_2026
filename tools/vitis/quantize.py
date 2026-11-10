#!/usr/bin/env python3
"""Vitis AI 3.5 PTQ of the raw-head pose graph (runs inside the container).

The container Python cannot import ``cpsa_ml`` (the package targets 3.10), so
this script rebuilds the deploy module from the host bundle: the same model
YAML, the host ``state_dict`` loaded strictly, and the host reference output
checked before calibration.  ``ml/src/cpsa_ml/deploy.py`` itself is loaded by
path, so the graph definition has a single source of truth.

Stages (``vai_q_pytorch``):
  1. ``calib``  -- quantizer calibration over the calibration frames,
     writes ``quant_info.json``.
  2. ``test``   -- quantized forward pass, compared against the float module.
  3. ``deploy`` -- ``export_xmodel`` writes the quantized ``.xmodel`` that
     ``vai_c_xir`` compiles.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


def _configure_thread_env() -> None:
    """Set the thread caps before PyTorch is imported.

    Calibration on a busy host spends most of its time in BLAS/OpenMP thread
    synchronisation unless the pool is capped; ``--threads`` therefore has to
    reach the process environment before ``import torch``.
    """

    argv = sys.argv[1:]
    threads = None
    for index, value in enumerate(argv):
        if value == "--threads" and index + 1 < len(argv):
            threads = argv[index + 1]
        elif value.startswith("--threads="):
            threads = value.split("=", 1)[1]
    if threads is not None:
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("MKL_NUM_THREADS", threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)


_configure_thread_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from pytorch_nndct.apis import torch_quantizer  # noqa: E402
from ultralytics.nn.tasks import PoseModel  # noqa: E402

DEFAULT_DEPLOY_MODULE = Path(__file__).resolve().parents[2] / "ml/src/cpsa_ml/deploy.py"


def load_deploy_module(path: Path) -> ModuleType:
    """Load ``cpsa_ml/deploy.py`` by path, bypassing the 3.10-only package."""

    spec = importlib.util.spec_from_file_location("cpsa_ml_deploy", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load deploy module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_deploy(deploy_module: ModuleType, artifacts: Path) -> tuple[torch.nn.Module, dict]:
    """Rebuild the deploy graph from the host bundle and load its weights."""

    config = yaml.safe_load((artifacts / "model.yaml").read_text(encoding="utf-8"))
    pose = PoseModel(
        cfg=config,
        ch=3,
        nc=int(config["nc"]),
        data_kpt_shape=tuple(config["kpt_shape"]),
        verbose=False,
    )
    deploy = deploy_module.deploy_from_yolo(pose)
    state = torch.load(str(artifacts / "deploy_state_dict.pt"), map_location="cpu")
    deploy.load_state_dict(state, strict=True)
    spec = json.loads((artifacts / "deploy_spec.json").read_text(encoding="utf-8"))
    deploy.input_size = tuple(int(value) for value in spec["input_size"])
    return deploy, spec


def check_reference(
    deploy: torch.nn.Module,
    artifacts: Path,
    spec: dict,
    tolerance: float = 1e-4,
) -> dict[str, dict[str, Any]]:
    """Verify the container rebuild matches the host reference output."""

    reference = np.load(artifacts / "deploy_reference.npz")
    images = torch.from_numpy(reference["images"].astype(np.float32))
    with torch.no_grad():
        outputs = deploy(images)
    report: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(spec["output_names"]):
        output = outputs[index]
        expected = reference[name]
        actual = output.numpy()
        difference = float(np.max(np.abs(actual - expected)))
        report[name] = {"shape": list(actual.shape), "max_abs_diff": difference}
        if difference > tolerance:
            raise SystemExit(
                f"container rebuild mismatch on {name}: max abs diff {difference:.3e} "
                f"> {tolerance:.1e}"
            )
    return report


def load_frames(directory: Path, limit: int) -> list[torch.Tensor]:
    """Load calibration frames as ``(1, 3, H, W)`` float32 tensors."""

    files = sorted(Path(directory).glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy calibration frames in {directory}")
    frames = []
    for path in files[:limit]:
        array = np.load(path).astype(np.float32)
        if array.ndim != 3:
            raise SystemExit(f"{path} is not (3, H, W), got {array.shape}")
        frames.append(torch.from_numpy(array[None, ...]))
    return frames


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def compare_outputs(
    float_outputs: list[list[np.ndarray]],
    quant_outputs: list[list[np.ndarray]],
    names: list[str],
) -> dict[str, dict[str, float]]:
    """Per-output cosine similarity and max absolute error over the frames."""

    report: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        float_stack = np.stack([frame[index] for frame in float_outputs])
        quant_stack = np.stack([frame[index] for frame in quant_outputs])
        report[name] = {
            "cosine": cosine_similarity(float_stack, quant_stack),
            "max_abs_error": float(np.max(np.abs(float_stack - quant_stack))),
            "mean_abs_error": float(np.mean(np.abs(float_stack - quant_stack))),
            "float_abs_max": float(np.max(np.abs(float_stack))),
            "quant_abs_max": float(np.max(np.abs(quant_stack))),
        }
    return report


def time_forwards(
    deploy: torch.nn.Module,
    quantized: torch.nn.Module,
    frames: list[torch.Tensor],
) -> dict[str, Any]:
    """Time float and quantized forwards on the first frames (baseline)."""

    timings: dict[str, Any] = {"frames": len(frames), "float_s": [], "quant_s": []}
    with torch.no_grad():
        for frame in frames[:1]:
            deploy(frame)
            quantized(frame)
        for frame in frames:
            started = time.time()
            deploy(frame)
            timings["float_s"].append(time.time() - started)
            started = time.time()
            quantized(frame)
            timings["quant_s"].append(time.time() - started)
    timings["float_s_per_frame"] = float(np.mean(timings["float_s"]))
    timings["quant_s_per_frame"] = float(np.mean(timings["quant_s"]))
    timings["float_s"] = [round(value, 3) for value in timings["float_s"]]
    timings["quant_s"] = [round(value, 3) for value in timings["quant_s"]]
    return timings


def parse_args(argv: list[str]) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=root / "build/vitis/artifacts")
    parser.add_argument("--calib", type=Path, default=root / "build/vitis/calib")
    parser.add_argument("--out", type=Path, default=root / "build/vitis/quant")
    parser.add_argument("--name", default="pose_halfbust62", help="xmodel base name")
    parser.add_argument("--deploy-module", type=Path, default=DEFAULT_DEPLOY_MODULE)
    parser.add_argument("--calib-count", type=int, default=120)
    parser.add_argument("--test-count", type=int, default=24)
    parser.add_argument("--save-frames", type=int, default=4)
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="torch/OpenMP thread cap (0 keeps the torch default)",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--time-only",
        action="store_true",
        help="time float and quantized forwards on the first frame, then exit",
    )
    parser.add_argument("--report", type=Path, default=None, help="JSON summary path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    deploy_module = load_deploy_module(args.deploy_module)
    deploy, spec = build_deploy(deploy_module, args.artifacts)
    reference_report = check_reference(deploy, args.artifacts, spec)
    print(json.dumps({"reference_check": reference_report}, indent=2))

    calib_frames = load_frames(args.calib, args.calib_count)
    test_frames = calib_frames[: min(args.test_count, len(calib_frames))]
    height, width = spec["input_size"]
    example = torch.zeros((1, 3, int(height), int(width)), dtype=torch.float32)
    device = torch.device("cpu")

    # -- calibration ------------------------------------------------------
    calib_quantizer = torch_quantizer(
        "calib", deploy, (example,), output_dir=str(args.out), device=device
    )
    quantized = calib_quantizer.quant_model

    if args.time_only:
        timing = time_forwards(deploy, quantized, calib_frames[: args.calib_count])
        timing["threads"] = args.threads
        timing["torch_threads"] = torch.get_num_threads()
        print(json.dumps(timing, indent=2))
        return 0

    def run_calibration(frames: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for index, frame in enumerate(frames):
                quantized(frame)
                if args.progress_every and (index + 1) % args.progress_every == 0:
                    elapsed = time.time() - calib_started
                    print(
                        f"[calib] {index + 1}/{len(frames)} frames in {elapsed:.1f}s "
                        f"({elapsed / (index + 1):.2f} s/frame)",
                        flush=True,
                    )

    calib_started = time.time()
    calib_quantizer.quantize(run_calibration, (calib_frames,))
    calib_seconds = time.time() - calib_started
    print(f"[calib] done in {calib_seconds:.1f}s", flush=True)

    # -- test (quantized vs float) ---------------------------------------
    test_quantizer = torch_quantizer(
        "test", deploy, (example,), output_dir=str(args.out), device=device
    )
    quant_model = test_quantizer.quant_model
    float_outputs: list[list[np.ndarray]] = []
    quant_outputs: list[list[np.ndarray]] = []
    with torch.no_grad():
        for frame in test_frames:
            float_outputs.append([tensor.numpy() for tensor in deploy(frame)])
            quant_outputs.append([tensor.numpy() for tensor in quant_model(frame)])
    test_quantizer.export_xmodel(output_dir=str(args.out), deploy_check=False)

    xmodels = sorted(args.out.glob("*.xmodel"))
    if not xmodels:
        raise SystemExit(f"export_xmodel wrote no .xmodel in {args.out}")
    produced = xmodels[0]
    target = args.out / f"{args.name}.xmodel"
    if produced != target:
        shutil.copyfile(produced, target)

    names = [str(name) for name in spec["output_names"]]
    comparison = compare_outputs(float_outputs, quant_outputs, names)

    saved = min(args.save_frames, len(test_frames))
    np.savez_compressed(
        args.out / "test_outputs.npz",
        frames=np.stack([frame.numpy() for frame in test_frames[:saved]]),
        **{
            f"float_{name}": np.stack([frame[index] for frame in float_outputs[:saved]])
            for index, name in enumerate(names)
        },
        **{
            f"quant_{name}": np.stack([frame[index] for frame in quant_outputs[:saved]])
            for index, name in enumerate(names)
        },
    )

    report = {
        "torch": torch.__version__,
        "nndct": __import__("pytorch_nndct").__version__,
        "input_size": [int(height), int(width)],
        "strides": spec["strides"],
        "output_names": names,
        "channel_layout": spec["channel_layout"],
        "calib_frames": len(calib_frames),
        "test_frames": len(test_frames),
        "calib_seconds": calib_seconds,
        "calib_seconds_per_frame": calib_seconds / max(len(calib_frames), 1),
        "threads": args.threads,
        "total_seconds": time.time() - started,
        "xmodel": str(target),
        "reference_check": reference_report,
        "quant_vs_float": comparison,
    }
    report_path = args.report or args.out / "quant_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
