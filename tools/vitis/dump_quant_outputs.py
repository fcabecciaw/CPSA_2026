#!/usr/bin/env python3
"""Dump quantized raw-head outputs for one trained variant (runs in the container).

The variant study needs the exact INT8 inputs of the runtime decoder: the
``vai_q_pytorch`` test-mode head tensors, converted to the int8 values the DPU
produces (``clip(round(value * 2 ** fix_point))``) together with the
``fix_point`` of every output tensor.  ``tools/vitis/eval_quant.py`` then
decodes them with ``cpsa.vision.pose_decode`` on the host and scores them with
``ml/src/cpsa_ml/eval.py``, so FP32 and INT8 accuracy use the same metric code.

Stages (all reuse :mod:`tools.vitis.quantize` helpers by import -- that module
stays the single source of the container-side PTQ flow):

1. rebuild the deploy graph from ``best.pt`` (or a host ``--artifacts``
   bundle), exactly as the deploy pipeline does;
2. calibrate from N train images, preprocessed exactly like
   :class:`cpsa.adapters.pose.dpu.DpuPoseEstimator` (RGB, letterbox to the
   deploy input, ``float32 / 255``); the letterbox below is a Python 3.8 port
   of ``cpsa.vision.pose_decode.letterbox`` and is checked bit-for-bit against
   it by ``tests/unit/test_eval_quant.py`` on the host;
3. run the test mode on M val images;
4. read each output tensor's ``fix_point`` from the exported quantized
   ``.xmodel`` (``fix`` op attribute -- the value the compiler carries to the
   DPU), quantize the test-mode float outputs to int8 and write
   ``quant_outputs.npz`` with the per-image letterbox parameters, so the host
   can map predictions back to the original image.

Example::

    tools/vitis/run_vitis.sh python tools/vitis/dump_quant_outputs.py \\
        --weights build/variant_study/hswish_480x640/best.pt \\
        --data /data/halfbust62 --calib-split train --test-split val \\
        --calib-count 64 --test-count 64 --height 480 --width 640 \\
        --out build/variant_study/hswish_480x640/quant --threads 1

Python 3.8 (the container interpreter); numpy and PIL only at frame level,
torch/PIL/NNDCT are imported inside the functions that need them so the
letterbox can be tested on the host without the Vitis AI stack.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

DEFAULT_DEPLOY_MODULE = Path(__file__).resolve().parents[2] / "ml/src/cpsa_ml/deploy.py"
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
DEFAULT_PAD_VALUE = 114
DEFAULT_INPUT_SIZE = (480, 640)


# ---------------------------------------------------------------------------
# Preprocessing (port of cpsa.vision.pose_decode.letterbox for Python 3.8)
# ---------------------------------------------------------------------------


def _resize_bilinear(
    image: np.ndarray,
    out_height: int,
    out_width: int,
) -> np.ndarray:
    """Resize an ``(H, W, 3)`` uint8 image with bilinear interpolation."""

    height, width = image.shape[:2]
    if (height, width) == (out_height, out_width):
        return image.copy()
    y = (np.arange(out_height, dtype=np.float64) + 0.5) * (height / out_height) - 0.5
    x = (np.arange(out_width, dtype=np.float64) + 0.5) * (width / out_width) - 0.5
    y0 = np.clip(np.floor(y).astype(np.intp), 0, height - 1)
    x0 = np.clip(np.floor(x).astype(np.intp), 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x1 = np.clip(x0 + 1, 0, width - 1)
    wy = np.clip(y - y0, 0.0, 1.0)[:, None, None]
    wx = np.clip(x - x0, 0.0, 1.0)[None, :, None]
    top_left = image[y0][:, x0].astype(np.float64)
    top_right = image[y0][:, x1].astype(np.float64)
    bottom_left = image[y1][:, x0].astype(np.float64)
    bottom_right = image[y1][:, x1].astype(np.float64)
    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    resized = top * (1.0 - wy) + bottom * wy
    return np.clip(np.rint(resized), 0, 255).astype(np.uint8)


def letterbox_rgb(
    rgb: np.ndarray,
    size: tuple[int, int],
    pad_value: int = DEFAULT_PAD_VALUE,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Port of ``cpsa.vision.pose_decode.letterbox`` (same pixels).

    Args:
        rgb: ``(H, W, 3)`` RGB uint8 image.
        size: Target ``(height, width)``.
        pad_value: Fill value for the padding.

    Returns:
        ``(image, scale, (pad_x, pad_y))``; ``image`` is ``(H, W, 3)`` uint8.
    """

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must be (H, W, 3), got shape {image.shape}")
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("rgb must not be empty")
    out_height, out_width = int(size[0]), int(size[1])
    if out_height <= 0 or out_width <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= pad_value <= 255:
        raise ValueError(f"pad_value must be in [0, 255], got {pad_value}")
    if (height, width) == (out_height, out_width) and image.dtype == np.uint8:
        return image, 1.0, (0.0, 0.0)
    scale = min(out_height / height, out_width / width)
    resized_height = max(1, min(out_height, round(height * scale)))
    resized_width = max(1, min(out_width, round(width * scale)))
    resized = _resize_bilinear(image, resized_height, resized_width)
    canvas = np.full((out_height, out_width, 3), pad_value, dtype=np.uint8)
    top = (out_height - resized_height) // 2
    left = (out_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, float(scale), (float(left), float(top))


def to_model_tensor(image: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` uint8 RGB -> ``(3, H, W)`` float32 in ``[0, 1]``.

    Matches ``DpuPoseEstimator``'s ``np.float32(value) / np.float32(255.0)``
    normalization (and hence ``tools/vitis/make_calib.py``).
    """

    transposed = np.ascontiguousarray(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))
    return transposed.astype(np.float32) / np.float32(255.0)


# ---------------------------------------------------------------------------
# Dataset and checkpoint loading
# ---------------------------------------------------------------------------


def resolve_images(root: Path, split: str | None, limit: int) -> list[Path]:
    """Return the sorted images of ``root`` (or ``root/<split>``)."""

    directory = Path(root)
    if split:
        nested = directory / split
        if nested.is_dir():
            directory = nested
    if not directory.is_dir():
        raise SystemExit(f"image directory not found: {directory}")
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"no images found in {directory}")
    return images[:limit]


def load_image(path: Path) -> np.ndarray:
    """Read one image as ``(H, W, 3)`` uint8 RGB."""

    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def preprocess_frames(
    images: list[Path],
    input_size: tuple[int, int],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Letterbox images and return tensors plus per-image metadata."""

    import torch

    frames: list[Any] = []
    metadata: list[dict[str, Any]] = []
    for path in images:
        image = load_image(path)
        letterboxed, scale, (pad_x, pad_y) = letterbox_rgb(image, input_size)
        tensor = torch.from_numpy(to_model_tensor(letterboxed)[None, ...])
        frames.append(tensor)
        metadata.append(
            {
                "name": path.stem,
                "path": str(path),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "scale": float(scale),
                "pad_x": float(pad_x),
                "pad_y": float(pad_y),
            }
        )
    return frames, metadata


def load_deploy_from_weights(
    deploy_module: ModuleType,
    weights: Path,
    input_size: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    """Rebuild the deploy graph from an Ultralytics ``best.pt``.

    Ultralytics trainer checkpoints store the live weights in ``ema`` and set
    ``model`` to ``None``; both layouts are accepted, together with a plain
    ``state_dict`` of the deploy graph.  The EMA is stored in half precision
    and is converted back to float, like ``attempt_load_one_weight`` does.
    """

    import torch

    path = Path(weights)
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} is not an Ultralytics checkpoint")
    source = None
    for key in ("ema", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, torch.nn.Module):
            source = key
            break
    if source is None:
        raise SystemExit(
            f"{path} has no 'ema' or 'model' module; export a host bundle with "
            "`python -m cpsa_ml.deploy --weights ...` and pass --artifacts"
        )
    model = payload[source]
    model = model.float().eval()
    deploy = deploy_module.deploy_from_yolo(model)
    deploy.input_size = (int(input_size[0]), int(input_size[1]))
    info = {
        "weights": str(path),
        "checkpoint_source": source,
        "checkpoint_keys": sorted(str(key) for key in payload),
        "epoch": payload.get("epoch"),
        "train_args_imgsz": (payload.get("train_args") or {}).get("imgsz"),
    }
    return deploy, info


def read_output_fix_points(
    xmodel: Path,
    input_size: tuple[int, int],
    channels: int,
) -> dict[int, dict[str, Any]]:
    """Map stride -> ``{fix_point, name, dims}`` from the quantized ``.xmodel``.

    ``export_xmodel`` leaves every graph output behind a ``fix`` op whose
    ``fix_point`` attribute is the tensor's quantization exponent.
    """

    import xir

    graph = xir.Graph.deserialize(str(xmodel))
    root = graph.get_root_subgraph()
    height, width = input_size
    resolved: dict[int, dict[str, Any]] = {}
    for tensor in root.get_output_tensors():
        dims = [int(dim) for dim in tensor.dims]
        if len(dims) != 4 or dims[0] != 1:
            raise SystemExit(f"unexpected output tensor shape {dims} for {tensor.name}")
        if dims[3] == channels:
            rows, columns = dims[1], dims[2]
        elif dims[1] == channels:
            rows, columns = dims[2], dims[3]
        else:
            raise SystemExit(
                f"cannot find {channels} channels in output tensor {tensor.name} {dims}"
            )
        if rows <= 0 or columns <= 0 or height % rows or width % columns:
            raise SystemExit(f"output {tensor.name} {dims} does not divide the input {input_size}")
        stride_y, stride_x = height // rows, width // columns
        if stride_y != stride_x:
            raise SystemExit(f"non-square stride for output {tensor.name} {dims}")
        producer = root.get_tensor_producer(tensor)
        attributes = producer.get_attrs() if producer is not None else {}
        fix_point = attributes.get("fix_point")
        if fix_point is None and tensor.has_attr("fix_point"):
            fix_point = tensor.get_attr("fix_point")
        if isinstance(fix_point, bytes):
            fix_point = fix_point.decode("utf-8")
        if fix_point is None:
            raise SystemExit(f"no fix_point on output {tensor.name} of {xmodel}")
        entry = {
            "fix_point": int(fix_point),
            "name": str(tensor.name),
            "dims": dims,
            "producer": producer.get_name() if producer is not None else None,
        }
        if stride_y in resolved:
            raise SystemExit(f"two outputs share stride {stride_y} in {xmodel}")
        resolved[int(stride_y)] = entry
    if not resolved:
        raise SystemExit(f"no outputs found in {xmodel}")
    return resolved


def quantize_to_int8(values: np.ndarray, fix_point: int) -> np.ndarray:
    """``clip(round(value * 2 ** fix_point))`` to int8, like the DPU does."""

    scaled = np.rint(np.asarray(values, dtype=np.float64) * (2.0**fix_point))
    return np.clip(scaled, -128.0, 127.0).astype(np.int8)


def to_nhwc(array: np.ndarray, channels: int) -> np.ndarray:
    """Convert one ``(1, C, H, W)`` or ``(1, H, W, C)`` output to NHWC."""

    squeezed = np.asarray(array)[0]
    if squeezed.shape[0] == channels and squeezed.shape[-1] != channels:
        return np.transpose(squeezed, (1, 2, 0))
    if squeezed.shape[-1] == channels:
        return squeezed
    raise ValueError(f"cannot find {channels} channels in shape {squeezed.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path, help="trained Ultralytics best.pt")
    source.add_argument(
        "--artifacts",
        type=Path,
        help="host deploy bundle from `python -m cpsa_ml.deploy --weights ...`",
    )
    parser.add_argument("--data", type=Path, default=None, help="half-bust dataset root")
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--test-split", default="val")
    parser.add_argument("--calib-images", type=Path, default=None)
    parser.add_argument("--test-images", type=Path, default=None)
    parser.add_argument(
        "--calib-frames",
        type=Path,
        default=None,
        help="directory of preprocessed (3, H, W) float32 .npy frames (overrides images)",
    )
    parser.add_argument("--calib-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--height", type=int, default=DEFAULT_INPUT_SIZE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_INPUT_SIZE[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="pose_halfbust62", help="exported xmodel base name")
    parser.add_argument("--deploy-module", type=Path, default=DEFAULT_DEPLOY_MODULE)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.calib_count < 1 or args.test_count < 1:
        raise SystemExit("--calib-count and --test-count must be positive")

    # Importing quantize configures OMP/MKL/OpenBLAS thread caps from sys.argv
    # before it imports torch; reuse its helpers instead of duplicating the
    # container-side PTQ flow.
    import quantize as qz
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    deploy_module = qz.load_deploy_module(args.deploy_module)
    reference_report: dict[str, Any] | None = None
    checkpoint_info: dict[str, Any]
    if args.artifacts is not None:
        deploy, spec = qz.build_deploy(deploy_module, args.artifacts)
        reference_report = qz.check_reference(deploy, args.artifacts, spec)
        input_size = (int(spec["input_size"][0]), int(spec["input_size"][1]))
        if (args.height, args.width) != input_size:
            raise SystemExit(
                f"--height/--width {(args.height, args.width)} do not match the artifact "
                f"input size {input_size}"
            )
        checkpoint_info = {"artifacts": str(args.artifacts)}
    else:
        input_size = (int(args.height), int(args.width))
        deploy, checkpoint_info = load_deploy_from_weights(deploy_module, args.weights, input_size)
        spec = deploy.spec()

    channels = int(spec["output_channels"])
    n_kpt = int(spec["n_kpt"])
    reg_max = int(spec["reg_max"])
    strides = [int(stride) for stride in spec["strides"]]

    if args.calib_frames is not None:
        calib_frames = qz.load_frames(args.calib_frames, args.calib_count)
        calib_meta: list[dict[str, Any]] = []
    else:
        if args.calib_images is not None:
            calib_root = args.calib_images
            calib_split = None
        elif args.data is not None:
            calib_root = args.data / "images"
            calib_split = args.calib_split
        else:
            raise SystemExit("pass --calib-images, --calib-frames or --data")
        calib_images = resolve_images(calib_root, calib_split, args.calib_count)
        calib_frames, calib_meta = preprocess_frames(calib_images, input_size)
    if not calib_frames:
        raise SystemExit("no calibration frames")

    if args.test_images is not None:
        test_root = args.test_images
        test_split = None
    elif args.data is not None:
        test_root = args.data / "images"
        test_split = args.test_split
    else:
        raise SystemExit("pass --test-images or --data for the test images")
    test_images = resolve_images(test_root, test_split, args.test_count)
    test_frames, test_meta = preprocess_frames(test_images, input_size)
    if not test_frames:
        raise SystemExit("no test frames")

    height, width = input_size
    example = torch.zeros((1, 3, height, width), dtype=torch.float32)
    device = torch.device("cpu")

    # -- calibration -------------------------------------------------------
    calib_quantizer = qz.torch_quantizer(
        "calib", deploy, (example,), output_dir=str(args.out), device=device
    )
    quantized = calib_quantizer.quant_model

    def run_calibration(frames: list[Any]) -> None:
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

    # -- test mode ---------------------------------------------------------
    test_quantizer = qz.torch_quantizer(
        "test", deploy, (example,), output_dir=str(args.out), device=device
    )
    quant_model = test_quantizer.quant_model
    float_outputs: list[list[np.ndarray]] = []
    quant_outputs: list[list[np.ndarray]] = []
    test_started = time.time()
    with torch.no_grad():
        for index, frame in enumerate(test_frames):
            float_outputs.append([tensor.numpy() for tensor in deploy(frame)])
            quant_outputs.append([tensor.numpy() for tensor in quant_model(frame)])
            if args.progress_every and (index + 1) % args.progress_every == 0:
                print(f"[test] {index + 1}/{len(test_frames)} frames", flush=True)
    test_seconds = time.time() - test_started

    test_quantizer.export_xmodel(output_dir=str(args.out), deploy_check=False)
    xmodels = sorted(args.out.glob("*.xmodel"))
    if not xmodels:
        raise SystemExit(f"export_xmodel wrote no .xmodel in {args.out}")
    quantized_xmodel = xmodels[0]
    target_xmodel = args.out / f"{args.name}.xmodel"
    if quantized_xmodel != target_xmodel:
        target_xmodel.write_bytes(quantized_xmodel.read_bytes())

    fix_points = read_output_fix_points(quantized_xmodel, input_size, channels)
    missing = sorted(set(strides) - set(fix_points))
    if missing:
        raise SystemExit(f"no quantized output for stride(s) {missing} in {quantized_xmodel}")
    ordered_fix_points = [int(fix_points[stride]["fix_point"]) for stride in strides]

    # -- int8-equivalent outputs -------------------------------------------
    arrays: dict[str, np.ndarray] = {}
    for index, stride in enumerate(strides):
        fix_point = ordered_fix_points[index]
        arrays[f"int8_head_stride{stride}"] = np.stack(
            [
                quantize_to_int8(to_nhwc(per_frame[index], channels), fix_point)
                for per_frame in quant_outputs
            ]
        )
    max_step_error = 0.0
    for index, stride in enumerate(strides):
        fix_point = ordered_fix_points[index]
        int8_stack = arrays[f"int8_head_stride{stride}"]
        for frame_index, per_frame in enumerate(quant_outputs):
            array = to_nhwc(per_frame[index], channels)
            reconstructed = int8_stack[frame_index].astype(np.float64) * (2.0**-fix_point)
            max_step_error = max(max_step_error, float(np.max(np.abs(array - reconstructed))))

    np.savez_compressed(
        args.out / "quant_outputs.npz",
        names=np.asarray([entry["name"] for entry in test_meta]),
        image_widths=np.asarray([entry["width"] for entry in test_meta], dtype=np.int32),
        image_heights=np.asarray([entry["height"] for entry in test_meta], dtype=np.int32),
        scales=np.asarray([entry["scale"] for entry in test_meta], dtype=np.float64),
        pad_x=np.asarray([entry["pad_x"] for entry in test_meta], dtype=np.float64),
        pad_y=np.asarray([entry["pad_y"] for entry in test_meta], dtype=np.float64),
        strides=np.asarray(strides, dtype=np.int32),
        fix_points=np.asarray(ordered_fix_points, dtype=np.int32),
        input_size=np.asarray(input_size, dtype=np.int32),
        n_kpt=np.int32(n_kpt),
        reg_max=np.int32(reg_max),
        num_classes=np.int32(spec["num_classes"]),
        channel_layout=np.asarray(
            [
                int(spec["channel_layout"]["box"]),
                int(spec["channel_layout"]["class"]),
                int(spec["channel_layout"]["keypoint"]),
            ],
            dtype=np.int32,
        ),
        **arrays,
    )

    names = [str(name) for name in spec["output_names"]]
    comparison = qz.compare_outputs(float_outputs, quant_outputs, names)
    report = {
        "weights": str(args.weights) if args.weights is not None else None,
        "artifacts": str(args.artifacts) if args.artifacts is not None else None,
        "checkpoint": checkpoint_info,
        "dataset": str(args.data) if args.data is not None else None,
        "calib_split": args.calib_split,
        "test_split": args.test_split,
        "calib_images": len(calib_frames),
        "test_images": len(test_frames),
        "calib_meta": calib_meta[:1],
        "torch": torch.__version__,
        "nndct": __import__("pytorch_nndct").__version__,
        "threads": args.threads,
        "input_size": [int(height), int(width)],
        "strides": strides,
        "fix_points": {
            stride: ordered_fix_points[position] for position, stride in enumerate(strides)
        },
        "output_names": names,
        "channel_layout": spec["channel_layout"],
        "n_kpt": n_kpt,
        "quant_vs_float": comparison,
        "max_int8_step_error": max_step_error,
        "reference_check": reference_report,
        "calib_seconds": calib_seconds,
        "test_seconds": test_seconds,
        "total_seconds": time.time() - started,
        "npz": str(args.out / "quant_outputs.npz"),
        "xmodel": str(target_xmodel),
    }
    (args.out / "deploy_spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    (args.out / "fix_points.json").write_text(
        json.dumps(
            {
                "strides": strides,
                "fix_points": ordered_fix_points,
                "outputs": {str(stride): fix_points[stride] for stride in strides},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out / "dump_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
