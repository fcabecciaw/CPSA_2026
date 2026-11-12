#!/usr/bin/env python3
"""YOLOX-nano reference check for the showcase detector (AC-S2).

The board runs the compiled ``yolox_nano.xmodel`` on the DPU; this tool checks
the CPU half of that pipeline against the Vitis AI model-zoo package itself on
real images.  It has three modes, because the reference model needs PyTorch and
``pytorch_nndct`` (the Vitis AI container, Python 3.8) while ``cpsa`` runs on
the host:

1. ``prepare`` (host, ``uv run``): letterbox every input image exactly as
   :func:`cpsa.yolox.prepare_input` does (BGR, pad 114) and
   save the model inputs plus the geometry (``scale``, ``pad``, source size).
2. ``dump`` (Vitis AI container, ``tools/vitis/run_vitis.sh``): run the package
   model on those saved inputs, apply the package's own post-processing
   (``sigmoid`` on channels ``4:``, grid/stride decode, class-aware NMS) and
   save the raw head tensors with the resulting detections.
3. ``check`` (host): re-run the preprocessing, assert it reproduces the saved
   inputs pixel for pixel, round-trip the reference heads through int8 with
   ``fix_point 3`` (the DPU output precision from
   ``inspect_xmodel.py``), decode them with
   :func:`cpsa.yolox.detections_from_heads` and compare the
   boxes against the package's detections: same class, IoU >= 0.9, no extra or
   missing boxes.  Writes ``build/vitis/report/yolox_reference.json`` and exits
   non-zero on any mismatch.

Example::

    uv run --no-sync python tools/vitis/yolox_reference.py prepare \\
        --images build/vitis/yolox_reference/images --work build/vitis/yolox_reference
    tools/vitis/run_vitis.sh python tools/vitis/yolox_reference.py dump \\
        --model build/models/zoo/yolox_nano/pt_yolox-nano_coco_416_416_1G_3.0 \\
        --work build/vitis/yolox_reference
    uv run --no-sync python tools/vitis/yolox_reference.py check \\
        --work build/vitis/yolox_reference \\
        --out build/vitis/report/yolox_reference.json

The reference path is the QAT-converted deployable model of the package
(``qat/convert_qat_results/deployable.pth``), the same weights the ``int``
xmodel was dumped from.  Re-running ``pytorch_nndct``'s quantized simulation in
the container is not possible: the package's ``quant_info.json`` was produced by
Vitis AI 3.0 and the 3.5 processor rejects its node names.  The decode
comparison is unaffected, because both sides decode the very same head tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_WORK = Path("build/vitis/yolox_reference")
DEFAULT_OUT = Path("build/vitis/report/yolox_reference.json")

#: Input/output ``fix_point`` of the compiled graph (``inspect_xmodel.py``).
INPUT_FIX_POINT = -2
OUTPUT_FIX_POINT = 3

#: Boxes smaller than this (source pixels squared) carry no usable information:
#: predictions into the letterbox padding collapse against the frame edge, and
#: their IoU is unstable.  They are excluded from the comparison on both sides.
COMPARISON_MIN_AREA = 1.0


def _image_paths(directory: Path, limit: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"no images found in {directory}")
    if limit is not None:
        paths = paths[:limit]
    return paths


def _read_bgr(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image {path}")
    return image


def _meta_path(work: Path) -> Path:
    return work / "meta.json"


def _quantize_head(head: np.ndarray, fix_point: int) -> np.ndarray:
    """Round-trip one head through int8, exactly as the DPU does on its outputs."""

    scale = 2.0**fix_point
    return np.clip(np.rint(head * scale), -128.0, 127.0) / scale


def _quantize_input(bgr: np.ndarray, fix_point: int) -> np.ndarray:
    """Round-trip the letterboxed frame through the xint8 input grid (``fp=-2``: steps of 4)."""

    step = 2.0 ** (-fix_point)
    return np.clip(np.rint(bgr / step), -128.0, 127.0) * step


def _load_meta(work: Path) -> dict[str, Any]:
    path = _meta_path(work)
    if not path.is_file():
        raise SystemExit(f"missing {path}; run 'prepare' first")
    return json.loads(path.read_text())


def prepare(images: Path, work: Path, limit: int | None) -> dict[str, Any]:
    """Letterbox the images with the runtime preprocessing and save the model inputs."""

    import cv2

    from cpsa.yolox import YOLOX_INPUT_SIZE, prepare_input

    inputs = work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for path in _image_paths(images, limit):
        bgr = _read_bgr(path)
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        letterboxed, scale, pad = prepare_input(rgb, YOLOX_INPUT_SIZE)
        stem = path.stem.replace(" ", "_")
        target = inputs / f"{stem}.png"
        if not cv2.imwrite(str(target), letterboxed):
            raise SystemExit(f"cannot write {target}")
        entries.append(
            {
                "name": path.name,
                "stem": stem,
                "source": str(path.resolve()),
                "height": int(bgr.shape[0]),
                "width": int(bgr.shape[1]),
                "scale": float(scale),
                "pad": [float(pad[0]), float(pad[1])],
            }
        )
        print(f"prepare: {path.name} -> {target.name} scale={scale:.4f} pad={pad}")
    meta = {"input_size": list(YOLOX_INPUT_SIZE), "images": entries}
    _meta_path(work).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _build_reference_model(model_dir: Path) -> Any:
    """Instantiate the package's QAT-converted deployable model."""

    import torch

    code = model_dir / "code"
    if not code.is_dir():
        raise SystemExit(f"{code} not found; extract the zoo package first")
    sys.path.insert(0, str(code))
    from yolox.models.yolo_head_q import YOLOXHead  # type: ignore[import-not-found]
    from yolox.models.yolo_pafpn_deploy_q import YOLOPAFPN  # type: ignore[import-not-found]
    from yolox.models.yolox_q import YOLOX  # type: ignore[import-not-found]

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(0.33, 0.25, in_channels=in_channels, act="relu", depthwise=True)
    head = YOLOXHead(80, 0.25, in_channels=in_channels, act="relu", depthwise=True)
    network = YOLOX(backbone, head)
    weights = model_dir / "qat" / "convert_qat_results" / "deployable.pth"
    if not weights.is_file():
        raise SystemExit(f"{weights} not found; extract the zoo package first")
    network.load_state_dict(torch.load(str(weights), map_location="cpu"))
    network.eval()
    return network


def _package_postprocess(
    heads: list[np.ndarray],
    input_size: tuple[int, int],
    conf_thr: float,
    iou_thr: float,
) -> np.ndarray:
    """Apply the package's own post-processing to raw ``(C, H, W)`` heads.

    Mirrors ``YOLOXHead.postprocess`` (sigmoid on channels ``4:``, grid/stride
    decode) and then the package's canonical inference path,
    ``yolox.utils.boxes.postprocess`` (best class per cell, ``obj * cls``
    threshold, class-aware ``torchvision.ops.batched_nms``).
    """

    import torch
    from yolox.utils.boxes import postprocess  # type: ignore[import-not-found]
    from yolox.utils.demo_utils import demo_postprocess  # type: ignore[import-not-found]

    predictions = []
    for head in heads:
        activated = head[None].copy()
        activated[:, 4:] = 1.0 / (1.0 + np.exp(-activated[:, 4:]))
        _, channels, height, width = activated.shape
        predictions.append(activated.reshape(1, channels, height * width))
    flat = np.concatenate(predictions, axis=2).transpose(0, 2, 1)[0]
    decoded = demo_postprocess(flat[None], input_size, p6=False)[0]

    # ``postprocess`` expects ``(cx, cy, w, h)`` and converts to corners itself.
    prediction = torch.from_numpy(decoded.astype(np.float32)[None].copy())
    detections = postprocess(
        prediction,
        num_classes=prediction.shape[2] - 5,
        conf_thre=conf_thr,
        nms_thre=iou_thr,
        class_agnostic=False,
    )[0]
    if detections is None:
        return np.zeros((0, 6), dtype=np.float64)
    rows = detections.numpy().astype(np.float64)
    result = np.empty((len(rows), 6), dtype=np.float64)
    result[:, :4] = rows[:, :4]
    result[:, 4] = rows[:, 4] * rows[:, 5]
    result[:, 5] = rows[:, 6]
    return result


def dump(model: Path, work: Path, conf_thr: float, iou_thr: float, limit: int | None) -> None:
    """Run the package model on the prepared inputs and save heads + detections."""

    import torch

    meta = _load_meta(work)
    input_size = (int(meta["input_size"][0]), int(meta["input_size"][1]))
    network = _build_reference_model(model)
    reference = work / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    entries = meta["images"][:limit] if limit is not None else meta["images"]
    dumped: list[dict[str, Any]] = []
    for entry in entries:
        bgr = _read_bgr(work / "inputs" / f"{entry['stem']}.png")
        # The DPU input is xint8 with this fix point, so the graph sees the
        # frame on a 4-wide grey grid; round before the forward pass.
        quantized = _quantize_input(bgr.astype(np.float64), INPUT_FIX_POINT)
        tensor = torch.from_numpy(quantized.astype(np.float32).transpose(2, 0, 1)[None])
        with torch.no_grad():
            raw = network(tensor)
        nchw = [head[0].numpy().astype(np.float32) for head in raw]
        # The DPU returns int8 heads with this fix point; round-trip them here so
        # both the package post-processing and the runtime decode see exactly
        # the values the board decodes.
        nchw = [_quantize_head(head, OUTPUT_FIX_POINT) for head in nchw]
        nhwc = [head.transpose(1, 2, 0).copy() for head in nchw]
        strides = [input_size[0] // head.shape[0] for head in nhwc]
        detections = _package_postprocess(nchw, input_size, conf_thr, iou_thr)
        arrays: dict[str, Any] = {"strides": np.asarray(strides, dtype=np.int64)}
        arrays.update({f"head_{index}": head for index, head in enumerate(nhwc)})
        arrays["detections"] = detections
        np.savez_compressed(reference / f"{entry['stem']}.npz", **arrays)
        dumped.append(
            {"name": entry["name"], "stem": entry["stem"], "detections": int(len(detections))}
        )
        print(f"dump: {entry['name']} -> {len(detections)} package detections")
    (work / "dump.json").write_text(
        json.dumps(
            {
                "model": str(model),
                "weights": "qat/convert_qat_results/deployable.pth",
                "input_fix_point": INPUT_FIX_POINT,
                "conf_thr": conf_thr,
                "iou_thr": iou_thr,
                "images": dumped,
            },
            indent=2,
        )
        + "\n"
    )


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _match(
    ours: list[Any],
    reference: np.ndarray,
    min_iou: float,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    used_reference: set[int] = set()
    used_ours: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for index, detection in enumerate(ours):
        best_index, best_iou = -1, 0.0
        for ref_index, row in enumerate(reference):
            if ref_index in used_reference or int(row[5]) != detection.class_id:
                continue
            overlap = _iou(np.asarray(detection.box), row[:4])
            if overlap > best_iou:
                best_index, best_iou = ref_index, overlap
        if best_index >= 0 and best_iou >= min_iou:
            used_reference.add(best_index)
            used_ours.add(index)
            pairs.append({"ours": index, "reference": best_index, "iou": best_iou})
    missing = [index for index in range(len(reference)) if index not in used_reference]
    extras = [index for index in range(len(ours)) if index not in used_ours]
    return pairs, extras, missing


def check(
    work: Path,
    out: Path,
    conf_thr: float,
    iou_thr: float,
    min_iou: float,
) -> dict[str, Any]:
    """Compare the runtime decode with the package detections and write the report."""

    from cpsa.yolox import (
        YOLOX_INPUT_SIZE,
        boxes_to_source,
        detections_from_heads,
        prepare_input,
    )

    meta = _load_meta(work)
    dump_info_path = work / "dump.json"
    dump_info = json.loads(dump_info_path.read_text()) if dump_info_path.is_file() else {}
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    matched_total = 0
    min_iou_seen = 1.0
    for entry in meta["images"]:
        bgr = _read_bgr(Path(entry["source"]))
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        letterboxed, scale, pad = prepare_input(rgb, YOLOX_INPUT_SIZE)
        saved = _read_bgr(work / "inputs" / f"{entry['stem']}.png")
        preprocess_identical = bool(
            letterboxed.shape == saved.shape and np.array_equal(letterboxed, saved)
        )

        reference_path = work / "reference" / f"{entry['stem']}.npz"
        if not reference_path.is_file():
            raise SystemExit(f"missing {reference_path}; run 'dump' first")
        data = np.load(reference_path)
        strides = [int(stride) for stride in data["strides"]]
        heads = [data[f"head_{index}"] for index in range(len(strides))]
        reference = data["detections"]
        source_hw = (int(entry["height"]), int(entry["width"]))
        ours = detections_from_heads(
            heads,
            strides,
            conf_thr=conf_thr,
            iou_thr=iou_thr,
            scale=float(entry["scale"]),
            pad=(float(entry["pad"][0]), float(entry["pad"][1])),
            source_hw=source_hw,
            fix_points=(OUTPUT_FIX_POINT,) * len(heads),
        )
        reference_source = reference.copy()
        reference_source[:, :4] = boxes_to_source(
            reference[:, :4],
            float(entry["scale"]),
            (float(entry["pad"][0]), float(entry["pad"][1])),
            source_hw,
        )
        # Boxes that collapse against the frame edge (predictions into the
        # letterbox padding) have no usable area in source pixels: the runtime
        # drops zero-area ones, and sub-pixel IoU is unstable float noise.  Both
        # sides are filtered to the same minimum area before matching.
        reference_boxes = reference_source[:, :4]
        reference_area = (reference_boxes[:, 2] - reference_boxes[:, 0]) * (
            reference_boxes[:, 3] - reference_boxes[:, 1]
        )
        reference_source = reference_source[reference_area >= COMPARISON_MIN_AREA]
        comparable = [
            detection
            for detection in ours
            if (detection.box[2] - detection.box[0]) * (detection.box[3] - detection.box[1])
            >= COMPARISON_MIN_AREA
        ]
        pairs, extras, missing = _match(comparable, reference_source, min_iou)
        matched_total += len(pairs)
        image_min_iou = min((pair["iou"] for pair in pairs), default=None)
        if image_min_iou is not None:
            min_iou_seen = min(min_iou_seen, image_min_iou)
        ok = (
            preprocess_identical
            and not extras
            and not missing
            and len(pairs) == len(reference_source)
            and len(pairs) == len(comparable)
        )
        if not ok:
            failures.append(entry["name"])
        results.append(
            {
                "image": entry["name"],
                "preprocess_identical": preprocess_identical,
                "reference": int(len(reference_source)),
                "reference_degenerate": int(len(reference) - len(reference_source)),
                "ours": len(comparable),
                "ours_subpixel": len(ours) - len(comparable),
                "matched": len(pairs),
                "min_iou": image_min_iou,
                "extras": extras,
                "missing": missing,
                "ok": ok,
                "reference_boxes": [
                    {
                        "box": [round(float(value), 2) for value in row[:4]],
                        "score": round(float(row[4]), 4),
                        "class_id": int(row[5]),
                    }
                    for row in reference_source
                ],
                "our_boxes": [
                    {
                        "box": [round(value, 2) for value in det.box],
                        "score": round(det.score, 4),
                        "class_id": det.class_id,
                    }
                    for det in comparable
                ],
            }
        )
        print(
            f"check: {entry['name']} reference={len(reference_source)} "
            f"(+{len(reference) - len(reference_source)} collapsed) "
            f"ours={len(comparable)} matched={len(pairs)} "
            f"min_iou={image_min_iou} ok={ok}"
        )

    report = {
        "model": dump_info.get("model"),
        "weights": dump_info.get("weights"),
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
        "min_iou": min_iou,
        "input_fix_point": INPUT_FIX_POINT,
        "output_fix_point": OUTPUT_FIX_POINT,
        "comparison_min_area": COMPARISON_MIN_AREA,
        "images": len(results),
        "matched_boxes": matched_total,
        "min_matched_iou": None if matched_total == 0 else round(min_iou_seen, 6),
        "failures": failures,
        "passed": not failures,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="letterbox images on the host")
    prepare_parser.add_argument("--images", type=Path, required=True)
    prepare_parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    prepare_parser.add_argument("--limit", type=int, default=None)

    dump_parser = subparsers.add_parser("dump", help="run the package model in the container")
    dump_parser.add_argument("--model", type=Path, required=True)
    dump_parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    dump_parser.add_argument("--conf-thr", type=float, default=0.35)
    dump_parser.add_argument("--iou-thr", type=float, default=0.45)
    dump_parser.add_argument("--limit", type=int, default=None)

    check_parser = subparsers.add_parser("check", help="compare decodes and write the report")
    check_parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    check_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    check_parser.add_argument("--conf-thr", type=float, default=0.35)
    check_parser.add_argument("--iou-thr", type=float, default=0.45)
    check_parser.add_argument("--min-iou", type=float, default=0.9)

    args = parser.parse_args(argv)
    if args.mode == "prepare":
        meta = prepare(args.images, args.work, args.limit)
        print(f"prepare: {len(meta['images'])} images in {args.work / 'inputs'}")
        return 0
    if args.mode == "dump":
        dump(args.model, args.work, args.conf_thr, args.iou_thr, args.limit)
        return 0
    report = check(args.work, args.out, args.conf_thr, args.iou_thr, args.min_iou)
    print(
        f"check: {report['images']} images, {report['matched_boxes']} matched boxes, "
        f"min IoU {report['min_matched_iou']}, failures {len(report['failures'])}"
    )
    print(f"check: report written to {args.out}")
    if not report["passed"]:
        print(f"check: FAILED on {', '.join(report['failures'])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
