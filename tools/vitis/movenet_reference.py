#!/usr/bin/env python3
"""MoveNet reference check for the runtime decode (M1, AC-E1).

The board runs the compiled ``movenet.xmodel`` on the DPU; this tool checks the
CPU half of that pipeline against the Vitis AI model-zoo package itself on real
images with people.  It has three modes, because the reference model needs
PyTorch and ``pytorch_nndct`` (the Vitis AI container, Python 3.8) while
``cpsa`` runs on the host:

1. ``prepare`` (host, ``uv run``): resize every image exactly as
   :func:`cpsa.movenet.preprocess` does (RGB, square resize to ``192x192``, no
   letterbox) and quantize it with the runtime LUT (``pixel / 128`` into the
   ``fix_point 6`` int8 grid).  Saves the resized RGB frame and the int8 model
   input.
2. ``dump`` (Vitis AI container, ``tools/vitis/run_vitis.sh``): run the
   package's quantized TorchScript model on the dequantized inputs, invert the
   ``nn.Sigmoid`` the package ends its heatmap and center heads with (the
   compiled graph exports those heads before the sigmoid), round-trip the four
   heads through the DPU output fix points, and decode them with the package's
   own ``movenetDecode``.  Saves the heads, the package keypoints and their
   scores.
3. ``check`` (host): re-run the preprocessing, assert it reproduces the saved
   frames and int8 inputs byte for byte, decode the same heads with
   :func:`cpsa.movenet.decode` and compare the keypoints the package scores at
   >= 0.3.  Requires at least 20 images with one comparable keypoint and a mean
   distance <= 2 px on the ``192`` grid.  Writes
   ``build/vitis/report/movenet_reference.json`` and exits non-zero on failure.

Example::

    uv run --no-sync python tools/vitis/movenet_reference.py prepare \\
        --images build/vitis/movenet_reference/images \\
        --work build/vitis/movenet_reference
    tools/vitis/run_vitis.sh python tools/vitis/movenet_reference.py dump \\
        --model build/models/zoo/movenet/pt_movenet_coco_192_192_0.5G_3.0 \\
        --work build/vitis/movenet_reference
    uv run --no-sync python tools/vitis/movenet_reference.py check \\
        --work build/vitis/movenet_reference \\
        --out build/vitis/report/movenet_reference.json

The reference decode is the package's ``lib/task/task_tools.py:movenetDecode``,
the very function ``src/cpsa/movenet.py`` mirrors, run on the same heads the
board decodes (both sides see the heads after the int8 round trip), so the
comparison measures the decode and not the quantization error.  ``movenetDecode``
returns no scores, so the tool reads each score off the package's heatmap at the
package's own peak cell and asserts that this peak reproduces the decode
coordinates; a drift there aborts the dump instead of producing a bogus score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_WORK = Path("build/vitis/movenet_reference")
DEFAULT_OUT = Path("build/vitis/report/movenet_reference.json")

#: ``fix_point`` of the compiled graph's tensors (``inspect_xmodel.py``).
INPUT_FIX_POINT = 6
HEAD_FIX_POINTS = {"heatmaps": 3, "centers": 3, "regs": 2, "offsets": 7}

#: The package decodes its heads in this order; so does the runtime.
HEAD_ORDER = ("heatmaps", "centers", "regs", "offsets")
HEAD_CHANNELS = {"heatmaps": 17, "centers": 1, "regs": 34, "offsets": 34}

#: The package drops heatmap values below this before searching for a peak.
HEATMAP_THRESHOLD = 0.1

#: Acceptance gates of AC-E1.
DEFAULT_SCORE_THRESHOLD = 0.3
DEFAULT_MAX_MEAN_DISTANCE = 2.0
DEFAULT_MIN_IMAGES = 20


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


def _load_meta(work: Path) -> dict[str, Any]:
    path = _meta_path(work)
    if not path.is_file():
        raise SystemExit(f"missing {path}; run 'prepare' first")
    return json.loads(path.read_text())


def _input_path(work: Path, stem: str) -> Path:
    return work / "inputs" / f"{stem}.npz"


def _reference_path(work: Path, stem: str) -> Path:
    return work / "reference" / f"{stem}.npz"


def _quantize_head(head: np.ndarray, fix_point: int) -> np.ndarray:
    """Round-trip one head through int8, exactly as the DPU does on its outputs."""

    scale = 2.0**fix_point
    return (np.clip(np.rint(head * scale), -128.0, 127.0) / scale).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-values.astype(np.float64)))).astype(np.float32)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """Invert the package's ``sigmoid``, keeping the values finite for the int8 grid."""

    clipped = np.clip(probabilities.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def prepare(images: Path, work: Path, limit: int | None) -> dict[str, Any]:
    """Resize and quantize the images with the runtime preprocessing."""

    from cpsa.dpu import input_quantization_lut_scaled
    from cpsa.movenet import (
        MOVENET_INPUT_SCALE,
        MOVENET_INPUT_SIZE,
        preprocess,
        preprocess_into,
    )

    lut = input_quantization_lut_scaled(INPUT_FIX_POINT, MOVENET_INPUT_SCALE)
    inputs = work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for path in _image_paths(images, limit):
        bgr = _read_bgr(path)
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        model_rgb = preprocess(rgb)
        quantized = np.empty((*MOVENET_INPUT_SIZE, 3), dtype=np.int8)
        preprocess_into(rgb, lut, quantized)
        stem = path.stem.replace(" ", "_")
        np.savez_compressed(_input_path(work, stem), rgb=model_rgb, input=quantized)
        entries.append(
            {
                "name": path.name,
                "stem": stem,
                "source": str(path.resolve()),
                "height": int(bgr.shape[0]),
                "width": int(bgr.shape[1]),
            }
        )
        print(f"prepare: {path.name} -> {stem}.npz")
    meta = {
        "input_size": list(MOVENET_INPUT_SIZE),
        "input_scale": MOVENET_INPUT_SCALE,
        "input_fix_point": INPUT_FIX_POINT,
        "images": entries,
    }
    _meta_path(work).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _load_package(model_dir: Path) -> Any:
    """Import the zoo package's quantized model and decode helpers.

    The zoo modules are written for the Vitis AI container: ``lib`` imports
    ``albumentations`` at module scope, ``pytorch_nndct`` has to be imported
    before the TorchScript graph is loaded, and ``lib/utils/utils.py`` loads its
    center-prior table from a path relative to the package's ``code`` directory,
    so the working directory changes.  Callers pass absolute paths.
    """

    import os
    import types

    import torch

    code = model_dir / "code"
    if not (code / "lib" / "task" / "task_tools.py").is_file():
        raise SystemExit(f"{code} not found; extract the zoo package first")
    sys.modules.setdefault("albumentations", types.ModuleType("albumentations"))
    sys.path.insert(0, str(code))
    os.chdir(code)

    # ``config.py`` runs ``argparse.parse_args()`` on ``sys.argv`` at import
    # time, so hide this tool's arguments while the package is imported.
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        import pytorch_nndct  # noqa: F401  (registers the vai:: custom ops)
        from lib.task.task_tools import movenetDecode  # type: ignore[import-not-found]
        from lib.utils.utils import maxPoint  # type: ignore[import-not-found]

        from config import cfg  # type: ignore[import-not-found]
    finally:
        sys.argv = argv

    weights = model_dir / "quantized" / "MoveNet_int.pt"
    if not weights.is_file():
        raise SystemExit(f"{weights} not found; extract the zoo package first")
    network = torch.jit.load(str(weights), map_location="cpu")
    network.eval()
    return {
        "network": network,
        "movenetDecode": movenetDecode,
        "maxPoint": maxPoint,
        "cfg": cfg,
        "weights": weights,
    }


def _package_pose(
    heads: dict[str, np.ndarray],
    movenet_decode: Any,
    max_point: Any,
    feature: int,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the heads with the package and return ``((17, 2) px, scores)``.

    ``movenetDecode`` only returns normalised coordinates and a ``-1`` flag, so
    the score of each keypoint is read from the package's heatmap at the peak
    cell the package's search picks.  The peak is re-derived here with the same
    arithmetic, and :func:`_assert_peaks_reproduce` checks it against the
    package decode before the score is trusted.
    """

    import torch

    tensors = [torch.from_numpy(heads[name][None].copy()) for name in HEAD_ORDER]
    decoded = np.asarray(movenet_decode(tensors, mode="output", img_size=img_size))
    xy = decoded.reshape(17, 2) * float(img_size)

    heatmaps = heads["heatmaps"].copy()
    heatmaps[heatmaps < HEATMAP_THRESHOLD] = 0.0
    regs = heads["regs"]
    offsets = heads["offsets"]
    center_x, center_y = max_point(heads["centers"])
    cx, cy = int(center_x[0, 0]), int(center_y[0, 0])
    columns = np.tile(np.arange(feature), (feature, 1))
    rows = columns.T
    scores = np.zeros(17, dtype=np.float64)
    for index in range(17):
        target_x = int(regs[index * 2, cy, cx] + 0.5) + cx
        target_y = int(regs[index * 2 + 1, cy, cx] + 0.5) + cy
        distance = np.sqrt((columns - target_x) ** 2 + (rows - target_y) ** 2) + 1.8
        weighted = heatmaps[index] / distance
        peak = int(np.argmax(weighted.reshape(-1)))
        peak_x = min(max(peak % feature, 0), feature - 1)
        peak_y = min(max(peak // feature, 0), feature - 1)
        scores[index] = float(heatmaps[index, peak_y, peak_x])
        if heatmaps[index, peak_y, peak_x] > 0.0:
            cell_x = (peak_x + float(offsets[index * 2, peak_y, peak_x])) / feature
            cell_y = (peak_y + float(offsets[index * 2 + 1, peak_y, peak_x])) / feature
            if abs(cell_x - decoded[0, index * 2]) > 1e-6:
                raise SystemExit(
                    f"score peak for keypoint {index} does not reproduce the package "
                    f"decode: x {cell_x} != {decoded[0, index * 2]}"
                )
            if abs(cell_y - decoded[0, index * 2 + 1]) > 1e-6:
                raise SystemExit(
                    f"score peak for keypoint {index} does not reproduce the package "
                    f"decode: y {cell_y} != {decoded[0, index * 2 + 1]}"
                )
    return xy, scores


def dump(model: Path, work: Path, limit: int | None) -> None:
    """Run the package model on the prepared inputs and save heads + package poses."""

    import torch

    work = work.resolve()
    model_argument = model
    model = model.resolve()
    meta = _load_meta(work)
    img_size = int(meta["input_size"][0])
    feature = img_size // 4
    package = _load_package(model)
    network = package["network"]
    reference = work / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    entries = meta["images"][:limit] if limit is not None else meta["images"]
    dumped: list[dict[str, Any]] = []
    for entry in entries:
        data = np.load(_input_path(work, entry["stem"]))
        dequantized = data["input"].astype(np.float32) * float(2.0**-INPUT_FIX_POINT)
        tensor = torch.from_numpy(np.ascontiguousarray(dequantized.transpose(2, 0, 1))[None])
        with torch.no_grad():
            raw = network(tensor)
        heads = [head[0].numpy().astype(np.float32) for head in raw]
        channels = [head.shape[0] for head in heads]
        expected = [HEAD_CHANNELS[name] for name in HEAD_ORDER]
        if channels != expected:
            raise SystemExit(f"package heads {channels} != expected {expected}")
        if any(head.shape[1:] != (feature, feature) for head in heads):
            shapes = [head.shape for head in heads]
            raise SystemExit(f"package heads are not {feature}x{feature}: {shapes}")

        # The package's TorchScript model ends its heatmap and center heads with
        # ``nn.Sigmoid``, while the compiled graph exports those two heads from
        # the DPU *before* the sigmoid (``inspect_xmodel.py`` shows the sigmoid
        # in a CPU subgraph).  Invert it to get the DPU domain, then round-trip
        # every head through the int8 grid the board actually decodes.
        dpu_heads = {}
        for index, name in enumerate(HEAD_ORDER):
            head = heads[index]
            if name in ("heatmaps", "centers"):
                head = _logit(head)
            dpu_heads[name] = _quantize_head(head, HEAD_FIX_POINTS[name])
        # The package decode expects probabilities again, i.e. exactly what its
        # own CPU subgraph produces from the quantized logits.
        decode_heads = {
            name: _sigmoid(dpu_heads[name]) if name in ("heatmaps", "centers") else dpu_heads[name]
            for name in HEAD_ORDER
        }
        xy, scores = _package_pose(
            decode_heads, package["movenetDecode"], package["maxPoint"], feature, img_size
        )
        valid = xy[:, 0] >= 0.0
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[valid, 0] = xy[valid, 0]
        keypoints[valid, 1] = xy[valid, 1]
        keypoints[valid, 2] = scores[valid]

        arrays: dict[str, Any] = {f"head_{name}": dpu_heads[name] for name in HEAD_ORDER}
        arrays["keypoints"] = keypoints
        arrays["valid"] = valid
        np.savez_compressed(_reference_path(work, entry["stem"]), **arrays)
        dumped.append(
            {
                "name": entry["name"],
                "stem": entry["stem"],
                "valid_keypoints": int(valid.sum()),
                "scored_keypoints": int((keypoints[:, 2] >= DEFAULT_SCORE_THRESHOLD).sum()),
            }
        )
        print(
            f"dump: {entry['name']} -> {int(valid.sum())} valid, "
            f"{int((keypoints[:, 2] >= DEFAULT_SCORE_THRESHOLD).sum())} scored"
        )
    (work / "dump.json").write_text(
        json.dumps(
            {
                "model": str(model_argument),
                "weights": "quantized/MoveNet_int.pt",
                "input_fix_point": INPUT_FIX_POINT,
                "head_fix_points": HEAD_FIX_POINTS,
                "heatmap_threshold": HEATMAP_THRESHOLD,
                "images": dumped,
            },
            indent=2,
        )
        + "\n"
    )


def check(
    work: Path,
    out: Path,
    score_threshold: float,
    max_mean_distance: float,
    min_images: int,
) -> dict[str, Any]:
    """Compare the runtime decode with the package keypoints and write the report."""

    from cpsa.dpu import input_quantization_lut_scaled
    from cpsa.movenet import (
        KEYPOINTS,
        MOVENET_INPUT_SCALE,
        MOVENET_INPUT_SIZE,
        MovenetHeads,
        decode,
        preprocess,
        preprocess_into,
    )

    lut = input_quantization_lut_scaled(INPUT_FIX_POINT, MOVENET_INPUT_SCALE)
    meta = _load_meta(work)
    dump_info_path = work / "dump.json"
    dump_info = json.loads(dump_info_path.read_text()) if dump_info_path.is_file() else {}
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    distances: list[float] = []
    images_with_keypoints = 0
    score_mismatch_total = 0
    for entry in meta["images"]:
        bgr = _read_bgr(Path(entry["source"]))
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        model_rgb = preprocess(rgb)
        quantized = np.empty((*MOVENET_INPUT_SIZE, 3), dtype=np.int8)
        preprocess_into(rgb, lut, quantized)
        saved = np.load(_input_path(work, entry["stem"]))
        preprocess_identical = bool(
            model_rgb.shape == saved["rgb"].shape and np.array_equal(model_rgb, saved["rgb"])
        )
        input_identical = bool(
            quantized.shape == saved["input"].shape and np.array_equal(quantized, saved["input"])
        )

        reference_arrays = np.load(_reference_path(work, entry["stem"]))
        heads = MovenetHeads(**{name: reference_arrays[f"head_{name}"] for name in HEAD_ORDER})
        pose = decode(heads, size=MOVENET_INPUT_SIZE)
        reference = reference_arrays["keypoints"]
        valid = reference_arrays["valid"]

        ours_kept = pose.keypoints[:, 2] >= score_threshold
        reference_kept = (reference[:, 2] >= score_threshold) & valid
        compared = ours_kept & reference_kept
        mismatch = [
            KEYPOINTS[index]
            for index in range(len(KEYPOINTS))
            if bool(ours_kept[index]) != bool(reference_kept[index])
        ]
        deltas = np.linalg.norm(
            reference[compared, :2].astype(np.float64)
            - pose.keypoints[compared, :2].astype(np.float64),
            axis=1,
        )
        image_mean = float(np.mean(deltas)) if len(deltas) else None
        image_max = float(np.max(deltas)) if len(deltas) else None
        worst_index = int(np.argmax(deltas)) if len(deltas) else None
        worst_name = (
            KEYPOINTS[int(np.flatnonzero(compared)[worst_index])]
            if worst_index is not None
            else None
        )
        distances.extend(float(value) for value in deltas)
        if len(deltas):
            images_with_keypoints += 1
        score_mismatch_total += len(mismatch)
        ok = preprocess_identical and input_identical and not mismatch
        if not ok:
            failures.append(entry["name"])
        results.append(
            {
                "image": entry["name"],
                "preprocess_identical": preprocess_identical,
                "input_identical": input_identical,
                "reference_valid": int(valid.sum()),
                "reference_keypoints": int(reference_kept.sum()),
                "compared": int(compared.sum()),
                "mean_distance_px": None if image_mean is None else round(image_mean, 4),
                "max_distance_px": None if image_max is None else round(image_max, 4),
                "worst_keypoint": worst_name,
                "score_mismatches": mismatch,
                "ok": ok,
            }
        )
        print(
            f"check: {entry['name']} reference={int(reference_kept.sum())} "
            f"compared={int(compared.sum())} mean={image_mean} max={image_max} "
            f"mismatches={len(mismatch)} ok={ok}"
        )

    mean_distance = float(np.mean(distances)) if distances else None
    problems: list[str] = []
    if score_mismatch_total:
        problems.append(f"{score_mismatch_total} keypoints disagree on the {score_threshold} score")
    if images_with_keypoints < min_images:
        problems.append(
            f"only {images_with_keypoints} images have a comparable keypoint, need {min_images}"
        )
    if mean_distance is None:
        problems.append("no comparable keypoints")
    elif mean_distance > max_mean_distance:
        problems.append(f"mean keypoint distance {mean_distance:.2f} px > {max_mean_distance} px")

    report = {
        "model": dump_info.get("model"),
        "weights": dump_info.get("weights"),
        "input_fix_point": INPUT_FIX_POINT,
        "head_fix_points": HEAD_FIX_POINTS,
        "input_scale": meta.get("input_scale"),
        "heatmap_threshold": HEATMAP_THRESHOLD,
        "score_threshold": score_threshold,
        "max_mean_distance_px": max_mean_distance,
        "min_images": min_images,
        "images": len(results),
        "images_with_keypoints": images_with_keypoints,
        "compared_keypoints": len(distances),
        "mean_distance_px": None if mean_distance is None else round(mean_distance, 4),
        "max_distance_px": None if not distances else round(float(np.max(distances)), 4),
        "score_mismatches": score_mismatch_total,
        "failures": failures,
        "problems": problems,
        "passed": not failures and not problems,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="resize and quantize the images")
    prepare_parser.add_argument("--images", type=Path, required=True)
    prepare_parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    prepare_parser.add_argument("--limit", type=int, default=None)

    dump_parser = subparsers.add_parser("dump", help="run the package model in the container")
    dump_parser.add_argument("--model", type=Path, required=True)
    dump_parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    dump_parser.add_argument("--limit", type=int, default=None)

    check_parser = subparsers.add_parser("check", help="compare decodes and write the report")
    check_parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    check_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    check_parser.add_argument("--score-thr", type=float, default=DEFAULT_SCORE_THRESHOLD)
    check_parser.add_argument("--max-mean-distance", type=float, default=DEFAULT_MAX_MEAN_DISTANCE)
    check_parser.add_argument("--min-images", type=int, default=DEFAULT_MIN_IMAGES)

    args = parser.parse_args(argv)
    if args.mode == "prepare":
        meta = prepare(args.images, args.work, args.limit)
        print(f"prepare: {len(meta['images'])} images in {args.work / 'inputs'}")
        return 0
    if args.mode == "dump":
        dump(args.model, args.work, args.limit)
        return 0
    report = check(args.work, args.out, args.score_thr, args.max_mean_distance, args.min_images)
    print(
        f"check: {report['images']} images, {report['images_with_keypoints']} with a "
        f"comparable keypoint, {report['compared_keypoints']} keypoints, "
        f"mean {report['mean_distance_px']} px, max {report['max_distance_px']} px"
    )
    print(f"check: report written to {args.out}")
    if not report["passed"]:
        details = report["failures"] + report["problems"]
        print(f"check: FAILED: {', '.join(details)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
