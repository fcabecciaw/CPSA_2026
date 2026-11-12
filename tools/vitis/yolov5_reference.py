#!/usr/bin/env python3
"""YOLOv5-nano reference check for the showcase detector (Y5).

The board runs the compiled ``yolov5_nano.xmodel`` on the DPU; this tool checks
the CPU half of that pipeline against the Vitis AI model-zoo post-processing on
real images.  It has three modes, because the reference model needs PyTorch (the
Vitis AI container, Python 3.8) while ``cpsa`` runs on the host:

1. ``prepare`` (host, ``uv run``): letterbox every input image exactly as
   :func:`cpsa.yolov5.prepare_input` does (RGB, pad 114) and save the model
   inputs plus the geometry (``scale``, ``pad``, source size).
2. ``dump`` (Vitis AI container, ``tools/vitis/run_vitis.sh``): run the reference
   YOLOv5n forward pass on those saved inputs, take the three raw head tensors
   (the ``Detect`` convolutions, before sigmoid), round-trip them through the
   DPU output ``fix_point`` of each head, and post-process them with a literal
   port of the zoo's own C++ post-processing
   (``vitis::ai::yolov3_post_process`` with ``type: YOLOV5`` from
   ``src/vai_library/xnnpp/src/yolov3.cpp`` at Vitis-AI v3.0), including its
   per-class NMS.  Saves the float heads with the resulting detections.
3. ``check`` (host): re-run the preprocessing, assert it reproduces the saved
   inputs pixel for pixel, decode the saved heads with
   :func:`cpsa.yolov5.detections_from_heads` and compare the boxes against the
   zoo post-processing: same class, IoU >= 0.9, no extra or missing boxes.  Writes
   ``build/vitis/report/yolov5_reference.json`` and exits non-zero on any
   mismatch.

Example::

    uv run --no-sync python tools/vitis/yolov5_reference.py prepare \\
        --images build/vitis/yolov5_reference/images --work build/vitis/yolov5_reference
    tools/vitis/run_vitis.sh python tools/vitis/yolov5_reference.py dump \\
        --work build/vitis/yolov5_reference
    uv run --no-sync python tools/vitis/yolov5_reference.py check \\
        --work build/vitis/yolov5_reference \\
        --out build/vitis/report/yolov5_reference.json

The reference network is the public Ultralytics YOLOv5n v7.0 checkpoint, the
float architecture the zoo entry was converted and quantized from (the zoo
README reports ``FLOAT AP 27.0`` for it).  The AMD download of the zoo's
float+QAT package now returns a README-only archive (the YOLO packages are no
longer published there), so the QAT weights themselves are not available; the
decode under test is independent of them, because both sides decode the very
same head tensors.  ``dump`` asserts that the reference model has the zoo's
strides, 85 channels per anchor and the prototxt's anchors, so the head layout
it exercises is the xmodel's.  The heads are rounded through the output
``fix_point`` reported by ``inspect_xmodel.py`` (3, 4, 4), which is what the
board decodes, and the input is rounded through the input ``fix_point`` (6) after
the zoo's ``1 / 255`` normalisation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

#: The container image runs this file with the system interpreter, which has no
#: editable install of ``cpsa``; point it at the worktree's source tree.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_WORK = Path("build/vitis/yolov5_reference")
DEFAULT_OUT = Path("build/vitis/report/yolov5_reference.json")
DEFAULT_CODE = Path("build/third_party/yolov5-7.0")
DEFAULT_WEIGHTS = Path("build/third_party/yolov5n.pt")

#: Input/output ``fix_point`` of the compiled graph (``inspect_xmodel.py``).
INPUT_FIX_POINT = 6
OUTPUT_FIX_POINTS = (3, 4, 4)

#: Minimum number of images the comparison has to cover.
MIN_IMAGES = 20

#: Boxes smaller than this (source pixels squared) carry no usable information:
#: predictions into the letterbox padding collapse against the frame edge, and
#: their IoU is unstable.  They are excluded from the comparison on both sides.
COMPARISON_MIN_AREA = 1.0

#: Class ids used to report how many images really show people and objects.
PERSON_CLASS = 0


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


def _quantize_units(values: np.ndarray, fix_point: int) -> np.ndarray:
    """Round normalised ``0..1`` samples through the xint8 input grid."""

    scale = 2.0**fix_point
    return np.clip(np.rint(values * scale), -128.0, 127.0) / scale


def _quantize_head(head: np.ndarray, fix_point: int) -> np.ndarray:
    """Round-trip one head through int8, exactly as the DPU does on its outputs."""

    scale = 2.0**fix_point
    return np.clip(np.rint(head * scale), -128.0, 127.0) / scale


def prepare(images: Path, work: Path, limit: int | None) -> dict[str, Any]:
    """Letterbox the images with the runtime preprocessing and save the model inputs."""

    import cv2

    from cpsa.yolov5 import YOLOV5_INPUT_SIZE, prepare_input

    inputs = work / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for path in _image_paths(images, limit):
        bgr = _read_bgr(path)
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        letterboxed, scale, pad = prepare_input(rgb, YOLOV5_INPUT_SIZE)
        stem = path.stem.replace(" ", "_")
        target = inputs / f"{stem}.png"
        # ``prepare_input`` keeps the RGB order; PNG + imread is BGR, so store
        # the frame swapped and let ``dump`` swap it back, unchanged.
        if not cv2.imwrite(str(target), np.ascontiguousarray(letterboxed[..., ::-1])):
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
    meta = {"input_size": list(YOLOV5_INPUT_SIZE), "images": entries}
    _meta_path(work).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _allow_missing_seaborn() -> None:
    """Make ``utils.plots`` importable in the Vitis AI image.

    YOLOv5 v7.0 imports seaborn at module scope for its plotting helpers, which
    pull a dependency the pinned image does not ship.  The reference only needs
    the model graph and never plots, so an empty module stands in rather than
    changing the container image.
    """

    try:
        import seaborn  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["seaborn"] = types.ModuleType("seaborn")


def _load_reference_model(code: Path, weights: Path) -> Any:
    """Instantiate the Ultralytics YOLOv5n checkpoint in raw-head mode."""

    import torch

    if not code.is_dir():
        raise SystemExit(f"{code} not found; unpack the Ultralytics YOLOv5 v7.0 tree there")
    if not weights.is_file():
        raise SystemExit(f"{weights} not found; download the YOLOv5n checkpoint first")
    sys.path.insert(0, str(code.resolve()))
    _allow_missing_seaborn()
    checkpoint = torch.load(str(weights), map_location="cpu")
    # The checkpoint holds the trained ``DetectionModel`` itself; ``attempt_load``
    # would wrap it in an ensemble that decodes the heads, which is exactly what
    # this tool needs to avoid.
    model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
    # PyTorch 2.0 removed ``recompute_scale_factor`` from ``Upsample.forward``
    # but v7.0's saved modules still call it; torch 1.13 in the image needs the
    # attribute present (``None`` is the default behaviour).
    for module in model.modules():
        if isinstance(module, torch.nn.Upsample):
            module.recompute_scale_factor = None
    detect = model.model[-1]
    # The reference needs the raw head convolutions, not the decoded boxes:
    # YOLOv5's ``Detect`` returns them when it believes it is training.
    detect.training = True
    detect.export = False
    return model


def _assert_head_layout(model: Any) -> tuple[int, ...]:
    """Check that the reference head is the zoo head and return its strides."""

    from cpsa.yolov5 import YOLOV5_ANCHORS

    detect = model.model[-1]
    strides = tuple(int(value) for value in detect.stride)
    if strides != tuple(sorted(YOLOV5_ANCHORS)):
        raise SystemExit(f"reference strides {strides} are not the zoo's {sorted(YOLOV5_ANCHORS)}")
    if int(detect.no) != 85 or int(detect.na) != 3:
        raise SystemExit(f"reference head has {detect.na} anchors of {detect.no}, expected 3 of 85")
    for index, stride in enumerate(strides):
        # ``Detect.anchors`` is stored relative to the stride.
        anchors = detect.anchors[index].numpy() * stride
        expected = np.asarray(YOLOV5_ANCHORS[stride], dtype=np.float32)
        if not np.allclose(anchors, expected, atol=1e-3):
            raise SystemExit(
                f"reference anchors at stride {stride} are {anchors.tolist()}, "
                f"the zoo prototxt has {expected.tolist()}"
            )
    return strides


def _sigmoid(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-values))


def _iou_centre(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one normalised ``(cx, cy, w, h)`` box against many, as the zoo does."""

    left = np.maximum(box[0] - box[2] / 2.0, boxes[:, 0] - boxes[:, 2] / 2.0)
    right = np.minimum(box[0] + box[2] / 2.0, boxes[:, 0] + boxes[:, 2] / 2.0)
    top = np.maximum(box[1] - box[3] / 2.0, boxes[:, 1] - boxes[:, 3] / 2.0)
    bottom = np.minimum(box[1] + box[3] / 2.0, boxes[:, 1] + boxes[:, 3] / 2.0)
    width = np.maximum(right - left, 0.0)
    height = np.maximum(bottom - top, 0.0)
    intersection = width * height
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0.0, intersection / union, 0.0)


def _apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    nms_thr: float,
    conf_thr: float,
) -> list[int]:
    """Greedy NMS of ``vitis::ai::applyNMS``: stable, ``score < conf`` skipped.

    The zoo calls it once per class, over the boxes of *all* classes, which is
    what makes its post-processing multi-label.
    """

    order = np.argsort(-scores, kind="stable")
    alive = np.ones(scores.size, dtype=np.bool_)
    kept: list[int] = []
    for position, index in enumerate(order):
        if not alive[index]:
            continue
        if scores[index] < conf_thr:
            alive[index] = False
            continue
        kept.append(int(index))
        overlap = _iou_centre(boxes[index], boxes[order[position + 1 :]])
        alive[order[position + 1 :]] &= overlap < nms_thr
    return kept


def _package_postprocess(
    heads: list[np.ndarray],
    strides: list[int],
    input_size: tuple[int, int],
    conf_thr: float,
    iou_thr: float,
) -> np.ndarray:
    """Apply the zoo's YOLOv5 post-processing to raw ``(H, W, 255)`` heads.

    A literal port of ``detect()`` and the per-class ``applyNMS`` of
    ``src/vai_library/xnnpp/src/yolov3.cpp`` (Vitis-AI v3.0) for
    ``type: YOLOV5``, with the anchors of ``yolov5_nano_pt.prototxt``.  The
    decoders are deliberately written differently from
    :func:`cpsa.yolov5.decode_yolov5` (a scalar loop over the zoo's index
    arithmetic) so the comparison exercises two independent implementations.

    Returns:
        ``(N, 6)`` rows ``x1, y1, x2, y2, score, class_id`` in input pixels.
    """

    from cpsa.yolov5 import YOLOV5_ANCHORS

    num_classes = 80
    boxes_per_class = 5 + num_classes
    input_height, input_width = input_size
    # ``conf_desigmoid``: the zoo skips a slot before the sigmoid when even a
    # perfect class probability could not reach the threshold.
    conf_desigmoid = -math.log(1.0 / conf_thr - 1.0) if 0.0 < conf_thr < 1.0 else -math.inf

    if len(heads) != len(strides):
        raise ValueError(f"{len(heads)} heads for {len(strides)} strides")

    candidates: list[list[float]] = []
    # Python 3.8 in the container has no ``zip(strict=...)``; the length check above
    # is the container-compatible equivalent.
    for head, stride in zip(heads, strides):  # noqa: B905
        anchors = YOLOV5_ANCHORS[stride]
        height, width = head.shape[:2]
        flat = head.reshape(height * width, len(anchors), boxes_per_class)
        for cell in range(height * width):
            row, column = divmod(cell, width)
            for anchor, slot in enumerate(flat[cell]):
                if slot[4] < conf_desigmoid:
                    continue
                objectness = float(_sigmoid(slot[4]))
                candidate = [
                    (float(_sigmoid(slot[0])) * 2.0 - 0.5 + column) / width,
                    (float(_sigmoid(slot[1])) * 2.0 - 0.5 + row) / height,
                    (float(_sigmoid(slot[2])) * 2.0) ** 2 * anchors[anchor][0] / input_width,
                    (float(_sigmoid(slot[3])) * 2.0) ** 2 * anchors[anchor][1] / input_height,
                    -1.0,
                    objectness,
                ]
                candidate.extend((objectness * _sigmoid(slot[5:])).tolist())
                candidates.append(candidate)

    if not candidates:
        return np.zeros((0, 6), dtype=np.float64)
    matrix = np.asarray(candidates, dtype=np.float64)
    centres = matrix[:, :4]
    per_class = matrix[:, 6:]
    rows: list[list[float]] = []
    for class_id in range(num_classes):
        scores = per_class[:, class_id]
        for index in _apply_nms(centres, scores, iou_thr, conf_thr):
            if scores[index] <= conf_thr:
                continue
            centre_x, centre_y, width, height = centres[index]
            rows.append(
                [
                    (centre_x - width / 2.0) * input_width,
                    (centre_y - height / 2.0) * input_height,
                    (centre_x + width / 2.0) * input_width,
                    (centre_y + height / 2.0) * input_height,
                    float(scores[index]),
                    float(class_id),
                ]
            )
    return np.asarray(rows, dtype=np.float64).reshape(-1, 6)


def dump(
    work: Path,
    code: Path,
    weights: Path,
    conf_thr: float,
    iou_thr: float,
    limit: int | None,
) -> None:
    """Run the reference model on the prepared inputs and save heads + detections."""

    import torch

    meta = _load_meta(work)
    input_size = (int(meta["input_size"][0]), int(meta["input_size"][1]))
    model = _load_reference_model(code, weights)
    strides = _assert_head_layout(model)
    reference = work / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    entries = meta["images"][:limit] if limit is not None else meta["images"]
    dumped: list[dict[str, Any]] = []
    for entry in entries:
        # The PNG holds the letterboxed frame with the channels swapped, so a
        # plain read returns the model's RGB order.
        bgr = _read_bgr(work / "inputs" / f"{entry['stem']}.png")
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        # The DPU input is xint8 with this fix point, so the graph sees the
        # frame on that grey grid; round the normalised frame before the pass.
        units = _quantize_units(rgb.astype(np.float64) * (1.0 / 255.0), INPUT_FIX_POINT)
        tensor = torch.from_numpy(units.astype(np.float32).transpose(2, 0, 1)[None])
        with torch.no_grad():
            raw = model(tensor)
        # ``Detect`` in v7.0 returns ``(1, anchors, H, W, 85)`` per level; the DPU
        # returns the same values flattened anchor-major into ``(1, H, W, 255)``
        # channels, which is the runtime's head layout.
        raw = [level[0].numpy().astype(np.float32) for level in raw]
        if len(raw) != len(OUTPUT_FIX_POINTS):
            raise SystemExit(f"the reference model returned {len(raw)} heads")
        # The DPU returns int8 heads with these fix points; round-trip them here
        # so both the zoo post-processing and the runtime decode see exactly the
        # values the board decodes.
        nhwc = []
        # The head count is checked just above; ``strict=`` needs Python 3.10 and
        # the container runs 3.8.
        for head, fix_point in zip(raw, OUTPUT_FIX_POINTS):  # noqa: B905
            cells = np.ascontiguousarray(head.transpose(1, 2, 0, 3))
            nhwc.append(_quantize_head(cells.reshape(*head.shape[1:3], -1), fix_point))
        actual = [input_size[0] // head.shape[0] for head in nhwc]
        if actual != list(strides):
            raise SystemExit(f"head shapes {[head.shape for head in nhwc]} give strides {actual}")
        detections = _package_postprocess(nhwc, actual, input_size, conf_thr, iou_thr)
        arrays: dict[str, Any] = {"strides": np.asarray(actual, dtype=np.int64)}
        arrays.update({f"head_{index}": head for index, head in enumerate(nhwc)})
        arrays["detections"] = detections
        np.savez_compressed(reference / f"{entry['stem']}.npz", **arrays)
        dumped.append(
            {"name": entry["name"], "stem": entry["stem"], "detections": int(len(detections))}
        )
        print(f"dump: {entry['name']} -> {len(detections)} zoo detections")
    (work / "dump.json").write_text(
        json.dumps(
            {
                "code": str(code),
                "weights": str(weights),
                "input_fix_point": INPUT_FIX_POINT,
                "output_fix_points": list(OUTPUT_FIX_POINTS),
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
    """Compare the runtime decode with the zoo detections and write the report."""

    from cpsa.yolov5 import (
        YOLOV5_INPUT_SIZE,
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
    person_and_object = 0
    for entry in meta["images"]:
        bgr = _read_bgr(Path(entry["source"]))
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        letterboxed, scale, pad = prepare_input(rgb, YOLOV5_INPUT_SIZE)
        saved = _read_bgr(work / "inputs" / f"{entry['stem']}.png")
        preprocess_identical = bool(
            letterboxed.shape == saved.shape
            and np.array_equal(letterboxed, np.ascontiguousarray(saved[..., ::-1]))
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
            fix_points=OUTPUT_FIX_POINTS,
        )
        reference_source = reference.copy()
        if reference_source.size:
            reference_source[:, :4] = boxes_to_source(
                reference_source[:, :4],
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
        classes = {int(row[5]) for row in reference_source}
        if PERSON_CLASS in classes and len(classes - {PERSON_CLASS}) > 0:
            person_and_object += 1
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
        "model": dump_info.get("weights"),
        "reference_code": dump_info.get("code"),
        "input_size": list(YOLOV5_INPUT_SIZE),
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
        "min_iou": min_iou,
        "input_fix_point": dump_info.get("input_fix_point"),
        "output_fix_points": dump_info.get("output_fix_points"),
        "comparison_min_area": COMPARISON_MIN_AREA,
        "images": len(results),
        "images_with_person_and_object": person_and_object,
        "matched_boxes": matched_total,
        "min_matched_iou": None if matched_total == 0 else round(min_iou_seen, 9),
        "failures": failures,
        "passed": not failures and len(results) >= MIN_IMAGES and matched_total > 0,
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

    dump_parser = subparsers.add_parser("dump", help="run the reference model in the container")
    dump_parser.add_argument(
        "--code", type=Path, default=DEFAULT_CODE, help="Ultralytics YOLOv5 v7.0 tree"
    )
    dump_parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLOv5n float checkpoint"
    )
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
        dump(args.work, args.code, args.weights, args.conf_thr, args.iou_thr, args.limit)
        return 0
    report = check(args.work, args.out, args.conf_thr, args.iou_thr, args.min_iou)
    print(
        f"check: {report['images']} images "
        f"({report['images_with_person_and_object']} with a person and an object), "
        f"{report['matched_boxes']} matched boxes, min IoU {report['min_matched_iou']}, "
        f"failures {len(report['failures'])}"
    )
    print(f"check: report written to {args.out}")
    if not report["passed"]:
        print(
            f"check: FAILED on {', '.join(report['failures']) or 'the image count'}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
