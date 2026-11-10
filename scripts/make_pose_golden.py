#!/usr/bin/env python3
"""Generate the independent YOLOv8-pose raw-head decode fixture (P-10).

Run with the ML environment that pins Ultralytics 8.4.203::

    ml/.venv/bin/python scripts/make_pose_golden.py

Random raw head tensors (box DFL logits, class logits, keypoint logits) are
pushed through the upstream ``ultralytics.nn.modules.head.Pose`` decode.  The
raw inputs and the upstream outputs are written to
``tests/fixtures/pose/yolov8_pose_raw_golden.npz``; the unit test loads only
the ``.npz`` so the root environment needs no Ultralytics or Torch.

The confidence threshold stored per case is the midpoint between the highest
and second-highest class scores, so exactly one candidate survives.  That makes
the golden comparison independent of NMS tie-breaking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SEED = 20261109
INPUT_HW = (96, 96)
STRIDES = (8, 16, 32)
REG_MAX = 16
N_CLASS = 1
CASES = (17, 62)
UPSTREAM_VERSION = "8.4.203"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "pose"
        / "yolov8_pose_raw_golden.npz",
    )
    return parser.parse_args()


def build_case(
    rng: np.random.Generator,
    n_kpt: int,
) -> dict[str, np.ndarray]:
    import torch
    from ultralytics.nn.modules.head import Pose

    feats = []
    raw_per_scale: list[np.ndarray] = []
    boxes = []
    scores = []
    keypoints = []
    for stride in STRIDES:
        height = INPUT_HW[0] // stride
        width = INPUT_HW[1] // stride
        box = rng.standard_normal((1, 4 * REG_MAX, height * width)).astype(np.float32)
        score = rng.standard_normal((1, N_CLASS, height * width)).astype(np.float32)
        keypoint = rng.standard_normal((1, 3 * n_kpt, height * width)).astype(np.float32)
        boxes.append(torch.from_numpy(box))
        scores.append(torch.from_numpy(score))
        keypoints.append(torch.from_numpy(keypoint))
        feats.append(torch.zeros(1, 8, height, width))
        concatenated = np.concatenate([box, score, keypoint], axis=1)
        raw_per_scale.append(
            concatenated.reshape(1, -1, height, width).transpose(0, 2, 3, 1).copy()
        )

    head = Pose(nc=N_CLASS, kpt_shape=(n_kpt, 3), reg_max=REG_MAX, ch=[8] * len(STRIDES))
    head.stride = torch.tensor(STRIDES, dtype=torch.float32)
    head.export = True
    with torch.no_grad():
        decoded = head._inference(
            {
                "boxes": torch.cat(boxes, dim=2),
                "scores": torch.cat(scores, dim=2),
                "kpts": torch.cat(keypoints, dim=2),
                "feats": feats,
            }
        ).numpy()

    class_scores = decoded[0, 4 : 4 + N_CLASS].max(axis=0)
    order = np.argsort(-class_scores, kind="stable")
    top = int(order[0])
    second = float(class_scores[order[1]]) if order.size > 1 else 0.0
    conf_thr = (float(class_scores[top]) + second) / 2.0
    case: dict[str, np.ndarray] = {
        "strides": np.asarray(STRIDES, dtype=np.int32),
        "n_kpt": np.asarray(n_kpt, dtype=np.int32),
        "reg_max": np.asarray(REG_MAX, dtype=np.int32),
        "conf_thr": np.asarray(conf_thr, dtype=np.float64),
        "iou_thr": np.asarray(0.5, dtype=np.float64),
        "input_hw": np.asarray(INPUT_HW, dtype=np.int32),
        "decoded": decoded.astype(np.float32),
        "top": np.asarray(top, dtype=np.int32),
    }
    for index, raw in enumerate(raw_per_scale):
        case[f"raw_{index}"] = raw.astype(np.float32)
    return case


def main() -> int:
    args = parse_args()
    try:
        import torch  # noqa: F401  (availability check only)
        import ultralytics
    except ImportError as exc:
        print(
            "Ultralytics/Torch are not importable; run this script with the ML "
            f"environment (ml/.venv/bin/python): {exc}",
            file=sys.stderr,
        )
        return 1
    version = getattr(ultralytics, "__version__", "unknown")
    if version != UPSTREAM_VERSION:
        print(
            f"warning: Ultralytics {version} differs from the pinned {UPSTREAM_VERSION}",
            file=sys.stderr,
        )

    rng = np.random.default_rng(args.seed)
    payload: dict[str, np.ndarray] = {
        "version": np.asarray(version),
        "seed": np.asarray(args.seed, dtype=np.int64),
    }
    for n_kpt in CASES:
        case = build_case(rng, n_kpt)
        prefix = f"kpt{n_kpt}"
        for key, value in case.items():
            payload[f"{prefix}_{key}"] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
