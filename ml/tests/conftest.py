"""Synthetic COCO-WholeBody fixture for the ml tests (no downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

BODY_POINTS = {
    0: (320, 150),
    1: (310, 145),
    2: (330, 145),
    3: (300, 140),
    4: (340, 140),
    5: (280, 200),
    6: (360, 200),
    7: (270, 250),
    8: (370, 250),
    9: (260, 300),
    10: (380, 300),
    11: (300, 340),
    12: (340, 340),
    13: (300, 400),
    14: (340, 400),
    15: (300, 460),
    16: (340, 460),
}

# iBUG indices used by HALFBUST_62, in a plausible face layout.
FACE_POINTS = {
    0: (300, 190),
    16: (340, 190),
    8: (320, 205),
    19: (310, 160),
    24: (330, 160),
    48: (312, 195),
    54: (328, 195),
    51: (320, 192),
    57: (320, 200),
}


def _array(points: dict[int, tuple[float, float]], count: int, dx: float = 0.0) -> list[float]:
    values = np.zeros((count, 3), dtype=np.float32)
    for index, (x, y) in points.items():
        values[index] = (x + dx, y, 2.0)
    return values.reshape(-1).tolist()


def _hand_points(dx: float, base_x: float) -> dict[int, tuple[float, float]]:
    points: dict[int, tuple[float, float]] = {}
    for index in range(21):
        points[index] = (base_x + dx + (index % 5) * 8, 290 + (index // 5) * 8)
    return points


def _annotation(
    annotation_id: int,
    image_id: int,
    dx: float = 0.0,
    outside_wrists: bool = False,
) -> dict[str, object]:
    body = {index: (x + dx, y) for index, (x, y) in BODY_POINTS.items()}
    if outside_wrists:
        body[9] = (5.0, 470.0)
        body[10] = (5.0, 475.0)
    xs = [point[0] for point in body.values()]
    ys = [point[1] for point in body.values()]
    x1, y1 = min(xs) - 30, min(ys) - 30
    x2, y2 = max(xs) + 30, max(ys) + 30
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "area": (x2 - x1) * (y2 - y1),
        "num_keypoints": 133,
        "keypoints": _array(body, 17),
        "foot_kpts": _array({}, 6),
        "foot_valid": False,
        "face_kpts": _array(FACE_POINTS, 68, dx),
        "face_valid": True,
        "lefthand_kpts": _array(_hand_points(dx, 245), 21),
        "lefthand_valid": True,
        "righthand_kpts": _array(_hand_points(dx, 385), 21),
        "righthand_valid": True,
    }


def _draw_image(path: Path, annotations: list[dict[str, object]]) -> None:
    image = Image.new("RGB", (640, 480), (35, 38, 44))
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        keypoints = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(17, 3)
        for _, (x, y, visibility) in enumerate(keypoints):
            if visibility > 0:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(240, 210, 170))
        nose = keypoints[0]
        draw.ellipse((nose[0] - 40, nose[1] - 45, nose[0] + 40, nose[1] + 45), fill=(220, 180, 140))
        draw.line(
            (keypoints[5][0], keypoints[5][1], keypoints[6][0], keypoints[6][1]),
            fill=(90, 160, 220),
            width=12,
        )
        draw.line(
            (keypoints[5][0], keypoints[5][1], keypoints[9][0], keypoints[9][1]),
            fill=(90, 160, 220),
            width=8,
        )
        draw.line(
            (keypoints[6][0], keypoints[6][1], keypoints[10][0], keypoints[10][1]),
            fill=(90, 160, 220),
            width=8,
        )
    image.save(path, format="JPEG", quality=95)


@pytest.fixture()
def synthetic_coco(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(coco_root, wholebody_json)`` for three synthetic images."""

    root = tmp_path / "coco"
    train_dir = root / "train2017"
    val_dir = root / "val2017"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    image_specs = [
        (1, "000000000001.jpg", [_annotation(1, 1), _annotation(2, 1, dx=80)]),
        (2, "000000000002.jpg", [_annotation(3, 2)]),
        (3, "000000000003.jpg", [_annotation(4, 3, outside_wrists=True)]),
    ]
    images = []
    annotations = []
    for image_id, file_name, image_annotations in image_specs:
        _draw_image(train_dir / file_name, image_annotations)
        (val_dir / file_name).write_bytes((train_dir / file_name).read_bytes())
        images.append({"id": image_id, "file_name": file_name, "width": 640, "height": 480})
        annotations.extend(image_annotations)

    payload = {
        "info": {"description": "synthetic COCO-WholeBody fixture"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }
    json_path = tmp_path / "coco_wholebody_train_v1.0.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return root, json_path
