"""The per-schema OKS sigmas that feed Ultralytics' keypoint loss.

A ``kpt_shape`` other than ``[17, 3]`` makes Ultralytics' ``v8PoseLoss`` fall
back to ``ones(nkpt) / nkpt`` unless the dataset YAML declares
``kpt_oks_sigmas``; with 1/62 the OKS loss saturates and far-away joints never
converge.  These tests pin the COCO-WholeBody source table, the derivation and
the YAML round-trip up to the loss object the trainer builds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cpsa_ml.schema import (
    COCO17,
    COCO_WHOLEBODY_SIGMAS,
    HALFBUST_62,
    HAND_21,
    OKS_SIGMA_FLOOR,
)


def test_wholebody_sigma_table_is_the_authoritative_133() -> None:
    assert len(COCO_WHOLEBODY_SIGMAS) == 133
    # Body block 0-16: the COCO-17 sigmas Ultralytics' OKS_SIGMA uses / 10.
    assert COCO_WHOLEBODY_SIGMAS[:17] == (
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    )
    # Spot values from mmpose configs/_base_/datasets/coco_wholebody.py:
    # face-0, left hand root, right hand root and the last right-hand point.
    assert COCO_WHOLEBODY_SIGMAS[23] == pytest.approx(0.042)
    assert COCO_WHOLEBODY_SIGMAS[91] == pytest.approx(0.029)
    assert COCO_WHOLEBODY_SIGMAS[112] == pytest.approx(0.029)
    assert COCO_WHOLEBODY_SIGMAS[132] == pytest.approx(0.031)


@pytest.mark.parametrize("schema", [HALFBUST_62, COCO17, HAND_21])
def test_sigmas_are_one_positive_value_per_keypoint(schema) -> None:
    sigmas = schema.oks_sigmas
    assert len(sigmas) == schema.num_keypoints
    assert all(sigma > 0.0 for sigma in sigmas)
    assert min(sigmas) >= OKS_SIGMA_FLOOR


@pytest.mark.parametrize("schema", [HALFBUST_62, COCO17, HAND_21])
def test_sigmas_are_symmetric_under_flip_idx(schema) -> None:
    sigmas = schema.oks_sigmas
    for index, mirror in enumerate(schema.flip_idx):
        assert sigmas[index] == sigmas[mirror], (schema.name, index, mirror)


def test_halfbust_sigmas_derive_from_wholebody_with_a_floor() -> None:
    sigmas = HALFBUST_62.oks_sigmas
    for index, mirror in enumerate(HALFBUST_62.flip_idx):
        source = max(
            COCO_WHOLEBODY_SIGMAS[HALFBUST_62.wholebody_index[index]],
            COCO_WHOLEBODY_SIGMAS[HALFBUST_62.wholebody_index[mirror]],
        )
        assert sigmas[index] == pytest.approx(max(OKS_SIGMA_FLOOR, source))


def test_body_sigmas_are_the_coco17_values() -> None:
    assert HALFBUST_62.oks_sigmas[:11] == COCO17.oks_sigmas[:11]
    assert COCO17.oks_sigmas == COCO_WHOLEBODY_SIGMAS[:17]


def test_face_and_hand_points_hit_the_floor() -> None:
    # iBUG-8 (chin, WholeBody 31) and the left/right thumb first joints
    # (WholeBody 92/113) have annotation-scale sigmas below the eye sigma.
    assert COCO_WHOLEBODY_SIGMAS[31] < OKS_SIGMA_FLOOR
    assert COCO_WHOLEBODY_SIGMAS[92] < OKS_SIGMA_FLOOR
    assert COCO_WHOLEBODY_SIGMAS[113] < OKS_SIGMA_FLOOR
    assert HALFBUST_62.oks_sigmas[13] == OKS_SIGMA_FLOOR
    assert HALFBUST_62.oks_sigmas[21] == OKS_SIGMA_FLOOR
    assert HALFBUST_62.oks_sigmas[42] == OKS_SIGMA_FLOOR
    # The two jaw ends are the only schema points above the floor in the face.
    assert HALFBUST_62.oks_sigmas[11] == HALFBUST_62.oks_sigmas[12] == pytest.approx(0.045)
    assert HALFBUST_62.oks_sigmas[20:41] == HALFBUST_62.oks_sigmas[41:62]
    assert HAND_21.oks_sigmas == HALFBUST_62.oks_sigmas[20:41]


def _mixed_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "images/train",
                "val": "images/val",
                "kpt_shape": list(HALFBUST_62.kpt_shape),
                "flip_idx": list(HALFBUST_62.flip_idx),
                "names": {0: "person"},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_mixed_yaml_feeds_v8_pose_loss(tmp_path: Path) -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8PoseLoss

    from cpsa_ml.models import build_model
    from cpsa_ml.variant_study import build_mixed_data_yaml

    half = _mixed_dataset(tmp_path / "halfbust")
    target = tmp_path / "data_mixed.yaml"
    build_mixed_data_yaml(half, None, target)
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data["kpt_oks_sigmas"] == list(HALFBUST_62.oks_sigmas)

    # PoseTrainer.set_model_attributes does exactly this assignment from the
    # dataset YAML before v8PoseLoss is constructed.
    model = build_model("halfbust62", variant="n", activation="lrelu")
    model.args = get_cfg()
    model.kpt_oks_sigmas = data["kpt_oks_sigmas"]
    criterion = v8PoseLoss(model)
    assert criterion.keypoint_loss.sigmas.tolist() == pytest.approx(list(HALFBUST_62.oks_sigmas))

    # The control: without the YAML key the 62-point head gets 1/62 for every
    # joint, which is the saturated loss this fix removes.
    control = build_model("halfbust62", variant="n", activation="lrelu")
    control.args = get_cfg()
    control_criterion = v8PoseLoss(control)
    assert control_criterion.keypoint_loss.sigmas.tolist() == pytest.approx([1 / 62] * 62)
