"""Tests for the HALFBUST_62 accuracy metrics (hand-built cases)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cpsa_ml.eval import (
    HALFBUST62_FACE_ANCHOR_INDICES,
    HALFBUST62_GROUPS,
    HALFBUST62_N_KPT,
    HALFBUST62_OKS_SIGMAS,
    EvalConfig,
    ImageSample,
    PersonPrediction,
    PersonTarget,
    average_precision,
    evaluate,
    face_width_px,
    oks_matrix,
    read_label_file,
    read_yolo_pose_labels,
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FACE_WIDTH = 100.0  # jaw ends at x=100 and x=200


def _keypoints() -> np.ndarray:
    """All 62 keypoints visible, jaw width exactly 100 px."""

    keypoints = np.zeros((HALFBUST62_N_KPT, 3), dtype=np.float64)
    keypoints[:, 0] = 320.0
    keypoints[:, 1] = 240.0
    keypoints[:, 2] = 1.0
    keypoints[1] = (310.0, 150.0, 1.0)  # left eye
    keypoints[2] = (350.0, 150.0, 1.0)  # right eye
    keypoints[3] = (300.0, 50.0, 1.0)  # left ear
    keypoints[4] = (400.0, 50.0, 1.0)  # right ear
    keypoints[11] = (100.0, 200.0, 1.0)  # jaw right
    keypoints[12] = (200.0, 200.0, 1.0)  # jaw left
    for index in range(62):
        keypoints[index] = (
            keypoints[index, 0] + 0.5 * (index % 7),
            keypoints[index, 1] + 0.5 * (index % 5),
            1.0,
        )
    keypoints[1] = (310.0, 150.0, 1.0)
    keypoints[2] = (350.0, 150.0, 1.0)
    keypoints[3] = (300.0, 50.0, 1.0)
    keypoints[4] = (400.0, 50.0, 1.0)
    keypoints[11] = (100.0, 200.0, 1.0)
    keypoints[12] = (200.0, 200.0, 1.0)
    return keypoints


def _target(**kwargs: object) -> PersonTarget:
    return PersonTarget(
        box=np.array([0.0, 0.0, IMAGE_WIDTH, IMAGE_HEIGHT], dtype=np.float64),
        keypoints=_keypoints(),
        **kwargs,
    )


def _prediction(
    target: PersonTarget,
    *,
    shift: float = 0.0,
    left_hand_shift: float = 0.0,
    keypoint_conf: float = 1.0,
    score: float = 0.9,
    box: np.ndarray | None = None,
) -> PersonPrediction:
    keypoints = target.keypoints.copy()
    keypoints[:, 2] = keypoint_conf
    keypoints[:, 0] += shift
    keypoints[20:41, 0] += left_hand_shift
    return PersonPrediction(
        box=target.box.copy() if box is None else box,
        score=score,
        keypoints=keypoints,
    )


def test_oks_sigmas_match_the_documented_convention() -> None:
    assert len(HALFBUST62_OKS_SIGMAS) == HALFBUST62_N_KPT
    assert HALFBUST62_OKS_SIGMAS[:11] == pytest.approx(
        (0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062)
    )
    assert set(HALFBUST62_OKS_SIGMAS[11:]) == {0.026}
    assert len(HALFBUST62_FACE_ANCHOR_INDICES) == 14


def test_groups_cover_the_schema() -> None:
    groups = dict(HALFBUST62_GROUPS)
    assert groups["face"] == tuple(range(11, 20))
    assert groups["shoulders"] == (5, 6)
    assert groups["elbows"] == (7, 8)
    assert groups["wrists"] == (9, 10)
    assert groups["left_hand"] == tuple(range(20, 41))
    assert groups["right_hand"] == tuple(range(41, 62))


def test_read_yolo_pose_labels_scales_to_pixels() -> None:
    keypoints = np.zeros((HALFBUST62_N_KPT, 3), dtype=np.float64)
    keypoints[:, 0] = 0.5
    keypoints[:, 1] = 0.5
    keypoints[:, 2] = 1.0
    keypoints[11] = (0.25, 0.5, 1.0)
    values = [0, 0.5, 0.5, 1.0, 1.0]
    for x, y, visibility in keypoints:
        values.extend((x, y, visibility))
    text = " ".join(str(value) for value in values)

    targets = read_yolo_pose_labels(text, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    assert len(targets) == 1
    target = targets[0]
    assert target.class_id == 0
    np.testing.assert_allclose(target.box, [0.0, 0.0, IMAGE_WIDTH, IMAGE_HEIGHT])
    np.testing.assert_allclose(target.keypoints[11], [0.25 * IMAGE_WIDTH, 0.5 * IMAGE_HEIGHT, 1.0])
    assert target.keypoints.shape == (HALFBUST62_N_KPT, 3)


def test_read_label_file_rejects_wrong_keypoint_count(tmp_path: Path) -> None:
    path = tmp_path / "broken.txt"
    path.write_text("0 0.5 0.5 1 1 0.5 0.5 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="are needed for 62 keypoints"):
        read_label_file(path, IMAGE_WIDTH, IMAGE_HEIGHT)


def test_read_yolo_pose_labels_rejects_non_finite() -> None:
    values = ["0", "0.5", "0.5", "1", "1"] + ["nan", "0", "1"] * HALFBUST62_N_KPT
    with pytest.raises(ValueError, match="non-finite"):
        read_yolo_pose_labels(" ".join(values), IMAGE_WIDTH, IMAGE_HEIGHT)


def test_face_width_uses_jaw_then_ears_then_eyes() -> None:
    keypoints = _keypoints()
    assert face_width_px(keypoints) == pytest.approx(FACE_WIDTH)

    ears = keypoints.copy()
    ears[[11, 12], 2] = 0.0
    assert face_width_px(ears) == pytest.approx(100.0)

    eyes = ears.copy()
    eyes[[3, 4], 2] = 0.0
    assert face_width_px(eyes) == pytest.approx(40.0 * 2.2)

    assert face_width_px(eyes, source="jaw") is None
    assert face_width_px(eyes, source="ears") is None
    assert face_width_px(eyes, source="eyes") == pytest.approx(88.0)


def test_perfect_prediction_scores_one() -> None:
    target = _target()
    sample = ImageSample(
        name="perfect",
        predictions=(_prediction(target),),
        targets=(target,),
    )

    report = evaluate([sample])

    assert report["detection"]["ap50"] == pytest.approx(1.0)
    assert report["pose"]["map50_95"] == pytest.approx(1.0)
    assert report["pose"]["map50"] == pytest.approx(1.0)
    assert report["counts"]["person_missing_rate"] == 0.0
    for group in report["groups"].values():
        assert group["missing_rate"] == 0.0
        assert group["pck"]["0.10"] == pytest.approx(1.0)
        assert group["pck"]["0.20"] == pytest.approx(1.0)
        assert group["pck_all"]["0.20"] == pytest.approx(1.0)
    assert report["centroids"]["face"]["mean_fw"] == pytest.approx(0.0)
    assert report["centroids"]["hand"]["all"]["mean_fw"] == pytest.approx(0.0)


def test_known_shift_gives_the_expected_pck_and_oks() -> None:
    target = _target()
    shifted = _prediction(target, shift=15.0)
    report = evaluate([ImageSample("shift", (shifted,), (target,))])

    for group in report["groups"].values():
        assert group["pck"]["0.10"] == pytest.approx(0.0)
        assert group["pck"]["0.20"] == pytest.approx(1.0)
        assert group["pck_all"]["0.20"] == pytest.approx(1.0)

    # 15 px with area 640*480*0.53: OKS is the mean of exp(-d^2/(2 sigma)^2 area 2)
    # over the visible keypoints, so it uses the per-keypoint sigmas.
    area = IMAGE_WIDTH * IMAGE_HEIGHT * 0.53
    expected_oks = float(
        np.mean(
            [
                math.exp(-(15.0**2) / ((2 * sigma) ** 2 * area * 2))
                for sigma in HALFBUST62_OKS_SIGMAS
            ]
        )
    )
    thresholds = report["config"]["oks_thresholds"]
    expected_map = sum(1.0 for threshold in thresholds if threshold <= expected_oks) / len(
        thresholds
    )
    assert report["pose"]["map50_95"] == pytest.approx(expected_map)

    assert report["centroids"]["face"]["mean_fw"] == pytest.approx(0.15)
    assert report["centroids"]["hand"]["all"]["mean_fw"] == pytest.approx(0.15)


def test_hand_centroid_error_is_per_side() -> None:
    target = _target()
    prediction = _prediction(target, left_hand_shift=30.0)
    report = evaluate([ImageSample("hand", (prediction,), (target,))])

    assert report["centroids"]["hand"]["left"]["mean_fw"] == pytest.approx(0.30)
    assert report["centroids"]["hand"]["right"]["mean_fw"] == pytest.approx(0.0)
    assert report["centroids"]["face"]["mean_fw"] == pytest.approx(0.0)


def test_missing_detection_counts_as_a_miss_everywhere() -> None:
    target = _target()
    report = evaluate([ImageSample("empty", (), (target,))])

    assert report["detection"]["ap50"] == 0.0
    assert report["pose"]["map50_95"] == 0.0
    assert report["counts"]["person_missing"] == 1
    assert report["counts"]["person_missing_rate"] == 1.0
    for group in report["groups"].values():
        assert group["missing_rate"] == pytest.approx(1.0)
        assert group["evaluated"] == 0
        assert group["pck"]["0.20"] is None
        assert group["pck_all"]["0.20"] == 0.0


def test_low_keypoint_confidence_counts_as_missing() -> None:
    target = _target()
    low = _prediction(target, keypoint_conf=0.2)
    report = evaluate([ImageSample("low", (low,), (target,))])

    assert report["detection"]["ap50"] == pytest.approx(1.0)
    assert report["counts"]["person_missing"] == 0
    for group in report["groups"].values():
        assert group["missing_rate"] == pytest.approx(1.0)
        assert group["evaluated"] == 0
        assert group["pck"]["0.20"] is None
        assert group["pck_all"]["0.20"] == 0.0


def test_false_positive_above_the_true_positive_halves_ap() -> None:
    target = _target()
    far_box = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float64)
    false_positive = PersonPrediction(
        box=far_box,
        score=0.95,
        keypoints=target.keypoints.copy(),
    )
    true_positive = _prediction(target, score=0.8)
    report = evaluate(
        [ImageSample("fp", (true_positive, false_positive), (target,))],
        EvalConfig(match_iou=0.5),
    )

    assert report["detection"]["ap50"] == pytest.approx(0.5)
    # The association only scores predictions above match_iou, so the true
    # positive is still found for the per-keypoint metrics.
    assert report["groups"]["wrists"]["pck_all"]["0.20"] == pytest.approx(1.0)


def test_large_keypoint_error_drops_pose_map_but_not_detection() -> None:
    target = _target()
    shifted = _prediction(target, shift=200.0)
    report = evaluate([ImageSample("bad_pose", (shifted,), (target,))])

    assert report["detection"]["ap50"] == pytest.approx(1.0)
    assert report["pose"]["map50_95"] == 0.0
    assert report["groups"]["face"]["pck"]["0.20"] == 0.0


def test_instance_without_face_width_is_excluded_from_normalized_metrics() -> None:
    keypoints = _keypoints()
    keypoints[[1, 2, 3, 4, 11, 12], 2] = 0.0
    target = PersonTarget(
        box=np.array([0.0, 0.0, IMAGE_WIDTH, IMAGE_HEIGHT], dtype=np.float64),
        keypoints=keypoints,
    )
    prediction = _prediction(target)
    report = evaluate([ImageSample("no_scale", (prediction,), (target,))])

    assert report["counts"]["instances_without_face_width"] == 1
    assert report["counts"]["instances_with_face_width"] == 0
    wrists = report["groups"]["wrists"]
    assert wrists["missing"] == 0
    assert wrists["evaluated"] == 0
    assert wrists["pck"]["0.20"] is None
    assert wrists["pck_all"]["0.20"] == 0.0


def test_average_precision_edge_cases() -> None:
    assert math.isnan(average_precision([], 0))
    assert average_precision([], 2) == 0.0
    assert average_precision([True, True], 2) == pytest.approx(1.0)
    assert average_precision([True, False, True], 2) < 1.0


def test_evaluate_on_no_samples_is_defined() -> None:
    report = evaluate([])
    assert report["counts"]["images"] == 0
    assert math.isnan(report["detection"]["ap50"])
    assert math.isnan(report["pose"]["map50_95"])


def test_oks_matrix_matches_the_ultralytics_formula() -> None:
    target = _target()
    keypoints = target.keypoints
    shifted = keypoints.copy()
    shifted[:, 0] += 10.0
    oks = oks_matrix(
        keypoints[None, ...],
        target.box[None, ...],
        shifted[None, ...],
        HALFBUST62_OKS_SIGMAS,
    )

    expected = float(
        np.mean(
            [
                math.exp(-(10.0**2) / ((2 * sigma) ** 2 * 640 * 480 * 0.53 * 2))
                for sigma in HALFBUST62_OKS_SIGMAS
            ]
        )
    )
    assert oks.shape == (1, 1)
    assert oks[0, 0] == pytest.approx(expected)


def test_visibility_mask_limits_oks_to_annotated_points() -> None:
    target = _target()
    keypoints = target.keypoints.copy()
    keypoints[:, 2] = 0.0
    keypoints[11] = (100.0, 200.0, 1.0)
    predicted = keypoints.copy()
    predicted[:, 2] = 1.0
    predicted[11, 0] += 5.0
    oks = oks_matrix(
        keypoints[None, ...],
        target.box[None, ...],
        predicted[None, ...],
        HALFBUST62_OKS_SIGMAS,
    )
    expected = math.exp(-(5.0**2) / ((2 * 0.026) ** 2 * 640 * 480 * 0.53 * 2))
    assert oks[0, 0] == pytest.approx(expected)
