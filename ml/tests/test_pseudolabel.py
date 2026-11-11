"""Tests for the teacher auto-labeler (injected models, no ONNX downloads)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from cpsa_ml import pseudolabel
from cpsa_ml.pseudolabel import (
    PALM_RATIO_RANGE,
    AutoLabelConfig,
    GroupThresholds,
    PersonDetector,
    WholeBodyTeacher,
    _hand_sample,
    _person_label,
    auto_label,
    split_for,
)
from cpsa_ml.schema import HALFBUST_62

IMAGE_SIZE = 160
BOX = np.asarray([[10.0, 10.0, 150.0, 150.0]], dtype=np.float32)


def teacher_keypoints(size: int = IMAGE_SIZE) -> np.ndarray:
    """Return a synthetic 133-keypoint teacher result with high scores."""

    keypoints = np.zeros((133, 3), dtype=np.float32)
    for index in range(17):  # body
        keypoints[index] = (60.0 + 3.0 * index, 40.0 + 2.0 * index, 2.5)
    for index in range(23, 91):  # face
        keypoints[index] = (70.0 + (index % 8), 50.0 + (index % 5), 2.0)
    keypoints[91] = (20.0, 118.0, 2.0)  # left wrist
    for finger in range(5):  # left hand 92-111
        for joint in range(4):
            index = 92 + finger * 4 + joint
            keypoints[index] = (20.0 + 5.0 * finger, 100.0 - 8.0 * joint, 2.0)
    keypoints[112] = (120.0, 118.0, 2.0)  # right wrist
    for finger in range(5):  # right hand 113-132
        for joint in range(4):
            index = 113 + finger * 4 + joint
            keypoints[index] = (120.0 + 5.0 * finger, 100.0 - 8.0 * joint, 2.0)
    assert size == IMAGE_SIZE
    return keypoints


def fake_detector(image: np.ndarray) -> np.ndarray:
    return BOX.copy()


def empty_detector(image: np.ndarray) -> np.ndarray:
    return np.zeros((0, 4), dtype=np.float32)


def fake_teacher(image: np.ndarray, boxes) -> list[np.ndarray]:
    return [teacher_keypoints() for _ in boxes]


def low_hand_teacher(image: np.ndarray, boxes) -> list[np.ndarray]:
    keypoints = teacher_keypoints()
    keypoints[91:133, 2] = 0.2  # below the hand threshold
    return [keypoints for _ in boxes]


def draw_frame(path: Path, prompt: str = "peace") -> None:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (30, 32, 38))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 130, 140), fill=(90, 110, 140))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=95)


def make_frames(root: Path, count: int = 3) -> Path:
    for index in range(count):
        draw_frame(root / f"{root.name}" / f"frame_{index:05d}.jpg")
    return root


def read_config(**overrides) -> AutoLabelConfig:
    values = dict(
        out=Path("/tmp/unused"),
        frames=Path("/tmp/unused"),
        thresholds=GroupThresholds(),
        val_fraction=0.0,
    )
    values.update(overrides)
    return AutoLabelConfig(**values)


class FakeOnnxSession:
    """Stand-in for an ``onnxruntime.InferenceSession`` provider list."""

    def __init__(self, providers: tuple[str, ...]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return list(self._providers)


def fake_rtmlib(providers: tuple[str, ...], events: list[str] | None = None) -> SimpleNamespace:
    """Return an rtmlib stand-in whose tools build a session with ``providers``."""

    class FakeTool:
        def __init__(self, *args, **kwargs) -> None:
            if events is not None:
                events.append("model")
            self.session = FakeOnnxSession(providers)

    return SimpleNamespace(RTMPose=FakeTool, YOLOX=FakeTool)


def test_auto_label_writes_the_halfbust_and_hand_datasets(tmp_path: Path) -> None:
    frames = make_frames(tmp_path / "frames")
    out = tmp_path / "labeled"
    payload = auto_label(
        read_config(out=out, frames=frames),
        detector=fake_detector,
        teacher=fake_teacher,
    )

    assert payload["frames"]["labeled"] == 3
    assert payload["frames"]["processed"] == 3
    assert payload["persons"]["total"] == 3
    assert payload["keypoints"]["visible"] == 3 * 62  # only above-threshold points count

    halfbust = out / "halfbust"
    data = yaml.safe_load((halfbust / "data.yaml").read_text())
    assert data["names"] == {0: "person"}
    assert data["kpt_shape"] == list(HALFBUST_62.kpt_shape)
    assert (halfbust / "schema.json").is_file()
    labels = sorted((halfbust / "labels" / "train").glob("*.txt"))
    assert len(labels) == 3
    tokens = labels[0].read_text().split()
    assert len(tokens) == 5 + 62 * 3
    keypoints = np.asarray([float(value) for value in tokens[5:]], dtype=np.float32).reshape(62, 3)
    assert int(np.count_nonzero(keypoints[:, 2] > 0)) > 30
    assert set(np.unique(keypoints[:, 2])) <= {0.0, 2.0}

    hands = out / "hand21"
    hand_data = yaml.safe_load((hands / "data.yaml").read_text())
    assert hand_data["names"] == {0: "hand"}
    assert hand_data["kpt_shape"] == [21, 3]
    assert hand_data["flip_idx"] == list(range(21))
    crops = sorted((hands / "images" / "train").glob("*.jpg"))
    assert len(crops) == 6  # two hands per frame
    with Image.open(crops[0]) as crop:
        assert crop.size == (192, 192)
    handedness = payload["hand_crops"]["train"]["handedness"]
    assert handedness == {"left": 3, "right": 3}
    assert payload["hand_crops"]["train"]["sources"] == {"recorded": 6}


def test_auto_label_skips_frames_without_a_person(tmp_path: Path) -> None:
    frames = make_frames(tmp_path / "frames", count=2)
    out = tmp_path / "labeled"
    payload = auto_label(
        read_config(out=out, frames=frames),
        detector=empty_detector,
        teacher=fake_teacher,
    )
    assert payload["frames"]["no_person"] == 2
    assert payload["frames"]["labeled"] == 0
    assert not list((out / "halfbust" / "images").rglob("*.jpg"))


def test_auto_label_skips_hands_below_the_score_threshold(tmp_path: Path) -> None:
    frames = make_frames(tmp_path / "frames", count=1)
    out = tmp_path / "labeled"
    payload = auto_label(
        read_config(out=out, frames=frames),
        detector=fake_detector,
        teacher=low_hand_teacher,
    )
    assert payload["frames"]["labeled"] == 1
    assert payload["persons"]["no_left_hand"] == 1
    assert payload["persons"]["no_right_hand"] == 1
    assert not list((out / "hand21" / "images").rglob("*.jpg"))


def test_auto_label_is_resumable(tmp_path: Path) -> None:
    frames = make_frames(tmp_path / "frames", count=2)
    out = tmp_path / "labeled"
    config = read_config(out=out, frames=frames)
    auto_label(config, detector=fake_detector, teacher=fake_teacher)
    crops = sorted((out / "hand21" / "images" / "train").glob("*.jpg"))

    payload = auto_label(config, detector=fake_detector, teacher=fake_teacher)
    assert payload["frames"]["existing"] == 2
    assert payload["frames"]["processed"] == 2  # restored, no new work
    assert payload["frames"]["labeled"] == 2  # counters are restored, not recounted
    assert sorted((out / "hand21" / "images" / "train").glob("*.jpg")) == crops


def test_auto_label_rejects_changed_settings(tmp_path: Path) -> None:
    frames = make_frames(tmp_path / "frames", count=1)
    out = tmp_path / "labeled"
    auto_label(read_config(out=out, frames=frames), detector=fake_detector, teacher=fake_teacher)
    with pytest.raises(ValueError, match="fresh output directory"):
        auto_label(
            read_config(out=out, frames=frames, seed=3),
            detector=fake_detector,
            teacher=fake_teacher,
        )


def test_person_label_marks_teacher_points() -> None:
    wholebody = HALFBUST_62.extract(teacher_keypoints())
    wholebody[0, 2] = 0.1  # below the body threshold
    line = _person_label(
        wholebody, BOX[0], (IMAGE_SIZE, IMAGE_SIZE), GroupThresholds(), HALFBUST_62
    )
    tokens = line.split()
    assert len(tokens) == 5 + 62 * 3
    keypoints = np.asarray([float(value) for value in tokens[5:]], dtype=np.float32).reshape(62, 3)
    assert keypoints[0, 2] == 0.0
    assert np.all(keypoints[1:, 2] == 2.0)
    x1, y1, x2, y2 = (float(value) for value in tokens[1:5])
    assert x1 < x2 and y1 < y2


def test_hand_sample_covers_the_landmarks_and_records_the_handedness(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    draw_frame(frame)
    sample, reason = _hand_sample(
        teacher_keypoints(),
        tuple(range(91, 112)),
        "left",
        frame_path=frame,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        person_box=BOX[0],
        thresholds=GroupThresholds(),
        min_keypoints=10,
        expand=1.2,
        max_fraction=0.6,
        palm_ratio=PALM_RATIO_RANGE,
        min_score=2.0,
    )
    assert reason == "ok"
    assert sample is not None
    assert sample.handedness == "left"
    assert sample.source == "recorded"
    assert sample.keypoints.shape == (21, 3)
    assert int(np.count_nonzero(sample.keypoints[:, 2] > 0)) == 21
    x1, y1, x2, y2 = sample.box
    assert x1 <= sample.keypoints[:, 0].min() and sample.keypoints[:, 0].max() <= x2
    assert y1 <= sample.keypoints[:, 1].min() and sample.keypoints[:, 1].max() <= y2

    keypoints = teacher_keypoints()
    keypoints[91:133, 2] = 0.2
    sample, reason = _hand_sample(
        keypoints,
        tuple(range(91, 112)),
        "left",
        frame_path=frame,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        person_box=BOX[0],
        thresholds=GroupThresholds(),
        min_keypoints=10,
        expand=1.2,
        max_fraction=0.6,
        palm_ratio=PALM_RATIO_RANGE,
        min_score=2.0,
    )
    assert sample is None
    assert reason == "few_keypoints"


def test_hand_sample_rejects_a_hand_that_spans_the_torso(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    draw_frame(frame)
    keypoints = teacher_keypoints()
    # A teacher block that straddles the whole torso: the hand is not visible
    # but the score is high, so only the extent relative to the person can
    # reject it.
    left = keypoints[91:112, :2]
    centre = 0.5 * (left.min(axis=0) + left.max(axis=0))
    keypoints[91:112, :2] = centre + (left - centre) * 6.0
    sample, reason = _hand_sample(
        keypoints,
        tuple(range(91, 112)),
        "left",
        frame_path=frame,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        person_box=BOX[0],
        thresholds=GroupThresholds(),
        min_keypoints=10,
        expand=1.2,
        max_fraction=0.6,
        palm_ratio=PALM_RATIO_RANGE,
        min_score=2.0,
    )
    assert sample is None
    assert reason == "too_large"


def test_hand_sample_rejects_an_impossible_finger_chain(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    draw_frame(frame)
    keypoints = teacher_keypoints()
    # The middle fingertip jumps far away while the palm stays put: the block
    # is small enough for the extent gate but anatomically impossible.
    keypoints[103, :2] = (130.0, 60.0)  # local landmark 12 of the left hand
    sample, reason = _hand_sample(
        keypoints,
        tuple(range(91, 112)),
        "left",
        frame_path=frame,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        person_box=(0.0, 0.0, 400.0, 400.0),
        thresholds=GroupThresholds(),
        min_keypoints=10,
        expand=1.2,
        max_fraction=0.6,
        palm_ratio=PALM_RATIO_RANGE,
        min_score=2.0,
    )
    assert sample is None
    assert reason == "bad_topology"


def test_hand_sample_rejects_a_low_confidence_hand(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    draw_frame(frame)
    keypoints = teacher_keypoints()
    keypoints[91:112, 2] = 1.5  # above the group threshold, below the crop gate
    sample, reason = _hand_sample(
        keypoints,
        tuple(range(91, 112)),
        "left",
        frame_path=frame,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        person_box=BOX[0],
        thresholds=GroupThresholds(),
        min_keypoints=10,
        expand=1.2,
        max_fraction=0.6,
        palm_ratio=PALM_RATIO_RANGE,
        min_score=2.0,
    )
    assert sample is None
    assert reason == "low_score"


def test_split_for_is_stable_and_balanced() -> None:
    paths = [f"prompt/frame_{index:05d}.jpg" for index in range(400)]
    first = [split_for(path, seed=0, val_fraction=0.1) for path in paths]
    assert first == [split_for(path, seed=0, val_fraction=0.1) for path in paths]
    assert 0.05 <= first.count("val") / len(paths) <= 0.16
    assert split_for(paths[0], seed=0, val_fraction=0.0) == "train"
    assert split_for(paths[0], seed=0, val_fraction=1.0 - 1e-9) in {"train", "val"}


def test_auto_label_samples_a_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (IMAGE_SIZE, IMAGE_SIZE)
    )
    assert writer.isOpened(), "the cv2 build cannot write mp4"
    frame = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 40, dtype=np.uint8)
    for _ in range(6):
        writer.write(frame)
    writer.release()

    out = tmp_path / "labeled"
    payload = auto_label(
        read_config(out=out, frames=None, video=video, video_stride=3),
        detector=fake_detector,
        teacher=fake_teacher,
    )
    assert payload["frames"]["processed"] == 2
    assert payload["frames"]["labeled"] == 2
    assert len(list((out / "halfbust" / "labels" / "train").glob("*.txt"))) == 2


def test_config_requires_a_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frames and/or video"):
        AutoLabelConfig(out=tmp_path)


def test_config_requires_an_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="halfbust and/or hands"):
        AutoLabelConfig(out=tmp_path, frames=tmp_path, halfbust=False, hands=False)


def test_stats_are_written_as_json(tmp_path: Path) -> None:
    frames = make_frames(tmp_path / "frames", count=1)
    out = tmp_path / "labeled"
    payload = auto_label(
        read_config(out=out, frames=frames), detector=fake_detector, teacher=fake_teacher
    )
    stored = json.loads((out / "stats.json").read_text())
    assert stored == payload
    assert stored["meta"]["teacher"]["backend"] == "injected"
    assert stored["prompts"] == {"frames": 1}


def test_cuda_teacher_fails_fast_when_onnxruntime_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        pseudolabel, "_import_rtmlib", lambda: fake_rtmlib(("CPUExecutionProvider",))
    )
    monkeypatch.setattr(pseudolabel, "_preload_cuda_dlls", lambda device: None)
    with pytest.raises(RuntimeError, match="CPUExecutionProvider"):
        WholeBodyTeacher("rtmw-x", device="cuda")


def test_cuda_detector_fails_fast_when_onnxruntime_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        pseudolabel, "_import_rtmlib", lambda: fake_rtmlib(("CPUExecutionProvider",))
    )
    monkeypatch.setattr(pseudolabel, "_preload_cuda_dlls", lambda device: None)
    with pytest.raises(RuntimeError, match="--device cpu --teacher rtmw-l"):
        PersonDetector(device="cuda")


def test_cuda_device_accepts_a_cuda_session(monkeypatch) -> None:
    monkeypatch.setattr(
        pseudolabel,
        "_import_rtmlib",
        lambda: fake_rtmlib(("CUDAExecutionProvider", "CPUExecutionProvider")),
    )
    monkeypatch.setattr(pseudolabel, "_preload_cuda_dlls", lambda device: None)
    teacher = WholeBodyTeacher("rtmw-x", device="cuda")
    assert teacher._estimator.session.get_providers()[0] == "CUDAExecutionProvider"


def test_cpu_device_does_not_require_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        pseudolabel, "_import_rtmlib", lambda: fake_rtmlib(("CPUExecutionProvider",))
    )
    detector = PersonDetector(device="cpu")
    assert detector._detector.session.get_providers() == ["CPUExecutionProvider"]


def test_cuda_device_preloads_the_dlls_before_building_the_models(monkeypatch) -> None:
    events: list[str] = []

    def record_preload(device: str) -> None:
        events.append(f"preload:{device}")

    monkeypatch.setattr(pseudolabel, "_preload_cuda_dlls", record_preload)
    monkeypatch.setattr(
        pseudolabel,
        "_import_rtmlib",
        lambda: fake_rtmlib(("CUDAExecutionProvider", "CPUExecutionProvider"), events=events),
    )
    WholeBodyTeacher("rtmw-x", device="cuda")
    assert events == ["preload:cuda", "model"]


def test_auto_label_logs_throughput_and_an_eta(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    frames = make_frames(tmp_path / "frames", count=4)
    out = tmp_path / "labeled"
    with caplog.at_level(logging.INFO, logger="cpsa_ml.pseudolabel"):
        auto_label(
            read_config(out=out, frames=frames, chunk_size=2),
            detector=fake_detector,
            teacher=fake_teacher,
        )
    messages = [record.getMessage() for record in caplog.records]
    messages = [message for message in messages if "frames/s" in message]
    assert messages
    assert "ETA" in messages[0]
    assert "/4 frames" in messages[0]
