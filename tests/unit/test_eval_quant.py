"""Variant-study eval tooling: int8 dump parsing, decode and comparison (host side)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "vitis"
for _path in (TOOLS, ROOT / "src", ROOT / "ml" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import dump_quant_outputs as dq  # noqa: E402
import eval_quant as eq  # noqa: E402

from cpsa.vision.pose_decode import decode_raw_heads, letterbox  # noqa: E402

INPUT_SIZE = (64, 96)
STRIDES = (8, 16, 32)
FIX_POINTS = (4, 5, 6)
N_KPT = 62
REG_MAX = 16
NUM_CLASSES = 1
CHANNELS = 4 * REG_MAX + NUM_CLASSES + 3 * N_KPT


def _label_line(n_kpt: int = N_KPT) -> str:
    values = ["0", "0.5", "0.5", "1", "1"]
    for _ in range(n_kpt):
        values.extend(["0.5", "0.5", "2"])
    return " ".join(values)


def _random_heads(
    rng: np.random.Generator,
    count: int,
    input_size: tuple[int, int] = INPUT_SIZE,
    strides: tuple[int, ...] = STRIDES,
) -> list[np.ndarray]:
    heads = []
    for stride in strides:
        height, width = input_size[0] // stride, input_size[1] // stride
        heads.append(rng.integers(-128, 128, size=(count, height, width, CHANNELS), dtype=np.int8))
    return heads


def _write_dump(
    path: Path,
    heads: list[np.ndarray],
    *,
    names: list[str] | None = None,
    strides: tuple[int, ...] = STRIDES,
    fix_points: tuple[int, ...] = FIX_POINTS,
    input_size: tuple[int, int] = INPUT_SIZE,
    scales: list[float] | None = None,
    pads: list[tuple[float, float]] | None = None,
    **overrides: Any,
) -> Path:
    count = heads[0].shape[0]
    names = names or [f"frame_{index}" for index in range(count)]
    widths = overrides.pop("image_widths", [input_size[1]] * count)
    heights = overrides.pop("image_heights", [input_size[0]] * count)
    scales = scales or [1.0] * count
    pads = pads or [(0.0, 0.0)] * count
    payload: dict[str, Any] = {
        "names": np.asarray(names),
        "image_widths": np.asarray(widths, dtype=np.int32),
        "image_heights": np.asarray(heights, dtype=np.int32),
        "scales": np.asarray(scales, dtype=np.float64),
        "pad_x": np.asarray([pad[0] for pad in pads], dtype=np.float64),
        "pad_y": np.asarray([pad[1] for pad in pads], dtype=np.float64),
        "strides": np.asarray(strides, dtype=np.int32),
        "fix_points": np.asarray(fix_points, dtype=np.int32),
        "input_size": np.asarray(input_size, dtype=np.int32),
        "n_kpt": np.int32(N_KPT),
        "reg_max": np.int32(REG_MAX),
        "num_classes": np.int32(NUM_CLASSES),
        "channel_layout": np.asarray([4 * REG_MAX, NUM_CLASSES, 3 * N_KPT], dtype=np.int32),
    }
    for stride, head in zip(strides, heads, strict=True):
        payload[f"int8_head_stride{stride}"] = head
    payload.update(overrides)
    np.savez_compressed(path, **payload)
    return path


# ---------------------------------------------------------------------------
# Letterbox port (must match the runtime bit for bit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width", "pad"),
    [
        (64, 96, 114),
        (97, 61, 114),
        (480, 640, 114),
        (33, 200, 0),
    ],
)
def test_letterbox_rgb_matches_runtime(height: int, width: int, pad: int) -> None:
    rng = np.random.default_rng(height * 1000 + width)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    expected_image, expected_scale, expected_pad = letterbox(image, INPUT_SIZE, pad)
    actual_image, actual_scale, actual_pad = dq.letterbox_rgb(image, INPUT_SIZE, pad)

    np.testing.assert_array_equal(actual_image, expected_image)
    assert actual_scale == expected_scale
    assert actual_pad == expected_pad
    assert actual_image.dtype == np.uint8


def test_letterbox_rgb_identity_and_validation() -> None:
    image = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.uint8)
    result, scale, pad = dq.letterbox_rgb(image, INPUT_SIZE)
    assert result is image
    assert (scale, pad) == (1.0, (0.0, 0.0))

    with pytest.raises(ValueError):
        dq.letterbox_rgb(np.zeros((4, 4), dtype=np.uint8), INPUT_SIZE)
    with pytest.raises(ValueError):
        dq.letterbox_rgb(image, (0, 10))
    with pytest.raises(ValueError):
        dq.letterbox_rgb(image, INPUT_SIZE, pad_value=300)


def test_to_model_tensor_matches_dpu_normalization() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(8, 6, 3), dtype=np.uint8)
    tensor = dq.to_model_tensor(image)
    assert tensor.shape == (3, 8, 6)
    assert tensor.dtype == np.float32
    expected = image.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
    np.testing.assert_array_equal(tensor, expected)


# ---------------------------------------------------------------------------
# int8 conversion helpers
# ---------------------------------------------------------------------------


def test_quantize_to_int8_rounds_and_clips() -> None:
    values = np.asarray([0.0, 0.0625, 0.125, -0.0625, 8.0, -9.0], dtype=np.float32)
    quantized = dq.quantize_to_int8(values, 4)
    np.testing.assert_array_equal(quantized, np.asarray([0, 1, 2, -1, 127, -128], dtype=np.int8))
    reconstructed = quantized.astype(np.float64) * 2.0**-4
    np.testing.assert_allclose(reconstructed, [0.0, 0.0625, 0.125, -0.0625, 7.9375, -8.0])


def test_to_nhwc_accepts_both_layouts() -> None:
    nhwc = np.zeros((1, 5, 7, CHANNELS), dtype=np.int8)
    assert dq.to_nhwc(nhwc, CHANNELS).shape == (5, 7, CHANNELS)
    nchw = np.zeros((1, CHANNELS, 5, 7), dtype=np.int8)
    transposed = dq.to_nhwc(nchw, CHANNELS)
    assert transposed.shape == (5, 7, CHANNELS)
    with pytest.raises(ValueError):
        dq.to_nhwc(np.zeros((1, 3, 5, 7), dtype=np.int8), CHANNELS)


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------


def test_load_dump_reads_frames_and_metadata(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    heads = _random_heads(rng, count=2)
    path = _write_dump(
        tmp_path / "quant_outputs.npz",
        heads,
        names=["a", "b"],
        scales=[0.5, 0.25],
        pads=[(4.0, 8.0), (0.0, 12.0)],
        image_widths=[128, 256],
        image_heights=[72, 144],
    )

    dump = eq.load_dump(path)

    assert dump.names == ("a", "b")
    assert dump.strides == STRIDES
    assert dump.fix_points == FIX_POINTS
    assert dump.input_size == INPUT_SIZE
    assert dump.shapes == tuple((INPUT_SIZE[0] // s, INPUT_SIZE[1] // s, CHANNELS) for s in STRIDES)
    np.testing.assert_allclose(dump.scales, [0.5, 0.25])
    np.testing.assert_allclose(dump.pad_y, [8.0, 12.0])
    np.testing.assert_array_equal(dump.image_widths, [128, 256])

    limited = eq.load_dump(path, limit=1)
    assert limited.names == ("a",)
    assert all(head.shape[0] == 1 for head in limited.heads)


def test_load_dump_rejects_inconsistent_payload(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    heads = _random_heads(rng, count=1)

    path = _write_dump(
        tmp_path / "wrong_stride.npz",
        _random_heads(rng, count=1, strides=(8, 16, 64)),
        strides=(8, 16, 64),
    )
    with pytest.raises(SystemExit, match="implies stride"):
        eq.load_dump(path)

    path = _write_dump(tmp_path / "wrong_channels.npz", heads, num_classes=2)
    with pytest.raises(SystemExit, match="channels"):
        eq.load_dump(path)

    path = _write_dump(tmp_path / "wrong_dtype.npz", [head.astype(np.float32) for head in heads])
    with pytest.raises(SystemExit, match="int8"):
        eq.load_dump(path)

    payload = {
        "names": np.asarray(["a"]),
        "strides": np.asarray(STRIDES),
        "fix_points": np.asarray(FIX_POINTS),
    }
    path = tmp_path / "missing.npz"
    np.savez_compressed(path, **payload)
    with pytest.raises(SystemExit, match="input_size"):
        eq.load_dump(path)

    with pytest.raises(SystemExit, match="not found"):
        eq.load_dump(tmp_path / "absent.npz")


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def test_decode_int8_matches_dequantized_float_decode(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    heads = _random_heads(rng, count=1)
    dump = eq.load_dump(_write_dump(tmp_path / "quant_outputs.npz", heads))

    persons = eq.decode_int8(dump, conf_thr=0.5, iou_thr=0.5)
    assert len(persons) == 1
    assert persons[0], "the synthetic dump must produce at least one detection"

    reference = [
        decode_raw_heads(
            [
                head[0].astype(np.float64) * 2.0**-fix_point
                for head, fix_point in zip(heads, FIX_POINTS, strict=True)
            ],
            STRIDES,
            N_KPT,
            0.5,
            0.5,
            REG_MAX,
        )
    ]
    assert len(persons[0]) == len(reference[0])
    for actual, expected in zip(persons[0], reference[0], strict=True):
        np.testing.assert_allclose(actual.box, expected.box, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(actual.xy, expected.xy, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(actual.conf, expected.conf, rtol=0.0, atol=1e-12)
        assert actual.score == pytest.approx(expected.score, abs=1e-12)


def test_decode_int8_unletterboxes_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    heads = _random_heads(rng, count=1)
    dump = eq.load_dump(
        _write_dump(
            tmp_path / "quant_outputs.npz",
            heads,
            scales=[0.5],
            pads=[(4.0, 8.0)],
            image_widths=[192],
            image_heights=[128],
        )
    )

    raw = eq.int8_decoder(dump).decode([head[0] for head in dump.heads], 0.5, 0.5)
    scaled = eq.decode_int8(dump, conf_thr=0.5, iou_thr=0.5)[0]
    assert scaled
    assert len(scaled) == len(raw)
    for actual, expected in zip(scaled, raw, strict=True):
        box = expected.box.copy()
        box[[0, 2]] = (box[[0, 2]] - 4.0) / 0.5
        box[[1, 3]] = (box[[1, 3]] - 8.0) / 0.5
        np.testing.assert_allclose(actual.box, box, rtol=0.0, atol=1e-9)
        xy = (expected.xy - np.asarray([4.0, 8.0])) / 0.5
        np.testing.assert_allclose(actual.xy, xy, rtol=0.0, atol=1e-9)


def test_load_dump_rejects_mismatched_fix_point_count(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    heads = _random_heads(rng, count=1)
    path = _write_dump(tmp_path / "quant_outputs.npz", heads, fix_points=(4, 5))
    with pytest.raises(SystemExit, match="fix points"):
        eq.load_dump(path)


# ---------------------------------------------------------------------------
# Samples, labels and evaluation
# ---------------------------------------------------------------------------


def test_build_samples_pairs_frames_with_labels(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text(_label_line(), encoding="utf-8")
    (labels / "b.txt").write_text("", encoding="utf-8")

    frames: list[list[eq.PersonPose]] = [[], []]
    samples = eq.build_samples(frames, ["a", "b"], labels, [192, 96], [128, 64], N_KPT)

    assert [sample.name for sample in samples] == ["a", "b"]
    assert len(samples[0].targets) == 1
    assert samples[0].targets[0].keypoints.shape == (N_KPT, 3)
    assert samples[1].targets == ()
    with pytest.raises(SystemExit, match="missing label"):
        eq.build_samples(frames, ["a", "c"], labels, [192, 96], [128, 64], N_KPT)
    with pytest.raises(ValueError, match="frames"):
        eq.build_samples(frames, ["a"], labels, [192, 96], [128, 64], N_KPT)


def test_to_prediction_stacks_xy_and_confidence() -> None:
    person = eq.PersonPose(
        box=np.asarray([0.0, 0.0, 10.0, 20.0]),
        score=0.75,
        xy=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        conf=np.asarray([0.9, 0.1]),
    )
    prediction = eq.to_prediction(person)
    assert prediction.score == 0.75
    np.testing.assert_array_equal(prediction.box, person.box)
    np.testing.assert_array_equal(prediction.keypoints, [[1.0, 2.0, 0.9], [3.0, 4.0, 0.1]])

    with pytest.raises(ValueError, match="conf"):
        eq.to_prediction(
            eq.PersonPose(
                box=np.zeros(4),
                score=0.5,
                xy=np.zeros((2, 2)),
                conf=np.zeros(3),
            )
        )


def test_evaluate_samples_reports_expected_sections(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "frame_0.txt").write_text(_label_line(), encoding="utf-8")
    rng = np.random.default_rng(6)
    dump = eq.load_dump(_write_dump(tmp_path / "quant_outputs.npz", _random_heads(rng, count=1)))

    frames = eq.decode_int8(dump, conf_thr=0.5, iou_thr=0.5)
    samples = eq.build_samples(
        frames, dump.names, labels, dump.image_widths, dump.image_heights, dump.n_kpt
    )
    report = eq.evaluate_samples(samples)

    assert report["counts"]["images"] == 1
    assert report["counts"]["instances"] == 1
    assert set(report["groups"]) == {name for name, _ in eq.HALFBUST62_GROUPS}
    assert "ap50" in report["detection"]
    assert "map50_95" in report["pose"]


# ---------------------------------------------------------------------------
# Comparison and decision rule
# ---------------------------------------------------------------------------


def _report(
    pck: dict[str, float],
    *,
    ap50: float = 0.5,
    map50_95: float = 0.2,
) -> dict[str, Any]:
    return {
        "detection": {"ap50": ap50},
        "pose": {"map50_95": map50_95, "map50": 0.3},
        "counts": {"person_missing_rate": 0.0},
        "centroids": {"hand": {"all": {"mean_fw": 0.1}}, "face": {"mean_fw": 0.2}},
        "groups": {
            name: {
                "pck": {"0.20": pck[name]},
                "pck_all": {"0.20": pck[name]},
                "missing_rate": 0.0,
            }
            for name, _ in eq.HALFBUST62_GROUPS
        },
    }


def _all_groups(value: float) -> dict[str, float]:
    return {name: value for name, _ in eq.HALFBUST62_GROUPS}


def test_metric_value_follows_paths_and_rejects_missing() -> None:
    report = _report(_all_groups(0.8))
    assert eq.metric_value(report, ("groups", "wrists", "pck_all", "0.20")) == 0.8
    assert eq.metric_value(report, ("groups", "wrists", "pck_all", "0.10")) is None
    assert eq.metric_value(report, ("missing", "path")) is None
    assert eq.metric_value({"a": float("nan")}, ("a",)) is None
    assert eq.metric_value({"a": True}, ("a",)) is None


def test_comparison_rows_include_deltas_and_missing_values() -> None:
    int8 = _report(_all_groups(0.80), ap50=0.4)
    fp32 = _report(_all_groups(0.85), ap50=0.5)
    rows = eq.comparison_rows(int8, fp32, None)

    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["detection.ap50"]["delta_int8_fp32"] == pytest.approx(-0.1)
    assert by_metric["detection.ap50"]["baseline"] is None
    assert by_metric["detection.ap50"]["delta_int8_baseline"] is None
    assert by_metric["wrists.pck_all@0.20fw"]["delta_int8_fp32"] == pytest.approx(-0.05)
    assert "person_missing_rate" in by_metric

    rows_without_fp32 = eq.comparison_rows(int8, None, fp32)
    by_metric = {row["metric"]: row for row in rows_without_fp32}
    assert by_metric["detection.ap50"]["fp32_host"] is None
    assert by_metric["detection.ap50"]["delta_int8_baseline"] == pytest.approx(-0.1)


@pytest.mark.parametrize(
    ("int8_pck", "baseline_pck", "status"),
    [
        (0.80, 0.80, "pass"),
        (0.785, 0.80, "pass"),  # exactly -0.015 > -0.02
        (0.78, 0.80, "fail"),  # -0.02 is not "within" the margin
        (0.70, 0.80, "fail"),
    ],
)
def test_decision_uses_pck_all_with_two_point_margin(
    int8_pck: float, baseline_pck: float, status: str
) -> None:
    verdict = eq.decision(_report(_all_groups(int8_pck)), _report(_all_groups(baseline_pck)))
    assert verdict["status"] == status
    assert verdict["passed"] is (status == "pass")
    assert verdict["details"]["wrists"]["delta"] == pytest.approx(int8_pck - baseline_pck)


def test_decision_is_unknown_without_baseline() -> None:
    verdict = eq.decision(_report(_all_groups(0.8)), None)
    assert verdict["status"] == "unknown"
    assert verdict["passed"] is None
    assert set(verdict["unknown_groups"]) == set(eq.DEFAULT_DECISION_GROUPS)


def test_decision_is_unknown_when_the_baseline_has_no_accuracy() -> None:
    verdict = eq.decision(_report(_all_groups(0.0)), _report(_all_groups(0.0)))
    assert verdict["status"] == "unknown"
    assert verdict["passed"] is None
    assert verdict["details"]["wrists"]["ok"] is None


def test_render_markdown_lists_every_row() -> None:
    int8 = _report(_all_groups(0.8))
    fp32 = _report(_all_groups(0.85))
    table = eq.render_markdown(
        eq.comparison_rows(int8, fp32, None),
        {"int8": "INT8", "fp32_host": "FP32 host", "baseline": "n/a"},
    )
    lines = table.splitlines()
    assert lines[0].startswith("| metric | INT8 |")
    assert "wrists.pck_all@0.20fw" in table
    assert "n/a" in table
    assert len(lines) == 2 + len(eq.comparison_rows(int8, fp32, None))


# ---------------------------------------------------------------------------
# Image lookup for the FP32 reference
# ---------------------------------------------------------------------------


def test_resolve_test_images_follows_dump_order(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    dump = eq.load_dump(
        _write_dump(tmp_path / "quant_outputs.npz", _random_heads(rng, count=2), names=["b", "a"])
    )
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a", "b", "c"):
        (images / f"{name}.jpg").write_bytes(b"stub")
    (images / "notes.txt").write_text("ignored")

    resolved = eq.resolve_test_images(dump, images)
    assert [path.stem for path in resolved] == ["b", "a"]

    (images / "b.jpg").unlink()
    with pytest.raises(SystemExit, match="no image"):
        eq.resolve_test_images(dump, images)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_parse_args_defaults_and_required_source() -> None:
    args = eq.parse_args(["--dump", "d.npz", "--out", "out"])
    assert args.dump == Path("d.npz")
    assert args.out == Path("out")
    assert args.conf_thr == 0.25
    assert args.skip_fp32 is False
    with pytest.raises(SystemExit):
        eq.parse_args([])


def test_resolve_dirs_prefers_explicit_paths() -> None:
    explicit = eq.parse_args(
        [
            "--dump",
            "d.npz",
            "--out",
            "out",
            "--data",
            "data",
            "--test-images",
            "imgs",
            "--labels",
            "lbls",
        ]
    )
    assert eq.resolve_dirs(explicit) == (Path("imgs"), Path("lbls"))

    from_data = eq.parse_args(
        ["--dump", "d.npz", "--out", "out", "--data", "data", "--test-split", "train"]
    )
    assert eq.resolve_dirs(from_data) == (
        Path("data/images/train"),
        Path("data/labels/train"),
    )

    missing = eq.parse_args(["--dump", "d.npz", "--out", "out"])
    with pytest.raises(SystemExit, match="labels"):
        eq.resolve_dirs(missing)


def test_report_of_unwraps_metrics_payload() -> None:
    report = _report(_all_groups(0.8))
    assert eq.report_of({"meta": {}, "metrics": report}) is report
    assert eq.report_of(report) is report


def test_main_writes_reports_without_fp32(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "frame_0.txt").write_text(_label_line(), encoding="utf-8")
    rng = np.random.default_rng(8)
    dump = _write_dump(tmp_path / "quant_outputs.npz", _random_heads(rng, count=1))
    out = tmp_path / "out"

    assert (
        eq.main(
            [
                "--dump",
                str(dump),
                "--labels",
                str(labels),
                "--test-images",
                str(tmp_path),
                "--baseline",
                str(_baseline_json(tmp_path)),
                "--variant",
                "hswish_480x640",
                "--out",
                str(out),
                "--limit",
                "1",
            ]
        )
        == 0
    )

    metrics = json.loads((out / "metrics_int8.json").read_text(encoding="utf-8"))
    assert metrics["meta"]["variant"] == "hswish_480x640"
    assert metrics["meta"]["frames"] == 1
    assert metrics["meta"]["fix_points"] == list(FIX_POINTS)
    assert "metrics" in metrics
    assert (out / "comparison.md").read_text(encoding="utf-8").startswith("| metric |")

    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["variant"] == "hswish_480x640"
    assert comparison["inputs"]["baseline"].endswith("baseline.json")
    assert comparison["decision"]["status"] in {"pass", "fail", "unknown"}
    assert len(comparison["rows"]) > 0


def _baseline_json(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.json"
    payload = {"metrics": _report(_all_groups(0.8))}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_main_requires_a_reference(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "frame_0.txt").write_text(_label_line(), encoding="utf-8")
    rng = np.random.default_rng(9)
    dump = _write_dump(tmp_path / "quant_outputs.npz", _random_heads(rng, count=1))

    with pytest.raises(SystemExit, match="baseline"):
        eq.main(
            [
                "--dump",
                str(dump),
                "--labels",
                str(labels),
                "--test-images",
                str(tmp_path),
                "--skip-fp32",
                "--out",
                str(tmp_path / "out"),
            ]
        )


# ---------------------------------------------------------------------------
# Grouped dumps and side-by-side splits
# ---------------------------------------------------------------------------


def test_parse_image_source_accepts_a_name_and_a_plain_directory() -> None:
    assert dq.parse_image_source("half=/data/half/val") == ("half", Path("/data/half/val"))
    assert dq.parse_image_source("/data/fullframe62/images/val") == (
        "val",
        Path("/data/fullframe62/images/val"),
    )
    assert dq.parse_image_source("plain") == ("plain", Path("plain"))


def test_resolve_grouped_images_keeps_source_order_and_names(tmp_path: Path) -> None:
    half = tmp_path / "half"
    full = tmp_path / "full"
    for directory, names in ((half, ["h0", "h1"]), (full, ["f0", "f1"])):
        directory.mkdir()
        for name in names:
            (directory / f"{name}.jpg").write_bytes(b"stub")

    images, groups = dq.resolve_grouped_images([f"half={half}", f"full={full}"], limit=1)

    assert [path.name for path in images] == ["h0.jpg", "f0.jpg"]
    assert groups == ["half", "full"]
    with pytest.raises(SystemExit, match="not found"):
        dq.resolve_grouped_images([f"missing={tmp_path / 'absent'}"], limit=1)


def test_load_dump_reads_groups_and_subsets_frames(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    heads = _random_heads(rng, count=3)
    path = _write_dump(
        tmp_path / "quant_outputs.npz",
        heads,
        names=["a", "b", "c"],
        groups=["half", "full", "full"],
        scales=[0.1, 0.2, 0.3],
    )

    dump = eq.load_dump(path)
    assert dump.groups == ("half", "full", "full")

    subset = dump.subset([2, 0])
    assert subset.names == ("c", "a")
    assert subset.groups == ("full", "half")
    np.testing.assert_allclose(subset.scales, [0.3, 0.1])
    assert all(head.shape[0] == 2 for head in subset.heads)
    np.testing.assert_array_equal(subset.heads[0][0], heads[0][2])
    with pytest.raises(ValueError, match="empty"):
        dump.subset([])

    legacy = eq.load_dump(_write_dump(tmp_path / "legacy.npz", _random_heads(rng, count=1)))
    assert legacy.groups == ("",)


def test_parse_eval_split_reads_name_images_and_labels() -> None:
    split = eq.parse_eval_split("half=/data/half/val:/data/half/labels/val")
    assert split == eq.EvalSplit(
        name="half",
        images=Path("/data/half/val"),
        labels=Path("/data/half/labels/val"),
    )
    assert [item.name for item in eq.parse_eval_splits(["a=/i:/l", "b=/i2:/l2"])] == ["a", "b"]
    with pytest.raises(SystemExit, match="NAME=IMAGES:LABELS"):
        eq.parse_eval_split("half")
    with pytest.raises(SystemExit, match="NAME=IMAGES:LABELS"):
        eq.parse_eval_split("half=/images")
    with pytest.raises(SystemExit, match="duplicate"):
        eq.parse_eval_splits(["a=/i:/l", "a=/i2:/l2"])


def test_split_indices_prefers_dump_groups_then_labels(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    grouped = eq.load_dump(
        _write_dump(
            tmp_path / "grouped.npz",
            _random_heads(rng, count=3),
            names=["a", "a", "c"],
            groups=["half", "full", "half"],
        )
    )
    spec = eq.EvalSplit("half", tmp_path / "images", tmp_path / "labels")
    assert eq.split_indices(grouped, spec) == [0, 2]

    labels = tmp_path / "legacy_labels"
    labels.mkdir()
    (labels / "a.txt").write_text("", encoding="utf-8")
    legacy = eq.load_dump(
        _write_dump(tmp_path / "legacy.npz", _random_heads(rng, count=2), names=["a", "b"])
    )
    legacy_spec = eq.EvalSplit("half", tmp_path / "images", labels)
    assert eq.split_indices(legacy, legacy_spec) == [0]


def test_render_split_markdown_has_one_column_per_split() -> None:
    half = eq.comparison_rows(_report(_all_groups(0.8)), None, None)
    full = eq.comparison_rows(_report(_all_groups(0.6)), None, None)
    table = eq.render_split_markdown(
        {"half": half, "full": full},
        {"int8": "INT8", "fp32_host": "n/a", "baseline": "n/a"},
        ["half", "full"],
    )
    lines = table.splitlines()
    assert lines[0] == "| metric | half INT8 | full INT8 |"
    assert "wrists.pck_all@0.20fw" in table
    assert len(lines) == 2 + len(half)


def test_comparison_rows_include_framing_fractions() -> None:
    int8 = _report(_all_groups(0.8))
    int8["framing"] = {
        "gt_box": {"full_body_fraction": 0.9},
        "predicted_box": {"full_body_fraction": 0.4, "unclassified_fraction": 0.1},
    }
    rows = {row["metric"]: row for row in eq.comparison_rows(int8, None, None)}

    assert rows["framing.gt_full_body_fraction"]["int8"] == 0.9
    assert rows["framing.pred_full_body_fraction"]["int8"] == 0.4
    assert rows["framing.pred_unclassified_fraction"]["int8"] == 0.1
    assert rows["framing.pred_half_body_fraction"]["int8"] is None
    assert rows["framing.gt_half_body_fraction"]["int8"] is None


def test_main_scores_named_dump_splits_side_by_side(tmp_path: Path) -> None:
    half_labels = tmp_path / "half_labels"
    full_labels = tmp_path / "full_labels"
    for labels in (half_labels, full_labels):
        labels.mkdir()
        (labels / "frame_0.txt").write_text(_label_line(), encoding="utf-8")
    rng = np.random.default_rng(13)
    dump = _write_dump(
        tmp_path / "quant_outputs.npz",
        _random_heads(rng, count=2),
        names=["frame_0", "frame_0"],
        groups=["half", "full"],
    )
    out = tmp_path / "out"

    assert (
        eq.main(
            [
                "--dump",
                str(dump),
                "--eval-split",
                f"half={tmp_path / 'half_images'}:{half_labels}",
                "--eval-split",
                f"full={tmp_path / 'full_images'}:{full_labels}",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    for name in ("half", "full"):
        metrics = json.loads((out / f"metrics_int8_{name}.json").read_text(encoding="utf-8"))
        assert metrics["meta"]["split"] == name
        assert metrics["meta"]["frames"] == 1

    table = (out / "comparison_splits.md").read_text(encoding="utf-8")
    assert table.splitlines()[0] == "| metric | half INT8 | full INT8 |"
    summary = json.loads((out / "comparison_splits.json").read_text(encoding="utf-8"))
    assert set(summary["splits"]) == {"half", "full"}
    assert summary["decision"]["status"] == "unknown"

    primary = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert primary["split"] == "half"
    assert (out / "metrics_int8.json").is_file()
    assert (out / "comparison.md").is_file()


def test_split_evaluation_rejects_an_unmatched_name(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "frame_0.txt").write_text(_label_line(), encoding="utf-8")
    rng = np.random.default_rng(14)
    dump = _write_dump(
        tmp_path / "quant_outputs.npz",
        _random_heads(rng, count=1),
        names=["frame_0"],
        groups=["half"],
    )

    with pytest.raises(SystemExit, match="no dumped frame"):
        eq.main(
            [
                "--dump",
                str(dump),
                "--eval-split",
                f"full={tmp_path / 'images'}:{labels}",
                "--out",
                str(tmp_path / "out"),
            ]
        )
