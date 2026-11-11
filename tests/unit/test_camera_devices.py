"""``auto`` camera selection from fake V4L2 listings (NFR-05, no hardware)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cpsa.adapters.camera.devices import (
    DEFAULT_MATCH,
    DEPTH_FOURCC,
    RGB_FOURCC,
    VideoNode,
    list_video_nodes,
    resolve_device,
    select_video_node,
)


def rgb_node(path: str = "/dev/video1", card: str = "F200 RGB") -> VideoNode:
    return VideoNode(path=path, card=card, formats=frozenset({RGB_FOURCC, "MJPG"}))


def depth_node(path: str = "/dev/video2", card: str = "F200 Depth") -> VideoNode:
    return VideoNode(path=path, card=card, formats=frozenset({DEPTH_FOURCC}))


def test_rgb_node_is_chosen_by_card_and_formats() -> None:
    nodes = [
        VideoNode("/dev/video0", "Integrated Webcam", frozenset({RGB_FOURCC})),
        depth_node(),
        rgb_node("/dev/video3"),
    ]

    chosen = select_video_node(nodes, match="F200")

    assert chosen.path == "/dev/video3"


def test_a_node_offering_yuyv_and_z16_is_not_the_rgb_node() -> None:
    both = VideoNode("/dev/video0", "F200", frozenset({RGB_FOURCC, DEPTH_FOURCC}))
    depth = depth_node("/dev/video1")

    with pytest.raises(RuntimeError, match="no camera node"):
        select_video_node([both], match="F200")
    chosen = select_video_node([depth], match="F200", require=DEPTH_FOURCC, forbid=())
    assert chosen.path == "/dev/video1"


def test_match_is_case_insensitive_and_skips_other_cards() -> None:
    nodes = [rgb_node("/dev/video0", "creative vf0800 f200"), rgb_node("/dev/video1", "F200 RGB")]

    assert select_video_node(nodes, match="f200").path == "/dev/video0"
    with pytest.raises(RuntimeError, match="no camera node"):
        select_video_node(nodes, match="C270")


def test_the_default_match_finds_the_real_f200_card_names() -> None:
    # Card names and formats as the KV260 kernel reports the Creative VF0800 (F200).
    card = "Intel(R) RealSense(TM) 3D Camer"
    nodes = [
        VideoNode("/dev/video1", card, frozenset({RGB_FOURCC})),
        VideoNode("/dev/video2", card, frozenset()),
        VideoNode("/dev/video3", card, frozenset({RGB_FOURCC, DEPTH_FOURCC, "INZI", "Y10 "})),
        VideoNode("/dev/video4", card, frozenset()),
    ]

    assert select_video_node(nodes, match=DEFAULT_MATCH).path == "/dev/video1"
    depth = select_video_node(nodes, match=DEFAULT_MATCH, require=DEPTH_FOURCC, forbid=())
    assert depth.path == "/dev/video3"


def test_no_candidate_lists_what_was_found() -> None:
    nodes = [VideoNode("/dev/video0", "Integrated Webcam", frozenset({"MJPG"}))]

    with pytest.raises(RuntimeError) as info:
        select_video_node(nodes, match="F200")

    message = str(info.value)
    assert "F200" in message
    assert "/dev/video0 'Integrated Webcam' [MJPG]" in message


def test_explicit_device_is_returned_unchanged_without_listing() -> None:
    def lister() -> list[VideoNode]:
        raise AssertionError("an explicit device must not probe V4L2")

    assert resolve_device("/dev/video0", lister=lister) == "/dev/video0"


def test_auto_device_is_resolved_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="cpsa.adapters.camera.devices"):
        resolved = resolve_device("auto", match="F200", lister=lambda: [depth_node(), rgb_node()])

    assert resolved == "/dev/video1"
    assert "Camera device 'auto' selected /dev/video1" in caplog.text


def test_auto_without_a_camera_raises() -> None:
    with pytest.raises(RuntimeError, match="no V4L2 nodes"):
        resolve_device("auto", lister=list)


def test_list_video_nodes_orders_by_index_and_skips_unreadable_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    linuxpy_device = pytest.importorskip("linuxpy.video.device")
    for name in ("video10", "video2", "video0"):
        (tmp_path / name).touch()

    class FakeFormat:
        def __init__(self, name: str) -> None:
            self.pixel_format = type("PixelFormat", (), {"name": name})()

    class FakeDevice:
        def __init__(self, path: str, read_write: bool = True) -> None:
            self.path = path

        def __enter__(self) -> FakeDevice:
            if self.path.endswith("video10"):
                raise OSError("device is busy")
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        @property
        def info(self) -> object:
            return type(
                "Info",
                (),
                {"card": "F200 RGB", "formats": [FakeFormat(RGB_FOURCC)]},
            )()

    monkeypatch.setattr(linuxpy_device, "Device", FakeDevice)

    nodes = list_video_nodes(tmp_path)

    assert [node.path for node in nodes] == [str(tmp_path / "video0"), str(tmp_path / "video2")]
    assert nodes[0].card == "F200 RGB"
    assert nodes[0].formats == frozenset({RGB_FOURCC})
