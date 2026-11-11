"""V4L2 node discovery for ``auto`` camera selection (NFR-05).

The Creative VF0800 / Intel RealSense F200 re-enumerates when the USB cable is
re-plugged: the RGB node that was ``/dev/video0`` may come back as
``/dev/video1``, so a hard-coded path silently captures nothing. A source
configured with ``device: auto`` resolves the node at every start instead.

Selection is by card name and offered pixel formats: the RGB node of the F200
offers ``YUYV`` and no ``Z16``, while its depth node offers ``Z16``. Explicit
``/dev/videoN`` paths are used unchanged.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

AUTO = "auto"
"""Device value that asks the source to discover its V4L2 node."""

DEFAULT_MATCH = "F200"
"""Card-name fragment that the demo camera reports over V4L2."""

RGB_FOURCC = "YUYV"
DEPTH_FOURCC = "Z16"

__all__ = [
    "AUTO",
    "DEFAULT_MATCH",
    "DEPTH_FOURCC",
    "RGB_FOURCC",
    "VideoNode",
    "list_video_nodes",
    "resolve_device",
    "select_video_node",
]


@dataclass(frozen=True)
class VideoNode:
    """One V4L2 node: its device path, card name and offered pixel formats."""

    path: str
    card: str
    formats: frozenset[str]

    def __str__(self) -> str:
        offered = ", ".join(sorted(self.formats)) or "no pixel formats"
        return f"{self.path} {self.card!r} [{offered}]"


def _node_index(path: str) -> int:
    """Numeric suffix of a ``videoN`` path, so video2 sorts before video10."""

    match = re.search(r"(\d+)$", path)
    return -1 if match is None else int(match.group(1))


def list_video_nodes(directory: Path = Path("/dev")) -> list[VideoNode]:
    """Enumerate the V4L2 capture nodes in ``directory``.

    Nodes that cannot be queried (metadata nodes, busy devices, permission
    errors) are skipped: a working camera may sit next to them.

    Raises:
        RuntimeError: If ``linuxpy`` is unavailable.
    """

    try:
        from linuxpy.video.device import Device
    except ImportError as exc:  # pragma: no cover - depends on the installed extra
        raise RuntimeError(
            "linuxpy is required for camera auto-detection; install the 'camera' extra"
        ) from exc

    nodes: list[VideoNode] = []
    for path in sorted(directory.glob("video*"), key=lambda entry: _node_index(str(entry))):
        try:
            with Device(str(path), read_write=False) as device:
                card = device.info.card
                formats = frozenset(fmt.pixel_format.name for fmt in device.info.formats)
        except Exception:
            logger.debug("Skipping %s: it cannot be queried", path, exc_info=True)
            continue
        nodes.append(VideoNode(path=str(path), card=card, formats=formats))
    return nodes


def select_video_node(
    nodes: Sequence[VideoNode],
    *,
    match: str,
    require: str = RGB_FOURCC,
    forbid: tuple[str, ...] = (DEPTH_FOURCC,),
) -> VideoNode:
    """Return the first node whose card contains ``match`` and whose formats fit.

    Args:
        nodes: Candidate nodes, already in preference order.
        match: Case-insensitive fragment of the V4L2 card name.
        require: Pixel format the node must offer.
        forbid: Pixel formats the node must not offer; the RGB rule excludes the
            depth format so the two nodes of one camera never collide.

    Raises:
        RuntimeError: If no candidate fits; the message lists what was found.
    """

    wanted = match.casefold()
    excluded = frozenset(forbid)
    for node in nodes:
        if (
            wanted in node.card.casefold()
            and require in node.formats
            and not excluded & node.formats
        ):
            return node
    found = "; ".join(str(node) for node in nodes) or "no V4L2 nodes"
    constraints = require if not forbid else f"{require} without {'/'.join(forbid)}"
    raise RuntimeError(f"no camera node matching {match!r} with {constraints}; found: {found}")


def resolve_device(
    device: str,
    *,
    match: str = DEFAULT_MATCH,
    require: str = RGB_FOURCC,
    forbid: tuple[str, ...] = (DEPTH_FOURCC,),
    lister: Callable[[], Sequence[VideoNode]] = list_video_nodes,
) -> str:
    """Resolve a configured device to a V4L2 path.

    An explicit path is returned unchanged; ``"auto"`` picks the matching node
    and logs the choice, so a log records which node served a run.

    Args:
        device: Configured device path, or ``"auto"``.
        match: Card-name fragment used for ``"auto"``.
        require: Pixel format the node must offer.
        forbid: Pixel formats the node must not offer.
        lister: Test seam returning the available nodes; defaults to
            :func:`list_video_nodes`.
    """

    if device.strip().casefold() != AUTO:
        return device
    node = select_video_node(list(lister()), match=match, require=require, forbid=forbid)
    logger.info(
        "Camera device 'auto' selected %s (card %r, formats %s)",
        node.path,
        node.card,
        ", ".join(sorted(node.formats)),
    )
    return node.path
