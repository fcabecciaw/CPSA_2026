"""File evidence sink: skeleton clip and keyframe per episode (FR-021, NFR-005).

Layout::

    <base>/<YYYY-MM-DD>/<episode_id>/clip.mp4   (MP4V; MJPG AVI fallback)
    <base>/<YYYY-MM-DD>/<episode_id>/keyframe.jpg

The clip is written with OpenCV's ``mp4v`` encoder because the KV260 ships the
same OpenCV build the rest of the vision extra uses. If the fourcc is not
available (some minimal builds), the sink falls back to an MJPG AVI and keeps the
same folder layout.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cpsa.vision.types import KeypointLayout, KeypointStatus, TrackedFrame

logger = logging.getLogger(__name__)

__all__ = ["FileEvidenceSink"]

_KEYPOINT_CONF = 0.3
_KEYFRAME_QUALITY = 95


class FileEvidenceSink:
    """Write one clip and one keyframe per episode and purge old day folders.

    Args:
        base_path: root of the evidence tree.
        retention_days: day folders older than this many days are removed by
            :meth:`purge`; ``0`` keeps only today.
        layout: keypoint layout used for the skeleton overlay; ``None`` draws
            keypoints only.
    """

    def __init__(
        self,
        base_path: Path,
        retention_days: int,
        layout: KeypointLayout | None = None,
    ) -> None:
        if retention_days < 0:
            raise ValueError("retention_days must be >= 0")
        self._base_path = Path(base_path)
        self._retention_days = int(retention_days)
        self._layout = layout

    @property
    def base_path(self) -> Path:
        return self._base_path

    def episode_folder(self, episode_id: str, day: date | None = None) -> Path:
        """Return the folder of one episode; ``day`` defaults to today."""
        current = day if day is not None else datetime.now().astimezone().date()
        return self._base_path / current.isoformat() / episode_id

    def write_episode(
        self,
        episode_id: str,
        frames: Sequence[tuple[NDArray[np.uint8], TrackedFrame | None]],
        fps: float,
        keyframe: int,
    ) -> None:
        """Save the clip with a skeleton overlay and the keyframe ``frames[keyframe]``."""
        if not episode_id or "/" in episode_id or "\\" in episode_id:
            raise ValueError(f"invalid episode_id {episode_id!r}")
        if not frames:
            raise ValueError("evidence needs at least one frame")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be a finite positive number")
        if not 0 <= keyframe < len(frames):
            raise ValueError(f"keyframe {keyframe} is outside 0..{len(frames) - 1}")
        images = [self._overlay(rgb, tracked) for rgb, tracked in frames]
        if any(image.shape != images[0].shape for image in images):
            raise ValueError("all evidence frames must have the same shape")
        folder = self.episode_folder(episode_id)
        folder.mkdir(parents=True, exist_ok=True)
        self._write_keyframe(images[keyframe], folder / "keyframe.jpg")
        self._write_clip(images, fps, folder / "clip.mp4")

    def purge(self, now: datetime) -> None:
        """Remove day folders older than the configured retention (NFR-005)."""
        if not self._base_path.is_dir():
            return
        cutoff = now.date() - timedelta(days=self._retention_days)
        for child in sorted(self._base_path.iterdir()):
            if not child.is_dir():
                continue
            try:
                day = date.fromisoformat(child.name)
            except ValueError:
                continue
            if day < cutoff:
                try:
                    shutil.rmtree(child)
                except OSError:
                    logger.exception("Failed to purge evidence folder %s", child)
                else:
                    logger.info("Purged evidence folder %s", child)

    # -- rendering --------------------------------------------------------

    def _overlay(self, rgb: NDArray[np.uint8], tracked: TrackedFrame | None) -> NDArray[np.uint8]:
        """Return a BGR frame with the tracked skeleton drawn on top."""
        cv2 = _cv2()
        array = np.asarray(rgb)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected uint8 RGB image, got {array.dtype} {array.shape}")
        canvas = cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_RGB2BGR)
        if tracked is None:
            return canvas
        layout = self._layout
        xy = np.asarray(tracked.xy, dtype=np.float64)
        conf = np.asarray(tracked.conf, dtype=np.float64)
        status = np.asarray(tracked.status, dtype=np.int8)
        visible = (
            (status != int(KeypointStatus.LOST))
            & (conf >= _KEYPOINT_CONF)
            & np.all(np.isfinite(xy), axis=1)
        )
        if layout is not None:
            for first, second in layout.edges:
                if not (visible[first] and visible[second]):
                    continue
                start = (int(round(xy[first][0])), int(round(xy[first][1])))
                end = (int(round(xy[second][0])), int(round(xy[second][1])))
                cv2.line(canvas, start, end, (0, 255, 0), 2)
        for index in np.flatnonzero(visible):
            point = (int(round(xy[index][0])), int(round(xy[index][1])))
            cv2.circle(canvas, point, 2, (0, 0, 255), -1)
        return canvas

    def _write_keyframe(self, image: NDArray[np.uint8], path: Path) -> None:
        cv2 = _cv2()
        parameters: list[Any] = [int(cv2.IMWRITE_JPEG_QUALITY), _KEYFRAME_QUALITY]
        if not cv2.imwrite(str(path), image, parameters):
            raise RuntimeError(f"cv2 could not write {path}")

    def _write_clip(self, images: Sequence[NDArray[np.uint8]], fps: float, path: Path) -> None:
        cv2 = _cv2()
        height, width = images[0].shape[:2]
        fourcc, target = cv2.VideoWriter_fourcc(*"mp4v"), path
        writer = cv2.VideoWriter(str(target), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            writer.release()
            target = path.with_suffix(".avi")
            writer = cv2.VideoWriter(
                str(target), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (width, height)
            )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"cv2 could not open a video writer for {path}")
        try:
            for image in images:
                writer.write(image)
        finally:
            writer.release()


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the vision extra
        raise RuntimeError(
            "OpenCV is required for file evidence; install the 'vision' extra"
        ) from exc
    return cv2
