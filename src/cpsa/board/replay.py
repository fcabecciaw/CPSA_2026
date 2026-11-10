"""``cpsa-replay-imu SESSION_DIR --out tags.jsonl``: replay a session's raw IMU.

The rows are fed in recorded timestamp order through the live pipeline stages
(``IMUSynchronizer`` -> ``Windower`` -> ``ClassificationStage``) with a clock that
follows the recorded timestamps, so the synchronizer's skew/stale behavior matches
a live run. The native extractor/classifier are aarch64-only.

Exit codes: 0 on success, 1 on an unexpected error, 2 for an invalid session or an
unsupported host/native library.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from cpsa.adapters.native.classifier import (
    NativeClassifier,
    NativeFeatureExtractor,
    NativeLibraryError,
)
from cpsa.adapters.recording.reader import Session, SessionFormatError
from cpsa.application.classification import ClassificationStage
from cpsa.application.slot import LatestSlot
from cpsa.domain.model import Detection, SampleKind, Side
from cpsa.imu.sync import IMUSynchronizer
from cpsa.imu.window import Windower
from cpsa.ports import Classifier, FeatureExtractor

logger = logging.getLogger("cpsa.replay")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STARTUP = 2


class ReplayUnavailable(RuntimeError):
    """Native classification is unavailable on this host."""


@dataclass(frozen=True)
class ReplayTag:
    """One detection written to ``tags.jsonl``."""

    window_end_ts: float
    tag: str
    timestamp: str

    def to_json(self) -> dict[str, object]:
        return {
            "window_end_ts": self.window_end_ts,
            "tag": self.tag,
            "timestamp": self.timestamp,
        }


class ReplayClock:
    """A clock that follows the recorded timestamps instead of wall time."""

    def __init__(self, base: datetime, first_ts: float) -> None:
        self._base = base
        self._first_ts = first_ts
        self._ts = first_ts

    def set(self, ts: float) -> None:
        self._ts = ts

    def monotonic(self) -> float:
        return self._ts

    def now(self) -> datetime:
        return self._base + timedelta(seconds=self._ts - self._first_ts)


def native_stages() -> tuple[FeatureExtractor, Classifier]:
    """Load the aarch64 native extractor and classifier in baseline order."""
    if platform.machine() != "aarch64":
        raise ReplayUnavailable(
            f"cpsa-replay-imu needs the aarch64 native libraries; this host is {platform.machine()}"
        )
    try:
        return NativeFeatureExtractor(), NativeClassifier()
    except NativeLibraryError as exc:
        raise ReplayUnavailable(str(exc)) from exc


def replay_session(
    session: Session,
    out_path: Path,
    *,
    extractor: FeatureExtractor,
    classifier: Classifier,
) -> tuple[ReplayTag, ...]:
    """Replay ``session.imu`` and write one JSON line per detection."""
    rows = session.imu
    order = np.argsort(rows["ts"], kind="stable") if rows.size else np.empty(0, dtype=np.int64)
    first_ts = float(rows["ts"][order[0]]) if rows.size else 0.0

    clock = ReplayClock(session.created, first_ts)
    slot: LatestSlot[Detection] = LatestSlot()
    stage = ClassificationStage(extractor, classifier, slot, clock)
    windower = Windower(stage.on_window)
    synchronizer = IMUSynchronizer(
        session.sync_max_skew_s,
        session.sync_stale_s,
        windower.add,
        clock,
    )

    tags: list[ReplayTag] = []
    for index in order.tolist():
        row = rows[index]
        ts = float(row["ts"])
        clock.set(ts)
        side = Side(str(row["side"]))
        kind = SampleKind(str(row["kind"]))
        count = 4 if kind is SampleKind.QUAT else 3
        values = tuple(float(row[f"v{number}"]) for number in range(count))
        synchronizer.update(side, kind, values, ts)
        while True:
            detection = slot.get(0.0)
            if detection is None:
                break
            tags.append(
                ReplayTag(
                    window_end_ts=float(detection.window_end_ts),
                    tag=detection.tag.name,
                    timestamp=detection.timestamp.isoformat(),
                )
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as stream:
        for tag in tags:
            stream.write(json.dumps(tag.to_json()) + "\n")
    return tuple(tags)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpsa-replay-imu",
        description="Replay a recorded session's IMU through the native classifier.",
    )
    parser.add_argument("session_dir", type=Path, help="session directory (contains session.json)")
    parser.add_argument("--out", required=True, type=Path, help="output JSON Lines file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        session = Session.open(args.session_dir)
    except (SessionFormatError, OSError) as exc:
        print(f"cpsa-replay-imu: {exc}", file=sys.stderr)
        return EXIT_STARTUP

    try:
        extractor, classifier = native_stages()
    except ReplayUnavailable as exc:
        print(f"cpsa-replay-imu: {exc}", file=sys.stderr)
        return EXIT_STARTUP

    try:
        tags = replay_session(session, args.out, extractor=extractor, classifier=classifier)
    except Exception:
        logger.exception("Replay failed")
        return EXIT_ERROR

    print(f"Replayed {int(session.imu.size)} IMU samples: {len(tags)} tags -> {args.out}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
