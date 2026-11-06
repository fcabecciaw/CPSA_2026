"""Wall-clock and monotonic time from the operating system."""

from __future__ import annotations

import time
from datetime import datetime


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def now(self) -> datetime:
        """Current local time, timezone-aware."""
        return datetime.now().astimezone()
