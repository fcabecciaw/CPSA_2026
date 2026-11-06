"""Single-slot, drop-oldest hand-off between the classification stage and the dispatcher."""

from __future__ import annotations

import logging
import threading
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Holds at most one item; a new item replaces any item not yet taken.

    ``put`` never waits for the consumer, so producers are never blocked by a slow
    dispatcher; ``get`` waits up to ``timeout`` seconds for an item.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._full = False

    def put(self, item: T) -> None:
        with self._cond:
            if self._full:
                logger.debug("Unprocessed item superseded by a newer one")
            self._item = item
            self._full = True
            self._cond.notify()

    def get(self, timeout: float) -> T | None:
        with self._cond:
            if not self._cond.wait_for(lambda: self._full, timeout):
                return None
            item, self._item, self._full = self._item, None, False
            return item
