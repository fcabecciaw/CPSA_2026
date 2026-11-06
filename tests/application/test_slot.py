import threading
import time

from cpsa.application.slot import LatestSlot


def test_get_returns_none_when_empty_after_timeout() -> None:
    assert LatestSlot[int]().get(timeout=0.01) is None


def test_newest_item_supersedes_unprocessed_one() -> None:
    slot = LatestSlot[int]()
    slot.put(1)
    slot.put(2)
    slot.put(3)
    assert slot.get(timeout=0) == 3
    assert slot.get(timeout=0) is None


def test_put_does_not_wait_for_a_consumer() -> None:
    slot = LatestSlot[int]()
    started = time.monotonic()
    for i in range(1000):
        slot.put(i)
    assert time.monotonic() - started < 1.0
    assert slot.get(timeout=0) == 999


def test_get_wakes_up_when_an_item_is_put() -> None:
    slot = LatestSlot[str]()
    result: list[str | None] = []
    consumer = threading.Thread(target=lambda: result.append(slot.get(timeout=5)))
    consumer.start()
    slot.put("detection")
    consumer.join(timeout=5)
    assert result == ["detection"]
