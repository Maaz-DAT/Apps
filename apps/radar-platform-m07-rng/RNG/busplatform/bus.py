"""
platform/bus.py — in-process pub/sub data bus (REQ-M02-1/2).

Design for speed, not generality:
  - publish() does direct synchronous callback dispatch to subscribers.
    No queue hop, no asyncio, no locking on the hot path (list.append /
    iteration are already atomic under the GIL) -- this is what gets
    >10k msg/s out of pure Python.
  - Subscribers that need to cross a thread/async boundary (e.g. the
    websocket dashboard) register a callback that itself does the
    minimal possible work (push onto a bounded deque) and drains it
    elsewhere -- see webapp/main.py.
  - A global sequence counter gives every message a monotonic seq,
    used by the recorder/replayer to detect reordering.
"""
from __future__ import annotations
import itertools
import threading
import time
from collections import defaultdict
from typing import Callable

from .wire import Frame

Callback = Callable[[Frame], None]


class MessageBus:
    __slots__ = ("_subs", "_seq", "_lock", "stats")

    def __init__(self) -> None:
        self._subs: dict[str, list[Callback]] = defaultdict(list)
        self._seq = itertools.count()
        self._lock = threading.Lock()  # only guards subscribe/unsubscribe
        self.stats = {"published": 0, "dropped": 0}

    def subscribe(self, topic: str, callback: Callback) -> None:
        with self._lock:
            self._subs[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callback) -> None:
        with self._lock:
            subs = self._subs.get(topic, [])
            if callback in subs:
                subs.remove(callback)

    def subscribe_ordered(
        self,
        topic: str,
        callback: Callback,
        window_ns: int = 200_000_000,
    ):
        """
        Subscribe with an online reorder window (REQ-M02-4).

        Frames are held up to `window_ns` (default 200 ms) and delivered
        to `callback` in ascending t_ns order.  Messages older than the
        window are counted as late and delivered immediately.

        Returns the ReorderBuffer so the caller can read late_count /
        pending and call flush() at end-of-stream.
        """
        from .reorder import ReorderBuffer
        buf = ReorderBuffer(window_ns=window_ns, on_ready=callback)
        self.subscribe(topic, buf.push)
        return buf

    def publish(self, topic: str, payload: bytes, t_ns: int | None = None) -> Frame:
        """Hot path. No lock: list snapshot read is fine, missing a
        subscriber added microseconds ago is an acceptable race for a
        pub/sub bus (new subscribers naturally only see future messages).
        """
        frame = Frame(
            topic=topic,
            t_ns=t_ns if t_ns is not None else time.time_ns(),
            seq=next(self._seq),
            payload=payload,
        )
        self.stats["published"] += 1
        for cb in self._subs.get(topic, ()):
            cb(frame)
        for cb in self._subs.get("*", ()):   # wildcard subscribers (e.g. recorder)
            cb(frame)
        return frame
