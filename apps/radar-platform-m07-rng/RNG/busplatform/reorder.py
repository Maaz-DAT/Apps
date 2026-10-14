"""
platform/reorder.py — online reorder window (REQ-M02-4).

Out-of-order frames are held in a time-ordered buffer and released in
ascending t_ns order once the watermark advances past them.

Default window = 200 ms.  A frame whose t_ns is older than
(max_seen_t_ns − window_ns) is counted as late and delivered
immediately (still in the caller's callback) so no data is silently
dropped.

Usage patterns
--------------
1. Stand-alone (recommended for consumers that need ordered delivery):

    buf = ReorderBuffer(window_ns=200_000_000, on_ready=my_handler)
    bus.subscribe("pdw", buf.push)

2. Helper on the bus:

    bus.subscribe_ordered("pdw", my_handler, window_ns=200_000_000)
"""

from __future__ import annotations

import heapq
import threading
import time
from typing import Callable

from .wire import Frame

Callback = Callable[[Frame], None]

DEFAULT_WINDOW_NS = 200_000_000  # 200 ms


class ReorderBuffer:
    """
    Hold frames for up to `window_ns` and release them sorted by t_ns.

    Thread-safe.  push() may be called from multiple producer threads;
    on_ready callbacks run under the same lock (keep them short).
    """

    __slots__ = (
        "window_ns",
        "_on_ready",
        "_heap",          # list of (t_ns, seq, frame)
        "_max_t_ns",
        "_late",
        "_released",
        "_lock",
    )

    def __init__(
        self,
        window_ns: int = DEFAULT_WINDOW_NS,
        on_ready: Callback | None = None,
    ) -> None:
        if window_ns < 0:
            raise ValueError("window_ns must be >= 0")
        self.window_ns = window_ns
        self._on_ready = on_ready
        self._heap: list[tuple[int, int, Frame]] = []
        self._max_t_ns: int = 0
        self._late: int = 0
        self._released: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, frame: Frame) -> None:
        """Insert a frame and release everything that is now due.

        Frames older than (max_seen_t_ns − window_ns) are counted as late
        and dropped so the delivered stream stays strictly sorted by t_ns.
        """
        with self._lock:
            t = frame.t_ns
            if t > self._max_t_ns:
                self._max_t_ns = t

            watermark = self._max_t_ns - self.window_ns

            # Late: window already closed for this timestamp → count & drop
            if t < watermark:
                self._late += 1
                ready = self._drain_locked(watermark)
            else:
                heapq.heappush(self._heap, (t, frame.seq, frame))
                ready = self._drain_locked(watermark)

        if self._on_ready is not None:
            for f in ready:
                self._on_ready(f)

    def flush(self) -> None:
        """Force-release every still-buffered frame (e.g. end of stream)."""
        with self._lock:
            ready = []
            while self._heap:
                _, _, f = heapq.heappop(self._heap)
                ready.append(f)
                self._released += 1
        if self._on_ready is not None:
            for f in ready:
                self._on_ready(f)

    def set_callback(self, on_ready: Callback | None) -> None:
        with self._lock:
            self._on_ready = on_ready

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def late_count(self) -> int:
        return self._late

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def released_count(self) -> int:
        return self._released

    def stats(self) -> dict:
        with self._lock:
            return {
                "window_ns": self.window_ns,
                "pending": len(self._heap),
                "late": self._late,
                "released": self._released,
                "max_t_ns": self._max_t_ns,
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _drain_locked(self, watermark: int) -> list[Frame]:
        """Pop every frame with t_ns <= watermark. Caller holds lock."""
        ready: list[Frame] = []
        while self._heap and self._heap[0][0] <= watermark:
            _, _, f = heapq.heappop(self._heap)
            ready.append(f)
            self._released += 1
        return ready


# ---------------------------------------------------------------------------
# Convenience: ordered subscribe helper (monkey-patched onto MessageBus)
# ---------------------------------------------------------------------------

def subscribe_ordered(
    bus,
    topic: str,
    callback: Callback,
    window_ns: int = DEFAULT_WINDOW_NS,
) -> ReorderBuffer:
    """
    Subscribe with an online reorder window.

    Returns the ReorderBuffer so the caller can inspect late_count /
    pending / call flush() at the end of a session.
    """
    buf = ReorderBuffer(window_ns=window_ns, on_ready=callback)
    bus.subscribe(topic, buf.push)
    return buf
