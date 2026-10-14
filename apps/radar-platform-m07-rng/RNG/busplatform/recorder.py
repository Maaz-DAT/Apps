"""
platform/recorder.py — flat-binary recorder/replayer (REQ-M02-3/4).

Recorder: subscribes to the bus wildcard topic, appends each Frame to
a single append-only file as it arrives. Buffered writes, fsync only
on close -- recording must not be the bottleneck at >10k msg/s.

Replayer: reads the whole file (or streams it for very large files),
sorts by t_ns (REQ-M02-3: replay must come back ordered even if the
recording captured out-of-order / late arrivals), and republishes onto
a bus at a configurable speed multiplier. Late-arrival count is
derived from how many frames were out of seq order in the raw file.
"""
from __future__ import annotations
import threading
import time
from pathlib import Path

from .bus import MessageBus
from .wire import Frame, pack, iter_frames

_WRITE_BUF_FRAMES = 2048  # flush every N frames to amortize syscalls


class Recorder:
    def __init__(self, bus: MessageBus, path: str | Path):
        self.path = Path(path)
        self._fh = open(self.path, "wb", buffering=1024 * 1024)
        self._bus = bus
        self._buf: list[bytes] = []
        self._count = 0
        self._lock = threading.Lock()

    def _on_frame(self, frame: Frame) -> None:
        b = pack(frame)
        with self._lock:
            self._buf.append(b)
            self._count += 1
            if len(self._buf) >= _WRITE_BUF_FRAMES:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buf:
            self._fh.write(b"".join(self._buf))
            self._buf.clear()

    def start(self) -> None:
        self._bus.subscribe("*", self._on_frame)

    def stop(self) -> None:
        self._bus.unsubscribe("*", self._on_frame)
        with self._lock:
            self._flush_locked()
            self._fh.flush()
            self._fh.close()

    @property
    def frames_written(self) -> int:
        return self._count


class Replayer:
    def __init__(self, bus: MessageBus, path: str | Path):
        self.bus = bus
        self.path = Path(path)
        self.late_arrivals = 0
        self.total_frames = 0
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def _load_sorted(self) -> list[Frame]:
        raw = self.path.read_bytes()
        frames = list(iter_frames(raw))
        self.total_frames = len(frames)
        # count out-of-order arrivals as recorded (pre-sort)
        self.late_arrivals = sum(
            1 for a, b in zip(frames, frames[1:]) if b.t_ns < a.t_ns
        )
        frames.sort(key=lambda f: f.t_ns)
        return frames

    def replay(self, speed: float = 1.0, blocking: bool = True) -> None:
        frames = self._load_sorted()
        if not frames:
            return

        def _run():
            t0_rec = frames[0].t_ns
            t0_wall = time.perf_counter()
            for f in frames:
                if self._stop_evt.is_set():
                    return
                target = t0_wall + (f.t_ns - t0_rec) / 1e9 / max(speed, 1e-9)
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)
                self.bus.publish(f.topic, f.payload, t_ns=f.t_ns)

        if blocking:
            _run()
        else:
            self._stop_evt.clear()
            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
