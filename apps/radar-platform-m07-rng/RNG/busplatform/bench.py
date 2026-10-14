"""
platform/bench.py — quick TM-02/TM-03-style check: throughput, zero loss,
record -> replay -> sorted-by-t_ns, reordering detection.

Run: python -m platform.bench   (from project root)
"""
from __future__ import annotations
import os
import random
import tempfile
import time

from .bus import MessageBus
from .recorder import Recorder, Replayer

# Real M-00 PDW payload for a realistic bench
try:
    from common.messages import PDW, FLAG_VALID
    _sample = PDW(
        t_ns=0, source_id="bench", toa_ns=0,
        frequency_hz=9.4e9, pulse_width_s=1e-6,
        phase=[0.1, 0.2, 0.3, 0.4],
        amplitude=[1.0, 1.0, 1.0, 1.0],
        flags=FLAG_VALID,
    )
    REAL_PDW_PAYLOAD = _sample.to_bytes()
except Exception:
    REAL_PDW_PAYLOAD = b"\x00" * 120


def bench_throughput(n_messages: int = 1_000_000) -> None:
    bus = MessageBus()
    received = {"n": 0}

    def _sink(frame):
        received["n"] += 1

    bus.subscribe("pdw", _sink)
    payload = REAL_PDW_PAYLOAD

    t0 = time.perf_counter()
    for i in range(n_messages):
        bus.publish("pdw", payload)
    dt = time.perf_counter() - t0

    rate = n_messages / dt
    print(f"[throughput] {n_messages:,} msgs in {dt:.3f}s -> {rate:,.0f} msg/s "
          f"(target >=10,000) -- {'PASS' if rate >= 10_000 else 'FAIL'}")
    print(f"[throughput] received={received['n']} published={bus.stats['published']} "
          f"loss={n_messages - received['n']}  payload={len(payload)} B")


def bench_record_replay(n_messages: int = 5_000) -> None:
    bus = MessageBus()
    path = os.path.join(tempfile.gettempdir(), "bench_record.bin")

    rec = Recorder(bus, path)
    rec.start()

    # publish out of order on purpose to exercise reordering handling
    base_t = time.time_ns()
    timestamps = [base_t + i * 1_000_000 for i in range(n_messages)]
    shuffled = timestamps[:]
    random.Random(42).shuffle(shuffled)
    for t_ns in shuffled:
        bus.publish("pdw", b"\x11" * 32, t_ns=t_ns)

    rec.stop()
    print(f"[record] wrote {rec.frames_written} frames to {path}")

    replay_bus = MessageBus()
    out_order = []
    replay_bus.subscribe("pdw", lambda f: out_order.append(f.t_ns))

    replayer = Replayer(replay_bus, path)
    replayer.replay(speed=1000.0, blocking=True)  # fast-forward for the test

    is_sorted = out_order == sorted(out_order)
    print(f"[replay] {len(out_order)} frames replayed, sorted={is_sorted} "
          f"late_arrivals_detected={replayer.late_arrivals} "
          f"-- {'PASS' if is_sorted and len(out_order) == n_messages else 'FAIL'}")


def bench_online_reorder(n_messages: int = 2_000, window_ms: float = 200.0) -> None:
    """
    REQ-M02-4 / TM-03 style: feed a deliberately shuffled stream through
    an online ReorderBuffer and verify:
      - output is sorted by t_ns
      - frames older than the window are counted as late
      - no frames are lost
    """
    from .reorder import ReorderBuffer

    window_ns = int(window_ms * 1e6)
    delivered: list[int] = []

    buf = ReorderBuffer(
        window_ns=window_ns,
        on_ready=lambda f: delivered.append(f.t_ns),
    )

    base_t = 1_000_000_000_000  # arbitrary epoch
    # spacing 1 ms so a 200 ms window holds ~200 messages
    timestamps = [base_t + i * 1_000_000 for i in range(n_messages)]
    shuffled = timestamps[:]
    rng = random.Random(42)
    rng.shuffle(shuffled)

    # Also inject a few *very* late frames (older than window) near the end
    n_forced_late = 5
    late_ts = [base_t - (i + 1) * 1_000_000 for i in range(n_forced_late)]

    for t_ns in shuffled:
        from .wire import Frame
        buf.push(Frame(topic="pdw", t_ns=t_ns, seq=0, payload=b""))
    for t_ns in late_ts:
        from .wire import Frame
        buf.push(Frame(topic="pdw", t_ns=t_ns, seq=0, payload=b""))

    buf.flush()  # end-of-stream: release anything still held

    is_sorted = delivered == sorted(delivered)
    total_pushed = n_messages + n_forced_late
    # Invariant: every pushed frame is either delivered or counted late
    accounted = len(delivered) + buf.late_count
    no_loss = accounted == total_pushed
    late_ok = buf.late_count >= n_forced_late

    print(
        f"[reorder] window={window_ms:.0f} ms  delivered={len(delivered)}  "
        f"late={buf.late_count}  accounted={accounted}/{total_pushed}  "
        f"sorted={is_sorted}  pending={buf.pending}  "
        f"-- {'PASS' if is_sorted and no_loss and late_ok else 'FAIL'}"
    )


if __name__ == "__main__":
    bench_throughput()
    bench_record_replay()
    bench_online_reorder()
