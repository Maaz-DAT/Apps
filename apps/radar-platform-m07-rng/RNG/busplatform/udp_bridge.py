"""
platform/udp_bridge.py — cross-process bridge (optional, REQ-M02-1 extension).

UdpSender: subscribes to the local bus, fires each frame out as one UDP
datagram (best-effort, no ack -- matches the bus's own "fast, lossy
under extreme load" character; use a wired TCP/IPC transport instead if
guaranteed delivery across processes is ever required).

UdpReceiver: listens on a socket and re-publishes every frame it
receives onto a local bus, so a second process sees the same stream.

Datagram size note: default UDP payload ceiling is ~65507 bytes; large
IQBlock frames should be chunked by the caller or sent via a different
transport -- this bridge is sized for PDW/IQDW/Bearing-class messages.
"""
from __future__ import annotations
import socket
import threading

from .bus import MessageBus
from .wire import Frame, pack, unpack_from

_MAX_DATAGRAM = 65000


class UdpSender:
    def __init__(self, bus: MessageBus, dest: tuple[str, int], topics: tuple[str, ...] = ("*",)):
        self.bus = bus
        self.dest = dest
        self.topics = topics
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dropped_oversize = 0

    def _on_frame(self, frame: Frame) -> None:
        b = pack(frame)
        if len(b) > _MAX_DATAGRAM:
            self.dropped_oversize += 1
            return
        self._sock.sendto(b, self.dest)

    def start(self) -> None:
        for t in self.topics:
            self.bus.subscribe(t, self._on_frame)

    def stop(self) -> None:
        for t in self.topics:
            self.bus.unsubscribe(t, self._on_frame)
        self._sock.close()


class UdpReceiver:
    def __init__(self, bus: MessageBus, bind: tuple[str, int]):
        self.bus = bus
        self.bind = bind
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(bind)
        self._sock.settimeout(0.5)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                data, _addr = self._sock.recvfrom(_MAX_DATAGRAM + 64)
            except socket.timeout:
                continue
            try:
                frame, _ = unpack_from(data, 0)
            except Exception:
                continue  # malformed datagram, drop (REQ-M02-4 style tolerance)
            self.bus.publish(frame.topic, frame.payload, t_ns=frame.t_ns)

    def start(self) -> None:
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sock.close()
