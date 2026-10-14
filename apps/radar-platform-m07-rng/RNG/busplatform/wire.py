"""
platform/wire.py — binary frame format shared by the bus, recorder/replayer
and UDP bridge (REQ-M02-*).

Frame layout (little-endian, no padding):
    topic_len : uint8
    topic     : topic_len bytes, ascii
    t_ns      : uint64   UTC nanoseconds (message timestamp, not wall clock)
    seq       : uint64   monotonic publisher sequence number
    length    : uint32   payload length in bytes
    payload   : length bytes (already-encoded message, e.g. encode_pdw())

Kept as raw bytes end-to-end (bus never decodes payload) so publish/record/
replay/UDP all stay on the fast path — decoding only happens where a
consumer actually needs the fields (e.g. the dashboard).
"""
from __future__ import annotations
import struct
from dataclasses import dataclass

_HDR_FIXED = struct.Struct("<B")        # topic_len
_HDR_REST = struct.Struct("<QQI")       # t_ns, seq, length


@dataclass(slots=True)
class Frame:
    topic: str
    t_ns: int
    seq: int
    payload: bytes


def pack(frame: Frame) -> bytes:
    topic_b = frame.topic.encode("ascii")
    if len(topic_b) > 255:
        raise ValueError("topic must be <= 255 bytes")
    return (
        _HDR_FIXED.pack(len(topic_b))
        + topic_b
        + _HDR_REST.pack(frame.t_ns, frame.seq, len(frame.payload))
        + frame.payload
    )


def unpack_from(buf: bytes | memoryview, offset: int = 0) -> tuple[Frame, int]:
    """Decode one frame starting at offset. Returns (frame, new_offset)."""
    (topic_len,) = _HDR_FIXED.unpack_from(buf, offset)
    offset += 1
    topic = bytes(buf[offset:offset + topic_len]).decode("ascii")
    offset += topic_len
    t_ns, seq, length = _HDR_REST.unpack_from(buf, offset)
    offset += _HDR_REST.size
    payload = bytes(buf[offset:offset + length])
    offset += length
    return Frame(topic, t_ns, seq, payload), offset


def iter_frames(buf: bytes):
    """Yield every complete frame in buf. Stops cleanly on a truncated tail."""
    offset = 0
    n = len(buf)
    while offset < n:
        try:
            # need at least 1 + 16 bytes of header before we know payload length
            if offset + 1 > n:
                break
            (topic_len,) = _HDR_FIXED.unpack_from(buf, offset)
            needed = 1 + topic_len + _HDR_REST.size
            if offset + needed > n:
                break
            frame, new_offset = unpack_from(buf, offset)
            if new_offset > n:
                break
        except struct.error:
            break
        yield frame
        offset = new_offset
