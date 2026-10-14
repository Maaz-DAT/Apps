"""
common/messages.py — Frozen M-00 schemas (Maaz ownership).

Every message carries:
  - schema_version : int
  - t_ns           : int64 UTC nanoseconds
  - source_id      : str

Fields document units. WGS-84 lat/lon only at the boundary; internal
angles are radians, ranges/positions in metres.

REQ-M00-3: each message supports lossless to_bytes / from_bytes and
NumPy .npz round-trip.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) > 65535:
        raise ValueError("string too long for uint16 length prefix")
    return struct.pack("<H", len(b)) + b


def _unpack_str(buf: bytes, offset: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<H", buf, offset)
    offset += 2
    s = buf[offset : offset + n].decode("utf-8")
    return s, offset + n


def _pack_f64_list(vals: list[float]) -> bytes:
    n = len(vals)
    return struct.pack("<H", n) + struct.pack(f"<{n}d", *vals)


def _unpack_f64_list(buf: bytes, offset: int) -> tuple[list[float], int]:
    (n,) = struct.unpack_from("<H", buf, offset)
    offset += 2
    vals = list(struct.unpack_from(f"<{n}d", buf, offset))
    return vals, offset + n * 8


# ---------------------------------------------------------------------------
# PDW — Pulse Descriptor Word
# ---------------------------------------------------------------------------
# Fields (user M-00):
#   pulse width, ToA, phase per channel, freq, flag (valid / saturated / …)
# Extra common: schema_version, t_ns, source_id
# Amplitude kept because phase without amplitude is rarely useful in practice.
# ---------------------------------------------------------------------------

# flags bitfield
FLAG_VALID      = 1 << 0
FLAG_SATURATED  = 1 << 1
FLAG_CLIPPED    = 1 << 2
FLAG_SIDELOBE   = 1 << 3


@dataclass(slots=True)
class PDW:
    """Pulse Descriptor Word."""

    schema_version: int = SCHEMA_VERSION
    t_ns: int = 0                    # UTC ns (usually == toa_ns)
    source_id: str = ""

    toa_ns: int = 0                  # Time of Arrival, UTC ns
    frequency_hz: float = 0.0
    pulse_width_s: float = 0.0
    phase: list[float] = field(default_factory=list)   # rad, one per channel
    amplitude: list[float] = field(default_factory=list)  # linear, one per channel
    flags: int = FLAG_VALID          # bitfield: VALID | SATURATED | …

    def to_bytes(self) -> bytes:
        body = struct.pack(
            "<Iq",                      # schema_version, t_ns
            self.schema_version,
            self.t_ns,
        )
        body += _pack_str(self.source_id)
        body += struct.pack(
            "<qddI",                    # toa_ns, frequency_hz, pulse_width_s, flags
            self.toa_ns,
            self.frequency_hz,
            self.pulse_width_s,
            self.flags,
        )
        body += _pack_f64_list(self.phase)
        body += _pack_f64_list(self.amplitude)
        return body

    @classmethod
    def from_bytes(cls, data: bytes) -> "PDW":
        offset = 0
        schema_version, t_ns = struct.unpack_from("<Iq", data, offset)
        offset += 12
        source_id, offset = _unpack_str(data, offset)
        toa_ns, frequency_hz, pulse_width_s, flags = struct.unpack_from(
            "<qddI", data, offset
        )
        offset += 28
        phase, offset = _unpack_f64_list(data, offset)
        amplitude, offset = _unpack_f64_list(data, offset)
        return cls(
            schema_version=schema_version,
            t_ns=t_ns,
            source_id=source_id,
            toa_ns=toa_ns,
            frequency_hz=frequency_hz,
            pulse_width_s=pulse_width_s,
            phase=phase,
            amplitude=amplitude,
            flags=flags,
        )

    def to_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            schema_version=self.schema_version,
            t_ns=self.t_ns,
            source_id=np.array(self.source_id),
            toa_ns=self.toa_ns,
            frequency_hz=self.frequency_hz,
            pulse_width_s=self.pulse_width_s,
            phase=np.asarray(self.phase, dtype=np.float64),
            amplitude=np.asarray(self.amplitude, dtype=np.float64),
            flags=self.flags,
        )

    @classmethod
    def from_npz(cls, path: str) -> "PDW":
        z = np.load(path, allow_pickle=False)
        return cls(
            schema_version=int(z["schema_version"]),
            t_ns=int(z["t_ns"]),
            source_id=str(z["source_id"]),
            toa_ns=int(z["toa_ns"]),
            frequency_hz=float(z["frequency_hz"]),
            pulse_width_s=float(z["pulse_width_s"]),
            phase=z["phase"].tolist(),
            amplitude=z["amplitude"].tolist(),
            flags=int(z["flags"]),
        )


# ---------------------------------------------------------------------------
# IQDW — IQ Descriptor Word = full PDW + short IQ snippet
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IQDW:
    """IQ Descriptor Word = PDW + IQ snippet around the pulse."""

    schema_version: int = SCHEMA_VERSION
    t_ns: int = 0
    source_id: str = ""

    pdw: PDW = field(default_factory=PDW)
    iq_snippet: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.complex64)
    )  # shape (n_channels, n_samples), complex64
    sample_rate_hz: float = 0.0

    def to_bytes(self) -> bytes:
        pdw_bytes = self.pdw.to_bytes()
        n_ch, n_s = self.iq_snippet.shape
        header = struct.pack(
            "<Iq", self.schema_version, self.t_ns
        )
        header += _pack_str(self.source_id)
        header += struct.pack("<I", len(pdw_bytes))
        snip_hdr = struct.pack("<HId", n_ch, n_s, self.sample_rate_hz)
        raw = np.ascontiguousarray(self.iq_snippet, dtype=np.complex64).tobytes()
        return header + pdw_bytes + snip_hdr + raw

    @classmethod
    def from_bytes(cls, data: bytes) -> "IQDW":
        offset = 0
        schema_version, t_ns = struct.unpack_from("<Iq", data, offset)
        offset += 12
        source_id, offset = _unpack_str(data, offset)
        (pdw_len,) = struct.unpack_from("<I", data, offset)
        offset += 4
        pdw = PDW.from_bytes(data[offset : offset + pdw_len])
        offset += pdw_len
        n_ch, n_s, fs = struct.unpack_from("<HId", data, offset)
        offset += 14
        nbytes = n_ch * n_s * 8
        snippet = np.frombuffer(
            data[offset : offset + nbytes], dtype=np.complex64
        ).reshape(n_ch, n_s).copy()
        return cls(
            schema_version=schema_version,
            t_ns=t_ns,
            source_id=source_id,
            pdw=pdw,
            iq_snippet=snippet,
            sample_rate_hz=fs,
        )

    def to_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            schema_version=self.schema_version,
            t_ns=self.t_ns,
            source_id=np.array(self.source_id),
            pdw_bytes=np.frombuffer(self.pdw.to_bytes(), dtype=np.uint8),
            iq_snippet=self.iq_snippet,
            sample_rate_hz=self.sample_rate_hz,
        )

    @classmethod
    def from_npz(cls, path: str) -> "IQDW":
        z = np.load(path, allow_pickle=False)
        pdw = PDW.from_bytes(bytes(z["pdw_bytes"]))
        return cls(
            schema_version=int(z["schema_version"]),
            t_ns=int(z["t_ns"]),
            source_id=str(z["source_id"]),
            pdw=pdw,
            iq_snippet=z["iq_snippet"],
            sample_rate_hz=float(z["sample_rate_hz"]),
        )


# ---------------------------------------------------------------------------
# IQBlock — multi-channel complex IQ block
# ---------------------------------------------------------------------------
# Fields (user M-00):
#   N channels × M complex64 samples
#   samp rate, center freq, time stamp
#   channel to element map, array id
#   platform position, heading at time of block
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IQBlock:
    """Multi-channel complex IQ block with platform context."""

    schema_version: int = SCHEMA_VERSION
    t_ns: int = 0                    # UTC ns of first sample
    source_id: str = ""

    samples: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.complex64)
    )  # shape (n_channels, n_samples), complex64
    sample_rate_hz: float = 0.0
    centre_frequency_hz: float = 0.0
    channel_map: list[int] = field(default_factory=list)  # channel → element index
    array_id: str = "default"

    # platform state at t_ns
    platform_lat_deg: float = 0.0    # WGS-84
    platform_lon_deg: float = 0.0
    platform_alt_m: float = 0.0
    platform_heading_rad: float = 0.0

    @property
    def n_channels(self) -> int:
        return int(self.samples.shape[0]) if self.samples.ndim == 2 else 0

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[1]) if self.samples.ndim == 2 else 0

    def to_bytes(self) -> bytes:
        n_ch, n_s = self.samples.shape
        header = struct.pack("<Iq", self.schema_version, self.t_ns)
        header += _pack_str(self.source_id)
        header += struct.pack(
            "<dd",                      # sample_rate_hz, centre_frequency_hz
            self.sample_rate_hz,
            self.centre_frequency_hz,
        )
        header += _pack_str(self.array_id)
        # channel_map as int32 list
        header += struct.pack("<H", len(self.channel_map))
        header += struct.pack(f"<{len(self.channel_map)}i", *self.channel_map)
        header += struct.pack(
            "<dddd",                    # lat, lon, alt, heading
            self.platform_lat_deg,
            self.platform_lon_deg,
            self.platform_alt_m,
            self.platform_heading_rad,
        )
        header += struct.pack("<II", n_ch, n_s)
        raw = np.ascontiguousarray(self.samples, dtype=np.complex64).tobytes()
        return header + raw

    @classmethod
    def from_bytes(cls, data: bytes) -> "IQBlock":
        offset = 0
        schema_version, t_ns = struct.unpack_from("<Iq", data, offset)
        offset += 12
        source_id, offset = _unpack_str(data, offset)
        sample_rate_hz, centre_frequency_hz = struct.unpack_from("<dd", data, offset)
        offset += 16
        array_id, offset = _unpack_str(data, offset)
        (n_map,) = struct.unpack_from("<H", data, offset)
        offset += 2
        channel_map = list(struct.unpack_from(f"<{n_map}i", data, offset))
        offset += n_map * 4
        lat, lon, alt, heading = struct.unpack_from("<dddd", data, offset)
        offset += 32
        n_ch, n_s = struct.unpack_from("<II", data, offset)
        offset += 8
        nbytes = n_ch * n_s * 8
        samples = np.frombuffer(
            data[offset : offset + nbytes], dtype=np.complex64
        ).reshape(n_ch, n_s).copy()
        return cls(
            schema_version=schema_version,
            t_ns=t_ns,
            source_id=source_id,
            samples=samples,
            sample_rate_hz=sample_rate_hz,
            centre_frequency_hz=centre_frequency_hz,
            channel_map=channel_map,
            array_id=array_id,
            platform_lat_deg=lat,
            platform_lon_deg=lon,
            platform_alt_m=alt,
            platform_heading_rad=heading,
        )

    def to_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            schema_version=self.schema_version,
            t_ns=self.t_ns,
            source_id=np.array(self.source_id),
            samples=self.samples,
            sample_rate_hz=self.sample_rate_hz,
            centre_frequency_hz=self.centre_frequency_hz,
            channel_map=np.asarray(self.channel_map, dtype=np.int32),
            array_id=np.array(self.array_id),
            platform_lat_deg=self.platform_lat_deg,
            platform_lon_deg=self.platform_lon_deg,
            platform_alt_m=self.platform_alt_m,
            platform_heading_rad=self.platform_heading_rad,
        )

    @classmethod
    def from_npz(cls, path: str) -> "IQBlock":
        z = np.load(path, allow_pickle=False)
        return cls(
            schema_version=int(z["schema_version"]),
            t_ns=int(z["t_ns"]),
            source_id=str(z["source_id"]),
            samples=z["samples"],
            sample_rate_hz=float(z["sample_rate_hz"]),
            centre_frequency_hz=float(z["centre_frequency_hz"]),
            channel_map=z["channel_map"].tolist(),
            array_id=str(z["array_id"]),
            platform_lat_deg=float(z["platform_lat_deg"]),
            platform_lon_deg=float(z["platform_lon_deg"]),
            platform_alt_m=float(z["platform_alt_m"]),
            platform_heading_rad=float(z["platform_heading_rad"]),
        )


# ---------------------------------------------------------------------------
# Convenience registry (topic name → class)
# ---------------------------------------------------------------------------

MESSAGE_TYPES = {
    "pdw": PDW,
    "iqdw": IQDW,
    "iqblock": IQBlock,
}
