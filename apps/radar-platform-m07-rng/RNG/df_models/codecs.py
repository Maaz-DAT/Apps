"""
A-02 — Raw IQ, PDW and IQ-DW codecs.

REQ-A02-1: Raw IQ in a simple SigMF-like layout (binary + JSON metadata)
REQ-A02-2: PDW and IQ-DW as fixed-length binary records; encode→decode is bit-exact
REQ-A02-4: Truncated / malformed records are counted and skipped, never crash
"""

from __future__ import annotations
import json
import struct
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# Message dataclasses (frozen schemas from A-00)
# ---------------------------------------------------------------------------

@dataclass
class IQBlock:
    """Raw multi-channel IQ block."""
    samples: np.ndarray          # shape (n_channels, n_samples), complex64
    sample_rate: float
    centre_frequency: float
    timestamp_ns: int            # UTC nanoseconds
    channel_map: list[int]       # channel index → element index
    array_id: str = "default"
    meta: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        return self.samples.shape[1]


@dataclass
class PDW:
    """Pulse Descriptor Word."""
    toa_ns: int                  # UTC nanoseconds
    frequency_hz: float
    pulse_width_s: float
    amplitude: list[float]       # per-channel amplitude
    phase: list[float] | None = None   # per-channel phase (rad), optional
    flags: int = 0
    aoa_rad: float | None = None
    snr_db: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class IQDW:
    """IQ Descriptor Word = PDW + short IQ snippet."""
    pdw: PDW
    iq_snippet: np.ndarray       # shape (n_channels, n_snippet), complex64
    sample_rate: float


@dataclass
class Bearing:
    """Output of a DF algorithm."""
    azimuth_rad: float
    uncertainty_rad: float       # 1σ
    confidence: float
    frequency_hz: float
    snr_db: float
    method: str                  # "CIDF" | "SODA" | "SBI" | "SDF"
    timestamp_ns: int
    array_id: str = "default"
    array_lat: float | None = None
    array_lon: float | None = None
    platform_heading_rad: float | None = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Binary layout for PDW (fixed-length, little-endian)
# ---------------------------------------------------------------------------
# toa_ns          : uint64
# frequency_hz    : float64
# pulse_width_s   : float64
# n_channels      : uint16
# amplitude[0..7] : 8 × float32   (padded to 8 channels max)
# phase[0..7]     : 8 × float32   (NaN if not present)
# flags           : uint32
# aoa_rad         : float32       (NaN if not present)
# snr_db          : float32       (NaN if not present)
# reserved        : 4 bytes
# Total values packed = 4 + 8 + 8 + 3 = 23
# Byte size = 8+8+8+2 + 32 + 32 + 4 + 4 + 4 + 4 = 106 → pad with 4x → 108

PDW_STRUCT = struct.Struct("<QddH8f8fIff4x")
PDW_SIZE = PDW_STRUCT.size          # 108


def encode_pdw(pdw: PDW) -> bytes:
    n_ch = len(pdw.amplitude)
    if n_ch > 8:
        raise ValueError("PDW codec supports max 8 channels")
    amp = list(pdw.amplitude) + [0.0] * (8 - n_ch)
    if pdw.phase is not None:
        ph = list(pdw.phase) + [float("nan")] * (8 - len(pdw.phase))
    else:
        ph = [float("nan")] * 8
    aoa = pdw.aoa_rad if pdw.aoa_rad is not None else float("nan")
    snr = pdw.snr_db if pdw.snr_db is not None else float("nan")
    return PDW_STRUCT.pack(
        pdw.toa_ns,
        pdw.frequency_hz,
        pdw.pulse_width_s,
        n_ch,
        *amp,
        *ph,
        pdw.flags,
        aoa,
        snr,
    )


def decode_pdw(data: bytes) -> PDW | None:
    """Return PDW or None if the record is truncated/malformed."""
    if len(data) < PDW_SIZE:
        return None
    try:
        fields = PDW_STRUCT.unpack(data[:PDW_SIZE])
    except struct.error:
        return None
    toa, freq, pw, n_ch = fields[0], fields[1], fields[2], fields[3]
    amp = list(fields[4:12][:n_ch])
    ph_raw = list(fields[12:20][:n_ch])
    phase = None if all(np.isnan(p) for p in ph_raw) else ph_raw
    flags = fields[20]
    aoa = fields[21]
    snr = fields[22]
    return PDW(
        toa_ns=toa,
        frequency_hz=freq,
        pulse_width_s=pw,
        amplitude=amp,
        phase=phase,
        flags=flags,
        aoa_rad=None if np.isnan(aoa) else aoa,
        snr_db=None if np.isnan(snr) else snr,
    )


def write_pdw_file(path: str | Path, pdws: list[PDW]) -> None:
    path = Path(path)
    with path.open("wb") as f:
        for p in pdws:
            f.write(encode_pdw(p))


def read_pdw_file(path: str | Path) -> tuple[list[PDW], int]:
    """
    Read a PDW file.
    Returns (list of valid PDWs, number of skipped/malformed records).
    """
    path = Path(path)
    data = path.read_bytes()
    pdws = []
    skipped = 0
    offset = 0
    while offset + PDW_SIZE <= len(data):
        rec = decode_pdw(data[offset:offset + PDW_SIZE])
        if rec is None:
            skipped += 1
        else:
            pdws.append(rec)
        offset += PDW_SIZE
    # leftover bytes count as one truncated record
    if offset < len(data):
        skipped += 1
    return pdws, skipped


# ---------------------------------------------------------------------------
# IQ-DW = PDW header + IQ snippet
# ---------------------------------------------------------------------------

def encode_iqdw(iqdw: IQDW) -> bytes:
    header = encode_pdw(iqdw.pdw)
    # snippet header: n_channels (uint16), n_samples (uint32), sample_rate (float64)
    n_ch, n_s = iqdw.iq_snippet.shape
    snip_hdr = struct.pack("<HId", n_ch, n_s, iqdw.sample_rate)
    # raw complex64 samples, channel-major
    raw = iqdw.iq_snippet.astype(np.complex64).tobytes()
    return header + snip_hdr + raw


def decode_iqdw(data: bytes) -> IQDW | None:
    if len(data) < PDW_SIZE + 14:
        return None
    pdw = decode_pdw(data[:PDW_SIZE])
    if pdw is None:
        return None
    try:
        n_ch, n_s, fs = struct.unpack_from("<HId", data, PDW_SIZE)
    except struct.error:
        return None
    expected = PDW_SIZE + 14 + n_ch * n_s * 8
    if len(data) < expected:
        return None
    raw = data[PDW_SIZE + 14:expected]
    snippet = np.frombuffer(raw, dtype=np.complex64).reshape(n_ch, n_s)
    return IQDW(pdw=pdw, iq_snippet=snippet.copy(), sample_rate=fs)


# ---------------------------------------------------------------------------
# Simple SigMF-like IQ storage (REQ-A02-1)
# ---------------------------------------------------------------------------

def write_iq_sigmf(path_stem: str | Path, iqblock: IQBlock) -> None:
    """
    Write a two-file SigMF-like recording:
      <stem>.sigmf-data  – interleaved complex64 (channel-major)
      <stem>.sigmf-meta  – JSON metadata
    """
    path_stem = Path(path_stem)
    data_path = path_stem.with_suffix(".sigmf-data")
    meta_path = path_stem.with_suffix(".sigmf-meta")

    # channel-major interleaved
    data_path.write_bytes(iqblock.samples.astype(np.complex64).tobytes())

    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": iqblock.sample_rate,
            "core:version": "1.0.0",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": iqblock.centre_frequency,
                "core:datetime": iqblock.timestamp_ns,
            }
        ],
        "annotations": [],
        "df_extensions": {
            "channel_map": iqblock.channel_map,
            "array_id": iqblock.array_id,
            "n_channels": iqblock.n_channels,
            "n_samples": iqblock.n_samples,
            "meta": iqblock.meta,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def read_iq_sigmf(path_stem: str | Path) -> IQBlock:
    path_stem = Path(path_stem)
    data_path = path_stem.with_suffix(".sigmf-data")
    meta_path = path_stem.with_suffix(".sigmf-meta")

    meta = json.loads(meta_path.read_text())
    ext = meta["df_extensions"]
    n_ch = ext["n_channels"]
    n_s = ext["n_samples"]
    raw = data_path.read_bytes()
    samples = np.frombuffer(raw, dtype=np.complex64).reshape(n_ch, n_s)

    return IQBlock(
        samples=samples.copy(),
        sample_rate=meta["global"]["core:sample_rate"],
        centre_frequency=meta["captures"][0]["core:frequency"],
        timestamp_ns=meta["captures"][0]["core:datetime"],
        channel_map=ext["channel_map"],
        array_id=ext["array_id"],
        meta=ext.get("meta", {}),
    )
