"""
ASTERIX Category 240 — Radar Video Transmission (Ed. 1.3)
Bit-accurate encoder / decoder / generator.

Scale factors from EUROCONTROL-SPEC-0149-240:
  START_AZ / END_AZ : LSB = 360° / 2^16
  CELL_DUR (nano)   : nanoseconds
  START_RG          : cell index (integer)
  TOD               : LSB = 1/128 s
  RES               : 1,2,4,8,16 bits per cell (I240/048)
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

CAT = 240
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
AZ_LSB_DEG = 360.0 / 65536.0
TOD_LSB_S = 1.0 / 128.0
C_LIGHT = 299_792_458.0  # m/s

# RES coding in I240/048 (bits 5-1 of second octet in common implementations)
# Spec: bit resolution used for amplitude coding
RES_BITS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}  # code → bits
BITS_TO_RES = {v: k for k, v in RES_BITS.items()}


@dataclass(slots=True)
class Cat240Sweep:
    """One decoded video radial / group of cells (SI units at boundary)."""

    sac: int = 0
    sic: int = 0
    msg_type: int = 2                    # 1=summary, 2=video
    msg_index: int = 0                   # I240/020 sequence
    start_az_rad: float = 0.0
    end_az_rad: float = 0.0
    start_range_m: float = 0.0           # range of first cell
    cell_size_m: float = 0.0             # range extent of one cell
    resolution_bits: int = 8
    compressed: bool = False
    amplitudes: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.uint16)
    )  # per cell, 0..(2^bits-1)
    time_of_day_ns: int = 0
    unknown_skipped: int = 0


# ---------------------------------------------------------------------------
# FSPEC helpers (same pattern as CAT 048)
# ---------------------------------------------------------------------------

def _read_fspec(data: bytes, offset: int) -> tuple[list[int], int]:
    frns: list[int] = []
    base = 1
    while offset < len(data):
        octet = data[offset]
        offset += 1
        for bit in range(7, 0, -1):
            if octet & (1 << bit):
                frns.append(base + (7 - bit))
        if not (octet & 1):
            break
        base += 7
    return frns, offset


def _write_fspec(frns: set[int]) -> bytes:
    if not frns:
        return b"\x00"
    max_frn = max(frns)
    n = (max_frn + 6) // 7
    out = bytearray(n)
    for frn in frns:
        oi = (frn - 1) // 7
        bi = 7 - ((frn - 1) % 7)
        out[oi] |= 1 << bi
    for i in range(n - 1):
        out[i] |= 1
    return bytes(out)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode_record(data: bytes, offset: int = 0) -> tuple[Cat240Sweep | None, int]:
    """Decode one CAT 240 record. Never raises (REQ style)."""
    start = offset
    try:
        if offset + 3 > len(data) or data[offset] != CAT:
            return None, offset
        length = int.from_bytes(data[offset + 1: offset + 3], "big")
        if length < 3 or offset + length > len(data):
            return None, offset
        end = offset + length
        offset += 3

        frns, offset = _read_fspec(data, offset)
        present = set(frns)
        sw = Cat240Sweep()
        skipped = 0
        n_cells = 0
        n_octets = 0
        cell_dur_ns = 0
        start_rg = 0
        res_bits = 8
        video_raw = b""

        # FRN map for CAT 240 UAP (Ed 1.3)
        # 1:010  2:000  3:020  4:030  5:040  6:041  7:048  FX
        # 8:049  9:050  10:051  11:052  12:140

        for frn in sorted(present):
            if offset >= end:
                break

            if frn == 1:  # I240/010
                if offset + 2 > end:
                    break
                sw.sac, sw.sic = data[offset], data[offset + 1]
                offset += 2

            elif frn == 2:  # I240/000 Message Type
                if offset + 1 > end:
                    break
                sw.msg_type = data[offset]
                offset += 1

            elif frn == 3:  # I240/020 Video Record Header (4 octets)
                if offset + 4 > end:
                    break
                sw.msg_index = int.from_bytes(data[offset: offset + 4], "big")
                offset += 4

            elif frn == 4:  # I240/030 Video Summary (variable ASCII)
                if offset >= end:
                    break
                # first byte = length of summary string in some impls;
                # spec: 1+n characters. Consume until we hit a safe bound.
                # Practical: treat as FX-style or length-prefixed.
                # Ed 1.3: variable length, characters. Skip by reading remaining
                # until next known fixed item is risky — use LEN residual later.
                # Simple: one length octet then n chars (common practice).
                n = data[offset]
                offset += 1
                offset = min(offset + n, end)

            elif frn == 5:  # I240/040 Video Header Nano (12 octets)
                if offset + 12 > end:
                    break
                start_az = int.from_bytes(data[offset: offset + 2], "big")
                end_az = int.from_bytes(data[offset + 2: offset + 4], "big")
                start_rg = int.from_bytes(data[offset + 4: offset + 8], "big")
                cell_dur_ns = int.from_bytes(data[offset + 8: offset + 12], "big")
                sw.start_az_rad = start_az * AZ_LSB_DEG * DEG_TO_RAD
                sw.end_az_rad = end_az * AZ_LSB_DEG * DEG_TO_RAD
                offset += 12

            elif frn == 6:  # I240/041 Video Header Femto (12 octets) — same layout, fs
                if offset + 12 > end:
                    break
                start_az = int.from_bytes(data[offset: offset + 2], "big")
                end_az = int.from_bytes(data[offset + 2: offset + 4], "big")
                start_rg = int.from_bytes(data[offset + 4: offset + 8], "big")
                # femto-seconds
                cell_dur_fs = int.from_bytes(data[offset + 8: offset + 12], "big")
                cell_dur_ns = cell_dur_fs // 1_000_000
                sw.start_az_rad = start_az * AZ_LSB_DEG * DEG_TO_RAD
                sw.end_az_rad = end_az * AZ_LSB_DEG * DEG_TO_RAD
                offset += 12

            elif frn == 7:  # I240/048 Resolution & Compression (2 octets)
                if offset + 2 > end:
                    break
                # octet1: spare + compression indicator (bit)

                # Common layout (Ed 1.3 corrected): 
                # byte0: CI (bit 8) + spare
                # byte1: RES code
                ci = data[offset]
                res_code = data[offset + 1] & 0x1F
                sw.compressed = bool(ci & 0x80)
                res_bits = RES_BITS.get(res_code, 8)
                sw.resolution_bits = res_bits
                offset += 2
                if sw.compressed:
                    # reject compressed for now (REQ-M04-2 style)
                    return None, end

            elif frn == 8:  # I240/049 Counters (5 octets): NB_OCTETS u16 + NB_CELLS u24
                if offset + 5 > end:
                    break
                n_octets = int.from_bytes(data[offset: offset + 2], "big")
                n_cells = int.from_bytes(data[offset + 2: offset + 5], "big")
                offset += 5

            elif frn in (9, 10, 11):  # Video blocks 050/051/052
                if offset >= end:
                    break
                # REP (1 octet) then REP * block_size
                block_size = {9: 4, 10: 64, 11: 256}[frn]
                rep = data[offset]
                offset += 1
                nbytes = rep * block_size
                if offset + nbytes > end:
                    break
                video_raw += data[offset: offset + nbytes]
                offset += nbytes

            elif frn == 12:  # I240/140 Time of Day
                if offset + 3 > end:
                    break
                raw = int.from_bytes(data[offset: offset + 3], "big")
                sw.time_of_day_ns = int(raw * TOD_LSB_S * 1e9)
                offset += 3

            else:
                skipped += 1
                # unknown fixed-ish: skip 1 byte minimum to progress
                offset += 1

        # Unpack amplitudes
        if n_cells == 0 and video_raw:
            # infer from raw length and resolution
            bits = res_bits
            n_cells = (len(video_raw) * 8) // bits

        if n_cells > 0 and video_raw:
            amps = _unpack_cells(video_raw, n_cells, res_bits)
            sw.amplitudes = amps
        else:
            sw.amplitudes = np.zeros(0, dtype=np.uint16)

        # Range geometry: cell duration → metres (two-way)
        # D = CELL_DUR * c / 2
        if cell_dur_ns > 0:
            sw.cell_size_m = (cell_dur_ns * 1e-9) * C_LIGHT / 2.0
        sw.start_range_m = start_rg * sw.cell_size_m
        sw.unknown_skipped = skipped
        return sw, end
    except Exception:
        return None, start + 1


def _unpack_cells(raw: bytes, n_cells: int, bits: int) -> np.ndarray:
    """Unpack tightly packed cell amplitudes."""
    if bits == 8:
        arr = np.frombuffer(raw, dtype=np.uint8)
        return arr[:n_cells].astype(np.uint16)
    if bits == 16:
        # big-endian per ASTERIX convention
        n = min(n_cells, len(raw) // 2)
        return np.frombuffer(raw[: n * 2], dtype=">u2").astype(np.uint16)
    if bits == 4:
        out = np.zeros(n_cells, dtype=np.uint16)
        for i in range(n_cells):
            byte = raw[i // 2]
            out[i] = (byte >> 4) if (i % 2 == 0) else (byte & 0x0F)
        return out
    if bits == 2:
        out = np.zeros(n_cells, dtype=np.uint16)
        for i in range(n_cells):
            byte = raw[i // 4]
            shift = 6 - 2 * (i % 4)
            out[i] = (byte >> shift) & 0x03
        return out
    if bits == 1:
        out = np.zeros(n_cells, dtype=np.uint16)
        for i in range(n_cells):
            byte = raw[i // 8]
            out[i] = (byte >> (7 - (i % 8))) & 1
        return out
    # fallback 8-bit
    return np.frombuffer(raw[:n_cells], dtype=np.uint8).astype(np.uint16)


def decode_buffer(data: bytes) -> tuple[list[Cat240Sweep], int]:
    sweeps: list[Cat240Sweep] = []
    errors = 0
    offset = 0
    while offset < len(data):
        if data[offset] != CAT:
            offset += 1
            errors += 1
            continue
        sw, new = decode_record(data, offset)
        if sw is None or new <= offset:
            offset += 1
            errors += 1
            continue
        sweeps.append(sw)
        offset = new
    return sweeps, errors


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def _pack_cells(amps: np.ndarray, bits: int) -> bytes:
    amps = np.asarray(amps)
    if bits == 8:
        return amps.astype(np.uint8).tobytes()
    if bits == 16:
        return amps.astype(">u2").tobytes()
    if bits == 4:
        n = len(amps)
        out = bytearray((n + 1) // 2)
        for i, v in enumerate(amps):
            if i % 2 == 0:
                out[i // 2] = (int(v) & 0x0F) << 4
            else:
                out[i // 2] |= int(v) & 0x0F
        return bytes(out)
    # default 8
    return amps.astype(np.uint8).tobytes()


def encode_record(sw: Cat240Sweep) -> bytes:
    """Encode one Cat240Sweep as a Video Message (type 2)."""
    items: list[tuple[int, bytes]] = []

    # FRN1 I240/010
    items.append((1, bytes([sw.sac & 0xFF, sw.sic & 0xFF])))
    # FRN2 I240/000
    items.append((2, bytes([2])))  # video message
    # FRN3 I240/020
    items.append((3, (sw.msg_index & 0xFFFFFFFF).to_bytes(4, "big")))

    # FRN5 I240/040 Video Header Nano
    start_az = int(round(sw.start_az_rad * RAD_TO_DEG / AZ_LSB_DEG)) & 0xFFFF
    end_az = int(round(sw.end_az_rad * RAD_TO_DEG / AZ_LSB_DEG)) & 0xFFFF
    cell_dur_ns = 0
    start_rg = 0
    if sw.cell_size_m > 0:
        # two-way: cell_dur = 2 * cell_size / c
        cell_dur_ns = int(round(sw.cell_size_m * 2.0 / C_LIGHT * 1e9))
        start_rg = int(round(sw.start_range_m / sw.cell_size_m))
    hdr = (
        start_az.to_bytes(2, "big")
        + end_az.to_bytes(2, "big")
        + start_rg.to_bytes(4, "big")
        + cell_dur_ns.to_bytes(4, "big")
    )
    items.append((5, hdr))

    # FRN7 I240/048
    res_code = BITS_TO_RES.get(sw.resolution_bits, 4)  # default 8-bit → code 4
    items.append((7, bytes([0x00, res_code & 0x1F])))  # no compression

    # Pack video
    bits = sw.resolution_bits
    raw = _pack_cells(sw.amplitudes, bits)
    n_cells = len(sw.amplitudes)
    n_octets = len(raw)

    # FRN8 I240/049
    items.append((8, n_octets.to_bytes(2, "big") + n_cells.to_bytes(3, "big")))

    # Choose block type by size
    if n_octets <= 1020:
        block_size = 4
        frn_vid = 9
    elif n_octets <= 16320:
        block_size = 64
        frn_vid = 10
    else:
        block_size = 256
        frn_vid = 11

    # Pad to multiple of block_size
    pad = (-len(raw)) % block_size
    if pad:
        raw = raw + b"\x00" * pad
    rep = len(raw) // block_size
    items.append((frn_vid, bytes([rep & 0xFF]) + raw))

    # FRN12 I240/140
    tod_raw = int(round(sw.time_of_day_ns / 1e9 / TOD_LSB_S)) & 0xFFFFFF
    items.append((12, tod_raw.to_bytes(3, "big")))

    frns = {f for f, _ in items}
    fspec = _write_fspec(frns)
    body = b"".join(p for _, p in sorted(items, key=lambda x: x[0]))
    total = 3 + len(fspec) + len(body)
    return bytes([CAT]) + total.to_bytes(2, "big") + fspec + body


def encode_block(sweeps: list[Cat240Sweep]) -> bytes:
    return b"".join(encode_record(s) for s in sweeps)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@dataclass
class TruthVideoTarget:
    range_m: float
    az_rad: float
    amplitude: int = 200          # 0-255 for 8-bit
    width_m: float = 50.0         # radial extent


def generate_sweep(
    *,
    az_rad: float,
    az_width_rad: float = 0.01,
    n_cells: int = 1024,
    cell_size_m: float = 15.0,
    start_range_m: float = 0.0,
    targets: list[TruthVideoTarget] | None = None,
    noise_sigma: float = 8.0,
    clutter_range_m: float = 3000.0,
    clutter_level: float = 40.0,
    resolution_bits: int = 8,
    sac: int = 0x50,
    sic: int = 0x01,
    msg_index: int = 0,
    time_of_day_ns: int = 0,
    seed: int | None = None,
) -> Cat240Sweep:
    """Synthetic video radial with Rayleigh-ish noise, near clutter, targets."""
    rng = np.random.default_rng(seed)
    max_val = (1 << resolution_bits) - 1

    # noise (approx Rayleigh → positive)
    noise = rng.rayleigh(noise_sigma, size=n_cells)
    amps = noise.copy()

    # ground clutter near radar
    for i in range(n_cells):
        r = start_range_m + i * cell_size_m
        if r < clutter_range_m:
            amps[i] += clutter_level * (1.0 - r / clutter_range_m)

    # targets
    if targets:
        for t in targets:
            # only paint if target azimuth falls in this radial
            daz = (t.az_rad - az_rad + math.pi) % (2 * math.pi) - math.pi
            if abs(daz) > az_width_rad / 2:
                continue
            centre = int(round((t.range_m - start_range_m) / cell_size_m))
            half = max(1, int(round(t.width_m / cell_size_m / 2)))
            for i in range(max(0, centre - half), min(n_cells, centre + half + 1)):
                amps[i] += t.amplitude

    amps = np.clip(amps, 0, max_val).astype(np.uint16)

    return Cat240Sweep(
        sac=sac,
        sic=sic,
        msg_type=2,
        msg_index=msg_index,
        start_az_rad=az_rad,
        end_az_rad=az_rad + az_width_rad,
        start_range_m=start_range_m,
        cell_size_m=cell_size_m,
        resolution_bits=resolution_bits,
        amplitudes=amps,
        time_of_day_ns=time_of_day_ns,
    )
