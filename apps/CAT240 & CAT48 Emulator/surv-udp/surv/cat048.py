"""
ASTERIX Category 048 — Monoradar Target Reports
Encoder, decoder and synthetic generator (M-03).

Scale factors verified against EUROCONTROL SPEC-0149-4 (CAT 048):
  I048/040 RHO   : LSB = 1/256 NM
  I048/040 THETA : LSB = 360° / 2^16
  I048/140 TOD   : LSB = 1/128 s
  I048/090 FL    : LSB = 1/4 FL
  I048/200 SPEED : LSB = 2^-14 NM/s
  I048/200 HEAD  : LSB = 360° / 2^16

SI conversion is performed at the decoder boundary (range → metres,
angles → radians, time → seconds / ns).
"""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass, field
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants (EUROCONTROL)
# ---------------------------------------------------------------------------

NM_TO_M = 1852.0
RHO_LSB_NM = 1.0 / 256.0
THETA_LSB_DEG = 360.0 / 65536.0
TOD_LSB_S = 1.0 / 128.0
FL_LSB = 0.25                     # 1/4 Flight Level
SPEED_LSB_NM_S = 2.0 ** -14
HEAD_LSB_DEG = 360.0 / 65536.0

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

CAT = 48

# Standard UAP FRN → (item id, fixed length or None if variable/FX)
# Lengths used for skipping unknown items.
# Variable items (020, 170, 130, …) use FX bit chaining; we parse 020 and 170
# fully and skip others by reading FX chain or fixed size.
UAP = [
    # FRN 1-7
    (1,  "010", 2),      # Data Source Identifier
    (2,  "140", 3),      # Time of Day
    (3,  "020", None),   # Target Report Descriptor (1+ FX)
    (4,  "040", 4),      # Measured Position Polar
    (5,  "070", 2),      # Mode-3/A
    (6,  "090", 2),      # Flight Level
    (7,  "130", None),   # Radar Plot Characteristics (compound) — skip
    # FRN 8-14
    (8,  "220", 3),      # Aircraft Address
    (9,  "240", 6),      # Aircraft Identification — skip
    (10, "250", None),   # Mode S MB Data — skip
    (11, "161", 2),      # Track Number
    (12, "042", 4),      # Calculated Position Cartesian — skip
    (13, "200", 4),      # Track Velocity Polar
    (14, "170", None),   # Track Status (1+ FX)
]


# ---------------------------------------------------------------------------
# Decoded target report (SI units at boundary)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Cat048Target:
    """One decoded / encoded CAT 048 target report in SI units."""

    # I048/010
    sac: int = 0
    sic: int = 0

    # I048/140 — time of day as UTC nanoseconds of day (0 … 86400e9)
    time_of_day_ns: int = 0

    # I048/020 (simplified — key flags only)
    typ: int = 0          # 0=no detect, 1=single PSR, 2=single SSR, 3=combined, …
    simulated: bool = False
    test: bool = False

    # I048/040 — polar position
    range_m: float = 0.0
    az_rad: float = 0.0

    # I048/070 Mode-3/A (0–7777 octal as integer 0–4095)
    mode3a: int | None = None
    mode3a_v: bool = True   # validated
    mode3a_g: bool = False  # garbled
    mode3a_l: bool = False  # smoothed / from track

    # I048/090 Flight Level → altitude in feet (FL * 100)
    flight_level_ft: float | None = None

    # I048/161
    track_number: int | None = None

    # I048/200 polar velocity
    ground_speed_m_s: float | None = None
    heading_rad: float | None = None

    # I048/220 24-bit ICAO address
    aircraft_address: int | None = None

    # I048/170 simplified
    track_confirmed: bool = True
    multipath: bool = False

    # Decoder bookkeeping
    unknown_items_skipped: int = 0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _read_fspec(data: bytes, offset: int) -> tuple[list[int], int]:
    """Read FSPEC octets (FX chaining). Returns (list of present FRNs, new offset)."""
    frns: list[int] = []
    frn_base = 1
    while True:
        if offset >= len(data):
            break
        octet = data[offset]
        offset += 1
        for bit in range(7, 0, -1):  # bits 8..2
            if octet & (1 << bit):
                frns.append(frn_base + (7 - bit))
        if not (octet & 0x01):  # FX = 0 → end
            break
        frn_base += 7
    return frns, offset


def _write_fspec(frns: set[int]) -> bytes:
    """Build minimal FSPEC covering the given FRNs."""
    if not frns:
        return b"\x00"
    max_frn = max(frns)
    n_octets = (max_frn + 6) // 7
    out = bytearray(n_octets)
    for frn in frns:
        octet_idx = (frn - 1) // 7
        bit_in_octet = 7 - ((frn - 1) % 7)  # bit 8 = FRN base, … bit 2
        out[octet_idx] |= 1 << bit_in_octet
    # set FX bits on all but last
    for i in range(n_octets - 1):
        out[i] |= 0x01
    return bytes(out)


def _skip_fx_item(data: bytes, offset: int) -> int:
    """Skip a variable-length FX-chained item (1+ octets)."""
    while offset < len(data):
        octet = data[offset]
        offset += 1
        if not (octet & 0x01):
            break
    return offset


def _skip_compound_item(data: bytes, offset: int) -> int:
    """
    Skip I048/130-style compound: primary subfield indicates which subfields
    follow.  We only need to advance the cursor safely.
    Primary is 1 octet; bit 1 = FX for more primary octets (rare).
    Then fixed subfields according to bits — for safety we treat unknown
    compound items by reading primary and skipping a conservative amount.
    For M-03 we simply consume the primary FX chain and stop; residual
    bytes are absorbed by LEN checks at record level.
    """
    return _skip_fx_item(data, offset)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode_record(data: bytes, offset: int = 0) -> tuple[Cat048Target | None, int]:
    """
    Decode one CAT 048 record starting at offset.
    Returns (target or None on fatal error, new_offset).
    Never raises on malformed data (REQ-M03-3).
    """
    start = offset
    try:
        if offset + 3 > len(data):
            return None, offset
        cat = data[offset]
        if cat != CAT:
            return None, offset
        length = int.from_bytes(data[offset + 1 : offset + 3], "big")
        if length < 3 or offset + length > len(data):
            return None, offset
        end = offset + length
        offset += 3

        frns, offset = _read_fspec(data, offset)
        tgt = Cat048Target()
        skipped = 0

        # Map FRN → presence for ordered parse matching UAP
        present = set(frns)

        for frn, item_id, fixed_len in UAP:
            if frn not in present:
                continue
            if offset >= end:
                break

            if item_id == "010":  # SAC, SIC
                if offset + 2 > end:
                    break
                tgt.sac = data[offset]
                tgt.sic = data[offset + 1]
                offset += 2

            elif item_id == "140":  # Time of Day
                if offset + 3 > end:
                    break
                raw = int.from_bytes(data[offset : offset + 3], "big")
                tgt.time_of_day_ns = int(raw * TOD_LSB_S * 1e9)
                offset += 3

            elif item_id == "020":  # Target Report Descriptor
                if offset >= end:
                    break
                first = data[offset]
                offset += 1
                # bits 8-6 TYP (simplified mapping of common encoding)
                typ_bits = (first >> 5) & 0x07
                tgt.typ = typ_bits
                tgt.simulated = bool(first & 0x10)
                # FX chain
                while first & 0x01 and offset < end:
                    first = data[offset]
                    offset += 1
                    if first & 0x80:
                        tgt.test = True

            elif item_id == "040":  # RHO / THETA
                if offset + 4 > end:
                    break
                rho_raw = int.from_bytes(data[offset : offset + 2], "big")
                theta_raw = int.from_bytes(data[offset + 2 : offset + 4], "big")
                tgt.range_m = rho_raw * RHO_LSB_NM * NM_TO_M
                tgt.az_rad = (theta_raw * THETA_LSB_DEG) * DEG_TO_RAD
                offset += 4

            elif item_id == "070":  # Mode-3/A
                if offset + 2 > end:
                    break
                w = int.from_bytes(data[offset : offset + 2], "big")
                tgt.mode3a_v = not bool(w & 0x8000)
                tgt.mode3a_g = bool(w & 0x4000)
                tgt.mode3a_l = bool(w & 0x2000)
                tgt.mode3a = w & 0x0FFF
                offset += 2

            elif item_id == "090":  # Flight Level
                if offset + 2 > end:
                    break
                w = int.from_bytes(data[offset : offset + 2], "big")
                # bit 16 = V, bit 15 = G; value in bits 14-1 as 1/4 FL
                fl_raw = w & 0x3FFF
                # two's complement for negative (rare)
                if fl_raw & 0x2000:
                    fl_raw -= 0x4000
                tgt.flight_level_ft = fl_raw * FL_LSB * 100.0
                offset += 2

            elif item_id == "220":  # Aircraft Address
                if offset + 3 > end:
                    break
                tgt.aircraft_address = int.from_bytes(data[offset : offset + 3], "big")
                offset += 3

            elif item_id == "161":  # Track Number
                if offset + 2 > end:
                    break
                tgt.track_number = int.from_bytes(data[offset : offset + 2], "big") & 0x0FFF
                offset += 2

            elif item_id == "200":  # Track Velocity Polar
                if offset + 4 > end:
                    break
                spd_raw = int.from_bytes(data[offset : offset + 2], "big")
                hdg_raw = int.from_bytes(data[offset + 2 : offset + 4], "big")
                tgt.ground_speed_m_s = spd_raw * SPEED_LSB_NM_S * NM_TO_M
                tgt.heading_rad = (hdg_raw * HEAD_LSB_DEG) * DEG_TO_RAD
                offset += 4

            elif item_id == "170":  # Track Status
                if offset >= end:
                    break
                first = data[offset]
                offset += 1
                tgt.track_confirmed = not bool(first & 0x80)  # CNF bit inverted convention varies
                tgt.multipath = bool(first & 0x08)
                while first & 0x01 and offset < end:
                    first = data[offset]
                    offset += 1

            else:
                # Unknown / intentionally skipped item
                skipped += 1
                if fixed_len is not None:
                    offset += fixed_len
                else:
                    # variable / compound — best-effort skip
                    if item_id in ("130", "250"):
                        offset = _skip_compound_item(data, offset)
                    else:
                        offset = _skip_fx_item(data, offset)

        tgt.unknown_items_skipped = skipped
        return tgt, end
    except Exception:
        return None, start + 1  # advance one byte so fuzzing progresses


def decode_buffer(data: bytes) -> tuple[list[Cat048Target], int]:
    """
    Decode every record in a byte buffer.
    Returns (list of targets, number of bytes/records that caused a skip).
    """
    targets: list[Cat048Target] = []
    errors = 0
    offset = 0
    while offset < len(data):
        # look for CAT=48
        if data[offset] != CAT:
            offset += 1
            errors += 1
            continue
        tgt, new_off = decode_record(data, offset)
        if tgt is None or new_off <= offset:
            offset += 1
            errors += 1
            continue
        targets.append(tgt)
        offset = new_off
    return targets, errors


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_record(tgt: Cat048Target) -> bytes:
    """Encode one Cat048Target into a complete CAT 048 record (CAT+LEN+FSPEC+items)."""
    items: list[tuple[int, bytes]] = []  # (FRN, payload)

    # 010
    items.append((1, bytes([tgt.sac & 0xFF, tgt.sic & 0xFF])))

    # 140
    tod_raw = int(round(tgt.time_of_day_ns / 1e9 / TOD_LSB_S)) & 0xFFFFFF
    items.append((2, tod_raw.to_bytes(3, "big")))

    # 020 — minimal one-octet descriptor
    typ = tgt.typ & 0x07
    sim = 0x10 if tgt.simulated else 0
    octet = (typ << 5) | sim
    if tgt.test:
        # put TST in first extension
        items.append((3, bytes([octet | 0x01, 0x80])))  # FX + TST
    else:
        items.append((3, bytes([octet])))

    # 040
    rho_raw = int(round(tgt.range_m / NM_TO_M / RHO_LSB_NM)) & 0xFFFF
    theta_raw = int(round((tgt.az_rad * RAD_TO_DEG) / THETA_LSB_DEG)) & 0xFFFF
    items.append((4, rho_raw.to_bytes(2, "big") + theta_raw.to_bytes(2, "big")))

    # 070
    if tgt.mode3a is not None:
        w = tgt.mode3a & 0x0FFF
        if not tgt.mode3a_v:
            w |= 0x8000
        if tgt.mode3a_g:
            w |= 0x4000
        if tgt.mode3a_l:
            w |= 0x2000
        items.append((5, w.to_bytes(2, "big")))

    # 090
    if tgt.flight_level_ft is not None:
        fl_raw = int(round(tgt.flight_level_ft / 100.0 / FL_LSB)) & 0x3FFF
        items.append((6, fl_raw.to_bytes(2, "big")))

    # 220
    if tgt.aircraft_address is not None:
        items.append((8, (tgt.aircraft_address & 0xFFFFFF).to_bytes(3, "big")))

    # 161
    if tgt.track_number is not None:
        items.append((11, (tgt.track_number & 0x0FFF).to_bytes(2, "big")))

    # 200
    if tgt.ground_speed_m_s is not None and tgt.heading_rad is not None:
        spd_raw = int(round(tgt.ground_speed_m_s / NM_TO_M / SPEED_LSB_NM_S)) & 0xFFFF
        hdg_raw = int(round((tgt.heading_rad * RAD_TO_DEG) / HEAD_LSB_DEG)) & 0xFFFF
        items.append((13, spd_raw.to_bytes(2, "big") + hdg_raw.to_bytes(2, "big")))

    # 170 — minimal
    cnf = 0x00 if tgt.track_confirmed else 0x80
    mp = 0x08 if tgt.multipath else 0
    items.append((14, bytes([cnf | mp])))

    frns = {frn for frn, _ in items}
    fspec = _write_fspec(frns)
    body = b"".join(payload for _, payload in sorted(items, key=lambda x: x[0]))
    record = bytes([CAT]) + (3 + len(fspec) + len(body)).to_bytes(2, "big") + fspec + body
    return record


def encode_block(targets: list[Cat048Target]) -> bytes:
    """Encode multiple targets into a concatenated CAT 048 byte stream."""
    return b"".join(encode_record(t) for t in targets)


# ---------------------------------------------------------------------------
# Generator (REQ-M03-5)
# ---------------------------------------------------------------------------

@dataclass
class TruthTarget:
    """Ground-truth target used by the generator."""
    range_m: float
    az_rad: float
    speed_m_s: float = 0.0
    heading_rad: float = 0.0
    mode3a: int | None = None
    flight_level_ft: float | None = None
    track_number: int | None = None
    aircraft_address: int | None = None
    detection_prob: float = 1.0


def generate(
    truths: list[TruthTarget],
    *,
    sac: int = 0x50,
    sic: int = 0x01,
    time_of_day_ns: int = 0,
    range_noise_m: float = 0.0,
    az_noise_rad: float = 0.0,
    seed: int | None = None,
) -> list[Cat048Target]:
    """
    Turn truth targets into CAT 048 reports with optional noise and Pd.
    """
    rng = random.Random(seed)
    out: list[Cat048Target] = []
    for i, t in enumerate(truths):
        if rng.random() > t.detection_prob:
            continue
        r = t.range_m + rng.gauss(0.0, range_noise_m)
        a = t.az_rad + rng.gauss(0.0, az_noise_rad)
        # wrap azimuth
        a = a % (2 * math.pi)
        out.append(
            Cat048Target(
                sac=sac,
                sic=sic,
                time_of_day_ns=time_of_day_ns,
                typ=2,  # SSR
                range_m=max(0.0, r),
                az_rad=a,
                mode3a=t.mode3a,
                flight_level_ft=t.flight_level_ft,
                track_number=t.track_number if t.track_number is not None else (i + 1),
                ground_speed_m_s=t.speed_m_s,
                heading_rad=t.heading_rad,
                aircraft_address=t.aircraft_address,
                track_confirmed=True,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Hand-built test vectors (step 1 of M-03)
# ---------------------------------------------------------------------------

def hand_built_vectors() -> list[tuple[bytes, dict]]:
    """
    Three hand-crafted records with known field values.
    Used to prove the decoder before any encode path exists.
    """
    vectors = []

    # --- Vector 1: minimal — 010 + 140 + 020 + 040 ---
    # SAC=0x50, SIC=0x01
    # TOD = 3600.0 s → raw = 3600 * 128 = 460800 = 0x070800
    # 020: TYP=2 (SSR), no FX → 0x40
    # RHO = 10 NM → 10*256 = 2560 = 0x0A00
    # THETA = 90° → 90 / (360/65536) = 16384 = 0x4000
    fspec1 = bytes([0xF0])  # FRN 1,2,3,4 present (bits 8,7,6,5)
    body1 = (
        bytes([0x50, 0x01])                          # 010
        + bytes([0x07, 0x08, 0x00])                   # 140
        + bytes([0x40])                              # 020
        + bytes([0x0A, 0x00, 0x40, 0x00])             # 040
    )
    rec1 = bytes([CAT]) + (3 + len(fspec1) + len(body1)).to_bytes(2, "big") + fspec1 + body1
    vectors.append((
        rec1,
        {
            "sac": 0x50, "sic": 0x01,
            "tod_s": 3600.0,
            "range_nm": 10.0,
            "az_deg": 90.0,
        },
    ))

    # --- Vector 2: adds Mode-3/A = 1234 octal (= 0x029C), FL = 350 ---
    # Mode-3/A 1234 octal = 1*512+2*64+3*8+4 = 668 = 0x029C, V=1 → 0x029C
    # FL 350 → 350 / 0.25 = 1400 = 0x0578
    fspec2 = bytes([0xFC])  # FRN 1-6
    body2 = (
        bytes([0x50, 0x01])
        + bytes([0x00, 0x00, 0x00])                   # TOD = 0
        + bytes([0x40])
        + bytes([0x14, 0x00, 0x00, 0x00])             # 20 NM, 0°
        + bytes([0x02, 0x9C])                         # Mode-3/A
        + bytes([0x05, 0x78])                         # FL
    )
    rec2 = bytes([CAT]) + (3 + len(fspec2) + len(body2)).to_bytes(2, "big") + fspec2 + body2
    vectors.append((
        rec2,
        {
            "sac": 0x50, "sic": 0x01,
            "range_nm": 20.0,
            "az_deg": 0.0,
            "mode3a": 0x029C,
            "fl_ft": 35000.0,
        },
    ))

    # --- Vector 3: track with velocity + address + track number ---
    # Also exercises second FSPEC octet (FRN 8,11,13,14)
    # Aircraft address 0xABC123
    # Track number 42
    # Speed 250 m/s ≈ 0.135 NM/s → raw = 0.135 / 2^-14 ≈ 2211.8 → 2212
    # Heading 45°
    fspec3 = bytes([0xF1, 0x8A])  # FRN 1-4 + FX; FRN 8,11,13
    # Recalculate carefully:
    # octet0: FRN1-7 → bits for 1,2,3,4 = 0xF0, need FX for second octet → 0xF1
    # octet1: FRN8=bit8, FRN11=bit5, FRN13=bit3 → 0x80|0x10|0x04 = 0x94
    fspec3 = bytes([0xF1, 0x94])
    spd_raw = int(round(250.0 / NM_TO_M / SPEED_LSB_NM_S))
    hdg_raw = int(round(45.0 / HEAD_LSB_DEG))
    body3 = (
        bytes([0x50, 0x02])
        + bytes([0x01, 0x00, 0x00])                   # TOD ≈ 2 s
        + bytes([0x60])                              # TYP=3 combined
        + bytes([0x28, 0x00, 0x20, 0x00])             # 40 NM, 45°
        + bytes([0xAB, 0xC1, 0x23])                   # address
        + bytes([0x00, 0x2A])                         # track 42
        + spd_raw.to_bytes(2, "big") + hdg_raw.to_bytes(2, "big")
    )
    rec3 = bytes([CAT]) + (3 + len(fspec3) + len(body3)).to_bytes(2, "big") + fspec3 + body3
    vectors.append((
        rec3,
        {
            "sac": 0x50, "sic": 0x02,
            "range_nm": 40.0,
            "az_deg": 45.0,
            "address": 0xABC123,
            "track": 42,
            "speed_m_s": 250.0,
            "heading_deg": 45.0,
        },
    ))

    return vectors
