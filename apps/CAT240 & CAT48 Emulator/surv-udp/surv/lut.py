"""
Polar → Cartesian scan-conversion lookup table (REQ-M05-1).

For each pixel (x, y) in a square PPI image of size `size × size`,
precompute the polar coordinates (range_bin, az_bin) into the video
history buffer of shape (n_az, n_range).

Convention
----------
- Image centre = radar origin
- +x = east (right), +y = north (up)  →  screen row 0 is north
- Azimuth 0 = north, increasing clockwise (radar convention) OR
  counter-clockwise — selected by `clockwise`.
- north_up=True keeps north at top; heading_up rotates by platform heading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ScanLUT:
    """Precomputed mapping from PPI pixels → (az_index, range_index)."""

    size: int
    n_az: int
    n_range: int
    max_range_m: float
    clockwise: bool = True
    # arrays of shape (size, size)
    az_idx: np.ndarray = None  # type: ignore
    range_idx: np.ndarray = None  # type: ignore
    valid: np.ndarray = None  # type: ignore  # bool mask inside max range circle


def build_scan_lut(
    size: int = 1024,
    n_az: int = 4096,
    n_range: int = 1024,
    max_range_m: float = 100_000.0,
    clockwise: bool = True,
) -> ScanLUT:
    """
    Build a polar→Cartesian LUT.

    Pixel (col, row) with origin at centre:
        dx = col - size/2
        dy = size/2 - row     # y increases upward (north)
        r_pix = hypot(dx, dy)
        az = atan2(dx, dy)    # 0 = north; +east
    """
    centre = size / 2.0
    # column and row grids
    cols = np.arange(size, dtype=np.float64)
    rows = np.arange(size, dtype=np.float64)
    dx = cols[np.newaxis, :] - centre          # shape (1, size) broadcast → (size, size)
    dy = centre - rows[:, np.newaxis]          # north-up

    r_pix = np.hypot(dx, dy)
    # azimuth: 0 = north, increasing toward east (clockwise when north-up).
    # atan2(east, north) already gives that convention.
    az = np.arctan2(dx, dy)                    # [-π, π], 0 = north, +east
    if not clockwise:
        az = -az
    az = az % (2 * math.pi)                    # [0, 2π)

    # map to bins
    range_m = r_pix / centre * max_range_m
    range_idx = np.floor(range_m / max_range_m * n_range).astype(np.int32)
    az_idx = np.floor(az / (2 * math.pi) * n_az).astype(np.int32)

    valid = (r_pix <= centre) & (range_idx >= 0) & (range_idx < n_range)
    range_idx = np.clip(range_idx, 0, n_range - 1)
    az_idx = np.clip(az_idx, 0, n_az - 1)

    # invalidate outside
    range_idx = np.where(valid, range_idx, 0)
    az_idx = np.where(valid, az_idx, 0)

    return ScanLUT(
        size=size,
        n_az=n_az,
        n_range=n_range,
        max_range_m=max_range_m,
        clockwise=clockwise,
        az_idx=az_idx,
        range_idx=range_idx,
        valid=valid,
    )


def scan_convert(
    video: np.ndarray,
    lut: ScanLUT,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert polar video (n_az, n_range) → Cartesian image (size, size).

    `video` dtype any numeric; output is same dtype (zeros outside valid disk).
    """
    if video.shape != (lut.n_az, lut.n_range):
        raise ValueError(f"video shape {video.shape} != ({lut.n_az}, {lut.n_range})")
    if out is None:
        out = np.zeros((lut.size, lut.size), dtype=video.dtype)
    else:
        out.fill(0)
    # gather
    vals = video[lut.az_idx, lut.range_idx]
    out[lut.valid] = vals[lut.valid]
    return out


def pixel_for_polar(
    range_m: float,
    az_rad: float,
    size: int,
    max_range_m: float,
    clockwise: bool = True,
) -> tuple[int, int]:
    """
    Hand-computable pixel (col, row) for a polar point.
    Used by unit tests for 0°, 90°, 180°, 270°.
    """
    centre = size / 2.0
    r_pix = (range_m / max_range_m) * centre
    # az=0 north, +east: dx = r sin(az), dy = r cos(az)
    if not clockwise:
        az_rad = -az_rad
    dx = r_pix * math.sin(az_rad)
    dy = r_pix * math.cos(az_rad)
    col = int(round(centre + dx))
    row = int(round(centre - dy))  # row 0 = top = north
    return col, row
