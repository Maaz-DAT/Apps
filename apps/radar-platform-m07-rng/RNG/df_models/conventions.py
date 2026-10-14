"""
A-01 / REQ-A01-4 — Element phase conventions and azimuth references.

These conventions are used by every DF app (CIDF, SODA, SDF) and must
never be changed without updating all consumers.
"""

from __future__ import annotations
import numpy as np

# Physical constants
C = 299_792_458.0          # speed of light (m/s)
C_NS = C * 1e-9            # m/ns  (useful for TOA)

# ---------------------------------------------------------------------------
# Angle conventions (document once, use everywhere)
# ---------------------------------------------------------------------------

# Uniform Circular Array (CIDF)
#   azimuth = 0  → North
#   positive     → clockwise
UCA_AZIMUTH_ZERO = "North"
UCA_POSITIVE_DIRECTION = "clockwise"

# 3-element Linear SODA array
#   θ = 0        → broadside (perpendicular to array axis)
#   positive θ   → direction of increasing element index
SODA_THETA_ZERO = "broadside"
SODA_POSITIVE_DIRECTION = "increasing element index"

# Spinning directional antenna (SDF)
#   antenna_angle = 0 → boresight of the antenna pattern
SPINNING_ANGLE_ZERO = "boresight"


def deg2rad(deg: float | np.ndarray) -> float | np.ndarray:
    return np.deg2rad(deg)


def rad2deg(rad: float | np.ndarray) -> float | np.ndarray:
    return np.rad2deg(rad)


def wrap_phase(phase: float | np.ndarray) -> float | np.ndarray:
    """Wrap phase into [-π, π]."""
    return (phase + np.pi) % (2 * np.pi) - np.pi


def wavelength(frequency_hz: float) -> float:
    """Wavelength in metres."""
    return C / frequency_hz
