"""
A-01 — Array geometry and steering-vector models.

REQ-A01-1: Steering vectors for
  - Uniform Circular Array (UCA) – used by CIDF
  - 3-element linear SODA array
  - Spinning directional antenna (amplitude pattern only)
"""

from __future__ import annotations
import numpy as np
from .conventions import C, wrap_phase, wavelength


# ---------------------------------------------------------------------------
# Uniform Circular Array (CIDF)
# ---------------------------------------------------------------------------

def uca_element_positions(n_elements: int, radius: float) -> np.ndarray:
    """
    Return (N, 2) array of (x, y) element positions in metres.
    Element 0 is at azimuth 0 (North) relative to array centre.
    Positive azimuth is clockwise.
    """
    if n_elements < 3:
        raise ValueError("UCA requires at least 3 elements")
    # Angles of elements measured clockwise from North
    angles = 2 * np.pi * np.arange(n_elements) / n_elements
    # North = +y, East = +x  → clockwise from North
    x = radius * np.sin(angles)
    y = radius * np.cos(angles)
    return np.column_stack([x, y])


def steering_vector_uca(
    n_elements: int,
    radius: float,
    frequency_hz: float,
    azimuth_rad: float,
) -> np.ndarray:
    """
    Complex steering vector of shape (n_elements,).

    Phase reference is the geometric centre of the array.
    Azimuth = 0 → North, positive clockwise (REQ-A01-4).
    """
    pos = uca_element_positions(n_elements, radius)
    # Direction of arrival unit vector (clockwise from North)
    # u = (sin(az), cos(az))  because North = +y
    u = np.array([np.sin(azimuth_rad), np.cos(azimuth_rad)])
    # Path length difference relative to centre
    path_diff = pos @ u                       # metres
    phase = 2 * np.pi * frequency_hz / C * path_diff
    return np.exp(1j * phase).astype(np.complex64)


# ---------------------------------------------------------------------------
# 3-element Linear SODA array
# ---------------------------------------------------------------------------

def validate_soda_geometry(d21: float, d31: float, frequency_hz: float) -> float:
    """
    Enforce d31 = 2·d21 + dΔ with dΔ ≤ λ_min/2 (REQ-A05-1 / REQ-A01-1).
    Returns dΔ.
    Raises ValueError if the geometry is invalid.
    """
    d_delta = d31 - 2 * d21
    if d_delta <= 0:
        raise ValueError(
            f"Invalid SODA geometry: dΔ = {d_delta:.6f} m must be > 0 "
            f"(d31 must be > 2·d21)"
        )
    lam = wavelength(frequency_hz)
    if d_delta > lam / 2 + 1e-12:
        raise ValueError(
            f"Invalid SODA geometry: dΔ = {d_delta:.6f} m > λ/2 = {lam/2:.6f} m "
            f"at {frequency_hz/1e9:.3f} GHz. Ambiguity cannot be resolved."
        )
    return d_delta


def soda_element_positions(d21: float, d31: float) -> np.ndarray:
    """
    Return (3,) array of element x-positions (array lies on the x-axis).
    Element 0 at x=0, element 1 at x=d21, element 2 at x=d31.
    """
    return np.array([0.0, d21, d31], dtype=np.float64)


def steering_vector_soda(
    d21: float,
    d31: float,
    frequency_hz: float,
    theta_rad: float,
) -> np.ndarray:
    """
    Complex steering vector of shape (3,).

    θ = 0 is broadside (REQ-A01-4).
    Phase of element 0 is used as reference (phase = 0).
    """
    validate_soda_geometry(d21, d31, frequency_hz)
    pos = soda_element_positions(d21, d31)
    # Path difference relative to element 0
    # for a wave arriving from angle θ (from broadside)
    path_diff = pos * np.sin(theta_rad)
    phase = 2 * np.pi * frequency_hz / C * path_diff
    return np.exp(1j * phase).astype(np.complex64)


# ---------------------------------------------------------------------------
# Spinning directional antenna (SDF)
# ---------------------------------------------------------------------------

def spinning_pattern(
    antenna_angle_rad: float,
    beamwidth_rad: float = np.deg2rad(30.0),
    pattern: str = "cos2",
) -> float:
    """
    Amplitude gain of a rotating directional antenna.

    antenna_angle_rad : angle of the antenna boresight relative to the
                        emitter (0 = emitter is on boresight).
    Returns a real amplitude gain ≥ 0.
    """
    # Fold into [-π, π]
    a = wrap_phase(antenna_angle_rad)
    half_bw = beamwidth_rad / 2.0

    if pattern == "cos2":
        # Classic cos² pattern truncated outside the main beam
        if abs(a) > half_bw * 2:          # practical cutoff
            return 0.0
        return float(np.cos(a * (np.pi / (2 * half_bw))) ** 2)

    if pattern == "gaussian":
        sigma = beamwidth_rad / 2.355     # FWHM ≈ beamwidth
        return float(np.exp(-0.5 * (a / sigma) ** 2))

    raise ValueError(f"Unknown pattern type: {pattern}")


def spinning_steering_amplitude(
    n_angles: int,
    beamwidth_rad: float = np.deg2rad(30.0),
    pattern: str = "cos2",
) -> np.ndarray:
    """Return amplitude pattern sampled over a full rotation (for tests)."""
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    return np.array([spinning_pattern(a, beamwidth_rad, pattern) for a in angles])
