"""
A-01 / REQ-A01-2 — Multi-channel baseband generator.

Produces complex baseband IQ for UCA, SODA or spinning antenna
with one or more emitters + AWGN at a specified per-channel SNR.
"""

from __future__ import annotations
import numpy as np
from .arrays import (
    steering_vector_uca,
    steering_vector_soda,
    spinning_pattern,
)
from .impairments import apply_impairments


def _add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN so that SNR (per channel) equals snr_db."""
    power = np.mean(np.abs(signal) ** 2, axis=1, keepdims=True)
    # Avoid division by zero for silent channels
    power = np.maximum(power, 1e-20)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = power / snr_linear
    noise = (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    ) / np.sqrt(2.0)
    noise *= np.sqrt(noise_power)
    return (signal + noise).astype(np.complex64)


def generate_baseband(
    array_type: str,
    geometry: dict,
    emitters: list[dict],
    sample_rate: float,
    n_samples: int,
    snr_db: float = 20.0,
    centre_frequency: float | None = None,
    phase_imbalance_deg: list[float] | None = None,
    gain_imbalance_db: list[float] | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Generate multi-channel complex baseband IQ.

    Parameters
    ----------
    array_type : "uca" | "soda" | "spinning"
    geometry   : dict of array parameters
        UCA:      {"n_elements": int, "radius": float}
        SODA:     {"d21": float, "d31": float}
        spinning: {"beamwidth_deg": float, "rotation_rate_rps": float}
    emitters   : list of dicts, each with
        {"frequency": Hz, "bearing_deg" or "theta_deg": float, "power_db": float}
    sample_rate : Hz
    n_samples   : number of samples
    snr_db      : per-channel SNR after all emitters are summed
    centre_frequency : optional LO frequency (for metadata only)
    phase_imbalance_deg / gain_imbalance_db : optional impairments
    seed : RNG seed for reproducibility

    Returns
    -------
    iq   : complex64 array shape (n_channels, n_samples)
    meta : dict with ground-truth information
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    if array_type == "uca":
        n_ch = geometry["n_elements"]
        radius = geometry["radius"]
        iq = np.zeros((n_ch, n_samples), dtype=np.complex128)
        for em in emitters:
            freq = em["frequency"]
            az = np.deg2rad(em.get("bearing_deg", em.get("azimuth_deg", 0.0)))
            power_lin = 10 ** (em.get("power_db", 0.0) / 10.0)
            sv = steering_vector_uca(n_ch, radius, freq, az)
            tone = np.sqrt(power_lin) * np.exp(2j * np.pi * freq * t)
            iq += sv[:, np.newaxis] * tone[np.newaxis, :]

    elif array_type == "soda":
        d21 = geometry["d21"]
        d31 = geometry["d31"]
        n_ch = 3
        iq = np.zeros((n_ch, n_samples), dtype=np.complex128)
        for em in emitters:
            freq = em["frequency"]
            theta = np.deg2rad(em.get("theta_deg", em.get("bearing_deg", 0.0)))
            power_lin = 10 ** (em.get("power_db", 0.0) / 10.0)
            sv = steering_vector_soda(d21, d31, freq, theta)
            tone = np.sqrt(power_lin) * np.exp(2j * np.pi * freq * t)
            iq += sv[:, np.newaxis] * tone[np.newaxis, :]

    elif array_type == "spinning":
        # Single-channel amplitude-modulated by the rotating pattern
        n_ch = 1
        bw = np.deg2rad(geometry.get("beamwidth_deg", 30.0))
        rot_rate = geometry.get("rotation_rate_rps", 1.0)  # revolutions per second
        iq = np.zeros((n_ch, n_samples), dtype=np.complex128)
        antenna_angle = 2 * np.pi * rot_rate * t
        for em in emitters:
            freq = em["frequency"]
            # bearing of emitter relative to North; antenna starts at 0
            em_bearing = np.deg2rad(em.get("bearing_deg", 0.0))
            power_lin = 10 ** (em.get("power_db", 0.0) / 10.0)
            # relative angle between antenna and emitter
            rel_angle = antenna_angle - em_bearing
            amp = np.array([spinning_pattern(a, bw) for a in rel_angle])
            tone = np.sqrt(power_lin) * amp * np.exp(2j * np.pi * freq * t)
            iq[0] += tone

    else:
        raise ValueError(f"Unknown array_type: {array_type}")

    # Apply impairments before noise
    if phase_imbalance_deg is not None or gain_imbalance_db is not None:
        iq = apply_impairments(iq, phase_imbalance_deg, gain_imbalance_db)

    # Add AWGN
    iq = _add_awgn(iq, snr_db, rng)

    meta = {
        "array_type": array_type,
        "geometry": geometry,
        "emitters": emitters,
        "sample_rate": sample_rate,
        "n_samples": n_samples,
        "snr_db": snr_db,
        "centre_frequency": centre_frequency,
        "seed": seed,
    }
    return iq.astype(np.complex64), meta
