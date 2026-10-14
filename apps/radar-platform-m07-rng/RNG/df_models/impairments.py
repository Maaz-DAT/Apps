"""
A-01 / REQ-A01-3 — Impairment injection.

Supports:
  - per-channel phase imbalance (degrees)
  - per-channel gain imbalance (dB)
  - baseline (element position) error – applied by the caller on geometry
"""

from __future__ import annotations
import numpy as np


def apply_impairments(
    iq: np.ndarray,
    phase_imbalance_deg: list[float] | np.ndarray | None = None,
    gain_imbalance_db: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply per-channel phase and gain imbalances to an IQ block.

    Parameters
    ----------
    iq : complex array, shape (n_channels, n_samples)
    phase_imbalance_deg : list of length n_channels (degrees)
    gain_imbalance_db   : list of length n_channels (dB)

    Returns
    -------
    impaired IQ of the same shape and dtype.
    """
    if iq.ndim != 2:
        raise ValueError("iq must have shape (n_channels, n_samples)")

    n_ch = iq.shape[0]
    out = iq.copy()

    if phase_imbalance_deg is not None:
        phase = np.asarray(phase_imbalance_deg, dtype=np.float64)
        if phase.shape != (n_ch,):
            raise ValueError(f"phase_imbalance_deg length must be {n_ch}")
        out *= np.exp(1j * np.deg2rad(phase))[:, np.newaxis]

    if gain_imbalance_db is not None:
        gain = np.asarray(gain_imbalance_db, dtype=np.float64)
        if gain.shape != (n_ch,):
            raise ValueError(f"gain_imbalance_db length must be {n_ch}")
        linear = 10 ** (gain / 20.0)
        out *= linear[:, np.newaxis]

    return out.astype(iq.dtype)


def apply_position_error(
    positions: np.ndarray,
    position_error_m: list[tuple[float, float]] | np.ndarray,
) -> np.ndarray:
    """
    Add (dx, dy) position errors to element positions.

    positions : (N, 2) array
    position_error_m : list/array of (dx, dy) for each element
    """
    err = np.asarray(position_error_m, dtype=np.float64)
    if err.shape != positions.shape:
        raise ValueError("position_error_m must match positions shape")
    return positions + err
