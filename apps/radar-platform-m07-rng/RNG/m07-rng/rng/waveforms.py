"""
rng/waveforms.py — Reference transmit waveforms for pulse compression (M-07).

Two selectable waveforms, both returned as complex64 baseband samples
at a given sample rate:

  * "chirp"  — linear FM (up-chirp), bandwidth_hz over duration_s.
               Matched filtering gives range compression gain
               approx. time-bandwidth product (duration_s * bandwidth_hz).
  * "simple" — unmodulated rectangular pulse of duration_s.
               Matched filtering is a simple boxcar correlation; no
               compression gain, range resolution set by pulse length.

Both waveforms are normalised to unit energy so that detection
thresholds behave consistently across waveform choice.
"""
from __future__ import annotations
import numpy as np


def chirp_waveform(bandwidth_hz: float, duration_s: float, sample_rate_hz: float) -> np.ndarray:
    """Linear FM up-chirp, baseband, unit energy, complex64."""
    if bandwidth_hz <= 0 or duration_s <= 0 or sample_rate_hz <= 0:
        raise ValueError("bandwidth_hz, duration_s and sample_rate_hz must be positive")
    n = max(2, int(round(duration_s * sample_rate_hz)))
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    k = bandwidth_hz / duration_s  # chirp rate, Hz/s
    phase = np.pi * k * t ** 2 - np.pi * bandwidth_hz * t  # centred FM sweep
    wf = np.exp(1j * phase)
    wf /= np.sqrt(np.sum(np.abs(wf) ** 2))
    return wf.astype(np.complex64)


def simple_pulse(duration_s: float, sample_rate_hz: float) -> np.ndarray:
    """Unmodulated rectangular pulse, unit energy, complex64."""
    if duration_s <= 0 or sample_rate_hz <= 0:
        raise ValueError("duration_s and sample_rate_hz must be positive")
    n = max(2, int(round(duration_s * sample_rate_hz)))
    wf = np.ones(n, dtype=np.complex128)
    wf /= np.sqrt(np.sum(np.abs(wf) ** 2))
    return wf.astype(np.complex64)


def make_waveform(kind: str, bandwidth_hz: float, duration_s: float, sample_rate_hz: float) -> np.ndarray:
    if kind == "chirp":
        return chirp_waveform(bandwidth_hz, duration_s, sample_rate_hz)
    if kind == "simple":
        return simple_pulse(duration_s, sample_rate_hz)
    raise ValueError(f"unknown waveform kind: {kind!r}")
