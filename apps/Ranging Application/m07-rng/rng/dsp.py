"""
rng/dsp.py — Signal processing core for the RNG application (M-07).

Pipeline stages, each a pure function so they are independently testable:

  1. pulse_compress(rx, waveform)   -> complex range profile (matched filter)
  2. range_axis(n, sample_rate_hz)  -> metres for each range-profile cell
  3. ca_cfar(power_db, ...)         -> boolean detections + adaptive threshold
     os_cfar(power_db, ...)         -> boolean detections + adaptive threshold
  4. peaks_from_mask(...)           -> one peak per contiguous detection run
  5. estimate_range_rate(history)   -> m/s from successive range measurements

Units: ranges in metres, amplitudes in dB, range rate in m/s (positive =
opening / increasing range).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


# ---------------------------------------------------------------------------
# Pulse compression (matched filter)
# ---------------------------------------------------------------------------

def pulse_compress(rx: np.ndarray, waveform: np.ndarray) -> np.ndarray:
    """
    Matched-filter `rx` (1-D complex samples) against `waveform`.

    Returns the complex range profile, same length as `rx`, aligned so
    that index 0 corresponds to zero range (the leading edge of a
    target echo overlapping the start of the receive window).
    """
    rx = np.asarray(rx, dtype=np.complex64)
    waveform = np.asarray(waveform, dtype=np.complex64)
    if rx.ndim != 1:
        raise ValueError("rx must be 1-D")
    matched = np.conj(waveform[::-1])
    full = np.convolve(rx, matched, mode="full")
    # The matched-filter peak for an echo starting at sample k in rx lands
    # at index k + len(waveform) - 1 in `full`; shift back by that so the
    # output's index 0 means "echo starts at sample 0 of rx" (zero range).
    start = len(waveform) - 1
    return full[start : start + len(rx)]


def range_axis(n_cells: int, sample_rate_hz: float) -> np.ndarray:
    """Range (m) of each range-profile cell for a two-way time-of-flight."""
    dt = 1.0 / sample_rate_hz
    return np.arange(n_cells, dtype=np.float64) * dt * C_LIGHT / 2.0


# ---------------------------------------------------------------------------
# CFAR detectors
# ---------------------------------------------------------------------------

@dataclass
class CfarResult:
    mask: np.ndarray        # bool, True where a detection was declared
    threshold_db: np.ndarray  # adaptive threshold per cell, dB


def _cfar_windows(n: int, guard: int, train: int):
    """Yield (cell_index, lag_slice, lead_slice) training-cell index pairs."""
    for i in range(n):
        lag_start = max(0, i - guard - train)
        lag_end = max(0, i - guard)
        lead_start = min(n, i + guard + 1)
        lead_end = min(n, i + guard + 1 + train)
        yield i, (lag_start, lag_end), (lead_start, lead_end)


def ca_cfar(power_db: np.ndarray, guard: int = 2, train: int = 8, pfa: float = 1e-3) -> CfarResult:
    """
    Cell-Averaging CFAR. Threshold = mean(training cells, linear power)
    scaled by the CA-CFAR alpha factor for the requested Pfa, converted
    back to dB. Falls back gracefully near the array edges (fewer
    training cells there still produce a threshold).
    """
    n = len(power_db)
    power_lin = 10 ** (np.asarray(power_db, dtype=np.float64) / 10.0)
    mask = np.zeros(n, dtype=bool)
    thresh_db = np.full(n, np.nan)
    for i, (ls, le), (ds, de) in _cfar_windows(n, guard, train):
        cells = np.concatenate([power_lin[ls:le], power_lin[ds:de]])
        n_train = len(cells)
        if n_train == 0:
            continue
        alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
        noise_est = np.mean(cells)
        thresh_lin = alpha * noise_est
        thresh_db[i] = 10 * np.log10(max(thresh_lin, 1e-30))
        if power_lin[i] > thresh_lin:
            mask[i] = True
    return CfarResult(mask=mask, threshold_db=thresh_db)


def os_cfar(power_db: np.ndarray, guard: int = 2, train: int = 8, pfa: float = 1e-3, k_rank: float = 0.75) -> CfarResult:
    """
    Ordered-Statistic CFAR. Instead of averaging the training cells,
    sort them and take the k-th order statistic (k_rank fraction of the
    way through, e.g. 0.75 -> 75th percentile) as the noise estimate.
    More robust to multiple closely-spaced targets / clutter edges
    falling inside the training window than CA-CFAR.
    """
    n = len(power_db)
    power_lin = 10 ** (np.asarray(power_db, dtype=np.float64) / 10.0)
    mask = np.zeros(n, dtype=bool)
    thresh_db = np.full(n, np.nan)
    for i, (ls, le), (ds, de) in _cfar_windows(n, guard, train):
        cells = np.concatenate([power_lin[ls:le], power_lin[ds:de]])
        n_train = len(cells)
        if n_train == 0:
            continue
        k = max(1, min(n_train, int(round(k_rank * n_train))))
        sorted_cells = np.sort(cells)
        noise_est = sorted_cells[k - 1]
        # OS-CFAR alpha approximation (same Pfa inversion as CA, applied
        # to the order-statistic estimate rather than the mean).
        alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
        thresh_lin = alpha * noise_est
        thresh_db[i] = 10 * np.log10(max(thresh_lin, 1e-30))
        if power_lin[i] > thresh_lin:
            mask[i] = True
    return CfarResult(mask=mask, threshold_db=thresh_db)


def run_cfar(power_db: np.ndarray, kind: str, guard: int, train: int, pfa: float, k_rank: float = 0.75) -> CfarResult:
    if kind == "ca":
        return ca_cfar(power_db, guard=guard, train=train, pfa=pfa)
    if kind == "os":
        return os_cfar(power_db, guard=guard, train=train, pfa=pfa, k_rank=k_rank)
    raise ValueError(f"unknown CFAR kind: {kind!r}")


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------

@dataclass
class Peak:
    index: int
    range_m: float
    amplitude_db: float
    threshold_db: float


def peaks_from_mask(power_db: np.ndarray, ranges_m: np.ndarray, mask: np.ndarray, threshold_db: np.ndarray) -> list[Peak]:
    """Collapse contiguous runs of `mask` into one peak (the local max) each."""
    peaks: list[Peak] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        stop = i
        local = power_db[start:stop]
        rel = int(np.argmax(local))
        idx = start + rel
        peaks.append(
            Peak(
                index=idx,
                range_m=float(ranges_m[idx]),
                amplitude_db=float(power_db[idx]),
                threshold_db=float(threshold_db[idx]) if not np.isnan(threshold_db[idx]) else float("-inf"),
            )
        )
    return peaks


# ---------------------------------------------------------------------------
# Range-rate estimation
# ---------------------------------------------------------------------------

def estimate_range_rate(times_s: list[float], ranges_m: list[float]) -> float | None:
    """
    Least-squares slope of range vs time over the supplied history
    (most recent measurements of one track). Returns m/s, or None if
    there are fewer than two points or all timestamps coincide.
    """
    if len(times_s) < 2 or len(times_s) != len(ranges_m):
        return None
    t = np.asarray(times_s, dtype=np.float64)
    r = np.asarray(ranges_m, dtype=np.float64)
    t = t - t[0]
    if np.ptp(t) <= 0:
        return None
    a, _b = np.polyfit(t, r, 1)
    return float(a)
