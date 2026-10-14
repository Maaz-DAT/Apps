"""
A-02 / REQ-A02-3 — PDW extraction from multi-channel IQ.

Envelope detection with noise-based threshold,
TOA at leading-edge half-amplitude point,
pulse width, frequency from FFT peak,
amplitude and phase per channel.
"""

from __future__ import annotations
import numpy as np
from .codecs import PDW


def _envelope(x: np.ndarray) -> np.ndarray:
    """Simple magnitude envelope."""
    return np.abs(x)


def extract_pdws(
    iq: np.ndarray,
    sample_rate: float,
    timestamp_ns: int = 0,
    threshold_db: float = 10.0,
    min_pulse_samples: int = 8,
    max_pulse_samples: int = 10_000,
) -> list[PDW]:
    """
    Extract PDWs from a multi-channel IQ block.

    Parameters
    ----------
    iq : complex array (n_channels, n_samples)
    sample_rate : Hz
    timestamp_ns : absolute time of sample 0
    threshold_db : detection threshold above noise floor
    min_pulse_samples / max_pulse_samples : length gates

    Returns
    -------
    list of PDW objects (one per detected pulse)
    """
    if iq.ndim != 2:
        raise ValueError("iq must be (n_channels, n_samples)")

    n_ch, n_s = iq.shape
    # Use sum of channel powers for detection
    env = np.sqrt(np.sum(np.abs(iq) ** 2, axis=0))
    # Noise floor estimate from lowest 20 % of samples
    sorted_env = np.sort(env)
    noise_floor = np.median(sorted_env[: max(1, n_s // 5)])
    if noise_floor < 1e-12:
        noise_floor = 1e-12
    thresh = noise_floor * (10 ** (threshold_db / 20.0))

    # Find contiguous regions above threshold
    above = env > thresh
    pdws: list[PDW] = []
    i = 0
    while i < n_s:
        if not above[i]:
            i += 1
            continue
        start = i
        while i < n_s and above[i]:
            i += 1
        stop = i
        length = stop - start
        if length < min_pulse_samples or length > max_pulse_samples:
            continue

        # Leading-edge half-amplitude TOA
        half_amp = 0.5 * np.max(env[start:stop])
        # search from start for the first sample that crosses half_amp
        toa_idx = start
        for k in range(start, stop):
            if env[k] >= half_amp:
                toa_idx = k
                break
        toa_ns = timestamp_ns + int(round(toa_idx / sample_rate * 1e9))

        # Pulse width (samples above half-amp inside the region)
        above_half = env[start:stop] >= half_amp
        pw_samples = int(np.sum(above_half))
        pw_s = pw_samples / sample_rate

        # Frequency from FFT peak of the strongest channel
        pulse = iq[:, start:stop]
        # pick channel with highest energy
        ch_energy = np.sum(np.abs(pulse) ** 2, axis=1)
        best_ch = int(np.argmax(ch_energy))
        spectrum = np.fft.fft(pulse[best_ch] * np.hanning(length))
        freqs = np.fft.fftfreq(length, d=1.0 / sample_rate)
        peak_bin = int(np.argmax(np.abs(spectrum)))
        freq_hz = float(abs(freqs[peak_bin]))

        # Amplitude & phase per channel (at the peak sample)
        peak_idx = start + int(np.argmax(env[start:stop]))
        amp = [float(np.abs(iq[c, peak_idx])) for c in range(n_ch)]
        phase = [float(np.angle(iq[c, peak_idx])) for c in range(n_ch)]

        # Crude SNR
        signal_power = np.mean(env[start:stop] ** 2)
        snr_db = 10 * np.log10(signal_power / (noise_floor ** 2 + 1e-30))

        pdws.append(
            PDW(
                toa_ns=toa_ns,
                frequency_hz=freq_hz,
                pulse_width_s=pw_s,
                amplitude=amp,
                phase=phase,
                flags=0,
                snr_db=float(snr_db),
            )
        )
    return pdws
