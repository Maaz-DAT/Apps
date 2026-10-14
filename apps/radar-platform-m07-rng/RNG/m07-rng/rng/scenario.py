"""
rng/scenario.py — Synthetic IQBlock source for RNG (both input paths).

Generates single-channel complex64 IQBlock messages on the frozen M-00
schema (common.messages.IQBlock) containing a transmitted waveform
(chirp or simple pulse) convolved with one or more target echoes at
configured ranges/velocities/RCS, plus complex AWGN at a configured
SNR. This drives both RNG input paths:

  * "video"   — RNG only looks at |compressed profile| vs range.
  * "raw_iq"  — RNG receives the IQBlock itself and runs pulse
                compression + CFAR end-to-end, exactly as it would on
                real IQBlock traffic from the bus.

Seeded for repeatable tests (REQ-M04 style determinism).
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .waveforms import make_waveform
from .dsp import C_LIGHT

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.messages import IQBlock  # noqa: E402


@dataclass
class Target:
    range_m: float
    velocity_mps: float = 0.0   # positive = opening range
    rcs_db: float = 0.0         # relative echo amplitude, dB


@dataclass
class ScenarioConfig:
    waveform_kind: str = "chirp"          # "chirp" | "simple"
    bandwidth_hz: float = 20e6
    duration_s: float = 2e-6
    sample_rate_hz: float = 50e6
    centre_frequency_hz: float = 9.4e9
    n_samples: int = 4096
    snr_db: float = 15.0
    targets: list[Target] = field(default_factory=lambda: [Target(range_m=15_000.0)])
    seed: int | None = None


def _add_awgn(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sig_power = np.mean(np.abs(sig) ** 2) if np.any(sig) else 1.0
    sig_power = max(sig_power, 1e-20)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape)) / np.sqrt(2.0)
    return sig + noise * np.sqrt(noise_power)


def generate_iq(cfg: ScenarioConfig, t0_s: float = 0.0, seed: int | None = None) -> np.ndarray:
    """Build one receive-window of complex64 IQ containing the configured targets."""
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    wf = make_waveform(cfg.waveform_kind, cfg.bandwidth_hz, cfg.duration_s, cfg.sample_rate_hz)
    rx = np.zeros(cfg.n_samples, dtype=np.complex128)
    wlen = len(wf)
    for tgt in cfg.targets:
        delay_s = 2.0 * (tgt.range_m + tgt.velocity_mps * t0_s) / C_LIGHT
        delay_samples = delay_s * cfg.sample_rate_hz
        start = int(round(delay_samples))
        if start >= cfg.n_samples or start + wlen < 0:
            continue
        amp = 10 ** (tgt.rcs_db / 20.0)
        # Doppler phase ramp across the pulse from radial velocity.
        if cfg.centre_frequency_hz > 0:
            doppler_hz = -2.0 * tgt.velocity_mps * cfg.centre_frequency_hz / C_LIGHT
        else:
            doppler_hz = 0.0
        t = np.arange(wlen, dtype=np.float64) / cfg.sample_rate_hz
        echo = amp * wf.astype(np.complex128) * np.exp(1j * 2 * np.pi * doppler_hz * t)
        lo = max(0, start)
        hi = min(cfg.n_samples, start + wlen)
        if hi <= lo:
            continue
        echo_lo = lo - start
        echo_hi = echo_lo + (hi - lo)
        rx[lo:hi] += echo[echo_lo:echo_hi]
    rx = _add_awgn(rx, cfg.snr_db, rng)
    return rx.astype(np.complex64)


def generate_iqblock(cfg: ScenarioConfig, t_ns: int, source_id: str = "rng-synth", seed: int | None = None) -> IQBlock:
    samples = generate_iq(cfg, t0_s=t_ns * 1e-9, seed=seed)
    return IQBlock(
        schema_version=1,
        t_ns=t_ns,
        source_id=source_id,
        samples=samples.reshape(1, -1),
        sample_rate_hz=cfg.sample_rate_hz,
        centre_frequency_hz=cfg.centre_frequency_hz,
        channel_map=[0],
        array_id="rng-single-channel",
        platform_lat_deg=0.0,
        platform_lon_deg=0.0,
        platform_alt_m=0.0,
        platform_heading_rad=0.0,
    )
