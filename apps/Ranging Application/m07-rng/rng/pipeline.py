"""
rng/pipeline.py — Ties waveform + dsp together into the RNG processing
chain, and tracks range rate across successive blocks per detection bin.

Also handles the CAT240-video input path: when RNG is given an already
decoded video sweep (amplitude vs range, no raw IQ) it skips pulse
compression and runs CFAR directly on the supplied profile.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from . import dsp
from .waveforms import make_waveform


@dataclass
class PipelineConfig:
    waveform_kind: str = "chirp"
    bandwidth_hz: float = 20e6
    duration_s: float = 2e-6
    sample_rate_hz: float = 50e6
    cfar_kind: str = "ca"            # "ca" | "os"
    guard: int = 2
    train: int = 8
    pfa: float = 1e-3
    k_rank: float = 0.75
    range_gate_m: float = 300.0      # bin width for associating detections across blocks


@dataclass
class DetectionRecord:
    range_m: float
    amplitude_db: float
    threshold_db: float
    timestamp_ns: int
    range_rate_mps: float | None = None


class RngPipeline:
    """Stateful: remembers recent detections per range bin to estimate range rate."""

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self._history: dict[int, list[tuple[float, float]]] = {}  # bin -> [(t_s, range_m), ...]
        self.last_profile_db: np.ndarray | None = None
        self.last_ranges_m: np.ndarray | None = None
        self.last_threshold_db: np.ndarray | None = None
        self.last_detections: list[DetectionRecord] = []
        self.input_count = 0
        self.processed_count = 0
        self.dropped_count = 0

    def _bin_of(self, range_m: float) -> int:
        return int(round(range_m / self.cfg.range_gate_m))

    def _update_range_rate(self, range_m: float, t_ns: int) -> float | None:
        t_s = t_ns * 1e-9
        b = self._bin_of(range_m)
        hist = self._history.setdefault(b, [])
        hist.append((t_s, range_m))
        if len(hist) > 8:
            del hist[: len(hist) - 8]
        times = [h[0] for h in hist]
        ranges = [h[1] for h in hist]
        return dsp.estimate_range_rate(times, ranges)

    def process_raw_iq(self, samples: np.ndarray, sample_rate_hz: float, t_ns: int) -> list[DetectionRecord]:
        """Raw-IQ input path: pulse compression + CFAR on a single channel."""
        self.input_count += 1
        if samples.ndim == 2:
            rx = samples[0]
        else:
            rx = samples
        if sample_rate_hz != self.cfg.sample_rate_hz:
            # Not fatal: process at the block's own rate, but matched filter
            # must be built against that same rate.
            fs = sample_rate_hz
        else:
            fs = self.cfg.sample_rate_hz
        wf = make_waveform(self.cfg.waveform_kind, self.cfg.bandwidth_hz, self.cfg.duration_s, fs)
        if len(wf) > len(rx):
            self.dropped_count += 1
            return []
        compressed = dsp.pulse_compress(rx, wf)
        power_db = 20 * np.log10(np.maximum(np.abs(compressed), 1e-12))
        ranges_m = dsp.range_axis(len(compressed), fs)
        return self._cfar_and_track(power_db, ranges_m, t_ns)

    def process_video(self, amplitude_db: np.ndarray, ranges_m: np.ndarray, t_ns: int) -> list[DetectionRecord]:
        """CAT240-video input path: CFAR directly on a decoded amplitude-vs-range profile."""
        self.input_count += 1
        return self._cfar_and_track(np.asarray(amplitude_db, dtype=np.float64), np.asarray(ranges_m, dtype=np.float64), t_ns)

    def _cfar_and_track(self, power_db: np.ndarray, ranges_m: np.ndarray, t_ns: int) -> list[DetectionRecord]:
        result = dsp.run_cfar(power_db, self.cfg.cfar_kind, self.cfg.guard, self.cfg.train, self.cfg.pfa, self.cfg.k_rank)
        peaks = dsp.peaks_from_mask(power_db, ranges_m, result.mask, result.threshold_db)
        detections = []
        for p in peaks:
            rate = self._update_range_rate(p.range_m, t_ns)
            detections.append(
                DetectionRecord(
                    range_m=p.range_m,
                    amplitude_db=p.amplitude_db,
                    threshold_db=p.threshold_db,
                    timestamp_ns=t_ns,
                    range_rate_mps=rate,
                )
            )
        self.last_profile_db = power_db
        self.last_ranges_m = ranges_m
        self.last_threshold_db = result.threshold_db
        self.last_detections = detections
        self.processed_count += 1
        return detections
