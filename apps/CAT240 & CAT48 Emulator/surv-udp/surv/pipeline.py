"""
End-to-end codec pipeline for SURV:
  truth → CAT 240 / CAT 048 generate → encode → decode → display data

Exercises the full encode/decode path (M-06 step 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from surv.cat048 import Cat048Target, TruthTarget, encode_block as enc048, decode_buffer as dec048, generate as gen048
from surv.cat240 import Cat240Sweep, TruthVideoTarget, encode_record as enc240, decode_buffer as dec240, generate_sweep
from surv.scenario import ScenarioTarget, make_scenario_20, target_at_time


@dataclass
class PlotSymbol:
    """Decoded plot for overlay."""
    range_m: float
    az_rad: float
    mode3a: int | None
    fl_ft: float | None
    track_number: int | None
    address: int | None
    age_scans: int = 0
    # full decoded fields for detail panel
    fields: dict = field(default_factory=dict)


@dataclass
class FrameBundle:
    """One antenna scan worth of data for the PPI."""
    polar: np.ndarray          # (n_az, n_range) float32 video
    plots: list[PlotSymbol]
    scan_index: int
    n_az: int
    n_range: int
    max_range_m: float
    cell_size_m: float
    dropped: int = 0


def build_scan(
    targets: list[ScenarioTarget],
    *,
    scan_index: int = 0,
    n_az: int = 360,
    n_range: int = 512,
    max_range_m: float = 100_000.0,
    cell_size_m: float = None,
    scan_period_s: float = 4.0,
    noise_sigma: float = 6.0,
    clutter_range_m: float = 4000.0,
    sac: int = 0x50,
    sic: int = 0x01,
    seed: int = 0,
    use_codec: bool = True,
) -> FrameBundle:
    """
    Produce one full scan: polar video + decoded CAT 048 plots.

    If use_codec=True, each radial goes through CAT 240 encode→decode
    and plots through CAT 048 encode→decode.
    """
    if cell_size_m is None:
        cell_size_m = max_range_m / n_range

    t_s = scan_index * scan_period_s
    # truth positions at this scan
    truth_pos = []
    for tgt in targets:
        r, az = target_at_time(tgt, t_s)
        truth_pos.append((tgt, r, az))

    # --- Video via CAT 240 path ---
    polar = np.zeros((n_az, n_range), dtype=np.float32)
    video_targets = [
        TruthVideoTarget(range_m=r, az_rad=az, amplitude=tgt.amplitude, width_m=80.0)
        for tgt, r, az in truth_pos
    ]

    for i in range(n_az):
        az = 2 * math.pi * i / n_az
        sw = generate_sweep(
            az_rad=az,
            az_width_rad=2 * math.pi / n_az,
            n_cells=n_range,
            cell_size_m=cell_size_m,
            start_range_m=0.0,
            targets=video_targets,
            noise_sigma=noise_sigma,
            clutter_range_m=clutter_range_m,
            resolution_bits=8,
            sac=sac, sic=sic,
            msg_index=scan_index * n_az + i,
            seed=seed + scan_index * n_az + i,
        )
        if use_codec:
            raw = enc240(sw)
            decoded, _ = dec240(raw)
            if decoded:
                amps = decoded[0].amplitudes.astype(np.float32)
                n = min(n_range, len(amps))
                polar[i, :n] = amps[:n]
        else:
            polar[i, :] = sw.amplitudes.astype(np.float32)

    # --- Plots via CAT 048 path ---
    truths_048 = []
    for tgt, r, az in truth_pos:
        truths_048.append(TruthTarget(
            range_m=r,
            az_rad=az,
            speed_m_s=tgt.speed_m_s,
            heading_rad=tgt.heading_rad,
            mode3a=tgt.mode3a,
            flight_level_ft=tgt.fl_ft,
            track_number=tgt.track_id,
            aircraft_address=tgt.address,
            detection_prob=1.0,
        ))
    reports = gen048(
        truths_048,
        sac=sac, sic=sic,
        time_of_day_ns=int(t_s * 1e9),
        range_noise_m=0.0,   # alignment test needs geometric match
        az_noise_rad=0.0,
        seed=seed,
    )
    plots: list[PlotSymbol] = []
    if use_codec:
        raw = enc048(reports)
        decoded, _ = dec048(raw)
        for t in decoded:
            plots.append(PlotSymbol(
                range_m=t.range_m,
                az_rad=t.az_rad,
                mode3a=t.mode3a,
                fl_ft=t.flight_level_ft,
                track_number=t.track_number,
                address=t.aircraft_address,
                fields={
                    "sac": t.sac, "sic": t.sic,
                    "range_m": t.range_m,
                    "az_rad": t.az_rad,
                    "mode3a": t.mode3a,
                    "flight_level_ft": t.flight_level_ft,
                    "track_number": t.track_number,
                    "aircraft_address": t.aircraft_address,
                    "ground_speed_m_s": t.ground_speed_m_s,
                    "heading_rad": t.heading_rad,
                    "time_of_day_ns": t.time_of_day_ns,
                },
            ))
    else:
        for t in reports:
            plots.append(PlotSymbol(
                range_m=t.range_m, az_rad=t.az_rad,
                mode3a=t.mode3a, fl_ft=t.flight_level_ft,
                track_number=t.track_number, address=t.aircraft_address,
            ))

    return FrameBundle(
        polar=polar,
        plots=plots,
        scan_index=scan_index,
        n_az=n_az,
        n_range=n_range,
        max_range_m=max_range_m,
        cell_size_m=cell_size_m,
    )
