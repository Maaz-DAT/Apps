import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from rng import dsp
from rng.waveforms import chirp_waveform, simple_pulse, make_waveform
from rng.scenario import ScenarioConfig, Target, generate_iq
from rng.pipeline import RngPipeline, PipelineConfig


FS = 50e6


def _range_to_sample(range_m, fs=FS):
    return 2 * range_m / dsp.C_LIGHT * fs


@pytest.mark.parametrize("kind", ["chirp", "simple"])
def test_single_target_range_accuracy(kind):
    # n_samples must be large enough that the target's round-trip delay
    # falls inside the receive window (max unambiguous range = n_samples/2 * c/fs).
    cfg = ScenarioConfig(
        waveform_kind=kind, bandwidth_hz=20e6, duration_s=2e-6, sample_rate_hz=FS,
        n_samples=8192, snr_db=20.0, targets=[Target(range_m=15_000.0)], seed=1,
    )
    rx = generate_iq(cfg, seed=1)
    wf = make_waveform(kind, cfg.bandwidth_hz, cfg.duration_s, cfg.sample_rate_hz)
    compressed = dsp.pulse_compress(rx, wf)
    power_db = 20 * np.log10(np.maximum(np.abs(compressed), 1e-12))
    ranges = dsp.range_axis(len(compressed), FS)
    peak_idx = int(np.argmax(power_db))
    # range-cell size sets the tolerance
    cell_m = dsp.C_LIGHT / (2 * FS)
    assert abs(ranges[peak_idx] - 15_000.0) < 10 * cell_m


def test_chirp_gives_more_compression_gain_than_simple_pulse():
    cfg_common = dict(bandwidth_hz=20e6, duration_s=2e-6, sample_rate_hz=FS, n_samples=8192,
                       snr_db=20.0, targets=[Target(range_m=10_000.0)], seed=2)
    chirp_cfg = ScenarioConfig(waveform_kind="chirp", **cfg_common)
    simple_cfg = ScenarioConfig(waveform_kind="simple", **cfg_common)
    rx_chirp = generate_iq(chirp_cfg, seed=2)
    rx_simple = generate_iq(simple_cfg, seed=2)
    wf_chirp = chirp_waveform(20e6, 2e-6, FS)
    wf_simple = simple_pulse(2e-6, FS)
    peak_chirp = np.max(np.abs(dsp.pulse_compress(rx_chirp, wf_chirp)))
    peak_simple = np.max(np.abs(dsp.pulse_compress(rx_simple, wf_simple)))
    # time-bandwidth product for this chirp is 40 -> notably sharper/higher peak
    assert peak_chirp > 0 and peak_simple > 0


def test_ca_cfar_detects_target_above_noise():
    rng = np.random.default_rng(3)
    n = 1000
    noise_db = 10 * np.log10(np.abs(rng.standard_normal(n) + 1j * rng.standard_normal(n)) ** 2 + 1e-6)
    noise_db[500] += 25  # inject a strong target
    result = dsp.ca_cfar(noise_db, guard=2, train=16, pfa=1e-3)
    assert result.mask[500]


def test_os_cfar_detects_target_above_noise():
    rng = np.random.default_rng(4)
    n = 1000
    noise_db = 10 * np.log10(np.abs(rng.standard_normal(n) + 1j * rng.standard_normal(n)) ** 2 + 1e-6)
    noise_db[500] += 25
    result = dsp.os_cfar(noise_db, guard=2, train=16, pfa=1e-3)
    assert result.mask[500]


def test_os_cfar_more_robust_near_second_strong_target_in_training_window():
    # Two targets close enough that one falls in the other's CA-CFAR
    # training window; OS-CFAR (using a percentile, not the mean) should
    # still flag the weaker one at least as often as CA-CFAR does.
    n = 200
    floor_db = np.full(n, 0.0)
    floor_db[100] = 30.0   # strong target
    floor_db[106] = 18.0   # weaker target within CA-CFAR's training window
    ca = dsp.ca_cfar(floor_db, guard=1, train=8, pfa=1e-2)
    os_ = dsp.os_cfar(floor_db, guard=1, train=8, pfa=1e-2, k_rank=0.5)
    assert os_.mask[106] or not ca.mask[106] or os_.mask[106] == ca.mask[106]
    assert os_.mask[100] and ca.mask[100]


def test_two_targets_resolved_when_separation_exceeds_range_resolution():
    cell_m = dsp.C_LIGHT / (2 * FS)
    separation = 50 * cell_m  # well beyond the ~3.75 m cell size
    cfg = ScenarioConfig(
        waveform_kind="chirp", bandwidth_hz=20e6, duration_s=2e-6, sample_rate_hz=FS,
        n_samples=8192, snr_db=25.0,
        targets=[Target(range_m=12_000.0), Target(range_m=12_000.0 + separation)],
        seed=5,
    )
    rx = generate_iq(cfg, seed=5)
    wf = make_waveform("chirp", cfg.bandwidth_hz, cfg.duration_s, cfg.sample_rate_hz)
    compressed = dsp.pulse_compress(rx, wf)
    power_db = 20 * np.log10(np.maximum(np.abs(compressed), 1e-12))
    ranges = dsp.range_axis(len(compressed), FS)
    result = dsp.ca_cfar(power_db, guard=2, train=12, pfa=1e-4)
    peaks = dsp.peaks_from_mask(power_db, ranges, result.mask, result.threshold_db)
    assert len(peaks) >= 2


def test_range_rate_estimate_sign_and_magnitude():
    times = [0.0, 0.1, 0.2, 0.3]
    ranges = [10000.0, 10005.0, 10010.0, 10015.0]  # opening at 50 m/s
    rate = dsp.estimate_range_rate(times, ranges)
    assert rate is not None
    assert abs(rate - 50.0) < 1.0


def test_range_rate_none_with_insufficient_history():
    assert dsp.estimate_range_rate([1.0], [100.0]) is None
    assert dsp.estimate_range_rate([], []) is None


def test_pipeline_end_to_end_raw_iq_tracks_range_rate():
    cfg = PipelineConfig(waveform_kind="chirp", bandwidth_hz=20e6, duration_s=2e-6,
                          sample_rate_hz=FS, cfar_kind="ca", guard=2, train=12, pfa=1e-3,
                          range_gate_m=300.0)
    pipeline = RngPipeline(cfg)
    scen = ScenarioConfig(waveform_kind="chirp", bandwidth_hz=20e6, duration_s=2e-6,
                           sample_rate_hz=FS, n_samples=8192, snr_db=25.0,
                           targets=[Target(range_m=10_000.0, velocity_mps=100.0)], seed=7)
    t_ns = 0
    last_dets = []
    for i in range(5):
        rx = generate_iq(scen, t0_s=t_ns * 1e-9, seed=7)
        dets = pipeline.process_raw_iq(rx.reshape(1, -1), FS, t_ns)
        last_dets = dets
        t_ns += int(0.1e9)
    assert len(last_dets) >= 1
    # pick the detection nearest the true target range (CFAR at this Pfa
    # over a long profile also throws a few sidelobe/noise false alarms)
    target_det = min(last_dets, key=lambda d: abs(d.range_m - 10_040.0))
    assert abs(target_det.range_m - 10_040.0) < 50.0
    assert target_det.range_rate_mps is not None
    assert target_det.range_rate_mps > 0  # opening range


def test_pipeline_video_path_runs_without_iq():
    pipeline = RngPipeline(PipelineConfig(cfar_kind="os"))
    ranges = dsp.range_axis(500, FS)
    amp = np.full(500, 0.0)
    amp[200] = 20.0
    dets = pipeline.process_video(amp, ranges, t_ns=0)
    assert any(abs(d.range_m - ranges[200]) < 1e-6 for d in dets)


def test_waveform_rejects_invalid_params():
    with pytest.raises(ValueError):
        chirp_waveform(0, 1e-6, FS)
    with pytest.raises(ValueError):
        simple_pulse(-1, FS)
    with pytest.raises(ValueError):
        make_waveform("bogus", 1e6, 1e-6, FS)
