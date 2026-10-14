#!/usr/bin/env python3
"""
SURV — Surveillance Display
CAT 240 video + CAT 048 plots, with optional UDP transmission
on separate ports (compatible with tools such as SPx / RadarView-240).

Run:  python3 server.py
      http://localhost:8000
"""
from __future__ import annotations

import base64
import json
import math
import socket
import sys
import time
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from surv.pipeline import build_scan, FrameBundle, PlotSymbol
from surv.scenario import make_scenario_20, ScenarioTarget, target_at_time
from surv.lut import build_scan_lut, scan_convert, ScanLUT
from surv.cat048 import (
    TruthTarget, generate as gen048, encode_block as enc048_block,
)
from surv.cat240 import (
    TruthVideoTarget, generate_sweep, encode_record as enc240,
)

HOST, PORT = "0.0.0.0", 8000


class SURVState:
    def __init__(self):
        self.targets = make_scenario_20(seed=42)
        self.scan_index = 0
        self.n_az = 180
        self.n_range = 512
        self.max_range_m = 100_000.0
        self.cell_size_m = self.max_range_m / self.n_range
        self.scan_period_s = 4.0
        self.noise_sigma = 6.0
        self.clutter_range_m = 4000.0
        self.gain = 1.0
        self.clutter_threshold = 0.0
        self.plot_fade_scans = 8
        self.decay = 0.88
        self.ppi_size = 720
        self.dropped = 0
        self.seed = 42
        self.use_codec = True
        self.persist = np.zeros((self.ppi_size, self.ppi_size), dtype=np.float32)
        self.lut = build_scan_lut(
            size=self.ppi_size, n_az=self.n_az, n_range=self.n_range,
            max_range_m=self.max_range_m,
        )
        self.plot_history: list[PlotSymbol] = []
        self.track_trails: dict[int, list[tuple[float, float]]] = {}
        self.last_frame: FrameBundle | None = None

        # UDP transmission (separate ports for CAT 240 and CAT 048)
        self.udp_enabled_240 = False
        self.udp_enabled_048 = False
        self.udp_host = "127.0.0.1"
        self.udp_port_240 = 8600
        self.udp_port_048 = 8601
        self.udp_bytes_240 = 0
        self.udp_bytes_048 = 0
        self.udp_pkts_240 = 0
        self.udp_pkts_048 = 0
        self.udp_errors = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def reset(self, seed=42):
        self.seed = seed
        self.targets = make_scenario_20(seed=seed)
        self.scan_index = 0
        self.dropped = 0
        self.persist[:] = 0
        self.plot_history.clear()
        self.track_trails.clear()
        self.last_frame = None
        self.udp_bytes_240 = self.udp_bytes_048 = 0
        self.udp_pkts_240 = self.udp_pkts_048 = 0
        self.udp_errors = 0

    def rebuild_lut(self):
        self.cell_size_m = self.max_range_m / self.n_range
        self.lut = build_scan_lut(
            size=self.ppi_size, n_az=self.n_az, n_range=self.n_range,
            max_range_m=self.max_range_m,
        )
        self.persist = np.zeros((self.ppi_size, self.ppi_size), dtype=np.float32)

    def send_udp(self, data: bytes, port: int) -> bool:
        try:
            self._sock.sendto(data, (self.udp_host, port))
            return True
        except OSError:
            self.udp_errors += 1
            return False


state = SURVState()


def _gain_thr(polar: np.ndarray) -> np.ndarray:
    out = polar.astype(np.float32) * state.gain
    if state.clutter_threshold > 0:
        thr = state.clutter_threshold * 255.0
        out = np.where(out < thr, 0.0, out)
    return np.clip(out, 0, 255)


def _encode_and_send_asterix(targets, scan_index: int, n_az: int, n_range: int,
                             max_range_m: float, cell_size_m: float) -> dict:
    """Build CAT 048 block + CAT 240 radials and optionally UDP-transmit them."""
    t_s = scan_index * state.scan_period_s
    tod_ns = int(t_s * 1e9)
    stats = {"cat048_bytes": 0, "cat240_bytes": 0, "cat240_radials": 0}

    # --- CAT 048 plots ---
    truths = []
    for tgt in targets:
        r, az = target_at_time(tgt, t_s)
        truths.append(TruthTarget(
            range_m=r, az_rad=az,
            speed_m_s=tgt.speed_m_s, heading_rad=tgt.heading_rad,
            mode3a=tgt.mode3a, flight_level_ft=tgt.fl_ft,
            track_number=tgt.track_id, aircraft_address=tgt.address,
            detection_prob=1.0,
        ))
    reports = gen048(truths, sac=0x50, sic=0x01, time_of_day_ns=tod_ns,
                     range_noise_m=0.0, az_noise_rad=0.0, seed=state.seed)
    raw048 = enc048_block(reports)
    stats["cat048_bytes"] = len(raw048)
    if state.udp_enabled_048:
        # Split into MTU-safe chunks if needed (UDP practical limit ~1400)
        mtu = 1400
        for i in range(0, len(raw048), mtu):
            chunk = raw048[i:i + mtu]
            if state.send_udp(chunk, state.udp_port_048):
                state.udp_pkts_048 += 1
                state.udp_bytes_048 += len(chunk)

    # --- CAT 240 video radials ---
    video_targets = [
        TruthVideoTarget(range_m=r, az_rad=az, amplitude=tgt.amplitude, width_m=80.0)
        for tgt, (r, az) in [(t, target_at_time(t, t_s)) for t in targets]
    ]
    total_240 = 0
    n_sent = 0
    for i in range(n_az):
        az = 2 * math.pi * i / n_az
        sw = generate_sweep(
            az_rad=az,
            az_width_rad=2 * math.pi / n_az,
            n_cells=n_range,
            cell_size_m=cell_size_m,
            start_range_m=0.0,
            targets=video_targets,
            noise_sigma=state.noise_sigma,
            clutter_range_m=state.clutter_range_m,
            resolution_bits=8,
            sac=0x50, sic=0x01,
            msg_index=scan_index * n_az + i,
            time_of_day_ns=tod_ns,
            seed=state.seed + scan_index * n_az + i,
        )
        raw240 = enc240(sw)
        total_240 += len(raw240)
        n_sent += 1
        if state.udp_enabled_240:
            mtu = 1400
            for j in range(0, len(raw240), mtu):
                chunk = raw240[j:j + mtu]
                if state.send_udp(chunk, state.udp_port_240):
                    state.udp_pkts_240 += 1
                    state.udp_bytes_240 += len(chunk)
    stats["cat240_bytes"] = total_240
    stats["cat240_radials"] = n_sent
    return stats


def do_scan() -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        bundle = build_scan(
            state.targets,
            scan_index=state.scan_index,
            n_az=state.n_az,
            n_range=state.n_range,
            max_range_m=state.max_range_m,
            cell_size_m=state.cell_size_m,
            scan_period_s=state.scan_period_s,
            noise_sigma=state.noise_sigma,
            clutter_range_m=state.clutter_range_m,
            seed=state.seed + state.scan_index,
            use_codec=state.use_codec,
        )
    except Exception as e:
        state.dropped += 1
        traceback.print_exc()
        return {"error": str(e), "dropped": state.dropped}

    # Optional UDP TX of ASTERIX (separate from display pipeline)
    tx_stats = {"cat048_bytes": 0, "cat240_bytes": 0, "cat240_radials": 0}
    if state.udp_enabled_240 or state.udp_enabled_048:
        try:
            tx_stats = _encode_and_send_asterix(
                state.targets, state.scan_index,
                state.n_az, state.n_range, state.max_range_m, state.cell_size_m,
            )
        except Exception:
            state.udp_errors += 1
            traceback.print_exc()

    polar = _gain_thr(bundle.polar)
    cartesian = scan_convert(polar, state.lut)
    state.persist *= state.decay
    state.persist = np.maximum(state.persist, cartesian)
    display = state.persist.copy()

    for p in state.plot_history:
        p.age_scans += 1
    state.plot_history = [p for p in state.plot_history if p.age_scans < state.plot_fade_scans]
    for p in bundle.plots:
        p.age_scans = 0
        state.plot_history.append(p)
        if p.track_number is not None:
            x = p.range_m * math.sin(p.az_rad)
            y = p.range_m * math.cos(p.az_rad)
            trail = state.track_trails.setdefault(p.track_number, [])
            trail.append((x, y))
            if len(trail) > 60:
                del trail[:-60]

    aged = FrameBundle(
        polar=bundle.polar, plots=list(state.plot_history),
        scan_index=bundle.scan_index, n_az=bundle.n_az, n_range=bundle.n_range,
        max_range_m=bundle.max_range_m, cell_size_m=bundle.cell_size_m,
        dropped=state.dropped,
    )
    state.last_frame = aged
    state.scan_index += 1

    cart_u8 = np.clip(display, 0, 255).astype(np.uint8)
    b64 = base64.b64encode(cart_u8.tobytes()).decode("ascii")

    plots = []
    for p in aged.plots:
        plots.append({
            "range_m": round(p.range_m, 1),
            "az_rad": round(p.az_rad, 5),
            "az_deg": round(math.degrees(p.az_rad), 2),
            "mode3a": f"{p.mode3a:04o}" if p.mode3a is not None else None,
            "fl_ft": p.fl_ft,
            "track_number": p.track_number,
            "address": f"{p.address:06X}" if p.address is not None else None,
            "age_scans": p.age_scans,
            "fields": {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in (p.fields or {}).items()
            },
        })
    trails = {
        str(tid): [{"x": round(x, 1), "y": round(y, 1)} for x, y in pts[-40:]]
        for tid, pts in state.track_trails.items()
    }

    return {
        "scan_index": aged.scan_index,
        "n_az": aged.n_az,
        "n_range": aged.n_range,
        "max_range_m": aged.max_range_m,
        "cell_size_m": round(aged.cell_size_m, 2),
        "dropped": state.dropped,
        "ppi_size": state.ppi_size,
        "image_b64": b64,
        "plots": plots,
        "trails": trails,
        "n_plots": len(plots),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "udp": {
            "host": state.udp_host,
            "port_240": state.udp_port_240,
            "port_048": state.udp_port_048,
            "enabled_240": state.udp_enabled_240,
            "enabled_048": state.udp_enabled_048,
            "bytes_240": state.udp_bytes_240,
            "bytes_048": state.udp_bytes_048,
            "pkts_240": state.udp_pkts_240,
            "pkts_048": state.udp_pkts_048,
            "errors": state.udp_errors,
            "last_tx": tx_stats,
        },
        "timestamp": time.time(),
    }


def status_dict():
    return {
        "scan_index": state.scan_index,
        "n_az": state.n_az,
        "n_range": state.n_range,
        "max_range_m": state.max_range_m,
        "gain": state.gain,
        "clutter_threshold": state.clutter_threshold,
        "plot_fade_scans": state.plot_fade_scans,
        "decay": state.decay,
        "dropped": state.dropped,
        "use_codec": state.use_codec,
        "n_targets": len(state.targets),
        "ppi_size": state.ppi_size,
        "udp": {
            "host": state.udp_host,
            "port_240": state.udp_port_240,
            "port_048": state.udp_port_048,
            "enabled_240": state.udp_enabled_240,
            "enabled_048": state.udp_enabled_048,
            "bytes_240": state.udp_bytes_240,
            "bytes_048": state.udp_bytes_048,
            "pkts_240": state.udp_pkts_240,
            "pkts_048": state.udp_pkts_048,
            "errors": state.udp_errors,
        },
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, fmt, *args):
        if "/api/" in str(args[0] if args else ""):
            return
        super().log_message(fmt, *args)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode())
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            data = (ROOT / "templates" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.startswith("/static/"):
            f = ROOT / "static" / path[len("/static/"):]
            if f.is_file():
                data = f.read_bytes()
                ctype = "text/css" if f.suffix == ".css" else (
                    "application/javascript" if f.suffix == ".js" else "application/octet-stream"
                )
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)
            return
        if path == "/api/status":
            self._json(200, status_dict())
            return
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)
        if path == "/api/scan":
            self._json(200, do_scan())
            return
        if path == "/api/reset":
            state.reset(seed=int(qs.get("seed", ["42"])[0]))
            self._json(200, {"ok": True, "scan_index": 0})
            return
        if path == "/api/config":
            body = self._read_json()
            if "gain" in body:
                state.gain = float(np.clip(body["gain"], 0.1, 4.0))
            if "clutter_threshold" in body:
                state.clutter_threshold = float(np.clip(body["clutter_threshold"], 0.0, 1.0))
            if "plot_fade_scans" in body:
                state.plot_fade_scans = int(np.clip(body["plot_fade_scans"], 1, 30))
            if "decay" in body:
                state.decay = float(np.clip(body["decay"], 0.5, 0.99))
            if "use_codec" in body:
                state.use_codec = bool(body["use_codec"])
            if "n_az" in body:
                state.n_az = int(np.clip(body["n_az"], 36, 720))
                state.rebuild_lut()
            if "n_range" in body:
                state.n_range = int(np.clip(body["n_range"], 64, 2048))
                state.rebuild_lut()
            if "max_range_m" in body:
                state.max_range_m = float(np.clip(body["max_range_m"], 10_000, 400_000))
                state.rebuild_lut()
            # UDP controls
            if "udp_host" in body:
                state.udp_host = str(body["udp_host"]).strip() or "127.0.0.1"
            if "udp_port_240" in body:
                state.udp_port_240 = int(np.clip(int(body["udp_port_240"]), 1024, 65535))
            if "udp_port_048" in body:
                state.udp_port_048 = int(np.clip(int(body["udp_port_048"]), 1024, 65535))
            if "udp_enabled_240" in body:
                state.udp_enabled_240 = bool(body["udp_enabled_240"])
            if "udp_enabled_048" in body:
                state.udp_enabled_048 = bool(body["udp_enabled_048"])
            self._json(200, status_dict())
            return
        self.send_error(404)


if __name__ == "__main__":
    print(f"SURV  →  http://localhost:{PORT}")
    print(f"  CAT 240 UDP default port {state.udp_port_240}")
    print(f"  CAT 048 UDP default port {state.udp_port_048}")
    HTTPServer((HOST, PORT), Handler).serve_forever()
