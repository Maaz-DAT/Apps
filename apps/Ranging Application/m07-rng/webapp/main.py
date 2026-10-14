from __future__ import annotations
import asyncio
import threading
import time
import pathlib
import sys

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # m07-rng/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))     # m02-bus-console/ (busplatform, common)

from busplatform.bus import MessageBus
from common.messages import IQBlock

from rng.scenario import ScenarioConfig, Target, generate_iqblock
from rng.pipeline import RngPipeline, PipelineConfig

app = FastAPI()
STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

bus = MessageBus()

# ---------------------------------------------------------------------------
# Shared mutable state (guarded informally — single-writer generator thread,
# readers are the websocket loop and the control endpoints; fine for a demo
# console, matches the pattern already used by the m02 webapp).
# ---------------------------------------------------------------------------

state = {
    "input_path": "raw_iq",          # "raw_iq" | "video"
    "generator_running": False,
    "rate_hz": 5.0,
    "waveform_kind": "chirp",
    "bandwidth_hz": 20e6,
    "duration_s": 2e-6,
    "sample_rate_hz": 50e6,
    "centre_frequency_hz": 9.4e9,
    "n_samples": 8192,
    "snr_db": 15.0,
    "cfar_kind": "ca",
    "guard": 2,
    "train": 12,
    "pfa": 1e-3,
    "k_rank": 0.75,
    "targets": [{"range_m": 15000.0, "velocity_mps": 50.0, "rcs_db": 0.0}],
}

pipeline = RngPipeline(PipelineConfig(
    waveform_kind=state["waveform_kind"], bandwidth_hz=state["bandwidth_hz"],
    duration_s=state["duration_s"], sample_rate_hz=state["sample_rate_hz"],
    cfar_kind=state["cfar_kind"], guard=state["guard"], train=state["train"],
    pfa=state["pfa"], k_rank=state["k_rank"],
))

_gen_stop = threading.Event()
_gen_thread: threading.Thread | None = None
_lock = threading.Lock()

_latest = {
    "ranges_m": [],
    "amplitude_db": [],
    "threshold_db": [],
    "detections": [],   # list of {range_m, amplitude_db, range_rate_mps, timestamp_ns}
    "detection_table": [],  # bounded history for the table
    "block_seq": 0,
}
_DETECTION_TABLE_MAX = 25


def _current_cfg() -> ScenarioConfig:
    targets = [Target(**t) for t in state["targets"]]
    return ScenarioConfig(
        waveform_kind=state["waveform_kind"],
        bandwidth_hz=state["bandwidth_hz"],
        duration_s=state["duration_s"],
        sample_rate_hz=state["sample_rate_hz"],
        centre_frequency_hz=state["centre_frequency_hz"],
        n_samples=state["n_samples"],
        snr_db=state["snr_db"],
        targets=targets,
    )


def _sync_pipeline_cfg():
    pipeline.cfg.waveform_kind = state["waveform_kind"]
    pipeline.cfg.bandwidth_hz = state["bandwidth_hz"]
    pipeline.cfg.duration_s = state["duration_s"]
    pipeline.cfg.sample_rate_hz = state["sample_rate_hz"]
    pipeline.cfg.cfar_kind = state["cfar_kind"]
    pipeline.cfg.guard = state["guard"]
    pipeline.cfg.train = state["train"]
    pipeline.cfg.pfa = state["pfa"]
    pipeline.cfg.k_rank = state["k_rank"]


def _on_iqblock(block: IQBlock):
    _sync_pipeline_cfg()
    with _lock:
        dets = pipeline.process_raw_iq(block.samples, block.sample_rate_hz, block.t_ns)
        _latest["ranges_m"] = pipeline.last_ranges_m.tolist() if pipeline.last_ranges_m is not None else []
        _latest["amplitude_db"] = pipeline.last_profile_db.tolist() if pipeline.last_profile_db is not None else []
        if pipeline.last_threshold_db is not None:
            _latest["threshold_db"] = [
                (None if np.isnan(v) else float(v)) for v in pipeline.last_threshold_db
            ]
        else:
            _latest["threshold_db"] = []
        _latest["detections"] = [
            {"range_m": d.range_m, "amplitude_db": d.amplitude_db, "range_rate_mps": d.range_rate_mps,
             "timestamp_ns": d.timestamp_ns}
            for d in dets
        ]
        for d in dets:
            _latest["detection_table"].append({
                "range_m": round(d.range_m, 1),
                "amplitude_db": round(d.amplitude_db, 1),
                "range_rate_mps": (round(d.range_rate_mps, 1) if d.range_rate_mps is not None else None),
                "timestamp_ns": d.timestamp_ns,
            })
        if len(_latest["detection_table"]) > _DETECTION_TABLE_MAX:
            del _latest["detection_table"][: len(_latest["detection_table"]) - _DETECTION_TABLE_MAX]
        _latest["block_seq"] += 1


def _on_bus_frame(frame):
    """Bus subscribers receive a raw Frame (topic/t_ns/seq/payload bytes);
    decode it back into an IQBlock before handing it to the pipeline."""
    block = IQBlock.from_bytes(frame.payload)
    _on_iqblock(block)


bus.subscribe("iqblock", _on_bus_frame)


def _generator_loop():
    seq = 0
    while not _gen_stop.is_set():
        cfg = _current_cfg()
        t_ns = time.time_ns()
        block = generate_iqblock(cfg, t_ns=t_ns, source_id="rng-webapp", seed=None)
        bus.publish("iqblock", block.to_bytes(), t_ns=t_ns)  # triggers _on_bus_frame -> _on_iqblock
        seq += 1
        interval = 1.0 / max(state["rate_hz"], 0.1)
        _gen_stop.wait(interval)


@app.post("/api/generator/start")
def start_generator():
    global _gen_thread
    if not state["generator_running"]:
        _gen_stop.clear()
        _gen_thread = threading.Thread(target=_generator_loop, daemon=True)
        _gen_thread.start()
        state["generator_running"] = True
    return {"ok": True, "state": state}


@app.post("/api/generator/stop")
def stop_generator():
    _gen_stop.set()
    state["generator_running"] = False
    return {"ok": True, "state": state}


@app.post("/api/config")
def update_config(payload: dict):
    """Partial update of waveform / CFAR / target parameters."""
    allowed_scalar = {
        "rate_hz", "waveform_kind", "bandwidth_hz", "duration_s", "sample_rate_hz",
        "centre_frequency_hz", "n_samples", "snr_db", "cfar_kind", "guard", "train",
        "pfa", "k_rank", "input_path",
    }
    for k, v in payload.items():
        if k in allowed_scalar:
            state[k] = v
    if "targets" in payload:
        state["targets"] = payload["targets"]
    _sync_pipeline_cfg()
    return {"ok": True, "state": state}


@app.get("/api/config")
def get_config():
    return {"state": state}


@app.get("/", response_class=HTMLResponse)
def index():
    return (pathlib.Path(__file__).parent / "templates" / "index.html").read_text()


@app.websocket("/ws")
async def ws_ascope(websocket: WebSocket):
    await websocket.accept()
    last_seq_sent = -1
    try:
        while True:
            await asyncio.sleep(0.1)
            with _lock:
                if _latest["block_seq"] == last_seq_sent:
                    continue
                last_seq_sent = _latest["block_seq"]
                payload = {
                    "ranges_m": _latest["ranges_m"],
                    "amplitude_db": _latest["amplitude_db"],
                    "threshold_db": _latest["threshold_db"],
                    "detections": _latest["detections"],
                    "detection_table": _latest["detection_table"],
                    "block_seq": _latest["block_seq"],
                    "bus_published": bus.stats["published"],
                    "input_count": pipeline.input_count,
                    "processed_count": pipeline.processed_count,
                    "dropped_count": pipeline.dropped_count,
                    "state": state,
                }
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
