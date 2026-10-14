from __future__ import annotations
import asyncio
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from busplatform.bus import MessageBus
from busplatform.recorder import Recorder, Replayer
from busplatform.udp_bridge import UdpSender, UdpReceiver
import random
from common.messages import PDW, FLAG_VALID, FLAG_SATURATED

app = FastAPI()
STATIC_DIR = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

bus = MessageBus()
RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"

state = {
    "generator_running": False,
    "generator_rate_hz": 1000,
    "recording": False,
    "replaying": False,
    "udp_tx": False,
    "udp_rx": False,
}
_recent: deque = deque(maxlen=30)
_counters = {"published_window": 0}
_last_seq = {"n": None, "reorder": 0}

recorder: Recorder | None = None
replayer: Replayer | None = None
recording_path: Path | None = None
udp_sender: UdpSender | None = None
udp_receiver: UdpReceiver | None = None

_gen_stop = threading.Event()
_gen_thread: threading.Thread | None = None


def _on_any_frame(frame):
    _counters["published_window"] += 1
    seq = frame.seq
    if _last_seq["n"] is not None and seq < _last_seq["n"]:
        _last_seq["reorder"] += 1
    _last_seq["n"] = seq
    if len(_recent) == _recent.maxlen:
        _recent.popleft()
    _recent.append({"topic": frame.topic, "t_ns": frame.t_ns, "seq": frame.seq,
                     "bytes": len(frame.payload)})


bus.subscribe("*", _on_any_frame)


def _synthetic_pdw(seed: int) -> PDW:
    """Synthetic PDW using the frozen M-00 schema."""
    r = random.Random(seed)
    n_ch = 4
    now = time.time_ns()
    flags = FLAG_VALID
    if r.random() < 0.02:
        flags |= FLAG_SATURATED
    return PDW(
        schema_version=1,
        t_ns=now,
        source_id="synth-gen",
        toa_ns=now,
        frequency_hz=9.4e9 + r.uniform(-2e6, 2e6),
        pulse_width_s=r.uniform(0.5e-6, 2e-6),
        phase=[r.uniform(-np.pi, np.pi) for _ in range(n_ch)],
        amplitude=[r.uniform(0.5, 2.0) for _ in range(n_ch)],
        flags=flags,
    )


def _generator_loop():
    seed = 0
    while not _gen_stop.is_set():
        batch_start = time.perf_counter()
        target_interval = 1.0 / max(state["generator_rate_hz"], 1)
        # tight-loop batches of ~50ms worth of messages, then check stop/rate
        batch_deadline = batch_start + 0.05
        while time.perf_counter() < batch_deadline and not _gen_stop.is_set():
            seed += 1
            p = _synthetic_pdw(seed)
            bus.publish("pdw", p.to_bytes(), t_ns=p.t_ns)
            if target_interval > 0.0005:
                time.sleep(target_interval)


@app.post("/api/generator/start")
def start_generator(rate_hz: int = 1000):
    global _gen_thread
    state["generator_rate_hz"] = max(1, rate_hz)
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


@app.post("/api/generator/rate")
def set_rate(rate_hz: int):
    state["generator_rate_hz"] = max(1, rate_hz)
    return {"ok": True, "state": state}


@app.post("/api/record/start")
def start_record(name: str = ""):
    global recorder, recording_path
    if not state["recording"]:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")
        if not safe_name:
            safe_name = "recording"
        recording_path = RECORDINGS_DIR / f"{safe_name}-{datetime.now():%Y%m%d-%H%M%S-%f}.bin"
        recorder = Recorder(bus, recording_path)
        recorder.start()
        state["recording"] = True
    return {"ok": True, "path": str(recording_path), "state": state}


@app.post("/api/record/stop")
def stop_record():
    global recorder
    if recorder and state["recording"]:
        recorder.stop()
        state["recording"] = False
        frames = recorder.frames_written
        recorder = None
        return {"ok": True, "frames_written": frames, "path": str(recording_path), "state": state}
    return {"ok": False, "state": state}


@app.get("/api/recordings")
def list_recordings():
    if not RECORDINGS_DIR.exists():
        return {"recordings": []}
    files = sorted(RECORDINGS_DIR.glob("*.bin"), key=lambda path: path.stat().st_mtime, reverse=True)
    return {"recordings": [
        {"name": path.name, "size_bytes": path.stat().st_size}
        for path in files if path.is_file()
    ]}


@app.post("/api/replay/start")
def start_replay(speed: float = 1.0, filename: str | None = None):
    global replayer
    if filename is None:
        recordings = sorted(RECORDINGS_DIR.glob("*.bin"), key=lambda path: path.stat().st_mtime, reverse=True) if RECORDINGS_DIR.exists() else []
        recording = recordings[0] if recordings else None
    else:
        recording = RECORDINGS_DIR / filename
        if Path(filename).name != filename or recording.suffix != ".bin":
            recording = None
    if recording is None or not recording.is_file():
        return {"ok": False, "error": "recording not found; choose or create a recording first"}
    if state["replaying"] and replayer and not replayer.is_running:
        state["replaying"] = False
    if not state["replaying"]:
        replayer = Replayer(bus, recording)
        replayer.replay(speed=speed, blocking=False)
        state["replaying"] = replayer.is_running
    return {"ok": True, "filename": recording.name, "state": state}


@app.post("/api/replay/stop")
def stop_replay():
    global replayer
    if replayer:
        replayer.stop()
    state["replaying"] = False
    return {"ok": True, "state": state}


@app.post("/api/udp/tx/start")
def start_udp_tx(host: str = "127.0.0.1", port: int = 9999):
    global udp_sender
    if not state["udp_tx"]:
        udp_sender = UdpSender(bus, (host, port))
        udp_sender.start()
        state["udp_tx"] = True
    return {"ok": True, "state": state}


@app.post("/api/udp/tx/stop")
def stop_udp_tx():
    global udp_sender
    if udp_sender:
        udp_sender.stop()
    state["udp_tx"] = False
    return {"ok": True, "state": state}


@app.post("/api/udp/rx/start")
def start_udp_rx(host: str = "0.0.0.0", port: int = 9999):
    global udp_receiver
    if not state["udp_rx"]:
        udp_receiver = UdpReceiver(bus, (host, port))
        udp_receiver.start()
        state["udp_rx"] = True
    return {"ok": True, "state": state}


@app.post("/api/udp/rx/stop")
def stop_udp_rx():
    global udp_receiver
    if udp_receiver:
        udp_receiver.stop()
    state["udp_rx"] = False
    return {"ok": True, "state": state}


@app.get("/", response_class=HTMLResponse)
def index():
    return (pathlib.Path(__file__).parent / "templates" / "index.html").read_text()


@app.websocket("/ws")
async def ws_stats(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(0.1)
            if state["replaying"] and replayer and not replayer.is_running:
                state["replaying"] = False
            rate = _counters["published_window"] * 10  # 100ms window -> per-second
            _counters["published_window"] = 0
            await websocket.send_json({
                "rate": rate,
                "total_published": bus.stats["published"],
                "reorder_count": _last_seq["reorder"],
                "recent": list(_recent)[-10:],
                "state": state,
                "recorder_frames": recorder.frames_written if recorder else 0,
            })
    except WebSocketDisconnect:
        pass
