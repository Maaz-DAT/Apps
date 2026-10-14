const $ = (id) => document.getElementById(id);

async function post(url) {
  const r = await fetch(url, { method: "POST" });
  const result = await r.json();
  if (!r.ok || result.ok === false) throw new Error(result.error || "Request failed");
  return result;
}

function startGen() {
  const rate = $("rate-input").value || 1000;
  post(`/api/generator/start?rate_hz=${rate}`);
}
function stopGen() { post("/api/generator/stop"); }
async function startRec() {
  try {
    const name = encodeURIComponent($("record-name").value.trim());
    await post(`/api/record/start?name=${name}`);
    $("recording-message").textContent = "Recording...";
  } catch (error) {
    $("recording-message").textContent = error.message;
  }
}
async function stopRec() {
  try {
    const result = await post("/api/record/stop");
    $("recording-message").textContent = `Saved ${result.path.split(/[\\/]/).pop()} (${result.frames_written.toLocaleString()} frames)`;
    await refreshRecordings();
  } catch (error) {
    $("recording-message").textContent = error.message;
  }
}
async function refreshRecordings() {
  const select = $("recording-select");
  const selected = select.value;
  try {
    const response = await fetch("/api/recordings");
    const data = await response.json();
    select.replaceChildren();
    if (!data.recordings.length) {
      select.add(new Option("No saved recordings", ""));
      return;
    }
    for (const recording of data.recordings) {
      const size = recording.size_bytes.toLocaleString();
      select.add(new Option(`${recording.name} (${size} bytes)`, recording.name));
    }
    if (data.recordings.some((recording) => recording.name === selected)) {
      select.value = selected;
    }
  } catch (error) {
    select.replaceChildren(new Option("Could not load recordings", ""));
    $("recording-message").textContent = error.message;
  }
}
async function startReplay() {
  const speed = $("speed-input").value || 1;
  const filename = $("recording-select").value;
  if (!filename) {
    $("recording-message").textContent = "Select a saved recording first";
    return;
  }
  try {
    const params = new URLSearchParams({ speed, filename });
    const result = await post(`/api/replay/start?${params}`);
    $("recording-message").textContent = `Replaying ${result.filename}`;
  } catch (error) {
    $("recording-message").textContent = error.message;
  }
}
function stopReplay() { post("/api/replay/stop"); }

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => { $("conn").textContent = "live"; };
  ws.onclose = () => { $("conn").textContent = "disconnected — retrying…"; setTimeout(connect, 1000); };

  ws.onmessage = (ev) => {
    const d = JSON.parse(ev.data);
    $("rate").textContent = d.rate.toLocaleString();
    $("total").textContent = d.total_published.toLocaleString();
    $("reorder").textContent = d.reorder_count.toLocaleString();
    $("recframes").textContent = d.recorder_frames.toLocaleString();

    $("led-gen").classList.toggle("on", d.state.generator_running);
    $("gen-label").textContent = d.state.generator_running
      ? `running @ ${d.state.generator_rate_hz} Hz target` : "stopped";

    $("led-rec").classList.toggle("on", d.state.recording);
    $("rec-label").textContent = d.state.recording ? "recording" : "idle";

    $("led-replay").classList.toggle("on", d.state.replaying);
    $("replay-label").textContent = d.state.replaying ? "replaying" : "idle";

    const log = $("log");
    log.innerHTML = d.recent.slice().reverse().map(f =>
      `<div><span class="topic">${f.topic}</span> seq=${f.seq} t_ns=${f.t_ns} bytes=${f.bytes}</div>`
    ).join("");
  };
}
refreshRecordings();
connect();
