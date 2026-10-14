const canvas = document.getElementById("ascope");
const ctx = canvas.getContext("2d");
let cssW = 0, cssH = 0; // canvas size in CSS pixels (post devicePixelRatio scaling)

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  cssW = Math.max(1, Math.round(rect.width));
  cssH = Math.max(1, Math.round(rect.height));
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS-pixel coordinates
  if (window.__lastAscopeMsg) drawAscope(window.__lastAscopeMsg);
}
window.addEventListener("resize", resizeCanvas);

let targets = [{ range_m: 15000, velocity_mps: 50, rcs_db: 0 }];

function renderTargets() {
  const el = document.getElementById("targets-list");
  el.innerHTML = "";
  targets.forEach((t, i) => {
    const row = document.createElement("div");
    row.className = "target-row";
    row.innerHTML = `
      <input type="number" step="100" value="${t.range_m}" title="range (m)" data-i="${i}" data-f="range_m" />
      <input type="number" step="5" value="${t.velocity_mps}" title="velocity (m/s)" data-i="${i}" data-f="velocity_mps" />
      <input type="number" step="1" value="${t.rcs_db}" title="RCS (dB)" data-i="${i}" data-f="rcs_db" />
      <button data-remove="${i}">×</button>
    `;
    el.appendChild(row);
  });
  el.querySelectorAll("input").forEach((inp) => {
    inp.addEventListener("change", (e) => {
      const i = +e.target.dataset.i, f = e.target.dataset.f;
      targets[i][f] = parseFloat(e.target.value);
    });
  });
  el.querySelectorAll("button[data-remove]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      targets.splice(+e.target.dataset.remove, 1);
      renderTargets();
    });
  });
}
renderTargets();

document.getElementById("add-target").addEventListener("click", () => {
  targets.push({ range_m: 10000, velocity_mps: 0, rcs_db: 0 });
  renderTargets();
});

function currentConfig() {
  return {
    waveform_kind: document.getElementById("waveform_kind").value,
    bandwidth_hz: parseFloat(document.getElementById("bandwidth_hz").value) * 1e6,
    duration_s: parseFloat(document.getElementById("duration_s").value) * 1e-6,
    sample_rate_hz: parseFloat(document.getElementById("sample_rate_hz").value) * 1e6,
    n_samples: parseInt(document.getElementById("n_samples").value, 10),
    snr_db: parseFloat(document.getElementById("snr_db").value),
    rate_hz: parseFloat(document.getElementById("rate_hz").value),
    cfar_kind: document.getElementById("cfar_kind").value,
    guard: parseInt(document.getElementById("guard").value, 10),
    train: parseInt(document.getElementById("train").value, 10),
    pfa: parseFloat(document.getElementById("pfa").value),
    k_rank: parseFloat(document.getElementById("k_rank").value),
    targets: targets,
  };
}

document.getElementById("apply").addEventListener("click", async () => {
  await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentConfig()),
  });
});

let generatorOn = false;
document.getElementById("gen-toggle").addEventListener("click", async () => {
  await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentConfig()),
  });
  const ep = generatorOn ? "/api/generator/stop" : "/api/generator/start";
  await fetch(ep, { method: "POST" });
  generatorOn = !generatorOn;
  const btn = document.getElementById("gen-toggle");
  const pill = document.getElementById("status-gen");
  btn.textContent = generatorOn ? "Stop generator" : "Start generator";
  pill.textContent = generatorOn ? "generator: on" : "generator: off";
  pill.className = "pill " + (generatorOn ? "on" : "off");
});

const PLOT_LEFT = 42, PLOT_RIGHT = 14, PLOT_TOP = 10, PLOT_BOTTOM = 26;

/**
 * Aggregate `values` (one per range cell) into one {min,max} pair per
 * output pixel column. When many range cells land in the same screen
 * column (common once a profile has thousands of cells), drawing a
 * single vertical segment spanning that column's min/max instead of
 * connecting every sample point-to-point avoids the dense jagged
 * "comb" artifact and gives a clean oscilloscope-style trace.
 */
function aggregateToColumns(ranges, values, xMax, plotW) {
  const cols = new Array(plotW);
  for (let i = 0; i < ranges.length; i++) {
    const v = values[i];
    if (v === null || v === undefined || !isFinite(v)) continue;
    let col = Math.floor((ranges[i] / xMax) * (plotW - 1));
    if (col < 0) col = 0;
    if (col >= plotW) col = plotW - 1;
    const c = cols[col];
    if (c === undefined) cols[col] = [v, v];
    else { if (v < c[0]) c[0] = v; if (v > c[1]) c[1] = v; }
  }
  return cols;
}

function drawAscope(msg) {
  window.__lastAscopeMsg = msg;
  if (!cssW) resizeCanvas();
  const w = cssW, h = cssH;
  ctx.clearRect(0, 0, w, h);
  const ranges = msg.ranges_m, amp = msg.amplitude_db, thr = msg.threshold_db;
  if (!ranges || ranges.length === 0) return;

  const finiteAmp = amp.filter((v) => isFinite(v));
  const finiteThr = thr.filter((v) => v !== null && isFinite(v));
  const allVals = finiteAmp.concat(finiteThr);
  const yMax = Math.max(...allVals, 0) + 5;
  const yMin = Math.min(...allVals, -40) - 5;
  const xMax = ranges[ranges.length - 1];

  const plotW = Math.max(1, w - PLOT_LEFT - PLOT_RIGHT);
  const plotH = Math.max(1, h - PLOT_TOP - PLOT_BOTTOM);
  const xPix = (col) => PLOT_LEFT + col;
  const xPixRange = (r) => PLOT_LEFT + (r / xMax) * plotW;
  const yPix = (db) => PLOT_TOP + plotH - ((db - yMin) / (yMax - yMin)) * plotH;

  // axes
  ctx.strokeStyle = "#223";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PLOT_LEFT, PLOT_TOP); ctx.lineTo(PLOT_LEFT, PLOT_TOP + plotH); ctx.lineTo(PLOT_LEFT + plotW, PLOT_TOP + plotH);
  ctx.stroke();
  ctx.fillStyle = "#7f93a6";
  ctx.font = "11px sans-serif";
  for (let i = 0; i <= 5; i++) {
    const r = (xMax / 5) * i;
    ctx.fillText((r / 1000).toFixed(1) + "k", xPixRange(r) - 12, h - 8);
  }
  ctx.fillText("range (m)", PLOT_LEFT + plotW - 60, h - 8);
  ctx.save();
  ctx.translate(10, PLOT_TOP + 14);
  ctx.fillText("dB", 0, 0);
  ctx.restore();

  // amplitude trace — one filled min/max segment per pixel column
  const ampCols = aggregateToColumns(ranges, amp, xMax, plotW);
  ctx.strokeStyle = "#4fd1c5";
  ctx.fillStyle = "rgba(79, 209, 197, 0.25)";
  ctx.lineWidth = 1;
  const baseline = PLOT_TOP + plotH;
  for (let col = 0; col < plotW; col++) {
    const c = ampCols[col];
    if (!c) continue;
    const yTop = yPix(c[1]);
    const yBot = Math.min(yPix(c[0]), baseline);
    ctx.fillRect(xPix(col), yTop, 1, Math.max(1, yBot - yTop));
  }
  // thin outline along the column maxima for a crisp top edge
  ctx.beginPath();
  let started = false;
  for (let col = 0; col < plotW; col++) {
    const c = ampCols[col];
    if (!c) { started = false; continue; }
    const x = xPix(col), y = yPix(c[1]);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // threshold trace (smoother/slower-varying — a simple per-column max is enough)
  const thrCols = aggregateToColumns(ranges, thr, xMax, plotW);
  ctx.strokeStyle = "#f6ad55";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  started = false;
  for (let col = 0; col < plotW; col++) {
    const c = thrCols[col];
    if (!c) { started = false; continue; }
    const x = xPix(col), y = yPix(c[1]);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // detections
  ctx.fillStyle = "#fc8181";
  (msg.detections || []).forEach((d) => {
    const x = xPixRange(d.range_m), y = yPix(d.amplitude_db);
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function updateTable(rows) {
  const tbody = document.querySelector("#det-table tbody");
  tbody.innerHTML = "";
  (rows || []).slice().reverse().forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.range_m.toFixed(1)}</td><td>${r.amplitude_db.toFixed(1)}</td>` +
      `<td>${r.range_rate_mps !== null ? r.range_rate_mps.toFixed(1) : "–"}</td><td>${r.timestamp_ns}</td>`;
    tbody.appendChild(tr);
  });
}

resizeCanvas();

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    drawAscope(msg);
    updateTable(msg.detection_table);
    document.getElementById("status-input").textContent = `input: ${msg.input_count}`;
    document.getElementById("status-processed").textContent = `processed: ${msg.processed_count}`;
    document.getElementById("status-dropped").textContent = `dropped: ${msg.dropped_count}`;
  };
  ws.onclose = () => setTimeout(connect, 1000);
}
connect();
