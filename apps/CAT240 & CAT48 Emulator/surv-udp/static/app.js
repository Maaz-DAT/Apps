const SIZE = 720;
const canvas = document.getElementById("ppi");
const overlay = document.getElementById("overlay");
const hud = document.getElementById("hud");
const ctx = canvas.getContext("2d");
const octx = overlay.getContext("2d");
const hctx = hud.getContext("2d");
const stage = document.getElementById("ppi-stage");
const area = document.getElementById("ppi-area");

let maxRangeM = 100000;
let currentPlots = [];
let currentTrails = {};
let selectedTrack = null;
let running = false;
let timerId = null;
let lastFrameTime = 0;
let fpsEma = 0;
let imageData = null;
let zoom = 1.0, panX = 0, panY = 0;
let dragging = false, dragStart = null;

function applyTransform() {
  const s = SIZE * zoom;
  stage.style.width = s + "px";
  stage.style.height = s + "px";
  stage.style.transform = `translate(${panX}px, ${panY}px)`;
}

function grayToRGBA(g) {
  const v = Math.min(255, g | 0);
  return [v * 0.2, v * 0.85, v * 0.25, v > 3 ? 255 : 0];
}

function drawCartesian(b64, size) {
  const raw = atob(b64);
  if (!imageData || imageData.width !== SIZE) imageData = ctx.createImageData(SIZE, SIZE);
  const d = imageData.data;
  const scale = SIZE / size;
  for (let row = 0; row < SIZE; row++) {
    const srcRow = Math.min(size - 1, Math.floor(row / scale));
    for (let col = 0; col < SIZE; col++) {
      const srcCol = Math.min(size - 1, Math.floor(col / scale));
      const g = raw.charCodeAt(srcRow * size + srcCol);
      const off = (row * SIZE + col) * 4;
      const [r, gr, b, a] = grayToRGBA(g);
      d[off] = r; d[off+1] = gr; d[off+2] = b; d[off+3] = a;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  drawHUD();
}

function drawHUD() {
  hctx.clearRect(0, 0, SIZE, SIZE);
  if (!document.getElementById("chk-rings").checked) return;
  const cx = SIZE / 2, cy = SIZE / 2, R = SIZE / 2 - 1;
  hctx.strokeStyle = "rgba(120,120,120,0.5)";
  hctx.lineWidth = 1;
  hctx.beginPath(); hctx.arc(cx, cy, R, 0, Math.PI * 2); hctx.stroke();
  hctx.fillStyle = "rgba(140,140,140,0.8)";
  hctx.font = "11px monospace";
  hctx.textAlign = "center";
  const km = maxRangeM / 1000;
  for (const f of [0.2, 0.4, 0.6, 0.8, 1.0]) {
    const r = f * R;
    hctx.beginPath(); hctx.arc(cx, cy, r, 0, Math.PI * 2); hctx.stroke();
    hctx.fillText(`${(f * km).toFixed(0)}`, cx + 4, cy - r + 12);
  }
  for (let deg = 0; deg < 360; deg += 30) {
    const rad = (deg * Math.PI) / 180;
    hctx.beginPath();
    hctx.moveTo(cx + Math.sin(rad) * (R - 6), cy - Math.cos(rad) * (R - 6));
    hctx.lineTo(cx + Math.sin(rad) * R, cy - Math.cos(rad) * R);
    hctx.stroke();
  }
  hctx.beginPath();
  hctx.moveTo(cx, 0); hctx.lineTo(cx, SIZE);
  hctx.moveTo(0, cy); hctx.lineTo(SIZE, cy);
  hctx.stroke();
  hctx.fillStyle = "rgba(180,180,180,0.9)";
  hctx.font = "bold 12px sans-serif";
  hctx.fillText("N", cx, 14);
  hctx.fillText("E", SIZE - 12, cy + 4);
  hctx.fillText("S", cx, SIZE - 6);
  hctx.fillText("W", 12, cy + 4);
}

function polarToPixel(rangeM, azRad) {
  const c = SIZE / 2;
  const r = (rangeM / maxRangeM) * c;
  return [c + r * Math.sin(azRad), c - r * Math.cos(azRad)];
}

function drawOverlay(plots, trails) {
  octx.clearRect(0, 0, SIZE, SIZE);
  currentPlots = plots || [];
  currentTrails = trails || {};
  const showTrails = document.getElementById("chk-trails").checked;
  const showLabels = document.getElementById("chk-labels").checked;
  if (showTrails) {
    octx.lineWidth = 1;
    for (const [, pts] of Object.entries(currentTrails)) {
      if (pts.length < 2) continue;
      octx.strokeStyle = "rgba(180,170,60,0.35)";
      octx.beginPath();
      const c = SIZE / 2, sc = c / maxRangeM;
      pts.forEach((pt, i) => {
        const px = c + pt.x * sc, py = c - pt.y * sc;
        if (i === 0) octx.moveTo(px, py); else octx.lineTo(px, py);
      });
      octx.stroke();
    }
  }
  for (const p of currentPlots) {
    const [px, py] = polarToPixel(p.range_m, p.az_rad);
    const age = p.age_scans || 0;
    const alpha = Math.max(0.3, 1 - age * 0.1);
    const sel = selectedTrack !== null && p.track_number === selectedTrack;
    octx.save();
    octx.globalAlpha = alpha;
    const s = sel ? 7 : 5;
    octx.fillStyle = sel ? "#fff" : "#d4c84a";
    octx.strokeStyle = sel ? "#888" : "rgba(0,0,0,0.4)";
    octx.lineWidth = sel ? 2 : 1;
    octx.beginPath();
    octx.moveTo(px, py - s); octx.lineTo(px + s, py);
    octx.lineTo(px, py + s); octx.lineTo(px - s, py);
    octx.closePath(); octx.fill(); octx.stroke();
    const f = p.fields || {};
    if (f.heading_rad != null) {
      const lead = 12;
      octx.strokeStyle = "rgba(180,170,60,0.6)";
      octx.beginPath();
      octx.moveTo(px, py);
      octx.lineTo(px + Math.sin(f.heading_rad) * lead, py - Math.cos(f.heading_rad) * lead);
      octx.stroke();
    }
    if (showLabels && p.track_number != null) {
      octx.font = "10px monospace";
      octx.fillStyle = sel ? "#ccc" : "rgba(212,200,74,0.9)";
      octx.fillText(String(p.track_number), px + 8, py - 3);
    }
    octx.restore();
  }
}

function updatePlotTable(plots) {
  const tbody = document.getElementById("plot-tbody");
  tbody.innerHTML = "";
  const byTrack = new Map();
  for (const p of plots) {
    const tid = p.track_number ?? -1;
    if (!byTrack.has(tid) || (p.age_scans || 0) < (byTrack.get(tid).age_scans || 0))
      byTrack.set(tid, p);
  }
  document.getElementById("n-plots-label").textContent = byTrack.size;
  [...byTrack.values()].sort((a, b) => (a.track_number||0) - (b.track_number||0)).forEach(p => {
    const tr = document.createElement("tr");
    if (selectedTrack !== null && p.track_number === selectedTrack) tr.classList.add("sel");
    tr.innerHTML = `<td>${p.track_number??"—"}</td><td>${(p.range_m/1000).toFixed(1)}</td>
      <td>${p.az_deg?.toFixed(1)??"—"}</td><td>${p.mode3a??"—"}</td>
      <td>${p.fl_ft??"—"}</td><td>${p.age_scans??0}</td>`;
    tr.onclick = () => selectPlot(p);
    tbody.appendChild(tr);
  });
}

function selectPlot(p) {
  selectedTrack = p.track_number;
  const f = p.fields || {};
  const rows = [
    ["Track", p.track_number], ["Mode-3/A", p.mode3a],
    ["FL (ft)", p.fl_ft ?? f.flight_level_ft],
    ["Range (m)", p.range_m?.toFixed?.(1)], ["Az (°)", p.az_deg?.toFixed?.(2)],
    ["Address", p.address],
    ["GS (m/s)", f.ground_speed_m_s != null ? Number(f.ground_speed_m_s).toFixed(1) : "—"],
    ["Hdg (°)", f.heading_rad != null ? (f.heading_rad * 180 / Math.PI).toFixed(1) : "—"],
    ["SAC/SIC", `${f.sac ?? "—"} / ${f.sic ?? "—"}`],
    ["Age", p.age_scans],
  ];
  document.getElementById("detail-panel").innerHTML = rows.map(([k,v]) =>
    `<div class="dr"><span class="k">${k}</span><span class="val">${v ?? "—"}</span></div>`
  ).join("");
  drawOverlay(currentPlots, currentTrails);
  updatePlotTable(currentPlots);
}

function updateUdpStats(u) {
  if (!u) return;
  const en = [];
  if (u.enabled_240) en.push(`240→${u.host}:${u.port_240}`);
  if (u.enabled_048) en.push(`048→${u.host}:${u.port_048}`);
  document.getElementById("st-udp").textContent = en.length ? "UDP " + en.join(" · ") : "UDP off";
  document.getElementById("st-udp").className = en.length ? "ok" : "";
  document.getElementById("udp-stats").textContent =
    `pkts 240:${u.pkts_240||0} (${u.bytes_240||0} B) · 048:${u.pkts_048||0} (${u.bytes_048||0} B)` +
    (u.errors ? ` · err ${u.errors}` : "");
}

function handleFrame(data) {
  if (data.error) {
    document.getElementById("st-drops").textContent = `Drops: ${data.dropped??"?"}`;
    return;
  }
  maxRangeM = data.max_range_m || maxRangeM;
  drawCartesian(data.image_b64, data.ppi_size || 640);
  drawOverlay(data.plots, data.trails);
  updatePlotTable(data.plots || []);
  updateUdpStats(data.udp);
  document.getElementById("st-scan").textContent = `Scan #${data.scan_index}`;
  document.getElementById("st-cells").textContent =
    `${data.n_az} az · ${data.n_range} cells · ${(maxRangeM/1000).toFixed(0)} km`;
  document.getElementById("st-drops").textContent = `Drops: ${data.dropped}`;
  if (data.elapsed_ms != null)
    document.getElementById("st-elapsed").textContent = `${data.elapsed_ms} ms/scan`;
  const now = performance.now();
  if (lastFrameTime > 0) {
    const fps = 1000 / (now - lastFrameTime);
    fpsEma = fpsEma === 0 ? fps : fpsEma * 0.8 + fps * 0.2;
    document.getElementById("st-fps").textContent = `${fpsEma.toFixed(1)} fps`;
  }
  lastFrameTime = now;
}

async function doScan() {
  handleFrame(await fetch("/api/scan", { method: "POST" }).then(x => x.json()));
}

async function applyConfig(extra = {}) {
  const body = {
    gain: document.getElementById("gain").value / 100,
    clutter_threshold: document.getElementById("thr").value / 100,
    plot_fade_scans: +document.getElementById("fade").value,
    decay: document.getElementById("decay").value / 100,
    use_codec: document.getElementById("use-codec").checked,
    ...extra,
  };
  await fetch("/api/config", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function applyUdp() {
  await applyConfig({
    udp_host: document.getElementById("udp-host").value,
    udp_port_240: +document.getElementById("udp-port-240").value,
    udp_port_048: +document.getElementById("udp-port-048").value,
    udp_enabled_240: document.getElementById("udp-en-240").checked,
    udp_enabled_048: document.getElementById("udp-en-048").checked,
  });
  const st = await fetch("/api/status").then(x => x.json());
  updateUdpStats(st.udp);
}

document.getElementById("btn-scan").onclick = () => doScan();
document.getElementById("btn-run").onclick = function () {
  running = !running;
  this.classList.toggle("active", running);
  this.textContent = running ? "Stop" : "Run";
  if (running) {
    const interval = +document.getElementById("sel-speed").value;
    const tick = async () => {
      if (!running) return;
      await doScan();
      timerId = setTimeout(tick, interval);
    };
    tick();
  } else if (timerId) { clearTimeout(timerId); timerId = null; }
};
document.getElementById("btn-reset").onclick = async () => {
  await fetch("/api/reset?seed=42", { method: "POST" });
  selectedTrack = null;
  document.getElementById("detail-panel").innerHTML = '<p class="muted">Click a plot on the PPI</p>';
  currentPlots = []; currentTrails = {};
  octx.clearRect(0,0,SIZE,SIZE); ctx.clearRect(0,0,SIZE,SIZE); drawHUD();
  document.getElementById("st-scan").textContent = "Scan #0";
  document.getElementById("plot-tbody").innerHTML = "";
  lastFrameTime = 0;
};
document.getElementById("btn-apply-cfg").onclick = async () => {
  await applyConfig({
    n_az: +document.getElementById("n-az").value,
    n_range: +document.getElementById("n-range").value,
    max_range_m: +document.getElementById("max-range").value * 1000,
  });
  maxRangeM = +document.getElementById("max-range").value * 1000;
  drawHUD();
};
document.getElementById("btn-udp-apply").onclick = () => applyUdp();
["udp-en-240","udp-en-048"].forEach(id => {
  document.getElementById(id).onchange = () => applyUdp();
});

["gain","thr","decay","fade"].forEach(id => {
  document.getElementById(id).oninput = function () {
    const m = {
      gain: () => document.getElementById("gain-val").textContent = this.value + "%",
      thr: () => document.getElementById("thr-val").textContent = this.value + "%",
      decay: () => document.getElementById("decay-val").textContent = (this.value/100).toFixed(2),
      fade: () => document.getElementById("fade-val").textContent = this.value,
    };
    m[id](); applyConfig();
  };
});
document.getElementById("use-codec").onchange = () => applyConfig();
["chk-rings","chk-trails","chk-labels"].forEach(id => {
  document.getElementById(id).onchange = () => { drawHUD(); drawOverlay(currentPlots, currentTrails); };
});

document.getElementById("btn-zoom-in").onclick = () => { zoom = Math.min(8, zoom * 1.25); applyTransform(); };
document.getElementById("btn-zoom-out").onclick = () => { zoom = Math.max(0.4, zoom / 1.25); applyTransform(); };
document.getElementById("btn-zoom-fit").onclick = () => { zoom = 1; panX = 0; panY = 0; applyTransform(); };
area.addEventListener("wheel", (e) => {
  e.preventDefault();
  zoom = Math.min(8, Math.max(0.4, zoom * (e.deltaY < 0 ? 1.12 : 1/1.12)));
  applyTransform();
}, { passive: false });
area.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  dragging = true;
  dragStart = { x: e.clientX - panX, y: e.clientY - panY };
  area.classList.add("panning");
});
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  panX = e.clientX - dragStart.x; panY = e.clientY - dragStart.y; applyTransform();
});
window.addEventListener("mouseup", () => { dragging = false; area.classList.remove("panning"); });

function stageCoords(e) {
  const rect = stage.getBoundingClientRect();
  return [(e.clientX - rect.left) * (SIZE / rect.width), (e.clientY - rect.top) * (SIZE / rect.height)];
}
area.addEventListener("mousemove", (e) => {
  if (dragging) return;
  const [col, row] = stageCoords(e);
  const cx = SIZE / 2, cy = SIZE / 2;
  const dx = col - cx, dy = cy - row;
  const rPix = Math.hypot(dx, dy);
  const rangeM = (rPix / cx) * maxRangeM;
  let az = Math.atan2(dx, dy); if (az < 0) az += 2 * Math.PI;
  const el = document.getElementById("cursor-box");
  el.textContent = rPix <= cx * 1.02
    ? `R ${(rangeM/1000).toFixed(2)} km  ·  Az ${(az*180/Math.PI).toFixed(1)}°`
    : "R — · Az —";
});
area.addEventListener("click", (e) => {
  if (dragging) return;
  const [col, row] = stageCoords(e);
  let best = null, bestD = 16;
  for (const p of currentPlots) {
    const [px, py] = polarToPixel(p.range_m, p.az_rad);
    const d = Math.hypot(px - col, py - row);
    if (d < bestD) { bestD = d; best = p; }
  }
  if (best) selectPlot(best);
});

stage.style.width = SIZE + "px";
stage.style.height = SIZE + "px";
drawHUD();
applyTransform();
setTimeout(() => doScan(), 200);
