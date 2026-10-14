# SURV — Surveillance Display (M-06) Web Application

Full-fledged web PPI showing **CAT 240 radar video** with **CAT 048 plots/tracks** overlaid.

Implements Maaz guide requirements **REQ-M06-1 … REQ-M06-6**.

## Quick start

```bash
cd surv-web
python3 server.py
# → open http://localhost:8000
```

No extra packages beyond **NumPy** (already used by the codecs).

## Features

| Feature | Requirement | How |
|---|---|---|
| CAT 240 video on PPI | REQ-M06-1 | Polar → Cartesian scan-conversion LUT + green phosphor palette |
| CAT 048 plot symbols | REQ-M06-1 | Diamond symbols + track labels + Mode-3/A / FL |
| Plot–video alignment | REQ-M06-2 | Full encode→decode path; zero noise in alignment mode |
| Gain / clutter threshold | REQ-M06-3 | Affect video only; plots untouched |
| Plot fade + track trails | REQ-M06-4 | Age in scans; trails drawn on overlay canvas |
| Drop counter | REQ-M06-5 | Status bar badge; increments on pipeline errors |
| Click → detail panel | REQ-M06-6 | All decoded CAT 048 fields shown |
| Persistence (afterglow) | REQ-M05-1 | Configurable decay on Cartesian buffer |
| Range rings / cursor | REQ-M05-4 | Rings + live R/Az readout under cursor |
| Continuous run | — | 0.5×–4× speed selectable |

## Pipeline

```
scenario (20 targets on known tracks)
  → CAT 240 generate → encode → decode → polar video
  → CAT 048 generate → encode → decode → plot symbols
  → gain/threshold → scan-convert → persistence → PPI
  → overlay symbols + trails + labels
```

The full codec path is exercised on every scan so codec bugs show up as misplaced symbols rather than hiding until live data arrives.

## Controls

- **Gain / Clutter threshold** — video only
- **Persistence decay** — afterglow strength
- **Plot fade (scans)** — how long symbols stay after last detection
- **Full encode→decode path** — toggle codec on/off for speed comparison
- **Geometry** — azimuths, range cells, max range (Apply geometry rebuilds LUT)
- **Next scan / Run continuous / Reset**

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | UI |
| GET | `/api/status` | Runtime state |
| POST | `/api/scan` | One scan → image + plots JSON |
| POST | `/api/config` | Update gain, threshold, fade, geometry… |
| POST | `/api/reset?seed=42` | Reset scenario |
| GET | `/api/plots` | Current plot list with full fields |
| GET | `/api/targets` | Ground-truth target positions |

## Project layout

```
surv-web/
├── server.py          # stdlib HTTP server + API
├── requirements.txt   # optional (fastapi path in app/)
├── surv/              # codecs + pipeline (from m06-surv)
│   ├── cat048.py
│   ├── cat240.py
│   ├── lut.py
│   ├── pipeline.py
│   └── scenario.py
├── templates/
│   └── index.html
└── static/
    ├── app.js
    └── style.css
```

## Desktop alternative

The original PySide6 desktop app remains in `m06-surv/`:

```bash
cd m06-surv
pip install numpy PySide6 pyqtgraph
python -m surv.app
```
