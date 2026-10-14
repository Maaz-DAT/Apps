# RNG — Ranging application (M-07)

A web app (FastAPI + browser A-scope, same pattern as `m02-bus-console/webapp`)
that implements the RNG application from the requirements:

- Estimates target range from radar returns via **pulse compression**
  (matched filtering against a linear-FM chirp or a simple unmodulated
  pulse — selectable in the UI) and **CFAR detection** (cell-averaging
  or ordered-statistic — also selectable).
- Supports both specified input paths:
  - **Raw IQ** — the `rng.pipeline.RngPipeline.process_raw_iq()` path
    runs pulse compression then CFAR directly on `IQBlock` messages
    (the frozen M-00 schema from `common/messages.py`, reused as-is
    from the `m02-bus-console` bus platform — no schema changes).
  - **CAT240 video** — `RngPipeline.process_video()` runs CFAR
    directly on an already-decoded amplitude-vs-range profile, for
    when RNG is fed a decoded video sweep instead of IQ.
- Displays an **A-scope**: amplitude (dB) vs range, with the adaptive
  CFAR threshold overlaid, detected peaks marked, a detection table,
  and **range-rate estimation** (least-squares slope of range vs time,
  per detection range-bin, across successive blocks).
- Ships as its own `uvicorn` service; it builds its own `IQBlock`
  messages via `common.messages.IQBlock` and publishes them on its own
  `busplatform.bus.MessageBus`, so it can be pointed at the same wire
  format (and UDP bridge / recorder) as `m02-bus-console/webapp`
  without any schema edits.

## Run

From `m02-bus-console/`:

```bash
pip install -r requirements.txt
python -m uvicorn m07-rng.webapp.main:app --reload --port 8010
```

Open http://127.0.0.1:8010 . Set the waveform, CFAR and target
scenario in the left panel, click **Apply config**, then **Start
generator**. The A-scope updates over a websocket as synthetic
`IQBlock`s are generated, pulse-compressed and CFAR-detected.

## Run tests

```bash
cd m07-rng
python -m pytest tests -q
```

12 tests cover: range accuracy for both waveforms, chirp vs
simple-pulse compression gain, CA-CFAR and OS-CFAR detection, OS-CFAR
robustness near a second strong target inside the training window,
two-target range resolution, range-rate sign/magnitude, the
end-to-end raw-IQ pipeline (pulse compression → CFAR → range-rate
tracking across blocks), the video-only CFAR path, and waveform
parameter validation.

## What's deliberately out of scope here

- Coherent multi-channel combining of `IQBlock`'s N channels (RNG
  currently processes channel 0; the M-00 `IQBlock` already carries
  `channel_map` for when per-channel/array combining is added).
- A real CAT240 decoder feeding `process_video()` — SURV's
  `m06-surv/surv/cat240.py` already exists and could supply that
  input path; wiring it in is a follow-up, not a schema change.
- Persisting RNG detections as a derived product back onto the bus
  (e.g. a `Detection`/`Plot`-style topic) for SURV or another
  consumer to subscribe to — straightforward to add once that
  derived-product schema is agreed, same boundary rule already
  stated in `REQUIREMENTS.md` for SURV.
