# dmk-camera

Test and capture tool for The Imaging Source **DMK 33UX273** (Sony IMX273, 1440x1080 mono,
3.45 um pixels, 12-bit, USB3).

The camera runs in UVC mode, so it works with the stock `uvcvideo` driver and no TIS
software. `xu.py` talks to the vendor UVC extension unit directly, which is needed to
**turn off auto-gain** (on by default, and the standard UVC controls can't reach it)
and to set exposure in 1 us steps (UVC's own control only has 100 us steps).

## Setup

    uv venv .venv && uv pip install -p .venv -r requirements.txt

You need `v4l2-ctl` (v4l-utils) and, for `preview`, gstreamer.

## Usage

    .venv/bin/python dmk.py info                    # controls, formats
    .venv/bin/python dmk.py fps                     # throughput test (236 fps @ 8-bit)
    .venv/bin/python dmk.py snap -e 3000 -n 4       # 3 ms, average 4 frames -> FITS + PNG
    .venv/bin/python dmk.py noise -e 1000           # dark stats, cap the sensor first
    .venv/bin/python dmk.py linearity               # exposure sweep
    .venv/bin/python dmk.py beam --dark             # laser spot: auto-exposure, centroid, D4sigma
    .venv/bin/python dmk.py focus -e 10000          # live sharpness meter, for focusing a lens
    .venv/bin/python dmk.py preview -e 5000         # live view window

Output goes to `captures/` (git-ignored). FITS files hold true 12-bit counts (0..4094).

## Measured (2026-09-30, serial 37524328)

- 235.8 fps at 1440x1080 8-bit, 117.6 fps at 16-bit, no dropped frames, ~360 MB/s
- black level 240 DN (the `brightness` control), clip at 4094 DN
- linear to 0.23% of range, 50 us to 11 ms

## Laser safety for the sensor

Without a lens the beam lands directly on the pixels. A few mW focused on the sensor can
permanently damage pixels, so attenuate it (ND filter, or image a spot on paper) and keep
the peak below saturation. `beam` auto-exposes to 70% of full scale.
