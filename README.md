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
    .venv/bin/python dmk.py record -t 60 -i 0.5     # a frame every 0.5 s for 60 s -> PNGs + cube.fits
    .venv/bin/python dmk.py focus -e 10000          # live sharpness meter, for focusing a lens
    .venv/bin/python dmk.py preview -e 5000         # live view window

### Fast steering mirror (`fsm.py`)

Closed-loop spot steering with the lab's piezo FSM controller (serial, 19200 baud,
`setall`/`measure` commands, -15..125 V), ported from the lab's Spyder script.

    .venv/bin/python dmk.py fsm move 55 55          # set volts, print where the spot is
    .venv/bin/python dmk.py fsm calibrate           # 7x7 voltage grid -> captures/fsm-cal.json
    .venv/bin/python dmk.py fsm goto 700 400        # put the spot on a pixel (to 0.2 px)
    .venv/bin/python dmk.py fsm track -t 30         # PI loop holding it there -> csv + plot
    .venv/bin/python dmk.py fsm settle --step 5     # step response of the mirror

Calibrate once per session (after anything in the optics moves). The model is
`cx = a0 + a1 vx + a2 vy + a3 vx^2` and `cy = b0 + b1 vx + b2 vy + b3 vy^2`; `goto` and
`track` use it for direction and the camera for the actual error. Default port is
`/dev/ttyUSB0` (`--port` to change); your user needs to be in the `uucp` group to open it.

Output goes to `captures/` (git-ignored). FITS files hold true 12-bit counts (0..4094).

## Measured (2026-09-30, serial 37524328)

- 235.8 fps at 1440x1080 8-bit, 117.6 fps at 16-bit, no dropped frames, ~360 MB/s
- black level 240 DN (the `brightness` control), clip at 4094 DN
- linear to 0.23% of range, 50 us to 11 ms

## Laser safety for the sensor

Without a lens the beam lands directly on the pixels. A few mW focused on the sensor can
permanently damage pixels, so attenuate it (ND filter, or image a spot on paper) and keep
the peak below saturation. `beam` auto-exposes to 70% of full scale.
