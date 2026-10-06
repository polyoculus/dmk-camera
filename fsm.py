"""Closed-loop laser spot steering with a piezo fast steering mirror (FSM).

Adapted from the lab's Spyder script (YK_FSM_Code_24July2026.py): same serial
protocol, same calibration model, same PI tracker, but driven by dmk.Camera
so it runs at the camera's real frame rate.

Controller: 19200 8N1 with XON/XOFF, ASCII commands terminated by CR LF
(piezosystem jena style). Channel 0 is Y, channel 1 is X, range -15..125 V.

Calibration model (fit per session, pixel = f(volts)):
    cx = a0 + a1*vx + a2*vy + a3*vx^2
    cy = b0 + b1*vx + b2*vy + b3*vy^2
The quadratic terms were needed in the lab's data (U/hump-shaped residuals
with a purely linear fit). Cross terms a2, b1 capture axis crosstalk.
"""

import json
import time

import numpy as np

V_MIN, V_MAX = -15.0, 125.0
V_MID = 55.0


class Mirror:
    def __init__(self, port="/dev/ttyUSB0"):
        import serial

        self.ser = serial.Serial(port, 19200, bytesize=8, parity="N", stopbits=1,
                                 xonxoff=True, timeout=0.1)
        for ch in range(3):
            self._send(f"setk,{ch},1")  # computer control
        self.move(V_MID, V_MID)

    def _send(self, cmd):
        self.ser.write(f"{cmd}\r\n".encode())

    def move(self, vx, vy):
        """Fire and forget; the camera tells us where the spot actually went."""
        vx, vy = clip(vx), clip(vy)
        self._send(f"setall,{vy:.3f},{vx:.3f}")
        return vx, vy

    def read(self):
        """Voltages the controller reports, as (vx, vy)."""
        self.ser.reset_input_buffer()
        self._send("measure")
        fields = self.ser.read_until(b"\r\n", 64).decode(errors="replace").strip().split(",")
        try:
            return float(fields[2]), float(fields[1])
        except (IndexError, ValueError):
            raise IOError(f"unexpected reply to measure: {fields!r}")

    def close(self):
        self.ser.close()


def clip(v):
    return float(np.clip(v, V_MIN, V_MAX))


def centroid(img, window=None):
    """Spot centroid in full-frame pixels, or None if there's no spot.

    Background is the frame median; pixels below a quarter of the peak are
    dropped (the lab script's threshold, but applied after background
    subtraction so the 240 DN black level doesn't bias it). Uses row/column
    projections, which is a few ms on a full frame. `window=(cx, cy, r)`
    restricts the search to a box around the last known position.
    """
    x0 = y0 = 0
    if window is not None:
        cx, cy, r = window
        x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
        img = img[y0:int(cy) + r, x0:int(cx) + r]
    a = img.astype(np.float32) - np.median(img)
    peak = a.max()
    if peak < 20:  # DN; well above the ~2 DN read noise
        return None
    a[a < peak / 4] = 0
    tot = a.sum()
    return (x0 + a.sum(0) @ np.arange(a.shape[1]) / tot,
            y0 + a.sum(1) @ np.arange(a.shape[0]) / tot)


# ---------------------------------------------------------------- model

def _design(vx, vy):
    one = np.ones_like(vx)
    return (np.column_stack([one, vx, vy, vx ** 2]),
            np.column_stack([one, vx, vy, vy ** 2]))


def fit(vx, vy, cx, cy):
    """Least-squares fit of the model. Returns (cal dict, rms_x, rms_y)."""
    vx, vy, cx, cy = map(np.asarray, (vx, vy, cx, cy))
    ax, ay = _design(vx, vy)
    a = np.linalg.lstsq(ax, cx, rcond=None)[0]
    b = np.linalg.lstsq(ay, cy, rcond=None)[0]
    rx, ry = cx - ax @ a, cy - ay @ b
    return dict(a=a.tolist(), b=b.tolist()), rx.std(), ry.std()


def predict(cal, vx, vy):
    a0, a1, a2, a3 = cal["a"]
    b0, b1, b2, b3 = cal["b"]
    return a0 + a1 * vx + a2 * vy + a3 * vx ** 2, b0 + b1 * vx + b2 * vy + b3 * vy ** 2


def jacobian(cal, vx, vy):
    """d(cx, cy) / d(vx, vy) at the given voltages, in px/V."""
    _, a1, a2, a3 = cal["a"]
    _, b1, b2, b3 = cal["b"]
    return np.array([[a1 + 2 * a3 * vx, a2],
                     [b1, b2 + 2 * b3 * vy]])


def invert(cal, x, y, vx=V_MID, vy=V_MID, iters=20):
    """Voltages the model says put the spot at (x, y), by Newton's method."""
    for _ in range(iters):
        px, py = predict(cal, vx, vy)
        dv = np.linalg.solve(jacobian(cal, vx, vy), [x - px, y - py])
        vx, vy = clip(vx + dv[0]), clip(vy + dv[1])
        if np.hypot(*dv) < 1e-4:
            break
    return vx, vy


def save_cal(cal, path):
    path.write_text(json.dumps(cal, indent=2))


def load_cal(path):
    if not path.exists():
        raise SystemExit(f"no calibration at {path}; run `dmk.py fsm calibrate` first")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- procedures
# `cam` needs .frame() and .flush(n); `mirror` needs .move(vx, vy).

def measure(cam, settle):
    """Centroid from a frame exposed after the mirror has settled."""
    time.sleep(settle)
    cam.flush(1)  # the buffered frame may have been exposing during the move
    return centroid(cam.frame())


def calibrate(cam, mirror, n=7, settle=0.05, log=print):
    """Visit an n x n voltage grid, then fit the model to where the spot landed."""
    vs = np.linspace(V_MIN, V_MAX, n)
    pts = []
    for i, vy in enumerate(vs):
        # serpentine order keeps steps small, so less ringing to wait out
        for vx in (vs if i % 2 == 0 else vs[::-1]):
            mirror.move(vx, vy)
            c = measure(cam, settle)
            if c is None:
                log(f"  V=({vx:6.1f}, {vy:6.1f})  spot lost, skipped")
                continue
            pts.append((vx, vy, *c))
            log(f"  V=({vx:6.1f}, {vy:6.1f})  spot ({c[0]:7.1f}, {c[1]:7.1f})")
    mirror.move(V_MID, V_MID)
    if len(pts) < 8:
        raise SystemExit(f"only {len(pts)} usable points; is the spot on the sensor?")
    vx, vy, cx, cy = np.array(pts).T
    cal, rx, ry = fit(vx, vy, cx, cy)
    cal.update(points=np.array(pts).tolist(), rms_x=rx, rms_y=ry,
               reach=dict(x=[cx.min(), cx.max()], y=[cy.min(), cy.max()]))
    return cal


def goto(cam, mirror, cal, x, y, tol=0.2, gain=0.9, max_iter=10, settle=0.02, log=print):
    """Open-loop jump using the model, then camera-corrected Newton steps.

    Returns (vx, vy, cx, cy, error_px).
    """
    vx, vy = invert(cal, x, y)
    err = np.inf
    cx = cy = np.nan
    for i in range(max_iter):
        vx, vy = mirror.move(vx, vy)
        c = measure(cam, settle)
        if c is None:
            raise SystemExit("spot lost")
        cx, cy = c
        e = np.array([x - cx, y - cy])
        err = np.hypot(*e)
        log(f"  iter {i}: V=({vx:7.3f}, {vy:7.3f})  spot ({cx:8.2f}, {cy:8.2f})  "
            f"error {err:.3f} px")
        if err < tol:
            break
        dv = np.linalg.solve(jacobian(cal, vx, vy), e)
        vx, vy = vx + gain * dv[0], vy + gain * dv[1]
    return vx, vy, cx, cy, err


def track(cam, mirror, cal, x, y, duration, kp=0.5, ki=0.02, ilimit=20.0, settle=0.0,
          window=40, log=print):
    """PI loop holding the spot at (x, y) until `duration` s or Ctrl+C.

    Returns a log array with columns t, vx, vy, cx, cy, error.
    """
    vx, vy = invert(cal, x, y)
    integ = np.zeros(2)
    rows = []
    last = None
    t0 = time.perf_counter()
    try:
        while (t := time.perf_counter() - t0) < duration:
            vx, vy = mirror.move(vx, vy)
            if settle:
                time.sleep(settle)
            img = cam.frame()
            c = centroid(img, (*last, window)) if last else None
            c = c or centroid(img)  # fall back to the full frame if the spot left the box
            if c is None:
                log("\n  spot lost, holding voltage")
                last = None
                continue
            last = c
            e = np.array([x - c[0], y - c[1]])
            integ = np.clip(integ + e, -ilimit, ilimit)
            dv = np.linalg.solve(jacobian(cal, vx, vy), kp * e + ki * integ)
            rows.append((t, vx, vy, *c, np.hypot(*e)))
            vx, vy = vx + dv[0], vy + dv[1]
            if len(rows) % 20 == 0:
                log(f"\r  {t:6.1f}s  error {rows[-1][5]:6.3f} px  V=({vx:7.2f}, {vy:7.2f})  "
                    f"{len(rows) / t:6.1f} Hz", end="", flush=True)
    except KeyboardInterrupt:
        pass
    log()
    return np.array(rows).reshape(-1, 6)


def settle_curve(cam, mirror, step=5.0, axis="x", n=60, before=5):
    """Step one axis and record the spot every frame, to see how fast the mirror settles.

    Returns (t, position_px) with t = 0 at the step command; the first
    `before` samples are taken before the step (negative t).
    """
    vx = vy = V_MID
    mirror.move(vx, vy)
    time.sleep(0.5)
    cam.flush()
    k = 0 if axis == "x" else 1
    t, pos = [], []
    for i in range(before + n):
        if i == before:
            t0 = time.perf_counter()
            mirror.move(vx + step * (axis == "x"), vy + step * (axis == "y"))
        c = centroid(cam.frame())
        t.append(time.perf_counter())
        pos.append(np.nan if c is None else c[k])
    mirror.move(V_MID, V_MID)
    return np.array(t) - t0, np.array(pos)
