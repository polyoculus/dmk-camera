#!/usr/bin/env python3
"""Test / capture tool for The Imaging Source DMK 33UX273 (USB3, UVC mode).

The camera enumerates as a plain UVC device, so no TIS driver is needed:
frames come through OpenCV's V4L2 backend and controls are set with v4l2-ctl.

Sensor: Sony IMX273, 1440x1080 mono, 3.45 um pixels, 12-bit ADC.
'Y16 ' frames carry the 12-bit value in the top bits (low nibble always 0),
so we shift right by 4 and store true 0..4095 counts.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import xu

PIXEL_UM = 3.45
WIDTH, HEIGHT = 1440, 1080
FULL_SCALE = 4094  # 0xFFE0 >> 4; the sensor clips here
# fps modes the camera advertises for 1440x1080 Y16 (GREY also has 236)
FPS_MODES = [120, 60, 30, 15, 5, 1]
OUT = Path(__file__).parent / "captures"


def find_device():
    for p in sorted(Path("/sys/class/video4linux").glob("video*")):
        if "DMK 33UX273" in (p / "name").read_text():
            dev = f"/dev/{p.name}"
            # first node of the pair is the capture node, second is metadata
            if subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats"],
                              capture_output=True, text=True).stdout.count("GREY"):
                return dev
    sys.exit("DMK 33UX273 not found (is it plugged into a USB3 port?)")


def v4l2_set(dev, **ctrls):
    arg = ",".join(f"{k}={v}" for k, v in ctrls.items())
    subprocess.run(["v4l2-ctl", "-d", dev, "-c", arg], check=True)


def v4l2_get(dev, name):
    out = subprocess.run(["v4l2-ctl", "-d", dev, "-C", name],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split(":")[1].split()[0])


class Camera:
    """exposure in microseconds, gain in dB (0..48)."""

    def __init__(self, exposure_us=1000, gain_db=0.0, fps=None):
        self.dev = find_device()
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        # fastest mode whose frame period still fits the exposure
        self.fps = fps or next((f for f in FPS_MODES if 1e6 / f >= exposure_us), 1)
        self.cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self.cap.isOpened():
            sys.exit(f"could not open {self.dev}")
        self.xu = xu.XU(self.dev)
        self.apply()

    def apply(self):
        # auto gain lives only in the vendor XU and defaults on; kill it first
        v4l2_set(self.dev, auto_exposure=1)
        self.xu.set(xu.GAIN_AUTO, 0)
        self.xu.set(xu.AUTO_SHUTTER, 0)
        # XU exposure is in us (the UVC one only has 100 us steps)
        self.xu.set(xu.EXPOSURE_US, max(1, round(self.exposure_us)))
        v4l2_set(self.dev, gain=round(self.gain_db * 10))
        self.flush()

    def set(self, exposure_us=None, gain_db=None):
        if exposure_us is not None:
            self.exposure_us = exposure_us
        if gain_db is not None:
            self.gain_db = gain_db
        self.apply()

    def flush(self, n=6):
        # drain frames that were exposed with the old settings
        for _ in range(n):
            self.cap.grab()

    def frame(self):
        ok, f = self.cap.read()
        if not ok or f.shape != (HEIGHT, WIDTH):
            sys.exit(f"bad frame: ok={ok} shape={None if f is None else f.shape}")
        return f >> 4

    def frames(self, n):
        return np.stack([self.frame() for _ in range(n)])

    def close(self):
        self.cap.release()
        self.xu.close()


def stats(img):
    sat = (img >= FULL_SCALE).mean() * 100
    return (f"min {img.min():4d}  max {img.max():4d}  mean {img.mean():7.1f}  "
            f"std {img.std():6.1f}  saturated {sat:.3f}%")


def beam_params(img, dark=None):
    """Centroid and ISO 11146 second-moment (D4-sigma) diameters, in pixels."""
    a = img.astype(np.float64)
    a -= dark if dark is not None else np.median(a)
    a[a < 0] = 0
    # crude background cut so the wings of noise don't blow up the moments
    a[a < 0.02 * a.max()] = 0
    tot = a.sum()
    if tot == 0:
        return None
    y, x = np.indices(a.shape)
    cx, cy = (a * x).sum() / tot, (a * y).sum() / tot
    sx = np.sqrt((a * (x - cx) ** 2).sum() / tot)
    sy = np.sqrt((a * (y - cy) ** 2).sum() / tot)
    return dict(cx=cx, cy=cy, d4x=4 * sx, d4y=4 * sy, peak=img.max())


def save(img, name, meta):
    from astropy.io import fits
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    stem = OUT / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
    hdr = fits.Header()
    for k, v in meta.items():
        hdr[k[:8].upper()] = v
    fits.writeto(f"{stem}.fits", img.astype(np.uint16), hdr, overwrite=True)
    plt.imsave(f"{stem}.png", img, cmap="gray", vmin=0, vmax=max(1, img.max()))
    return stem


# ---------------------------------------------------------------- commands

def cmd_info(a):
    dev = find_device()
    print(f"device: {dev}")
    subprocess.run(["v4l2-ctl", "-d", dev, "-D", "-V", "-P", "-L"])


def cmd_fps(a):
    dev = find_device()
    pix = "GREY" if a.bits == 8 else "Y16 "
    subprocess.run(["v4l2-ctl", "-d", dev,
                    f"--set-fmt-video=width={WIDTH},height={HEIGHT},pixelformat={pix}",
                    "-p", str(a.rate)], check=True)
    v4l2_set(dev, auto_exposure=1, exposure_time_absolute=1)
    t = time.time()
    subprocess.run(["v4l2-ctl", "-d", dev, "--stream-mmap",
                    f"--stream-count={a.n}", "--stream-to=/dev/null"],
                   capture_output=True, check=True)
    dt = time.time() - t
    print(f"{a.n} frames {pix.strip()} @ requested {a.rate} fps: "
          f"{a.n / dt:.1f} fps measured (incl. startup), "
          f"{a.n * WIDTH * HEIGHT * (a.bits // 8) / dt / 1e6:.0f} MB/s")


def cmd_snap(a):
    cam = Camera(a.exposure, a.gain)
    stack = cam.frames(a.n)
    cam.close()
    img = stack.mean(0).round().astype(np.uint16) if a.n > 1 else stack[0]
    print(stats(img))
    stem = save(img, a.name, dict(exptime=a.exposure * 1e-6, gain_db=a.gain,
                                  nframes=a.n, camera="DMK 33UX273",
                                  pixsize=PIXEL_UM))
    print(f"saved {stem}.fits / .png")


def cmd_noise(a):
    """Dark frames: read noise + fixed pattern. Cover the sensor first."""
    cam = Camera(a.exposure, a.gain)
    stack = cam.frames(a.n).astype(np.float64)
    cam.close()
    temporal = stack.std(0).mean()
    fpn = stack.mean(0).std()
    print(f"{a.n} dark frames @ {a.exposure} us, {a.gain} dB")
    print(f"  black level (mean):   {stack.mean():.2f} DN")
    print(f"  temporal noise:       {temporal:.2f} DN rms")
    print(f"  fixed-pattern noise:  {fpn:.2f} DN rms")
    hot = (stack.mean(0) > stack.mean() + 10 * temporal).sum()
    print(f"  hot pixels (>10 sig): {hot}")


def cmd_linearity(a):
    """Sweep exposure at constant illumination, check signal scales linearly."""
    cam = Camera(a.start)
    print("(keep the scene/illumination fixed during the sweep)")
    exps = np.geomspace(a.start, a.stop, a.steps).round()
    rows = []
    for e in exps:
        cam.set(exposure_us=e)
        m = cam.frames(3).mean()
        rows.append((e, m))
        print(f"  {e:9.0f} us  mean {m:8.1f} DN")
    cam.close()
    e, m = np.array(rows).T
    ok = m < 0.9 * FULL_SCALE
    if ok.sum() >= 3:
        k, b = np.polyfit(e[ok], m[ok], 1)
        resid = (m[ok] - (k * e[ok] + b)) / (m[ok].max() - b) * 100
        print(f"fit: {k:.4g} DN/us, offset {b:.1f} DN, "
              f"max nonlinearity {np.abs(resid).max():.2f}% of range")


def autoexpose(cam, target=0.7, lo=1, hi=500_000):
    """Binary-search exposure so the peak pixel sits at `target` of full scale."""
    for _ in range(14):
        e = np.sqrt(lo * hi)
        cam.set(exposure_us=e)
        peak = np.percentile(cam.frame(), 99.999)
        if peak > target * FULL_SCALE:
            hi = e
        else:
            lo = e
        if hi / lo < 1.1:
            break
    cam.set(exposure_us=lo)
    return lo


def cmd_beam(a):
    cam = Camera(a.exposure or 1000, a.gain)
    if a.exposure is None:
        e = autoexpose(cam)
        print(f"auto exposure: {e:.0f} us")
    dark = None
    if a.dark:
        input("block the laser, then press enter to take a dark frame...")
        cam.flush()
        dark = cam.frames(a.n).mean(0)
        input("unblock the laser, press enter...")
        cam.flush()
    img = cam.frames(a.n).mean(0).round().astype(np.uint16)
    cam.close()
    print(stats(img))
    if img.max() >= FULL_SCALE:
        print("WARNING: saturated - the diameters below are wrong. "
              "Lower exposure or add attenuation.")
    p = beam_params(img, dark)
    if p is None:
        sys.exit("no signal")
    um = PIXEL_UM
    print(f"centroid: ({p['cx']:.1f}, {p['cy']:.1f}) px")
    print(f"D4sigma:  x {p['d4x']:.1f} px = {p['d4x'] * um:.0f} um   "
          f"y {p['d4y']:.1f} px = {p['d4y'] * um:.0f} um")
    stem = save(img, "beam", dict(exptime=cam.exposure_us * 1e-6, gain_db=a.gain,
                                  nframes=a.n, cx=p["cx"], cy=p["cy"],
                                  d4x_um=p["d4x"] * um, d4y_um=p["d4y"] * um))
    plot_beam(img, p, f"{stem}-profile.png")
    print(f"saved {stem}.fits / .png / -profile.png")


def plot_beam(img, p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    cx, cy = p["cx"], p["cy"]
    r = int(max(p["d4x"], p["d4y"], 20) * 1.5)
    x0, x1 = max(0, int(cx) - r), min(WIDTH, int(cx) + r)
    y0, y1 = max(0, int(cy) - r), min(HEIGHT, int(cy) + r)
    crop = img[y0:y1, x0:x1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(img, cmap="inferno")
    ax[0].set_title("full frame")
    ax[1].imshow(crop, cmap="inferno", extent=(x0, x1, y1, y0))
    ax[1].add_patch(Ellipse((cx, cy), p["d4x"], p["d4y"], fill=False,
                            color="cyan", lw=1))
    ax[1].set_title("D4σ ellipse")
    ax[2].plot(np.arange(x0, x1), img[int(round(cy)), x0:x1], label="x cut")
    ax[2].plot(np.arange(y0, y1), img[y0:y1, int(round(cx))], label="y cut")
    ax[2].axhline(FULL_SCALE, color="r", ls="--", lw=0.8, label="saturation")
    ax[2].set_xlabel("px")
    ax[2].set_ylabel("DN")
    ax[2].legend()
    ax[2].set_title(f"D4σ {p['d4x'] * PIXEL_UM:.0f} x {p['d4y'] * PIXEL_UM:.0f} µm")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def cmd_focus(a):
    """Live sharpness readout: move the lens until the number peaks."""
    cam = Camera(a.exposure, a.gain)
    best = 0
    print("move the lens slowly; Ctrl+C to stop")
    try:
        while True:
            f = cam.frame().astype(np.float32)
            # variance of the Laplacian, normalised by brightness so exposure
            # changes don't masquerade as focus changes
            sharp = cv2.Laplacian(f, cv2.CV_32F).var() / max(f.mean(), 1) ** 2 * 1e4
            best = max(best, sharp)
            bar = "#" * int(40 * sharp / best)
            print(f"\rsharpness {sharp:8.2f}  best {best:8.2f}  {bar:<40}  "
                  f"peak {f.max():4.0f}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()


def cmd_preview(a):
    dev = find_device()
    x = xu.XU(dev)
    v4l2_set(dev, auto_exposure=1, gain=round(a.gain * 10))
    x.set(xu.GAIN_AUTO, 0)
    x.set(xu.AUTO_SHUTTER, 0)
    x.set(xu.EXPOSURE_US, max(1, round(a.exposure)))
    x.close()
    subprocess.run(["gst-launch-1.0", "-q", "v4l2src", f"device={dev}", "!",
                    f"video/x-raw,format=GRAY8,width={WIDTH},height={HEIGHT},framerate=30/1",
                    "!", "videoconvert", "!", "autovideosink", "sync=false"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="device info and controls")

    p = sub.add_parser("fps", help="measure streaming throughput")
    p.add_argument("--bits", type=int, choices=[8, 16], default=8)
    p.add_argument("--rate", type=int, default=236)
    p.add_argument("-n", type=int, default=1000)

    def common(p, exposure=1000):
        p.add_argument("-e", "--exposure", type=float, default=exposure, help="us")
        p.add_argument("-g", "--gain", type=float, default=0.0, help="dB, 0..48")
        p.add_argument("-n", type=int, default=1, help="frames to average")

    p = sub.add_parser("snap", help="save a 12-bit frame as FITS + PNG")
    common(p)
    p.add_argument("--name", default="snap")

    p = sub.add_parser("noise", help="dark-frame noise stats (cover the sensor)")
    common(p)
    p.set_defaults(n=50)

    p = sub.add_parser("linearity", help="exposure sweep")
    p.add_argument("--start", type=float, default=100)
    p.add_argument("--stop", type=float, default=100_000)
    p.add_argument("--steps", type=int, default=10)

    p = sub.add_parser("beam", help="laser spot: centroid + D4sigma")
    common(p, exposure=None)
    p.add_argument("--dark", action="store_true", help="take a dark frame first")
    p.set_defaults(n=5)

    p = sub.add_parser("focus", help="live sharpness meter for focusing a lens")
    common(p, exposure=10000)

    p = sub.add_parser("preview", help="live view window (gstreamer)")
    p.add_argument("-e", "--exposure", type=float, default=10000, help="us")
    p.add_argument("-g", "--gain", type=float, default=0.0)

    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
