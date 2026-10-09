"""Time fsm.centroid on simulated DMK 33UX273 frames.

Answers whether guiding needs the GPU: compares centroid time against the
frame budget at each camera fps mode. Run on the Jetson Nano itself:

    python bench_centroid.py
"""

import platform
import time

import numpy as np

from dmk import FPS_MODES, HEIGHT, WIDTH
from fsm import centroid

rng = np.random.default_rng(0)
BLACK, READ_NOISE, PEAK, SIGMA = 240, 2.0, 2000.0, 3.0  # DN, DN, DN, px
yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]


def frame(cx, cy):
    spot = PEAK * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA**2))
    img = BLACK + spot + rng.normal(0, READ_NOISE, (HEIGHT, WIDTH))
    return np.clip(img, 0, 65535).astype(np.uint16)


def bench(name, fn, frames, truths):
    fn(frames[0], 0)  # warm up
    times, errs = [], []
    for i, (img, (cx, cy)) in enumerate(zip(frames, truths)):
        t = time.perf_counter()
        x, y = fn(img, i)
        times.append(time.perf_counter() - t)
        errs.append(np.hypot(x - cx, y - cy))
    ms = np.array(times) * 1e3
    print(f"{name:<22} median {np.median(ms):7.3f} ms   p99 {np.percentile(ms, 99):7.3f} ms   "
          f"max error {max(errs):.3f} px")
    return np.median(ms)


truths = [(rng.uniform(100, WIDTH - 100), rng.uniform(100, HEIGHT - 100)) for _ in range(30)]
frames = [frame(cx, cy) for cx, cy in truths]
# tracking: the last known position is a few px off the true one
last = [(cx + rng.uniform(-5, 5), cy + rng.uniform(-5, 5)) for cx, cy in truths]

print(f"{platform.machine()}, {platform.processor() or platform.node()}, numpy {np.__version__}, "
      f"{WIDTH}x{HEIGHT} uint16\n")
full = bench("full frame", lambda img, i: centroid(img), frames, truths)
win = bench("tracking window r=40", lambda img, i: centroid(img, (*last[i], 40)), frames, truths)

print("\nfps   budget    full frame    tracking window")
for fps in FPS_MODES:
    budget = 1e3 / fps
    print(f"{fps:>3}  {budget:6.1f} ms   {full / budget:6.0%} used   {win / budget:6.1%} used")
