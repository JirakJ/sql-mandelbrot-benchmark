"""
RBrot - GNU R Mandelbrot via a persistent Rscript worker process.

R is single-threaded, so the worker uses NumPy-style whole-grid vectorization
(flat double vectors, escape bookkeeping, periodic compaction of escaped
cells) plus cardioid/bulb early-out and y-axis symmetry. Worker startup and
bytecode warm-up happen at import time (untimed). run_rbrot only writes a
request line and reads raw uint16 pixels back.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import shutil
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "rbrot.R")

_RSCRIPT = shutil.which("Rscript")
if _RSCRIPT is None:
    raise RuntimeError("rbrot: Rscript not found on PATH")

# One persistent R process for the whole benchmark; stderr swallowed.
_proc = subprocess.Popen(
    [_RSCRIPT, "--vanilla", _SRC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_rbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the R worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("rbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then one full frame so R's
# bytecode compiler has compiled the hot function before timing starts.
run_rbrot(16, 16, 8)
run_rbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_rbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "rbrot.png")
