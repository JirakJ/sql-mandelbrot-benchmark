"""
ChickenBrot - CHICKEN Scheme Mandelbrot via a pool of persistent workers.

CHICKEN Scheme has only green threads (no OS-level parallelism), so all
concurrency comes from N worker processes spawned at import time (untimed).
The worker is compiled Scheme -> C -> native with csc when stale. Each worker
reads a request line "w h mi a b" and writes the top-half row band [a,b) as
raw little-endian uint16 rows. run_chickenbrot deals one contiguous band per
worker, reads the bands into the numpy buffer, and mirrors the bottom half
(y-axis symmetry).

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
_SRC = os.path.join(_DIR, "chickenbrot.scm")
_BIN = os.path.join(_DIR, "chickenbrot_bin")

_CSC = shutil.which("csc") or "/opt/homebrew/bin/csc"


def _build():
    """Compile the worker if the binary is missing or stale (untimed)."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return
    res = subprocess.run(
        [_CSC, "-O5", "-d0", "-local", "-strict-types", _SRC, "-o", _BIN],
        cwd=_DIR,
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError("chickenbrot: build failed:\n" + res.stderr.decode())


_build()

_NW = min(14, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(_NW)
]


def _shutdown():
    for p in _procs:
        p.terminate()


atexit.register(_shutdown)


def run_chickenbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the CHICKEN pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2
    base, rem = divmod(top, _NW)

    # One contiguous band per worker; skip workers with no rows.
    bands = []
    r0 = 0
    for i in range(_NW):
        n = base + (1 if i < rem else 0)
        if n:
            bands.append((_procs[i], r0, r0 + n))
            r0 += n

    for p, a, b in bands:
        p.stdin.write(f"{width} {height} {max_iterations} {a} {b}\n".encode())
        p.stdin.flush()

    for p, a, b in bands:
        view = memoryview(out[a:b]).cast("B")
        total = (b - a) * width * 2
        got = 0
        while got < total:
            n = p.stdout.readinto(view[got:])
            if not n:
                raise RuntimeError("chickenbrot: worker died mid-frame")
            got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check plus one full-size frame so
# pages and pipes are hot before timing starts.
run_chickenbrot(16, 16, 8)
run_chickenbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_chickenbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "chickenbrot.png")
