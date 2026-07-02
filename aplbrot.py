"""
AplBrot - GNU APL Mandelbrot via a pool of persistent `apl` worker processes.

APL is a whole-array (vectorized) language with no easy in-process threads, so
parallelism comes from N `apl` interpreters spawned at import time (untimed).
Each worker runs aplbrot.apl: it reads a request line "w h mi a b" (rows a..b-1
of the top half), computes the entire (b-a, w) band with vectorized array ops
(the escape-time recurrence is iterated on the full complex matrix under a
boolean mask -- no per-pixel loop, only a loop over the <=mi steps), and writes
the band back as raw little-endian uint16 bytes.

run_aplbrot deals interleaved row bands across the workers, pipelines all
requests, reads the uint16 bands straight into the numpy buffer, and mirrors the
bottom half (y-axis symmetry).

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
_SRC = os.path.join(_DIR, "aplbrot.apl")


def _find_apl():
    for cand in ("apl",):
        p = shutil.which(cand)
        if p:
            return p
    if os.path.exists("/opt/homebrew/bin/apl"):
        return "/opt/homebrew/bin/apl"
    raise RuntimeError("aplbrot: no apl interpreter found")


_APL = _find_apl()
_NW = min(14, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_APL, "--script", "-s", "-f", _SRC],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(_NW)
]


def _shutdown():
    for p in _procs:
        try:
            p.terminate()
        except Exception:
            pass


atexit.register(_shutdown)


def run_aplbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the apl pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2

    # Interleaved bands: band c -> worker c % NW, small enough that no single
    # band overruns the OS pipe buffer while requests are still being written.
    chunk = max(1, top // (_NW * 2))
    per = [[] for _ in range(_NW)]
    a = 0
    c = 0
    while a < top:
        b = min(a + chunk, top)
        per[c % _NW].append((a, b))
        a = b
        c += 1

    for p, bands in zip(_procs, per):
        if bands:
            msg = "".join(
                f"{width} {height} {max_iterations} {a} {b}\n" for a, b in bands
            )
            p.stdin.write(msg.encode())
            p.stdin.flush()

    for p, bands in zip(_procs, per):
        for a, b in bands:
            view = memoryview(out[a:b]).cast("B")
            total = (b - a) * width * 2
            got = 0
            while got < total:
                n = p.stdout.readinto(view[got:])
                if not n:
                    raise RuntimeError("aplbrot: worker died mid-frame")
                got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check, then one full-size frame so every
# interpreter has parsed the workspace and is at steady state before timing.
run_aplbrot(16, 16, 8)
run_aplbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_aplbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "aplbrot.png")
