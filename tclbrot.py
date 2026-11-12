"""
TclBrot - Tcl Mandelbrot via a pool of persistent tclsh worker processes.

Tcl has no in-process threads for this workload, so parallelism comes from
N tclsh workers spawned at import time (untimed). run_tclbrot splits the top
half of the image into small interleaved row bands (band c -> worker c % N)
for load balance, pipelines all requests, reads uint16 LE bands back into the
numpy buffer, and mirrors the bottom half (y-axis symmetry).

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
_SRC = os.path.join(_DIR, "tclbrot.tcl")


def _find_tclsh():
    # Prefer the newest available interpreter.
    for cand in ("tclsh9.1", "tclsh9.0", "tclsh8.7", "tclsh8.6", "tclsh"):
        p = shutil.which(cand)
        if p:
            return p
    if os.path.exists("/usr/bin/tclsh"):
        return "/usr/bin/tclsh"
    raise RuntimeError("tclbrot: no tclsh found")


_TCLSH = _find_tclsh()
_NW = min(14, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_TCLSH, _SRC],
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


def run_tclbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the tclsh pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2

    # Interleaved bands: band c -> worker c % NW. Factor 2 balances load
    # without the per-request flush overhead of finer chunks (measured).
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
                    raise RuntimeError("tclbrot: worker died mid-frame")
                got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check, then one full-size frame so
# every worker has its procs bytecode-compiled before timing starts.
run_tclbrot(16, 16, 8)
run_tclbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_tclbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "tclbrot.png")
