"""
SmalltalkBrot - GNU Smalltalk Mandelbrot via a pool of persistent gst workers.

Smalltalk Processes are green threads (one native VM thread), so parallelism
comes from N gst worker processes spawned at import time (untimed). Each worker
runs a read-eval loop over stdin (smalltalkbrot.st), computes interleaved
top-half row bands with FloatD arithmetic and the cardioid/bulb early-outs, and
writes each row as raw uint16 LE bytes. run_smalltalkbrot pipelines all
requests, reads the bands back into the numpy buffer, and mirrors the bottom
half (y-axis symmetry).

The gst binary is /opt/homebrew/bin/gst (the interactive shell aliases
gst=git status, so the full path is always used).

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "smalltalkbrot.st")
_GST = "/opt/homebrew/bin/gst"

_NW = min(14, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_GST, _SRC],
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


def run_smalltalkbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the gst pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2

    # Interleaved bands (band c -> worker c % NW) balance the uneven per-row
    # cost (in-set rows early-out cheaply, boundary rows iterate to the cap).
    chunk = max(1, top // (_NW * 4))
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
                    raise RuntimeError("smalltalkbrot: worker died mid-frame")
                got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check plus one full-size frame so every
# worker's bytecode and pipes are hot before timing starts.
run_smalltalkbrot(16, 16, 8)
run_smalltalkbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_smalltalkbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "smalltalkbrot.png")
