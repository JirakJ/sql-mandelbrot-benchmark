"""
SquirrelBrot - Squirrel Mandelbrot via a pool of persistent worker processes.

The Squirrel VM is single-threaded, so parallelism comes from N worker
processes spawned at import time (untimed). run_squirrelbrot gives each
worker an interleaved set of top-half rows (row i, i+N, ... for load
balance), reads the raw uint16 bands back, scatters them into the numpy
buffer, and mirrors the bottom half (y-axis symmetry).

License: MIT
"""

import atexit
import os
import shutil
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "squirrelbrot.nut")

_SQ = "/opt/homebrew/bin/sq"
if not os.path.exists(_SQ):
    _SQ = shutil.which("sq")
if _SQ is None:
    raise RuntimeError("squirrelbrot: sq (Squirrel) not found")

_NW = min(14, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_SQ, _SRC],
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


def run_squirrelbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Squirrel pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2
    nw = min(_NW, top)

    # Interleaved rows per worker: worker i computes rows i, i+nw, ...
    for i in range(nw):
        _procs[i].stdin.write(
            f"{width} {height} {max_iterations} {i} {top} {nw}\n".encode()
        )
        _procs[i].stdin.flush()

    for i in range(nw):
        nrows = len(range(i, top, nw))
        band = np.empty((nrows, width), dtype=np.uint16)
        view = memoryview(band).cast("B")
        total = nrows * width * 2
        got = 0
        p = _procs[i]
        while got < total:
            n = p.stdout.readinto(view[got:])
            if not n:
                raise RuntimeError("squirrelbrot: worker died mid-frame")
            got += n
        out[i:top:nw] = band

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check, then one full-size frame.
run_squirrelbrot(16, 16, 8)
run_squirrelbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_squirrelbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "squirrelbrot.png")
