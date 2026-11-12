"""
BashBrot - pure Bash Mandelbrot via a pool of persistent worker processes.

Bash arithmetic is 64-bit integer only, so the worker (bashbrot.sh) runs the
escape loop in Q26 fixed point. Parallelism comes from N persistent bash
processes spawned at import (untimed). Rows are interleaved across workers
(stride = pool size) for load balance; each worker emits one text line of
space-separated decimal counts per row - the honest bash way to move data.
One drain thread per worker keeps pipes from filling and stalling the pool.
run_bashbrot assembles the top half and mirrors the bottom (y-axis symmetry).

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess
import threading

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "bashbrot.sh")

# Prefer a newer brew bash if present; /bin/bash 3.2 works fine.
_BASH = "/opt/homebrew/bin/bash"
if not os.path.exists(_BASH):
    _BASH = "/bin/bash"

_NW = min(14, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_BASH, _SRC],
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


def _drain(p, out, start, stride, top):
    """Read one text line per row (start, start+stride, ... < top)."""
    for r in range(start, top, stride):
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("bashbrot: worker died mid-frame")
        out[r] = np.array(line.split(), dtype=np.uint16)


def run_bashbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the bash pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2

    threads = []
    for i in range(min(_NW, top)):
        p = _procs[i]
        p.stdin.write(f"{width} {height} {max_iterations} {i} {_NW} {top}\n".encode())
        p.stdin.flush()
        t = threading.Thread(target=_drain, args=(p, out, i, _NW, top))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): touches every worker, checks the protocol.
run_bashbrot(64, 32, 16)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_bashbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "bashbrot.png")
