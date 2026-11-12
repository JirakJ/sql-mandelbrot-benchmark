"""
JsBrot - Node.js worker_threads Mandelbrot.

One persistent node process (jsbrot.mjs) is spawned at import time (untimed)
with a thread pool sized to all CPU cores. run_jsbrot only writes a request
line and reads the raw uint16 frame back over the pipe.

Author: optimized for multi-core via worker_threads + SharedArrayBuffer
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_MJS = os.path.join(_DIR, "jsbrot.mjs")

_proc = subprocess.Popen(
    ["node", _MJS],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,  # binary frames only; worker logs go to stderr
    cwd=_DIR,
)
atexit.register(_proc.kill)


def _request(width, height, max_iterations):
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    mv = memoryview(out).cast("B")
    n = out.nbytes
    got = 0
    while got < n:
        k = _proc.stdout.readinto(mv[got:])
        if not k:
            raise RuntimeError("jsbrot: node worker died")
        got += k
    return out


# Warm-up at import (untimed): spins up workers and V8 JIT.
_request(16, 16, 8)


def run_jsbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the node worker. Returns (h, w) uint16."""
    return _request(width, height, max_iterations)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_jsbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "jsbrot.png")
