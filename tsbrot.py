"""
TsBrot - TypeScript Mandelbrot via Node.js native type-stripping + worker_threads.

One persistent node process runs tsbrot.ts directly (Node >= 23 strips the types
at load, no build step) with a thread pool sized to all CPU cores. run_tsbrot
only writes a request line and reads the raw uint16 frame back over the pipe.
Same algorithm as jsbrot — the point is that TypeScript's types are erased at
runtime, so it runs at JavaScript speed.

License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_TS = os.path.join(_DIR, "tsbrot.ts")

_proc = subprocess.Popen(
    ["node", _TS],
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
            raise RuntimeError("tsbrot: node worker died")
        got += k
    return out


# Warm-up at import (untimed): strips types, spins up workers and V8 JIT.
_request(16, 16, 8)


def run_tsbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the node/TS worker. Returns (h, w) uint16."""
    return _request(width, height, max_iterations)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_tsbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "tsbrot.png")
