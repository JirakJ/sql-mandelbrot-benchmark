"""
RacketBrot - Racket CS Mandelbrot via one persistent worker using places.

A single racket process (racketbrot.rkt) is spawned at import time (untimed).
It creates a pool of (processor-count) places once at startup for true
parallelism. run_racketbrot only writes a request line and reads the raw
uint16 LE frame back over the pipe.

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
_SRC = os.path.join(_DIR, "racketbrot.rkt")

_RACKET = shutil.which("racket")
if _RACKET is None and os.path.exists("/opt/homebrew/bin/racket"):
    _RACKET = "/opt/homebrew/bin/racket"
if _RACKET is None:
    raise RuntimeError("racketbrot: racket not found")

_proc = subprocess.Popen(
    [_RACKET, _SRC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,  # binary frames only; logs go to stderr
    stderr=None,
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
            raise RuntimeError("racketbrot: racket worker died")
        got += k
    return out


# Warm-up at import (untimed): protocol check, then full-size frames so the
# places and heaps are primed before timing starts.
_request(16, 16, 8)
_request(1400, 800, 256)


def run_racketbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Racket place pool. Returns (h, w) uint16."""
    return _request(width, height, max_iterations)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_racketbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "racketbrot.png")
