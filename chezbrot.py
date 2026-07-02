"""
ChezBrot - Chez Scheme Mandelbrot via one persistent threaded worker.

The homebrew Chez 10.x build is threaded, so a single chez process
(chezbrot.ss) is spawned at import time (untimed) with a fork-thread pool
sized to all CPU cores. run_chezbrot only writes a request line and reads
the raw uint16 frame back over the pipe.

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
_SRC = os.path.join(_DIR, "chezbrot.ss")

_CHEZ = shutil.which("chez") or shutil.which("chezscheme")
if _CHEZ is None and os.path.exists("/opt/homebrew/bin/chez"):
    _CHEZ = "/opt/homebrew/bin/chez"
if _CHEZ is None:
    raise RuntimeError("chezbrot: chez scheme not found")

_NT = min(14, os.cpu_count() or 1)

_proc = subprocess.Popen(
    [_CHEZ, "--script", _SRC, str(_NT)],
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
            raise RuntimeError("chezbrot: chez worker died")
        got += k
    return out


# Warm-up at import (untimed): protocol check, then a full-size frame so the
# thread pool and heap are primed before timing starts.
_request(16, 16, 8)
_request(1400, 800, 256)


def run_chezbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Chez worker. Returns (h, w) uint16."""
    return _request(width, height, max_iterations)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_chezbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "chezbrot.png")
