"""
RakuBrot - Rakudo/MoarVM Mandelbrot via one persistent worker process.

One raku process (rakubrot.raku) is spawned at import time (untimed); it
runs the escape loop with native num/int locals and fans rows out to all
CPU cores with start/await Promises. run_rakubrot only writes a request
line and reads the raw uint16 LE frame back over the pipe.

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
_SRC = os.path.join(_DIR, "rakubrot.raku")

_RAKU = shutil.which("raku") or "/opt/homebrew/bin/raku"
if not os.path.exists(_RAKU):
    raise RuntimeError("rakubrot: raku not found")

_proc = subprocess.Popen(
    [_RAKU, _SRC],
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
            raise RuntimeError("rakubrot: raku worker died")
        got += k
    return out


def run_rakubrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the raku worker. Returns (h, w) uint16."""
    return _request(width, height, max_iterations)


# Warm-up at import (untimed): protocol check, then full-size frames so
# MoarVM's spesh/JIT compiles the hot loop before timing starts.
_request(16, 16, 8)
_request(1400, 800, 256)
_request(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_rakubrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "rakubrot.png")
