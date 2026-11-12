"""
JanetBrot - Janet Mandelbrot via a persistent multi-threaded worker process.

Janet has real OS threads (ev/spawn-thread); one worker process computes the
top half of the image on a thread pool, mirrors the bottom half, and streams
raw uint16 LE pixels back. Process spawn and warm-up happen at import time.

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
_SRC = os.path.join(_DIR, "janetbrot.janet")

_JANET = shutil.which("janet") or "/opt/homebrew/bin/janet"
if not os.path.exists(_JANET):
    raise RuntimeError("janetbrot: janet not found")

_NW = str(min(16, os.cpu_count() or 1))

# One persistent worker for the whole benchmark; logging to stderr only.
_proc = subprocess.Popen(
    [_JANET, _SRC, _NW],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_janetbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Janet worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("janetbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then a full-size frame so the
# thread pool and VM are hot before timing starts.
run_janetbrot(16, 16, 8)
run_janetbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_janetbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "janetbrot.png")
