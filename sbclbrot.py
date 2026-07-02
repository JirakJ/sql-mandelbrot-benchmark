"""
SbclBrot - Common Lisp (SBCL) multithreaded Mandelbrot via a persistent worker.

sbcl --script compiles the kernel at import time (untimed). run_sbclbrot only
writes a request line and reads raw uint16 LE pixels back over a pipe.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "sbclbrot.lisp")

# One persistent SBCL image for the whole benchmark; stderr swallowed.
_proc = subprocess.Popen(
    ["sbcl", "--script", _SRC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env={**os.environ, "SBCLBROT_THREADS": str(os.cpu_count() or 8)},
)
atexit.register(_proc.terminate)


def run_sbclbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the SBCL worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("sbclbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then one full frame so threads
# and GC pages are hot before timing.
run_sbclbrot(16, 16, 8)
run_sbclbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_sbclbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "sbclbrot.png")
