"""
ClojureBrot - Clojure (JVM) Mandelbrot via a persistent worker process.

JVM startup and JIT warm-up happen at import time (untimed). run_clojurebrot
only writes a request line and reads raw uint16 pixels back. The worker
parallelizes rows over a fixed thread pool and mirrors the bottom half.

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
_SRC = os.path.join(_DIR, "clojurebrot.clj")

_CLOJURE = shutil.which("clojure") or "/opt/homebrew/bin/clojure"
if not os.path.exists(_CLOJURE):
    raise RuntimeError("clojurebrot: clojure CLI not found")

# One persistent JVM for the whole benchmark; stderr swallowed (boxed-math warnings).
_proc = subprocess.Popen(
    [_CLOJURE, "-M", _SRC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_clojurebrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Clojure worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("clojurebrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then full-size frames so the
# JIT compiles the hot loop and the thread pool spins up before timing.
run_clojurebrot(16, 16, 8)
run_clojurebrot(1400, 800, 256)
run_clojurebrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_clojurebrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "clojurebrot.png")
