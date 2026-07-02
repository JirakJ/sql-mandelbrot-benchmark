"""
GoBrot - Go c-shared Mandelbrot kernel called via ctypes.

The dylib is built from gobrot_src/ on first import (free, not timed).
run_gobrot only does the compute.

License: MIT
"""

import ctypes
import os
import shutil
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_DIR, "gobrot_src")
_SRC = os.path.join(_SRC_DIR, "gobrot.go")
_LIB = os.path.join(_DIR, "libgobrot.dylib")

_GO = "/opt/homebrew/bin/go"
if not os.path.exists(_GO):
    _GO = shutil.which("go") or "go"


def _build():
    """Compile the dylib if missing or stale."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    r = subprocess.run(
        [_GO, "build", "-buildmode=c-shared", "-o", _LIB, "."],
        cwd=_SRC_DIR,
    )
    if r.returncode != 0:
        raise RuntimeError("gobrot: go build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.MandelbrotGo.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.MandelbrotGo.restype = None

# Warm-up at import (untimed): starts the Go runtime and its worker threads.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.MandelbrotGo(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_gobrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the Go kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.MandelbrotGo(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_gobrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "gobrot.png")
