"""
C3Brot - C3 (c3c/LLVM) float[<8>] SIMD + OS-thread Mandelbrot.

The native kernel (c3brot.c3) is compiled to a dylib on first import (free,
not timed) and called via ctypes. run_c3brot only does the compute.

License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "c3brot.c3")
_LIB = os.path.join(_DIR, "libc3brot.dylib")

_C3C = "/opt/homebrew/bin/c3c"
_CMD = [_C3C, "dynamic-lib", "-O5", "c3brot.c3", "-o", "libc3brot"]


def _build():
    """Compile the dylib if missing or stale."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    r = subprocess.run(
        _CMD, cwd=_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if r.returncode != 0 or not os.path.exists(_LIB):
        raise RuntimeError("c3brot: c3c build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot.restype = None


def run_c3brot(width, height, max_iterations):
    """Compute the Mandelbrot set via the native C3 SIMD kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


# Warm-up at import (untimed): spins up the thread pool so the first timed
# call runs at steady state.
run_c3brot(16, 16, 8)
run_c3brot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_c3brot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "c3brot.png")
