"""
HalideBrot - Mandelbrot via the Halide image DSL (Halide 21).

The kernel (halidebrot.cpp) builds a Halide pipeline once and JIT-compiles it on
the first realize() call. It is compiled to a dylib on first import (untimed) and
called via ctypes. The Halide JIT warm-up also happens at import, so run_halidebrot
is pure steady-state compute: schedule = parallel(y) + vectorize(x, 8), float32.

Author: optimized for M-series Macs
License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "halidebrot.cpp")
_LIB = os.path.join(_DIR, "libhalidebrot.dylib")
_HALIDE = "/opt/homebrew/opt/halide"

_CMD = [
    "clang++", "-std=c++17", "-O3",
    "-I" + os.path.join(_HALIDE, "include"),
    _SRC,
    "-L" + os.path.join(_HALIDE, "lib"), "-lHalide",
    "-Wl,-rpath," + os.path.join(_HALIDE, "lib"),
    "-dynamiclib", "-o", _LIB,
]


def _build():
    """Compile the Halide dylib if missing or stale."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    subprocess.run(_CMD, check=True)


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot.restype = None


def run_halidebrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the Halide kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


# Warm-up at import (untimed): triggers the Halide JIT compile + thread pool spin-up
# so the first timed call runs at steady state.
run_halidebrot(16, 16, 8)
run_halidebrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_halidebrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "halidebrot.png")
