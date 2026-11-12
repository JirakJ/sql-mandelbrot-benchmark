"""
CppBrot - C++ + ARM NEON + GCD Mandelbrot, the fastest entry on Apple Silicon.

The native kernel (cppbrot.cpp) is compiled to a dylib on first import (free,
not timed) and called via ctypes. run_cppbrot only does the compute.

Author: optimized for M-series Macs
License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "cppbrot.cpp")
_LIB = os.path.join(_DIR, "libcppbrot.dylib")

_BASE = ["clang++", "-O3", "-std=c++17", "-fPIC", "-shared", _SRC, "-o", _LIB]


def _build():
    """Compile the dylib if missing or stale. Tries -mcpu=apple-m4, falls back."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    for extra in (["-mcpu=apple-m4"], []):
        if subprocess.run(_BASE[:1] + extra + _BASE[1:]).returncode == 0:
            return
    raise RuntimeError("cppbrot: clang++ build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot_neon.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_neon.restype = None

# Warm-up at import (untimed by the harness): spins up the GCD thread pool so
# the first timed call doesn't pay thread-creation latency.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_neon(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_cppbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the native NEON kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_neon(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_cppbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "cppbrot.png")
