"""
SwiftBrot - Swift + SIMD8<Float> + GCD Mandelbrot.

The native kernel (swiftbrot.swift) is compiled to a dylib on first import
(free, not timed) and called via ctypes. run_swiftbrot only does the compute.

License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "swiftbrot.swift")
_LIB = os.path.join(_DIR, "libswiftbrot.dylib")

_CMD = ["swiftc", "-O", "-wmo", "-emit-library", _SRC, "-o", _LIB]


def _build():
    """Compile the dylib if missing or stale."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    if subprocess.run(_CMD).returncode != 0:
        raise RuntimeError("swiftbrot: swiftc build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot_swift.argtypes = [
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_swift.restype = None

# Warm-up at import (untimed by the harness): spins up the GCD thread pool.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_swift(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_swiftbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the Swift SIMD kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_swift(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    from utils import save_mandelbrot_image

    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_swiftbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "swiftbrot.png")
