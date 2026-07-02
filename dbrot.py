"""
DBrot - D (LDC) + float4 SIMD (NEON) + std.parallelism Mandelbrot.

The native kernel (dbrot.d) is compiled to a dylib on first import (free,
not timed) and called via ctypes. run_dbrot only does the compute.

License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "dbrot.d")
_LIB = os.path.join(_DIR, "libdbrot.dylib")

_LDC = "/opt/homebrew/bin/ldc2"
_CMD = [
    _LDC, "-O3", "-release", "-boundscheck=off", "-mcpu=native",
    "--fp-contract=fast", "-shared", _SRC, "-of=" + _LIB,
]


def _build():
    """Compile the dylib if missing or stale."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    if subprocess.run(_CMD).returncode != 0:
        raise RuntimeError("dbrot: ldc2 build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.dbrot_init.argtypes = []
_lib.dbrot_init.restype = None
_lib.dbrot_init()  # rt_init + thread pool sizing (cores - 1 workers + caller)
_lib.mandelbrot_d.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_d.restype = None

# Warm-up at import (untimed): spins up the taskPool worker threads.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_d(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_dbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the native D kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_d(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_dbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "dbrot.png")
