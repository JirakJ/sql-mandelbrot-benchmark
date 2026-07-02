"""
AsmBrot - hand-written AArch64 NEON assembly Mandelbrot kernel + C shim (GCD).

The dylib is built from asmbrot.s + asmbrot_shim.c on first import (untimed)
and called via ctypes. run_asmbrot only does the compute.

Author: optimized for M-series Macs
License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_C = os.path.join(_DIR, "asmbrot_shim.c")
_SRC_S = os.path.join(_DIR, "asmbrot.s")
_LIB = os.path.join(_DIR, "libasmbrot.dylib")


def _build():
    """Compile the dylib if missing or stale."""
    if os.path.exists(_LIB):
        lib_m = os.path.getmtime(_LIB)
        if lib_m >= os.path.getmtime(_SRC_C) and lib_m >= os.path.getmtime(_SRC_S):
            return
    cmd = ["clang", "-O3", "-fPIC", "-shared", _SRC_C, _SRC_S, "-o", _LIB]
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError("asmbrot: clang build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot_asm.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_asm.restype = None

# Warm-up at import (untimed): spins up the GCD thread pool.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_asm(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_asmbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the NEON asm kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_asm(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_asmbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "asmbrot.png")
