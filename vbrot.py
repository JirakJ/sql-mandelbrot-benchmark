"""
VBrot - V (vlang) Mandelbrot via a native shared library.

vbrot.v is compiled to libvbrot.dylib at import time (untimed) with
`v -prod -shared -gc none` and called through ctypes. The V kernel runs a
16-lane f32 escape loop (auto-vectorized to NEON by V's clang backend) on a
persistent semaphore-driven pool of one thread per CPU core (created during
the untimed warm-up call). run_vbrot only does the compute.

Author: optimized for M-series Macs
License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "vbrot.v")
_LIB = os.path.join(_DIR, "libvbrot.dylib")
_V = "/opt/homebrew/bin/v"

_BASE = [_V, "-prod", "-shared", "-gc", "none", "-no-bounds-checking",
         "-enable-globals", _SRC, "-o", _LIB]


def _build():
    """Compile the dylib if missing or stale. Tries -mcpu=apple-m4, falls back."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    for extra in (["-cflags", "-mcpu=apple-m4"], []):
        if subprocess.run(_BASE + extra, capture_output=True).returncode == 0:
            return
    raise RuntimeError("vbrot: v build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot_v.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_v.restype = None

# Warm-up at import (untimed by the harness).
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_v(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_vbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the V kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_v(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_vbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "vbrot.png")
