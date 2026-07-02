"""
PascalBrot - Free Pascal float32 scalar Mandelbrot via a native dylib.

The kernel (pascalbrot.pas) keeps a persistent thread pool over an atomic
row counter and exploits y-axis symmetry. Compiled with fpc on first import
(free, not timed) and called via ctypes. run_pascalbrot only does compute.

Author: optimized for M-series Macs
License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "pascalbrot.pas")
_LIB = os.path.join(_DIR, "libpascalbrot.dylib")

_FPC = "/opt/homebrew/bin/fpc" if os.path.exists("/opt/homebrew/bin/fpc") else "fpc"
_BASE = [_FPC, "-O4", "-CX", "-XX", "-CF32", _SRC, "-o" + _LIB]


def _build():
    """Compile the dylib if missing or stale. Tries -CpARMV8, falls back."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    res = None
    for extra in (["-CpARMV8"], []):
        res = subprocess.run(
            _BASE[:1] + extra + _BASE[1:], capture_output=True, cwd=_DIR
        )
        if res.returncode == 0:
            return
    raise RuntimeError(
        "pascalbrot: fpc build failed:\n"
        + res.stdout.decode(errors="replace")
        + res.stderr.decode(errors="replace")
    )


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot_pascal.argtypes = [
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_pascal.restype = None

# Warm-up at import (untimed): spins up the persistent thread pool.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_pascal(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_pascalbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the Pascal kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_pascal(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_pascalbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "pascalbrot.png")
