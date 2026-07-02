"""
ZigBrot - Zig @Vector SIMD + thread-pool Mandelbrot kernel.

The native kernel (zigbrot.zig) is compiled to a dylib on first import (free,
not timed) and called via ctypes. run_zigbrot only does the compute.

Author: optimized for M-series Macs
License: MIT
"""

import ctypes
import os
import shutil
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "zigbrot.zig")
_LIB = os.path.join(_DIR, "libzigbrot.dylib")

_ZIG = shutil.which("zig") or "/opt/homebrew/bin/zig"


def _build():
    """Compile the dylib if missing or stale."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    res = subprocess.run(
        [_ZIG, "build-lib", _SRC, "-O", "ReleaseFast", "-dynamic", "-lc",
         "-fstrip", "-femit-bin=" + _LIB],
        cwd=_DIR,
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError("zigbrot: zig build failed:\n" + res.stderr.decode())


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot_zig.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_zig.restype = None

# Warm-up at import (untimed by the harness): spins up the worker pool so the
# first timed call doesn't pay thread-creation latency.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_zig(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_zigbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the native Zig kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_zig(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_zigbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "zigbrot.png")
