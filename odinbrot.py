"""
OdinBrot - Odin + #simd[8]f32 + persistent thread pool Mandelbrot.

The native kernel (odinbrot.odin) is compiled to a dylib on first import (free,
not timed) and called via ctypes. run_odinbrot only does the compute.

License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "odinbrot.odin")
_LIB = os.path.join(_DIR, "libodinbrot.dylib")

_BASE = [
    "odin", "build", _SRC, "-file", "-build-mode:shared",
    "-o:speed", "-no-bounds-check", f"-out:{_LIB}",
]


def _build():
    """Compile the dylib if missing or stale. Tries -microarch:native, falls back."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    for extra in (["-microarch:native"], []):
        if subprocess.run(_BASE + extra).returncode == 0:
            return
    raise RuntimeError("odinbrot: odin build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot_odin.argtypes = [
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_odin.restype = None

# Warm-up at import (untimed by the harness): spawns the persistent worker
# threads so the first timed call doesn't pay thread-creation latency.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_odin(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_odinbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the native Odin SIMD kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_odin(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_odinbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "odinbrot.png")
