"""
ISPCBrot - Intel ISPC (SPMD SIMD) Mandelbrot, NEON i32x8, compiled at import.

An ISPC kernel (ispcbrot.ispc) computes one image row 8 lanes wide; a tiny C
shim (ispcbrot_shim.c) fans rows across all cores with libdispatch and mirrors
the bottom half. Both are compiled to libispcbrot.dylib on first import (free,
not timed) and driven via ctypes. run_ispcbrot only does compute.

Author: optimized for M-series Macs
License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "ispcbrot.ispc")
_SHIM = os.path.join(_DIR, "ispcbrot_shim.c")
_OBJ = os.path.join(_DIR, "ispcbrot.o")
_HDR = os.path.join(_DIR, "ispcbrot.h")
_LIB = os.path.join(_DIR, "libispcbrot.dylib")

_ISPC = "/opt/homebrew/bin/ispc"


def _stale(target, *deps):
    if not os.path.exists(target):
        return True
    tmt = os.path.getmtime(target)
    return any(os.path.getmtime(d) > tmt for d in deps)


def _build():
    """Compile the ISPC kernel + C shim into a dylib if missing or stale."""
    if not _stale(_LIB, _SRC, _SHIM):
        return
    subprocess.run(
        [_ISPC, "-O3", "--target=neon-i32x8", "--arch=aarch64", "--pic",
         "--opt=fast-math", _SRC, "-o", _OBJ, "-h", _HDR],
        check=True,
    )
    subprocess.run(
        ["clang", "-O3", "-dynamiclib", "-o", _LIB, _OBJ, _SHIM],
        check=True,
    )


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot.restype = None


def run_ispcbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the ISPC kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


# Warm-up at import (untimed): spins up the GCD thread pool, steady-state timing.
run_ispcbrot(16, 16, 8)
run_ispcbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_ispcbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "ispcbrot.png")
