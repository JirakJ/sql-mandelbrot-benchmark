"""
Metal4Brot - Mandelbrot on the Apple Silicon GPU via the Metal 4 command model.

Mirrors metalbrot.py, but the host (metal4brot.mm) drives the dispatch through the
WWDC 2025/2026 Metal 4 command API (MTL4CommandQueue / MTL4CommandBuffer /
MTL4CommandAllocator / MTL4ArgumentTable / MTLResidencySet + MTLSharedEvent) instead
of the Metal 3 command queue. Same MSL kernel and math as metalbrot; the point is to
measure whether the new command model reduces fixed submit/execute latency for a tiny
single dispatch.

The dylib is compiled on first import (untimed, mtime-guarded) and called via ctypes.
Device/queue/pipeline/residency setup and a warm-up dispatch happen at import;
run_metal4brot only does the compute + one memcpy view.

Author: optimized for M-series Macs (Metal 4)
License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "metal4brot.mm")
_LIB = os.path.join(_DIR, "libmetal4brot.dylib")


def _build():
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    cmd = [
        "clang++", "-O3", "-std=c++17", "-fobjc-arc", "-fPIC", "-shared",
        "-framework", "Metal", "-framework", "Foundation",
        _SRC, "-o", _LIB,
    ]
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError("metal4brot: clang++ build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.metal4_init.restype = ctypes.c_int
_lib.mandelbrot_metal4.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_metal4.restype = None
_lib.mandelbrot_metal4_ptr.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.mandelbrot_metal4_ptr.restype = ctypes.POINTER(ctypes.c_uint16)

if not _lib.metal4_init():
    raise ImportError("metal4brot: no Metal 4 device / shader compile failed")

# Warm-up dispatch at import (untimed): first command buffer on a fresh queue
# pays GPU wake-up + pipeline residency costs.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_metal4(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_metal4brot(width, height, max_iterations):
    """Compute the Mandelbrot set on the GPU via Metal 4. Returns (h, w) uint16.

    Unified memory zero-copy: the returned array is a view into the shared
    MTLBuffer (valid until the next call overwrites it).
    """
    ptr = _lib.mandelbrot_metal4_ptr(width, height, max_iterations)
    return np.ctypeslib.as_array(ptr, shape=(height, width))


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_metal4brot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "metal4brot.png")
