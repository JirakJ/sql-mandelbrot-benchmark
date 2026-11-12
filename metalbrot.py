"""
MetalBrot - Mandelbrot on the Apple Silicon GPU via a Metal compute shader.

The host (metalbrot.mm) is compiled to a dylib on first import (untimed) and
called via ctypes; the Metal shader itself is compiled at runtime from source,
so no metallib toolchain is needed. Device/queue/pipeline setup and a warm-up
dispatch happen at import; run_metalbrot only does the compute.

Author: optimized for M-series Macs
License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "metalbrot.mm")
_LIB = os.path.join(_DIR, "libmetalbrot.dylib")


def _build():
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    cmd = [
        "clang++", "-O3", "-std=c++17", "-fobjc-arc", "-fPIC", "-shared",
        "-framework", "Metal", "-framework", "Foundation",
        _SRC, "-o", _LIB,
    ]
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError("metalbrot: clang++ build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.metal_init.restype = ctypes.c_int
_lib.mandelbrot_metal.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_metal.restype = None
_lib.mandelbrot_metal_ptr.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.mandelbrot_metal_ptr.restype = ctypes.POINTER(ctypes.c_uint16)

if not _lib.metal_init():
    raise ImportError("metalbrot: no Metal device / shader compile failed")

# Warm-up dispatch at import (untimed): first command buffer on a fresh queue
# pays GPU wake-up + pipeline residency costs.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_metal(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_metalbrot(width, height, max_iterations):
    """Compute the Mandelbrot set on the GPU. Returns (h, w) uint16.

    Unified memory zero-copy: the returned array is a view into the shared
    MTLBuffer (valid until the next call overwrites it).
    """
    ptr = _lib.mandelbrot_metal_ptr(width, height, max_iterations)
    return np.ctypeslib.as_array(ptr, shape=(height, width))


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_metalbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "metalbrot.png")
