"""
HybridBrot - CPU and GPU computing one Mandelbrot frame concurrently.

Unified memory lets the Metal GPU and the ext_vector CPU kernel write disjoint
row ranges of the same shared buffer at the same time — the fastest entry in
the suite. Split between engines is calibrated once at import (untimed).

License: MIT
"""

import ctypes
import os
import subprocess
import time

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "hybridbrot.mm")
_LIB = os.path.join(_DIR, "libhybridbrot.dylib")


def _build():
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    for extra in (["-mcpu=apple-m4"], []):
        cmd = ["clang++", "-O3", "-std=c++17", "-fobjc-arc", "-fPIC", "-shared",
               *extra, "-framework", "Metal", "-framework", "Foundation",
               _SRC, "-o", _LIB]
        if subprocess.run(cmd).returncode == 0:
            return
    raise RuntimeError("hybridbrot: clang++ build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.hybrid_init.restype = ctypes.c_int
_lib.mandelbrot_hybrid.argtypes = [ctypes.c_int] * 4
_lib.mandelbrot_hybrid.restype = ctypes.POINTER(ctypes.c_uint16)
_lib.mandelbrot_hybrid_strided.argtypes = [ctypes.c_int] * 4
_lib.mandelbrot_hybrid_strided.restype = ctypes.POINTER(ctypes.c_uint16)

if not _lib.hybrid_init():
    raise ImportError("hybridbrot: no Metal device")

# Calibrate the GPU/CPU split once at import (untimed). Contiguous split wins
# over strided assignment (strided costs the GPU row locality — measured, kept
# in the source as a documented dead end).
_lib.mandelbrot_hybrid(1400, 800, 256, 56)  # warm both engines
_best = None
for _s in (46, 50, 52, 54, 56, 60, 64):
    _ts = []
    for _ in range(5):
        _t = time.perf_counter()
        _lib.mandelbrot_hybrid(1400, 800, 256, _s)
        _ts.append(time.perf_counter() - _t)
    _m = sorted(_ts)[1]
    if _best is None or _m < _best[1]:
        _best = (_s, _m)
_SPLIT = _best[0]


def run_hybridbrot(width, height, max_iterations):
    """CPU+GPU concurrent frame. Returns (h, w) uint16 view of shared memory."""
    ptr = _lib.mandelbrot_hybrid(width, height, max_iterations, _SPLIT)
    return np.ctypeslib.as_array(ptr, shape=(height, width))


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing (hybrid, split={_SPLIT}% GPU)...")
    result = run_hybridbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "hybridbrot.png")
