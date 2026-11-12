"""
Hybrid4Brot - CPU and GPU computing one Mandelbrot frame concurrently, GPU
driven through the Metal 4 command model.

Same concurrent CPU+GPU design as hybridbrot.py (ext_vector float8 CPU kernel +
Metal compute GPU kernel writing disjoint row ranges of one shared buffer), but
the GPU half uses the WWDC 2025/2026 Metal 4 command API and stays async so the
CPU works while the GPU renders. Split between engines is calibrated once at
import (untimed). The optimal GPU share may differ from the Metal 3 hybrid, so
the sweep is run independently here.

License: MIT
"""

import ctypes
import os
import subprocess
import time

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "hybrid4brot.mm")
_LIB = os.path.join(_DIR, "libhybrid4brot.dylib")


def _build():
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    for extra in (["-mcpu=apple-m4"], []):
        cmd = ["clang++", "-O3", "-std=c++17", "-fobjc-arc", "-fPIC", "-shared",
               *extra, "-framework", "Metal", "-framework", "Foundation",
               _SRC, "-o", _LIB]
        if subprocess.run(cmd).returncode == 0:
            return
    raise RuntimeError("hybrid4brot: clang++ build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.hybrid4_init.restype = ctypes.c_int
_lib.mandelbrot_hybrid4.argtypes = [ctypes.c_int] * 4
_lib.mandelbrot_hybrid4.restype = ctypes.POINTER(ctypes.c_uint16)

if not _lib.hybrid4_init():
    raise ImportError("hybrid4brot: no Metal 4 device / shader compile failed")

# Calibrate the GPU/CPU split once at import (untimed). The Metal 4 GPU submit is
# cheaper, so the optimal GPU share can shift; sweep a slightly wider set.
for _s in (56, 64, 72, 76):  # warm across the range
    _lib.mandelbrot_hybrid4(1400, 800, 256, _s)
_best = None
for _s in (52, 56, 60, 64, 68, 70, 72, 74, 76, 80):
    _ts = []
    for _ in range(11):
        _t = time.perf_counter()
        _lib.mandelbrot_hybrid4(1400, 800, 256, _s)
        _ts.append(time.perf_counter() - _t)
    _m = sorted(_ts)[len(_ts) // 2]  # median
    if _best is None or _m < _best[1]:
        _best = (_s, _m)
_SPLIT = _best[0]


def run_hybrid4brot(width, height, max_iterations):
    """CPU+GPU concurrent frame (Metal 4 GPU). Returns (h, w) uint16 view."""
    ptr = _lib.mandelbrot_hybrid4(width, height, max_iterations, _SPLIT)
    return np.ctypeslib.as_array(ptr, shape=(height, width))


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing (hybrid4, split={_SPLIT}% GPU)...")
    result = run_hybrid4brot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "hybrid4brot.png")
