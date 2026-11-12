"""
NimBrot - Nim compiled to a C dylib (clang backend), threaded row kernel.

The native kernel (nimbrot.nim) is compiled to a dylib on first import (free,
not timed) and called via ctypes. run_nimbrot only does the compute.

License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "nimbrot.nim")
_LIB = os.path.join(_DIR, "libnimbrot.dylib")
_NIM = "/opt/homebrew/bin/nim"


def _build():
    """Compile the dylib if missing or stale. Tries -mcpu=apple-m4, falls back."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    base = [
        _NIM, "c", "--app:lib", "-d:danger", "--threads:on", "--mm:arc",
        f"--nimcache:{os.path.join(_DIR, 'nimcache')}", f"-o:{_LIB}", _SRC,
    ]
    for extra in (["--passC:-mcpu=apple-m4"], []):
        r = subprocess.run(
            base[:2] + extra + base[2:],
            cwd=_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if r.returncode == 0:
            return
    raise RuntimeError("nimbrot: nim build failed")


_build()
_lib = ctypes.CDLL(_LIB)
try:
    _lib.NimMain()  # init Nim runtime once
except AttributeError:
    pass
_lib.mandelbrot_nim.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_nim.restype = None

# Warm-up at import (untimed): spins up the thread pool.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_nim(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_nimbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the native Nim kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_nim(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_nimbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "nimbrot.png")
