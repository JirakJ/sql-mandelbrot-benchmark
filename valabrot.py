"""
ValaBrot - Vala (GLib) Mandelbrot via a C-ABI dylib.

valac compiles valabrot.vala to libvalabrot.dylib at import (untimed) and the
kernel is called through ctypes. All cores via GLib threads over an atomic
row counter; y-mirror inside the kernel. run_valabrot only does the compute.

License: MIT
"""

import ctypes
import os
import subprocess
import tempfile

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "valabrot.vala")
_LIB = os.path.join(_DIR, "libvalabrot.dylib")
_VALAC = "/opt/homebrew/bin/valac"

_BASE = ["--library=valabrot", "-X", "-O3", "-X", "-fPIC", "-X", "-shared",
         _SRC, "-o", _LIB]


def _build():
    """Compile the dylib if missing or stale. Tries -mcpu=apple-m4, falls back."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    with tempfile.TemporaryDirectory() as tmp:  # keeps .vapi side files out of the repo
        for extra in (["-X", "-mcpu=apple-m4"], []):
            if subprocess.run([_VALAC] + extra + _BASE, cwd=tmp).returncode == 0:
                return
    raise RuntimeError("valabrot: valac build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot_vala.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_vala.restype = None

# Warm-up at import (untimed): touches the code path and thread creation.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_vala(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_valabrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the Vala kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_vala(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_valabrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "valabrot.png")
