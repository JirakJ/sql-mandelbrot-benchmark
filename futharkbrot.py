"""
FutharkBrot - Mandelbrot via Futhark's multicore backend.

futhark multicore --library compiles futharkbrot.fut to C with a C API; that C
is built into a dylib at import (untimed) and driven via ctypes. The kernel
computes only the top half (y-axis symmetry); numpy mirrors the bottom.

License: MIT
"""

import ctypes
import os
import subprocess
import tempfile

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "futharkbrot.fut")
_LIB = os.path.join(_DIR, "libfutharkbrot.dylib")


def _build():
    """futhark multicore -> C (in a temp dir), then clang -> dylib. Skip if fresh."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, "futharkbrot")
        subprocess.run(
            ["futhark", "multicore", "--library", _SRC, "-o", stem], check=True
        )
        base = ["clang", "-O3", "-fPIC", "-shared", stem + ".c",
                "-o", _LIB, "-lpthread", "-lm"]
        for extra in (["-mcpu=apple-m4"], []):
            if subprocess.run(base[:1] + extra + base[1:]).returncode == 0:
                return
    raise RuntimeError("futharkbrot: clang build failed")


_build()
_lib = ctypes.CDLL(_LIB)

_lib.futhark_context_config_new.restype = ctypes.c_void_p
_lib.futhark_context_new.argtypes = [ctypes.c_void_p]
_lib.futhark_context_new.restype = ctypes.c_void_p
_lib.futhark_entry_mandel.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int32,
]
_lib.futhark_entry_mandel.restype = ctypes.c_int
_lib.futhark_context_sync.argtypes = [ctypes.c_void_p]
_lib.futhark_context_sync.restype = ctypes.c_int
_lib.futhark_values_u16_2d.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.futhark_values_u16_2d.restype = ctypes.c_int
_lib.futhark_free_u16_2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.futhark_free_u16_2d.restype = ctypes.c_int

_cfg = _lib.futhark_context_config_new()
_ctx = _lib.futhark_context_new(_cfg)  # num_threads defaults to all cores


def _compute(width, height, max_iterations):
    """Run the entry point, mirror in numpy, return (h, w) uint16."""
    half = (height + 1) // 2
    out = np.empty((height, width), dtype=np.uint16)
    arr = ctypes.c_void_p()
    if _lib.futhark_entry_mandel(
        _ctx, ctypes.byref(arr), width, height, max_iterations
    ):
        raise RuntimeError("futhark_entry_mandel failed")
    _lib.futhark_context_sync(_ctx)
    # Writes half*width u16 row-major, i.e. exactly out[:half].
    _lib.futhark_values_u16_2d(
        _ctx, arr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    )
    _lib.futhark_context_sync(_ctx)
    _lib.futhark_free_u16_2d(_ctx, arr)
    out[half:] = out[: height - half][::-1]  # mirror; skips odd middle row
    return out


_compute(16, 16, 8)  # warm-up: spin up the thread pool at import


def run_futharkbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the Futhark multicore kernel."""
    return _compute(width, height, max_iterations)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_futharkbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "futharkbrot.png")
