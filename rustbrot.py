"""
RustBrot - Rust + ARM NEON + rayon Mandelbrot entry.

The native kernel (rustbrot/src/lib.rs) is compiled to a dylib at import time
(free, not timed) and called via ctypes. run_rustbrot only does the compute.

License: MIT
"""

import ctypes
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_CRATE = os.path.join(_DIR, "rustbrot")
_MANIFEST = os.path.join(_CRATE, "Cargo.toml")
_SRC = os.path.join(_CRATE, "src", "lib.rs")
_LIB = os.path.join(_CRATE, "target", "release", "librustbrot.dylib")


def _build():
    """Compile the dylib if missing or stale."""
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= max(
        os.path.getmtime(_SRC), os.path.getmtime(_MANIFEST)
    ):
        return
    env = dict(os.environ)
    env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " -C target-cpu=native").strip()
    cmd = ["cargo", "build", "--release", "--manifest-path", _MANIFEST]
    if subprocess.run(cmd, env=env).returncode != 0:
        raise RuntimeError("rustbrot: cargo build failed")


_build()
_lib = ctypes.CDLL(_LIB)
_lib.mandelbrot_rust.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint16),
]
_lib.mandelbrot_rust.restype = None

# Warm-up at import (untimed): spins up the rayon thread pool so the first
# timed call doesn't pay thread-creation latency.
_warm = np.empty((16, 16), dtype=np.uint16)
_lib.mandelbrot_rust(16, 16, 8, _warm.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
del _warm


def run_rustbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the native NEON kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _lib.mandelbrot_rust(
        width, height, max_iterations,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_rustbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "rustbrot.png")
