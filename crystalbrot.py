"""
CrystalBrot - Crystal (multi-threaded execution contexts) Mandelbrot via a
persistent worker process.

Compilation and worker startup happen at import time (untimed). run_crystalbrot
only writes a request line and reads raw uint16 pixels back.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "crystalbrot.cr")
_BIN = os.path.join(_DIR, "crystalbrot_bin")

_BASE = ["crystal", "build", "--release", "-Dpreview_mt", "-Dexecution_context",
         _SRC, "-o", _BIN]


def _build():
    """Compile the worker if the binary is missing or stale."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return
    err = b""
    for extra in (["--mcpu", "apple-m4"], []):
        res = subprocess.run(_BASE[:3] + extra + _BASE[3:], capture_output=True)
        if res.returncode == 0:
            return
        err = res.stderr
    raise RuntimeError("crystalbrot: crystal build failed:\n" + err.decode())


_build()

# One persistent worker for the whole benchmark; all its logging goes to stderr.
_env = dict(os.environ, CRYSTAL_WORKERS="14")
_proc = subprocess.Popen(
    [_BIN],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env=_env,
)
atexit.register(_proc.terminate)


def run_crystalbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Crystal worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("crystalbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check + one full frame so threads and
# memory paths are hot before timing.
run_crystalbrot(16, 16, 8)
run_crystalbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_crystalbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "crystalbrot.png")
