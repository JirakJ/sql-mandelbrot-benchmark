"""
PonyBrot - Pony (actor-model, LLVM-compiled) Mandelbrot via a persistent
worker process.

ponyc compilation and worker startup happen at import time (untimed).
run_ponybrot only writes a request line and reads raw uint16 pixels back;
the Pony runtime work-steals the row actors across all cores.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_DIR, "ponybrot")
_SRC = os.path.join(_PKG, "main.pony")
_BIN = os.path.join(_DIR, "ponybrot_bin")
_PONYC = "/opt/homebrew/bin/ponyc"


def _build():
    """Compile the worker if the binary is missing or stale."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return
    err = b""
    for extra in (["--cpu", "apple-m4"], []):
        res = subprocess.run(
            [_PONYC] + extra + ["-o", _DIR, "-b", "ponybrot_bin", _PKG],
            capture_output=True,
        )
        if res.returncode == 0:
            return
        err = res.stdout + res.stderr
    raise RuntimeError("ponybrot: ponyc build failed:\n" + err.decode())


_build()

# One persistent worker for the whole benchmark; stderr is discarded.
_proc = subprocess.Popen(
    [_BIN],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_ponybrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Pony worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("ponybrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check + one full frame so scheduler
# threads and memory paths are hot before timing.
run_ponybrot(16, 16, 8)
run_ponybrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_ponybrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "ponybrot.png")
