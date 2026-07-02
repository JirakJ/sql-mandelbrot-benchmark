"""
HaskellBrot - GHC (threaded RTS) Mandelbrot via a persistent worker process.

Compilation and worker spawn happen at import time (untimed).
run_haskellbrot only writes a request line and reads raw uint16 pixels back.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "haskellbrot.hs")
_BIN = os.path.join(_DIR, "haskellbrot_bin")


def _build():
    """Compile the worker if the binary is missing or stale."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return
    res = subprocess.run(
        ["ghc", "-O2", "-threaded", "-rtsopts", "-o", _BIN, _SRC],
        cwd=_DIR,
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError("haskellbrot: ghc failed:\n" + res.stderr.decode())


_build()

# One persistent worker for the whole benchmark; all its logging goes to stderr.
_proc = subprocess.Popen(
    [_BIN, "+RTS", "-N"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_haskellbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Haskell worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("haskellbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then a full frame so the RTS
# spins up its capabilities before timing.
run_haskellbrot(16, 16, 8)
run_haskellbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_haskellbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "haskellbrot.png")
