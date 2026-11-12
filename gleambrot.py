"""
GleamBrot - Gleam (typed BEAM) Mandelbrot via a persistent worker process.

`gleam build` and BEAM startup happen at import time (untimed).
run_gleambrot only writes a request line and reads raw uint16 pixels back.
The kernel is pure Gleam (tail-recursive escape loop, cardioid/period-2
early-outs, append-optimised bit-array rows, y-axis mirror); a minimal
Erlang FFI does binary stdio and spawns one process per row band.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import glob
import os
import shutil
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_DIR, "gleambrot_src")
_SOURCES = [
    os.path.join(_SRC_DIR, "gleam.toml"),
    os.path.join(_SRC_DIR, "src", "gleambrot.gleam"),
    os.path.join(_SRC_DIR, "src", "gleambrot_ffi.erl"),
]
_BEAM = os.path.join(_SRC_DIR, "build", "dev", "erlang", "gleambrot", "ebin", "gleambrot.beam")

_GLEAM = shutil.which("gleam") or "/opt/homebrew/bin/gleam"
_ERL = shutil.which("erl") or "/opt/homebrew/bin/erl"


def _build():
    """gleam build if the .beam is missing or stale."""
    if os.path.exists(_BEAM) and os.path.getmtime(_BEAM) >= max(
        os.path.getmtime(s) for s in _SOURCES
    ):
        return
    res = subprocess.run([_GLEAM, "build"], cwd=_SRC_DIR, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError("gleambrot: gleam build failed:\n" + res.stderr.decode())


_build()

# One persistent BEAM for the whole benchmark; stderr swallowed.
_cmd = [_ERL, "-noshell"]
for _pa in sorted(glob.glob(os.path.join(_SRC_DIR, "build", "dev", "erlang", "*", "ebin"))):
    _cmd += ["-pa", _pa]
_cmd += ["-s", "gleambrot", "main"]
_proc = subprocess.Popen(
    _cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_gleambrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Gleam worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("gleambrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then a full frame so the BEAM
# scheduler pool and allocator arenas are hot before timing starts.
run_gleambrot(16, 16, 8)
run_gleambrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_gleambrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "gleambrot.png")
