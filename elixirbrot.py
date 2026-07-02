"""
ElixirBrot - Elixir/BEAM Mandelbrot via a persistent worker process.

BEAM startup and module compilation happen at import time (untimed).
run_elixirbrot only writes a request line and reads raw uint16 pixels back.
BEAM floats are boxed, so this is an honest "as fast as idiomatic Elixir
goes" entry: recursive escape loop, Task.async_stream row bands, iodata rows.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import shutil
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "elixirbrot.exs")

_ELIXIR = shutil.which("elixir")
if _ELIXIR is None:
    raise RuntimeError("elixirbrot: elixir not found on PATH")

# One persistent BEAM for the whole benchmark; stderr swallowed.
_proc = subprocess.Popen(
    [_ELIXIR, _SRC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_elixirbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Elixir worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("elixirbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then a full frame so the BEAM
# JIT, scheduler pool, and allocator arenas are hot before timing starts.
run_elixirbrot(16, 16, 8)
run_elixirbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_elixirbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "elixirbrot.png")
