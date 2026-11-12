"""
JuliaBrot - Julia (multithreaded, Float64 scalar) Mandelbrot via a persistent worker.

One Julia process is spawned at import time (untimed) with --threads=auto and
JIT-warmed by a tiny render, so run_juliabrot only pays for pipe I/O + compute.
Protocol: "<w> <h> <iters>\n" on stdin -> w*h*2 bytes of uint16 LE on stdout.

License: MIT
"""

import atexit
import os
import shutil
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "juliabrot.jl")

_JULIA = shutil.which("julia")
if _JULIA is None:
    raise RuntimeError("juliabrot: julia executable not found on PATH")

_proc = subprocess.Popen(
    [_JULIA, "--threads=auto", "-O3", "--startup-file=no", _SRC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,  # binary pipe; worker logs (if any) go to stderr
    cwd=_DIR,
)
atexit.register(_proc.kill)


def _render(width, height, max_iterations):
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    got, total = 0, out.nbytes
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("juliabrot: worker exited unexpectedly")
        got += n
    return out


# Warm-up at import (untimed): triggers Julia JIT compilation of the kernel.
_render(16, 16, 8)


def run_juliabrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Julia worker. Returns (h, w) uint16."""
    return _render(width, height, max_iterations)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_juliabrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "juliabrot.png")
