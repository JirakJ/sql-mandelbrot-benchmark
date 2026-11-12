"""
PerlBrot - Perl Mandelbrot via a pool of persistent worker processes.

Perl ithreads are heavyweight, so parallelism comes from N worker processes
spawned at import time (untimed). run_perlbrot splits the top half of the
image into one contiguous row band per worker, reads the bands back into the
numpy buffer, and mirrors the bottom half (y-axis symmetry).

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
_SRC = os.path.join(_DIR, "perlbrot.pl")

_PERL = shutil.which("perl")
if _PERL is None:
    raise RuntimeError("perlbrot: perl not found on PATH")

_NW = min(14, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_PERL, _SRC],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(_NW)
]


def _shutdown():
    for p in _procs:
        p.terminate()


atexit.register(_shutdown)


def run_perlbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Perl pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2
    base, rem = divmod(top, _NW)

    # One contiguous band per worker; skip workers with no rows.
    bands = []
    r0 = 0
    for i in range(_NW):
        n = base + (1 if i < rem else 0)
        if n:
            bands.append((_procs[i], r0, r0 + n))
            r0 += n

    for p, a, b in bands:
        p.stdin.write(f"{width} {height} {max_iterations} {a} {b}\n".encode())
        p.stdin.flush()

    for p, a, b in bands:
        view = memoryview(out[a:b]).cast("B")
        total = (b - a) * width * 2
        got = 0
        while got < total:
            n = p.stdout.readinto(view[got:])
            if not n:
                raise RuntimeError("perlbrot: worker died mid-frame")
            got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check across all workers. Perl has
# no JIT, so no full-frame warm-up is needed.
run_perlbrot(16, 16, 8)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_perlbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "perlbrot.png")
