"""
ElispBrot - Emacs Lisp Mandelbrot via a pool of persistent batch workers.

Emacs Lisp has no shared-memory threads, so parallelism comes from N
`emacs --batch -Q --load elispbrot.el` workers spawned at import time
(untimed; each byte-compiles / native-compiles the hot kernel at startup).
run_elispbrot hands out single-row bands of the top half round-robin for
near-perfect load balance, reads raw uint16 LE rows back into the numpy
buffer, and mirrors the bottom half (y-axis symmetry).

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
_SRC = os.path.join(_DIR, "elispbrot.el")

_EMACS = "/opt/homebrew/bin/emacs"
if not os.path.exists(_EMACS):
    _EMACS = shutil.which("emacs")
if _EMACS is None:
    raise RuntimeError("elispbrot: emacs not found")

_NW = min(16, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_EMACS, "--batch", "-Q", "--load", _SRC],
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


def run_elispbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Emacs pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2

    # Single-row bands, round-robin: adjacent rows have similar cost, so the
    # expensive rows near the real axis spread evenly across workers.
    for r in range(top):
        _procs[r % _NW].stdin.write(
            f"{width} {height} {max_iterations} {r} {r + 1}\n".encode()
        )
    for p in _procs:
        p.stdin.flush()

    rowbytes = width * 2
    for r in range(top):
        p = _procs[r % _NW]
        view = memoryview(out[r : r + 1]).cast("B")
        got = 0
        while got < rowbytes:
            n = p.stdout.readinto(view[got:])
            if not n:
                raise RuntimeError("elispbrot: worker died mid-frame")
            got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check; kernels were compiled at
# worker startup, so no full-frame JIT warm-up is needed.
run_elispbrot(16, 16, 8)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_elispbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "elispbrot.png")
