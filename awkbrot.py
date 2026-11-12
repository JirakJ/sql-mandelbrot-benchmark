"""
AwkBrot - GNU AWK Mandelbrot via a pool of persistent gawk worker processes.

AWK has no threads, so parallelism comes from N gawk workers spawned at
import time (untimed). Top-half rows are dealt round-robin across the pool
(adjacent rows differ hugely in cost near the set, so striding balances the
load), each worker emits rows as raw uint16 LE bytes built with sprintf
"%c%c" under LC_ALL=C, and the wrapper reassembles them into the numpy
buffer and mirrors the bottom half (y-axis symmetry).

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
_SRC = os.path.join(_DIR, "awkbrot.awk")

_GAWK = "/opt/homebrew/bin/gawk"
if not os.path.exists(_GAWK):
    _GAWK = shutil.which("gawk")
if _GAWK is None:
    raise RuntimeError("awkbrot: gawk not found")

_NW = min(14, os.cpu_count() or 1)

# C locale + -b (bytes mode) keep "%c" byte-exact for 0-255, incl. NUL.
_ENV = dict(os.environ, LC_ALL="C", LANG="C")

_procs = [
    subprocess.Popen(
        [_GAWK, "-b", "-f", _SRC],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_ENV,
    )
    for _ in range(_NW)
]


def _shutdown():
    for p in _procs:
        p.terminate()


atexit.register(_shutdown)


def run_awkbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the gawk pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2

    # Worker i computes rows i, i+NW, i+2*NW, ... of the top half.
    rows = [range(i, top, _NW) for i in range(_NW)]

    for p, rs in zip(_procs, rows):
        if rs:
            reqs = "".join(
                f"{width} {height} {max_iterations} {r} {r + 1}\n" for r in rs
            )
            p.stdin.write(reqs.encode())
            p.stdin.flush()

    rowbytes = width * 2
    for p, rs in zip(_procs, rows):
        for r in rs:
            view = memoryview(out[r]).cast("B")
            got = 0
            while got < rowbytes:
                n = p.stdout.readinto(view[got:])
                if not n:
                    raise RuntimeError("awkbrot: worker died mid-frame")
                got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): checks the byte protocol survives the pipe.
# No full-size frame here - gawk has no JIT to warm.
_chk = run_awkbrot(16, 16, 8)
assert _chk.shape == (16, 16) and int(_chk.max()) == 8, "awkbrot: protocol mangled"
del _chk


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_awkbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "awkbrot.png")
