"""
GambitBrot - Gambit Scheme Mandelbrot via a pool of persistent workers.

Gambit's threads are green (cooperative, not parallel), so parallelism comes
from N worker processes spawned at import time (untimed). Each worker is a
standalone native executable compiled by gsc (Scheme -> C -> machine code).
run_gambitbrot splits the top half of the image into one contiguous row band
per worker, reads the raw uint16 bands back into the numpy buffer, and mirrors
the bottom half (y-axis symmetry).

Build note: the Homebrew Gambit bottle is configured to build its runtime with
GCC, and gsc's default `gsc` name on this machine collides with Ghostscript, so
we invoke Gambit by its Cellar path and force a matching gcc-* C compiler
(clang-built modules abort with a runtime-incompatibility / exit 71).

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
_SRC = os.path.join(_DIR, "gambitbrot.scm")
_BIN = os.path.join(_DIR, "gambitbrot_bin")

# Real Gambit gsc (the bare `gsc` on PATH is Ghostscript on this box).
_GSC = next(
    (p for p in (
        "/opt/homebrew/opt/gambit-scheme/bin/gsc",
        "/opt/homebrew/bin/gsc-gambit",
    ) if os.path.exists(p)),
    shutil.which("gsc") or "gsc",
)


def _find_gcc():
    """Gambit's runtime is GCC-built; a matching gcc-* keeps modules loadable."""
    for name in ("gcc-16", "gcc-15", "gcc-14"):
        p = shutil.which(name)
        if p:
            return name
    cands = sorted(glob.glob("/opt/homebrew/bin/gcc-[0-9]*"))
    if cands:
        return os.path.basename(cands[-1])
    return shutil.which("gcc") or "gcc"


def _build():
    """Compile the worker exe if the binary is missing or stale (untimed)."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return
    cc = _find_gcc()
    res = subprocess.run(
        [_GSC, "-cc", cc, "-exe", "-o", _BIN, _SRC],
        cwd=_DIR,
        capture_output=True,
    )
    if res.returncode != 0 or not os.path.exists(_BIN):
        raise RuntimeError("gambitbrot: build failed:\n" + res.stderr.decode())


_build()

_NW = min(14, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(_NW)
]


def _shutdown():
    for p in _procs:
        try:
            p.terminate()
        except Exception:
            pass


atexit.register(_shutdown)


def run_gambitbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Gambit pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2
    base, rem = divmod(top, _NW)

    # One contiguous band per worker; skip workers that get no rows.
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
                raise RuntimeError("gambitbrot: worker died mid-frame")
            got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check plus one full-size frame so
# pages and pipes are hot before timing starts.
run_gambitbrot(16, 16, 8)
run_gambitbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_gambitbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "gambitbrot.png")
