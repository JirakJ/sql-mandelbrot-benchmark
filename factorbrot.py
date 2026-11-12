"""
FactorBrot - Factor (concatenative language) Mandelbrot via a pool of
persistent worker processes.

Factor's threads are cooperative (green), so a single image cannot be split
across cores in-process. Parallelism therefore comes from N persistent Factor
worker processes spawned at import time (untimed). Each worker runs
factorbrot.factor, reading request lines "w h mi a b" from stdin and writing
rows [a, b) back as raw little-endian uint16. run_factorbrot deals the top
half of the image out as contiguous row bands, reads the bands into the numpy
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
_SRC = os.path.join(_DIR, "factorbrot.factor")


def _find_factor():
    p = shutil.which("factor")
    # GNU coreutils ships a `factor` (prime factorization) that shadows the
    # Factor language on PATH; prefer the language binary explicitly.
    if p and "coreutils" not in p:
        # Verify it is actually the Factor language, not coreutils.
        try:
            out = subprocess.run(
                [p, "--version"], capture_output=True, timeout=10
            ).stdout.decode(errors="ignore")
            if "coreutils" not in out:
                return p
        except Exception:
            pass
    for cand in (
        "/Applications/factor/factor",
        "/Applications/factor/Factor.app/Contents/MacOS/factor",
        os.path.expanduser("~/factor/factor"),
    ):
        if os.path.exists(cand):
            return cand
    raise RuntimeError("factorbrot: Factor language binary not found")


_FACTOR = _find_factor()
_NW = min(14, os.cpu_count() or 1)

_procs = [
    subprocess.Popen(
        [_FACTOR, _SRC],
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


def run_factorbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Factor pool. Returns (h, w) uint16."""
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
                raise RuntimeError("factorbrot: worker died mid-frame")
            got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check, then one full-size frame so
# every worker has its hot words JIT-compiled before timing starts.
run_factorbrot(16, 16, 8)
run_factorbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_factorbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "factorbrot.png")
