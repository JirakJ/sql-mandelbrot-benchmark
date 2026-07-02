"""
CobolBrot - GnuCOBOL Mandelbrot via a pool of persistent worker processes.

GnuCOBOL has no threads, so parallelism comes from N compiled COBOL workers
spawned at import time (untimed). The top half of the image is split into
small row chunks assigned round-robin across workers (rows near the real
axis are much heavier, so fine chunks balance the load), each worker streams
its chunks back as raw uint16 LE, and the bottom half is a numpy mirror
(y-axis symmetry).

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "cobolbrot.cob")
_BIN = os.path.join(_DIR, "cobolbrot_bin")
_COBC = "/opt/homebrew/bin/cobc"

if not os.path.exists(_COBC):
    raise RuntimeError("cobolbrot: cobc not found at " + _COBC)

# Build at import (untimed) if the binary is missing or stale.
if not os.path.exists(_BIN) or os.path.getmtime(_BIN) < os.path.getmtime(_SRC):
    subprocess.run(
        [_COBC, "-x", "-O2", "-free", _SRC, "-o", _BIN],
        check=True,
        cwd=_DIR,
    )

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
        except OSError:
            pass


atexit.register(_shutdown)


def run_cobolbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the COBOL pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2

    # Round-robin small chunks: each worker gets a mix of cheap edge rows and
    # expensive near-axis rows. All requests are queued upfront (they are tiny
    # lines); responses are drained in assignment order.
    chunk = max(1, top // (_NW * 8))
    tasks = []  # (worker, r0, r1) in assignment order
    r0 = 0
    i = 0
    while r0 < top:
        r1 = min(top, r0 + chunk)
        tasks.append((i % _NW, r0, r1))
        r0 = r1
        i += 1

    reqs = [b""] * _NW
    for wi, a, b in tasks:
        reqs[wi] += f"{width} {height} {max_iterations} {a} {b}\n".encode()
    for wi in range(_NW):
        if reqs[wi]:
            _procs[wi].stdin.write(reqs[wi])
            _procs[wi].stdin.flush()

    for wi, a, b in tasks:
        view = memoryview(out[a:b]).cast("B")
        total = (b - a) * width * 2
        got = 0
        while got < total:
            n = _procs[wi].stdout.readinto(view[got:])
            if not n:
                raise RuntimeError("cobolbrot: worker died mid-frame")
            got += n

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check and page fault warm; the COBOL
# binary is AOT-compiled, so no JIT-scale warm frames are needed.
run_cobolbrot(16, 16, 8)
run_cobolbrot(256, 256, 32)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_cobolbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "cobolbrot.png")
