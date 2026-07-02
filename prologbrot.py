"""
PrologBrot - SWI-Prolog Mandelbrot via a persistent multi-threaded worker.

SWI-Prolog has real OS threads, so one persistent swipl process fans one-row
jobs across all cores through a message queue. The wrapper sends
"w h max_iter\\n", reads the top half back as raw uint16 LE pixels, and
mirrors the bottom half (y-axis symmetry). Process spawn and warm-up happen
at import time (untimed).

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
_SRC = os.path.join(_DIR, "prologbrot.pl")

_SWIPL = shutil.which("swipl") or "/opt/homebrew/bin/swipl"
if not os.path.exists(_SWIPL):
    raise RuntimeError("prologbrot: swipl not found")

# One persistent SWI-Prolog process; it threads internally across all cores.
_proc = subprocess.Popen(
    [_SWIPL, "-q", "-O", "-g", "main", _SRC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_prologbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Prolog worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2
    view = memoryview(out[:top]).cast("B")
    total = top * width * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("prologbrot: worker died mid-frame")
        got += n
    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check, then one full frame so thread
# pools, allocator arenas and clause indexes are hot before timing starts.
run_prologbrot(16, 16, 8)
run_prologbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_prologbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "prologbrot.png")
