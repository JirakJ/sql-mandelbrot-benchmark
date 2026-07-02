"""
GuileBrot - GNU Guile 3 Mandelbrot via a single persistent worker process.

Guile 3 has real POSIX threads (call-with-new-thread) and a JIT, so a single
persistent process holds an in-process thread pool over row bands and uses all
cores. The worker is compiled/JIT-warmed at import (untimed): Guile auto-compiles
guilebrot.scm to a cached .go on first launch and a full-size warm-up frame
steadies the JIT. run_guilebrot is pure compute: it sends a "w h mi" request line
and reads the whole frame back as raw uint16 little-endian bytes.

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
_SRC = os.path.join(_DIR, "guilebrot.scm")

_GUILE = shutil.which("guile") or "/opt/homebrew/bin/guile"


def _precompile():
    """Compile guilebrot.scm to a cached .go if stale (untimed).

    Guile's auto-compiler does its own source-vs-cache mtime check; invoking
    `guild compile` here forces the compile to happen at import instead of on
    first request, so timing later is steady-state.
    """
    try:
        subprocess.run(
            [_GUILE, "-c",
             f'(compile-file "{_SRC}")'],
            cwd=_DIR, capture_output=True, timeout=120,
        )
    except Exception:
        pass  # auto-compile on launch will still handle it


_precompile()

_proc = subprocess.Popen(
    [_GUILE, _SRC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    cwd=_DIR,
)


def _shutdown():
    try:
        _proc.terminate()
    except Exception:
        pass


atexit.register(_shutdown)


def run_guilebrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Guile worker. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()

    view = memoryview(out).cast("B")
    total = height * width * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("guilebrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check plus one full-size frame so the
# JIT and pipes are hot before timing starts.
run_guilebrot(16, 16, 8)
run_guilebrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_guilebrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "guilebrot.png")
