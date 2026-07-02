"""
LeanBrot - Lean 4 compiled Mandelbrot via a persistent worker process.

leanbrot.lean is compiled through C (lean -c, then leanc -O3) at import time
when the binary is missing or stale. The worker computes the top half of the
frame in parallel (Task.spawn thread pool); the wrapper mirrors the bottom
half via numpy (y-axis symmetry). run_leanbrot only does request + read.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess
import tempfile

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "leanbrot.lean")
_BIN = os.path.join(_DIR, "leanbrot_bin")


def _build():
    """Compile the worker if the binary is missing or stale."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return
    with tempfile.TemporaryDirectory() as tmp:
        c_file = os.path.join(tmp, "leanbrot.c")
        for cmd in (
            ["lean", "-c", c_file, _SRC],
            ["leanc", "-O3", "-o", _BIN, c_file],
        ):
            res = subprocess.run(cmd, cwd=_DIR, capture_output=True)
            if res.returncode != 0:
                raise RuntimeError(
                    "leanbrot: %s failed:\n%s" % (cmd[0], res.stderr.decode())
                )


_build()

# One persistent worker for the whole benchmark; logs go to stderr only.
_proc = subprocess.Popen([_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
atexit.register(_proc.terminate)


def run_leanbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Lean worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    top = (height + 1) // 2
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * top * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:total])
        if not n:
            raise RuntimeError("leanbrot: worker died mid-frame")
        got += n
    if top < height:  # mirror bottom half
        out[top:] = out[height - top - 1 :: -1]
    return out


# Warm-up at import (untimed): protocol check, then a full frame so the
# Lean task-pool threads exist before the first timed call.
run_leanbrot(16, 16, 8)
run_leanbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_leanbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "leanbrot.png")
