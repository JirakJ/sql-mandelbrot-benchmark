"""
ChapelBrot - Chapel (HPC parallel language) Mandelbrot via a persistent worker.

The worker (chapelbrot.chpl) is compiled with `chpl --fast` at import time if
missing or stale (untimed). run_chapelbrot writes a request line and reads raw
uint16 pixels back from the persistent process.

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
_SRC = os.path.join(_DIR, "chapelbrot.chpl")
_BIN = os.path.join(_DIR, "chapelbrot_bin")

_CHPL = shutil.which("chpl") or "/opt/homebrew/bin/chpl"


def _build():
    """Compile the worker if the binary is missing or stale (slow, ~30-60s)."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return
    res = subprocess.run(
        [_CHPL, "--fast", _SRC, "-o", _BIN],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError("chapelbrot: chpl failed:\n" + res.stderr.decode())


_build()

# One persistent Chapel runtime for the whole benchmark; all cores by default.
_env = dict(os.environ)
_env.setdefault("CHPL_RT_NUM_THREADS_PER_LOCALE", str(os.cpu_count() or 1))

_proc = subprocess.Popen(
    [_BIN],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env=_env,
)
atexit.register(_proc.terminate)


def run_chapelbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Chapel worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("chapelbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then a full frame so the
# tasking layer has spun up all threads before timing starts.
run_chapelbrot(16, 16, 8)
run_chapelbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_chapelbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "chapelbrot.png")
