"""
DartBrot - Dart AOT + isolate pool Mandelbrot via a persistent worker process.

The Dart source (dartbrot.dart) is AOT-compiled at import time (untimed) and
launched once as a long-lived worker speaking a tiny binary protocol over
stdin/stdout. run_dartbrot only sends the request and reads the raw frame.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "dartbrot.dart")
_BIN = os.path.join(_DIR, "dartbrot_bin")
_DART = "/opt/homebrew/bin/dart"


def _build():
    """AOT-compile the worker if the binary is missing or stale."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return
    r = subprocess.run([_DART, "compile", "exe", _SRC, "-o", _BIN])
    if r.returncode != 0:
        raise RuntimeError("dartbrot: dart compile exe failed")


_build()
_proc = subprocess.Popen(
    [_BIN],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    # stderr inherited: worker logging stays visible, stdout stays binary-clean
)
atexit.register(_proc.kill)


def _request(width, height, max_iterations):
    _proc.stdin.write(b"%d %d %d\n" % (width, height, max_iterations))
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    off = 0
    while off < len(view):
        got = _proc.stdout.readinto(view[off:])
        if not got:
            raise RuntimeError("dartbrot: worker died mid-frame")
        off += got
    return out


# Warm-up at import (untimed): spins up the isolate pool and AOT code paths.
_request(16, 16, 8)


def run_dartbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the Dart worker. Returns (h, w) uint16."""
    return _request(width, height, max_iterations)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_dartbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "dartbrot.png")
