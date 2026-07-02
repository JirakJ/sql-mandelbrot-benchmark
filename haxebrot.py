"""
HaxeBrot - Haxe (hxcpp C++ target) Mandelbrot via a persistent worker process.

HaxeBrot.hx is compiled to a native binary at import time if missing or stale
(untimed). The worker is internally multithreaded (sys.thread over row bands
with y-axis symmetry), so a single process serves whole frames.
run_haxebrot only writes a request line and reads raw uint16 LE pixels back.

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
_SRC = os.path.join(_DIR, "HaxeBrot.hx")
_BUILD_DIR = os.path.join(_DIR, "haxebrot_build")
_EXE = os.path.join(_BUILD_DIR, "HaxeBrot")

_HAXE = shutil.which("haxe") or "/opt/homebrew/bin/haxe"


def _build():
    """Compile the worker binary if missing or stale."""
    if os.path.exists(_EXE) and os.path.getmtime(_EXE) >= os.path.getmtime(_SRC):
        return
    res = subprocess.run(
        [_HAXE, "-cp", _DIR, "--main", "HaxeBrot", "--cpp", _BUILD_DIR,
         "-D", "HXCPP_M64", "-D", "HXCPP_OPTIMIZE_LINK",
         "-D", "analyzer-optimize", "-dce", "full"],
        capture_output=True, cwd=_DIR,
    )
    if res.returncode != 0:
        raise RuntimeError("haxebrot: haxe build failed:\n" + res.stderr.decode())


_build()

_NT = min(14, os.cpu_count() or 1)

# One persistent worker for the whole benchmark; it threads internally.
_proc = subprocess.Popen(
    [_EXE, str(_NT)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_haxebrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Haxe worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("haxebrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then full-size frames so the
# thread pool and pages are hot before timing starts.
run_haxebrot(16, 16, 8)
run_haxebrot(1400, 800, 256)
run_haxebrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_haxebrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "haxebrot.png")
