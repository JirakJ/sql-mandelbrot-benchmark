"""
KotlinBrot - Kotlin/JVM Vector API (SIMD) Mandelbrot via a persistent worker.

Compilation and JVM startup/JIT warm-up happen at import time (untimed).
run_kotlinbrot only writes a request line and reads raw uint16 pixels back.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "KotlinBrot.kt")
_JAR = os.path.join(_DIR, "kotlinbrot.jar")


def _build():
    """Compile the worker jar if missing or stale (slow, ~1 min, import-only)."""
    if os.path.exists(_JAR) and os.path.getmtime(_JAR) >= os.path.getmtime(_SRC):
        return
    res = subprocess.run(
        ["/opt/homebrew/bin/kotlinc", _SRC, "-include-runtime", "-d", _JAR],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError("kotlinbrot: kotlinc failed:\n" + res.stderr.decode())


_build()

# One persistent JVM for the whole benchmark; stderr swallowed (incubator warning).
_proc = subprocess.Popen(
    ["java", "--add-modules", "jdk.incubator.vector", "-jar", _JAR],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_kotlinbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Kotlin worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("kotlinbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then two full frames so C2
# compiles the vector kernel and the fork-join pool spins up before timing.
run_kotlinbrot(16, 16, 8)
run_kotlinbrot(1400, 800, 256)
run_kotlinbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_kotlinbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "kotlinbrot.png")
