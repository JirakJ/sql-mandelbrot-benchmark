"""
JavaBrot - Java 21 Vector API (SIMD) Mandelbrot via a persistent worker process.

Compilation and JVM startup/JIT warm-up happen at import time (untimed).
run_javabrot only writes a request line and reads raw uint16 pixels back.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "JavaBrot.java")
_CLS_DIR = os.path.join(_DIR, "javabrot_classes")
_CLS = os.path.join(_CLS_DIR, "JavaBrot.class")


def _build():
    """Compile the worker if the .class is missing or stale."""
    if os.path.exists(_CLS) and os.path.getmtime(_CLS) >= os.path.getmtime(_SRC):
        return
    res = subprocess.run(
        ["javac", "--release", "21", "--add-modules", "jdk.incubator.vector",
         "-d", _CLS_DIR, _SRC],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError("javabrot: javac failed:\n" + res.stderr.decode())


_build()

# One persistent JVM for the whole benchmark; stderr swallowed (incubator warning).
_proc = subprocess.Popen(
    ["java", "--add-modules", "jdk.incubator.vector", "-cp", _CLS_DIR, "JavaBrot"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_javabrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Java worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("javabrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then bigger frames so the JIT
# compiles the vector kernel and the fork-join pool spins up before timing.
run_javabrot(16, 16, 8)
run_javabrot(1400, 800, 256)
run_javabrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_javabrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "javabrot.png")
