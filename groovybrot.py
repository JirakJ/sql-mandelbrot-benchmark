"""
GroovyBrot - Groovy/JVM Vector API (SIMD) Mandelbrot via a persistent worker.

`groovy GroovyBrot.groovy` compiles the @CompileStatic worker in-process at
launch; JVM startup, groovy compile and JIT warm-up all happen at import time
(untimed). run_groovybrot only writes a request line and reads raw uint16
pixels back.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "GroovyBrot.groovy")
_GROOVY = "/opt/homebrew/bin/groovy"

# JAVA_OPTS reaches the JVM the groovy launcher starts; the incubator module
# must be visible both to the in-process groovy compiler and at runtime.
_ENV = dict(os.environ)
_ENV["JAVA_OPTS"] = (
    _ENV.get("JAVA_OPTS", "") + " --add-modules jdk.incubator.vector"
).strip()

# One persistent JVM for the whole benchmark; stderr swallowed (incubator warning).
_proc = subprocess.Popen(
    [_GROOVY, _SRC],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    env=_ENV,
)
atexit.register(_proc.terminate)


def run_groovybrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Groovy worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("groovybrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then two full frames so C2
# compiles the vector kernel and the fork-join pool spins up before timing.
run_groovybrot(16, 16, 8)
run_groovybrot(1400, 800, 256)
run_groovybrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_groovybrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "groovybrot.png")
