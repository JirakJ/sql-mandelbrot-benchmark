"""
ScalaBrot - Scala 3/JVM Vector API (SIMD) Mandelbrot via a persistent worker.

Compilation and JVM startup/JIT warm-up happen at import time (untimed).
run_scalabrot only writes a request line and reads raw uint16 pixels back.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import glob
import os
import shutil
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "ScalaBrot.scala")
_CLS_DIR = os.path.join(_DIR, "scalabrot_classes")
_CLS = os.path.join(_CLS_DIR, "ScalaBrot.class")


def _find_jar(pattern):
    """Locate a Scala runtime jar under common install roots."""
    roots = ["/opt/homebrew/Cellar/scala", "/usr/local/Cellar/scala"]
    if os.environ.get("SCALA_HOME"):
        roots.append(os.environ["SCALA_HOME"])
    scalac = shutil.which("scalac")
    if scalac:  # e.g. .../scala/3.8.4/bin/scalac -> search its install tree
        roots.append(os.path.dirname(os.path.dirname(os.path.realpath(scalac))))
    for root in roots:
        hits = [
            h
            for h in sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
            if "sources" not in h and "javadoc" not in h
        ]
        if hits:
            return hits[-1]  # newest version sorts last
    raise RuntimeError(f"scalabrot: cannot find {pattern} (is scala installed?)")


def _build():
    """Compile the worker if the .class is missing or stale."""
    if os.path.exists(_CLS) and os.path.getmtime(_CLS) >= os.path.getmtime(_SRC):
        return
    os.makedirs(_CLS_DIR, exist_ok=True)
    res = subprocess.run(
        ["scalac", "-release", "21", "-d", _CLS_DIR, _SRC],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError("scalabrot: scalac failed:\n" + res.stderr.decode())


_build()

_CP = os.pathsep.join(
    [_CLS_DIR, _find_jar("scala3-library*.jar"), _find_jar("scala-library*.jar")]
)

# One persistent JVM for the whole benchmark; stderr swallowed (incubator warning).
_proc = subprocess.Popen(
    ["java", "--add-modules", "jdk.incubator.vector", "-cp", _CP, "ScalaBrot"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_scalabrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Scala worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("scalabrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then two full frames so C2
# compiles the vector kernel and the fork-join pool spins up before timing.
run_scalabrot(16, 16, 8)
run_scalabrot(1400, 800, 256)
run_scalabrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_scalabrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "scalabrot.png")
