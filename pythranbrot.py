"""
PythranBrot - Pythran (Python -> C++ with xsimd) Mandelbrot. The kernel in
pythranbrot_impl.py is ahead-of-time compiled to a native .so at import
(mtime-guarded, untimed) with -O3 -march=native -fopenmp -ffast-math, then
imported and called. run_pythranbrot is pure compute over the compiled kernel.

Author: optimized for M-series Macs
License: MIT
"""

import os
import sys
import glob
import shutil
import importlib
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "pythranbrot_impl.py")
# Compile in an isolated dir: pythran shells out to setuptools, whose flat-layout
# package discovery otherwise trips over the many sibling dirs in the repo root.
_BUILD = os.path.join(_DIR, ".pythranbrot_build")


def _artifact():
    hits = glob.glob(os.path.join(_BUILD, "pythranbrot_impl*.so"))
    return hits[0] if hits else None


def _build():
    art = _artifact()
    if art and os.path.getmtime(art) >= os.path.getmtime(_SRC):
        return  # up to date
    os.makedirs(_BUILD, exist_ok=True)
    if art:
        os.remove(art)
    src_copy = os.path.join(_BUILD, "pythranbrot_impl.py")
    shutil.copyfile(_SRC, src_copy)
    env = dict(os.environ)
    # Apple's /usr/bin/c++ rejects -fopenmp; use a Homebrew LLVM clang (with its
    # bundled libomp) when present so the OpenMP pragma actually parallelizes.
    for base in ("/opt/homebrew/opt/llvm/bin", "/usr/local/opt/llvm/bin"):
        cxx = os.path.join(base, "clang++")
        cc = os.path.join(base, "clang")
        if os.path.exists(cxx):
            env["CXX"], env["CC"] = cxx, cc
            break
    subprocess.run(
        ["uv", "run", "pythran", "-O3", "-march=native", "-fopenmp",
         "-ffast-math", "pythranbrot_impl.py"],
        cwd=_BUILD, check=True, env=env,
    )


_build()

if _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)
_impl = importlib.import_module("pythranbrot_impl")
_mandel = _impl.mandel


def run_pythranbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the compiled Pythran kernel. (h, w) uint16."""
    return _mandel(width, height, max_iterations)


# Warm-up at import so later timing is steady-state (thread pool spun up).
run_pythranbrot(16, 16, 8)
run_pythranbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_pythranbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "pythranbrot.png")
