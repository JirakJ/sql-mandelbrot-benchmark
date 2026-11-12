"""
CythonBrot - Cython 3 native Mandelbrot: nogil float32 kernel, OpenMP prange
over top-half rows, y-axis mirror via memcpy.

The extension is cythonized in a build dir on first import (untimed) and
imported from there. If the OpenMP build fails, a plain build is used with a
ThreadPoolExecutor over nogil row bands instead.

Author: optimized for M-series Macs
License: MIT
"""

import glob
import os
import shutil
import subprocess
import sys

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "cythonbrot.pyx")
_BUILD = os.path.join(_DIR, ".cythonbrot_build")
_IMPL = "_cythonbrot_impl"  # distinct name so the .so doesn't shadow this module
_STAMP = os.path.join(_BUILD, "mode.txt")
_OMP = "/opt/homebrew/opt/libomp"

_SETUP = r"""
import sys
from setuptools import Extension, setup
from Cython.Build import cythonize

impl, omp, mcpu = sys.argv[1], sys.argv[2], sys.argv[3]
cflags = ["-O3", "-fno-math-errno"]
ldflags = []
if mcpu:
    cflags.append(mcpu)
if omp:
    cflags += ["-Xpreprocessor", "-fopenmp", "-I" + omp + "/include"]
    ldflags += ["-L" + omp + "/lib", "-lomp", "-Wl,-rpath," + omp + "/lib"]
setup(
    script_args=["build_ext", "--inplace"],
    ext_modules=cythonize(
        [Extension(impl, [impl + ".pyx"],
                   extra_compile_args=cflags, extra_link_args=ldflags)],
        compiler_directives={"language_level": 3},
        quiet=True,
    ),
)
"""


def _so_path():
    hits = glob.glob(os.path.join(_BUILD, _IMPL + "*.so"))
    return hits[0] if hits else None


def _build():
    so = _so_path()
    if so and os.path.getmtime(so) >= os.path.getmtime(_SRC):
        return
    os.makedirs(_BUILD, exist_ok=True)
    shutil.copyfile(_SRC, os.path.join(_BUILD, _IMPL + ".pyx"))
    omp = _OMP if os.path.exists(os.path.join(_OMP, "lib", "libomp.dylib")) else ""
    for use_omp in ([omp] if omp else []) + [""]:
        for mcpu in ("-mcpu=apple-m4", ""):
            res = subprocess.run(
                [sys.executable, "-c", _SETUP, _IMPL, use_omp, mcpu],
                cwd=_BUILD, stdout=subprocess.DEVNULL,
            )
            if res.returncode == 0 and _so_path():
                with open(_STAMP, "w") as f:
                    f.write("omp" if use_omp else "threads")
                return
    raise RuntimeError("cythonbrot: cythonize build failed")


_build()
sys.path.insert(0, _BUILD)
_impl = __import__(_IMPL)
sys.path.remove(_BUILD)

with open(_STAMP) as f:
    _MODE = f.read().strip()

if _MODE == "threads":
    from concurrent.futures import ThreadPoolExecutor

    _NWORK = os.cpu_count() or 4
    _pool = ThreadPoolExecutor(_NWORK)


def _run(width, height, max_iterations, out):
    if _MODE == "omp":
        _impl.compute(width, height, max_iterations, out)
        return
    half = (height + 1) // 2
    n = min(_NWORK * 4, half)  # small bands for load balance
    bounds = [half * i // n for i in range(n + 1)]
    futs = [
        _pool.submit(_impl.compute_band, width, height, max_iterations,
                     bounds[i], bounds[i + 1], out)
        for i in range(n)
    ]
    for f in futs:
        f.result()


# Warm-up at import (untimed): spins up the OpenMP/thread pool.
_warm = np.empty((16, 16), dtype=np.uint16)
_run(16, 16, 8, _warm)
del _warm


def run_cythonbrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the Cython kernel. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    _run(width, height, max_iterations, out)
    return out


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_cythonbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "cythonbrot.png")
