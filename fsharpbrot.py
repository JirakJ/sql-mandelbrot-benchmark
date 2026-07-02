"""
FSharpBrot - F# .NET ARM NEON (AdvSimd) Mandelbrot via a persistent worker process.

dotnet publish and JIT warm-up happen at import time (untimed).
run_fsharpbrot only writes a request line and reads raw uint16 pixels back.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.join(_DIR, "fsharpbrot")
_SRCS = [os.path.join(_PROJ, "Program.fs"), os.path.join(_PROJ, "fsharpbrot.fsproj")]
_PUB = os.path.join(_PROJ, "bin", "publish")
_DLL = os.path.join(_PUB, "fsharpbrot.dll")
_DOTNET = "/opt/homebrew/bin/dotnet"


def _build():
    """Publish the worker if the binary is missing or stale."""
    if os.path.exists(_DLL) and os.path.getmtime(_DLL) >= max(
        os.path.getmtime(s) for s in _SRCS
    ):
        return
    res = subprocess.run(
        [_DOTNET, "publish", "-c", "Release", "-o", _PUB, _PROJ],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            "fsharpbrot: dotnet publish failed:\n"
            + res.stdout.decode() + res.stderr.decode()
        )


_build()

# One persistent .NET worker for the whole benchmark; stderr swallowed.
# Launched via the dotnet host: Homebrew's runtime isn't in the apphost's
# default search path.
_proc = subprocess.Popen(
    [_DOTNET, _DLL],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_fsharpbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the F# worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("fsharpbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then bigger frames so tiered
# compilation promotes the NEON kernel and the thread pool spins up.
run_fsharpbrot(16, 16, 8)
run_fsharpbrot(1400, 800, 256)
run_fsharpbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_fsharpbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "fsharpbrot.png")
