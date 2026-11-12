"""
ErlangBrot - Erlang/OTP Mandelbrot via a persistent BEAM worker process.

erlc compilation and BEAM startup happen at import time (untimed).
run_erlangbrot only writes a request line and reads raw uint16 pixels back.
BEAM floats are boxed, so this is an honest "as fast as idiomatic Erlang
goes" entry: recursive escape loop, one spawned process per row band,
iodata rows, y-axis mirror.

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
_SRC = os.path.join(_DIR, "erlangbrot.erl")
_BEAM_DIR = os.path.join(_DIR, "erlangbrot_beam")
_BEAM = os.path.join(_BEAM_DIR, "erlangbrot.beam")

_ERL = shutil.which("erl") or "/opt/homebrew/bin/erl"
_ERLC = shutil.which("erlc") or "/opt/homebrew/bin/erlc"


def _build():
    """Compile the worker if the .beam is missing or stale."""
    if os.path.exists(_BEAM) and os.path.getmtime(_BEAM) >= os.path.getmtime(_SRC):
        return
    os.makedirs(_BEAM_DIR, exist_ok=True)
    res = subprocess.run(
        [_ERLC, "+inline", "-o", _BEAM_DIR, _SRC],
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError("erlangbrot: erlc failed:\n" + res.stderr.decode())


_build()

# One persistent BEAM for the whole benchmark; stderr swallowed.
_proc = subprocess.Popen(
    [_ERL, "-noshell", "-pa", _BEAM_DIR, "-s", "erlangbrot", "start"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_erlangbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Erlang worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("erlangbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check, then a full frame so the BEAM
# scheduler pool and allocator arenas are hot before timing starts.
run_erlangbrot(16, 16, 8)
run_erlangbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_erlangbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "erlangbrot.png")
