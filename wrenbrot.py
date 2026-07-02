"""
WrenBrot - Wren Mandelbrot via a pool of persistent worker processes.

Wren's VM is single-threaded, so parallelism comes from N wren_cli workers
spawned at import time (untimed). run_wrenbrot splits the top half of the
image into one contiguous row band per worker, decodes each band (wren_cli
stdout can't carry NUL bytes, so pixels arrive as 3 NUL-free base-255 bytes)
into the numpy buffer, and mirrors the bottom half (y-axis symmetry).

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
_SRC = os.path.join(_DIR, "wrenbrot.wren")

_WREN = shutil.which("wren_cli") or "/opt/homebrew/bin/wren_cli"
if not os.path.exists(_WREN):
    raise RuntimeError("wrenbrot: wren_cli not found")

# 2x oversubscription: interleaved bands are equal-cost, so extra workers let
# the scheduler balance P- vs E-cores on Apple Silicon.
_NW = min(28, 2 * (os.cpu_count() or 1))

_procs = [
    subprocess.Popen(
        [_WREN, _SRC],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(_NW)
]


def _shutdown():
    for p in _procs:
        p.terminate()


atexit.register(_shutdown)

# Decode weights for the 3-byte base-255 (+1 per byte) pixel encoding.
_W3 = np.array([65025, 255, 1], dtype=np.uint32)
_BIAS = np.uint32(65025 + 255 + 1)


def run_wrenbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the Wren pool. Returns (h, w) uint16."""
    out = np.empty((height, width), dtype=np.uint16)
    top = (height + 1) // 2
    nw = min(_NW, top)

    # Interleaved rows (worker i does rows i, i+nw, ...) to balance load:
    # rows near the middle of the image cost far more than the top rows.
    for i in range(nw):
        _procs[i].stdin.write(
            f"{width} {height} {max_iterations} {i} {top} {nw}\n".encode()
        )
        _procs[i].stdin.flush()

    for i in range(nw):
        p = _procs[i]
        nrows = len(range(i, top, nw))
        total = nrows * width * 3
        buf = bytearray(total)
        view = memoryview(buf)
        got = 0
        while got < total:
            n = p.stdout.readinto(view[got:])
            if not n:
                raise RuntimeError("wrenbrot: worker died mid-frame")
            got += n
        d = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        vals = d @ _W3 - _BIAS
        out[i:top:nw] = vals.astype(np.uint16).reshape(nrows, width)

    # Mirror bottom half; middle row of an odd height stays as computed.
    if top < height:
        out[top:] = out[height - 1 - top::-1]
    return out


# Warm-up at import (untimed): protocol check, then a full-size frame so every
# worker has touched its hot paths before timing starts.
run_wrenbrot(16, 16, 8)
run_wrenbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_wrenbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "wrenbrot.png")
