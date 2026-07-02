"""
NumbaBrot - Numba JIT Mandelbrot: parallel prange over top-half rows,
float64 scalar escape loop, cardioid/period-2 early-outs, y-axis mirror
done inside the jitted kernel. JIT compile + thread-pool spin-up happen
at import (untimed); run_numbabrot is pure compute.

Author: optimized for M-series Macs
License: MIT
"""

import numpy as np
import numba
from numba import njit, prange

from utils import save_mandelbrot_image


@njit(parallel=True, fastmath=True, cache=True)
def _mandelbrot(width, height, max_iter):
    out = np.empty((height, width), np.uint16)
    dx = 3.5 / (width - 1)
    dy = 2.0 / (height - 1)
    top = (height + 1) // 2  # includes odd middle row
    # chunk size 1 = near-dynamic scheduling; middle rows are far heavier
    old_chunk = numba.set_parallel_chunksize(1)
    for r in prange(top):
        ci = -1.0 + r * dy
        ci2 = ci * ci
        for x in range(width):
            cr = -2.5 + x * dx
            crm = cr - 0.25
            q = crm * crm + ci2
            # cardioid / period-2 bulb: in-set without iterating
            if q * (q + crm) <= 0.25 * ci2 or (cr + 1.0) * (cr + 1.0) + ci2 <= 0.0625:
                out[r, x] = max_iter
                continue
            zr = 0.0
            zi = 0.0
            it = 0
            while it < max_iter:
                zr2 = zr * zr
                zi2 = zi * zi
                if zr2 + zi2 > 4.0:
                    break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                it += 1
            out[r, x] = it
    numba.set_parallel_chunksize(old_chunk)
    # mirror top half to bottom (odd middle row already excluded)
    for r in prange(height // 2):
        dst = height - 1 - r
        for x in range(width):
            out[dst, x] = out[r, x]
    return out


# Warm-up at import: compiles (or loads cached machine code, cache=True)
# and spins up the threading layer before the first timed call.
_mandelbrot(16, 16, 8)
_mandelbrot(1400, 800, 256)


def run_numbabrot(width, height, max_iterations):
    """Compute the Mandelbrot set via the Numba kernel. Returns (h, w) uint16."""
    return _mandelbrot(width, height, max_iterations)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_numbabrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "numbabrot.png")
