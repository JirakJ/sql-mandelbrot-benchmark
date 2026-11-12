"""
PythranBrot kernel — Python source ahead-of-time compiled to C++ (xsimd +
OpenMP) by Pythran. The nested float64 escape loop with cardioid/period-2
early-outs is auto-vectorized; the `# omp parallel for` pragma over the
top-half rows fans work across all cores. Mirror applied in-kernel.

Author: optimized for M-series Macs
License: MIT
"""

import numpy as np


# pythran export mandel(int, int, int)
def mandel(width, height, max_iter):
    out = np.empty((height, width), np.uint16)
    dx = 3.5 / (width - 1)
    dy = 2.0 / (height - 1)
    top = (height + 1) // 2  # includes odd middle row
    # omp parallel for schedule(dynamic)
    for r in range(top):
        ci = -1.0 + r * dy
        ci2 = ci * ci
        for x in range(width):
            cr = -2.5 + x * dx
            crm = cr - 0.25
            q = crm * crm + ci2
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
    # mirror top half to bottom (odd middle row already excluded)
    for r in range(height // 2):
        dst = height - 1 - r
        for x in range(width):
            out[dst, x] = out[r, x]
    return out
