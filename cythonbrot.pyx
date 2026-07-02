# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""CythonBrot kernel: float32 scalar escape loop, OpenMP prange over top-half rows."""

cimport cython
from cython.parallel cimport prange
from libc.string cimport memcpy


cdef inline unsigned short _pixel(float cr, float ci, int max_iter) noexcept nogil:
    cdef float zr = 0.0
    cdef float zi = 0.0
    cdef float zr2, zi2, nzr, crm, q, ci2
    cdef int it = 0

    # Cardioid early-out
    crm = cr - <float>0.25
    ci2 = ci * ci
    q = crm * crm + ci2
    if q * (q + crm) <= <float>0.25 * ci2:
        return <unsigned short>max_iter
    # Period-2 bulb early-out
    if (cr + <float>1.0) * (cr + <float>1.0) + ci2 <= <float>0.0625:
        return <unsigned short>max_iter

    while it < max_iter:
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > <float>4.0:
            break
        nzr = zr2 - zi2 + cr
        zi = <float>2.0 * zr * zi + ci
        zr = nzr
        it += 1
    return <unsigned short>it


cdef void _rows(int width, int height, int max_iter, double dx, double dy,
                int r0, int r1, unsigned short[:, ::1] out) noexcept nogil:
    # Serial band over top-half rows [r0, r1), mirroring each to height-1-r.
    cdef int r, x, mr
    cdef float cr, ci
    for r in range(r0, r1):
        ci = <float>(-1.0 + r * dy)
        for x in range(width):
            cr = <float>(-2.5 + x * dx)
            out[r, x] = _pixel(cr, ci, max_iter)
        mr = height - 1 - r
        if mr != r:
            memcpy(&out[mr, 0], &out[r, 0], width * sizeof(unsigned short))


def compute(int width, int height, int max_iter, unsigned short[:, ::1] out):
    """OpenMP path: prange over top-half rows (serial without -fopenmp)."""
    cdef double dx = 3.5 / (width - 1)
    cdef double dy = 2.0 / (height - 1)
    cdef int half = (height + 1) // 2
    cdef int r
    for r in prange(half, nogil=True, schedule="dynamic", chunksize=1):
        _rows(width, height, max_iter, dx, dy, r, r + 1, out)


def compute_band(int width, int height, int max_iter, int r0, int r1,
                 unsigned short[:, ::1] out):
    """Thread-pool fallback path: nogil band, callable from Python threads."""
    cdef double dx = 3.5 / (width - 1)
    cdef double dy = 2.0 / (height - 1)
    with nogil:
        _rows(width, height, max_iter, dx, dy, r0, r1, out)
