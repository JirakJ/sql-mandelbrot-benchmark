// CBrot - pure C11 Mandelbrot: scalar float loop, no intrinsics.
// The "what does clang -O3 autovectorization give you for free" datapoint.
//
// Semantics match pybrot.py: value = iterations survived before |z|^2 > 4
// (check-then-iterate), capped at max_iter (in-set pixels -> max_iter).
// Cardioid + period-2 bulb early-out, y-axis symmetry (top half computed,
// bottom half mirrored), rows fanned across all cores via libdispatch.

#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

void mandelbrot_c(int width, int height, int max_iter, uint16_t *out) {
    const float dx = 3.5f / (float)(width - 1);
    const float dy = 2.0f / (float)(height - 1);
    const float x0 = -2.5f;
    const float y0 = -1.0f;

    // rows r and height-1-r have opposite ci -> identical escape counts
    const size_t half = ((size_t)height + 1) / 2;

    dispatch_apply(half,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t row) {
        const float ci = y0 + (float)row * dy;
        const float ci2 = ci * ci;
        uint16_t *orow = out + row * (size_t)width;

        for (int x = 0; x < width; ++x) {
            const float cr = x0 + (float)x * dx;

            // main cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2
            const float crm = cr - 0.25f;
            const float q = crm * crm + ci2;
            if (q * (q + crm) <= 0.25f * ci2) {
                orow[x] = (uint16_t)max_iter;
                continue;
            }
            // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
            const float crp = cr + 1.0f;
            if (crp * crp + ci2 <= 0.0625f) {
                orow[x] = (uint16_t)max_iter;
                continue;
            }

            float zr = 0.f, zi = 0.f;
            int it = 0;
            for (; it < max_iter; ++it) {
                const float zr2 = zr * zr;
                const float zi2 = zi * zi;
                if (zr2 + zi2 > 4.0f) break;
                const float nzr = zr2 - zi2 + cr;
                zi = 2.f * zr * zi + ci;
                zr = nzr;
            }
            orow[x] = (uint16_t)it;
        }

        const size_t mrow = (size_t)height - 1 - row;
        if (mrow != row)
            memcpy(out + mrow * (size_t)width, orow, (size_t)width * sizeof(uint16_t));
    });
}
