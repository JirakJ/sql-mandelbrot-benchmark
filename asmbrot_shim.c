// AsmBrot shim - row dispatch (GCD), cardioid/bulb early-out, scalar tail,
// y-axis symmetry mirror. The 4-pixel inner escape loop lives in asmbrot.s.
// Semantics match pybrot.py: value = iterations survived, in-set = max_iter.

#include <dispatch/dispatch.h>
#include <stdint.h>
#include <string.h>

void mandel4(const float cr[4], float ci, int max_iter,
             uint32_t count[4], const uint32_t active0[4]);

void mandelbrot_asm(int width, int height, int max_iter, uint16_t *out) {
    const float dx = 3.5f / (float)(width - 1);
    const float dy = 2.0f / (float)(height - 1);
    // Rows r and height-1-r have opposite ci; escape counts are
    // conjugation-invariant, so only the top half is computed.
    const size_t half = ((size_t)height + 1) / 2;

    dispatch_apply(half, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t row) {
        const float ci = -1.0f + (float)row * dy;
        const float ci2 = ci * ci;
        uint16_t *orow = out + row * (size_t)width;

        int x = 0;
        for (; x + 4 <= width; x += 4) {
            float cr[4];
            uint32_t act[4], cnt[4];
            int nact = 0;
            for (int k = 0; k < 4; ++k) {
                float c = -2.5f + (float)(x + k) * dx;
                cr[k] = c;
                // main cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2
                float crm = c - 0.25f;
                float q = crm * crm + ci2;
                // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
                float crp = c + 1.0f;
                int inset = (q * (q + crm) <= 0.25f * ci2) ||
                            (crp * crp + ci2 <= 0.0625f);
                act[k] = inset ? 0u : 0xffffffffu;
                nact += inset ? 0 : 1;
            }
            if (nact) {
                mandel4(cr, ci, max_iter, cnt, act);
                for (int k = 0; k < 4; ++k)
                    orow[x + k] = act[k] ? (uint16_t)cnt[k] : (uint16_t)max_iter;
            } else {
                for (int k = 0; k < 4; ++k)
                    orow[x + k] = (uint16_t)max_iter;
            }
        }
        for (; x < width; ++x) { // scalar tail (width % 4 != 0)
            float c = -2.5f + (float)x * dx;
            float crm = c - 0.25f;
            float q = crm * crm + ci2;
            float crp = c + 1.0f;
            if (q * (q + crm) <= 0.25f * ci2 || crp * crp + ci2 <= 0.0625f) {
                orow[x] = (uint16_t)max_iter;
                continue;
            }
            float zr = 0.f, zi = 0.f;
            int it = 0;
            for (; it < max_iter; ++it) {
                float a = zr * zr, b = zi * zi;
                if (a + b > 4.0f) break;
                float nzr = a - b + c;
                zi = 2.f * zr * zi + ci;
                zr = nzr;
            }
            orow[x] = (uint16_t)it;
        }

        const size_t mrow = (size_t)height - 1 - row;
        if (mrow != row) // guard the middle row when height is odd
            memcpy(out + mrow * (size_t)width, orow,
                   (size_t)width * sizeof(uint16_t));
    });
}
