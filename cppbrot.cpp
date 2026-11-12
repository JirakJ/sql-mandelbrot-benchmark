// CppBrot - fastest Mandelbrot on Apple Silicon: C++ + ARM NEON + GCD threads.
//
// - float32x4 NEON: 4 pixels/vector, FMA, masked escape-time counting.
// - G independent vector groups interleaved (4*G pixels/iter): the z = z^2 + c
//   recurrence is a latency-bound dependency chain, so running G chains in
//   parallel hides FP/FMA latency and saturates the M-series vector pipes.
// - y-axis symmetry: the fixed viewport (-2.5..1) x (-1..1) is symmetric about
//   the real axis and Mandelbrot escape counts are conjugation-invariant
//   (conj(z^2+c) = conj(z)^2 + conj(c), and IEEE negation is exact), so only
//   the top half of the rows is computed; the rest is a memcpy mirror.
// - amortized escape reduction: the per-iteration horizontal vmaxvq + branch is
//   the serializing op; it runs every 4th iteration. Per-lane escape counts are
//   still exact (dead lanes are masked out; inf/nan compares are always false).
// - libdispatch (dispatch_apply): rows fanned across all P+E cores, work-stealing.
// - cardioid + period-2 bulb early-out: the big in-set regions (the expensive
//   full-max_iter pixels) are skipped analytically.
//
// float32 is exact enough for the default view (pixel pitch ~2.5e-3 >> f32 eps);
// the image is visually identical to the f64 reference implementations.
//
// Semantics match pybrot.py: value = iterations survived before |z|^2 > 4
// (check-then-iterate), capped at max_iter (in-set pixels -> max_iter).

#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <cstdint>
#include <cstddef>
#include <cstring>

#ifndef G
#define G 2 // independent vector groups (4*G pixels per inner iteration)
#endif
#ifndef QOS
#define QOS QOS_CLASS_USER_INITIATED
#endif

extern "C" void mandelbrot_neon(int width, int height, int max_iter, uint16_t *out) {
    const float dx = 3.5f / (float)(width - 1);
    const float dy = 2.0f / (float)(height - 1);
    const float x0 = -2.5f;
    const float y0 = -1.0f;

    const float32x4_t lane = {0.f, 1.f, 2.f, 3.f};
    const float32x4_t four = vdupq_n_f32(4.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);
    const float32x4_t vdx = vdupq_n_f32(dx);
    const float32x4_t vx0 = vdupq_n_f32(x0);
    const float32x4_t quarter = vdupq_n_f32(0.25f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t bulb_r2 = vdupq_n_f32(0.0625f); // (1/4)^2
    const uint32x4_t vmax = vdupq_n_u32((uint32_t)max_iter);

    // Rows r and height-1-r have exactly opposite cy up to 1 ulp of grid
    // rounding, and opposite cy gives bit-identical escape counts.
    const size_t half = ((size_t)height + 1) / 2;

    dispatch_apply(half,
                   dispatch_get_global_queue(QOS, 0),
                   ^(size_t row) {
        const float cy = y0 + (float)row * dy;
        const float32x4_t ciq = vdupq_n_f32(cy);
        const float32x4_t ci2 = vmulq_f32(ciq, ciq);
        uint16_t *orow = out + row * (size_t)width;

        int x = 0;
        for (; x + 4 * G <= width; x += 4 * G) {
            float32x4_t crq[G], zr[G], zi[G], zr2[G], zi2[G];
            uint32x4_t in_set[G], active[G], count[G];

            #pragma clang loop unroll(full)
            for (int k = 0; k < G; ++k) {
                float32x4_t xv = vaddq_f32(vdupq_n_f32((float)(x + 4 * k)), lane);
                crq[k] = vmlaq_f32(vx0, xv, vdx); // cr = x0 + (x+4k+lane)*dx
                // main cardioid: q*(q + (cr-1/4)) <= (1/4) ci^2, q=(cr-1/4)^2+ci^2
                float32x4_t crm = vsubq_f32(crq[k], quarter);
                float32x4_t q = vmlaq_f32(ci2, crm, crm);
                uint32x4_t card =
                    vcleq_f32(vmulq_f32(q, vaddq_f32(q, crm)), vmulq_f32(quarter, ci2));
                // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
                float32x4_t crp = vaddq_f32(crq[k], one);
                uint32x4_t bulb = vcleq_f32(vmlaq_f32(ci2, crp, crp), bulb_r2);
                in_set[k] = vorrq_u32(card, bulb);
                active[k] = vmvnq_u32(in_set[k]);
                zr[k] = vdupq_n_f32(0.f);
                zi[k] = vdupq_n_f32(0.f);
                count[k] = vdupq_n_u32(0);
            }

            for (int it = 0; it < max_iter; ++it) {
                #pragma clang loop unroll(full)
                for (int k = 0; k < G; ++k) {
                    zr2[k] = vmulq_f32(zr[k], zr[k]);
                    zi2[k] = vmulq_f32(zi[k], zi[k]);
                    float32x4_t mag2 = vaddq_f32(zr2[k], zi2[k]);
                    active[k] = vandq_u32(active[k], vcleq_f32(mag2, four));
                    count[k] = vsubq_u32(count[k], active[k]); // +1 where active
                }
                // horizontal reduce + branch only every 4th iteration; dead
                // lanes just run masked (inf/nan never re-activates them)
                if ((it & 3) == 3) {
                    uint32x4_t any = active[0];
                    #pragma clang loop unroll(full)
                    for (int k = 1; k < G; ++k) any = vorrq_u32(any, active[k]);
                    if (vmaxvq_u32(any) == 0) break;
                }
                #pragma clang loop unroll(full)
                for (int k = 0; k < G; ++k) {
                    float32x4_t nzr = vaddq_f32(vsubq_f32(zr2[k], zi2[k]), crq[k]);
                    zi[k] = vmlaq_f32(ciq, vmulq_f32(two, zr[k]), zi[k]); // 2*zr*zi+ci
                    zr[k] = nzr;
                }
            }

            #pragma clang loop unroll(full)
            for (int k = 0; k < G; ++k) {
                uint32x4_t cnt = vbslq_u32(in_set[k], vmax, count[k]);
                vst1_u16(orow + x + 4 * k, vmovn_u32(cnt));
            }
        }
        // scalar tail (width not a multiple of 4*G)
        for (; x < width; ++x) {
            float cr = x0 + (float)x * dx;
            float zrs = 0.f, zis = 0.f;
            int it = 0;
            for (; it < max_iter; ++it) {
                float a = zrs * zrs, b = zis * zis;
                if (a + b > 4.0f) break;
                float nzr = a - b + cr;
                zis = 2.f * zrs * zis + cy;
                zrs = nzr;
            }
            orow[x] = (uint16_t)it;
        }

        // mirror to the conjugate row
        const size_t mrow = (size_t)height - 1 - row;
        if (mrow != row)
            memcpy(out + mrow * (size_t)width, orow, (size_t)width * sizeof(uint16_t));
    });
}
