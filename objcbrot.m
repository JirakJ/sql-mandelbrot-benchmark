// ObjCBrot - Objective-C + libdispatch Mandelbrot, Apple-heritage entry.
//
// Hot kernel is plain C float32 SIMD via clang ext_vector_type (NO arm_neon.h
// intrinsics - that's cppbrot's shtick): comparisons on ext vectors return
// int-vector masks (-1/0), so escape counting is `count -= active` with an
// amortized __builtin_reduce_or any-check every 4th iteration. Two independent
// 8-lane groups per strip hide FMA latency. Cardioid + period-2 bulb early-out,
// y-axis symmetry via memcpy mirror. Rows fan across all cores with
// dispatch_apply from an ObjC method on an NSObject worker.
//
// Semantics: value = iterations survived before |z|^2 > 4, capped at max_iter.

#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <string.h>

typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));

#define G 2 // independent vector groups (16 pixels per inner iteration)

static void render_row(int width, int max_iter, float x0, float dx, float ci,
                       uint16_t *orow) {
    const f32x8 lane = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    const float ci2s = ci * ci;
    const i32x8 vmax = (i32x8)max_iter; // scalar cast splats

    int x = 0;
    for (; x + 8 * G <= width; x += 8 * G) {
        f32x8 cr[G], zr[G], zi[G], zr2[G], zi2[G];
        i32x8 in_set[G], active[G], count[G];

#pragma clang loop unroll(full)
        for (int k = 0; k < G; ++k) {
            f32x8 xv = (float)(x + 8 * k) + lane;
            cr[k] = x0 + xv * dx;
            // main cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2, q=(cr-1/4)^2+ci^2
            f32x8 crm = cr[k] - 0.25f;
            f32x8 q = crm * crm + ci2s;
            i32x8 card = q * (q + crm) <= 0.25f * ci2s;
            // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
            f32x8 crp = cr[k] + 1.0f;
            i32x8 bulb = crp * crp + ci2s <= 0.0625f;
            in_set[k] = card | bulb;
            active[k] = ~in_set[k];
            zr[k] = (f32x8)0.f;
            zi[k] = (f32x8)0.f;
            count[k] = (i32x8)0;
        }

        for (int it = 0; it < max_iter; ++it) {
#pragma clang loop unroll(full)
            for (int k = 0; k < G; ++k) {
                zr2[k] = zr[k] * zr[k];
                zi2[k] = zi[k] * zi[k];
                active[k] &= (zr2[k] + zi2[k] <= 4.0f);
                count[k] -= active[k]; // mask is -1: +1 where active
            }
            // horizontal reduce + branch amortized to every 4th iteration;
            // dead lanes run masked (inf/nan compares never re-activate)
            if ((it & 3) == 3) {
                i32x8 any = active[0];
#pragma clang loop unroll(full)
                for (int k = 1; k < G; ++k) any |= active[k];
                if (!__builtin_reduce_or(any)) break;
            }
#pragma clang loop unroll(full)
            for (int k = 0; k < G; ++k) {
                f32x8 nzr = zr2[k] - zi2[k] + cr[k];
                zi[k] = 2.0f * zr[k] * zi[k] + ci;
                zr[k] = nzr;
            }
        }

#pragma clang loop unroll(full)
        for (int k = 0; k < G; ++k) {
            i32x8 cnt = (count[k] & ~in_set[k]) | (vmax & in_set[k]);
            u16x8 v = __builtin_convertvector(cnt, u16x8);
            memcpy(orow + x + 8 * k, &v, sizeof v); // unaligned store
        }
    }
    // scalar tail (width not a multiple of 8*G)
    for (; x < width; ++x) {
        float crs = x0 + (float)x * dx;
        float zrs = 0.f, zis = 0.f;
        int it = 0;
        for (; it < max_iter; ++it) {
            float a = zrs * zrs, b = zis * zis;
            if (a + b > 4.0f) break;
            float nzr = a - b + crs;
            zis = 2.f * zrs * zis + ci;
            zrs = nzr;
        }
        orow[x] = (uint16_t)it;
    }
}

@interface ObjCBrot : NSObject
- (void)renderWidth:(int)width
             height:(int)height
            maxIter:(int)maxIter
               into:(uint16_t *)out;
@end

@implementation ObjCBrot

- (void)renderWidth:(int)width
             height:(int)height
            maxIter:(int)maxIter
               into:(uint16_t *)out {
    const float dx = 3.5f / (float)(width - 1);
    const float dy = 2.0f / (float)(height - 1);
    const float x0 = -2.5f;
    const float y0 = -1.0f;

    // Rows r and height-1-r have opposite ci -> identical escape counts:
    // compute the top half only, memcpy-mirror the rest.
    const size_t half = ((size_t)height + 1) / 2;

    dispatch_apply(half,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t row) {
        const float ci = y0 + (float)row * dy;
        uint16_t *orow = out + row * (size_t)width;
        render_row(width, maxIter, x0, dx, ci, orow);
        const size_t mrow = (size_t)height - 1 - row;
        if (mrow != row)
            memcpy(out + mrow * (size_t)width, orow,
                   (size_t)width * sizeof(uint16_t));
    });
}

@end

void mandelbrot_objc(int width, int height, int max_iter, uint16_t *out) {
    @autoreleasepool {
        static ObjCBrot *worker;
        static dispatch_once_t once;
        dispatch_once(&once, ^{ worker = [[ObjCBrot alloc] init]; });
        [worker renderWidth:width height:height maxIter:max_iter into:out];
    }
}
