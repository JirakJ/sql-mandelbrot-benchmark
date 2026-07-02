// HybridBrot - CPU + GPU computing ONE frame concurrently.
//
// Unified memory makes this possible: the 40-core GPU and the 14-core CPU
// write disjoint row ranges of the SAME shared MTLBuffer, no copies, no
// synchronization beyond one waitUntilCompleted at the end.
//
// - GPU: Metal compute kernel takes rows [0, K) of the top half (async commit).
// - CPU: clang ext_vector float8 kernel (the fastest CPU approach measured in
//   this repo — see objcbrot.m) takes rows [K, halfH) via dispatch_apply while
//   the GPU works.
// - Both write their conjugate mirror rows too (y-symmetry).
// - split_pct tunes the GPU share of the top half.
//
// Semantics match pybrot.py: count = iterations survived, in-set = max_iter.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdint>
#include <cstring>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_pso;
static id<MTLComputePipelineState> g_pso_strided;
static id<MTLBuffer> g_out;
static size_t g_cap;

static const char *kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void mandel_range(device ushort *out [[buffer(0)]],
                         constant uint4 &p  [[buffer(1)]], // w, h, maxIter, rows
                         uint2 gid          [[thread_position_in_grid]])
{
    const uint w = p.x, h = p.y, maxIter = p.z;
    if (gid.x >= w || gid.y >= p.w) return;

    const float dx = 3.5f / (float)(w - 1);
    const float dy = 2.0f / (float)(h - 1);
    const float cr = -2.5f + (float)gid.x * dx;
    const float ci = -1.0f + (float)gid.y * dy;

    const float ci2 = ci * ci;
    const float crm = cr - 0.25f;
    const float q = crm * crm + ci2;
    const float crp = cr + 1.0f;

    uint it;
    if (q * (q + crm) <= 0.25f * ci2 || crp * crp + ci2 <= 0.0625f) {
        it = maxIter;
    } else {
        float zr = 0.0f, zi = 0.0f;
        for (it = 0; it < maxIter; ++it) {
            float zr2 = zr * zr, zi2 = zi * zi;
            if (zr2 + zi2 > 4.0f) break;
            float nzr = zr2 - zi2 + cr;
            zi = 2.0f * zr * zi + ci;
            zr = nzr;
        }
    }
    out[gid.y * w + gid.x] = (ushort)it;
    out[(h - 1 - gid.y) * w + gid.x] = (ushort)it;
}

// Strided variant: GPU owns slots [0, slots) of every 100-row window, so both
// engines get a uniform mix of cheap and expensive rows.
kernel void mandel_strided(device ushort *out [[buffer(0)]],
                           constant uint4 &p  [[buffer(1)]], // w, h, maxIter, slots
                           uint2 gid          [[thread_position_in_grid]])
{
    const uint w = p.x, h = p.y, maxIter = p.z, slots = p.w;
    const uint halfH = (h + 1) / 2;
    const uint row = (gid.y / slots) * 100 + (gid.y % slots);
    if (gid.x >= w || row >= halfH) return;

    const float dx = 3.5f / (float)(w - 1);
    const float dy = 2.0f / (float)(h - 1);
    const float cr = -2.5f + (float)gid.x * dx;
    const float ci = -1.0f + (float)row * dy;

    const float ci2 = ci * ci;
    const float crm = cr - 0.25f;
    const float q = crm * crm + ci2;
    const float crp = cr + 1.0f;

    uint it;
    if (q * (q + crm) <= 0.25f * ci2 || crp * crp + ci2 <= 0.0625f) {
        it = maxIter;
    } else {
        float zr = 0.0f, zi = 0.0f;
        for (it = 0; it < maxIter; ++it) {
            float zr2 = zr * zr, zi2 = zi * zi;
            if (zr2 + zi2 > 4.0f) break;
            float nzr = zr2 - zi2 + cr;
            zi = 2.0f * zr * zi + ci;
            zr = nzr;
        }
    }
    out[row * w + gid.x] = (ushort)it;
    out[(h - 1 - row) * w + gid.x] = (ushort)it;
}
)MSL";

extern "C" int hybrid_init(void) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;
        g_queue = [g_dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib =
            [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                options:[MTLCompileOptions new]
                                  error:&err];
        if (!lib) return 0;
        g_pso = [g_dev newComputePipelineStateWithFunction:
                           [lib newFunctionWithName:@"mandel_range"] error:&err];
        g_pso_strided = [g_dev newComputePipelineStateWithFunction:
                             [lib newFunctionWithName:@"mandel_strided"] error:&err];
        return g_pso != nil && g_pso_strided != nil;
    }
}

// ---- CPU kernel: clang ext_vector float8, G=2 groups (16 px per iteration) ----

typedef float float8 __attribute__((ext_vector_type(8)));
typedef int int8v __attribute__((ext_vector_type(8)));

struct RowJob {
    uint16_t *out;
    int width, height, max_iter, row0;
};

static void cpu_row(void *ctx, size_t idx) {
    const RowJob *j = (const RowJob *)ctx;
    const int w = j->width, h = j->height, max_iter = j->max_iter;
    const int row = j->row0 + (int)idx;
    const float dx = 3.5f / (float)(w - 1);
    const float cy = -1.0f + (float)row * (2.0f / (float)(h - 1));
    uint16_t *orow = j->out + (size_t)row * w;

    const float ci2s = cy * cy;
    int x = 0;
    for (; x + 16 <= w; x += 16) {
        float8 crq[2], zr[2], zi[2], zr2[2], zi2[2];
        int8v inset[2], active[2], count[2];
        for (int k = 0; k < 2; ++k) {
            float8 xv;
            for (int l = 0; l < 8; ++l) xv[l] = (float)(x + 8 * k + l);
            crq[k] = -2.5f + xv * dx;
            float8 crm = crq[k] - 0.25f;
            float8 q = crm * crm + ci2s;
            float8 crp = crq[k] + 1.0f;
            inset[k] = (q * (q + crm) <= 0.25f * ci2s) | (crp * crp + ci2s <= 0.0625f);
            active[k] = ~inset[k];
            zr[k] = 0.0f; zi[k] = 0.0f; count[k] = 0;
        }
        for (int it = 0; it < max_iter; ++it) {
            for (int k = 0; k < 2; ++k) {
                zr2[k] = zr[k] * zr[k];
                zi2[k] = zi[k] * zi[k];
                active[k] &= (zr2[k] + zi2[k] <= 4.0f);
                count[k] -= active[k];
            }
            if ((it & 3) == 3 &&
                !__builtin_reduce_or(active[0] | active[1])) break;
            for (int k = 0; k < 2; ++k) {
                float8 nzr = zr2[k] - zi2[k] + crq[k];
                zi[k] = 2.0f * zr[k] * zi[k] + cy;
                zr[k] = nzr;
            }
        }
        for (int k = 0; k < 2; ++k) {
            int8v cnt = (inset[k] & max_iter) | (~inset[k] & count[k]);
            for (int l = 0; l < 8; ++l) orow[x + 8 * k + l] = (uint16_t)cnt[l];
        }
    }
    for (; x < w; ++x) {
        float cr = -2.5f + (float)x * dx;
        float a = 0.f, b = 0.f;
        int it = 0;
        for (; it < max_iter; ++it) {
            float a2 = a * a, b2 = b * b;
            if (a2 + b2 > 4.0f) break;
            float na = a2 - b2 + cr;
            b = 2.f * a * b + cy;
            a = na;
        }
        orow[x] = (uint16_t)it;
    }
    // mirror
    const int mrow = h - 1 - row;
    if (mrow != row)
        memcpy(j->out + (size_t)mrow * w, orow, (size_t)w * sizeof(uint16_t));
}

// Contiguous split gives the GPU the light edge rows and the CPU the heavy
// near-axis rows (or vice versa) — poor balance. Strided assignment hands both
// engines a uniform mix of cheap and expensive rows: GPU gets stride-slots
// [0, gpuSlots) of every 100-row window, CPU the rest.
struct StridedJob {
    uint16_t *out;
    int width, height, max_iter, gpu_slots;
};

static void cpu_row_strided(void *ctx, size_t idx) {
    const StridedJob *sj = (const StridedJob *)ctx;
    const uint32_t halfH = ((uint32_t)sj->height + 1) / 2;
    // idx enumerates CPU rows: those with (row % 100) >= gpu_slots
    uint32_t cpu_per_win = 100 - (uint32_t)sj->gpu_slots;
    uint32_t win = (uint32_t)idx / cpu_per_win;
    uint32_t off = (uint32_t)idx % cpu_per_win;
    uint32_t row = win * 100 + (uint32_t)sj->gpu_slots + off;
    if (row >= halfH) return;
    RowJob j = {sj->out, sj->width, sj->height, sj->max_iter, (int)row};
    cpu_row(&j, 0);
}

// GPU takes rows [0, K) of the top half, CPU takes [K, halfH), both mirror.
extern "C" uint16_t *mandelbrot_hybrid(int width, int height, int max_iter,
                                       int split_pct) {
    @autoreleasepool {
        const size_t bytes = (size_t)width * (size_t)height * sizeof(uint16_t);
        if (!g_out || g_cap < bytes) {
            g_out = [g_dev newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared |
                                               MTLResourceHazardTrackingModeUntracked];
            g_cap = bytes;
        }
        const uint32_t halfH = ((uint32_t)height + 1) / 2;
        uint32_t K = (uint32_t)((uint64_t)halfH * (uint32_t)split_pct / 100);
        if (K > halfH) K = halfH;

        id<MTLCommandBuffer> cb = [g_queue commandBufferWithUnretainedReferences];
        if (K > 0) {
            const uint32_t params[4] = {(uint32_t)width, (uint32_t)height,
                                        (uint32_t)max_iter, K};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g_pso];
            [enc setBuffer:g_out offset:0 atIndex:0];
            [enc setBytes:params length:sizeof(params) atIndex:1];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)width, K, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
            [enc endEncoding];
        }
        [cb commit]; // async — CPU works while GPU renders

        if (K < halfH) {
            RowJob job = {(uint16_t *)g_out.contents, width, height, max_iter,
                          (int)K};
            dispatch_apply_f(halfH - K,
                             dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                             &job, cpu_row);
        }
        [cb waitUntilCompleted];
        return (uint16_t *)g_out.contents;
    }
}

// Strided split: GPU owns `slots` of every 100 top-half rows, CPU the rest.
extern "C" uint16_t *mandelbrot_hybrid_strided(int width, int height,
                                               int max_iter, int slots) {
    @autoreleasepool {
        const size_t bytes = (size_t)width * (size_t)height * sizeof(uint16_t);
        if (!g_out || g_cap < bytes) {
            g_out = [g_dev newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared |
                                               MTLResourceHazardTrackingModeUntracked];
            g_cap = bytes;
        }
        const uint32_t halfH = ((uint32_t)height + 1) / 2;
        if (slots < 0) slots = 0;
        if (slots > 100) slots = 100;
        const uint32_t wins = halfH / 100, rem = halfH % 100;
        const uint32_t gpuRows =
            wins * (uint32_t)slots + (rem < (uint32_t)slots ? rem : (uint32_t)slots);

        id<MTLCommandBuffer> cb = [g_queue commandBufferWithUnretainedReferences];
        if (gpuRows > 0) {
            const uint32_t params[4] = {(uint32_t)width, (uint32_t)height,
                                        (uint32_t)max_iter, (uint32_t)slots};
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:g_pso_strided];
            [enc setBuffer:g_out offset:0 atIndex:0];
            [enc setBytes:params length:sizeof(params) atIndex:1];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)width, gpuRows, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
            [enc endEncoding];
        }
        [cb commit];

        const uint32_t cpuPerWin = 100 - (uint32_t)slots;
        const uint32_t cpuDispatch = ((halfH + 99) / 100) * cpuPerWin;
        if (cpuDispatch > 0) {
            StridedJob job = {(uint16_t *)g_out.contents, width, height,
                              max_iter, slots};
            dispatch_apply_f(cpuDispatch,
                             dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                             &job, cpu_row_strided);
        }
        [cb waitUntilCompleted];
        return (uint16_t *)g_out.contents;
    }
}
