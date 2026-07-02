// Hybrid4Brot - CPU + GPU computing ONE frame concurrently, GPU via Metal 4.
//
// Same idea as hybridbrot.mm: the GPU renders rows [0, K) of the top half while
// the clang ext_vector float8 CPU kernel renders rows [K, halfH) into the SAME
// shared MTLBuffer, then one wait. Here the GPU half is driven through the WWDC
// 2025/2026 Metal 4 command model (MTL4CommandQueue / MTL4CommandAllocator /
// MTL4CommandBuffer / MTL4ArgumentTable bound by gpuAddress / MTLResidencySet,
// pipeline built by the MTL4 compiler), to test whether the lower GPU submit
// latency lets the hybrid beat the Metal 3 record.
//
// CRITICAL: the GPU commit stays ASYNC. We commit the Metal 4 command buffer,
// signal an MTLSharedEvent on the queue, then run the CPU dispatch_apply while
// the GPU works, and only block on the event at the very end - mirroring the
// Metal 3 hybrid's commit -> CPU -> wait ordering.
//
// Semantics match pybrot.py: count = iterations survived, in-set = max_iter.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdint>
#include <cstring>

static id<MTLDevice>               g_dev;
static id<MTL4CommandQueue>        g_q;
static id<MTL4CommandAllocator>    g_alloc;
static id<MTL4CommandBuffer>       g_cb;
static id<MTLComputePipelineState> g_pso;       // mandel_range
static id<MTL4ArgumentTable>       g_argTable;
static id<MTLResidencySet>         g_resSet;
static id<MTLSharedEvent>          g_event;
static id<MTLBuffer>               g_params;     // uint4 {w,h,maxIter,K}, shared
static id<MTLBuffer>               g_out;        // output, shared
static size_t                      g_cap;
static uint64_t                    g_evVal;
static NSUInteger                  g_tew;        // threadExecutionWidth (32)

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
)MSL";

extern "C" int hybrid4_init(void) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;

        NSError *err = nil;
        g_q = [g_dev newMTL4CommandQueue];
        if (!g_q) return 0;
        g_alloc = [g_dev newCommandAllocator];
        if (!g_alloc) return 0;
        g_cb = [g_dev newCommandBuffer];
        if (!g_cb) return 0;

        // Pipeline via the Metal 4 compiler.
        MTL4CompilerDescriptor *cdesc = [MTL4CompilerDescriptor new];
        id<MTL4Compiler> compiler = [g_dev newCompilerWithDescriptor:cdesc error:&err];
        if (!compiler) return 0;
        MTL4LibraryDescriptor *ldesc = [MTL4LibraryDescriptor new];
        ldesc.source = [NSString stringWithUTF8String:kSrc];
        id<MTLLibrary> lib = [compiler newLibraryWithDescriptor:ldesc error:&err];
        if (!lib) return 0;
        MTL4LibraryFunctionDescriptor *fdesc = [MTL4LibraryFunctionDescriptor new];
        fdesc.name = @"mandel_range";
        fdesc.library = lib;
        MTL4ComputePipelineDescriptor *pdesc = [MTL4ComputePipelineDescriptor new];
        pdesc.computeFunctionDescriptor = fdesc;
        g_pso = [compiler newComputePipelineStateWithDescriptor:pdesc
                                            compilerTaskOptions:nil
                                                          error:&err];
        if (!g_pso) return 0;
        g_tew = g_pso.threadExecutionWidth;

        // Argument table (bind by gpuAddress).
        MTL4ArgumentTableDescriptor *adesc = [MTL4ArgumentTableDescriptor new];
        adesc.maxBufferBindCount = 2;
        g_argTable = [g_dev newArgumentTableWithDescriptor:adesc error:&err];
        if (!g_argTable) return 0;

        // Params buffer (shared, uint4).
        g_params = [g_dev newBufferWithLength:sizeof(uint32_t) * 4
                                     options:MTLResourceStorageModeShared |
                                             MTLResourceHazardTrackingModeUntracked];
        if (!g_params) return 0;

        // Residency set replaces implicit residency.
        MTLResidencySetDescriptor *rdesc = [MTLResidencySetDescriptor new];
        rdesc.initialCapacity = 2;
        g_resSet = [g_dev newResidencySetWithDescriptor:rdesc error:&err];
        if (!g_resSet) return 0;
        [g_resSet addAllocation:g_params];
        [g_resSet commit];
        [g_resSet requestResidency];
        [g_q addResidencySet:g_resSet];

        g_event = [g_dev newSharedEvent];
        if (!g_event) return 0;
        g_evVal = 0;

        return 1;
    }
}

// ---- CPU kernel: clang ext_vector float8, G=2 groups (16 px per iteration) ----
// (Copied verbatim from hybridbrot.mm to keep the CPU half identical.)

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

// GPU takes rows [0, K) of the top half, CPU takes [K, halfH), both mirror.
// GPU commit is ASYNC: commit + signal, then CPU works, then wait on the event.
extern "C" uint16_t *mandelbrot_hybrid4(int width, int height, int max_iter,
                                        int split_pct) {
    @autoreleasepool {
        const size_t bytes = (size_t)width * (size_t)height * sizeof(uint16_t);
        if (!g_out || g_cap < bytes) {
            if (g_out) [g_resSet removeAllocation:g_out];
            g_out = [g_dev newBufferWithLength:bytes
                                      options:MTLResourceStorageModeShared |
                                              MTLResourceHazardTrackingModeUntracked];
            g_cap = bytes;
            [g_resSet addAllocation:g_out];
            [g_resSet commit];
            [g_resSet requestResidency];
        }

        const uint32_t halfH = ((uint32_t)height + 1) / 2;
        uint32_t K = (uint32_t)((uint64_t)halfH * (uint32_t)split_pct / 100);
        if (K > halfH) K = halfH;

        uint32_t *pp = (uint32_t *)g_params.contents;
        pp[0] = (uint32_t)width; pp[1] = (uint32_t)height;
        pp[2] = (uint32_t)max_iter; pp[3] = K;

        // Encode the GPU half (rows [0, K)).
        [g_alloc reset];
        [g_cb beginCommandBufferWithAllocator:g_alloc];
        if (K > 0) {
            id<MTL4ComputeCommandEncoder> enc = [g_cb computeCommandEncoder];
            [enc setComputePipelineState:g_pso];
            [g_argTable setAddress:g_out.gpuAddress atIndex:0];
            [g_argTable setAddress:g_params.gpuAddress atIndex:1];
            [enc setArgumentTable:g_argTable];
            [enc dispatchThreads:MTLSizeMake((NSUInteger)width, K, 1)
           threadsPerThreadgroup:MTLSizeMake(g_tew, 8, 1)];
            [enc endEncoding];
        }
        [g_cb endCommandBuffer];

        // ASYNC commit + signal: GPU renders while the CPU works below.
        g_evVal++;
        [g_q commit:&g_cb count:1];
        [g_q signalEvent:g_event value:g_evVal];

        // CPU half (rows [K, halfH)) concurrent with the GPU.
        if (K < halfH) {
            RowJob job = {(uint16_t *)g_out.contents, width, height, max_iter,
                          (int)K};
            dispatch_apply_f(halfH - K,
                             dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                             &job, cpu_row);
        }

        // Now block until the GPU half is done.
        [g_event waitUntilSignaledValue:g_evVal timeoutMS:10000];
        return (uint16_t *)g_out.contents;
    }
}
