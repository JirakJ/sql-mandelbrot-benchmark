// Metal4Brot - Mandelbrot on the Apple Silicon GPU via the Metal 4 command model.
//
// Same MSL kernel and math as metalbrot.mm (identical iteration + y-mirror write),
// but driven through the WWDC 2025/2026 Metal 4 command submission API instead of
// the Metal 3 one, to test whether the new command model shaves fixed submit/execute
// latency off a tiny single-dispatch workload.
//
// Metal 4 pieces used (vs Metal 3 equivalents):
//   Metal 3                         Metal 4 (this file)
//   ------------------------------  --------------------------------------------
//   [dev newCommandQueue]           [dev newMTL4CommandQueue]              (MTL4CommandQueue)
//   [queue commandBuffer]           [dev newCommandBuffer] +              (MTL4CommandBuffer)
//                                     beginCommandBufferWithAllocator:    (MTL4CommandAllocator)
//   [enc setBuffer:.. atIndex:]     MTL4ArgumentTable setAddress:atIndex: (bind by gpuAddress)
//   [enc setBytes:.. atIndex:]      params in a shared MTLBuffer, bound by gpuAddress
//   implicit residency              MTLResidencySet added to the queue
//   [cb commit]/[cb waitUntil..]    [queue commit:count:] + [queue signalEvent:value:]
//                                     + [MTLSharedEvent waitUntilSignaledValue:timeoutMS:]
//   newComputePipelineWithFunction  MTL4Compiler newComputePipelineStateWithDescriptor:
//
// All init happens in metal4_init (untimed). The compute path reuses the queue,
// allocator, command buffer, argument table, residency set, shared event and
// output buffer across calls.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdint>
#include <cstring>

static id<MTLDevice>              g_dev;
static id<MTL4CommandQueue>       g_q;
static id<MTL4CommandAllocator>   g_alloc;
static id<MTL4CommandBuffer>      g_cb;
static id<MTLComputePipelineState> g_pso;
static id<MTL4ArgumentTable>      g_argTable;
static id<MTLResidencySet>        g_resSet;
static id<MTLSharedEvent>         g_event;
static id<MTLBuffer>              g_params;   // 16-byte uint4, shared
static id<MTLBuffer>              g_out;      // output, shared
static size_t                    g_cap;
static uint64_t                  g_evVal;
static NSUInteger                g_tew;      // threadExecutionWidth (32 on Apple GPUs)

static const char *kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void mandel(device ushort *out       [[buffer(0)]],
                   constant uint4 &p        [[buffer(1)]], // w, h, maxIter, halfH
                   uint2 gid                [[thread_position_in_grid]])
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
        it = maxIter; // inside main cardioid / period-2 bulb
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
    out[(h - 1 - gid.y) * w + gid.x] = (ushort)it; // conjugate mirror row
}
)MSL";

extern "C" int metal4_init(void) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;

        // --- Metal 4 command model objects ---
        NSError *err = nil;
        g_q = [g_dev newMTL4CommandQueue];
        if (!g_q) return 0;
        g_alloc = [g_dev newCommandAllocator];
        if (!g_alloc) return 0;
        g_cb = [g_dev newCommandBuffer];
        if (!g_cb) return 0;

        // --- Compile the kernel + build the pipeline via the Metal 4 compiler ---
        MTL4CompilerDescriptor *cdesc = [MTL4CompilerDescriptor new];
        id<MTL4Compiler> compiler = [g_dev newCompilerWithDescriptor:cdesc error:&err];
        if (!compiler) return 0;

        MTL4LibraryDescriptor *ldesc = [MTL4LibraryDescriptor new];
        ldesc.source = [NSString stringWithUTF8String:kSrc];
        id<MTLLibrary> lib = [compiler newLibraryWithDescriptor:ldesc error:&err];
        if (!lib) return 0;

        MTL4LibraryFunctionDescriptor *fdesc = [MTL4LibraryFunctionDescriptor new];
        fdesc.name = @"mandel";
        fdesc.library = lib;

        MTL4ComputePipelineDescriptor *pdesc = [MTL4ComputePipelineDescriptor new];
        pdesc.computeFunctionDescriptor = fdesc;
        g_pso = [compiler newComputePipelineStateWithDescriptor:pdesc
                                            compilerTaskOptions:nil
                                                          error:&err];
        if (!g_pso) return 0;
        g_tew = g_pso.threadExecutionWidth;

        // --- Argument table (binds buffers by GPU address, not setBytes) ---
        MTL4ArgumentTableDescriptor *adesc = [MTL4ArgumentTableDescriptor new];
        adesc.maxBufferBindCount = 2;
        g_argTable = [g_dev newArgumentTableWithDescriptor:adesc error:&err];
        if (!g_argTable) return 0;

        // --- Params buffer (shared, holds the uint4) ---
        g_params = [g_dev newBufferWithLength:sizeof(uint32_t) * 4
                                     options:MTLResourceStorageModeShared |
                                             MTLResourceHazardTrackingModeUntracked];
        if (!g_params) return 0;

        // --- Residency set: replaces implicit residency in Metal 4 ---
        MTLResidencySetDescriptor *rdesc = [MTLResidencySetDescriptor new];
        rdesc.initialCapacity = 2;
        g_resSet = [g_dev newResidencySetWithDescriptor:rdesc error:&err];
        if (!g_resSet) return 0;
        [g_resSet addAllocation:g_params];
        [g_resSet commit];
        [g_resSet requestResidency];
        [g_q addResidencySet:g_resSet];

        // --- Shared event for CPU/GPU completion sync (no waitUntilCompleted in MTL4) ---
        g_event = [g_dev newSharedEvent];
        if (!g_event) return 0;
        g_evVal = 0;

        return 1;
    }
}

// Runs the kernel into the persistent shared buffer and returns its pointer
// (unified memory: CPU-visible without copying). Buffer reused across calls.
extern "C" uint16_t *mandelbrot_metal4_ptr(int width, int height, int max_iter) {
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
        uint32_t *pp = (uint32_t *)g_params.contents;
        pp[0] = (uint32_t)width; pp[1] = (uint32_t)height;
        pp[2] = (uint32_t)max_iter; pp[3] = halfH;

        // Reset the allocator (prior work is complete: we block on the event each call).
        [g_alloc reset];
        [g_cb beginCommandBufferWithAllocator:g_alloc];

        id<MTL4ComputeCommandEncoder> enc = [g_cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso];
        [g_argTable setAddress:g_out.gpuAddress atIndex:0];
        [g_argTable setAddress:g_params.gpuAddress atIndex:1];
        [enc setArgumentTable:g_argTable];

#ifndef TGH
#define TGH 8
#endif
        MTLSize tg = MTLSizeMake(g_tew, TGH, 1);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)width, halfH, 1)
       threadsPerThreadgroup:tg];
        [enc endEncoding];
        [g_cb endCommandBuffer];

        g_evVal++;
        [g_q commit:&g_cb count:1];
        [g_q signalEvent:g_event value:g_evVal];
        [g_event waitUntilSignaledValue:g_evVal timeoutMS:10000];

        return (uint16_t *)g_out.contents;
    }
}

// Copying variant kept for callers that own their buffer (used for warm-up).
extern "C" void mandelbrot_metal4(int width, int height, int max_iter, uint16_t *out) {
    uint16_t *src = mandelbrot_metal4_ptr(width, height, max_iter);
    memcpy(out, src, (size_t)width * (size_t)height * sizeof(uint16_t));
}
