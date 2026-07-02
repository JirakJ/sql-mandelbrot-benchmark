// MetalBrot - Mandelbrot on the Apple Silicon GPU (Metal compute shader).
//
// - Shader compiled at runtime via newLibraryWithSource (no metallib toolchain).
// - Unified memory: MTLResourceStorageModeShared output buffer, zero-copy for
//   the GPU, one memcpy back to the caller's numpy buffer.
// - y-axis symmetry: grid covers only the top half of rows; each thread writes
//   its pixel and the conjugate mirror pixel.
// - cardioid + period-2 bulb early-out, same as the CPU kernel.
// - Device/queue/pipeline are created once in metal_init (untimed, at import).
//
// Semantics match pybrot.py: value = iterations survived before |z|^2 > 4,
// capped at max_iter.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdint>
#include <cstring>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_pso;
static id<MTLBuffer> g_out;
static size_t g_cap;

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

extern "C" int metal_init(void) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;
        g_queue = [g_dev newCommandQueue];
        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        id<MTLLibrary> lib =
            [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                options:opts
                                  error:&err];
        if (!lib) return 0;
        id<MTLFunction> fn = [lib newFunctionWithName:@"mandel"];
        g_pso = [g_dev newComputePipelineStateWithFunction:fn error:&err];
        return g_pso != nil;
    }
}

// Runs the kernel into the persistent shared buffer and returns its pointer
// (unified memory: CPU-visible without copying). Buffer is reused across calls.
extern "C" uint16_t *mandelbrot_metal_ptr(int width, int height, int max_iter) {
    @autoreleasepool {
        const size_t bytes = (size_t)width * (size_t)height * sizeof(uint16_t);
        if (!g_out || g_cap < bytes) {
            g_out = [g_dev newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared |
                                               MTLResourceHazardTrackingModeUntracked];
            g_cap = bytes;
        }
        const uint32_t halfH = ((uint32_t)height + 1) / 2;
        const uint32_t params[4] = {(uint32_t)width, (uint32_t)height,
                                    (uint32_t)max_iter, halfH};

        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:g_pso];
        [enc setBuffer:g_out offset:0 atIndex:0];
        [enc setBytes:params length:sizeof(params) atIndex:1];

        const NSUInteger tew = g_pso.threadExecutionWidth; // 32 on Apple GPUs
#ifndef TGH
#define TGH 8
#endif
        MTLSize tg = MTLSizeMake(tew, TGH, 1);
        [enc dispatchThreads:MTLSizeMake((NSUInteger)width, halfH, 1)
            threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        return (uint16_t *)g_out.contents;
    }
}

// Copying variant kept for callers that own their buffer.
extern "C" void mandelbrot_metal(int width, int height, int max_iter, uint16_t *out) {
    uint16_t *src = mandelbrot_metal_ptr(width, height, max_iter);
    memcpy(out, src, (size_t)width * (size_t)height * sizeof(uint16_t));
}
