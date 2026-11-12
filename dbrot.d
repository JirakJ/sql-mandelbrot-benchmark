// DBrot - Mandelbrot in D (LDC): float4 SIMD lowered to NEON, std.parallelism.
//
// Kernel structure mirrors cppbrot.cpp:
// - __vector(float[4]) lanes, G independent groups to hide FMA latency
// - masked escape counting (count -= all-ones active mask)
// - horizontal any-active check + branch only every 4th iteration
// - cardioid + period-2 bulb analytic early-out
// - y-axis symmetry: compute top half, memcpy mirror the rest
// Semantics: value = iterations survived before |z|^2 > 4, capped at max_iter.

import core.simd;
import core.stdc.string : memcpy;
import ldc.intrinsics : llvm_fma;
import ldc.simd : greaterOrEqualMask;
import std.parallelism : defaultPoolThreads, parallel, totalCPUs;
import std.range : iota;

extern (C) int rt_init();

enum G = 2; // independent vector groups (4*G pixels per inner iteration)

// a <= b elementwise, ordered compare (NaN/inf never re-activates a lane)
pragma(inline, true)
private int4 leMask(float4 a, float4 b)
{
    return greaterOrEqualMask!float4(b, a);
}

// Called once from Python right after dlopen: boot druntime, size the pool.
extern (C) void dbrot_init()
{
    rt_init();
    defaultPoolThreads = totalCPUs > 1 ? totalCPUs - 1 : 1;
}

extern (C) void mandelbrot_d(int width, int height, int maxIter, ushort* outp)
{
    immutable float dx = 3.5f / (width - 1);
    immutable float dy = 2.0f / (height - 1);
    enum float x0 = -2.5f;
    enum float y0 = -1.0f;

    immutable float4 lane = [0.0f, 1.0f, 2.0f, 3.0f];
    immutable float4 vdx = dx;
    immutable float4 vx0 = x0;
    immutable float4 four = 4.0f;
    immutable float4 two = 2.0f;
    immutable float4 one = 1.0f;
    immutable float4 quarter = 0.25f;
    immutable float4 bulbR2 = 0.0625f; // (1/4)^2
    immutable float4 fzero = 0.0f;
    immutable int4 izero = 0;
    immutable int4 vmax = maxIter;

    // rows r and height-1-r have opposite ci -> identical escape counts
    immutable size_t half = (cast(size_t) height + 1) / 2;

    foreach (row; parallel(iota(size_t(0), half), 1))
    {
        immutable float cy = y0 + cast(float) row * dy;
        immutable float4 ciq = cy;
        immutable float4 ci2 = ciq * ciq;
        ushort* orow = outp + row * cast(size_t) width;

        int x = 0;
        for (; x + 4 * G <= width; x += 4 * G)
        {
            float4[G] crq, zr, zi, zr2, zi2;
            int4[G] inSet, active, count;

            static foreach (k; 0 .. G)
            {{
                float4 xv = cast(float)(x + 4 * k);
                xv += lane;
                crq[k] = llvm_fma(xv, vdx, vx0); // cr = x0 + (x+4k+lane)*dx
                // cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2, q=(cr-1/4)^2+ci^2
                float4 crm = crq[k] - quarter;
                float4 q = llvm_fma(crm, crm, ci2);
                int4 card = leMask(q * (q + crm), quarter * ci2);
                // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
                float4 crp = crq[k] + one;
                int4 bulb = leMask(llvm_fma(crp, crp, ci2), bulbR2);
                inSet[k] = card | bulb;
                active[k] = ~inSet[k];
                zr[k] = fzero;
                zi[k] = fzero;
                count[k] = izero;
            }}

            for (int it = 0; it < maxIter; ++it)
            {
                static foreach (k; 0 .. G)
                {{
                    zr2[k] = zr[k] * zr[k];
                    zi2[k] = zi[k] * zi[k];
                    active[k] &= leMask(zr2[k] + zi2[k], four);
                    count[k] -= active[k]; // +1 where active (mask is -1)
                }}
                // horizontal reduce + branch amortized over 4 iterations
                if ((it & 3) == 3)
                {
                    int4 any = active[0];
                    static foreach (k; 1 .. G)
                        any |= active[k];
                    if (!(any.array[0] | any.array[1] | any.array[2] | any.array[3]))
                        break;
                }
                static foreach (k; 0 .. G)
                {{
                    float4 nzr = (zr2[k] - zi2[k]) + crq[k];
                    zi[k] = llvm_fma(two * zr[k], zi[k], ciq); // 2*zr*zi + ci
                    zr[k] = nzr;
                }}
            }

            static foreach (k; 0 .. G)
            {{
                int4 cnt = (inSet[k] & vmax) | (~inSet[k] & count[k]);
                foreach (j; 0 .. 4)
                    orow[x + 4 * k + j] = cast(ushort) cnt.array[j];
            }}
        }

        // scalar tail (width not a multiple of 4*G)
        for (; x < width; ++x)
        {
            immutable float cr = x0 + cast(float) x * dx;
            float zrs = 0.0f, zis = 0.0f;
            int it = 0;
            for (; it < maxIter; ++it)
            {
                immutable float a = zrs * zrs, b = zis * zis;
                if (a + b > 4.0f)
                    break;
                immutable float nzr = a - b + cr;
                zis = 2.0f * zrs * zis + cy;
                zrs = nzr;
            }
            orow[x] = cast(ushort) it;
        }

        // mirror to the conjugate row
        immutable size_t mrow = cast(size_t) height - 1 - row;
        if (mrow != row)
            memcpy(outp + mrow * cast(size_t) width, orow, cast(size_t) width * ushort.sizeof);
    }
}
