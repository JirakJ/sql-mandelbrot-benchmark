// CSharpBrot - persistent Mandelbrot worker: C# + ARM NEON (AdvSimd) intrinsics.
//
// Protocol: reads "width height maxIter\n" lines on stdin, replies with
// width*height uint16 little-endian counts (row-major) on stdout, flushed.
// Count convention: iterations survived, in-set pixels = maxIter.
//
// Kernel: float32x4 NEON, G=2 independent vector groups to hide FMA latency,
// masked escape counting, all-dead MaxAcross check amortized to every 4th
// iteration, cardioid + period-2 bulb analytic early-out, y-axis symmetry
// (only top half computed, bottom mirrored), Parallel.For across rows.

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Threading.Tasks;

internal static class Program
{
    private const int G = 2; // independent vector groups (8 pixels per inner iter)

    private static void Main()
    {
        Stream stdout = Console.OpenStandardOutput();
        string? line;
        while ((line = Console.In.ReadLine()) != null)
        {
            line = line.Trim();
            if (line.Length == 0) continue;
            string[] p = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int w = int.Parse(p[0]);
            int h = int.Parse(p[1]);
            int mi = int.Parse(p[2]);
            ushort[] frame = Compute(w, h, mi);
            stdout.Write(MemoryMarshal.AsBytes(frame.AsSpan()));
            stdout.Flush();
        }
    }

    private static ushort[] Compute(int width, int height, int maxIter)
    {
        var frame = new ushort[(long)width * height];
        double dx = width > 1 ? 3.5 / (width - 1) : 0.0;
        double dy = height > 1 ? 2.0 / (height - 1) : 0.0;
        var crArr = new float[width];
        for (int x = 0; x < width; x++) crArr[x] = (float)(-2.5 + x * dx);

        int half = (height + 1) / 2; // y-symmetry: bottom half mirrors top
        Parallel.For(0, half, row =>
        {
            float ci = (float)(-1.0 + row * dy);
            int off = row * width;
            if (AdvSimd.Arm64.IsSupported)
                ComputeRowNeon(crArr, ci, maxIter, frame, off);
            else
                for (int x = 0; x < width; x++)
                    frame[off + x] = ScalarPixel(crArr[x], ci, maxIter);
            int mrow = height - 1 - row;
            if (mrow != row)
                Array.Copy(frame, off, frame, (long)mrow * width, width);
        });
        return frame;
    }

    private static unsafe void ComputeRowNeon(
        float[] crArr, float ci, int maxIter, ushort[] frame, int off)
    {
        int width = crArr.Length;
        Vector128<float> four = Vector128.Create(4.0f);
        Vector128<float> quarter = Vector128.Create(0.25f);
        Vector128<float> one = Vector128.Create(1.0f);
        Vector128<float> bulbR2 = Vector128.Create(0.0625f); // (1/4)^2
        Vector128<uint> vmax = Vector128.Create((uint)maxIter);
        Vector128<float> civ = Vector128.Create(ci);
        Vector128<float> ci2 = AdvSimd.Multiply(civ, civ);

        var cr = stackalloc Vector128<float>[G];
        var zr = stackalloc Vector128<float>[G];
        var zi = stackalloc Vector128<float>[G];
        var zr2 = stackalloc Vector128<float>[G];
        var zi2 = stackalloc Vector128<float>[G];
        var inSet = stackalloc Vector128<uint>[G];
        var active = stackalloc Vector128<uint>[G];
        var count = stackalloc Vector128<uint>[G];

        fixed (float* crp = crArr)
        fixed (ushort* fp = frame)
        {
            ushort* orow = fp + off;
            int x = 0;
            for (; x + 4 * G <= width; x += 4 * G)
            {
                for (int k = 0; k < G; k++)
                {
                    cr[k] = AdvSimd.LoadVector128(crp + x + 4 * k);
                    // cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2, q=(cr-1/4)^2+ci^2
                    Vector128<float> crm = AdvSimd.Subtract(cr[k], quarter);
                    Vector128<float> q = AdvSimd.FusedMultiplyAdd(ci2, crm, crm);
                    Vector128<uint> card = AdvSimd.CompareLessThanOrEqual(
                        AdvSimd.Multiply(q, AdvSimd.Add(q, crm)),
                        AdvSimd.Multiply(quarter, ci2)).AsUInt32();
                    // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
                    Vector128<float> crb = AdvSimd.Add(cr[k], one);
                    Vector128<uint> bulb = AdvSimd.CompareLessThanOrEqual(
                        AdvSimd.FusedMultiplyAdd(ci2, crb, crb), bulbR2).AsUInt32();
                    inSet[k] = AdvSimd.Or(card, bulb);
                    active[k] = AdvSimd.Not(inSet[k]);
                    zr[k] = Vector128<float>.Zero;
                    zi[k] = Vector128<float>.Zero;
                    count[k] = Vector128<uint>.Zero;
                }

                for (int it = 0; it < maxIter; it++)
                {
                    for (int k = 0; k < G; k++)
                    {
                        zr2[k] = AdvSimd.Multiply(zr[k], zr[k]);
                        zi2[k] = AdvSimd.Multiply(zi[k], zi[k]);
                        Vector128<float> mag2 = AdvSimd.Add(zr2[k], zi2[k]);
                        active[k] = AdvSimd.And(active[k],
                            AdvSimd.CompareLessThanOrEqual(mag2, four).AsUInt32());
                        count[k] = AdvSimd.Subtract(count[k], active[k]); // +1 where active
                    }
                    // horizontal reduce + branch only every 4th iteration; dead
                    // lanes run masked (inf/nan compares never re-activate them)
                    if ((it & 3) == 3)
                    {
                        Vector128<uint> any = active[0];
                        for (int k = 1; k < G; k++) any = AdvSimd.Or(any, active[k]);
                        if (AdvSimd.Arm64.MaxAcross(any).ToScalar() == 0) break;
                    }
                    for (int k = 0; k < G; k++)
                    {
                        Vector128<float> nzr =
                            AdvSimd.Add(AdvSimd.Subtract(zr2[k], zi2[k]), cr[k]);
                        zi[k] = AdvSimd.FusedMultiplyAdd(
                            civ, AdvSimd.Add(zr[k], zr[k]), zi[k]); // 2*zr*zi + ci
                        zr[k] = nzr;
                    }
                }

                for (int k = 0; k < G; k++)
                {
                    Vector128<uint> cnt = AdvSimd.BitwiseSelect(inSet[k], vmax, count[k]);
                    AdvSimd.Store(orow + x + 4 * k, AdvSimd.ExtractNarrowingLower(cnt));
                }
            }
            for (; x < width; x++) // scalar tail (width not a multiple of 4*G)
                orow[x] = ScalarPixel(crArr[x], ci, maxIter);
        }
    }

    private static ushort ScalarPixel(float cr, float ci, int maxIter)
    {
        float ci2 = ci * ci;
        float crm = cr - 0.25f;
        float q = crm * crm + ci2;
        if (q * (q + crm) <= 0.25f * ci2) return (ushort)maxIter;
        float crb = cr + 1.0f;
        if (crb * crb + ci2 <= 0.0625f) return (ushort)maxIter;
        float zr = 0f, zi = 0f;
        int it = 0;
        while (it < maxIter)
        {
            float zr2 = zr * zr, zi2 = zi * zi;
            if (zr2 + zi2 > 4.0f) break;
            float nzr = zr2 - zi2 + cr;
            zi = 2f * zr * zi + ci;
            zr = nzr;
            it++;
        }
        return (ushort)it;
    }
}
