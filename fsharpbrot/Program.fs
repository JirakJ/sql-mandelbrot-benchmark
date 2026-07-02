// FSharpBrot - persistent Mandelbrot worker: F# + ARM NEON (AdvSimd) intrinsics.
//
// Protocol: reads "width height maxIter\n" lines on stdin, replies with
// width*height uint16 little-endian counts (row-major) on stdout, flushed.
// Count convention: iterations survived, in-set pixels = maxIter.
//
// Kernel: float32x4 NEON, 2 independent vector groups (8 pixels per inner
// iter) to hide FMA latency, masked escape counting, all-dead MaxAcross check
// amortized to every 4th iteration, cardioid + period-2 bulb early-out,
// y-axis symmetry (top half computed, bottom mirrored), Parallel.For rows.

module FSharpBrot.Program

#nowarn "9"

open System
open System.Runtime.InteropServices
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.Arm
open System.Threading.Tasks
open Microsoft.FSharp.NativeInterop

let scalarPixel (cr: float32) (ci: float32) (maxIter: int) : uint16 =
    let ci2 = ci * ci
    let crm = cr - 0.25f
    let q = crm * crm + ci2
    if q * (q + crm) <= 0.25f * ci2 then uint16 maxIter
    elif (cr + 1.0f) * (cr + 1.0f) + ci2 <= 0.0625f then uint16 maxIter
    else
        let mutable zr = 0.0f
        let mutable zi = 0.0f
        let mutable it = 0
        let mutable go = true
        while go && it < maxIter do
            let zr2 = zr * zr
            let zi2 = zi * zi
            if zr2 + zi2 > 4.0f then go <- false
            else
                let nzr = zr2 - zi2 + cr
                zi <- 2.0f * zr * zi + ci
                zr <- nzr
                it <- it + 1
        uint16 it

let computeRowNeon (crArr: float32[]) (ci: float32) (maxIter: int)
                   (frame: uint16[]) (off: int) =
    let width = crArr.Length
    let four = Vector128.Create(4.0f)
    let quarter = Vector128.Create(0.25f)
    let one = Vector128.Create(1.0f)
    let bulbR2 = Vector128.Create(0.0625f) // (1/4)^2
    let vmax = Vector128.Create(uint32 maxIter)
    let civ = Vector128.Create(ci)
    let ci2 = AdvSimd.Multiply(civ, civ)

    use crp = fixed crArr
    use fp = fixed frame
    let orow = NativePtr.add fp off
    let mutable x = 0
    while x + 8 <= width do
        let cr0 = AdvSimd.LoadVector128(NativePtr.add crp x)
        let cr1 = AdvSimd.LoadVector128(NativePtr.add crp (x + 4))
        // cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2, q=(cr-1/4)^2+ci^2
        let crm0 = AdvSimd.Subtract(cr0, quarter)
        let crm1 = AdvSimd.Subtract(cr1, quarter)
        let q0 = AdvSimd.FusedMultiplyAdd(ci2, crm0, crm0)
        let q1 = AdvSimd.FusedMultiplyAdd(ci2, crm1, crm1)
        let card0 =
            AdvSimd.CompareLessThanOrEqual(
                AdvSimd.Multiply(q0, AdvSimd.Add(q0, crm0)),
                AdvSimd.Multiply(quarter, ci2)).AsUInt32()
        let card1 =
            AdvSimd.CompareLessThanOrEqual(
                AdvSimd.Multiply(q1, AdvSimd.Add(q1, crm1)),
                AdvSimd.Multiply(quarter, ci2)).AsUInt32()
        // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
        let crb0 = AdvSimd.Add(cr0, one)
        let crb1 = AdvSimd.Add(cr1, one)
        let bulb0 =
            AdvSimd.CompareLessThanOrEqual(
                AdvSimd.FusedMultiplyAdd(ci2, crb0, crb0), bulbR2).AsUInt32()
        let bulb1 =
            AdvSimd.CompareLessThanOrEqual(
                AdvSimd.FusedMultiplyAdd(ci2, crb1, crb1), bulbR2).AsUInt32()
        let inSet0 = AdvSimd.Or(card0, bulb0)
        let inSet1 = AdvSimd.Or(card1, bulb1)
        let mutable active0 = AdvSimd.Not(inSet0)
        let mutable active1 = AdvSimd.Not(inSet1)
        let mutable zr0 = Vector128<float32>.Zero
        let mutable zi0 = Vector128<float32>.Zero
        let mutable zr1 = Vector128<float32>.Zero
        let mutable zi1 = Vector128<float32>.Zero
        let mutable cnt0 = Vector128<uint32>.Zero
        let mutable cnt1 = Vector128<uint32>.Zero

        let mutable it = 0
        let mutable go = true
        while go && it < maxIter do
            let zr2_0 = AdvSimd.Multiply(zr0, zr0)
            let zi2_0 = AdvSimd.Multiply(zi0, zi0)
            let zr2_1 = AdvSimd.Multiply(zr1, zr1)
            let zi2_1 = AdvSimd.Multiply(zi1, zi1)
            active0 <- AdvSimd.And(active0,
                AdvSimd.CompareLessThanOrEqual(
                    AdvSimd.Add(zr2_0, zi2_0), four).AsUInt32())
            active1 <- AdvSimd.And(active1,
                AdvSimd.CompareLessThanOrEqual(
                    AdvSimd.Add(zr2_1, zi2_1), four).AsUInt32())
            cnt0 <- AdvSimd.Subtract(cnt0, active0) // +1 where active
            cnt1 <- AdvSimd.Subtract(cnt1, active1)
            // horizontal reduce + branch only every 4th iteration; dead lanes
            // run masked (inf/nan compares never re-activate them)
            if (it &&& 3) = 3
               && AdvSimd.Arm64.MaxAcross(
                      AdvSimd.Or(active0, active1)).ToScalar() = 0u then
                go <- false
            else
                let nzr0 = AdvSimd.Add(AdvSimd.Subtract(zr2_0, zi2_0), cr0)
                zi0 <- AdvSimd.FusedMultiplyAdd(
                    civ, AdvSimd.Add(zr0, zr0), zi0) // 2*zr*zi + ci
                zr0 <- nzr0
                let nzr1 = AdvSimd.Add(AdvSimd.Subtract(zr2_1, zi2_1), cr1)
                zi1 <- AdvSimd.FusedMultiplyAdd(
                    civ, AdvSimd.Add(zr1, zr1), zi1)
                zr1 <- nzr1
                it <- it + 1

        let out0 = AdvSimd.BitwiseSelect(inSet0, vmax, cnt0)
        let out1 = AdvSimd.BitwiseSelect(inSet1, vmax, cnt1)
        AdvSimd.Store(NativePtr.add orow x, AdvSimd.ExtractNarrowingLower(out0))
        AdvSimd.Store(NativePtr.add orow (x + 4),
                      AdvSimd.ExtractNarrowingLower(out1))
        x <- x + 8
    while x < width do // scalar tail (width not a multiple of 8)
        NativePtr.set orow x (scalarPixel crArr.[x] ci maxIter)
        x <- x + 1

let compute (width: int) (height: int) (maxIter: int) : uint16[] =
    let frame = Array.zeroCreate<uint16> (width * height)
    let dx = if width > 1 then 3.5 / float (width - 1) else 0.0
    let dy = if height > 1 then 2.0 / float (height - 1) else 0.0
    let crArr = Array.init width (fun x -> float32 (-2.5 + float x * dx))

    let half = (height + 1) / 2 // y-symmetry: bottom half mirrors top
    Parallel.For(0, half, fun row ->
        let ci = float32 (-1.0 + float row * dy)
        let off = row * width
        if AdvSimd.Arm64.IsSupported then
            computeRowNeon crArr ci maxIter frame off
        else
            for x in 0 .. width - 1 do
                frame.[off + x] <- scalarPixel crArr.[x] ci maxIter
        let mrow = height - 1 - row
        if mrow <> row then
            Array.Copy(frame, off, frame, mrow * width, width))
    |> ignore
    frame

[<EntryPoint>]
let main _ =
    let out = Console.OpenStandardOutput()
    let mutable line = Console.In.ReadLine()
    while not (isNull line) do
        let t = line.Trim()
        if t.Length > 0 then
            let p = t.Split(' ', StringSplitOptions.RemoveEmptyEntries)
            let frame = compute (int p.[0]) (int p.[1]) (int p.[2])
            out.Write(MemoryMarshal.AsBytes(ReadOnlySpan<uint16>(frame)))
            out.Flush()
        line <- Console.In.ReadLine()
    0
