// SwiftBrot - Swift + SIMD8<Float> + GCD Mandelbrot kernel.
//
// Semantics match pybrot.py: value = iterations survived before |z|^2 > 4,
// capped at max_iter (in-set pixels -> max_iter).
// y-axis symmetry: only the top half of rows is computed, the rest is mirrored.

import Dispatch
import Foundation

@_cdecl("mandelbrot_swift")
public func mandelbrotSwift(_ w: Int32, _ h: Int32, _ it: Int32,
                            _ out: UnsafeMutablePointer<UInt16>) {
    let width = Int(w), height = Int(h), maxIter = Int(it)
    let dx = 3.5 / Float(width - 1)
    let dy = 2.0 / Float(height - 1)
    let x0: Float = -2.5
    let y0: Float = -1.0

    let lane = SIMD8<Float>(0, 1, 2, 3, 4, 5, 6, 7)
    let four = SIMD8<Float>(repeating: 4.0)
    let quarter = SIMD8<Float>(repeating: 0.25)
    let bulbR2 = SIMD8<Float>(repeating: 0.0625)
    let vdx = SIMD8<Float>(repeating: dx)
    let vx0 = SIMD8<Float>(repeating: x0)
    let ones = SIMD8<UInt32>(repeating: 1)
    let vmax = SIMD8<UInt32>(repeating: UInt32(maxIter))

    // Rows r and height-1-r have opposite ci -> identical escape counts.
    let half = (height + 1) / 2

    DispatchQueue.concurrentPerform(iterations: half) { row in
        let ci = y0 + Float(row) * dy
        let ciq = SIMD8<Float>(repeating: ci)
        let ci2 = ciq * ciq
        let orow = out + row * width

        var x = 0
        while x + 8 <= width {
            let xv = SIMD8<Float>(repeating: Float(x)) + lane
            let cr = vx0 + xv * vdx
            // main cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2, q = (cr-1/4)^2 + ci^2
            let crm = cr - quarter
            let q = crm * crm + ci2
            let card = (q * (q + crm)) .<= (quarter * ci2)
            // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
            let crp = cr + SIMD8<Float>(repeating: 1.0)
            let bulb = (crp * crp + ci2) .<= bulbR2
            let inSet = card .| bulb

            var zr = SIMD8<Float>.zero
            var zi = SIMD8<Float>.zero
            var count = SIMD8<UInt32>.zero
            var active = .!inSet

            var iter = 0
            while iter < maxIter {
                let zr2 = zr * zr
                let zi2 = zi * zi
                // dead lanes never re-activate (inf/nan compares are false)
                active = active .& ((zr2 + zi2) .<= four)
                if !any(active) { break }
                count &+= SIMD8<UInt32>.zero.replacing(with: ones, where: active)
                let nzr = zr2 - zi2 + cr
                zi = 2 * zr * zi + ciq
                zr = nzr
                iter += 1
            }

            let result = count.replacing(with: vmax, where: inSet)
            let r16 = SIMD8<UInt16>(truncatingIfNeeded: result)
            for k in 0..<8 { orow[x + k] = r16[k] }
            x += 8
        }
        // scalar tail (width not a multiple of 8)
        while x < width {
            let crs = x0 + Float(x) * dx
            var zrs: Float = 0, zis: Float = 0
            var iter = 0
            while iter < maxIter {
                let a = zrs * zrs, b = zis * zis
                if a + b > 4.0 { break }
                let nzr = a - b + crs
                zis = 2 * zrs * zis + ci
                zrs = nzr
                iter += 1
            }
            orow[x] = UInt16(iter)
            x += 1
        }

        // mirror to the conjugate row (guard the middle row when height is odd)
        let mrow = height - 1 - row
        if mrow != row {
            (out + mrow * width).update(from: orow, count: width)
        }
    }
}
