// RustBrot - Rust + ARM NEON + rayon Mandelbrot kernel.
//
// Mirrors cppbrot.cpp: float32x4 lanes (4 px/vector), G=2 independent vector
// groups interleaved to hide FMA latency, masked escape counting with the
// horizontal vmaxvq reduction amortized to every 4th iteration, cardioid +
// period-2 bulb analytic early-out, y-axis symmetry (top half computed, bottom
// half memcpy-mirrored), rows fanned across all cores via rayon.
//
// Semantics match pybrot.py: value = iterations survived before |z|^2 > 4
// (check-then-iterate), capped at max_iter (in-set pixels -> max_iter).

use std::arch::aarch64::*;

const G: usize = 2; // independent vector groups (4*G pixels per inner iteration)

// Rows write disjoint slices of `out`, so sharing the raw pointer is race-free.
struct OutPtr(*mut u16);
unsafe impl Sync for OutPtr {}
unsafe impl Send for OutPtr {}

#[no_mangle]
pub extern "C" fn mandelbrot_rust(width: i32, height: i32, max_iter: i32, out: *mut u16) {
    use rayon::prelude::*;

    let w = width as usize;
    let h = height as usize;
    let dx = 3.5f32 / (width - 1) as f32;
    let dy = 2.0f32 / (height - 1) as f32;
    // Conjugate rows have opposite ci and identical escape counts (IEEE
    // negation is exact), so only the top half is computed.
    let half = (h + 1) / 2;
    let optr = OutPtr(out);

    (0..half).into_par_iter().for_each(|row| {
        let optr = &optr;
        unsafe { do_row(w, h, max_iter as u32, dx, dy, row, optr.0) }
    });
}

#[target_feature(enable = "neon")]
unsafe fn do_row(w: usize, h: usize, max_iter: u32, dx: f32, dy: f32, row: usize, out: *mut u16) {
    let cy = -1.0f32 + row as f32 * dy;
    let orow = out.add(row * w);

    let lane = vld1q_f32([0.0f32, 1.0, 2.0, 3.0].as_ptr());
    let four = vdupq_n_f32(4.0);
    let two = vdupq_n_f32(2.0);
    let vdx = vdupq_n_f32(dx);
    let vx0 = vdupq_n_f32(-2.5);
    let quarter = vdupq_n_f32(0.25);
    let one = vdupq_n_f32(1.0);
    let bulb_r2 = vdupq_n_f32(0.0625); // (1/4)^2
    let vmax = vdupq_n_u32(max_iter);
    let ciq = vdupq_n_f32(cy);
    let ci2 = vmulq_f32(ciq, ciq);

    let mut x = 0usize;
    while x + 4 * G <= w {
        let mut crq = [vdupq_n_f32(0.0); G];
        let mut zr = [vdupq_n_f32(0.0); G];
        let mut zi = [vdupq_n_f32(0.0); G];
        let mut zr2 = [vdupq_n_f32(0.0); G];
        let mut zi2 = [vdupq_n_f32(0.0); G];
        let mut in_set = [vdupq_n_u32(0); G];
        let mut active = [vdupq_n_u32(0); G];
        let mut count = [vdupq_n_u32(0); G];

        for k in 0..G {
            let xv = vaddq_f32(vdupq_n_f32((x + 4 * k) as f32), lane);
            crq[k] = vfmaq_f32(vx0, xv, vdx); // cr = -2.5 + (x+4k+lane)*dx
            // main cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2, q = (cr-1/4)^2 + ci^2
            let crm = vsubq_f32(crq[k], quarter);
            let q = vfmaq_f32(ci2, crm, crm);
            let card = vcleq_f32(vmulq_f32(q, vaddq_f32(q, crm)), vmulq_f32(quarter, ci2));
            // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
            let crp = vaddq_f32(crq[k], one);
            let bulb = vcleq_f32(vfmaq_f32(ci2, crp, crp), bulb_r2);
            in_set[k] = vorrq_u32(card, bulb);
            active[k] = vmvnq_u32(in_set[k]);
        }

        let mut it = 0u32;
        while it < max_iter {
            for k in 0..G {
                zr2[k] = vmulq_f32(zr[k], zr[k]);
                zi2[k] = vmulq_f32(zi[k], zi[k]);
                let mag2 = vaddq_f32(zr2[k], zi2[k]);
                active[k] = vandq_u32(active[k], vcleq_f32(mag2, four));
                count[k] = vsubq_u32(count[k], active[k]); // +1 where active
            }
            // horizontal reduce + branch only every 4th iteration; escaped
            // lanes just run masked (inf/nan compares never re-activate them)
            if it & 3 == 3 {
                let mut any = active[0];
                for k in 1..G {
                    any = vorrq_u32(any, active[k]);
                }
                if vmaxvq_u32(any) == 0 {
                    break;
                }
            }
            for k in 0..G {
                let nzr = vaddq_f32(vsubq_f32(zr2[k], zi2[k]), crq[k]);
                zi[k] = vfmaq_f32(ciq, vmulq_f32(two, zr[k]), zi[k]); // 2*zr*zi + ci
                zr[k] = nzr;
            }
            it += 1;
        }

        for k in 0..G {
            let cnt = vbslq_u32(in_set[k], vmax, count[k]);
            vst1_u16(orow.add(x + 4 * k), vmovn_u32(cnt));
        }
        x += 4 * G;
    }

    // scalar tail (width not a multiple of 4*G)
    while x < w {
        let cr = -2.5f32 + x as f32 * dx;
        let mut zrs = 0.0f32;
        let mut zis = 0.0f32;
        let mut it = 0u32;
        while it < max_iter {
            let a = zrs * zrs;
            let b = zis * zis;
            if a + b > 4.0 {
                break;
            }
            let nzr = a - b + cr;
            zis = 2.0 * zrs * zis + cy;
            zrs = nzr;
            it += 1;
        }
        *orow.add(x) = it as u16;
        x += 1;
    }

    // mirror to the conjugate row (guard the middle row when height is odd)
    let mrow = h - 1 - row;
    if mrow != row {
        std::ptr::copy_nonoverlapping(orow, out.add(mrow * w), w);
    }
}
