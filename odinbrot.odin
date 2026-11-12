// OdinBrot - Mandelbrot dylib: 8-wide f32 SIMD, persistent thread pool.
//
// - #simd[8]f32 masked escape-time counting (lanes_le mask, count -= mask).
// - G=2 independent vector groups (16 pixels/iter) to hide FMA latency.
// - amortized escape reduction: horizontal reduce_or + branch every 4th iter.
// - cardioid + period-2 bulb analytic early-out.
// - y-axis symmetry: only top half computed, bottom mirrored via mem_copy.
// - persistent worker threads (spawned on first call, i.e. import warm-up);
//   rows handed out via an atomic counter, semaphores for wake/done.
//
// Semantics match pybrot.py: value = iterations survived before |z|^2 > 4,
// capped at max_iter.
package odinbrot

import "base:intrinsics"
import "base:runtime"
import "core:os"
import "core:simd"
import "core:sync"
import "core:thread"

N :: 8
G :: 2 // independent vector groups; 16 pixels per inner iteration
Vf :: #simd[N]f32
Vu :: #simd[N]u32

IOTA := Vf{0, 1, 2, 3, 4, 5, 6, 7}

// current job (published before waking workers; sema post/wait order them)
job_w, job_h, job_it: i32
job_half: i32
job_out: [^]u16
row_next: i32 // atomic row dispenser

work_sema: sync.Sema
done_sema: sync.Sema
num_workers: int
pool_up: bool

compute_row :: proc "contextless" (w, h, maxit: i32, out: [^]u16, row: i32) {
	dx := f32(3.5) / f32(w - 1)
	dy := f32(2.0) / f32(h - 1)
	ci := f32(-1.0) + f32(row) * dy

	civ := Vf(ci)
	ci2 := civ * civ
	vdx := Vf(dx)
	vx0 := Vf(f32(-2.5))
	four := Vf(4)
	two := Vf(2)
	quarter := Vf(0.25)
	one := Vf(1)
	bulb_r2 := Vf(0.0625)
	vmax := Vu(u32(maxit))

	orow := out[int(row) * int(w):]

	x: i32 = 0
	for ; x + N * G <= w; x += N * G {
		crv, zr, zi, zr2, zi2: [G]Vf
		in_set, active, count: [G]Vu

		#unroll for k in 0 ..< G {
			xv := Vf(f32(x + i32(k) * N)) + IOTA
			crv[k] = vx0 + xv * vdx
			// main cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2, q = (cr-1/4)^2 + ci^2
			crm := crv[k] - quarter
			q := crm * crm + ci2
			card := simd.lanes_le(q * (q + crm), quarter * ci2)
			// period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
			crp := crv[k] + one
			bulb := simd.lanes_le(crp * crp + ci2, bulb_r2)
			in_set[k] = card | bulb
			active[k] = ~in_set[k]
		}

		any0 := active[0]
		#unroll for k in 1 ..< G {any0 |= active[k]}
		if simd.reduce_or(any0) != 0 {
			for it: i32 = 0; it < maxit; it += 1 {
				#unroll for k in 0 ..< G {
					zr2[k] = zr[k] * zr[k]
					zi2[k] = zi[k] * zi[k]
					mag2 := zr2[k] + zi2[k]
					active[k] &= simd.lanes_le(mag2, four)
					count[k] -= active[k] // mask is all-ones: -(-1) == +1
				}
				// horizontal reduce + branch only every 4th iteration; dead
				// lanes just run masked (inf/nan never re-activates them)
				if it & 3 == 3 {
					any := active[0]
					#unroll for k in 1 ..< G {any |= active[k]}
					if simd.reduce_or(any) == 0 {break}
				}
				#unroll for k in 0 ..< G {
					nzr := zr2[k] - zi2[k] + crv[k]
					zi[k] = two * zr[k] * zi[k] + civ
					zr[k] = nzr
				}
			}
		}

		#unroll for k in 0 ..< G {
			cnt := simd.select(in_set[k], vmax, count[k])
			cnt16 := cast(#simd[N]u16)cnt
			intrinsics.unaligned_store((^#simd[N]u16)(&orow[x + i32(k) * N]), cnt16)
		}
	}

	// scalar tail (width not a multiple of N*G)
	for ; x < w; x += 1 {
		cr := f32(-2.5) + f32(x) * dx
		zrs, zis: f32
		it: i32 = 0
		for ; it < maxit; it += 1 {
			a := zrs * zrs
			b := zis * zis
			if a + b > 4.0 {break}
			nzr := a - b + cr
			zis = 2 * zrs * zis + ci
			zrs = nzr
		}
		orow[x] = u16(it)
	}

	// mirror to the conjugate row (opposite ci -> identical escape counts)
	mrow := h - 1 - row
	if mrow != row {
		intrinsics.mem_copy_non_overlapping(&out[int(mrow) * int(w)], &orow[0], int(w) * size_of(u16))
	}
}

do_rows :: proc "contextless" () {
	w, h, maxit, half, out := job_w, job_h, job_it, job_half, job_out
	for {
		row := sync.atomic_add(&row_next, 1)
		if row >= half {break}
		compute_row(w, h, maxit, out, row)
	}
}

worker_proc :: proc(_: rawptr) {
	for {
		sync.sema_wait(&work_sema)
		do_rows()
		sync.sema_post(&done_sema)
	}
}

ensure_pool :: proc() {
	if pool_up {return}
	pool_up = true
	num_workers = os.get_processor_core_count() - 1 // caller thread works too
	if num_workers < 0 {num_workers = 0}
	for _ in 0 ..< num_workers {
		thread.create_and_start_with_data(nil, worker_proc, nil, .Normal, true)
	}
}

@(export)
mandelbrot_odin :: proc "c" (w, h, it: i32, out: [^]u16) {
	context = runtime.default_context()
	ensure_pool()
	job_w, job_h, job_it, job_out = w, h, it, out
	job_half = (h + 1) / 2
	sync.atomic_store(&row_next, 0)
	sync.sema_post(&work_sema, num_workers)
	do_rows()
	for _ in 0 ..< num_workers {sync.sema_wait(&done_sema)}
}
