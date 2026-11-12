// VBrot - V (vlang) Mandelbrot kernel, compiled to a shared library.
//
// - 16-lane f32 escape loop written so clang (V's C backend) auto-vectorizes
//   it to NEON: sticky per-lane alive masks, unconditional updates (escaped
//   lanes blow up to inf/nan, NaN compares stay false), horizontal
//   all-dead check only every 8th iteration.
// - cardioid + period-2 bulb early-out per lane.
// - y-axis symmetry: only rows 0..(height+1)/2-1 computed, rest mirrored.
// - persistent pool of one thread per CPU core (spawned on first call,
//   untimed), woken via semaphores; interleaved rows for load balance.
//
// Semantics: value = iterations survived before zr^2+zi^2 > 4 (check first,
// then update), capped at max_iter; in-set pixels = max_iter.
module vbrot

import runtime
import sync

const lanes = 16

__global (
	g_nt     int
	g_width  int
	g_height int
	g_iter   int
	g_out    &u16
	g_starts []&sync.Semaphore
	g_done   &sync.Semaphore
)

@[direct_array_access]
fn do_rows(tid int, nthreads int, width int, height int, max_iter int, out &u16) {
	dx := f32(3.5) / f32(width - 1)
	dy := f32(2.0) / f32(height - 1)
	half := (height + 1) / 2
	mut p := unsafe { out }
	for row := tid; row < half; row += nthreads {
		ci := f32(-1.0) + f32(row) * dy
		ci2 := ci * ci
		base := row * width
		mut x := 0
		for ; x + lanes <= width; x += lanes {
			mut cr := [lanes]f32{}
			mut zr := [lanes]f32{}
			mut zi := [lanes]f32{}
			mut cnt := [lanes]int{}
			mut alive := [lanes]int{}
			mut inset := [lanes]int{}
			for l in 0 .. lanes {
				c := f32(-2.5) + f32(x + l) * dx
				cr[l] = c
				crm := c - f32(0.25)
				q := crm * crm + ci2
				crp := c + f32(1.0)
				ins := q * (q + crm) <= f32(0.25) * ci2 || crp * crp + ci2 <= f32(0.0625)
				inset[l] = if ins { 1 } else { 0 }
				alive[l] = if ins { 0 } else { 1 }
			}
			for it := 0; it < max_iter; it++ {
				for l in 0 .. lanes {
					a := zr[l] * zr[l]
					b := zi[l] * zi[l]
					alive[l] = if a + b <= f32(4.0) { alive[l] } else { 0 }
					cnt[l] += alive[l]
					nzr := a - b + cr[l]
					zi[l] = f32(2.0) * zr[l] * zi[l] + ci
					zr[l] = nzr
				}
				if it & 7 == 7 {
					mut any := 0
					for l in 0 .. lanes {
						any |= alive[l]
					}
					if any == 0 {
						break
					}
				}
			}
			for l in 0 .. lanes {
				v := if inset[l] != 0 { max_iter } else { cnt[l] }
				unsafe {
					p[base + x + l] = u16(v)
				}
			}
		}
		// scalar f64 tail (width not a multiple of lanes)
		cid := f64(-1.0) + f64(row) * (2.0 / f64(height - 1))
		cid2 := cid * cid
		for ; x < width; x++ {
			crd := f64(-2.5) + f64(x) * (3.5 / f64(width - 1))
			crm := crd - 0.25
			q := crm * crm + cid2
			mut it := 0
			if q * (q + crm) <= 0.25 * cid2 || (crd + 1.0) * (crd + 1.0) + cid2 <= 0.0625 {
				it = max_iter
			} else {
				mut zrs := 0.0
				mut zis := 0.0
				for it < max_iter {
					a := zrs * zrs
					b := zis * zis
					if a + b > 4.0 {
						break
					}
					zis = 2.0 * zrs * zis + cid
					zrs = a - b + crd
					it++
				}
			}
			unsafe {
				p[base + x] = u16(it)
			}
		}
		// mirror to the conjugate row (middle row of odd height maps to itself)
		mrow := height - 1 - row
		if mrow != row {
			unsafe {
				vmemcpy(&u8(p) + mrow * width * 2, &u8(p) + base * 2, usize(width * 2))
			}
		}
	}
}

fn worker_loop(tid int) {
	for {
		g_starts[tid].wait()
		do_rows(tid, g_nt, g_width, g_height, g_iter, g_out)
		g_done.post()
	}
}

@[export: 'mandelbrot_v']
pub fn mandelbrot_v(width int, height int, max_iter int, out &u16) {
	if g_nt == 0 {
		g_nt = runtime.nr_cpus()
		if g_nt < 1 {
			g_nt = 1
		}
		g_done = sync.new_semaphore()
		for tid in 0 .. g_nt {
			g_starts << sync.new_semaphore()
			spawn worker_loop(tid)
		}
	}
	g_width = width
	g_height = height
	g_iter = max_iter
	g_out = unsafe { out }
	for mut s in g_starts {
		s.post()
	}
	for _ in 0 .. g_nt {
		g_done.wait()
	}
}
