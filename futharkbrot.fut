-- FutharkBrot: data-parallel Mandelbrot kernel, multicore backend.
-- Computes only the top half (y-axis symmetry); Python mirrors the rest.

def pixel (max_iter: i32) (cr: f64) (ci: f64) : u16 =
  let crm = cr - 0.25
  let q = crm * crm + ci * ci
  let in_cardioid = q * (q + crm) <= 0.25 * ci * ci
  let in_bulb = (cr + 1.0) * (cr + 1.0) + ci * ci <= 0.0625
  in if in_cardioid || in_bulb
     then u16.i32 max_iter
     else -- it = iterations survived; escape test before the update.
          let (it, _, _) =
            loop (it, zr, zi) = (0i32, 0.0, 0.0)
            while it < max_iter && zr * zr + zi * zi <= 4.0 do
              let zr2 = zr * zr
              let zi2 = zi * zi
              in (it + 1, zr2 - zi2 + cr, 2.0 * zr * zi + ci)
          in u16.i32 it

entry mandel (w: i64) (h: i64) (max_iter: i32) : [][]u16 =
  let dx = 3.5 / f64.i64 (w - 1)
  let dy = 2.0 / f64.i64 (h - 1)
  let half = (h + 1) / 2
  in tabulate_2d half w
       (\r x -> pixel max_iter (-2.5 + f64.i64 x * dx) (-1.0 + f64.i64 r * dy))
