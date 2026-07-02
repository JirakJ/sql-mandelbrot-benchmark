//// Gleambrot - persistent Mandelbrot worker in Gleam on the BEAM.
//// Protocol: "<width> <height> <iters>\n" on stdin ->
//// exactly width*height*2 bytes uint16 LE row-major on stdout.
//// Hot path is pure Gleam; a tiny Erlang FFI handles binary stdio and
//// per-band process spawning. Zero hex dependencies.

@external(erlang, "gleambrot_ffi", "set_binary_io")
fn set_binary_io() -> Nil

@external(erlang, "gleambrot_ffi", "read_request")
fn read_request() -> Result(#(Int, Int, Int), Nil)

@external(erlang, "gleambrot_ffi", "write_frame")
fn write_frame(rows: List(BitArray)) -> Nil

@external(erlang, "gleambrot_ffi", "halt_now")
fn halt_now() -> Nil

@external(erlang, "gleambrot_ffi", "schedulers")
fn schedulers() -> Int

@external(erlang, "gleambrot_ffi", "pmap")
fn pmap(items: List(a), f: fn(a) -> b) -> List(b)

@external(erlang, "lists", "append")
fn concat(lists: List(List(a))) -> List(a)

@external(erlang, "lists", "append")
fn append(a: List(a), b: List(a)) -> List(a)

@external(erlang, "erlang", "float")
fn to_float(n: Int) -> Float

pub fn main() {
  set_binary_io()
  serve(schedulers())
}

fn serve(nb: Int) -> Nil {
  case read_request() {
    Ok(#(w, h, it)) -> {
      write_frame(frame(w, h, it, nb))
      serve(nb)
    }
    Error(_) -> halt_now()
  }
}

// Compute the top half in parallel band processes, mirror the bottom
// (y-axis symmetry); the odd middle row is computed once, never mirrored.
fn frame(w: Int, h: Int, max: Int, nb: Int) -> List(BitArray) {
  let dx = 3.5 /. to_float(w - 1)
  let dy = 2.0 /. to_float(h - 1)
  let top = { h + 1 } / 2
  // 4 bands per scheduler for load balance near the set boundary.
  let chunk = int_max(1, { top + nb * 4 - 1 } / { nb * 4 })
  let parts = pmap(bands(0, top, chunk), fn(b) { rows(b.0, b.1, w, dx, dy, max) })
  let top_rows = concat(parts)
  append(top_rows, rev_take(top_rows, h - top, []))
}

fn bands(r0: Int, top: Int, chunk: Int) -> List(#(Int, Int)) {
  case r0 < top {
    False -> []
    True -> [#(r0, int_min(r0 + chunk, top)), ..bands(r0 + chunk, top, chunk)]
  }
}

fn rows(r: Int, r1: Int, w: Int, dx: Float, dy: Float, max: Int) -> List(BitArray) {
  case r < r1 {
    False -> []
    True -> {
      let ci = -1.0 +. to_float(r) *. dy
      [row(0, w, dx, ci, max, <<>>), ..rows(r + 1, r1, w, dx, dy, max)]
    }
  }
}

// Append-optimised bit array build: left-to-right, one uint16 LE per pixel.
fn row(x: Int, w: Int, dx: Float, ci: Float, max: Int, acc: BitArray) -> BitArray {
  case x < w {
    False -> acc
    True -> {
      let cr = -2.5 +. to_float(x) *. dx
      row(x + 1, w, dx, ci, max, <<acc:bits, pixel(cr, ci, max):size(16)-little>>)
    }
  }
}

fn pixel(cr: Float, ci: Float, max: Int) -> Int {
  let crm = cr -. 0.25
  let ci2 = ci *. ci
  let q = crm *. crm +. ci2
  // Main cardioid and period-2 bulb: in the set, skip the loop.
  case q *. { q +. crm } <=. 0.25 *. ci2 {
    True -> max
    False ->
      case { cr +. 1.0 } *. { cr +. 1.0 } +. ci2 <=. 0.0625 {
        True -> max
        False -> escape(0.0, 0.0, cr, ci, 0, max)
      }
  }
}

fn escape(zr: Float, zi: Float, cr: Float, ci: Float, it: Int, max: Int) -> Int {
  case it < max {
    False -> it
    True -> {
      let zr2 = zr *. zr
      let zi2 = zi *. zi
      case zr2 +. zi2 >. 4.0 {
        True -> it
        False ->
          escape(zr2 -. zi2 +. cr, 2.0 *. zr *. zi +. ci, cr, ci, it + 1, max)
      }
    }
  }
}

// Bottom half = reverse of the first h-top computed rows.
fn rev_take(l: List(a), n: Int, acc: List(a)) -> List(a) {
  case n <= 0 {
    True -> acc
    False ->
      case l {
        [] -> acc
        [x, ..rest] -> rev_take(rest, n - 1, [x, ..acc])
      }
  }
}

fn int_min(a: Int, b: Int) -> Int {
  case a < b {
    True -> a
    False -> b
  }
}

fn int_max(a: Int, b: Int) -> Int {
  case a > b {
    True -> a
    False -> b
  }
}
