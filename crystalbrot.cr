# Mandelbrot persistent worker.
# Protocol: STDIN line "<w> <h> <iters>\n" -> STDOUT w*h uint16 LE row-major.
# Build: crystal build --release -Dpreview_mt -Dexecution_context crystalbrot.cr -o crystalbrot_bin
require "wait_group"

WORKERS = (ENV["CRYSTAL_WORKERS"]?.try(&.to_i?) || System.cpu_count.to_i).clamp(1, 64)
CTX = Fiber::ExecutionContext::Parallel.new("mandel", WORKERS)

# One row, Float32 scalar; mirrors into row h-1-r (y-axis symmetry).
def compute_row(buf : Pointer(UInt16), r : Int32, w : Int32, h : Int32,
                dx : Float32, dy : Float32, max_iter : Int32) : Nil
  ci = -1.0_f32 + r.to_f32 * dy
  ci2 = ci * ci
  m = max_iter.to_u16!
  row = buf + r.to_i64 * w
  x = 0
  while x < w
    cr = -2.5_f32 + x.to_f32 * dx
    crm = cr - 0.25_f32
    q = crm * crm + ci2
    crp = cr + 1.0_f32
    # cardioid + period-2 bulb early-out
    if q * (q + crm) <= 0.25_f32 * ci2 || crp * crp + ci2 <= 0.0625_f32
      row[x] = m
    else
      zr = 0.0_f32
      zi = 0.0_f32
      it = 0
      while it < max_iter
        zr2 = zr * zr
        zi2 = zi * zi
        break if zr2 + zi2 > 4.0_f32
        zi = 2.0_f32 * zr * zi + ci
        zr = zr2 - zi2 + cr
        it += 1
      end
      row[x] = it.to_u16!
    end
    x += 1
  end
  mr = h - 1 - r
  (buf + mr.to_i64 * w).copy_from(row, w) if mr != r
end

STDOUT.sync = true
buf = Slice(UInt16).empty

while line = STDIN.gets
  parts = line.split
  next if parts.size < 3
  w = parts[0].to_i
  h = parts[1].to_i
  iters = parts[2].to_i
  n = w * h
  buf = Slice(UInt16).new(n) if buf.size != n
  dx = w > 1 ? 3.5_f32 / (w - 1) : 0.0_f32
  dy = h > 1 ? 2.0_f32 / (h - 1) : 0.0_f32
  half = (h + 1) // 2
  ptr = buf.to_unsafe

  counter = Atomic(Int32).new(0)
  wg = WaitGroup.new(WORKERS)
  WORKERS.times do
    CTX.spawn do
      while (r = counter.add(1)) < half
        compute_row(ptr, r, w, h, dx, dy, iters)
      end
      wg.done
    end
  end
  wg.wait

  STDOUT.write(Slice(UInt8).new(ptr.as(UInt8*), n * 2))
  STDOUT.flush
end
