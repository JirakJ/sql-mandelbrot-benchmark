# frozen_string_literal: true
# RubyBrot worker: Ractor pool (one per core) + YJIT scalar escape loop.
# Protocol: stdin line "<w> <h> <iters>\n" -> stdout w*h*2 bytes uint16 LE, row-major.
# Rows 0..(h+1)/2-1 computed; bottom half mirrored (y-axis symmetry).

require "etc"

$stdout.binmode
$stdin.binmode

# Compute one strided list of rows; send [y, packed_row] per row to the port.
def compute_rows(list, width, max_iter, dx, dy, out)
  list.each do |y|
    ci = -1.0 + y * dy
    ci2 = ci * ci
    row = Array.new(width, 0)
    x = 0
    while x < width
      cr = -2.5 + x * dx
      crm = cr - 0.25
      q = crm * crm + ci2
      cr1 = cr + 1.0
      if q * (q + crm) <= 0.25 * ci2 || cr1 * cr1 + ci2 <= 0.0625
        row[x] = max_iter # cardioid / period-2 bulb early-out
      else
        zr = 0.0
        zi = 0.0
        it = 0
        while it < max_iter
          zr2 = zr * zr
          zi2 = zi * zi
          break if zr2 + zi2 > 4.0
          zi = 2.0 * zr * zi + ci
          zr = zr2 - zi2 + cr
          it += 1
        end
        row[x] = it
      end
      x += 1
    end
    out.send([y, row.pack("v*")])
  end
end

NWORKERS = Etc.nprocessors
PORT = Ractor::Port.new
WORKERS = Array.new(NWORKERS) do
  Ractor.new(PORT) do |out|
    while (msg = Ractor.receive)
      compute_rows(msg[0], msg[1], msg[2], msg[3], msg[4], out)
    end
  end
end

while (line = $stdin.gets)
  parts = line.split
  w = Integer(parts[0])
  h = Integer(parts[1])
  mi = Integer(parts[2])
  dx = 3.5 / (w - 1)
  dy = 2.0 / (h - 1)
  half = (h + 1) / 2
  rows = Array.new(h)
  WORKERS.each_with_index do |r, k|
    list = k.step(half - 1, NWORKERS).to_a
    r.send([list, w, mi, dx, dy]) unless list.empty?
  end
  half.times do
    y, s = PORT.receive
    rows[y] = s
  end
  y = half
  while y < h
    rows[y] = rows[h - 1 - y]
    y += 1
  end
  $stdout.write(rows.join)
  $stdout.flush
end
