// Wren Mandelbrot worker: reads "w h it r0 r1 step\n", computes rows
// r0, r0+step, ... < r1 (interleaving balances load across the pool).
// wren_cli's stdout write is printf-based (truncates at NUL), so each
// uint16 is sent as 3 NUL-free bytes: base-255 digits, each + 1.
// Loops until stdin closes (readLine aborts; wrapper terminates us anyway).
import "io" for Stdin, Stdout

// Hot loop lives in an Fn so all state is in true locals, not module vars.
var compute = Fn.new {|w, h, maxit, r0, r1, step, enc|
  var dx = 3.5 / (w - 1)
  var dy = 2.0 / (h - 1)
  var rows = []
  var row = r0
  while (row < r1) {
    var ci = -1.0 + row * dy
    var ci2 = ci * ci
    var qci = 0.25 * ci2
    var px = List.filled(w, "")
    var x = 0
    while (x < w) {
      var cr = -2.5 + x * dx
      var res = 0
      var crm = cr - 0.25
      var q = crm * crm + ci2
      var cp1 = cr + 1.0
      if (q * (q + crm) <= qci || cp1 * cp1 + ci2 <= 0.0625) {
        res = maxit // cardioid / period-2 bulb: in set
      } else {
        var zr = 0.0
        var zi = 0.0
        var it = 0
        while (it < maxit) {
          var zr2 = zr * zr
          var zi2 = zi * zi
          if (zr2 + zi2 > 4.0) break
          zi = 2.0 * zr * zi + ci
          zr = zr2 - zi2 + cr
          it = it + 1
        }
        res = it
      }
      px[x] = enc[res]
      x = x + 1
    }
    rows.add(px.join())
    row = row + step
  }
  return rows.join()
}

var encMax = -1
var enc = []

while (true) {
  var line = Stdin.readLine()
  var parts = line.split(" ")
  if (parts.count < 6) break // EOF flushes a partial buffer: exit cleanly
  var w = Num.fromString(parts[0])
  var h = Num.fromString(parts[1])
  var maxit = Num.fromString(parts[2])
  var r0 = Num.fromString(parts[3])
  var r1 = Num.fromString(parts[4])
  var step = Num.fromString(parts[5])

  if (maxit != encMax) {
    // Lookup table: value -> 3-byte NUL-free string.
    enc = List.filled(maxit + 1, "")
    var v = 0
    while (v <= maxit) {
      var hi = (v / 65025).floor
      var rem = v - hi * 65025
      var mid = (rem / 255).floor
      var lo = rem - mid * 255
      enc[v] = String.fromByte(hi + 1) + String.fromByte(mid + 1) +
          String.fromByte(lo + 1)
      v = v + 1
    }
    encMax = maxit
  }

  System.write(compute.call(w, h, maxit, r0, r1, step, enc))
  Stdout.flush()
}
