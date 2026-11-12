/* ChapelBrot worker: reads "w h maxIter" lines on stdin, writes w*h
   little-endian uint16 pixels (row-major) on stdout, flushed per frame.
   Build: chpl --fast chapelbrot.chpl -o chapelbrot_bin */

use IO, DynamicIters;

inline proc pixel(cr: real(32), ci: real(32), maxIter: int(32)): uint(16) {
  param quarter = 0.25: real(32);
  param sixteenth = 0.0625: real(32);
  param one = 1.0: real(32);
  param two = 2.0: real(32);
  param four = 4.0: real(32);

  // Cardioid + period-2 bulb early-out.
  const ci2 = ci * ci;
  const crm = cr - quarter;
  const q = crm * crm + ci2;
  if q * (q + crm) <= quarter * ci2 then return maxIter: uint(16);
  const cp1 = cr + one;
  if cp1 * cp1 + ci2 <= sixteenth then return maxIter: uint(16);

  var zr = 0.0: real(32), zi = 0.0: real(32);
  var it: int(32) = 0;
  while it < maxIter {
    const zr2 = zr * zr, zi2 = zi * zi;
    if zr2 + zi2 > four then break;
    const nzr = zr2 - zi2 + cr;
    zi = two * zr * zi + ci;
    zr = nzr;
    it += 1;
  }
  return it: uint(16);
}

proc computeFrame(w: int, h: int, maxIter: int, ref img: [] uint(16)) {
  const dx = 3.5 / (if w > 1 then w - 1 else 1);
  const dy = 2.0 / (if h > 1 then h - 1 else 1);
  const top = (h + 1) / 2;
  const mi = maxIter: int(32);

  // Dynamic scheduling: row cost varies a lot across the set.
  forall r in dynamic(0..top-1, 1) {
    const ci = (-1.0 + r * dy): real(32);
    const rowOff = r * w;
    for x in 0..w-1 {
      const cr = (-2.5 + x * dx): real(32);
      img[rowOff + x] = pixel(cr, ci, mi);
    }
    // y-axis symmetry: mirror to the bottom half (odd middle row guarded).
    const mr = h - 1 - r;
    if mr != r then img[mr*w..#w] = img[rowOff..#w];
  }
}

proc main() throws {
  var w, h, mi: int;
  while stdin.read(w, h, mi) {
    var img: [0..#(w*h)] uint(16);
    computeFrame(w, h, mi, img);
    stdout.writeBinary(img, endianness.little);
    stdout.flush();
  }
}
