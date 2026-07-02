(* SmlBrot - Standard ML (MLton) Mandelbrot worker.
   Protocol: read "w h iters r0 r1" line on stdin, compute rows [r0,r1)
   of the top half, write (r1-r0)*w uint16 LE pixels row-major on stdout.
   Loops until EOF. Logging (none) would go to stderr only. *)

val stdoutFd = Posix.FileSys.stdout

fun writeAll vec =
  let
    val len = Word8Vector.length vec
    fun go i =
      if i >= len then ()
      else go (i + Posix.IO.writeVec
                     (stdoutFd, Word8VectorSlice.slice (vec, i, NONE)))
  in
    go 0
  end

(* Escape-time loop: returns iterations survived (max_iter = in set). *)
fun escape (zr, zi, cr, ci, it, maxIter) =
  if it >= maxIter then it
  else
    let
      val zr2 = zr * zr
      val zi2 = zi * zi
    in
      if zr2 + zi2 > 4.0 then it
      else escape (zr2 - zi2 + cr, 2.0 * zr * zi + ci, cr, ci, it + 1, maxIter)
    end

fun computeBand (w, h, maxIter, r0, r1) =
  let
    val dx = 3.5 / Real.fromInt (w - 1)
    val dy = 2.0 / Real.fromInt (h - 1)
    val buf = Word8Array.array ((r1 - r0) * w * 2, 0w0)
    fun pixel (cr, ci, ci2) =
      let
        (* cardioid + period-2 bulb early-outs *)
        val crm = cr - 0.25
        val q = crm * crm + ci2
        val cp1 = cr + 1.0
      in
        if q * (q + crm) <= 0.25 * ci2 orelse cp1 * cp1 + ci2 <= 0.0625
        then maxIter
        else escape (0.0, 0.0, cr, ci, 0, maxIter)
      end
    fun doRow row =
      let
        val ci = ~1.0 + Real.fromInt row * dy
        val ci2 = ci * ci
        val base = (row - r0) * w * 2
        fun px x =
          if x >= w then ()
          else
            let
              val res = pixel (~2.5 + Real.fromInt x * dx, ci, ci2)
              val i = base + x + x
            in
              Word8Array.update (buf, i, Word8.fromInt res);
              Word8Array.update (buf, i + 1, Word8.fromInt (Int.quot (res, 256)));
              px (x + 1)
            end
      in
        px 0
      end
    fun rows r = if r >= r1 then () else (doRow r; rows (r + 1))
  in
    rows r0;
    Word8Array.vector buf
  end

fun parse line =
  case List.mapPartial Int.fromString (String.tokens Char.isSpace line) of
    [w, h, it, r0, r1] => SOME (w, h, it, r0, r1)
  | _ => NONE

fun loop () =
  case TextIO.inputLine TextIO.stdIn of
    NONE => ()
  | SOME line =>
      (case parse line of
         SOME job => (writeAll (computeBand job); loop ())
       | NONE => loop ())

val () = loop ()
