/-
LeanBrot - Lean 4 Mandelbrot persistent worker.

Protocol: "<width> <height> <max_iter>\n" on stdin; writes the TOP HALF
((height+1)/2 rows) as uint16 LE row-major bytes to stdout, flushed.
The Python wrapper mirrors the bottom half (y-axis symmetry).
Rows are computed in parallel via Task.spawn (Lean runtime thread pool).
-/

/-- Escape loop; returns iterations survived (maxIter = in set). -/
partial def escapeLoop (cr ci : Float) (maxIter : UInt32)
    (zr zi : Float) (it : UInt32) : UInt32 :=
  if it == maxIter then it
  else
    let zr2 := zr * zr
    let zi2 := zi * zi
    if zr2 + zi2 > 4.0 then it
    else escapeLoop cr ci maxIter (zr2 - zi2 + cr) (2.0 * zr * zi + ci) (it + 1)

@[inline]
def pixel (cr ci : Float) (maxIter : UInt32) : UInt32 :=
  -- cardioid early-out
  let crm := cr - 0.25
  let ci2 := ci * ci
  let q := crm * crm + ci2
  if q * (q + crm) <= 0.25 * ci2 then maxIter
  -- period-2 bulb early-out
  else if (cr + 1.0) * (cr + 1.0) + ci2 <= 0.0625 then maxIter
  else escapeLoop cr ci maxIter 0.0 0.0 0

/-- Row pixels, tail-recursive; UInt64.toFloat is a plain cast
    (Float.ofNat would round-trip through GMP per pixel). -/
partial def rowLoop (w : USize) (maxIter : UInt32) (dx ci : Float)
    (x : USize) (acc : ByteArray) : ByteArray :=
  if x == w then acc
  else
    let cr := -2.5 + x.toUInt64.toFloat * dx
    let n := pixel cr ci maxIter
    -- uint16 LE: low byte, high byte
    rowLoop w maxIter dx ci (x + 1) ((acc.push n.toUInt8).push (n >>> 8).toUInt8)

def rowBytes (w : Nat) (maxIter : UInt32) (dx ci : Float) : ByteArray :=
  rowLoop (USize.ofNat w) maxIter dx ci 0 (ByteArray.emptyWithCapacity (w * 2))

def frame (w h : Nat) (maxIter : UInt32) : ByteArray := Id.run do
  let dx := if w > 1 then 3.5 / Float.ofNat (w - 1) else 0.0
  let dy := if h > 1 then 2.0 / Float.ofNat (h - 1) else 0.0
  let top := (h + 1) / 2
  let tasks := (List.range top).map fun r =>
    Task.spawn fun _ => rowBytes w maxIter dx (-1.0 + (USize.ofNat r).toUInt64.toFloat * dy)
  let mut out := ByteArray.emptyWithCapacity (w * top * 2)
  for t in tasks do
    out := out ++ t.get
  return out

partial def workerLoop (stdin stdout : IO.FS.Stream) : IO Unit := do
  let line ← stdin.getLine
  if line.isEmpty then return ()  -- EOF
  match line.trimAscii.toString.splitOn " " |>.filter (· ≠ "") with
  | [ws, hs, is] =>
    let w := ws.toNat!
    let h := hs.toNat!
    let mi := is.toNat!
    stdout.write (frame w h (UInt32.ofNat mi))
    stdout.flush
    workerLoop stdin stdout
  | _ => IO.eprintln s!"leanbrot: bad request: {line.trimAscii}"

def main : IO Unit := do
  workerLoop (← IO.getStdin) (← IO.getStdout)
