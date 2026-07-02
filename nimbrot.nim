# Mandelbrot kernel as a C dylib. float32 scalar loop, persistent thread
# pool over an atomic row counter, y-axis symmetry mirror.
import std/[locks, atomics, cpuinfo]

var
  poolLock: Lock
  jobCv: Cond      # workers wait here for a new job
  doneCv: Cond     # caller waits here for job completion
  jobGen: int      # bumped per job, guarded by poolLock
  activeWorkers: int
  rowCounter: Atomic[int]
  jw, jh, jit, jhalf: int
  jbuf: ptr UncheckedArray[uint16]
  workers: seq[Thread[int]]
  poolReady = false

proc computeRow(row: int) =
  let w = jw
  let maxIter = jit
  let dx = 3.5'f32 / float32(w - 1)
  let dy = 2.0'f32 / float32(jh - 1)
  let ci = -1.0'f32 + float32(row) * dy
  let ci2 = ci * ci
  let base = row * w
  for x in 0 ..< w:
    let cr = -2.5'f32 + float32(x) * dx
    # cardioid + period-2 bulb early-out
    let crm = cr - 0.25'f32
    let q = crm * crm + ci2
    let crp = cr + 1.0'f32
    if q * (q + crm) <= 0.25'f32 * ci2 or crp * crp + ci2 <= 0.0625'f32:
      jbuf[base + x] = uint16(maxIter)
      continue
    var zr = 0'f32
    var zi = 0'f32
    var it = 0
    while it < maxIter:
      let zr2 = zr * zr
      let zi2 = zi * zi
      if zr2 + zi2 > 4.0'f32: break
      let nzr = zr2 - zi2 + cr
      zi = 2.0'f32 * zr * zi + ci
      zr = nzr
      inc it
    jbuf[base + x] = uint16(it)
  # mirror row (skip middle row of odd heights)
  let mrow = jh - 1 - row
  if mrow != row:
    copyMem(addr jbuf[mrow * w], addr jbuf[base], w * 2)

proc drainRows() =
  while true:
    let r = rowCounter.fetchAdd(1, moRelaxed)
    if r >= jhalf: break
    computeRow(r)

proc workerLoop(id: int) {.thread.} =
  var myGen = 0
  while true:
    acquire(poolLock)
    while jobGen == myGen:
      wait(jobCv, poolLock)
    myGen = jobGen
    release(poolLock)
    drainRows()
    acquire(poolLock)
    dec activeWorkers
    if activeWorkers == 0:
      signal(doneCv)
    release(poolLock)

proc ensurePool() =
  if poolReady: return
  initLock(poolLock)
  initCond(jobCv)
  initCond(doneCv)
  let n = max(countProcessors() - 1, 0)
  workers.setLen(n)
  for i in 0 ..< n:
    createThread(workers[i], workerLoop, i)
  poolReady = true

proc mandelbrotNim(w, h, it: cint; buf: ptr UncheckedArray[uint16])
    {.exportc: "mandelbrot_nim", cdecl, dynlib.} =
  ensurePool()
  acquire(poolLock)
  jw = w.int
  jh = h.int
  jit = it.int
  jbuf = buf
  jhalf = (h.int + 1) div 2
  rowCounter.store(0)
  activeWorkers = workers.len
  inc jobGen
  broadcast(jobCv)
  release(poolLock)
  drainRows()  # caller thread works too
  acquire(poolLock)
  while activeWorkers > 0:
    wait(doneCv, poolLock)
  release(poolLock)
