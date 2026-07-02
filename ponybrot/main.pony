// Mandelbrot persistent worker in Pony.
// Protocol: stdin line "<w> <h> <iters>\n" -> stdout w*h uint16 LE row-major.
// Build: ponyc ponybrot -o . -b ponybrot_bin
//
// One behavior call per top-half row, striped over a fixed pool of Worker
// actors that the Pony runtime work-steals across all cores. A Collector
// orders the finished rows and emits the frame with a single writev, reusing
// the same immutable row bytes for the mirrored bottom half (y-axis symmetry).

actor Main
  let _env: Env
  let _buf: String ref = String
  let _nw: USize = 64
  let _workers: Array[Worker]
  let _pending: Array[(USize, USize, USize)] = Array[(USize, USize, USize)]
  var _busy: Bool = false

  new create(env: Env) =>
    _env = env
    _workers = Array[Worker](_nw)
    var i: USize = 0
    while i < _nw do
      _workers.push(Worker)
      i = i + 1
    end
    env.input(InputHandler(this), 1024)

  be chunk(data: Array[U8] iso) =>
    _buf.append(String.from_iso_array(consume data))
    _drain()

  be frame_done() =>
    _busy = false
    _kick()

  fun ref _drain() =>
    var go = true
    while go do
      try
        let nl = _buf.find("\n")?
        let line: String val = _buf.substring(0, nl)
        _buf.trim_in_place((nl + 1).usize())
        _handle(line)
      else
        go = false
      end
    end

  fun ref _handle(line: String val) =>
    let parts: Array[String] val = line.split()
    try
      let w = parts(0)?.usize()?
      let h = parts(1)?.usize()?
      let mi = parts(2)?.usize()?
      _pending.push((w, h, mi))
      _kick()
    end

  // Frames run strictly one at a time so output never interleaves.
  fun ref _kick() =>
    if _busy then return end
    try
      (let w: USize, let h: USize, let mi: USize) = _pending.shift()?
      _busy = true
      let coll = Collector(_env, this, w, h)
      let top = (h + 1) / 2
      var r: USize = 0
      while r < top do
        try _workers(r % _nw)?.row(coll, w, h, mi, r) end
        r = r + 1
      end
    end

class InputHandler is InputNotify
  let _main: Main

  new iso create(main: Main) =>
    _main = main

  fun ref apply(data: Array[U8] iso) =>
    _main.chunk(consume data)

  fun ref dispose() =>
    None

actor Collector
  let _env: Env
  let _main: Main
  let _h: USize
  let _top: USize
  let _rows: Array[(Array[U8] val | None)]
  var _remaining: USize

  new create(env: Env, main: Main, w: USize, h: USize) =>
    _env = env
    _main = main
    _h = h
    _top = (h + 1) / 2
    _rows = Array[(Array[U8] val | None)].init(None, _top)
    _remaining = _top

  be deliver(r: USize, data: Array[U8] val) =>
    try _rows(r)? = data end
    _remaining = _remaining - 1
    if _remaining == 0 then
      _flush()
    end

  fun ref _flush() =>
    let n = _h
    var seq: Array[ByteSeq] iso = recover Array[ByteSeq](n) end
    var r: USize = 0
    while r < _top do
      try
        match _rows(r)?
        | let a: Array[U8] val => seq.push(a)
        end
      end
      r = r + 1
    end
    // Bottom half mirrors computed rows (h-1-top) down to 0; an odd middle
    // row (index top-1) is emitted once above and skipped here.
    var s: USize = _h - _top
    while s > 0 do
      s = s - 1
      try
        match _rows(s)?
        | let a: Array[U8] val => seq.push(a)
        end
      end
    end
    _env.out.writev(consume seq)
    // stdout to a pipe is fully buffered; StdStream never flushes on its own.
    _env.out.flush()
    _main.frame_done()

actor Worker
  new create() =>
    None

  be row(coll: Collector, w: USize, h: USize, mi: USize, r: USize) =>
    coll.deliver(r, _compute(w, h, mi, r))

  // Cardioid + period-2 bulb membership test.
  fun _inset(cr: F64, ci2: F64, card_rhs: F64): Bool =>
    let crm: F64 = cr - 0.25
    let q: F64 = (crm * crm) + ci2
    let crp: F64 = cr + 1.0
    ((q * (q + crm)) <= card_rhs) or (((crp * crp) + ci2) <= 0.0625)

  // One row, F64 scalar, escape count = iterations survived (in-set = mi).
  // Two independent pixel chains per loop hide the FMA latency of the
  // z = z^2 + c dependency chain.
  fun _compute(w: USize, h: USize, mi: USize, r: USize): Array[U8] val =>
    let dx: F64 = if w > 1 then 3.5 / (w - 1).f64() else 0.0 end
    let dy: F64 = if h > 1 then 2.0 / (h - 1).f64() else 0.0 end
    let ci: F64 = -1.0 + (r.f64() * dy)
    let ci2: F64 = ci * ci
    let card_rhs: F64 = 0.25 * ci2
    var out: Array[U8] iso = recover Array[U8](w * 2) end
    var x: USize = 0
    while (x + 2) <= w do
      let cr0: F64 = -2.5 + (x.f64() * dx)
      let cr1: F64 = -2.5 + ((x + 1).f64() * dx)
      var it0: USize = 0
      var it1: USize = 0
      var zr0: F64 = 0
      var zi0: F64 = 0
      var zr1: F64 = 0
      var zi1: F64 = 0
      if _inset(cr0, ci2, card_rhs) then it0 = mi end
      if _inset(cr1, ci2, card_rhs) then it1 = mi end
      // joint loop while both pixels are still iterating
      if (it0 == 0) and (it1 == 0) then
        while it0 < mi do
          let a0: F64 = zr0 * zr0
          let b0: F64 = zi0 * zi0
          let a1: F64 = zr1 * zr1
          let b1: F64 = zi1 * zi1
          if (((a0 + b0) > 4.0) or ((a1 + b1) > 4.0)) then break end
          zi0 = ((2.0 * zr0) * zi0) + ci
          zr0 = (a0 - b0) + cr0
          zi1 = ((2.0 * zr1) * zi1) + ci
          zr1 = (a1 - b1) + cr1
          it0 = it0 + 1
        end
        it1 = it0
      end
      // finish each pixel individually from its current state
      while it0 < mi do
        let a: F64 = zr0 * zr0
        let b: F64 = zi0 * zi0
        if (a + b) > 4.0 then break end
        zi0 = ((2.0 * zr0) * zi0) + ci
        zr0 = (a - b) + cr0
        it0 = it0 + 1
      end
      while it1 < mi do
        let a: F64 = zr1 * zr1
        let b: F64 = zi1 * zi1
        if (a + b) > 4.0 then break end
        zi1 = ((2.0 * zr1) * zi1) + ci
        zr1 = (a - b) + cr1
        it1 = it1 + 1
      end
      let v0: U16 = it0.u16()
      let v1: U16 = it1.u16()
      out.push(v0.u8())
      out.push((v0 >> 8).u8())
      out.push(v1.u8())
      out.push((v1 >> 8).u8())
      x = x + 2
    end
    // odd-width tail pixel
    while x < w do
      let cr: F64 = -2.5 + (x.f64() * dx)
      var it: USize = 0
      if _inset(cr, ci2, card_rhs) then
        it = mi
      else
        var zr: F64 = 0
        var zi: F64 = 0
        while it < mi do
          let a: F64 = zr * zr
          let b: F64 = zi * zi
          if (a + b) > 4.0 then break end
          zi = ((2.0 * zr) * zi) + ci
          zr = (a - b) + cr
          it = it + 1
        end
      end
      let v: U16 = it.u16()
      out.push(v.u8())
      out.push((v >> 8).u8())
      x = x + 1
    end
    consume out
