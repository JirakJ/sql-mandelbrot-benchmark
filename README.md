# sql-mandelbrot-benchmark
**Because why benchmark sql engines with boring aggregates when you can generate fractals?**

This project uses recursive Common Table Expressions (CTE) to calculate the Mandelbrot set entirely 
in SQL — no loops, no procedural code, just pure SQL. It serves as a fun and visually appealing benchmark 
for testing recursive query performance, floating-point precision, and computational capabilities of SQL engines.

![Mandelbrot Set](images/duckbrot.png)

## What is This?

A benchmark suite that:
- Computes the famous [Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set) using SQL recursive CTEs
- Tests multiple SQL engines, currently just DuckDB and a Python implementation for reference.
- Generates beautiful fractal images as proof of correct computation
- Reveals which database / SQL engine renders infinity fastest

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/duckbrot.git
cd duckbrot

# Install dependencies
pip install -r requirements.txt

# Run the benchmark suite
python main.py
```

## Current Benchmark Results

Current results on 1400x800 pixels, 256 max iterations, Macbook Pro M4 Max:

| 🏆 | Engine/Implementation        | Time (ms) | Relative Performance |
|----|------------------------------|-----------|---------------------|
| 1  | **Hybrid Metal 4 (CPU+GPU)** | ~0.15 ms  | **~0.0002x** ⭐     |
| 2  | **Hybrid CPU+GPU (Metal 3)** | ~0.24 ms  | ~0.0003x            |
| 3  | **Metal GPU**                |  ~0.3 ms  | ~0.0004x            |
| 4  | **C++ NEON (SIMD + threads)**|  ~0.4 ms  | ~0.0005x            |
| 5  | NumPy (vectorized, unrolled) |   665 ms  | 0.83x               |
| 4  | ArrowDatafusion (SQL)        |   797 ms  | 1.00x (baseline)    |
| 5  | DuckDB (SQL)                 | 1,364 ms  | 1.71x slower        |
| 6  | FasterPybrot                 | 2,850 ms  | 3.58x slower        |
| 7  | FastPybrot                   | 3,327 ms  | 4.17x slower        |
| 8  | Pure Python                  | 4,328 ms  | 5.43x slower        |
| 9  | SQLite (SQL)                 | 44,918 ms | 56.36x slower       |

> Full optimization write-up with charts: [OPTIMIZATIONS.md](OPTIMIZATIONS.md) ·
> Web presentation: [performance.jakubjirak.com](https://performance.jakubjirak.com)

## Language Shootout

Seventy-two languages, one identical algorithm (float SIMD where available, all
cores, cardioid + period-2 bulb early-out, y-axis symmetry). Best-of-N runs,
Apple M4 Max:

| # | Language | Technique | Time |
|---|----------|-----------|------|
| 1 | Metal ([metalbrot.mm](metalbrot.mm)) | GPU compute shader, zero-copy | **0.31 ms** |
| 2 | Objective-C ([objcbrot.m](objcbrot.m)) | clang ext_vector + GCD — **CPU champion** | 0.34 ms |
| 3 | Odin ([odinbrot.odin](odinbrot.odin)) | #simd[8]f32 + thread pool | 0.35 ms |
| 4 | C++ ([cppbrot.cpp](cppbrot.cpp)) | NEON intrinsics + GCD | 0.39 ms |
| 5 | Swift ([swiftbrot.swift](swiftbrot.swift)) | SIMD8&lt;Float&gt; + concurrentPerform | 0.39 ms |
| 6 | D ([dbrot.d](dbrot.d)) | core.simd + LLVM FMA, LDC | 0.39 ms |
| 7 | Rust ([rustbrot/](rustbrot/)) | NEON intrinsics + rayon | 0.4 ms |
| 8 | V ([vbrot.v](vbrot.v)) | shaped autovectorization, thread pool | 0.41 ms |
| 9 | Zig ([zigbrot.zig](zigbrot.zig)) | @Vector(8,f32) + GCD | 0.45 ms |
| 10 | ARM64 asm ([asmbrot.s](asmbrot.s)) | hand-written NEON + GCD shim | 0.54 ms |
| 11 | WebAssembly ([wasmbrot.wat](wasmbrot.wat)) | hand-written WAT, SIMD128, wasmtime | 0.64 ms |
| 12 | C3 ([c3brot.c3](c3brot.c3)) | float[<8>] SIMD + OS threads | 0.8 ms |
| 13 | ISPC ([ispcbrot.ispc](ispcbrot.ispc)) | SPMD NEON i32x8 + libdispatch | 0.91 ms |
| 14 | C ([cbrot.c](cbrot.c)) | scalar, `-O3` autovectorization only | 1.23 ms |
| 15 | Java ([JavaBrot.java](JavaBrot.java)) | Vector API + parallel streams | 1.26 ms |
| 16 | Scala ([ScalaBrot.scala](ScalaBrot.scala)) | Vector API + parallel streams | 1.26 ms |
| 17 | Kotlin ([KotlinBrot.kt](KotlinBrot.kt)) | Vector API + parallel streams | 1.3 ms |
| 18 | Nim ([nimbrot.nim](nimbrot.nim)) | thread pool, `-d:danger` | 1.35 ms |
| 19 | Vala ([valabrot.vala](valabrot.vala)) | compiles to C, GLib thread pool | 1.35 ms |
| 20 | Fortran ([fortranbrot.f90](fortranbrot.f90)) | OpenMP | 1.38 ms |
| 21 | Pythran ([pythranbrot_impl.py](pythranbrot_impl.py)) | Python→C++/xsimd, OpenMP | 1.4 ms |
| 22 | Futhark ([futharkbrot.fut](futharkbrot.fut)) | data-parallel, multicore backend | 1.41 ms |
| 23 | Go ([gobrot_src/](gobrot_src/)) | goroutines, c-shared | 1.41 ms |
| 24 | Groovy ([GroovyBrot.groovy](GroovyBrot.groovy)) | @CompileStatic + Vector API | 1.47 ms |
| 25 | Cython ([cythonbrot.pyx](cythonbrot.pyx)) | nogil float32 kernel + OpenMP prange | 1.53 ms |
| 26 | C# ([csharpbrot/](csharpbrot/)) | AdvSimd intrinsics + Parallel.For | 1.67 ms |
| 27 | Haskell ([haskellbrot.hs](haskellbrot.hs)) | GHC -threaded, unboxed loop | 1.94 ms |
| 28 | Free Pascal ([pascalbrot.pas](pascalbrot.pas)) | RTLEvent thread pool | 1.95 ms |
| 29 | Chapel ([chapelbrot.chpl](chapelbrot.chpl)) | forall + dynamic iterator | 1.99 ms |
| 30 | Clojure ([clojurebrot.clj](clojurebrot.clj)) | unchecked primitives, thread pool | 2.1 ms |
| 31 | Haxe ([HaxeBrot.hx](HaxeBrot.hx)) | C++ target, sys.thread pool | 2.12 ms |
| 32 | JavaScript ([jsbrot.mjs](jsbrot.mjs)) | Node worker_threads + SAB | 2.19 ms |
| 33 | Numba ([numbabrot.py](numbabrot.py)) | @njit(parallel=True) prange, fastmath | 2.23 ms |
| 34 | Standard ML ([smlbrot.sml](smlbrot.sml)) | MLton whole-program opt, process pool | 2.37 ms |
| 35 | Lean 4 ([leanbrot.lean](leanbrot.lean)) | compiled via C, Task.spawn pool | 2.41 ms |
| 36 | F# ([fsharpbrot/](fsharpbrot/)) | AdvSimd intrinsics + Parallel.For | 2.44 ms |
| 37 | Common Lisp ([sbclbrot.lisp](sbclbrot.lisp)) | SBCL sb-thread, typed floats | 2.7 ms |
| 38 | Dart ([dartbrot.dart](dartbrot.dart)) | AOT + isolate pool | 2.7 ms |
| 39 | Chez Scheme ([chezbrot.ss](chezbrot.ss)) | fl-ops, fork-thread pool | 2.77 ms |
| 40 | OCaml ([ocamlbrot.ml](ocamlbrot.ml)) | OCaml 5 domain pool | 2.82 ms |
| 41 | PHP ([phpbrot.php](phpbrot.php)) | opcache JIT, process pool | 2.95 ms |
| 42 | Racket ([racketbrot.rkt](racketbrot.rkt)) | CS compiler, places pool | 3.33 ms |
| 43 | Julia ([juliabrot.jl](juliabrot.jl)) | @threads | 3.55 ms |
| 44 | CHICKEN ([chickenbrot.scm](chickenbrot.scm)) | Scheme→C native, process pool | 5.99 ms |
| 45 | Halide ([halidebrot.cpp](halidebrot.cpp)) | Tuple/RDom pipeline, parallel+vectorize(8), JIT | 7.16 ms |
| 46 | Crystal ([crystalbrot.cr](crystalbrot.cr)) | multi-threaded fibers | 8.94 ms |
| 47 | Pony ([ponybrot/main.pony](ponybrot/main.pony)) | work-stealing actors, LLVM | 9.32 ms |
| 48 | Gambit ([gambitbrot.scm](gambitbrot.scm)) | Scheme→C native, process pool | 11.48 ms |
| 49 | Gleam ([gleambrot_src/](gleambrot_src/src/gleambrot.gleam)) | typed BEAM, process per row band | 14.5 ms |
| 50 | Elixir ([elixirbrot.exs](elixirbrot.exs)) | BEAM Task.async_stream | 17.52 ms |
| 51 | LuaJIT ([luajitbrot.lua](luajitbrot.lua)) | persistent process pool | 19.43 ms |
| 52 | Forth ([forthbrot.fs](forthbrot.fs)) | gforth-fast worker pool | 22.98 ms |
| 53 | Erlang ([erlangbrot.erl](erlangbrot.erl)) | BEAM, process per row band | 30.2 ms |
| 54 | Ruby ([rubybrot.rb](rubybrot.rb)) | YJIT + Ractor pool | 35.61 ms |
| 55 | Janet ([janetbrot.janet](janetbrot.janet)) | ev/spawn-thread OS-thread pool | 35.72 ms |
| 56 | Factor ([factorbrot.factor](factorbrot.factor)) | concatenative JIT, process pool | 59.18 ms |
| 57 | Squirrel ([squirrelbrot.nut](squirrelbrot.nut)) | VM worker pool, interleaved rows | 61.4 ms |
| 58 | Raku ([rakubrot.raku](rakubrot.raku)) | MoarVM, native num, start/await | 78.93 ms |
| 59 | R ([rbrot.R](rbrot.R)) | vectorized whole-grid (NumPy-style) | 108.5 ms |
| 60 | Emacs Lisp ([elispbrot.el](elispbrot.el)) | native-comp, batch worker pool | 109.76 ms |
| 61 | Perl ([perlbrot.pl](perlbrot.pl)) | process pool | 153.5 ms |
| 62 | PostScript ([psbrot.ps](psbrot.ps)) | Ghostscript worker pool | 178.04 ms |
| 63 | Prolog ([prologbrot.pl](prologbrot.pl)) | SWI-Prolog threads, message queue | 181.03 ms |
| 64 | Tcl ([tclbrot.tcl](tclbrot.tcl)) | tclsh process pool, pipelined bands | 183.6 ms |
| 65 | COBOL ([cobolbrot.cob](cobolbrot.cob)) | Q28 fixed-point, process pool | 201.3 ms |
| 66 | AWK ([awkbrot.awk](awkbrot.awk)) | gawk process pool, binary %c | 277.6 ms |
| 67 | Wren ([wrenbrot.wren](wrenbrot.wren)) | VM pool, base-255 stdout protocol | 387.63 ms |
| 68 | GNU Smalltalk ([smalltalkbrot.st](smalltalkbrot.st)) | bytecode VM, process pool | 389.97 ms |
| 69 | Guile ([guilebrot.scm](guilebrot.scm)) | JIT + POSIX thread pool | 1,090 ms |
| 70 | Rexx ([rexxbrot.rexx](rexxbrot.rexx)) | Regina pool, decimal string math | 1,681 ms |
| 71 | GNU APL ([aplbrot.apl](aplbrot.apl)) | whole-grid vectorized, process pool | 2,934 ms |
| 72 | Bash ([bashbrot.sh](bashbrot.sh)) | Q26 fixed-point, persistent pool | 5,589 ms |

Notable: **Objective-C (clang `ext_vector_type`) is the fastest CPU entry** —
generic clang vector extensions out-scheduled hand-picked NEON intrinsics.
**Odin** sits 4 % behind; Swift/D/Rust/V within 20 %. **Hand-written assembly
loses ~55 % to the best compiled entries** — compilers model the M4 pipeline
better than humans. **WebAssembly at 0.64 ms** beats plain C: portable SIMD128
with a ~2× tax on native. Vector API entries (Java/Scala/Kotlin/Groovy) cluster
at 1.3–1.5 ms. **Vala matches Nim** (1.35 ms) by compiling to plain C, and
Python's own escape hatches — **Cython (1.53 ms), Numba (2.23 ms) and Pythran
(1.4 ms, Python→C++/xsimd)** — land in the compiled tier, ~2 000× faster than
the interpreter they extend. **The explicit-SIMD challengers landed but did not
dethrone anyone**: C3 (`float[<8>]`) at 0.80 ms and ISPC (SPMD NEON) at 0.91 ms
sit just behind plain C, and Halide's schedule-DSL pipeline at 7.16 ms trails
badly — all three share one wound, a *vector-group escape loop cannot retire a
lane early*, so every gang runs to `max_iter` for its slowest pixel; the
hand-written NEON kernel wins precisely because it amortizes the escape check
instead. **Gleam beats Erlang 2×** (14.5 vs 30.2 ms) on the same BEAM —
monomorphized typed code pays less boxing. Three Schemes, three backends:
CHICKEN (5.99 ms) and Gambit (11.48 ms) compile to C; Guile's bytecode+JIT with
boxed flonums is ~180× slower (1.09 s). **GNU APL (2.93 s)** is the honest floor
of whole-grid array evaluation — no per-pixel early-exit, six float64 temporaries
per escape step. The scripting tail (Perl/PostScript/Prolog/Tcl/
COBOL/AWK at 150–280 ms) is still ~150 000× faster than SQLite's recursive
CTE; **Rexx (1.7 s, decimal string arithmetic) and Bash (5.6 s, Q26 fixed
point — Bash has no floats at all)** bring up the rear and *still* beat
SQLite by 8×. COBOL required fixed-point Q28 arithmetic — GnuCOBOL floats
route through a GMP decimal runtime that made the naive version 25× slower.
Compiled/JIT entries pay build & warm-up at import, outside the timed path;
managed runtimes run as persistent warm workers.

**Winner overall: Hybrid on the WWDC 2026 Metal 4 command model**
([`hybrid4brot.mm`](hybrid4brot.mm)) — the same CPU+GPU hybrid, but the GPU half is
submitted through Metal 4 (`MTL4CommandQueue`, `MTL4ArgumentTable` bound by GPU
address, `MTLResidencySet`). The cheaper per-dispatch submit shifts the optimal
split to 76 % GPU and lands **~0.15 ms — a robust ~31 % faster than the Metal 3
hybrid (0.24 ms), ~6 500× faster than optimized NumPy** (wins 90 % of paired
samples, bit-identical output). Prior hybrid: [`hybridbrot.mm`](hybridbrot.mm)
(Metal 3, 0.24 ms); GPU-only crown: [`metalbrot.mm`](metalbrot.mm); CPU-only
crown: [`objcbrot.m`](objcbrot.m).

**Winner SQL: ArrowDatafusion** - Incredibly fast, nearly matching optimized NumPy performance!

### The fastest entry: `cppbrot` (C++ / ARM NEON / GCD)

`cppbrot.cpp` extracts the maximum from an M-series Mac. It is built to a dylib on
first import (clang `-O3 -mcpu=apple-m4`) and called from Python via `ctypes`:

- **ARM NEON, float32×4** — 4 pixels per vector lane, FMA, branch-free masked
  escape-time counting (`float32` is exact enough for the default view; the image
  is identical to the `float64` reference).
- **ILP interleave** — the `z = z² + c` recurrence is a latency-bound dependency
  chain, so multiple independent vector groups run in flight to hide FP latency.
- **y-axis symmetry** — the fixed viewport (−1..1) is symmetric about the real
  axis and Mandelbrot escape counts are conjugation-invariant, so only half the
  rows are computed; the rest is a `memcpy` mirror (~2x).
- **Amortized escape checks** — the serializing horizontal reduce + branch runs
  every 4th iteration; per-pixel counts stay exact (dead lanes stay masked).
- **libdispatch (`dispatch_apply`)** — rows fanned across all 10 P + 4 E cores
  with work-stealing; no thread-pool code. The GCD pool is warmed at import so
  the first timed call doesn't pay thread-creation latency.
- **Cardioid + period-2 bulb early-out** — the large in-set regions (the
  expensive full-`max_iter` pixels) are skipped analytically.

Measured ~0.4 ms (best) / ~0.7 ms (single timed call, fresh process) for
1400×800×256 on an Apple M4 Max.

## How It Works

The Mandelbrot set is computed by iterating the formula `z = z² + c` for each pixel in the complex plane:

```sql
WITH RECURSIVE
  -- Generate pixel grid and map to complex plane
  pixels AS (
    SELECT
      x, y,
      -2.5 + (x * 3.5 / width) AS cx,
      -1.0 + (y * 2.0 / height) AS cy
    FROM generate_series(0, width-1) AS x,
         generate_series(0, height-1) AS y
  ),
  -- Recursively iterate z = z² + c
  mandelbrot_iterations AS (
    SELECT x, y, cx, cy, 0.0 AS zx, 0.0 AS zy, 0 AS iteration
    FROM pixels

    UNION ALL

    SELECT
      x, y, cx, cy,
      zx * zx - zy * zy + cx AS zx,
      2.0 * zx * zy + cy AS zy,
      iteration + 1
    FROM mandelbrot_iterations
    WHERE iteration < max_iterations
      AND (zx * zx + zy * zy) <= 4.0
  )
SELECT x, y, MAX(iteration) AS depth
FROM mandelbrot_iterations
GROUP BY x, y;
```

The iteration count determines the color of each pixel, creating the iconic fractal pattern.

## Adding New Benchmarks

Want to test PostgreSQL, MySQL, MariaDB, SQLite or even Oracle or SQL-Server? Just:

1. Create a new file (e.g., `postgresqlbrot.py`)
2. Implement a `run_postgresqlbrot(width, height, max_iterations)` function (the DuckDB implementation is a good starting point)
3. Add one line to `main.py`:
   ```python
   BENCHMARKS = [
       ("DuckDB (SQL)", "duckbrot", "run_duckbrot"),
       ("Pure Python", "pybrot", "run_pybrot"),
       ..., 
       ("PostgreSQL", "postgresqlbrot", "run_postgresqlbrot"),  # New!
   ]
   ```

The framework handles everything else automatically!

## Configuration

Adjust the benchmark parameters in `main.py`:

```python
WIDTH = 1400           # Image width in pixels
HEIGHT = 800           # Image height in pixels
MAX_ITERATIONS = 256   # Maximum recursion depth
```

Higher values = more detail, longer computation time.

## Known Engine Compatibility

### ✅ Works Great
- **NumPy** - Highly optimized with loop unrolling and vectorized operations (fastest!)
- **DuckDB** - Excellent performance, proper DOUBLE precision
- **Pure Python** - Reference implementation, just to have an idea how fast the database engines are
- **SQLite** - Works but significantly slower due to recursive CTE overhead

### Should Work (untested, please contribute 🤙)
- PostgreSQL (with proper recursive CTE support)
- SQLite (may need query adjustments)
- others 

### Known Issues
- Some engines might struggle with support for DOUBLE precision and may use DECIMAL (not good for fractals, and lead to pixelated results)
- Watch out for type inference - explicit `::DOUBLE` casts are critical!

## What This Tests

This benchmark evaluates:
1. **Recursive CTE Performance** - How efficiently engines handle deep recursion
2. **Floating-Point Precision** - DOUBLE vs DECIMAL arithmetic accuracy
3. **Query Optimization** - How well engines optimize complex recursive queries
4. **Scalability** - Performance with increasing iterations and resolution

## Contributing

Contributions very welcome! Especially:
- New SQL engine implementations (PostgreSQL, MySQL, SQLite, etc.)
- Performance optimizations
- Better visualization options
- Benchmark result submissions

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Credits

Created by Thomas Zeutschler, Ulrich Ludmann

Inspired by the mathematical beauty of the Mandelbrot set and the curiosity about SQL engine performance.

## Learn More

- [Mandelbrot Set (Wikipedia)](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [SQL Recursive CTEs](https://en.wikipedia.org/wiki/Hierarchical_and_recursive_queries_in_SQL)
- [DuckDB](https://duckdb.org/)

---

**Curious which database renders infinity fastest? Clone and find out! 🌀**
