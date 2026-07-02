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
| 1  | **Metal GPU**                |  ~0.3 ms  | **~0.0004x** ⭐     |
| 2  | **C++ NEON (SIMD + threads)**|  ~0.4 ms  | ~0.0005x            |
| 3  | NumPy (vectorized, unrolled) |   665 ms  | 0.83x               |
| 4  | ArrowDatafusion (SQL)        |   797 ms  | 1.00x (baseline)    |
| 5  | DuckDB (SQL)                 | 1,364 ms  | 1.71x slower        |
| 6  | FasterPybrot                 | 2,850 ms  | 3.58x slower        |
| 7  | FastPybrot                   | 3,327 ms  | 4.17x slower        |
| 8  | Pure Python                  | 4,328 ms  | 5.43x slower        |
| 9  | SQLite (SQL)                 | 44,918 ms | 56.36x slower       |

> Full optimization write-up with charts: [OPTIMIZATIONS.md](OPTIMIZATIONS.md) ·
> Web presentation: [performance.jakubjirak.com](https://performance.jakubjirak.com)

## Language Shootout

Forty-two languages, one identical algorithm (float SIMD where available, all
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
| 7 | Rust ([rustbrot/](rustbrot/)) | NEON intrinsics + rayon | 0.40 ms |
| 8 | V ([vbrot.v](vbrot.v)) | shaped autovectorization, thread pool | 0.41 ms |
| 9 | Zig ([zigbrot.zig](zigbrot.zig)) | @Vector(8,f32) + GCD | 0.45 ms |
| 10 | ARM64 asm ([asmbrot.s](asmbrot.s)) | hand-written NEON + GCD shim | 0.54 ms |
| 11 | WebAssembly ([wasmbrot.wat](wasmbrot.wat)) | hand-written WAT, SIMD128, wasmtime | 0.64 ms |
| 12 | C ([cbrot.c](cbrot.c)) | scalar, `-O3` autovectorization only | 1.23 ms |
| 13 | Java ([JavaBrot.java](JavaBrot.java)) | Vector API + parallel streams | 1.26 ms |
| 14 | Scala ([ScalaBrot.scala](ScalaBrot.scala)) | Vector API + parallel streams | 1.26 ms |
| 15 | Kotlin ([KotlinBrot.kt](KotlinBrot.kt)) | Vector API + parallel streams | 1.30 ms |
| 16 | Nim ([nimbrot.nim](nimbrot.nim)) | thread pool, `-d:danger` | 1.35 ms |
| 17 | Fortran ([fortranbrot.f90](fortranbrot.f90)) | OpenMP | 1.38 ms |
| 18 | Futhark ([futharkbrot.fut](futharkbrot.fut)) | data-parallel, multicore backend | 1.41 ms |
| 19 | Go ([gobrot_src/](gobrot_src/)) | goroutines, c-shared | 1.41 ms |
| 20 | Groovy ([GroovyBrot.groovy](GroovyBrot.groovy)) | @CompileStatic + Vector API | 1.47 ms |
| 21 | C# ([csharpbrot/](csharpbrot/)) | AdvSimd intrinsics + Parallel.For | 1.67 ms |
| 22 | Haskell ([haskellbrot.hs](haskellbrot.hs)) | GHC -threaded, unboxed loop | 1.94 ms |
| 23 | Free Pascal ([pascalbrot.pas](pascalbrot.pas)) | RTLEvent thread pool | 1.95 ms |
| 24 | Chapel ([chapelbrot.chpl](chapelbrot.chpl)) | forall + dynamic iterator | 1.99 ms |
| 25 | Clojure ([clojurebrot.clj](clojurebrot.clj)) | unchecked primitives, thread pool | 2.10 ms |
| 26 | Haxe ([HaxeBrot.hx](HaxeBrot.hx)) | C++ target, sys.thread pool | 2.12 ms |
| 27 | JavaScript ([jsbrot.mjs](jsbrot.mjs)) | Node worker_threads + SAB | 2.19 ms |
| 28 | F# ([fsharpbrot/](fsharpbrot/)) | AdvSimd intrinsics + Parallel.For | 2.44 ms |
| 29 | Common Lisp ([sbclbrot.lisp](sbclbrot.lisp)) | SBCL sb-thread, typed floats | 2.70 ms |
| 30 | Dart ([dartbrot.dart](dartbrot.dart)) | AOT + isolate pool | 2.70 ms |
| 31 | Chez Scheme ([chezbrot.ss](chezbrot.ss)) | fl-ops, fork-thread pool | 2.77 ms |
| 32 | OCaml ([ocamlbrot.ml](ocamlbrot.ml)) | OCaml 5 domain pool | 2.82 ms |
| 33 | PHP ([phpbrot.php](phpbrot.php)) | opcache JIT, process pool | 2.95 ms |
| 34 | Julia ([juliabrot.jl](juliabrot.jl)) | @threads | 3.55 ms |
| 35 | Crystal ([crystalbrot.cr](crystalbrot.cr)) | multi-threaded fibers | 8.94 ms |
| 36 | Elixir ([elixirbrot.exs](elixirbrot.exs)) | BEAM Task.async_stream | 17.52 ms |
| 37 | LuaJIT ([luajitbrot.lua](luajitbrot.lua)) | persistent process pool | 19.43 ms |
| 38 | Ruby ([rubybrot.rb](rubybrot.rb)) | YJIT + Ractor pool | 35.61 ms |
| 39 | R ([rbrot.R](rbrot.R)) | vectorized whole-grid (NumPy-style) | 108.5 ms |
| 40 | Perl ([perlbrot.pl](perlbrot.pl)) | process pool | 153.5 ms |
| 41 | COBOL ([cobolbrot.cob](cobolbrot.cob)) | Q28 fixed-point, process pool | 201.3 ms |
| 42 | AWK ([awkbrot.awk](awkbrot.awk)) | gawk process pool, binary %c | 277.6 ms |

Notable: **Objective-C (clang `ext_vector_type`) is the fastest CPU entry** —
generic clang vector extensions out-scheduled hand-picked NEON intrinsics.
**Odin** sits 4 % behind; Swift/D/Rust/V within 20 %. **Hand-written assembly
loses ~55 % to the best compiled entries** — compilers model the M4 pipeline
better than humans. **WebAssembly at 0.64 ms** beats plain C: portable SIMD128
with a ~2× tax on native. Vector API entries (Java/Scala/Kotlin/Groovy) cluster
at 1.3–1.5 ms. The scripting tail (Perl/COBOL/AWK at 150–280 ms) is still
~250 000× faster than SQLite's recursive CTE. COBOL required fixed-point Q28
arithmetic — GnuCOBOL floats route through a GMP decimal runtime that made the
naive version 25× slower. Compiled/JIT entries pay build & warm-up at import,
outside the timed path; managed runtimes run as persistent warm workers.

**Winner overall: Metal GPU** ([`metalbrot.mm`](metalbrot.mm)) — a runtime-compiled
compute shader with zero-copy unified memory, ~**3000x faster than the SQL baseline**.
CPU crown: [`cppbrot.cpp`](cppbrot.cpp), C++ + NEON + all 14 cores.

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
