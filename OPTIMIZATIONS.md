# Squeezing an Apple M4 Max: Mandelbrot from 972 ms to 0.30 ms

A case study in extracting maximum performance from Apple Silicon —
every optimization applied to this repository's Mandelbrot benchmark,
what it bought, and how it was verified.

**Workload:** 1400×800 pixels × 256 max iterations, escape-time Mandelbrot
(`z = z² + c`, escape when `|z|² > 4`), identical semantics to the SQL and
Python reference implementations in this repo.

**Machine:** MacBook Pro, Apple M4 Max — 10 performance + 4 efficiency CPU
cores (128-bit NEON, FMA), 40-core GPU, unified memory, macOS 26.5.

## Results

| Implementation | Time | Speedup vs NumPy |
|---|---|---|
| SQLite (recursive CTE) | ~238 s | 0.004× |
| Pure Python | 4 848 ms | 0.2× |
| DuckDB (SQL) | 2 331 ms | 0.4× |
| ArrowDataFusion (SQL) | 1 135 ms | 0.9× |
| NumPy (vectorized, unrolled) | 972 ms | 1× |
| **C++ NEON, all cores (v1)** | **0.79 ms** | **1 230×** |
| **+ y-symmetry + amortized checks (v2)** | **0.39 ms** | **2 490×** |
| **Metal GPU, zero-copy (v3)** | **0.30 ms** | **3 240×** |

![Benchmark ladder](web/assets/chart-ladder.png)
![Optimization journey](web/assets/chart-journey.png)

## CPU kernel — [`cppbrot.cpp`](cppbrot.cpp)

### 1. Native code, tuned for the exact silicon

Compiled at first import with `clang++ -O3 -mcpu=apple-m4` into a dylib,
called from Python via `ctypes`. `-mcpu=apple-m4` lets clang schedule for the
M4's actual pipeline widths and latencies. Build cost is paid once, outside
the timed path.

### 2. ARM NEON SIMD — 4 pixels per instruction

The escape-time loop runs on `float32x4_t` vectors: 4 pixels per lane with
fused multiply-add (`vmlaq_f32`). Escaped lanes are masked out branch-free
(`vcleq_f32` + `vandq_u32`), and the per-pixel iteration count is accumulated
by subtracting the all-ones mask (`count -= active`).

**float32 vs float64:** f32 gives 4 lanes instead of 2. At this viewport the
pixel pitch (~2.5×10⁻³) is ~10⁴ ulp of f32 — precision is ample. Verified:
99.8 % of escaped pixels within ±1 iteration of the f64 reference; the
remainder are chaotic boundary pixels. Images are visually identical.

### 3. All 14 cores via libdispatch

`dispatch_apply` fans rows across all P+E cores with work-stealing — zero
thread-pool code. Two findings the profiler forced on us:

- `QOS_CLASS_USER_INTERACTIVE` was a **regression** (median 1.75 ms vs
  0.41 ms) — `USER_INITIATED` schedules this workload better.
- The GCD pool is warmed with a tiny dispatch at import, so the first timed
  call doesn't pay thread-creation latency (1.73 ms → ~0.7 ms single call).

### 4. Cardioid + period-2 bulb early-out

Points inside the main cardioid and the period-2 bulb never escape — they are
the *most expensive* pixels (full 256 iterations each). Both regions have
closed-form tests (a handful of FLOPs), evaluated vectorized before iterating.
~22 % of all pixels skip the loop entirely.

### 5. Instruction-level parallelism (G=2 interleave)

`z = z² + c` is a serial dependency chain — each iteration waits on the last.
One 4-lane chain cannot fill the M4's SIMD pipes. The kernel runs **G=2
independent vector groups** (8 pixels) through the loop body so two chains
are always in flight. Measured: G=1 → 0.71 ms, **G=2 → 0.38 ms**, G=3 →
0.46 ms (register pressure starts to bite).

### 6. y-axis symmetry — the free 2×

The viewport (−2.5..1) × (−1..1) is symmetric about the real axis, and
Mandelbrot escape counts are conjugation-invariant: `conj(z² + c) =
conj(z)² + conj(c)`, and IEEE negation is exact, so the counts are
bit-identical. Compute the top half, `memcpy` the mirror. Only caveat: the
mirrored row's `cy` differs from exact negation by ≤1 ulp of grid rounding —
973 px (0.09 %) differ vs direct computation, accuracy vs f64 unchanged
(99.814 %).

### 7. Amortized escape checks

The horizontal reduce (`vmaxvq_u32`) + branch that tests "any lane still
alive?" serializes the loop. It now runs every 4th iteration; per-lane
masking stays per-iteration, so **counts remain exact** — dead lanes just run
up to 3 wasted iterations (inf/NaN compares are always false and never
re-activate a lane). Combined with symmetry: 0.79 ms → **0.39 ms**.

## GPU kernel — [`metalbrot.mm`](metalbrot.mm)

### 8. Metal compute shader, compiled at runtime

The MSL kernel ships as a source string compiled via `newLibraryWithSource`
at import — no metallib toolchain required. One thread per pixel of the top
half; each thread writes its pixel and the conjugate mirror pixel (symmetry
again, now with zero extra memory traffic). Cardioid/bulb early-out included —
on the GPU it saves whole 32-wide SIMD groups in the bulk regions.

### 9. Unified memory, zero-copy

Apple Silicon's unified memory means the GPU writes into a
`MTLResourceStorageModeShared` buffer the CPU can read directly. The result
array returned to Python is a **numpy view of the GPU buffer** — no copy at
all. Hazard tracking is off (`MTLResourceHazardTrackingModeUntracked`);
`waitUntilCompleted` is the only sync. Cutting the 2.2 MB memcpy + tracking
took the GPU path from 0.41 ms to **0.30 ms**.

### 10. Threadgroup shape

Swept threadgroup heights (32×4, 32×8, 32×16, 32×32): flat between 4 and 16,
regression at 32. Shipped 32×8 = 256 threads/group, matching Apple's
occupancy guidance.

## CPU vs GPU: the crossover

At the benchmark size the GPU wins only 1.5× — a Metal dispatch has a fixed
~0.15–0.2 ms command-buffer cost that dominates sub-millisecond work. Scale
the workload and the 40-core GPU pulls away:

| Workload | C++ NEON (14 cores) | Metal (40-core GPU) | GPU advantage |
|---|---|---|---|
| 1400×800 (1.1 Mpx) | 0.52 ms | 0.34 ms | 1.5× |
| 2800×1600 (4.5 Mpx) | 1.38 ms | 0.61 ms | 2.2× |
| 5600×3200 (18 Mpx) | 4.94 ms | 1.58 ms | 3.1× |
| 11200×6400 (72 Mpx) | 24.5 ms | 1.87 ms | **13.1×** |
| 1400×800 @ 4096 iter | 3.05 ms | 0.80 ms | 3.8× |

![CPU vs GPU scaling](web/assets/chart-scaling.png)

72 megapixels of 256-iteration Mandelbrot in 1.87 ms ≈ **38 gigapixel-iterations
per second**.

## Language shootout — one algorithm, twenty-two implementations

To separate "language speed" from "algorithm speed", the same optimized algorithm
(SIMD where the language exposes it, all cores, cardioid/bulb early-out, y-axis
symmetry, identical escape semantics) was implemented in twenty-two languages.
Best of 15, 1400×800×256:

| Language | Time | Notes |
|---|---|---|
| Metal (GPU) | 0.31 ms | compute shader, zero-copy unified memory |
| C++ | 0.39 ms | NEON intrinsics, G=2 ILP, GCD |
| Swift | 0.39 ms | `SIMD8<Float>`, `concurrentPerform` — ties C++ |
| Rust | 0.40 ms | `std::arch::aarch64` NEON + rayon |
| Zig | 0.45 ms | `@Vector(8, f32)`, `@mulAdd` FMA, GCD |
| ARM64 assembly | 0.54 ms | hand-written NEON kernel + C/GCD shim |
| C | 1.23 ms | scalar, autovectorization only — no intrinsics |
| Java | 1.26 ms | Vector API (`FloatVector`), parallel streams, warm JVM worker |
| Kotlin | 1.30 ms | Vector API via JVM interop, parallel streams |
| Nim | 1.35 ms | `-d:danger`, persistent thread pool, clang backend |
| Fortran | 1.38 ms | OpenMP, gfortran |
| Go | 1.41 ms | goroutines, atomic row counter, c-shared |
| C# | 1.67 ms | `AdvSimd` Vector128 intrinsics, `Parallel.For`, .NET 10 |
| Haskell | 1.94 ms | GHC `-threaded`, strict unboxed loop, zero-copy ByteString |
| JavaScript | 2.19 ms | Node `worker_threads` + SharedArrayBuffer |
| Common Lisp | 2.70 ms | SBCL, typed single-floats, sb-thread |
| Dart | 2.70 ms | AOT-compiled isolate pool |
| OCaml | 2.82 ms | OCaml 5 multicore domain pool |
| Julia | 3.55 ms | `Threads.@threads`, warm worker |
| Crystal | 8.94 ms | MT fibers (execution contexts) — bimodal scheduler, median 16.5 ms |
| LuaJIT | 19.43 ms | no threads: pool of 10 persistent worker processes |
| Ruby | 35.61 ms | Ruby 4.0, YJIT, one Ractor per core |

![Language shootout](web/assets/chart-languages.png)

Findings:

- **Swift ties C++; Rust and Zig sit within 15 %** — all compile to nearly the
  same NEON machine code. Language choice in this tier is ergonomics, not speed.
- **Hand-written assembly (0.54 ms) loses ~35 % to compiler + intrinsics.**
  The human-scheduled loop can't beat LLVM's model of the M4 pipeline. Write
  intrinsics, let the compiler schedule.
- **Intrinsics vs autovectorization is 3×** (C++ 0.39 vs C 1.23): the early-exit
  escape loop defeats clang's autovectorizer, so scalar C is what most "fast C"
  actually ships.
- **SIMD APIs on managed runtimes deliver**: Java 1.26 / Kotlin 1.30 (Vector
  API) and C# 1.67 (`AdvSimd`) put GC'd runtimes within ~4× of native
  intrinsics — ahead of scalar Fortran and Go.
- **The scripting tail is parallelism-limited, not arithmetic-limited**:
  LuaJIT's superb single-core JIT still needs a process pool (19 ms), and Ruby's
  Ractors carry heavy per-frame coordination (36 ms). Even Common Lisp (2.7 ms)
  runs circles around them.
- Fairness: compile & JIT warm-up happen at import (untimed), managed runtimes
  run as persistent warm workers; the timed path is compute + (for workers) a
  2.2 MB pipe transfer. Crystal's number is reported with its measured
  bimodality, not hidden behind the best case.

## Verification methodology

Every step was checked before it was kept:

- **Bit-exactness across restructures** — v2's top half is bit-identical to
  v1; every G value produces identical output.
- **Accuracy vs float64** — compared against the f64 NumPy reference on every
  escaped pixel; kept ≥99.76 % within ±1 iteration at every step (f32 CPU:
  99.81 %, GPU fast-math: 99.77 %).
- **Semantics** — same escape convention as the repo's Python/SQL
  implementations (count = iterations survived before `|z|² > 4`).
- **Timing** — best & median of 15–31 runs for steady-state; fresh-process
  single calls for the "what the harness sees" number; contenders re-measured
  on an idle machine.

## What didn't work

- `QOS_CLASS_USER_INTERACTIVE` — 4× *worse* median than `USER_INITIATED`.
- G=3/G=4 interleave — register pressure beats latency hiding past G=2.
- Threadgroups of 1024 on the GPU — occupancy loss.
- (v1 era) deeper unrolling of the masked-count loop — the workload was
  overhead-bound until symmetry halved the work; only then did ILP tuning
  show real differences.

## Reproduce

```bash
uv pip install -r requirements.txt
uv run python main.py          # full suite, all engines
uv run python cppbrot.py       # CPU kernel alone
uv run python metalbrot.py     # GPU kernel alone
```
