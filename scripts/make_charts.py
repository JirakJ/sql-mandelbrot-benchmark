"""Generate benchmark charts for docs + landing page (jakubjirak.com dark palette)."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#07111f"
PANEL = "#0f1d32"
BLUE = "#1d4ed8"
LIGHT_BLUE = "#93c5fd"
TEAL = "#14b8a6"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
RED = "#ef4444"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": "#1e3a5f",
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "Helvetica Neue",
    "font.size": 12,
    "grid.color": "#1e3a5f",
    "grid.alpha": 0.5,
})

OUT = sys.argv[1] if len(sys.argv) > 1 else "web/assets"

# (name, ms, highlight) — clean single-run measurements, M4 Max, 1400x800x256
slow = json.load(open(
    "/private/tmp/claude-501/-Users-jirakj-work-sql-mandelbrot-benchmark/"
    "ac13c787-0d92-4813-a7b0-77fbfa304935/scratchpad/slow_bench.json"))
LADDER = [
    ("SQLite (SQL)", slow["sqlitebrot"], False),
    ("Pure Python", slow["pybrot"], False),
    ("FastPybrot", slow["fastpybrot"], False),
    ("FasterPybrot", slow["fasterpybrot"], False),
    ("DuckDB (SQL)", 2331.0, False),
    ("DataFusion (SQL)", 1135.0, False),
    ("NumPy (vectorized)", 972.0, False),
    ("C++ NEON (CPU)", 0.39, True),
    ("Metal (GPU)", 0.30, True),
    ("Hybrid — Metal 3", 0.24, True),
    ("Hybrid — Metal 4", 0.15, True),
]

fig, ax = plt.subplots(figsize=(10, 5.6))
names = [x[0] for x in LADDER]
vals = [x[1] for x in LADDER]
colors = [TEAL if x[2] else BLUE for x in LADDER]
bars = ax.barh(names, vals, color=colors, height=0.62)
ax.set_xscale("log")
ax.set_xlabel("time, ms — log scale (lower is better)")
ax.set_title("Mandelbrot 1400×800 × 256 iterations — Apple M4 Max", pad=14, fontsize=14)
for b, v in zip(bars, vals):
    label = f"{v:,.2f} ms" if v < 10 else f"{v:,.0f} ms"
    ax.text(b.get_width() * 1.25, b.get_y() + b.get_height() / 2, label,
            va="center", fontsize=10.5, color=TEXT)
ax.set_xlim(0.1, 400000)
span = LADDER[0][1] / LADDER[-1][1]
ax.text(0.985, 0.88, f"≈ {span:,.0f}× faster\nsix orders of magnitude",
        transform=ax.transAxes, ha="right", va="top", fontsize=12.5,
        color=TEAL, fontweight="bold", linespacing=1.4)
ax.grid(axis="x", which="both")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/chart-ladder.png", dpi=2.0 * 72, facecolor=BG)
plt.close(fig)

# Optimization journey
JOURNEY = [
    ("NumPy\n(starting point)", 972.0, BLUE),
    ("v1: NEON SIMD\n+ all cores + early-out", 0.79, LIGHT_BLUE),
    ("v2: + y-symmetry\n+ amortized checks", 0.39, TEAL),
    ("v3: Metal GPU\n+ zero-copy", 0.30, "#5eead4"),
    ("v4: hybrid\nCPU + GPU together", 0.24, "#99f6e4"),
    ("v5: Metal 4\ncommand model", 0.15, "#ccfbf1"),
]
fig, ax = plt.subplots(figsize=(10, 5.2))
xs = range(len(JOURNEY))
vals = [j[1] for j in JOURNEY]
ax.bar(xs, vals, color=[j[2] for j in JOURNEY], width=0.55)
ax.set_yscale("log")
ax.set_xticks(list(xs))
ax.set_xticklabels([j[0] for j in JOURNEY], fontsize=11)
ax.set_ylabel("time, ms — log scale")
ax.set_title("Optimization journey — same workload, same machine", pad=14, fontsize=14)
for x, v in zip(xs, vals):
    ax.text(x, v * 1.35, f"{v:g} ms", ha="center", fontsize=12, color=TEXT,
            fontweight="bold")
speedups = ["1×", "1 230×", "2 490×", "3 240×", "4 120×", "6 480×"]
for x, (v, s) in enumerate(zip(vals, speedups)):
    ax.text(x, v * 0.35, s, ha="center", fontsize=11, color=BG if x else TEXT,
            fontweight="bold")
ax.set_ylim(0.05, 4000)
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/chart-journey.png", dpi=2.0 * 72, facecolor=BG)
plt.close(fig)

# CPU vs GPU scaling
SCALING = [  # (label, Mpx, cpu_ms, gpu_ms)
    ("1400×800", 1.12, 0.52, 0.34),
    ("2800×1600", 4.48, 1.38, 0.61),
    ("5600×3200", 17.9, 4.94, 1.58),
    ("11200×6400", 71.7, 24.54, 1.87),
]
fig, ax = plt.subplots(figsize=(10, 5.2))
mpx = [s[1] for s in SCALING]
cpu = [s[2] for s in SCALING]
gpu = [s[3] for s in SCALING]
ax.plot(mpx, cpu, "o-", color=LIGHT_BLUE, lw=2.5, ms=8, label="C++ NEON — 14 CPU cores")
ax.plot(mpx, gpu, "o-", color=TEAL, lw=2.5, ms=8, label="Metal — 40-core GPU")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("megapixels")
ax.set_ylabel("time, ms")
ax.set_title("CPU vs GPU scaling — the gap grows with the workload", pad=14, fontsize=14)
for x, c, g, (lbl, *_ ) in zip(mpx, cpu, gpu, SCALING):
    ax.annotate(f"{c/g:.1f}×", (x, g), textcoords="offset points", xytext=(0, -22),
                ha="center", color=TEAL, fontsize=11, fontweight="bold")
ax.legend(facecolor=PANEL, edgecolor="#1e3a5f", labelcolor=TEXT, fontsize=11)
ax.grid(which="both")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/chart-scaling.png", dpi=2.0 * 72, facecolor=BG)
plt.close(fig)

# Language shootout (best-of-N, same algorithm: SIMD/threads/symmetry/early-out)
# teal = explicit SIMD (GPU / intrinsics / vector types / SIMD API / hand asm), blue = scalar
LANGS = [
    ("Metal (GPU)", 0.31, True),
    ("Objective-C (ext_vector)", 0.34, True),
    ("Odin (#simd)", 0.35, True),
    ("C++ (NEON intrinsics)", 0.39, True),
    ("Swift (SIMD8)", 0.39, True),
    ("D (core.simd, LDC)", 0.39, True),
    ("Rust (NEON + rayon)", 0.40, True),
    ("V (shaped autovec)", 0.41, False),
    ("Zig (@Vector)", 0.45, True),
    ("ARM64 assembly (NEON)", 0.54, True),
    ("WebAssembly (WAT SIMD128)", 0.64, True),
    ("C (autovectorized)", 1.23, False),
    ("Java (Vector API)", 1.26, True),
    ("Scala (Vector API)", 1.26, True),
    ("Kotlin (Vector API)", 1.30, True),
    ("Nim (threads)", 1.35, False),
    ("Vala (C backend, GLib)", 1.35, False),
    ("Fortran (OpenMP)", 1.38, False),
    ("Futhark (multicore)", 1.41, False),
    ("Go (goroutines)", 1.41, False),
    ("Groovy (Vector API)", 1.47, True),
    ("Cython (nogil + OpenMP)", 1.53, False),
    ("C# (AdvSimd)", 1.67, True),
    ("Haskell (GHC threads)", 1.94, False),
    ("Free Pascal (threads)", 1.95, False),
    ("Chapel (forall)", 1.99, False),
    ("Clojure (JVM primitives)", 2.10, False),
    ("Haxe (C++ target)", 2.12, False),
    ("JavaScript (Node workers)", 2.19, False),
    ("Numba (JIT prange)", 2.23, False),
    ("Standard ML (MLton)", 2.37, False),
    ("Lean 4 (Tasks)", 2.41, False),
    ("F# (AdvSimd)", 2.44, True),
    ("Common Lisp (SBCL)", 2.70, False),
    ("Dart (isolates)", 2.70, False),
    ("Chez Scheme (fl ops)", 2.77, False),
    ("OCaml (domains)", 2.82, False),
    ("PHP (JIT, process pool)", 2.95, False),
    ("Racket (places)", 3.33, False),
    ("Julia (threads)", 3.55, False),
    ("Crystal (MT fibers)", 8.94, False),
    ("Pony (actors)", 9.32, False),
    ("Gleam (typed BEAM)", 14.50, False),
    ("Elixir (BEAM)", 17.52, False),
    ("LuaJIT (process pool)", 19.43, False),
    ("Forth (gforth pool)", 22.98, False),
    ("Erlang (BEAM)", 30.20, False),
    ("Ruby (YJIT + Ractors)", 35.61, False),
    ("Janet (ev threads)", 35.72, False),
    ("Squirrel (VM pool)", 61.40, False),
    ("Raku (MoarVM)", 78.93, False),
    ("R (vectorized)", 108.47, False),
    ("Emacs Lisp (native-comp)", 109.76, False),
    ("Perl (process pool)", 153.54, False),
    ("PostScript (Ghostscript)", 178.04, False),
    ("Prolog (SWI threads)", 181.03, False),
    ("Tcl (process pool)", 183.60, False),
    ("COBOL (Q28 fixed-point)", 201.28, False),
    ("AWK (gawk pool)", 277.65, False),
    ("Wren (VM pool)", 387.63, False),
    ("Rexx (decimal string math)", 1680.88, False),
    ("Bash (Q26 fixed-point)", 5589.11, False),
    ("C3 (float[<8>] SIMD)", 0.80, True),
    ("ISPC (SPMD NEON)", 0.91, True),
    ("Pythran (Py->C++/xsimd)", 1.40, False),
    ("CHICKEN (Scheme->C)", 5.99, False),
    ("Halide (schedule DSL)", 7.16, True),
    ("Gambit (Scheme->C)", 11.48, False),
    ("Factor (concatenative)", 59.18, False),
    ("GNU Smalltalk (pool)", 389.97, False),
    ("Guile (JIT threads)", 1089.61, False),
    ("GNU APL (whole-grid)", 2933.82, False),
    # Overall records — combined CPU+GPU, NOT single languages (gold)
    ("★ Hybrid — Metal 4 (CPU+GPU)", 0.15, "REC"),
    ("★ Hybrid — Metal 3 (CPU+GPU)", 0.24, "REC"),
]
LANGS.sort(key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(10.5, 25.2))
names = [x[0] for x in LANGS][::-1]
vals = [x[1] for x in LANGS][::-1]
GOLD = "#fbbf24"
colors = [(GOLD if x[2] == "REC" else TEAL if x[2] else BLUE) for x in LANGS][::-1]
bars = ax.barh(names, vals, color=colors, height=0.62)
ax.set_xscale("log")
ax.set_xlabel("time, ms — log scale (lower is better)")
ax.set_title("72 languages + the CPU+GPU record — Apple M4 Max, 1400×800 × 256 it.",
             pad=14, fontsize=14)
for b, v in zip(bars, vals):
    lbl = f"{v:.2f} ms" if v < 100 else f"{v:,.0f} ms"
    ax.text(b.get_width() * 1.09, b.get_y() + b.get_height() / 2, lbl,
            va="center", fontsize=10, color=TEXT)
ax.set_xlim(0.1, 16000)
ax.grid(axis="x", which="both")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{OUT}/chart-languages.png", dpi=2.0 * 72, facecolor=BG)
plt.close(fig)

print("charts written to", OUT)
