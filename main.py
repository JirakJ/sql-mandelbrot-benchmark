"""
Mandelbrot Set Benchmark Suite

A comprehensive benchmark comparing different approaches to computing
the Mandelbrot set. Tests SQL engines, programming languages, and
various optimization techniques.

Author: Thomas Zeutschler
License: MIT
GitHub: https://github.com/Zeutschler/sql-mandelbrot-benchmark
"""

from utils import print_header, print_results, run_benchmark, save_mandelbrot_image

# Mandelbrot set configuration
WIDTH = 1400
HEIGHT = 800
MAX_ITERATIONS = 256

# Benchmark registry: (name, module, function)
BENCHMARKS = [
    ("Metal GPU", "metalbrot", "run_metalbrot"),
    ("C++ NEON (SIMD+threads)", "cppbrot", "run_cppbrot"),
    ("ARM64 Assembly (NEON)", "asmbrot", "run_asmbrot"),
    ("Rust (NEON+rayon)", "rustbrot", "run_rustbrot"),
    ("Swift (SIMD8+GCD)", "swiftbrot", "run_swiftbrot"),
    ("C (autovectorized)", "cbrot", "run_cbrot"),
    ("Fortran (OpenMP)", "fortranbrot", "run_fortranbrot"),
    ("Go (goroutines)", "gobrot", "run_gobrot"),
    ("Java (Vector API)", "javabrot", "run_javabrot"),
    ("Julia (threads)", "juliabrot", "run_juliabrot"),
    ("JavaScript (Node workers)", "jsbrot", "run_jsbrot"),
    ("Dart (isolates)", "dartbrot", "run_dartbrot"),
    ("Zig (@Vector SIMD)", "zigbrot", "run_zigbrot"),
    ("C# (AdvSimd)", "csharpbrot", "run_csharpbrot"),
    ("Kotlin (Vector API)", "kotlinbrot", "run_kotlinbrot"),
    ("Nim (threads)", "nimbrot", "run_nimbrot"),
    ("Crystal (MT)", "crystalbrot", "run_crystalbrot"),
    ("Haskell (GHC threads)", "haskellbrot", "run_haskellbrot"),
    ("OCaml (domains)", "ocamlbrot", "run_ocamlbrot"),
    ("Common Lisp (SBCL)", "sbclbrot", "run_sbclbrot"),
    ("LuaJIT (process pool)", "luajitbrot", "run_luajitbrot"),
    ("Ruby (YJIT+Ractors)", "rubybrot", "run_rubybrot"),
    ("NumPy (Vectorized)", "numpybrot", "run_numpybrot"),
    ("ArrowDatafusion", "arrow_datafusion", "run_arrow_datafusion"),
    ("DuckDB (SQL)", "duckbrot", "run_duckbrot"),
    ("FastPybrot", "fastpybrot", "run_pybrot"),
    ("FasterPybrot", "fasterpybrot", "run_pybrot"),
    ("Pure Python", "pybrot", "run_pybrot"),
    ("SQLite", "sqlitebrot", "run_sqlitebrot"),
    # Add more benchmarks here:
    # ("PostgreSQL", "postgresqlbrot", "run_postgresqlbrot"),
    # ("MySQL", "mysqlbrot", "run_mysqlbrot"),
]


def main():
    """Run all available benchmarks."""
    print_header(WIDTH, HEIGHT, MAX_ITERATIONS)

    results = []

    # Run all available benchmarks
    for name, module_name, func_name in BENCHMARKS:
        try:
            module = __import__(module_name)
            func = getattr(module, func_name)
            result, elapsed_ms = run_benchmark(
                name, func, WIDTH, HEIGHT, MAX_ITERATIONS
            )
            results.append((name, elapsed_ms))

            # Save the generated image
            if result is not None:
                filename = f"{module_name}.png"
                save_mandelbrot_image(result, MAX_ITERATIONS, filename)

        except ImportError as e:
            print(f"\n⊘ {name} benchmark not available: {e}")
        except AttributeError as e:
            print(f"\n⊘ {name} benchmark missing function: {e}")

    # Print summary
    print_results(results, WIDTH, HEIGHT, MAX_ITERATIONS)


if __name__ == "__main__":
    main()
