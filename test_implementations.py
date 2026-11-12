"""
Correctness tests for the Python-side Mandelbrot implementations.

Every implementation is checked against an independent scalar reference
(escape-time iteration written out in this file), on:

* hand-picked points whose escape count is known analytically, placed
  exactly on pixel centres of a small grid, and
* a full comparison grid, pixel by pixel.

The native / GPU / other-language entries need their own toolchains and
are not exercised here; for those only the benchmark registry is checked.
"""

from pathlib import Path

import numpy as np
import pytest

# (module, function) for every implementation that only needs requirements.txt
IMPLEMENTATIONS = [
    ("pybrot", "run_pybrot"),
    ("fastpybrot", "run_pybrot"),
    ("fasterpybrot", "run_pybrot"),
    ("numpybrot", "run_numpybrot"),
    ("duckbrot", "run_duckbrot"),
    ("sqlitebrot", "run_sqlitebrot"),
    ("arrow_datafusion", "run_arrow_datafusion"),
]

# View window shared by all implementations: re [-2.5, 1.0], im [-1.0, 1.0]
RE_MIN, RE_MAX = -2.5, 1.0
IM_MIN, IM_MAX = -1.0, 1.0

# Comparison grid: big enough to cover the interesting boundary, small
# enough that the slowest engine (SQLite recursive CTE) stays well < 1 s.
GRID_W, GRID_H, GRID_MAX = 64, 40, 64

# Fraction of pixels allowed to differ from the reference. The engines are
# currently bit-exact; the slack only absorbs a different FMA/rounding
# strategy on some CPU flipping a chaotic boundary pixel.
MAX_MISMATCH_FRACTION = 0.005


def reference_escape(cx: float, cy: float, max_iterations: int) -> int:
    """Number of z -> z^2 + c steps taken before |z| > 2 (capped)."""
    zx = zy = 0.0
    for i in range(max_iterations):
        if zx * zx + zy * zy > 4.0:
            return i
        zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
    return max_iterations


def reference_grid(width: int, height: int, max_iterations: int) -> np.ndarray:
    out = np.empty((height, width), dtype=np.int64)
    for y in range(height):
        cy = IM_MIN + y * (IM_MAX - IM_MIN) / (height - 1)
        for x in range(width):
            cx = RE_MIN + x * (RE_MAX - RE_MIN) / (width - 1)
            out[y, x] = reference_escape(cx, cy, max_iterations)
    return out


def run(module_name: str, func_name: str, width: int, height: int, max_iter: int):
    module = pytest.importorskip(module_name)
    result = getattr(module, func_name)(width, height, max_iter)
    return np.asarray(result).astype(np.int64)


impl_params = pytest.mark.parametrize(
    "module_name,func_name",
    IMPLEMENTATIONS,
    ids=[m for m, _ in IMPLEMENTATIONS],
)


@pytest.fixture(scope="module")
def reference():
    return reference_grid(GRID_W, GRID_H, GRID_MAX)


# --- the reference itself ---------------------------------------------------

# 15 x 9 pixels -> step 0.25 in both axes, so these c values are exact pixels.
KNOWN_W, KNOWN_H, KNOWN_MAX = 15, 9, 50
KNOWN_POINTS = [
    # (cx, cy, expected escape count)
    (0.0, 0.0, KNOWN_MAX),  # fixed point z = 0
    (-1.0, 0.0, KNOWN_MAX),  # period-2 cycle 0, -1, 0, ...
    (-2.0, 0.0, KNOWN_MAX),  # tip of the set: z = 2 forever, |z|^2 == 4
    (0.25, 0.0, KNOWN_MAX),  # cusp of the main cardioid
    (0.0, 1.0, KNOWN_MAX),  # c = i: preperiodic, bounded
    (-0.25, 0.25, KNOWN_MAX),  # inside the main cardioid
    (1.0, 0.0, 3),  # 0, 1, 2, 5 -> escapes on the 3rd check
    (-2.5, 0.0, 1),  # |c| > 2 escapes after one step
    (1.0, 1.0, 2),  # 0, 1+i, 1+3i
    (-2.5, 1.0, 1),  # far corner
]


def pixel_of(cx: float, cy: float) -> tuple[int, int]:
    x = round((cx - RE_MIN) / (RE_MAX - RE_MIN) * (KNOWN_W - 1))
    y = round((cy - IM_MIN) / (IM_MAX - IM_MIN) * (KNOWN_H - 1))
    return y, x


@pytest.mark.parametrize("cx,cy,expected", KNOWN_POINTS)
def test_reference_known_points(cx, cy, expected):
    assert reference_escape(cx, cy, KNOWN_MAX) == expected


def test_reference_is_symmetric_about_real_axis(reference):
    # Mandelbrot set is symmetric under conjugation; the grid is symmetric
    # in y, so rows mirror (allowing a stray boundary pixel from rounding).
    mismatch = np.count_nonzero(reference != reference[::-1, :])
    assert mismatch <= MAX_MISMATCH_FRACTION * reference.size


# --- every implementation ---------------------------------------------------


@impl_params
def test_output_shape_and_range(module_name, func_name):
    result = run(module_name, func_name, 31, 17, 40)
    assert result.shape == (17, 31)
    assert result.min() >= 0
    assert result.max() == 40  # the window always contains set points


@impl_params
def test_known_points(module_name, func_name):
    result = run(module_name, func_name, KNOWN_W, KNOWN_H, KNOWN_MAX)
    for cx, cy, expected in KNOWN_POINTS:
        y, x = pixel_of(cx, cy)
        assert result[y, x] == expected, f"c = {cx}{cy:+}i at pixel ({y}, {x})"


@impl_params
def test_matches_reference(module_name, func_name, reference):
    result = run(module_name, func_name, GRID_W, GRID_H, GRID_MAX)
    assert result.shape == reference.shape
    diff = result != reference
    assert np.count_nonzero(diff) <= MAX_MISMATCH_FRACTION * reference.size, (
        f"{np.count_nonzero(diff)} of {reference.size} pixels differ, "
        f"max |delta| = {np.abs(result - reference).max()}"
    )
    # Set membership must agree everywhere except at those rare pixels.
    in_set = result == GRID_MAX
    assert np.count_nonzero(in_set != (reference == GRID_MAX)) <= np.count_nonzero(diff)


@impl_params
def test_single_iteration(module_name, func_name):
    # With a cap of 1 every pixel takes exactly one step (catches off-by-one).
    result = run(module_name, func_name, KNOWN_W, KNOWN_H, 1)
    np.testing.assert_array_equal(result, reference_grid(KNOWN_W, KNOWN_H, 1))


# --- utilities and registry -------------------------------------------------


def test_save_mandelbrot_image(tmp_path, monkeypatch):
    from matplotlib import colormaps
    from PIL import Image

    from utils import save_mandelbrot_image

    monkeypatch.chdir(tmp_path)
    data = reference_grid(20, 15, 20)
    save_mandelbrot_image(data, 20, "test.png")

    out = tmp_path / "images" / "test.png"
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (20, 15)
        # in-set pixels get the darkest colour of the map, escaped ones not
        darkest = tuple(int(v * 255) for v in colormaps["hot"](0.0)[:3])
        y, x = np.argwhere(data == 20)[0]
        assert img.getpixel((int(x), int(y)))[:3] == darkest
        y, x = np.argwhere(data == data[data < 20].max())[0]
        assert img.getpixel((int(x), int(y)))[:3] != darkest


def test_run_benchmark_returns_result_and_time():
    from pybrot import run_pybrot
    from utils import run_benchmark

    result, elapsed_ms = run_benchmark("pure python", run_pybrot, 10, 8, 5)
    assert np.asarray(result).shape == (8, 10)
    assert elapsed_ms is not None and elapsed_ms >= 0


def test_benchmark_registry_points_at_existing_modules():
    from main import BENCHMARKS

    root = Path(__file__).parent
    names = [name for name, _, _ in BENCHMARKS]
    assert len(names) == len(set(names)), "duplicate benchmark names"
    for _, module_name, func_name in BENCHMARKS:
        source = root / f"{module_name}.py"
        assert source.exists(), f"missing {source.name}"
        assert f"def {func_name}(" in source.read_text(encoding="utf-8")
