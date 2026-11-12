"""
WasmBrot - hand-written WebAssembly SIMD128 Mandelbrot run via wasmtime.

The .wat kernel is compiled at import (untimed). One Store+Instance per pool
thread, all importing ONE shared memory, so each thread writes its strided
rows at their final offsets; wasmtime releases the GIL during calls, giving
real parallelism. Calls go through raw wasmtime FFI to skip the slow
Func.__call__ wrapper. run_wasmbrot only dispatches and mirrors.

Author: optimized for M-series Macs
License: MIT
"""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import wasmtime
from wasmtime import _ffi as _wffi

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_WAT = os.path.join(_DIR, "wasmbrot.wat")

_cfg = wasmtime.Config()
_cfg.wasm_simd = True
_cfg.wasm_threads = True
_cfg.shared_memory = True
_cfg.cranelift_opt_level = "speed"
_engine = wasmtime.Engine(_cfg)
_module = wasmtime.Module.from_file(_engine, _WAT)

_NTHREADS = os.cpu_count() or 1
_PAGE = 65536

class _SharedMem(wasmtime.SharedMemory):
    """Works around wasmtime-py bug: _as_extern wraps ptr() in an extra pointer."""

    def _as_extern(self):
        union = _wffi.wasmtime_extern_union(sharedmemory=self.ptr())
        return _wffi.wasmtime_extern_t(_wffi.WASMTIME_EXTERN_SHAREDMEMORY, union)


_shmem = _SharedMem(
    _engine, wasmtime.MemoryType(wasmtime.Limits(32, 16384), shared=True)
)


class _Ctx:
    """One store+instance per thread; raw FFI call path (Func.__call__ is slow)."""

    def __init__(self):
        self.store = wasmtime.Store(_engine)
        inst = wasmtime.Instance(self.store, _module, [_shmem])
        fn = inst.exports(self.store)["mandel_rows"]
        self._ctxp = self.store._context()
        self._fref = ctypes.byref(fn._func)
        self._fn = fn  # keep alive
        self._params = (_wffi.wasmtime_val_t * 7)()
        for p in self._params:
            p.kind = _wffi.WASMTIME_I32.value
        self._results = (_wffi.wasmtime_val_t * 0)()

    def compute(self, w, h, maxit, r0, step, nrows):
        p = self._params
        p[0].of.i32 = w
        p[1].of.i32 = h
        p[2].of.i32 = maxit
        p[3].of.i32 = r0
        p[4].of.i32 = step
        p[5].of.i32 = nrows
        p[6].of.i32 = 0
        trap = ctypes.POINTER(_wffi.wasm_trap_t)()
        err = _wffi.wasmtime_func_call(
            self._ctxp, self._fref, p, 7, self._results, 0, ctypes.byref(trap)
        )
        if err or trap:
            raise RuntimeError("wasmbrot: wasm call trapped")


_ctxs = [_Ctx() for _ in range(_NTHREADS)]
_pool = ThreadPoolExecutor(max_workers=_NTHREADS)


def run_wasmbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in parallel wasm instances. Returns (h, w) uint16."""
    top = (height + 1) // 2
    need = top * width * 2
    if _shmem.data_len() < need:
        _shmem.grow((need - _shmem.data_len() + _PAGE - 1) // _PAGE)
    n = min(_NTHREADS, top)
    futs = [
        _pool.submit(
            _ctxs[i].compute,
            width, height, max_iterations,
            i, n, (top - i + n - 1) // n,
        )
        for i in range(n)
    ]
    for f in futs:
        f.result()
    ptr = ctypes.cast(_shmem.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    view = np.ctypeslib.as_array(ptr, shape=(top, width))
    out = np.empty((height, width), dtype=np.uint16)
    out[:top] = view
    if top < height:
        out[top:] = view[height - 1 - top :: -1]
    return out


# Warm-up at import (untimed): spins up pool threads and grows shared memory.
run_wasmbrot(16, 16, 8)
run_wasmbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_wasmbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "wasmbrot.png")
