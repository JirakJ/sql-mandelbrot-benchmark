"""
OcamlBrot - OCaml 5 multicore (Domains) Mandelbrot via a persistent worker.

The native worker (ocamlbrot.ml) is compiled with ocamlopt at import time
(untimed) and kept alive for the whole benchmark. run_ocamlbrot only writes
a request line and reads raw uint16 pixels back.

Author: optimized for M-series Macs
License: MIT
"""

import atexit
import os
import subprocess

import numpy as np

from utils import save_mandelbrot_image

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "ocamlbrot.ml")
_BIN = os.path.join(_DIR, "ocamlbrot_bin")


def _build():
    """Compile the worker if the binary is missing or stale."""
    if os.path.exists(_BIN) and os.path.getmtime(_BIN) >= os.path.getmtime(_SRC):
        return
    cmds = [
        ["ocamlfind", "ocamlopt", "-package", "unix", "-linkpkg",
         "-unsafe", "-inline", "200", _SRC, "-o", _BIN],
        ["ocamlopt", "-unsafe", "-inline", "200", _SRC, "-o", _BIN],
        ["ocamlopt", "-I", "+unix", "unix.cmxa",
         "-unsafe", "-inline", "200", _SRC, "-o", _BIN],
    ]
    errs = []
    for cmd in cmds:
        try:
            res = subprocess.run(cmd, cwd=_DIR, capture_output=True)
        except FileNotFoundError as exc:
            errs.append(str(exc))
            continue
        if res.returncode == 0:
            return
        errs.append(res.stderr.decode())
    raise RuntimeError("ocamlbrot: build failed:\n" + "\n".join(errs))


_build()

# One persistent worker for the whole benchmark; logging goes to stderr only.
_proc = subprocess.Popen(
    [_BIN],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
)
atexit.register(_proc.terminate)


def run_ocamlbrot(width, height, max_iterations):
    """Compute the Mandelbrot set in the OCaml worker. Returns (h, w) uint16."""
    _proc.stdin.write(f"{width} {height} {max_iterations}\n".encode())
    _proc.stdin.flush()
    out = np.empty((height, width), dtype=np.uint16)
    view = memoryview(out).cast("B")
    total = width * height * 2
    got = 0
    while got < total:
        n = _proc.stdout.readinto(view[got:])
        if not n:
            raise RuntimeError("ocamlbrot: worker died mid-frame")
        got += n
    return out


# Warm-up at import (untimed): protocol check plus one full-size frame so the
# domain pool and allocator are hot before timing.
run_ocamlbrot(16, 16, 8)
run_ocamlbrot(1400, 800, 256)


if __name__ == "__main__":
    WIDTH, HEIGHT, MAX_ITERATIONS = 1400, 800, 256
    print(f"Computing Mandelbrot set ({WIDTH}x{HEIGHT}, max {MAX_ITERATIONS})...")
    result = run_ocamlbrot(WIDTH, HEIGHT, MAX_ITERATIONS)
    save_mandelbrot_image(result, MAX_ITERATIONS, "ocamlbrot.png")
