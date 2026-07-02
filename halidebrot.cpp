// halidebrot.cpp - Mandelbrot via the Halide image DSL (algorithm/schedule split).
//
// The pipeline is built once and JIT-compiled on the first realize() call (the
// Python wrapper triggers that as an untimed warm-up). extern "C" mandelbrot()
// then just sets the runtime Params and realizes into the caller's buffer.
//
// Algorithm matches the benchmark spec exactly: fixed viewport escape-time
// Mandelbrot with cardioid/bulb early-out, computed on the top half only and
// mirrored across the y axis in this shim.

#include <Halide.h>
#include <cstdint>
#include <cstring>

using namespace Halide;

namespace {

struct Kernel {
    Param<float> dx{"dx"}, dy{"dy"};
    Param<int> maxiter{"maxiter"};
    Func result{"result"};
    Var x{"x"}, y{"y"};
    bool built = false;

    void build() {
        // Map pixel -> complex plane.
        Expr cr = -2.5f + cast<float>(x) * dx;
        Expr ci = -1.0f + cast<float>(y) * dy;

        // Cardioid / period-2 bulb early-out.
        Expr crm = cr - 0.25f;
        Expr q = crm * crm + ci * ci;
        Expr in_cardioid = q * (q + crm) <= 0.25f * ci * ci;
        Expr in_bulb = (cr + 1.0f) * (cr + 1.0f) + ci * ci <= 0.0625f;
        Expr early = in_cardioid || in_bulb;

        // Escape-time iteration as a Tuple Func: (zr, zi, count).
        Func iter{"iter"};
        iter(x, y) = Tuple(0.0f, 0.0f, 0);
        RDom r(0, maxiter, "r");
        Expr zr = iter(x, y)[0];
        Expr zi = iter(x, y)[1];
        Expr cnt = iter(x, y)[2];
        Expr zr2 = zr * zr;
        Expr zi2 = zi * zi;
        Expr escaped = zr2 + zi2 > 4.0f;
        iter(x, y) = Tuple(
            select(escaped, zr, zr2 - zi2 + cr),
            select(escaped, zi, 2.0f * zr * zi + ci),
            cnt + select(escaped, 0, 1) + r * 0);  // r*0 keeps the RDom live

        result(x, y) = cast<uint16_t>(select(early, maxiter, iter(x, y)[2]));

        // Schedule: strip-mine columns into vectors of 8, parallelize rows,
        // materialize the iteration per vector so the Tuple reduction stays hot.
        Var xo{"xo"}, xi{"xi"};
        result.split(x, xo, xi, 8, TailStrategy::GuardWithIf)
              .parallel(y)
              .vectorize(xi);
        iter.compute_at(result, xo).vectorize(x, 8, TailStrategy::GuardWithIf);
        iter.update().vectorize(x, 8, TailStrategy::GuardWithIf);

        built = true;
    }
};

Kernel &pipe() {
    static Kernel p;
    if (!p.built) p.build();
    return p;
}

}  // namespace

extern "C" void mandelbrot(int w, int h, int mi, uint16_t *out) {
    Kernel &p = pipe();
    int top = (h + 1) / 2;
    p.dx.set(3.5f / (float)(w - 1));
    p.dy.set(2.0f / (float)(h - 1));
    p.maxiter.set(mi);

    // Wrap the first `top` rows of the caller's row-major buffer. Halide dim0
    // (x) has stride 1, dim1 (y) stride w -> matches out[y*w + x].
    Buffer<uint16_t> buf(out, {w, top});
    p.result.realize(buf);

    // Mirror across the y axis: out[h-1-r] = out[r] for r in [0, h/2).
    for (int r = 0; r < h / 2; ++r) {
        std::memcpy(out + (size_t)(h - 1 - r) * w, out + (size_t)r * w,
                    (size_t)w * sizeof(uint16_t));
    }
}
