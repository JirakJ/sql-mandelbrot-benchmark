// ISPC (SPMD SIMD) Mandelbrot row kernel, NEON i32x8.
//
// Semantics match the benchmark reference exactly: value = iterations
// survived before |z|^2 > 4 (check-then-iterate), capped at max_iter
// (in-set pixels -> max_iter). Cardioid + period-2 bulb early-out.
// One exported entry computes a single image row over all columns; the
// C shim fans rows across cores with libdispatch and mirrors the bottom half.

export void mandel_row(uniform int w, uniform float dx, uniform float x0,
                       uniform float ci, uniform int mi, uniform uint16 out[]) {
    uniform float ci2 = ci * ci;
    foreach (x = 0 ... w) {
        float cr = x0 + (float)x * dx;

        // main cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2
        float crm = cr - 0.25f;
        float q = crm * crm + ci2;
        bool in_card = (q * (q + crm) <= 0.25f * ci2);
        // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
        float crp = cr + 1.0f;
        bool in_bulb = (crp * crp + ci2 <= 0.0625f);

        int it = mi;
        if (!(in_card || in_bulb)) {
            float zr = 0.0f, zi = 0.0f;
            int i = 0;
            // per-lane escape-time loop; lanes that have escaped stop counting
            while (i < mi) {
                float zr2 = zr * zr;
                float zi2 = zi * zi;
                if (zr2 + zi2 > 4.0f)
                    break;
                zi = 2.0f * zr * zi + ci;
                zr = zr2 - zi2 + cr;
                i++;
            }
            it = i;
        }
        out[x] = (uint16)it;
    }
}
