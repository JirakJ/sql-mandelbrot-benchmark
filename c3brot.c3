// c3brot - C3 (LLVM) SIMD + threaded escape-time Mandelbrot kernel.
//
// Exposes a C-ABI `mandelbrot(int, int, int, ushort*)` called from Python via
// ctypes. Top-half rows are split into contiguous bands across up to 14 OS
// threads; each band is computed 8 columns at a time with float[<8>] SIMD and
// the cardioid/bulb early-out. The bottom half is mirrored from the top.

module c3brot;
import std::thread;
import std::os;
import std::math;

alias F8v = float[<8>];
alias B8v = bool[<8>];

struct Task
{
    int   w;
    int   h;
    int   mi;
    int   row_start;   // inclusive, top-half row index
    int   row_end;     // exclusive
    float dx;
    float dy;
    ushort* out;
}

// Element-wise a <= b, returning a lane mask.
macro B8v vle(F8v a, F8v b) => $$veccomple(a, b);

macro bool any_true(B8v m)
{
    F8v f = math::select(m, (F8v)1.0, (F8v)0.0);
    return f.sum() > 0.0;
}

fn void compute_band(Task* t)
{
    int w = t.w;
    int mi = t.mi;
    float dx = t.dx;
    float dy = t.dy;
    ushort* base = t.out;
    F8v four = 4.0;
    F8v quarter = 0.25;
    F8v fmi = (float)mi;

    // Per-lane column offset vector [0,1,2,...,7]
    F8v lane_idx;
    for (int i = 0; i < 8; i++) lane_idx[i] = (float)i;

    for (int r = t.row_start; r < t.row_end; r++)
    {
        float ci_s = -1.0f + (float)r * dy;
        F8v ci = ci_s;
        F8v ci2 = ci * ci;
        ushort* rowp = base + (usz)r * (usz)w;

        for (int x = 0; x < w; x += 8)
        {
            F8v xv = (float)x + lane_idx;
            F8v cr = -2.5f + xv * dx;
            F8v crm = cr - quarter;
            F8v q = crm * crm + ci2;

            // cardioid: q*(q+crm) <= 0.25*ci2 ; bulb: (cr+1)^2+ci2 <= 0.0625
            B8v card = vle(q * (q + crm), quarter * ci2);
            F8v crp1 = cr + 1.0f;
            B8v bulb = vle(crp1 * crp1 + ci2, (F8v)0.0625);
            B8v inset = card | bulb;

            F8v count = math::select(inset, fmi, (F8v)0.0);
            B8v active = !inset;

            F8v zr = 0.0;
            F8v zi = 0.0;
            for (int it = 0; it < mi; it++)
            {
                F8v zr2 = zr * zr;
                F8v zi2 = zi * zi;
                F8v mag = zr2 + zi2;
                B8v step = active & vle(mag, four);
                if (!any_true(step)) break;
                count = math::select(step, count + 1.0f, count);
                F8v nzi = 2.0f * zr * zi + ci;
                zr = zr2 - zi2 + cr;
                zi = nzi;
                active = step;
            }

            int lim = w - x;
            if (lim > 8) lim = 8;
            for (int k = 0; k < lim; k++)
            {
                rowp[x + k] = (ushort)count[k];
            }
        }
    }
}

fn int thread_entry(void* arg)
{
    compute_band((Task*)arg);
    return 0;
}

fn void mandelbrot(int width, int height, int max_iterations, ushort* out) @export("mandelbrot")
{
    int w = width;
    int h = height;
    int mi = max_iterations;
    if (w <= 0 || h <= 0) return;

    float dx = (float)(3.5 / (double)(w - 1));
    float dy = (float)(2.0 / (double)(h - 1));
    int top = (h + 1) / 2;

    int n = os::num_cpu();
    if (n > 14) n = 14;
    if (n < 1) n = 1;
    if (n > top) n = top;

    Task[14] tasks;
    Thread[14] threads;
    bool[14] joinable;
    for (int i = 0; i < 14; i++) joinable[i] = false;

    int per = (top + n - 1) / n;
    for (int i = 0; i < n; i++)
    {
        int rs = i * per;
        if (rs >= top) break;
        int re = rs + per;
        if (re > top) re = top;
        tasks[i] = {
            .w = w, .h = h, .mi = mi,
            .row_start = rs, .row_end = re,
            .dx = dx, .dy = dy, .out = out
        };
        if (i == n - 1)
        {
            // Run the last band on the calling thread.
            compute_band(&tasks[i]);
        }
        else if (catch threads[i].create(&thread_entry, &tasks[i]))
        {
            // Fallback: run inline if thread creation fails.
            compute_band(&tasks[i]);
        }
        else
        {
            joinable[i] = true;
        }
    }
    for (int i = 0; i < n; i++)
    {
        if (joinable[i]) (void)threads[i].join();
    }

    // Mirror top half into bottom half.
    for (int r = 0; r < h / 2; r++)
    {
        ushort* src = out + (usz)r * (usz)w;
        ushort* dst = out + (usz)(h - 1 - r) * (usz)w;
        for (int x = 0; x < w; x++) dst[x] = src[x];
    }
}
