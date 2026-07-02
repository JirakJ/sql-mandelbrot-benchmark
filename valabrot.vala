// ValaBrot - Mandelbrot kernel in Vala, compiled to a C-ABI dylib.
//
// - float64 scalar escape loop (exact vs the numpy reference)
// - cardioid + period-2 bulb analytic early-out
// - y-axis symmetry: top half computed, bottom half memcpy-mirrored
// - all cores: one GLib.Thread per CPU pulling rows off an AtomicInt counter

namespace ValaBrot {

    static void compute_row (uint16* outp, int width, int height,
                             int max_iter, double dx, double dy, int row) {
        double ci = -1.0 + row * dy;
        double ci2 = ci * ci;
        uint16* orow = outp + row * width;

        for (int x = 0; x < width; x++) {
            double cr = -2.5 + x * dx;
            double crm = cr - 0.25;
            double q = crm * crm + ci2;
            double crp = cr + 1.0;
            int result;
            if (q * (q + crm) <= 0.25 * ci2 || crp * crp + ci2 <= 0.0625) {
                result = max_iter; // main cardioid or period-2 bulb
            } else {
                double zr = 0.0, zi = 0.0;
                int it = 0;
                while (it < max_iter) {
                    double zr2 = zr * zr;
                    double zi2 = zi * zi;
                    if (zr2 + zi2 > 4.0) break;
                    double nzr = zr2 - zi2 + cr;
                    zi = 2.0 * zr * zi + ci;
                    zr = nzr;
                    it++;
                }
                result = it;
            }
            orow[x] = (uint16) result;
        }

        int mrow = height - 1 - row;
        if (mrow != row) {
            GLib.Memory.copy (outp + mrow * width, orow,
                              (size_t) width * sizeof (uint16));
        }
    }
}

[CCode (cname = "mandelbrot_vala")]
public void mandelbrot_vala (int width, int height, int max_iter,
                             [CCode (array_length = false)] uint16[] outbuf) {
    double dx = 3.5 / (double) (width - 1);
    double dy = 2.0 / (double) (height - 1);
    int half = (height + 1) / 2;

    int ncpu = (int) GLib.get_num_processors ();
    if (ncpu > half) ncpu = half;
    if (ncpu < 1) ncpu = 1;

    uint16* op = outbuf;
    int next_row = 0;

    var threads = new Thread<void*>[ncpu];
    for (int t = 0; t < ncpu; t++) {
        threads[t] = new Thread<void*> ("valabrot", () => {
            while (true) {
                int row = GLib.AtomicInt.add (ref next_row, 1);
                if (row >= half) break;
                ValaBrot.compute_row (op, width, height, max_iter, dx, dy, row);
            }
            return null;
        });
    }
    foreach (var th in threads) th.join ();
}
