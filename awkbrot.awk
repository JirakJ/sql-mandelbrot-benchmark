# Mandelbrot worker: reads "w h maxit r0 r1" lines, computes rows [r0,r1)
# of the top half, writes them as raw uint16 LE bytes to stdout. Loops to EOF.
# Run with gawk -b under LC_ALL=C so "%c" is byte-exact for 0-255 (incl. NUL).
{
    w = $1 + 0; h = $2 + 0; maxit = $3 + 0; r0 = $4 + 0; r1 = $5 + 0
    dx = 3.5 / (w - 1)
    dy = 2.0 / (h - 1)
    if (maxit != pmax) {  # cache little-endian byte pair per count value
        for (v = 0; v <= maxit; v++)
            pack[v] = sprintf("%c%c", v % 256, int(v / 256))
        pmax = maxit
    }
    for (row = r0; row < r1; row++) {
        ci = row * dy - 1.0
        ci2 = ci * ci
        q4 = 0.25 * ci2
        out = ""
        for (x = 0; x < w; x++) {
            cr = x * dx - 2.5
            crm = cr - 0.25
            q = crm * crm + ci2
            cp = cr + 1.0
            if (q * (q + crm) <= q4 || cp * cp + ci2 <= 0.0625) {
                out = out pack[maxit]  # cardioid / period-2 bulb: in set
                continue
            }
            zr = 0.0; zi = 0.0; it = 0
            while (it < maxit) {
                zr2 = zr * zr
                zi2 = zi * zi
                if (zr2 + zi2 > 4.0) break
                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr
                it++
            }
            out = out pack[it]
        }
        printf "%s", out
    }
    fflush()
}
