// Squirrel Mandelbrot worker. Protocol: reads "w h maxit r0 r1 stride\n"
// from stdin, computes rows r0, r0+stride, ... < r1, writes them as raw
// uint16 LE bytes to stdout, flushes, loops until EOF.
// Math: Q28 fixed point in 64-bit ints with round-to-nearest shifts
// (this Squirrel build has 32-bit floats; int64 fixed point is more
// accurate vs the float64 reference).

const SH = 28;
const SHM1 = 27;

function readLine() {
    local s = "";
    while (true) {
        local c;
        try { c = stdin.readn('b'); } catch (e) { return null; }
        if (c == 10) return s;
        s += c.tochar();
    }
}

local buf = null;

while (true) {
    local line = readLine();
    if (line == null || line == "") break;
    local p = split(line, " ");
    local w = p[0].tointeger();
    local h = p[1].tointeger();
    local maxit = p[2].tointeger();
    local r0 = p[3].tointeger();
    local r1 = p[4].tointeger();
    local stride = p[5].tointeger();

    local S = 1 << SH;
    local H1 = 1 << (SH - 1);   // rounding term for >> SH
    local H2 = 1 << (SH - 2);   // rounding term for >> (SH-1)
    local FOUR = 4 * S;
    local Q4 = S / 4;
    local S16 = S / 16;

    // cr[x] = -2.5 + x*3.5/(w-1), rounded to Q28
    local den = 2 * (w - 1);
    local hd = den / 2;
    local cr0 = -5 * S / 2;
    local crarr = array(w);
    for (local x = 0; x < w; x++) crarr[x] = cr0 + (7 * S * x + hd) / den;

    local nrows = 0;
    for (local r = r0; r < r1; r += stride) nrows++;
    local nbytes = nrows * w * 2;
    if (buf == null || buf.len() != nbytes) buf = blob(nbytes);
    else buf.seek(0);

    local de = h - 1;
    local hde = de / 2;
    for (local row = r0; row < r1; row += stride) {
        local ci = -S + (2 * S * row + hde) / de;
        local ci2 = (ci * ci + H1) >> SH;
        for (local x = 0; x < w; x++) {
            local cr = crarr[x];
            local res;
            local crm = cr - Q4;
            local q = ((crm * crm + H1) >> SH) + ci2;
            local cp1 = cr + S;
            if ((q * (q + crm) + H1) >> SH <= (ci2 + 2) >> 2
                || ((cp1 * cp1 + H1) >> SH) + ci2 <= S16) {
                res = maxit;  // cardioid / period-2 bulb
            } else {
                local zr = 0, zi = 0, it = 0;
                while (it < maxit) {
                    local zr2 = (zr * zr + H1) >> SH;
                    local zi2 = (zi * zi + H1) >> SH;
                    if (zr2 + zi2 > FOUR) break;
                    local nzr = zr2 - zi2 + cr;
                    zi = ((zr * zi + H2) >> SHM1) + ci;
                    zr = nzr;
                    it++;
                }
                res = it;
            }
            buf.writen(res, 'w');
        }
    }
    stdout.writeblob(buf);
    stdout.flush();
}
