# Rakudo/MoarVM Mandelbrot worker.
# Protocol: "<width> <height> <max_iter>\n" on stdin ->
# width*height*2 bytes uint16 LE row-major on stdout. Loops until EOF.
use nqp;

my int $DEG = $*KERNEL.cpu-cores;

# Compute rows $k, $k+$stride, ... of the top half into $buf (interleaved
# static partitioning balances the heavy middle rows across threads).
# Hot loop uses native num/int locals only; no allocation per pixel.
sub rows(Buf $buf, int $w, int $h, int $top, int $maxit,
         num $dx, num $dy, int $k, int $stride) {
    my int $row = $k;
    while $row < $top {
        my num $ci  = -1.0e0 + $row * $dy;
        my num $ci2 = $ci * $ci;
        my int $ofs = $row * $w;
        my int $x   = 0;
        while $x < $w {
            my num $cr  = -2.5e0 + $x * $dx;
            my num $crm = $cr - 0.25e0;
            my num $q   = $crm * $crm + $ci2;
            my num $cp1 = $cr + 1.0e0;
            my int $res;
            if $q * ($q + $crm) <= 0.25e0 * $ci2
            || $cp1 * $cp1 + $ci2 <= 0.0625e0 {
                $res = $maxit;                 # cardioid / period-2 bulb
            }
            else {
                my num $zr = 0e0;
                my num $zi = 0e0;
                my int $it = 0;
                while $it < $maxit {
                    my num $zr2 = $zr * $zr;
                    my num $zi2 = $zi * $zi;
                    last if $zr2 + $zi2 > 4e0;
                    $zi = 2e0 * $zr * $zi + $ci;
                    $zr = $zr2 - $zi2 + $cr;
                    $it = $it + 1;
                }
                $res = $it;
            }
            nqp::bindpos_i($buf, $ofs + $x, $res);
            $x = $x + 1;
        }
        $row = $row + $stride;
    }
}

sub frame(int $w, int $h, int $maxit --> Buf) {
    my int $top = ($h + 1) div 2;
    my $buf := Buf[uint16].allocate($w * $h);
    my num $dx = 3.5e0 / ($w - 1);
    my num $dy = 2.0e0 / ($h - 1);

    await (^$DEG).map: -> int $k {
        start rows($buf, $w, $h, $top, $maxit, $dx, $dy, $k, $DEG);
    };

    # Mirror bottom half (y-axis symmetry); odd middle row stays.
    my int $j = $top;
    while $j < $h {
        my int $src = ($h - 1 - $j) * $w;
        my int $dst = $j * $w;
        my int $x   = 0;
        while $x < $w {
            nqp::bindpos_i($buf, $dst + $x, nqp::atpos_i($buf, $src + $x));
            $x = $x + 1;
        }
        $j = $j + 1;
    }
    $buf
}

my $out = $*OUT;
loop {
    my $line = $*IN.get;
    last without $line;
    my ($w, $h, $it) = $line.words.map(*.Int);
    $out.write(frame($w, $h, $it));
    $out.flush;
}
