<?php
// PHP Mandelbrot worker: reads "w h it r0 r1\n", computes rows [r0,r1)
// of the top half, writes them as raw uint16 LE bytes to stdout. Loops to EOF.

function compute_rows(int $w, int $h, int $maxit, int $r0, int $r1): string
{
    $dx = 3.5 / ($w - 1);
    $dy = 2.0 / ($h - 1);
    $chunks = [];
    for ($row = $r0; $row < $r1; $row++) {
        $ci = -1.0 + $row * $dy;
        $ci2 = $ci * $ci;
        $line = [];
        for ($x = 0; $x < $w; $x++) {
            $cr = -2.5 + $x * $dx;
            $crm = $cr - 0.25;
            $q = $crm * $crm + $ci2;
            $cp1 = $cr + 1.0;
            if ($q * ($q + $crm) <= 0.25 * $ci2 || $cp1 * $cp1 + $ci2 <= 0.0625) {
                $line[] = $maxit; // cardioid / period-2 bulb: in set
            } else {
                $zr = 0.0;
                $zi = 0.0;
                $it = 0;
                while ($it < $maxit) {
                    $zr2 = $zr * $zr;
                    $zi2 = $zi * $zi;
                    if ($zr2 + $zi2 > 4.0) {
                        break;
                    }
                    $zi = 2.0 * $zr * $zi + $ci;
                    $zr = $zr2 - $zi2 + $cr;
                    $it++;
                }
                $line[] = $it;
            }
        }
        $chunks[] = pack('v*', ...$line);
    }
    return implode('', $chunks);
}

while (($line = fgets(STDIN)) !== false) {
    if (sscanf($line, '%d %d %d %d %d', $w, $h, $maxit, $r0, $r1) !== 5) {
        continue;
    }
    fwrite(STDOUT, compute_rows($w, $h, $maxit, $r0, $r1));
    fflush(STDOUT);
}
