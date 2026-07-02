#!/usr/bin/perl
# Mandelbrot band worker: reads "w h max_iter r0 r1" lines on STDIN,
# writes (r1-r0)*w uint16 little-endian counts (row-major) to STDOUT.
use strict;
use warnings;

$| = 1;
binmode(STDOUT);

# Per-column tables cached across requests (keyed on width).
my ($lw, @cr, @crm, @crm2, @cp1sq);

while (my $line = <STDIN>) {
    my ($w, $h, $mi, $r0, $r1) = split ' ', $line;
    last unless defined $r1;

    my $dy = 2.0 / ($h - 1);

    if (!defined($lw) || $lw != $w) {
        $lw = $w;
        my $dx = 3.5 / ($w - 1);
        @cr = @crm = @crm2 = @cp1sq = ();
        for my $x (0 .. $w - 1) {
            my $c = -2.5 + $x * $dx;
            $cr[$x] = $c;
            my $m = $c - 0.25;
            $crm[$x]  = $m;
            $crm2[$x] = $m * $m;
            my $p = $c + 1.0;
            $cp1sq[$x] = $p * $p;
        }
    }

    my $buf = '';
    for my $y ($r0 .. $r1 - 1) {
        my $ci  = -1.0 + $y * $dy;
        my $ci2 = $ci * $ci;
        my $bulb = 0.0625 - $ci2;    # period-2 bulb: cp1sq <= bulb
        my $card = 0.25 * $ci2;      # cardioid: q*(q+crm) <= card
        my @row;
        for my $x (0 .. $w - 1) {
            if ($cp1sq[$x] <= $bulb) { push @row, $mi; next; }
            my $q = $crm2[$x] + $ci2;
            if ($q * ($q + $crm[$x]) <= $card) { push @row, $mi; next; }
            my $cre = $cr[$x];
            my $zr  = 0.0;
            my $zi  = 0.0;
            my $it  = 0;
            my ($zr2, $zi2);
            while ($it < $mi) {
                $zr2 = $zr * $zr;
                $zi2 = $zi * $zi;
                last if $zr2 + $zi2 > 4.0;
                $zi = 2.0 * $zr * $zi + $ci;
                $zr = $zr2 - $zi2 + $cre;
                ++$it;
            }
            push @row, $it;
        }
        $buf .= pack('v*', @row);
    }

    my $off = 0;
    my $len = length $buf;
    while ($off < $len) {
        my $n = syswrite(STDOUT, $buf, $len - $off, $off);
        die "perlbrot worker: write failed: $!" unless defined $n;
        $off += $n;
    }
}
