# tclbrot.tcl - persistent Mandelbrot band worker.
# Protocol: stdin line "w h it r0 r1" -> (r1-r0)*w uint16 LE counts on stdout.
# Loops until EOF. All diagnostics would go to stderr; stdout is binary-only.

chan configure stdout -translation binary
chan configure stderr -buffering none

proc band {w h it r0 r1} {
    set dx [expr {3.5 / ($w - 1)}]
    set dy [expr {2.0 / ($h - 1)}]
    # Per-column constants, computed once per request.
    set crs {}
    set crms {}
    set crp2s {}
    for {set x 0} {$x < $w} {incr x} {
        set cr [expr {-2.5 + $x * $dx}]
        lappend crs $cr
        lappend crms [expr {$cr - 0.25}]
        lappend crp2s [expr {($cr + 1.0) * ($cr + 1.0)}]
    }
    set counts {}
    for {set row $r0} {$row < $r1} {incr row} {
        set ci [expr {-1.0 + $row * $dy}]
        set ci2 [expr {$ci * $ci}]
        set qci2 [expr {0.25 * $ci2}]
        foreach cr $crs crm $crms crp2 $crp2s {
            # Cardioid + period-2 bulb early-outs.
            set q [expr {$crm * $crm + $ci2}]
            if {$q * ($q + $crm) <= $qci2 || $crp2 + $ci2 <= 0.0625} {
                lappend counts $it
                continue
            }
            # First iteration folded: z0 = 0 -> z1 = c, one iteration survived.
            set zr $cr
            set zi $ci
            set i 1
            while {$i < $it} {
                set zr2 [expr {$zr * $zr}]
                set zi2 [expr {$zi * $zi}]
                if {$zr2 + $zi2 > 4.0} break
                set zi [expr {2.0 * $zr * $zi + $ci}]
                set zr [expr {$zr2 - $zi2 + $cr}]
                incr i
            }
            lappend counts $i
        }
    }
    puts -nonewline [binary format s* $counts]
    flush stdout
}

while {[gets stdin line] >= 0} {
    if {$line eq ""} continue
    lassign $line w h it r0 r1
    if {$it == 0} {
        puts -nonewline [binary format s* [lrepeat [expr {($r1 - $r0) * $w}] 0]]
        flush stdout
    } else {
        band $w $h $it $r0 $r1
    }
}
