"smalltalkbrot.st - GNU Smalltalk Mandelbrot worker (persistent process).

Reads request lines 'w h mi a b' from stdin, computes the top-half row band
[a, b) with FloatD arithmetic and the cardioid/bulb early-outs, packs each row
as width uint16 little-endian values into a ByteArray, and writes the raw bytes
to stdout. The Python wrapper (pattern B) runs a pool of these workers, one
per core, and mirrors the bottom half for y-axis symmetry.

Run:  /opt/homebrew/bin/gst smalltalkbrot.st"

Object subclass: Mandel [
    Mandel class >> computeW: w h: h mi: mi from: aRow to: bRow [
        | dx dy nbytes row ci ci2 cr crm q zr zi zr2 zi2 it val idx |
        dx := 3.5 / (w - 1).
        dy := 2.0 / (h - 1).
        nbytes := w * 2.
        aRow to: bRow - 1 do: [ :r |
            row := ByteArray new: nbytes.
            ci := -1.0 + (r * dy).
            ci2 := ci * ci.
            idx := 1.
            0 to: w - 1 do: [ :x |
                cr := -2.5 + (x * dx).
                crm := cr - 0.25.
                q := (crm * crm) + ci2.
                (((q * (q + crm)) <= (0.25 * ci2))
                    or: [ (((cr + 1.0) * (cr + 1.0)) + ci2) <= 0.0625 ])
                    ifTrue: [ val := mi ]
                    ifFalse: [
                        zr := 0.0. zi := 0.0. zr2 := 0.0. zi2 := 0.0. it := 0.
                        [ (it < mi) and: [ (zr2 + zi2) <= 4.0 ] ] whileTrue: [
                            zi := (2.0 * zr * zi) + ci.
                            zr := (zr2 - zi2) + cr.
                            zr2 := zr * zr.
                            zi2 := zi * zi.
                            it := it + 1 ].
                        val := it ].
                row at: idx put: (val bitAnd: 255).
                row at: idx + 1 put: ((val bitShift: -8) bitAnd: 255).
                idx := idx + 2 ].
            stdout nextPutAll: row ].
        stdout flush.
    ]
]

| line parts w h mi a b |
[ (line := stdin nextLine) notNil ] whileTrue: [
    parts := line substrings.
    parts isEmpty ifFalse: [
        w := (parts at: 1) asInteger.
        h := (parts at: 2) asInteger.
        mi := (parts at: 3) asInteger.
        a := (parts at: 4) asInteger.
        b := (parts at: 5) asInteger.
        Mandel computeW: w h: h mi: mi from: a to: b ] ].
