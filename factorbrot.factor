! FactorBrot - Mandelbrot worker in the Factor concatenative language.
!
! Factor threads are cooperative (green), so parallelism comes from a pool
! of persistent worker processes (see factorbrot.py). Each worker reads
! request lines "w h mi a b" from stdin, computes rows [a, b) of the escape
! time image, packs each row as raw little-endian uint16 and writes it to
! stdout (rebound to binary encoding). Hot words are JIT-compiled by Factor
! at load time; the pool is warmed at import so timing is steady-state.
!
! Author: optimized for M-series Macs
! License: MIT

USING: byte-arrays io io.encodings io.encodings.binary kernel
       math math.order math.parser sequences
       sequences.generalizations splitting typed ;
IN: factorbrot

! Escape-time iteration for one point c = (cr, ci). Caps at mi.
TYPED:: iterate ( cr: float ci: float mi: fixnum -- n: fixnum )
    0.0 :> zr!
    0.0 :> zi!
    0 :> it!
    [ it mi < [ zr zr * zi zi * + 4.0 <= ] [ f ] if ] [
        zr zr * :> zr2
        zi zi * :> zi2
        2.0 zr * zi * ci + :> nzi
        zr2 zi2 - cr + :> nzr
        nzr zr!
        nzi zi!
        it 1 + it!
    ] while
    it ;

! One pixel: cardioid/period-2 bulb early-out, else iterate.
TYPED:: pixel ( cr: float ci: float ci2: float mi: fixnum -- n: fixnum )
    cr 0.25 - :> crm
    crm crm * ci2 + :> qq
    qq qq crm + * 0.25 ci2 * <=
    cr 1.0 + cr 1.0 + * ci2 + 0.0625 <=
    or
    [ mi ] [ cr ci mi iterate ] if ;

! Compute one row r into a fresh little-endian uint16 byte-array.
TYPED:: do-row ( w: fixnum h: fixnum mi: fixnum r: fixnum -- bytes: byte-array )
    3.5 w 1 - >float / :> dx
    2.0 h 1 - >float / :> dy
    -1.0 r >float dy * + :> ci
    ci ci * :> ci2
    w 2 * <byte-array> :> buf
    w [ :> x
        -2.5 x >float dx * + :> cr
        cr ci ci2 mi pixel :> n
        n 255 bitand   x 2 *     buf set-nth
        n -8 shift 255 bitand   x 2 * 1 +  buf set-nth
    ] each-integer
    buf ;

! Handle one request: emit rows [a, b) back to back, then flush.
TYPED:: process ( w: fixnum h: fixnum mi: fixnum a: fixnum b: fixnum -- )
    b a - [ a + :> r  w h mi r do-row write ] each-integer
    flush ;

: main ( -- )
    binary encode-output
    [ readln dup ] [
        " " split harvest [ string>number ] map
        5 firstn process
    ] while drop ;

MAIN: main
