\ Forth Mandelbrot worker (gforth / gforth-fast).
\ Protocol: read "w h maxit r0 r1\n" from stdin, compute rows [r0,r1)
\ of the top half, write (r1-r0)*w uint16 LE bytes to stdout, flush.
\ Loops until EOF. Spawned as a pool by forthbrot.py.

256 constant lbuf#
create linebuf lbuf# allot

variable vw    variable vh    variable vit
variable vr0   variable vr1
variable band-buf   0 band-buf !
variable band-cap   0 band-cap !

: ensure-buf ( bytes -- )
  dup band-cap @ > if
    band-buf @ ?dup if free throw then
    dup allocate throw band-buf !
    band-cap !
  else drop then ;

\ Escape-time kernel with cardioid / period-2 bulb early-outs.
\ Returns iterations survived (max = maxit for in-set points).
: iters { maxit f: cr f: ci -- n }
  ci ci f* { f: ci2 }
  cr 0.25e f- { f: crm }
  crm crm f* ci2 f+ { f: q }
  q q crm f+ f* 0.25e ci2 f* f<= if maxit exit then
  cr 1e f+ fdup f* ci2 f+ 0.0625e f<= if maxit exit then
  0e 0e ( F: zr zi )
  0 ( it )
  begin
    dup maxit <
  while
    fover fdup f* fover fdup f*        ( F: zr zi zr2 zi2 )
    fover fover f+ 4e f> if
      fdrop fdrop fdrop fdrop exit
    then
    f- cr f+                           ( F: zr zi nzr )
    frot frot f* 2e f* ci f+           ( F: nzr nzi )
    1+
  repeat
  fdrop fdrop ;

\ One image row into addr as uint16 LE.
: do-row { w maxit addr f: ci f: dx -- }
  w 0 ?do
    addr i 2* +
    maxit  i s>d d>f dx f* -2.5e f+  ci  iters
    dup 255 and 2 pick c!
    8 rshift swap 1+ c!
  loop ;

: compute-band ( -- )
  vr1 @ vr0 @ ?do
    vw @ vit @  band-buf @ i vr0 @ - vw @ * 2* +
    i s>d d>f  2e vh @ 1- s>d d>f f/ f*  -1e f+   ( F: ci )
    3.5e vw @ 1- s>d d>f f/                       ( F: ci dx )
    do-row
  loop ;

: skip-bl ( addr u -- addr' u' )
  begin dup 0> if over c@ bl = else false then while 1 /string repeat ;

: grab-num ( addr u -- addr' u' n )
  skip-bl 0. 2swap >number 2swap d>s ;

: parse-req ( addr u -- )
  grab-num vw ! grab-num vh ! grab-num vit !
  grab-num vr0 ! grab-num vr1 ! 2drop ;

: main ( -- )
  begin
    linebuf lbuf# stdin read-line throw   ( u flag )
  while
    linebuf swap parse-req
    vr1 @ vr0 @ - vw @ * 2* ensure-buf
    compute-band
    band-buf @  vr1 @ vr0 @ - vw @ * 2*  stdout write-file throw
    stdout flush-file throw
  repeat
  drop ;

main bye
