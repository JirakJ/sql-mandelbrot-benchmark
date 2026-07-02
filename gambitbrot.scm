; gambitbrot.scm - Gambit Scheme Mandelbrot worker.
;
; Gambit threads are green (not parallel), so the Python wrapper spawns a pool
; of these compiled worker processes. Each worker reads "w h mi a b" request
; lines from stdin (compute rows [a, b) of the top half) and writes the band
; back to stdout as raw uint16 little-endian bytes, row-major.
;
; Compile: gsc -exe -o gambitbrot_bin gambitbrot.scm

(declare
  (standard-bindings)
  (extended-bindings)
  (not safe)
  (fixnum)
  (flonum))

;; Escape-time iteration for a single point. Returns iteration count (<= mi).
(define (mandel-iter cr ci mi)
  (let loop ((zr 0.0) (zi 0.0) (it 0))
    (if (fx>= it mi)
        it
        (let ((zr2 (fl* zr zr))
              (zi2 (fl* zi zi)))
          (if (fl> (fl+ zr2 zi2) 4.0)
              it
              (let ((nzi (fl+ (fl* 2.0 (fl* zr zi)) ci))
                    (nzr (fl+ (fl- zr2 zi2) cr)))
                (loop nzr nzi (fx+ it 1))))))))

;; Compute rows [a, b) into a fresh byte buffer, then emit it in one write.
(define (compute w h mi a b out)
  (let* ((dx    (fl/ 3.5 (exact->inexact (fx- w 1))))
         (dy    (fl/ 2.0 (exact->inexact (fx- h 1))))
         (nrows (fx- b a))
         (buf   (make-u8vector (fx* (fx* nrows w) 2) 0)))
    (let rloop ((r a) (off 0))
      (if (fx>= r b)
          (write-subu8vector buf 0 (u8vector-length buf) out)
          (let* ((ci  (fl+ -1.0 (fl* (exact->inexact r) dy)))
                 (ci2 (fl* ci ci)))
            (let cloop ((x 0) (o off))
              (if (fx>= x w)
                  (rloop (fx+ r 1) o)
                  (let* ((cr  (fl+ -2.5 (fl* (exact->inexact x) dx)))
                         (crm (fl- cr 0.25))
                         (q   (fl+ (fl* crm crm) ci2))
                         (cr1 (fl+ cr 1.0))
                         (val (if (or (fl<= (fl* q (fl+ q crm)) (fl* 0.25 ci2))
                                      (fl<= (fl+ (fl* cr1 cr1) ci2) 0.0625))
                                  mi
                                  (mandel-iter cr ci mi))))
                    (u8vector-set! buf o (fxand val 255))
                    (u8vector-set! buf (fx+ o 1)
                                   (fxand (fxarithmetic-shift-right val 8) 255))
                    (cloop (fx+ x 1) (fx+ o 2))))))))))

;; Parse "w h mi a b" and dispatch.
(define (process line out)
  (let* ((p  (open-input-string line))
         (w  (read p))
         (h  (read p))
         (mi (read p))
         (a  (read p))
         (b  (read p)))
    (compute w h mi a b out)))

(define (main)
  (let ((in  (current-input-port))
        (out (current-output-port)))
    (let loop ()
      (let ((line (read-line in)))
        (if (eof-object? line)
            #t
            (begin
              (process line out)
              (force-output out)
              (loop)))))))

(main)
