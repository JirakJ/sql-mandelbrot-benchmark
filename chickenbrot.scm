;;; ChickenBrot - CHICKEN Scheme Mandelbrot worker (persistent process, Pattern B)
;;;
;;; CHICKEN has only green threads, so parallelism lives in the Python wrapper
;;; which spawns a pool of these worker binaries. Each worker reads request
;;; lines "w h mi a b" from stdin and writes the top-half row band [a,b) as
;;; raw little-endian uint16 rows to stdout. Compiled Scheme -> C -> native.
;;;
;;; Author: optimized for M-series Macs
;;; License: MIT

(declare (usual-integrations))

(import (chicken base)
        (chicken io)
        (chicken flonum)
        (chicken fixnum)
        (chicken port)
        (chicken string)
        srfi-4)

;; Compute one row band and emit it as raw uint16 LE bytes.
(define (compute-band w h mi a b)
  (let* ((dx (fp/ 3.5 (exact->inexact (fx- w 1))))
         (dy (fp/ 2.0 (exact->inexact (fx- h 1))))
         (nrows (fx- b a))
         (buf (make-u8vector (fx* (fx* nrows w) 2) 0)))
    (let row-loop ((r a) (o 0))
      (if (fx>= r b)
          (write-u8vector buf)
          (let* ((ci (fp+ -1.0 (fp* (exact->inexact r) dy)))
                 (ci2 (fp* ci ci)))
            (let col-loop ((x 0) (o o))
              (if (fx>= x w)
                  (row-loop (fx+ r 1) o)
                  (let* ((cr (fp+ -2.5 (fp* (exact->inexact x) dx)))
                         (crm (fp- cr 0.25))
                         (qq (fp+ (fp* crm crm) ci2))
                         (cr1 (fp+ cr 1.0)))
                    (let ((v
                           (if (or (fp<= (fp* qq (fp+ qq crm)) (fp* 0.25 ci2))
                                   (fp<= (fp+ (fp* cr1 cr1) ci2) 0.0625))
                               mi
                               (let esc ((zr 0.0) (zi 0.0) (it 0))
                                 (if (fx>= it mi)
                                     it
                                     (let ((zr2 (fp* zr zr))
                                           (zi2 (fp* zi zi)))
                                       (if (fp> (fp+ zr2 zi2) 4.0)
                                           it
                                           (esc (fp+ (fp- zr2 zi2) cr)
                                                (fp+ (fp* 2.0 (fp* zr zi)) ci)
                                                (fx+ it 1)))))))))
                      (u8vector-set! buf o (fxand v #xff))
                      (u8vector-set! buf (fx+ o 1) (fxand (fxshr v 8) #xff))
                      (col-loop (fx+ x 1) (fx+ o 2)))))))))))

;; Main loop: one request line -> one raw band, flushed.
(let loop ()
  (let ((line (read-line)))
    (unless (eof-object? line)
      (let ((parts (map string->number (string-split line " "))))
        (compute-band (car parts) (cadr parts) (caddr parts)
                      (cadddr parts) (car (cddddr parts)))
        (flush-output))
      (loop))))
