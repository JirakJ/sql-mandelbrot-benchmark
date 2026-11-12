;;; guilebrot.scm - GNU Guile 3 Mandelbrot worker (in-process thread pool).
;;;
;;; Runs as ONE persistent process. Reads request lines "w h mi" from stdin,
;;; computes the top half of the image across a pool of native POSIX threads
;;; (call-with-new-thread) into a single bytevector (uint16 little-endian),
;;; mirrors the bottom half (y-axis symmetry), and writes the whole frame as
;;; raw bytes to stdout, then force-output. Loops until EOF.
;;;
;;; License: MIT

(use-modules (ice-9 threads)
             (ice-9 rdelim)
             (rnrs bytevectors)
             (rnrs io ports))

;; Compute rows [r0, r1) of the image into bv (row-major, uint16 LE).
(define (compute-band bv w mi dx dy r0 r1)
  (let loop-r ((r r0))
    (when (< r r1)
      (let* ((ci  (+ -1.0 (* r dy)))
             (ci2 (* ci ci))
             (row-base (* 2 (* r w))))
        (let loop-x ((x 0))
          (when (< x w)
            (let* ((cr  (+ -2.5 (* x dx)))
                   (crm (- cr 0.25))
                   (q   (+ (* crm crm) ci2))
                   (crp (+ cr 1.0))
                   (val
                    (if (or (<= (* q (+ q crm)) (* 0.25 ci2))
                            (<= (+ (* crp crp) ci2) 0.0625))
                        mi
                        (let iter ((zr 0.0) (zi 0.0) (it 0))
                          (if (>= it mi)
                              it
                              (let ((zr2 (* zr zr))
                                    (zi2 (* zi zi)))
                                (if (> (+ zr2 zi2) 4.0)
                                    it
                                    (iter (+ (- zr2 zi2) cr)
                                          (+ (* 2.0 zr zi) ci)
                                          (+ it 1)))))))))
              (bytevector-u16-set! bv (+ row-base (* 2 x)) val (endianness little))
              (loop-x (+ x 1)))))
        (loop-r (+ r 1))))))

(define nthreads (max 1 (min 14 (current-processor-count))))

(define (compute-frame w h mi)
  (let* ((dx  (/ 3.5 (- w 1)))
         (dy  (/ 2.0 (- h 1)))
         (top (quotient (+ h 1) 2))
         (bv  (make-bytevector (* 2 (* w h))))
         (base (quotient top nthreads))
         (rem  (remainder top nthreads)))
    ;; Split the top half into contiguous row bands, one native thread each.
    (let build ((i 0) (r0 0) (threads '()))
      (if (>= i nthreads)
          (for-each join-thread threads)
          (let* ((n  (+ base (if (< i rem) 1 0)))
                 (r1 (+ r0 n)))
            (if (> n 0)
                (build (+ i 1) r1
                       (cons (call-with-new-thread
                              (lambda () (compute-band bv w mi dx dy r0 r1)))
                             threads))
                (build (+ i 1) r1 threads)))))
    ;; Mirror the bottom half: row (h-1-r) = row r.
    (let ((rowbytes (* 2 w)))
      (let mloop ((r 0))
        (when (< r top)
          (let ((m (- (- h 1) r)))
            (when (>= m top)
              (bytevector-copy! bv (* r rowbytes) bv (* m rowbytes) rowbytes)))
          (mloop (+ r 1)))))
    bv))

;; Main loop: one request line "w h mi" -> one raw frame written back.
(let ((out (current-output-port)))
  (let main-loop ()
    (let ((line (read-line)))
      (unless (eof-object? line)
        (call-with-input-string line
          (lambda (p)
            (let* ((w  (read p))
                   (h  (read p))
                   (mi (read p)))
              (put-bytevector out (compute-frame w h mi))
              (force-output out))))
        (main-loop)))))
