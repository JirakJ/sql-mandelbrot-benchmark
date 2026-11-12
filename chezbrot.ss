;; chezbrot.ss - persistent Mandelbrot worker for Chez Scheme (threaded build).
;; Protocol: "<width> <height> <max-iter>\n" on stdin ->
;; width*height*2 bytes of little-endian uint16 counts (row-major) on stdout.
;; Parallelism: persistent fork-thread pool, interleaved rows over the top
;; half (y-axis symmetry), bottom half mirrored via bytevector-copy!.

(optimize-level 3)
(generate-inspector-information #f)

(define nt
  (let ([args (command-line)])
    (or (and (pair? args) (pair? (cdr args)) (string->number (cadr args))) 8)))

(define bout (standard-output-port (buffer-mode none)))

;; Iterations survived; in-set pixels return maxit.
(define (escape cr ci maxit)
  (let loop ([zr 0.0] [zi 0.0] [it 0])
    (if (fx= it maxit)
        maxit
        (let ([zr2 (fl* zr zr)] [zi2 (fl* zi zi)])
          (if (fl> (fl+ zr2 zi2) 4.0)
              it
              (loop (fl+ (fl- zr2 zi2) cr)
                    (fl+ (fl* 2.0 (fl* zr zi)) ci)
                    (fx+ it 1)))))))

;; Compute rows t, t+nt, ... below top into bv.
(define (do-rows t w top maxit bv dx dy)
  (let row-loop ([r t])
    (when (fx< r top)
      (let* ([ci (fl+ -1.0 (fl* (fixnum->flonum r) dy))]
             [ci2 (fl* ci ci)]
             [qci2 (fl* 0.25 ci2)]
             [base (fx* r (fx* w 2))])
        (let x-loop ([x 0])
          (when (fx< x w)
            (let* ([cr (fl+ -2.5 (fl* (fixnum->flonum x) dx))]
                   [crm (fl- cr 0.25)]
                   [q (fl+ (fl* crm crm) ci2)]
                   [cp1 (fl+ cr 1.0)])
              (bytevector-u16-native-set!
               bv (fx+ base (fx* x 2))
               ;; cardioid + period-2 bulb early-out
               (if (or (fl<= (fl* q (fl+ q crm)) qci2)
                       (fl<= (fl+ (fl* cp1 cp1) ci2) 0.0625))
                   maxit
                   (escape cr ci maxit)))
              (x-loop (fx+ x 1))))))
      (row-loop (fx+ r nt)))))

;; Persistent thread pool: generation counter guarded by mutex/condition.
(define m (make-mutex))
(define cv-start (make-condition))
(define cv-done (make-condition))
(define gen 0)
(define ndone 0)
(define jw 0) (define jtop 0) (define jit 0)
(define jdx 0.0) (define jdy 0.0)
(define jbv (make-bytevector 0))

(define (worker t)
  (let loop ([mygen 0])
    (with-mutex m
      (let wait ()
        (when (fx= gen mygen)
          (condition-wait cv-start m)
          (wait))))
    (do-rows t jw jtop jit jbv jdx jdy)
    (with-mutex m
      (set! ndone (fx+ ndone 1))
      (when (fx= ndone nt) (condition-signal cv-done)))
    (loop (fx+ mygen 1))))

(do ([t 0 (fx+ t 1)]) ((fx= t nt))
  (fork-thread (lambda () (worker t))))

(define cached-bv (make-bytevector 0))

(define (frame w h maxit)
  (let* ([top (fxdiv (fx+ h 1) 2)]
         [nbytes (fx* w (fx* h 2))]
         [bv (if (fx= nbytes (bytevector-length cached-bv))
                 cached-bv
                 (begin (set! cached-bv (make-bytevector nbytes)) cached-bv))]
         [dx (fl/ 3.5 (fixnum->flonum (fx- w 1)))]
         [dy (fl/ 2.0 (fixnum->flonum (fx- h 1)))])
    (with-mutex m
      (set! jw w) (set! jtop top) (set! jit maxit)
      (set! jdx dx) (set! jdy dy) (set! jbv bv)
      (set! ndone 0)
      (set! gen (fx+ gen 1))
      (condition-broadcast cv-start))
    (with-mutex m
      (let wait ()
        (unless (fx= ndone nt)
          (condition-wait cv-done m)
          (wait))))
    ;; Mirror bottom half; odd middle row (dst = r) stays as computed.
    (let ([rowb (fx* w 2)])
      (do ([r 0 (fx+ r 1)]) ((fx= r top))
        (let ([dst (fx- (fx- h 1) r)])
          (when (fx> dst r)
            (bytevector-copy! bv (fx* r rowb) bv (fx* dst rowb) rowb)))))
    (put-bytevector bout bv 0 nbytes)
    (flush-output-port bout)))

(let loop ()
  (let ([line (get-line (current-input-port))])
    (unless (eof-object? line)
      (let* ([p (open-input-string line)]
             [w (read p)] [h (read p)] [it (read p)])  ; force eval order
        (frame w h it))
      (loop))))
