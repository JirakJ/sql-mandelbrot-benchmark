#lang racket/base
;; racketbrot.rkt - persistent Mandelbrot worker for Racket CS.
;; Protocol: "<width> <height> <max-iter>\n" on stdin ->
;; width*height*2 bytes of little-endian uint16 counts (row-major) on stdout.
;; Parallelism: pool of (processor-count) places created once at startup;
;; each computes an interleaved row set over the top half (y-axis symmetry),
;; the main place scatters bands, mirrors the bottom half, writes the frame.

(require racket/place
         racket/future
         racket/string
         racket/unsafe/ops
         syntax/location)

(provide place-main)

(define this-mod (quote-module-path))

;; Iterations survived; in-set pixels return maxit.
(define (escape cr ci maxit)
  (let loop ([zr 0.0] [zi 0.0] [it 0])
    (if (unsafe-fx>= it maxit)
        maxit
        (let ([zr2 (unsafe-fl* zr zr)] [zi2 (unsafe-fl* zi zi)])
          (if (unsafe-fl> (unsafe-fl+ zr2 zi2) 4.0)
              it
              (loop (unsafe-fl+ (unsafe-fl- zr2 zi2) cr)
                    (unsafe-fl+ (unsafe-fl* 2.0 (unsafe-fl* zr zi)) ci)
                    (unsafe-fx+ it 1)))))))

;; Rows t, t+np, ... below top, packed sequentially into bv as uint16 LE.
(define (do-rows bv w top maxit dx dy t np)
  (define roww (unsafe-fx* w 2))
  (let row-loop ([r t] [base 0])
    (when (unsafe-fx< r top)
      (let* ([ci (unsafe-fl+ -1.0 (unsafe-fl* (unsafe-fx->fl r) dy))]
             [ci2 (unsafe-fl* ci ci)]
             [qci2 (unsafe-fl* 0.25 ci2)])
        (let x-loop ([x 0] [off base])
          (when (unsafe-fx< x w)
            (let* ([cr (unsafe-fl+ -2.5 (unsafe-fl* (unsafe-fx->fl x) dx))]
                   [crm (unsafe-fl- cr 0.25)]
                   [q (unsafe-fl+ (unsafe-fl* crm crm) ci2)]
                   [cp1 (unsafe-fl+ cr 1.0)]
                   ;; cardioid + period-2 bulb early-out
                   [v (if (or (unsafe-fl<= (unsafe-fl* q (unsafe-fl+ q crm)) qci2)
                              (unsafe-fl<= (unsafe-fl+ (unsafe-fl* cp1 cp1) ci2) 0.0625))
                          maxit
                          (escape cr ci maxit))])
              (unsafe-bytes-set! bv off (unsafe-fxand v 255))
              (unsafe-bytes-set! bv (unsafe-fx+ off 1) (unsafe-fxrshift v 8))
              (x-loop (unsafe-fx+ x 1) (unsafe-fx+ off 2))))))
      (row-loop (unsafe-fx+ r np) (unsafe-fx+ base roww)))))

(define (place-main ch)
  (define cfg (place-channel-get ch))       ; (t np)
  (define t (car cfg))
  (define np (cadr cfg))
  (let loop ()
    (define job (place-channel-get ch))     ; (w h maxit)
    (define w (car job))
    (define h (cadr job))
    (define maxit (caddr job))
    (define top (quotient (+ h 1) 2))
    (define nrows (if (< t top) (quotient (+ (- top t) np -1) np) 0))
    (define bv (make-bytes (* nrows w 2)))
    (when (> nrows 0)
      (do-rows bv w top maxit
               (/ 3.5 (exact->inexact (- w 1)))
               (/ 2.0 (exact->inexact (- h 1)))
               t np))
    (place-channel-put ch bv)
    (loop)))

(module+ main
  (define np (processor-count))
  (define places
    (for/list ([t (in-range np)])
      (define ch (dynamic-place this-mod 'place-main))
      (place-channel-put ch (list t np))
      ch))
  (define in (current-input-port))
  (define out (current-output-port))

  (define (frame w h maxit)
    (define top (quotient (+ h 1) 2))
    (define rowb (* w 2))
    (define fbv (make-bytes (* h rowb)))
    (for ([ch (in-list places)])
      (place-channel-put ch (list w h maxit)))
    (for ([ch (in-list places)] [t (in-naturals)])
      (define bv (place-channel-get ch))
      (let scatter ([r t] [src 0])
        (when (< r top)
          (bytes-copy! fbv (* r rowb) bv src (+ src rowb))
          (scatter (+ r np) (+ src rowb)))))
    ;; Mirror bottom half; odd middle row (dst = r) stays as computed.
    (for ([r (in-range top)])
      (define dst (- h 1 r))
      (when (> dst r)
        (bytes-copy! fbv (* dst rowb) fbv (* r rowb) (* (+ r 1) rowb))))
    (void (write-bytes fbv out))
    (flush-output out))

  (let loop ()
    (define line (read-line in 'any))
    (unless (eof-object? line)
      (define nums (map string->number (string-split line)))
      (frame (car nums) (cadr nums) (caddr nums))
      (loop))))
