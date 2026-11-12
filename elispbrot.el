;;; elispbrot.el --- Mandelbrot band worker for batch Emacs -*- lexical-binding: t; -*-

;; Protocol: read "w h maxit r0 r1\n" lines from stdin, compute rows [r0,r1)
;; of the top half, write them as raw uint16 LE bytes to stdout.  Loops to EOF.
;; Logs go to stderr only.  Hot function is native-compiled when the Emacs
;; build supports it, byte-compiled otherwise.

(require 'cl-lib)
(cl-declaim (optimize (speed 3) (safety 0)))

;; Millions of boxed floats per frame: keep GC out of the hot loop.
(setq gc-cons-threshold (* 1024 1024 1024)
      gc-cons-percentage 1.0)

(defun elispbrot--compute (w h maxit r0 r1)
  "Compute rows [R0,R1) of a WxH frame; return uint16 LE unibyte string."
  (let* ((dx (/ 3.5 (float (1- w))))
         (dy (/ 2.0 (float (1- h))))
         (crv (make-vector w 0.0))   ; cr per column
         (crmv (make-vector w 0.0))  ; cr - 0.25
         (crm2v (make-vector w 0.0)) ; (cr - 0.25)^2
         (cp2v (make-vector w 0.0))  ; (cr + 1)^2
         (buf (make-string (* (- r1 r0) w 2) 0))
         (idx 0))
    (let ((x 0) cr crm cp)
      (while (< x w)
        (setq cr (+ -2.5 (* x dx))
              crm (- cr 0.25)
              cp (+ cr 1.0))
        (aset crv x cr)
        (aset crmv x crm)
        (aset crm2v x (* crm crm))
        (aset cp2v x (* cp cp))
        (setq x (1+ x))))
    (let ((row r0))
      (while (< row r1)
        (let* ((ci (+ -1.0 (* row dy)))
               (ci2 (* ci ci))
               (card-rhs (* 0.25 ci2))
               (x 0))
          (while (< x w)
            (let ((q (+ (aref crm2v x) ci2))
                  (res 0))
              (if (or (<= (* q (+ q (aref crmv x))) card-rhs)
                      (<= (+ (aref cp2v x) ci2) 0.0625))
                  (setq res maxit)       ; cardioid / period-2 bulb: in set
                (let ((cr (aref crv x))
                      (zr 0.0) (zi 0.0) (zr2 0.0) (zi2 0.0)
                      (it 0))
                  (while (and (< it maxit)
                              (<= (+ (setq zr2 (* zr zr))
                                     (setq zi2 (* zi zi)))
                                  4.0))
                    (setq zi (+ (* 2.0 zr zi) ci)
                          zr (+ (- zr2 zi2) cr)
                          it (1+ it)))
                  (setq res it)))
              (aset buf idx (logand res 255))
              (aset buf (1+ idx) (ash res -8))
              (setq idx (+ idx 2)
                    x (1+ x)))))
        (setq row (1+ row))))
    buf))

(if (and (fboundp 'native-comp-available-p) (native-comp-available-p))
    (progn
      (setq native-comp-speed 3)
      (native-compile #'elispbrot--compute)
      (message "elispbrot: native-compiled (speed 3)"))
  (byte-compile #'elispbrot--compute)
  (message "elispbrot: byte-compiled (native comp unavailable)"))

(defun elispbrot--main ()
  (set-binary-mode 'stdout t)
  (let (line parts)
    ;; read-from-minibuffer reads stdin lines in batch; error means EOF.
    (while (setq line (condition-case nil (read-from-minibuffer "") (error nil)))
      (setq parts (mapcar #'string-to-number (split-string line)))
      (when (>= (length parts) 5)
        (send-string-to-terminal
         (elispbrot--compute (nth 0 parts) (nth 1 parts) (nth 2 parts)
                             (nth 3 parts) (nth 4 parts)))))))

(elispbrot--main)

;;; elispbrot.el ends here
