;;; SbclBrot - persistent SBCL Mandelbrot worker.
;;; Protocol: stdin text line "<width> <height> <max-iter>\n" ->
;;; stdout width*height*2 bytes, uint16 little-endian, row-major.
;;; Multithreaded across rows (atomic row counter), y-axis mirror.

(declaim (optimize (speed 3) (safety 0) (debug 0)))

(defpackage #:sbclbrot
  (:use #:cl))
(in-package #:sbclbrot)

(deftype u16vec () '(simple-array (unsigned-byte 16) (*)))
(deftype u8vec () '(simple-array (unsigned-byte 8) (*)))

(sb-alien:define-alien-routine ("memcpy" %memcpy) sb-sys:system-area-pointer
  (dst sb-sys:system-area-pointer)
  (src sb-sys:system-area-pointer)
  (n sb-alien:unsigned-long))

(declaim (inline mandel-pixel))
(defun mandel-pixel (cr ci ci2 max-iter)
  (declare (type double-float cr ci ci2)
           (type (unsigned-byte 16) max-iter))
  ;; Cardioid / period-2 bulb early-out.
  (let* ((crm (- cr 0.25d0))
         (cp1 (+ cr 1d0))
         (q (+ (* crm crm) ci2)))
    (declare (type double-float crm cp1 q))
    (when (or (<= (* q (+ q crm)) (* 0.25d0 ci2))
              (<= (+ (* cp1 cp1) ci2) 0.0625d0))
      (return-from mandel-pixel max-iter)))
  (let ((zr 0d0) (zi 0d0) (it 0))
    (declare (type double-float zr zi)
             (type (unsigned-byte 16) it))
    (loop while (< it max-iter)
          do (let ((zr2 (* zr zr)) (zi2 (* zi zi)))
               (declare (type double-float zr2 zi2))
               (when (> (+ zr2 zi2) 4d0) (return))
               (setf zi (+ (* 2d0 zr zi) ci)
                     zr (+ (- zr2 zi2) cr))
               (incf it)))
    it))

(defun worker (out w h half max-iter dx dy counter)
  (declare (type u16vec out)
           (type fixnum w h half)
           (type (unsigned-byte 16) max-iter)
           (type double-float dx dy)
           (type (simple-array sb-ext:word (1)) counter))
  (loop
    (let ((y (the fixnum (sb-ext:atomic-incf (aref counter 0)))))
      (when (>= y half) (return))
      (let* ((ci (+ -1d0 (* (coerce y 'double-float) dy)))
             (ci2 (* ci ci))
             (base (* y w)))
        (declare (type double-float ci ci2) (type fixnum base))
        (loop for x of-type fixnum below w
              do (setf (aref out (+ base x))
                       (mandel-pixel (+ -2.5d0 (* (coerce x 'double-float) dx))
                                     ci ci2 max-iter)))
        (let ((my (- h 1 y)))
          (declare (type fixnum my))
          (unless (= my y)
            (replace out out :start1 (* my w) :end1 (+ (* my w) w)
                             :start2 base)))))))

(defun compute (w h max-iter nthreads)
  (declare (type fixnum w h nthreads)
           (type (unsigned-byte 16) max-iter))
  (let* ((out (make-array (* w h) :element-type '(unsigned-byte 16)))
         (half (ceiling h 2))
         (dx (if (> w 1) (/ 3.5d0 (coerce (1- w) 'double-float)) 0d0))
         (dy (if (> h 1) (/ 2d0 (coerce (1- h) 'double-float)) 0d0))
         (counter (make-array 1 :element-type 'sb-ext:word :initial-element 0))
         (nt (max 1 (min nthreads half))))
    (declare (type fixnum half nt))
    (flet ((work () (worker out w h half max-iter dx dy counter)))
      (if (= nt 1)
          (work)
          (let ((threads (loop repeat (1- nt)
                               collect (sb-thread:make-thread #'work))))
            (work)                      ; main thread participates
            (mapc #'sb-thread:join-thread threads))))
    out))

(defun write-frame (out bin)
  (declare (type u16vec out))
  (let* ((nbytes (* (length out) 2))
         (bytes (make-array nbytes :element-type '(unsigned-byte 8))))
    (declare (type u8vec bytes))
    ;; arm64 is little-endian: raw copy of the u16 payload is already LE.
    (sb-sys:with-pinned-objects (out bytes)
      (%memcpy (sb-sys:vector-sap bytes) (sb-sys:vector-sap out) nbytes))
    (write-sequence bytes bin)
    (force-output bin)))

(defun main ()
  (let ((bin (sb-sys:make-fd-stream 1 :output t
                                       :element-type '(unsigned-byte 8)
                                       :buffering :full))
        (in *standard-input*)
        (nthreads (max 1 (or (ignore-errors
                               (parse-integer
                                (sb-ext:posix-getenv "SBCLBROT_THREADS")))
                             8))))
    ;; Swallow any stray prints so fd 1 stays pure binary.
    (setf *standard-output* (make-broadcast-stream))
    (loop for line = (read-line in nil nil)
          while line
          do (multiple-value-bind (w p1) (parse-integer line :junk-allowed t)
               (multiple-value-bind (h p2)
                   (parse-integer line :start p1 :junk-allowed t)
                 (let ((iters (parse-integer line :start p2 :junk-allowed t)))
                   (write-frame (compute w h iters nthreads) bin)))))))

(main)
