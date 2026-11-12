;; ClojureBrot - persistent Mandelbrot worker.
;; Protocol: "<width> <height> <max_iter>\n" on stdin ->
;; width*height uint16 little-endian pixels, row-major, on stdout.
;; Parallel over top-half rows (fixed thread pool, atomic row counter),
;; bottom half mirrored via System/arraycopy.

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(import '(java.io BufferedOutputStream BufferedReader FileDescriptor
                  FileOutputStream InputStreamReader OutputStream)
        '(java.nio ByteBuffer ByteOrder)
        '(java.util.concurrent Callable ExecutorService Executors Future)
        '(java.util.concurrent.atomic AtomicInteger))

(def ^:private ^ExecutorService pool
  (Executors/newFixedThreadPool (.availableProcessors (Runtime/getRuntime))))

(defn- mandel-row
  "Fill one row of escape counts (iterations survived) into buf at off."
  [^shorts buf off w dx ci max-iter]
  (let [off (long off) w (long w) dx (double dx)
        ci (double ci) max-iter (long max-iter)
        ci2 (* ci ci)
        lim (* 0.25 ci2)]
    (loop [x 0]
      (when (< x w)
        (let [cr  (+ -2.5 (* (double x) dx))
              crm (- cr 0.25)
              q   (+ (* crm crm) ci2)
              cr1 (+ cr 1.0)
              it  (if (or (<= (* q (+ q crm)) lim)            ; cardioid
                          (<= (+ (* cr1 cr1) ci2) 0.0625))    ; period-2 bulb
                    max-iter
                    (loop [zr 0.0 zi 0.0 it 0]
                      (if (< it max-iter)
                        (let [zr2 (* zr zr)
                              zi2 (* zi zi)]
                          (if (> (+ zr2 zi2) 4.0)
                            it
                            (recur (+ (- zr2 zi2) cr)
                                   (+ (* 2.0 zr zi) ci)
                                   (inc it))))
                        it)))]
          (aset buf (+ off x) (unchecked-short it))
          (recur (inc x)))))))

(defn- compute-frame
  "Compute a full frame into a fresh row-major short array."
  [w h max-iter]
  (let [w (long w) h (long h) max-iter (long max-iter)
        buf (short-array (* w h))
        dx (if (> w 1) (/ 3.5 (double (dec w))) 0.0)
        dy (if (> h 1) (/ 2.0 (double (dec h))) 0.0)
        top (quot (inc h) 2)
        next-row (AtomicInteger. 0)
        worker (fn []
                 (loop []
                   (let [r (long (.getAndIncrement next-row))]
                     (when (< r top)
                       (mandel-row buf (* r w) w dx
                                   (+ -1.0 (* (double r) dy)) max-iter)
                       (recur)))))
        futs (mapv (fn [_] (.submit pool ^Callable worker))
                   (range (.availableProcessors (Runtime/getRuntime))))]
    (run! (fn [^Future f] (.get f)) futs)
    (loop [r 0]                                   ; mirror; odd middle row stays
      (when (< r top)
        (let [mr (- (dec h) r)]
          (when (> mr r)
            (System/arraycopy buf (* r w) buf (* mr w) w)))
        (recur (inc r))))
    buf))

(defn- write-frame [^OutputStream out ^shorts buf]
  (let [n (alength buf)
        bb (doto (ByteBuffer/allocate (* 2 n))
             (.order ByteOrder/LITTLE_ENDIAN))]
    (.put (.asShortBuffer bb) buf)
    (.write out (.array bb) 0 (* 2 n))
    (.flush out)))

(let [in  (BufferedReader. (InputStreamReader. System/in))
      out (BufferedOutputStream. (FileOutputStream. FileDescriptor/out) 65536)]
  (loop []
    (when-let [line (.readLine in)]
      (let [parts (.split (.trim ^String line) "\\s+")]
        (when (>= (alength parts) 3)
          (write-frame out (compute-frame (Long/parseLong (aget parts 0))
                                          (Long/parseLong (aget parts 1))
                                          (Long/parseLong (aget parts 2))))))
      (recur)))
  (.shutdown pool))
