# Janet Mandelbrot worker: reads "w h iters\n" from stdin, computes the top
# half on N OS threads (ev/spawn-thread + ev/thread-chan), mirrors the bottom
# half, writes w*h*2 raw uint16 LE bytes to stdout. Loops until EOF.

(def nw (max 1 (or (scan-number (get (dyn :args) 1 "8")) 8)))

(defn compute-band
  "Rows [r0,r1) of the top half as a uint16 LE buffer."
  [w h iters r0 r1]
  (def dx (/ 3.5 (- w 1)))
  (def dy (/ 2.0 (- h 1)))
  (def buf (buffer/new (* (- r1 r0) w 2)))
  (var row r0)
  (while (< row r1)
    (def ci (+ -1.0 (* row dy)))
    (def ci2 (* ci ci))
    (var x 0)
    (while (< x w)
      (def cr (+ -2.5 (* x dx)))
      (def crm (- cr 0.25))
      (def q (+ (* crm crm) ci2))
      (def cp1 (+ cr 1.0))
      (var it 0)
      (if (or (<= (* q (+ q crm)) (* 0.25 ci2))
              (<= (+ (* cp1 cp1) ci2) 0.0625))
        (set it iters) # cardioid / period-2 bulb: in set
        (do
          (var zr 0.0)
          (var zi 0.0)
          (while (< it iters)
            (def zr2 (* zr zr))
            (def zi2 (* zi zi))
            (if (> (+ zr2 zi2) 4.0) (break))
            (def nzr (+ (- zr2 zi2) cr))
            (set zi (+ (* 2.0 (* zr zi)) ci))
            (set zr nzr)
            (++ it))))
      (buffer/push-uint16 buf :le it)
      (++ x))
    (++ row))
  buf)

(def out-chan (ev/thread-chan 32))
(def in-chans (seq [_ :range [0 nw]] (ev/thread-chan 4)))

(each ch in-chans
  (ev/spawn-thread
    (forever
      (def [idx w h iters r0 r1] (ev/take ch))
      (ev/give out-chan [idx (compute-band w h iters r0 r1)]))))

(defn frame [w h iters]
  (def top (div (+ h 1) 2))
  (def base (div top nw))
  (def rem (mod top nw))
  (var nb 0)
  (var r0 0)
  (loop [i :range [0 nw]]
    (def n (+ base (if (< i rem) 1 0)))
    (when (> n 0)
      (ev/give (in-chans i) [nb w h iters r0 (+ r0 n)])
      (set r0 (+ r0 n))
      (++ nb)))
  (def bands (array/new-filled nb nil))
  (repeat nb
    (def [idx buf] (ev/take out-chan))
    (put bands idx buf))
  (def out (buffer/new (* w h 2)))
  (each b bands (buffer/push out b))
  # Mirror bottom half; odd middle row stays as computed.
  (def rw (* w 2))
  (var r top)
  (while (< r h)
    (def s (- h 1 r))
    (buffer/push out (buffer/slice out (* s rw) (* (+ s 1) rw)))
    (++ r))
  (file/write stdout out)
  (file/flush stdout))

(while true
  (def line (file/read stdin :line))
  (if (nil? line) (break))
  (def parts (string/split " " (string/trim (string line))))
  (frame (scan-number (parts 0)) (scan-number (parts 1)) (scan-number (parts 2))))

# Worker threads block in ev/take forever; force exit at EOF.
(os/exit 0)
