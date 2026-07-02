;; WasmBrot - hand-written WebAssembly SIMD128 Mandelbrot kernel.
;; mandel_rows(w, h, max_iter, r0, step, nrows, out) fills image rows
;; r0, r0+step, r0+2*step, ... (nrows of them, strided for load balance)
;; at their natural row offsets (out + r*w*2) as uint16 LE. All worker
;; instances import one shared memory, so rows land pre-assembled.
(module
  (import "env" "mem" (memory 32 16384 shared))

  ;; Scalar f64 pixel (tail when width % 4 != 0). Returns iterations survived.
  (func $px (param $cr f64) (param $ci f64) (param $maxit i32) (result i32)
    (local $zr f64) (local $zi f64) (local $zr2 f64) (local $zi2 f64)
    (local $it i32) (local $crm f64) (local $q f64) (local $ci2 f64)
    (local.set $ci2 (f64.mul (local.get $ci) (local.get $ci)))
    ;; cardioid
    (local.set $crm (f64.sub (local.get $cr) (f64.const 0.25)))
    (local.set $q (f64.add (f64.mul (local.get $crm) (local.get $crm)) (local.get $ci2)))
    (if (f64.le (f64.mul (local.get $q) (f64.add (local.get $q) (local.get $crm)))
                (f64.mul (f64.const 0.25) (local.get $ci2)))
      (then (return (local.get $maxit))))
    ;; period-2 bulb
    (if (f64.le (f64.add (f64.mul (f64.add (local.get $cr) (f64.const 1))
                                  (f64.add (local.get $cr) (f64.const 1)))
                         (local.get $ci2))
                (f64.const 0.0625))
      (then (return (local.get $maxit))))
    (block $done
      (loop $l
        (br_if $done (i32.ge_s (local.get $it) (local.get $maxit)))
        (local.set $zr2 (f64.mul (local.get $zr) (local.get $zr)))
        (local.set $zi2 (f64.mul (local.get $zi) (local.get $zi)))
        (br_if $done (f64.gt (f64.add (local.get $zr2) (local.get $zi2)) (f64.const 4)))
        (local.set $zi (f64.add (f64.mul (f64.mul (f64.const 2) (local.get $zr)) (local.get $zi))
                                (local.get $ci)))
        (local.set $zr (f64.add (f64.sub (local.get $zr2) (local.get $zi2)) (local.get $cr)))
        (local.set $it (i32.add (local.get $it) (i32.const 1)))
        (br $l)))
    (local.get $it))

  ;; f32x4 SIMD kernel, two interleaved vectors (8 px per group) for ILP.
  (func (export "mandel_rows")
        (param $w i32) (param $h i32) (param $maxit i32)
        (param $r0 i32) (param $step i32) (param $nrows i32) (param $out i32)
    (local $r i32) (local $x i32) (local $it i32) (local $k i32)
    (local $rowbase i32) (local $addr i32) (local $any i32)
    (local $dx f64) (local $dy f64) (local $cid f64)
    (local $civ v128) (local $ci2 v128) (local $maxv v128)
    (local $four v128) (local $quart v128) (local $eps v128)
    (local $cra v128) (local $za v128) (local $wa v128)
    (local $za2 v128) (local $wa2 v128) (local $acta v128)
    (local $cnta v128) (local $inseta v128) (local $ta v128) (local $qa v128)
    (local $crb v128) (local $zb v128) (local $wb v128)
    (local $zb2 v128) (local $wb2 v128) (local $actb v128)
    (local $cntb v128) (local $insetb v128) (local $tb v128) (local $qb v128)

    (local.set $dx (f64.div (f64.const 3.5)
                            (f64.convert_i32_s (i32.sub (local.get $w) (i32.const 1)))))
    (local.set $dy (f64.div (f64.const 2)
                            (f64.convert_i32_s (i32.sub (local.get $h) (i32.const 1)))))
    (local.set $maxv (i32x4.splat (local.get $maxit)))
    (local.set $four (f32x4.splat (f32.const 4)))
    (local.set $quart (f32x4.splat (f32.const 0.25)))
    (local.set $eps (f32x4.splat (f32.const 0.0625)))

    (local.set $r (local.get $r0))
    (local.set $k (i32.const 0))
    (block $rows_done
      (loop $rows
        (br_if $rows_done (i32.ge_s (local.get $k) (local.get $nrows)))
        (local.set $cid (f64.add (f64.const -1)
                                 (f64.mul (f64.convert_i32_s (local.get $r)) (local.get $dy))))
        (local.set $civ (f32x4.splat (f32.demote_f64 (local.get $cid))))
        (local.set $ci2 (f32x4.mul (local.get $civ) (local.get $civ)))
        (local.set $rowbase (i32.add (local.get $out)
            (i32.mul (local.get $r) (i32.shl (local.get $w) (i32.const 1)))))
        (local.set $x (i32.const 0))

        ;; 8-pixel groups: vector a = x..x+3, vector b = x+4..x+7
        (block $vec_done
          (loop $vec
            (br_if $vec_done (i32.gt_s (i32.add (local.get $x) (i32.const 8)) (local.get $w)))
            ;; build cr lanes from f64 then demote (matches scalar grid exactly)
            (local.set $cra (f32x4.splat (f32.demote_f64 (f64.add (f64.const -2.5)
                (f64.mul (f64.convert_i32_s (local.get $x)) (local.get $dx))))))
            (local.set $cra (f32x4.replace_lane 1 (local.get $cra) (f32.demote_f64 (f64.add (f64.const -2.5)
                (f64.mul (f64.convert_i32_s (i32.add (local.get $x) (i32.const 1))) (local.get $dx))))))
            (local.set $cra (f32x4.replace_lane 2 (local.get $cra) (f32.demote_f64 (f64.add (f64.const -2.5)
                (f64.mul (f64.convert_i32_s (i32.add (local.get $x) (i32.const 2))) (local.get $dx))))))
            (local.set $cra (f32x4.replace_lane 3 (local.get $cra) (f32.demote_f64 (f64.add (f64.const -2.5)
                (f64.mul (f64.convert_i32_s (i32.add (local.get $x) (i32.const 3))) (local.get $dx))))))
            (local.set $crb (f32x4.splat (f32.demote_f64 (f64.add (f64.const -2.5)
                (f64.mul (f64.convert_i32_s (i32.add (local.get $x) (i32.const 4))) (local.get $dx))))))
            (local.set $crb (f32x4.replace_lane 1 (local.get $crb) (f32.demote_f64 (f64.add (f64.const -2.5)
                (f64.mul (f64.convert_i32_s (i32.add (local.get $x) (i32.const 5))) (local.get $dx))))))
            (local.set $crb (f32x4.replace_lane 2 (local.get $crb) (f32.demote_f64 (f64.add (f64.const -2.5)
                (f64.mul (f64.convert_i32_s (i32.add (local.get $x) (i32.const 6))) (local.get $dx))))))
            (local.set $crb (f32x4.replace_lane 3 (local.get $crb) (f32.demote_f64 (f64.add (f64.const -2.5)
                (f64.mul (f64.convert_i32_s (i32.add (local.get $x) (i32.const 7))) (local.get $dx))))))

            ;; cardioid + bulb per-lane -> initial in-set mask
            (local.set $ta (f32x4.sub (local.get $cra) (local.get $quart)))
            (local.set $qa (f32x4.add (f32x4.mul (local.get $ta) (local.get $ta)) (local.get $ci2)))
            (local.set $inseta (f32x4.le
                (f32x4.mul (local.get $qa) (f32x4.add (local.get $qa) (local.get $ta)))
                (f32x4.mul (local.get $quart) (local.get $ci2))))
            (local.set $ta (f32x4.add (local.get $cra) (f32x4.splat (f32.const 1))))
            (local.set $inseta (v128.or (local.get $inseta)
                (f32x4.le (f32x4.add (f32x4.mul (local.get $ta) (local.get $ta)) (local.get $ci2))
                          (local.get $eps))))
            (local.set $tb (f32x4.sub (local.get $crb) (local.get $quart)))
            (local.set $qb (f32x4.add (f32x4.mul (local.get $tb) (local.get $tb)) (local.get $ci2)))
            (local.set $insetb (f32x4.le
                (f32x4.mul (local.get $qb) (f32x4.add (local.get $qb) (local.get $tb)))
                (f32x4.mul (local.get $quart) (local.get $ci2))))
            (local.set $tb (f32x4.add (local.get $crb) (f32x4.splat (f32.const 1))))
            (local.set $insetb (v128.or (local.get $insetb)
                (f32x4.le (f32x4.add (f32x4.mul (local.get $tb) (local.get $tb)) (local.get $ci2))
                          (local.get $eps))))

            (local.set $acta (v128.not (local.get $inseta)))
            (local.set $actb (v128.not (local.get $insetb)))
            (local.set $za (v128.const i32x4 0 0 0 0))
            (local.set $wa (v128.const i32x4 0 0 0 0))
            (local.set $zb (v128.const i32x4 0 0 0 0))
            (local.set $wb (v128.const i32x4 0 0 0 0))
            (local.set $cnta (v128.const i32x4 0 0 0 0))
            (local.set $cntb (v128.const i32x4 0 0 0 0))
            (local.set $it (i32.const 0))

            (if (i32.or (v128.any_true (local.get $acta)) (v128.any_true (local.get $actb)))
              (then
                (block $done
                  (loop $iter
                    ;; z = zr ($za/$zb), w = zi ($wa/$wb)
                    (local.set $za2 (f32x4.mul (local.get $za) (local.get $za)))
                    (local.set $zb2 (f32x4.mul (local.get $zb) (local.get $zb)))
                    (local.set $wa2 (f32x4.mul (local.get $wa) (local.get $wa)))
                    (local.set $wb2 (f32x4.mul (local.get $wb) (local.get $wb)))
                    (local.set $acta (v128.and (local.get $acta)
                        (f32x4.le (f32x4.add (local.get $za2) (local.get $wa2)) (local.get $four))))
                    (local.set $actb (v128.and (local.get $actb)
                        (f32x4.le (f32x4.add (local.get $zb2) (local.get $wb2)) (local.get $four))))
                    (br_if $done (i32.eqz (i32.or
                        (v128.any_true (local.get $acta))
                        (v128.any_true (local.get $actb)))))
                    (local.set $cnta (i32x4.sub (local.get $cnta) (local.get $acta)))
                    (local.set $cntb (i32x4.sub (local.get $cntb) (local.get $actb)))
                    (local.set $wa (f32x4.add
                        (f32x4.mul (f32x4.add (local.get $za) (local.get $za)) (local.get $wa))
                        (local.get $civ)))
                    (local.set $wb (f32x4.add
                        (f32x4.mul (f32x4.add (local.get $zb) (local.get $zb)) (local.get $wb))
                        (local.get $civ)))
                    (local.set $za (f32x4.add (f32x4.sub (local.get $za2) (local.get $wa2)) (local.get $cra)))
                    (local.set $zb (f32x4.add (f32x4.sub (local.get $zb2) (local.get $wb2)) (local.get $crb)))
                    (local.set $it (i32.add (local.get $it) (i32.const 1)))
                    (br_if $iter (i32.lt_s (local.get $it) (local.get $maxit)))))))

            ;; in-set lanes -> max_iter, store 8 u16
            (local.set $cnta (v128.bitselect (local.get $maxv) (local.get $cnta) (local.get $inseta)))
            (local.set $cntb (v128.bitselect (local.get $maxv) (local.get $cntb) (local.get $insetb)))
            (local.set $addr (i32.add (local.get $rowbase) (i32.shl (local.get $x) (i32.const 1))))
            (i32.store16 (local.get $addr) (i32x4.extract_lane 0 (local.get $cnta)))
            (i32.store16 offset=2 (local.get $addr) (i32x4.extract_lane 1 (local.get $cnta)))
            (i32.store16 offset=4 (local.get $addr) (i32x4.extract_lane 2 (local.get $cnta)))
            (i32.store16 offset=6 (local.get $addr) (i32x4.extract_lane 3 (local.get $cnta)))
            (i32.store16 offset=8 (local.get $addr) (i32x4.extract_lane 0 (local.get $cntb)))
            (i32.store16 offset=10 (local.get $addr) (i32x4.extract_lane 1 (local.get $cntb)))
            (i32.store16 offset=12 (local.get $addr) (i32x4.extract_lane 2 (local.get $cntb)))
            (i32.store16 offset=14 (local.get $addr) (i32x4.extract_lane 3 (local.get $cntb)))
            (local.set $x (i32.add (local.get $x) (i32.const 8)))
            (br $vec)))

        ;; scalar f64 tail
        (block $tail_done
          (loop $tail
            (br_if $tail_done (i32.ge_s (local.get $x) (local.get $w)))
            (i32.store16
              (i32.add (local.get $rowbase) (i32.shl (local.get $x) (i32.const 1)))
              (call $px
                (f64.add (f64.const -2.5)
                         (f64.mul (f64.convert_i32_s (local.get $x)) (local.get $dx)))
                (local.get $cid)
                (local.get $maxit)))
            (local.set $x (i32.add (local.get $x) (i32.const 1)))
            (br $tail)))

        (local.set $r (i32.add (local.get $r) (local.get $step)))
        (local.set $k (i32.add (local.get $k) (i32.const 1)))
        (br $rows)))))
