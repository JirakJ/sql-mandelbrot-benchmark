// AsmBrot kernel - hand-written AArch64 NEON, 4 pixels per call.
//
// void mandel4(const float cr[4], float ci, int max_iter,
//              uint32_t count[4], const uint32_t active0[4]);
//
// AAPCS64: x0=cr, s0=ci, w1=max_iter, x2=count, x3=active0.
// count = iterations survived before |z|^2 > 4 (pybrot convention).
// active0 lets the shim pre-kill cardioid/bulb lanes so an in-set lane
// never forces the group to run the full max_iter.
// Uses only v0-v7/v16-v19 (caller-saved) - nothing to spill.
// Escaped lanes keep running masked; fcmge on inf/nan is false, so they
// never re-activate and their count stays exact. The serializing umaxv
// reduction therefore only needs to run every 4th iteration.

    .text
    .p2align 2
    .globl _mandel4
_mandel4:
    ld1     {v1.4s}, [x0]           // cr
    ld1     {v7.4s}, [x3]           // active mask
    dup     v2.4s, v0.s[0]          // ci splat
    fmov    v3.4s, #4.0
    movi    v4.4s, #0               // zr
    movi    v5.4s, #0               // zi
    movi    v6.4s, #0               // count
    mov     w3, #0                  // it
    cbz     w1, Ldone
Lloop:
    fmul    v16.4s, v4.4s, v4.4s    // zr2
    fmul    v17.4s, v5.4s, v5.4s    // zi2
    fadd    v18.4s, v16.4s, v17.4s  // |z|^2
    fcmge   v18.4s, v3.4s, v18.4s   // |z|^2 <= 4.0
    and     v7.16b, v7.16b, v18.16b // active &= survived
    sub     v6.4s, v6.4s, v7.4s     // count += 1 where active (mask = -1)
    add     w3, w3, #1
    tst     w3, #3
    b.ne    Lupd
    umaxv   s18, v7.4s
    fmov    w4, s18
    cbz     w4, Ldone               // all lanes dead
Lupd:
    fadd    v19.4s, v4.4s, v4.4s    // 2*zr (before zr is overwritten)
    fsub    v16.4s, v16.4s, v17.4s
    fadd    v4.4s, v16.4s, v1.4s    // zr' = zr2 - zi2 + cr
    mov     v16.16b, v2.16b
    fmla    v16.4s, v19.4s, v5.4s   // ci + 2*zr*zi
    mov     v5.16b, v16.16b         // zi'
    cmp     w3, w1
    b.lt    Lloop
Ldone:
    st1     {v6.4s}, [x2]
    ret
