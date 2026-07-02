// ZigBrot - Zig @Vector SIMD Mandelbrot, rows fanned out via libdispatch.
//
// - @Vector(VLEN, f32) masked escape counting: escaped lanes stay masked,
//   count accumulated via @select(u32), horizontal @reduce(.Or, active)
//   break only every 4th iteration (it is the serializing op).
// - cardioid + period-2 bulb analytic early-out.
// - y-axis symmetry: only the top half of rows is computed, rest is a memcpy.
// - threading: dispatch_apply_f on the USER_INITIATED global queue (same as
//   cppbrot). A spec-style std.Thread.spawn pool pulling rows from a
//   std.atomic.Value(u32) counter was implemented and measured first: its
//   best-case handoff is ~10% faster (0.39ms vs 0.43ms med @1400x800x256),
//   but macOS core placement makes it bimodal — roughly half the processes
//   land in a ~2.7ms mode (hot-spin handoff, QoS-tagged and condvar-park
//   variants all hit it). GCD is stable within ~0.03ms across every run, so
//   it wins on expected time. Zig 0.16 also moved std.Thread.Mutex/Condition
//   behind std.Io, so libc/libdispatch is the natural primitive here anyway.
//
// Semantics: value = iterations survived before |z|^2 > 4, capped at max_iter.

const std = @import("std");

// Measured on M4 Max at 1400x800x256: VLEN 8 ~0.42ms, 16 ~0.43ms, 4 ~0.53ms,
// 32 ~0.97ms (register spills). 8 = two NEON registers per op, enough ILP to
// hide the z = z^2 + c FMA latency chain.
const VLEN = 8;

const Vf = @Vector(VLEN, f32);
const Vu = @Vector(VLEN, u32);
const Vb = @Vector(VLEN, bool);

const lane: Vf = std.simd.iota(f32, VLEN);
const ones: Vu = @splat(1);
const zeroes: Vu = @splat(0);
const vtrue: Vb = @splat(true);
const vfalse: Vb = @splat(false);

inline fn vOr(a: Vb, b: Vb) Vb {
    return @select(bool, a, a, b);
}
inline fn vAnd(a: Vb, b: Vb) Vb {
    return @select(bool, a, b, a);
}
inline fn vNot(a: Vb) Vb {
    return @select(bool, a, vfalse, vtrue);
}

// job state; dispatch_apply_f is a barrier, so plain globals are fine
var jw: u32 = 0;
var jh: u32 = 0;
var jmax: u32 = 0;
var jhalf: u32 = 0;
var jout: [*]u16 = undefined;

const QOS_CLASS_USER_INITIATED: isize = 0x19;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;
extern "c" fn dispatch_apply_f(
    iterations: usize,
    queue: ?*anyopaque,
    ctx: ?*anyopaque,
    work: *const fn (?*anyopaque, usize) callconv(.c) void,
) void;

fn escapeRow(row: u32) void {
    const w = jw;
    const max_iter = jmax;
    const dx = 3.5 / @as(f32, @floatFromInt(w - 1));
    const dy = 2.0 / @as(f32, @floatFromInt(jh - 1));
    const cy = -1.0 + @as(f32, @floatFromInt(row)) * dy;
    const orow = jout + @as(usize, row) * @as(usize, w);

    const vdx: Vf = @splat(dx);
    const vx0: Vf = @splat(-2.5);
    const four: Vf = @splat(4.0);
    const two: Vf = @splat(2.0);
    const quarter: Vf = @splat(0.25);
    const one: Vf = @splat(1.0);
    const bulb_r2: Vf = @splat(0.0625);
    const ci: Vf = @splat(cy);
    const ci2 = ci * ci;
    const vmax: Vu = @splat(max_iter);

    var x: u32 = 0;
    while (x + VLEN <= w) : (x += VLEN) {
        const xv = @as(Vf, @splat(@as(f32, @floatFromInt(x)))) + lane;
        const cr = @mulAdd(Vf, xv, vdx, vx0);
        // main cardioid: q*(q + (cr-1/4)) <= (1/4)*ci^2, q = (cr-1/4)^2 + ci^2
        const crm = cr - quarter;
        const q = @mulAdd(Vf, crm, crm, ci2);
        const card = q * (q + crm) <= quarter * ci2;
        // period-2 bulb: (cr+1)^2 + ci^2 <= 1/16
        const crp = cr + one;
        const bulb = @mulAdd(Vf, crp, crp, ci2) <= bulb_r2;
        const in_set = vOr(card, bulb);

        var active = vNot(in_set);
        var zr: Vf = @splat(0.0);
        var zi: Vf = @splat(0.0);
        var count: Vu = @splat(0);
        var i: u32 = 0;
        while (i < max_iter) : (i += 1) {
            const zr2 = zr * zr;
            const zi2 = zi * zi;
            active = vAnd(active, (zr2 + zi2) <= four);
            count +%= @select(u32, active, ones, zeroes);
            // reduce + branch every 4th iter; dead lanes just run masked
            // (inf/nan compares are false, so they never re-activate)
            if ((i & 3) == 3 and !@reduce(.Or, active)) break;
            const nzr = zr2 - zi2 + cr;
            zi = @mulAdd(Vf, two * zr, zi, ci);
            zr = nzr;
        }
        const res = @select(u32, in_set, vmax, count);
        const res16: @Vector(VLEN, u16) = @truncate(res);
        const arr: [VLEN]u16 = res16;
        @memcpy(orow[x .. x + VLEN], &arr);
    }
    // scalar tail (width not a multiple of VLEN)
    while (x < w) : (x += 1) {
        const cr = -2.5 + @as(f32, @floatFromInt(x)) * dx;
        const crm = cr - 0.25;
        const q = crm * crm + cy * cy;
        var iters: u32 = max_iter;
        if (!(q * (q + crm) <= 0.25 * cy * cy) and
            !((cr + 1.0) * (cr + 1.0) + cy * cy <= 0.0625))
        {
            var zr: f32 = 0.0;
            var zi: f32 = 0.0;
            var i: u32 = 0;
            while (i < max_iter) : (i += 1) {
                const a = zr * zr;
                const b = zi * zi;
                if (a + b > 4.0) break;
                const nzr = a - b + cr;
                zi = 2.0 * zr * zi + cy;
                zr = nzr;
            }
            iters = i;
        }
        orow[x] = @truncate(iters);
    }
    // mirror to the conjugate row
    const mrow = jh - 1 - row;
    if (mrow != row) {
        const drow = jout + @as(usize, mrow) * @as(usize, w);
        @memcpy(drow[0..w], orow[0..w]);
    }
}

fn rowWork(_: ?*anyopaque, row: usize) callconv(.c) void {
    escapeRow(@intCast(row));
}

export fn mandelbrot_zig(w: c_int, h: c_int, it: c_int, out: [*]u16) callconv(.c) void {
    if (w <= 0 or h <= 0) return;
    jw = @intCast(w);
    jh = @intCast(h);
    jmax = if (it > 0) @intCast(it) else 0;
    jout = out;
    jhalf = (jh + 1) / 2;
    dispatch_apply_f(jhalf, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), null, &rowWork);
}
