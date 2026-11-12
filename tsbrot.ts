// TsBrot persistent worker: reads "<w> <h> <iters>\n" lines on stdin, writes
// exactly w*h*2 bytes (uint16 LE, row-major) per request to stdout.
// Identical algorithm to jsbrot.mjs — this entry exists to show that TypeScript
// types are erased at runtime (Node's native type-stripping), so it runs at
// JavaScript speed. Pool of worker_threads shares a SharedArrayBuffer; rows are
// handed out via an Atomics counter. Only the top half is computed (y-symmetry).
import { Worker, isMainThread, parentPort } from "node:worker_threads";
import { availableParallelism } from "node:os";
import { createInterface } from "node:readline";
import { once } from "node:events";

interface FrameMsg {
  width: number;
  height: number;
  maxIter: number;
  sab: SharedArrayBuffer;
  ctrl: SharedArrayBuffer;
}

function computeRow(out: Uint16Array, base: number, width: number,
                    dx: number, ci: number, maxIter: number): void {
  const ci2 = ci * ci;
  for (let x = 0; x < width; x++) {
    const cr = -2.5 + x * dx;
    const crm = cr - 0.25;
    const q = crm * crm + ci2;
    // cardioid + period-2 bulb early-out
    if (q * (q + crm) <= 0.25 * ci2 || (cr + 1) * (cr + 1) + ci2 <= 0.0625) {
      out[base + x] = maxIter;
      continue;
    }
    let zr = 0.0, zi = 0.0, it = 0;
    while (it < maxIter) {
      const zr2 = zr * zr, zi2 = zi * zi;
      if (zr2 + zi2 > 4.0) break;
      zi = 2.0 * zr * zi + ci;
      zr = zr2 - zi2 + cr;
      it++;
    }
    out[base + x] = it; // iterations survived: in-set pixels = maxIter
  }
}

function computeFrame(out: Uint16Array, ctrl: Int32Array,
                      width: number, height: number, maxIter: number): void {
  const dx = 3.5 / (width - 1);
  const dy = 2.0 / (height - 1);
  const half = (height + 1) >> 1;
  for (;;) {
    const row = Atomics.add(ctrl, 0, 1);
    if (row >= half) break;
    const base = row * width;
    computeRow(out, base, width, dx, -1.0 + row * dy, maxIter);
    const mirror = (height - 1 - row) * width;
    if (mirror !== base) out.copyWithin(mirror, base, base + width);
  }
}

if (isMainThread) {
  const N = availableParallelism();
  process.stdout.setMaxListeners(0);
  process.stderr.setMaxListeners(0);
  const ctrlSab = new SharedArrayBuffer(4);
  const ctrl = new Int32Array(ctrlSab);
  let sab = new SharedArrayBuffer(2);

  let doneCount = 0;
  let resolveFrame: (() => void) | null = null;
  const workers: Worker[] = [];
  for (let i = 0; i < N; i++) {
    const w = new Worker(new URL(import.meta.url));
    w.on("message", () => {
      if (++doneCount === N && resolveFrame) resolveFrame();
    });
    w.on("error", (e: Error) => {
      process.stderr.write(`tsbrot worker error: ${e}\n`);
      process.exit(1);
    });
    w.unref();
    workers.push(w);
  }

  const rl = createInterface({ input: process.stdin, terminal: false });
  for await (const line of rl) {
    const parts = line.trim().split(/\s+/);
    if (parts.length < 3) continue;
    const w = Number(parts[0]) | 0, h = Number(parts[1]) | 0, maxIter = Number(parts[2]) | 0;
    const nbytes = w * h * 2;
    if (sab.byteLength < nbytes) sab = new SharedArrayBuffer(nbytes);
    Atomics.store(ctrl, 0, 0);
    doneCount = 0;
    const frame = new Promise<void>((r) => { resolveFrame = r; });
    const msg: FrameMsg = { width: w, height: h, maxIter, sab, ctrl: ctrlSab };
    for (const worker of workers) worker.postMessage(msg);
    await frame;
    const buf = Buffer.from(new Uint8Array(sab, 0, nbytes));
    if (!process.stdout.write(buf)) await once(process.stdout, "drain");
  }
  process.exit(0);
} else {
  {
    const warm = new Uint16Array(128 * 128);
    const wctrl = new Int32Array(1);
    computeFrame(warm, wctrl, 128, 128, 256);
  }
  parentPort!.on("message", ({ width, height, maxIter, sab, ctrl }: FrameMsg) => {
    computeFrame(new Uint16Array(sab), new Int32Array(ctrl), width, height, maxIter);
    parentPort!.postMessage(1);
  });
}
