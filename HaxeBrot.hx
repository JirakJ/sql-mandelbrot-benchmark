// HaxeBrot - persistent Mandelbrot worker, Haxe/hxcpp target.
// Protocol: "<w> <h> <iters>\n" on stdin -> w*h*2 bytes uint16 LE row-major on stdout.
// Multithreaded over top-half rows (y-axis symmetry), f64 scalar kernel.

import haxe.io.Bytes;
import sys.thread.Deque;
import sys.thread.Thread;

class HaxeBrot {
	static var nWorkers:Int;
	static var jobQs:Array<Deque<Int>>;
	static var doneQ:Deque<Int>;

	static var W:Int;
	static var H:Int;
	static var MAXIT:Int;
	static var TOP:Int;
	static var buf:Bytes;

	static inline function pixel(cr:Float, ci:Float, maxIter:Int):Int {
		var ci2 = ci * ci;
		// cardioid early-out
		var crm = cr - 0.25;
		var q = crm * crm + ci2;
		if (q * (q + crm) <= 0.25 * ci2) return maxIter;
		// period-2 bulb
		var crp = cr + 1.0;
		if (crp * crp + ci2 <= 0.0625) return maxIter;

		var zr = 0.0;
		var zi = 0.0;
		var it = 0;
		while (it < maxIter) {
			var zr2 = zr * zr;
			var zi2 = zi * zi;
			if (zr2 + zi2 > 4.0) break;
			zi = 2.0 * zr * zi + ci;
			zr = zr2 - zi2 + cr;
			it++;
		}
		return it;
	}

	// Worker id computes interleaved rows id, id+N, ... of the top half.
	static function computeRows(id:Int) {
		var w = W;
		var h = H;
		var maxIter = MAXIT;
		var top = TOP;
		var dx = 3.5 / (w - 1);
		var dy = 2.0 / (h - 1);
		var b = buf;
		var r = id;
		while (r < top) {
			var ci = -1.0 + r * dy;
			var off = r * w * 2;
			for (x in 0...w) {
				var cr = -2.5 + x * dx;
				var v = pixel(cr, ci, maxIter);
				b.set(off, v & 0xFF);
				b.set(off + 1, (v >> 8) & 0xFF);
				off += 2;
			}
			var mr = h - 1 - r;
			if (mr != r) b.blit(mr * w * 2, b, r * w * 2, w * 2);
			r += nWorkers;
		}
	}

	static function worker(id:Int) {
		var q = jobQs[id];
		while (true) {
			if (q.pop(true) < 0) return;
			computeRows(id);
			doneQ.push(id);
		}
	}

	static function main() {
		var args = Sys.args();
		nWorkers = args.length > 0 ? Std.parseInt(args[0]) : 8;
		if (nWorkers < 1) nWorkers = 1;

		doneQ = new Deque();
		jobQs = [for (i in 0...nWorkers) new Deque()];
		for (i in 0...nWorkers) Thread.create(() -> worker(i));

		var stdin = Sys.stdin();
		var stdout = Sys.stdout();
		while (true) {
			var line:String = null;
			try {
				line = stdin.readLine();
			} catch (e:haxe.io.Eof) {
				break;
			}
			var parts = StringTools.trim(line).split(" ");
			W = Std.parseInt(parts[0]);
			H = Std.parseInt(parts[1]);
			MAXIT = Std.parseInt(parts[2]);
			TOP = (H + 1) >> 1;
			var size = W * H * 2;
			if (buf == null || buf.length != size) buf = Bytes.alloc(size);

			for (i in 0...nWorkers) jobQs[i].push(1);
			for (_ in 0...nWorkers) doneQ.pop(true);

			stdout.writeFullBytes(buf, 0, size);
			stdout.flush();
		}
	}
}
