// DartBrot worker: persistent process, isolate pool, binary frames on stdout.
//
// Protocol: reads "<width> <height> <max_iter>\n" lines from stdin, replies
// with width*height*2 bytes (uint16 little-endian, row-major) on stdout.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

void worker(SendPort mainPort) {
  final rp = ReceivePort();
  mainPort.send(rp.sendPort);
  rp.listen((msg) {
    final m = msg as List;
    final int w = m[0] as int;
    final int h = m[1] as int;
    final int maxIter = m[2] as int;
    final int r0 = m[3] as int;
    final int r1 = m[4] as int;
    final int workerId = m[5] as int;
    final dx = 3.5 / (w - 1);
    final dy = 2.0 / (h - 1);
    final buf = Uint16List((r1 - r0) * w);
    var o = 0;
    for (var row = r0; row < r1; row++) {
      final ci = -1.0 + row * dy;
      final ci2 = ci * ci;
      for (var x = 0; x < w; x++) {
        final cr = -2.5 + x * dx;
        final crm = cr - 0.25;
        final q = crm * crm + ci2;
        int it;
        // Cardioid + period-2 bulb early-out.
        if (q * (q + crm) <= 0.25 * ci2 ||
            (cr + 1.0) * (cr + 1.0) + ci2 <= 0.0625) {
          it = maxIter;
        } else {
          var zr = 0.0, zi = 0.0;
          it = 0;
          while (it < maxIter) {
            final zr2 = zr * zr;
            final zi2 = zi * zi;
            if (zr2 + zi2 > 4.0) break;
            zi = 2.0 * zr * zi + ci;
            zr = zr2 - zi2 + cr;
            it++;
          }
        }
        buf[o++] = it;
      }
    }
    mainPort.send([r0, r1, workerId, TransferableTypedData.fromList([buf])]);
  });
}

Future<void> main() async {
  final nWorkers = Platform.numberOfProcessors;
  final resultPort = ReceivePort();
  final workerPorts = <SendPort>[];
  final ready = Completer<void>();

  // Per-frame dispatch state, rebound for each request.
  void Function(List)? onResult;

  resultPort.listen((msg) {
    if (msg is SendPort) {
      workerPorts.add(msg);
      if (workerPorts.length == nWorkers && !ready.isCompleted) {
        ready.complete();
      }
    } else {
      onResult!(msg as List);
    }
  });

  for (var i = 0; i < nWorkers; i++) {
    await Isolate.spawn(worker, resultPort.sendPort);
  }
  await ready.future;

  Future<Uint16List> computeFrame(int w, int h, int maxIter) {
    final half = (h + 1) >> 1; // top half incl. middle row when odd
    final frame = Uint16List(h * w);
    final chunk = (half ~/ (nWorkers * 8)).clamp(1, 1 << 30);
    var nextRow = 0;
    var outstanding = 0;
    final done = Completer<Uint16List>();

    void dispatch(int workerId) {
      if (nextRow >= half) return;
      final r0 = nextRow;
      final r1 = (r0 + chunk < half) ? r0 + chunk : half;
      nextRow = r1;
      outstanding++;
      workerPorts[workerId].send([w, h, maxIter, r0, r1, workerId]);
    }

    onResult = (m) {
      final r0 = m[0] as int;
      final r1 = m[1] as int;
      final workerId = m[2] as int;
      final data = (m[3] as TransferableTypedData).materialize().asUint16List();
      frame.setRange(r0 * w, r1 * w, data);
      outstanding--;
      dispatch(workerId);
      if (outstanding == 0 && nextRow >= half) {
        // Mirror: row h-1-r duplicates row r (skip middle row when h is odd).
        for (var r = 0; r < half; r++) {
          final mr = h - 1 - r;
          if (mr != r) {
            frame.setRange(mr * w, mr * w + w, frame, r * w);
          }
        }
        done.complete(frame);
      }
    };

    for (var i = 0; i < nWorkers; i++) {
      dispatch(i);
    }
    return done.future;
  }

  final lines = stdin.transform(utf8.decoder).transform(const LineSplitter());
  await for (final line in lines) {
    final parts = line.trim().split(RegExp(r'\s+'));
    if (parts.length < 3 || parts[0].isEmpty) continue;
    final w = int.parse(parts[0]);
    final h = int.parse(parts[1]);
    final maxIter = int.parse(parts[2]);
    final frame = await computeFrame(w, h, maxIter);
    stdout.add(frame.buffer.asUint8List(0, w * h * 2));
    await stdout.flush();
  }
  exit(0);
}
