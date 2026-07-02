// Persistent Mandelbrot worker: reads "width height maxIter\n" on stdin,
// writes width*height uint16 little-endian counts (row-major) on stdout.
// Count = iterations survived; in-set pixels = maxIter.
// SIMD via jdk.incubator.vector, rows parallelized across all cores.

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.IntVector
import jdk.incubator.vector.VectorMask
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies
import java.io.BufferedOutputStream
import java.io.BufferedReader
import java.io.FileDescriptor
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.util.stream.IntStream

private val FS: VectorSpecies<Float> = FloatVector.SPECIES_128
private val IS: VectorSpecies<Int> = IntVector.SPECIES_128

fun main() {
    val reader = BufferedReader(InputStreamReader(System.`in`, Charsets.US_ASCII))
    // Raw fd 1 for binary output; never touch System.out.
    val out = BufferedOutputStream(FileOutputStream(FileDescriptor.out), 1 shl 20)
    while (true) {
        val line = reader.readLine() ?: break
        val t = line.trim()
        if (t.isEmpty()) continue
        val p = t.split(Regex("\\s+"))
        out.write(compute(p[0].toInt(), p[1].toInt(), p[2].toInt()))
        out.flush()
    }
}

fun compute(width: Int, height: Int, maxIter: Int): ByteArray {
    val frame = ByteArray(width * height * 2)
    val dx = if (width > 1) 3.5 / (width - 1) else 0.0
    val dy = if (height > 1) 2.0 / (height - 1) else 0.0
    val crArr = FloatArray(width) { x -> (-2.5 + x * dx).toFloat() }

    val half = (height + 1) / 2 // y-symmetry: bottom half mirrors top
    IntStream.range(0, half).parallel().forEach { row ->
        val ci = (-1.0 + row * dy).toFloat()
        val counts = IntArray(width)
        computeRow(crArr, ci, maxIter, counts)
        val off = row * width * 2
        for (x in 0 until width) {
            val v = counts[x]
            frame[off + 2 * x] = v.toByte()
            frame[off + 2 * x + 1] = (v ushr 8).toByte()
        }
        val mrow = height - 1 - row
        if (mrow != row) {
            System.arraycopy(frame, off, frame, mrow * width * 2, width * 2)
        }
    }
    return frame
}

private fun computeRow(crArr: FloatArray, ci: Float, maxIter: Int, counts: IntArray) {
    val width = crArr.size
    val civ = FloatVector.broadcast(FS, ci)
    val ci2v = civ.mul(civ)
    val lanes = FS.length()
    val upper = width - width % lanes
    var x = 0
    while (x < upper) {
        val cr = FloatVector.fromArray(FS, crArr, x)

        // cardioid + period-2 bulb early-out
        val crm = cr.sub(0.25f)
        val q = crm.mul(crm).add(ci2v)
        var inSet: VectorMask<Float> =
            q.mul(q.add(crm)).compare(VectorOperators.LE, ci2v.mul(0.25f))
        val crp = cr.add(1.0f)
        inSet = inSet.or(crp.mul(crp).add(ci2v).compare(VectorOperators.LE, 0.0625f))

        var count = IntVector.zero(IS).blend(maxIter.toLong(), inSet.cast(IS))
        var active = inSet.not()
        if (active.anyTrue()) {
            var zr = FloatVector.zero(FS)
            var zi = FloatVector.zero(FS)
            for (it in 0 until maxIter) {
                val zr2 = zr.mul(zr)
                val zi2 = zi.mul(zi)
                active = active.andNot(zr2.add(zi2).compare(VectorOperators.GT, 4.0f))
                if (!active.anyTrue()) break
                val nzr = zr2.sub(zi2).add(cr)
                zi = zr.add(zr).fma(zi, civ) // 2*zr*zi + ci
                zr = nzr
                count = count.sub(-1, active.cast(IS)) // +1 on active lanes
            }
        }
        count.intoArray(counts, x)
        x += lanes
    }
    while (x < width) {
        counts[x] = scalarPixel(crArr[x], ci, maxIter)
        x++
    }
}

private fun scalarPixel(cr: Float, ci: Float, maxIter: Int): Int {
    val crm = cr - 0.25f
    val ci2 = ci * ci
    val q = crm * crm + ci2
    if (q * (q + crm) <= 0.25f * ci2) return maxIter
    val crp = cr + 1.0f
    if (crp * crp + ci2 <= 0.0625f) return maxIter
    var zr = 0.0f
    var zi = 0.0f
    var it = 0
    while (it < maxIter) {
        val zr2 = zr * zr
        val zi2 = zi * zi
        if (zr2 + zi2 > 4.0f) break
        val nzr = zr2 - zi2 + cr
        zi = 2.0f * zr * zi + ci
        zr = nzr
        it++
    }
    return it
}
