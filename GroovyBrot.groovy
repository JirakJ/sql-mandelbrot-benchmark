import groovy.transform.CompileStatic

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.IntVector
import jdk.incubator.vector.VectorMask
import jdk.incubator.vector.VectorOperators
import jdk.incubator.vector.VectorSpecies

import java.nio.charset.StandardCharsets
import java.util.stream.IntStream

/**
 * Persistent Mandelbrot worker: reads "width height maxIter\n" lines on stdin,
 * writes width*height uint16 little-endian counts (row-major) on raw fd 1.
 * Count convention: iterations survived, in-set pixels = maxIter.
 */
@CompileStatic
class GroovyBrot {

    private static final VectorSpecies<Float> FS = FloatVector.SPECIES_128
    private static final VectorSpecies<Integer> IS = IntVector.SPECIES_128

    static void main(String[] args) {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.US_ASCII))
        // Raw fd 1 for binary output; System.out (PrintStream) is never used.
        OutputStream out = new BufferedOutputStream(
                new FileOutputStream(FileDescriptor.out), 1 << 20)
        String line
        while ((line = br.readLine()) != null) {
            line = line.trim()
            if (line.isEmpty()) continue
            String[] p = line.split("\\s+")
            int w = Integer.parseInt(p[0])
            int h = Integer.parseInt(p[1])
            int mi = Integer.parseInt(p[2])
            out.write(compute(w, h, mi))
            out.flush()
        }
    }

    static byte[] compute(int width, int height, int maxIter) {
        byte[] frame = new byte[width * height * 2]
        double dx = width > 1 ? 3.5d / (width - 1) : 0.0d
        double dy = height > 1 ? 2.0d / (height - 1) : 0.0d
        float[] crArr = new float[width]
        for (int x = 0; x < width; x++) crArr[x] = (float) (-2.5d + x * dx)

        int half = (height + 1).intdiv(2) // y-symmetry: bottom half mirrors top
        IntStream.range(0, half).parallel().forEach { int row ->
            float ci = (float) (-1.0d + row * dy)
            int[] counts = new int[width]
            computeRow(crArr, ci, maxIter, counts)
            int off = row * width * 2
            for (int x = 0; x < width; x++) {
                int v = counts[x]
                frame[off + 2 * x] = (byte) v
                frame[off + 2 * x + 1] = (byte) (v >>> 8)
            }
            int mrow = height - 1 - row
            if (mrow != row) {
                System.arraycopy(frame, off, frame, mrow * width * 2, width * 2)
            }
        }
        return frame
    }

    static void computeRow(float[] crArr, float ci, int maxIter, int[] counts) {
        int width = crArr.length
        FloatVector civ = FloatVector.broadcast(FS, ci)
        FloatVector ci2v = civ.mul(civ)
        int x = 0
        int upper = width - (width % FS.length())
        for (; x < upper; x += FS.length()) {
            FloatVector cr = FloatVector.fromArray(FS, crArr, x)

            // cardioid + period-2 bulb early-out
            FloatVector crm = cr.sub(0.25f)
            FloatVector q = crm.mul(crm).add(ci2v)
            VectorMask<Float> inSet =
                    q.mul(q.add(crm)).compare(VectorOperators.LE, ci2v.mul(0.25f))
            FloatVector crp = cr.add(1.0f)
            inSet = inSet.or(crp.mul(crp).add(ci2v).compare(VectorOperators.LE, 0.0625f))

            IntVector count = IntVector.zero(IS).blend((long) maxIter, inSet.cast(IS))
            VectorMask<Float> active = inSet.not()
            if (active.anyTrue()) {
                FloatVector zr = FloatVector.zero(FS)
                FloatVector zi = FloatVector.zero(FS)
                for (int it = 0; it < maxIter; it++) {
                    FloatVector zr2 = zr.mul(zr)
                    FloatVector zi2 = zi.mul(zi)
                    active = active.andNot(
                            zr2.add(zi2).compare(VectorOperators.GT, 4.0f))
                    if (!active.anyTrue()) break
                    FloatVector nzr = zr2.sub(zi2).add(cr)
                    zi = zr.add(zr).fma(zi, civ) // 2*zr*zi + ci
                    zr = nzr
                    count = count.sub(-1, active.cast(IS)) // +1 on active lanes
                }
            }
            count.intoArray(counts, x)
        }
        for (; x < width; x++) counts[x] = scalarPixel(crArr[x], ci, maxIter)
    }

    static int scalarPixel(float cr, float ci, int maxIter) {
        float crm = cr - 0.25f
        float ci2 = ci * ci
        float q = crm * crm + ci2
        if (q * (q + crm) <= 0.25f * ci2) return maxIter
        float crp = cr + 1.0f
        if (crp * crp + ci2 <= 0.0625f) return maxIter
        float zr = 0f
        float zi = 0f
        int it = 0
        while (it < maxIter) {
            float zr2 = zr * zr
            float zi2 = zi * zi
            if (zr2 + zi2 > 4.0f) break
            float nzr = zr2 - zi2 + cr
            zi = 2f * zr * zi + ci
            zr = nzr
            it++
        }
        return it
    }
}
