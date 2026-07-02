// GoBrot - Mandelbrot kernel built as a c-shared dylib, called via ctypes.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

//export MandelbrotGo
func MandelbrotGo(w, h, it C.int, out *C.uint16_t) {
	width := int(w)
	height := int(h)
	maxIter := uint16(it)
	buf := unsafe.Slice((*uint16)(unsafe.Pointer(out)), width*height)

	dx := float32(3.5) / float32(width-1)
	dy := float32(2.0) / float32(height-1)

	// y-axis symmetry: compute top half, mirror the rest.
	half := (height + 1) / 2
	var next int64 = -1
	var wg sync.WaitGroup
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				row := int(atomic.AddInt64(&next, 1))
				if row >= half {
					return
				}
				ci := float32(-1.0) + float32(row)*dy
				ci2 := ci * ci
				line := buf[row*width : (row+1)*width]
				for x := 0; x < width; x++ {
					cr := float32(-2.5) + float32(x)*dx
					// cardioid + period-2 bulb interior checks
					crm := cr - 0.25
					q := crm*crm + ci2
					if q*(q+crm) <= 0.25*ci2 || (cr+1)*(cr+1)+ci2 <= 0.0625 {
						line[x] = maxIter
						continue
					}
					var zr, zi float32
					var n uint16
					for n < maxIter {
						zr2 := zr * zr
						zi2 := zi * zi
						if zr2+zi2 > 4.0 {
							break
						}
						zi = 2*zr*zi + ci
						zr = zr2 - zi2 + cr
						n++
					}
					line[x] = n
				}
				mirror := height - 1 - row
				if mirror != row { // odd height: middle row has no twin
					copy(buf[mirror*width:(mirror+1)*width], line)
				}
			}
		}()
	}
	wg.Wait()
}

func main() {}
