// C shim: fans the ISPC row kernel across cores with libdispatch,
// computes only the top half of rows and mirrors the bottom half.
#include <dispatch/dispatch.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "ispcbrot.h"

void mandelbrot(int width, int height, int max_iter, uint16_t *out) {
    const float dx = 3.5f / (float)(width - 1);
    const float dy = 2.0f / (float)(height - 1);
    const float x0 = -2.5f;
    const float y0 = -1.0f;

    const size_t half = ((size_t)height + 1) / 2;

    dispatch_apply(half,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t row) {
        const float ci = y0 + (float)row * dy;
        uint16_t *orow = out + row * (size_t)width;
        mandel_row(width, dx, x0, ci, max_iter, orow);

        const size_t mrow = (size_t)height - 1 - row;
        if (mrow != row)
            memcpy(out + mrow * (size_t)width, orow,
                   (size_t)width * sizeof(uint16_t));
    });
}
