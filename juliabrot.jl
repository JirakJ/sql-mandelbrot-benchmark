# JuliaBrot persistent worker.
# Protocol: read "w h maxit" line from stdin, write w*h UInt16 pixels
# (little-endian, row-major) to stdout, flush, loop until EOF.
# Julia is column-major, so output is built as a flat row-major Vector{UInt16}.
# Semantics match pybrot: value = iterations survived (check-then-iterate),
# in-set pixels = maxit.

function compute_row!(buf::Vector{UInt16}, row::Int, w::Int, maxit::Int,
                      dx::Float64, dy::Float64)
    ci = -1.0 + row * dy
    ci2 = ci * ci
    base = row * w
    @inbounds for x in 0:w-1
        cr = -2.5 + x * dx
        # cardioid + period-2 bulb early-out
        crm = cr - 0.25
        q = crm * crm + ci2
        crp = cr + 1.0
        if q * (q + crm) <= 0.25 * ci2 || crp * crp + ci2 <= 0.0625
            buf[base + x + 1] = UInt16(maxit)
            continue
        end
        zr = 0.0
        zi = 0.0
        it = 0
        while it < maxit
            zr2 = zr * zr
            zi2 = zi * zi
            (zr2 + zi2 > 4.0) && break
            zi = 2.0 * zr * zi + ci
            zr = zr2 - zi2 + cr
            it += 1
        end
        buf[base + x + 1] = UInt16(it)
    end
    return nothing
end

function mandelbrot(w::Int, h::Int, maxit::Int)
    buf = Vector{UInt16}(undef, w * h)
    dx = 3.5 / (w - 1)
    dy = 2.0 / (h - 1)
    # y-axis symmetry: ci of row h-1-r is exactly -ci of row r on this grid
    half = (h + 1) >> 1
    Threads.@threads :dynamic for row in 0:half-1
        compute_row!(buf, row, w, maxit, dx, dy)
        mrow = h - 1 - row
        if mrow != row
            @inbounds copyto!(buf, mrow * w + 1, buf, row * w + 1, w)
        end
    end
    return buf
end

function main()
    while !eof(stdin)
        line = readline(stdin)
        isempty(strip(line)) && continue
        parts = split(line)
        w = parse(Int, parts[1])
        h = parse(Int, parts[2])
        it = parse(Int, parts[3])
        buf = mandelbrot(w, h, it)
        write(stdout, buf)  # raw bytes, host (little) endian
        flush(stdout)
    end
end

main()
