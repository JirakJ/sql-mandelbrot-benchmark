-- LuaJIT Mandelbrot worker: reads "w h it r0 r1\n", computes rows [r0,r1)
-- of the top half, writes them as raw uint16 LE bytes to stdout. Loops to EOF.
local ffi = require("ffi")

io.stdout:setvbuf("no")

local function compute(w, h, maxit, r0, r1, buf)
    local dx = 3.5 / (w - 1)
    local dy = 2.0 / (h - 1)
    local idx = 0
    for row = r0, r1 - 1 do
        local ci = -1.0 + row * dy
        local ci2 = ci * ci
        for x = 0, w - 1 do
            local cr = -2.5 + x * dx
            local res
            local crm = cr - 0.25
            local q = crm * crm + ci2
            local cp1 = cr + 1.0
            if q * (q + crm) <= 0.25 * ci2 or cp1 * cp1 + ci2 <= 0.0625 then
                res = maxit  -- cardioid / period-2 bulb: in set
            else
                local zr, zi = 0.0, 0.0
                local it = 0
                while it < maxit do
                    local zr2 = zr * zr
                    local zi2 = zi * zi
                    if zr2 + zi2 > 4.0 then break end
                    local nzr = zr2 - zi2 + cr
                    zi = 2.0 * zr * zi + ci
                    zr = nzr
                    it = it + 1
                end
                res = it
            end
            buf[idx] = res
            idx = idx + 1
        end
    end
end

while true do
    local line = io.stdin:read("*l")
    if not line then break end
    local w, h, maxit, r0, r1 =
        line:match("^(%d+)%s+(%d+)%s+(%d+)%s+(%d+)%s+(%d+)")
    w, h, maxit = tonumber(w), tonumber(h), tonumber(maxit)
    r0, r1 = tonumber(r0), tonumber(r1)
    local n = (r1 - r0) * w
    local buf = ffi.new("uint16_t[?]", n)
    compute(w, h, maxit, r0, r1, buf)
    io.write(ffi.string(buf, n * 2))
    io.flush()
end
