# Elixirbrot - persistent Mandelbrot worker on the BEAM.
# Protocol: "<width> <height> <iters>\n" on stdin ->
# exactly width*height*2 bytes uint16 LE row-major on stdout.
# stdout stays binary-clean; loops until EOF.

defmodule Elixirbrot do
  @moduledoc false

  def serve(nb) do
    # Raw byte mode: unicode mode would UTF-8-expand bytes >= 0x80.
    :ok = :io.setopts(:standard_io, encoding: :latin1)
    loop(nb)
  end

  defp loop(nb) do
    case IO.gets("") do
      :eof ->
        :ok

      {:error, _} ->
        :ok

      line ->
        [w, h, it] = line |> String.split() |> Enum.map(&String.to_integer/1)
        IO.binwrite(:stdio, frame(w, h, it, nb))
        loop(nb)
    end
  end

  # Compute top half in parallel row bands, mirror the bottom (y symmetry).
  def frame(w, h, max, nb) do
    dx = 3.5 / (w - 1)
    dy = 2.0 / (h - 1)
    crs = crs_desc(0, w, dx, [])
    top = div(h + 1, 2)
    # 4 chunks per scheduler for load balance near the set boundary.
    chunk = max(1, div(top + nb * 4 - 1, nb * 4))

    rows =
      0..(top - 1)//chunk
      |> Enum.map(fn r0 -> {r0, min(r0 + chunk, top)} end)
      |> Task.async_stream(fn {r0, r1} -> band(r0, r1, crs, dy, max) end,
        max_concurrency: nb,
        ordered: true,
        timeout: :infinity
      )
      |> Enum.flat_map(fn {:ok, rs} -> rs end)

    # Row binaries are refcounted, so the mirror is free.
    [rows | rows |> Enum.take(h - top) |> Enum.reverse()]
  end

  defp crs_desc(x, w, dx, acc) when x < w, do: crs_desc(x + 1, w, dx, [-2.5 + x * dx | acc])
  defp crs_desc(_, _, _, acc), do: acc

  defp band(r, r1, crs, dy, max) when r < r1 do
    ci = -1.0 + r * dy
    [row(crs, ci, max, []) | band(r + 1, r1, crs, dy, max)]
  end

  defp band(_, _, _, _, _), do: []

  # crs is descending in x; prepending yields ascending-x row bytes.
  defp row([cr | rest], ci, max, acc),
    do: row(rest, ci, max, [<<pixel(cr, ci, max)::16-little>> | acc])

  defp row([], _, _, acc), do: IO.iodata_to_binary(acc)

  defp pixel(cr, ci, max) do
    crm = cr - 0.25
    ci2 = ci * ci
    q = crm * crm + ci2

    cond do
      # Main cardioid and period-2 bulb: in the set, skip the loop.
      q * (q + crm) <= 0.25 * ci2 -> max
      (cr + 1.0) * (cr + 1.0) + ci2 <= 0.0625 -> max
      true -> escape(0.0, 0.0, cr, ci, 0, max)
    end
  end

  defp escape(zr, zi, cr, ci, it, max) when it < max do
    zr2 = zr * zr
    zi2 = zi * zi

    if zr2 + zi2 > 4.0 do
      it
    else
      escape(zr2 - zi2 + cr, 2.0 * zr * zi + ci, cr, ci, it + 1, max)
    end
  end

  defp escape(_, _, _, _, it, _), do: it
end

Elixirbrot.serve(System.schedulers_online())
