%% Erlangbrot - persistent Mandelbrot worker on the BEAM (plain Erlang/OTP).
%% Protocol: "<width> <height> <iters>\n" on stdin ->
%% exactly width*height*2 bytes uint16 LE row-major on stdout.
%% stdout stays binary-clean; loops until EOF.

-module(erlangbrot).
-export([start/0]).
-compile({inline, [pixel/3]}).

start() ->
    %% Raw byte mode: unicode mode would UTF-8-expand bytes >= 16#80.
    ok = io:setopts(standard_io, [binary, {encoding, latin1}]),
    serve(erlang:system_info(schedulers_online)).

serve(NB) ->
    case io:get_line(standard_io, "") of
        eof ->
            halt(0);
        {error, _} ->
            halt(0);
        Line ->
            [W, H, It | _] =
                [binary_to_integer(T)
                 || T <- binary:split(Line, [<<" ">>, <<"\n">>, <<"\r">>],
                                      [global, trim_all])],
            ok = file:write(standard_io, frame(W, H, It, NB)),
            serve(NB)
    end.

%% Compute the top half in parallel band processes, mirror the bottom
%% (y-axis symmetry). Row binaries are shared; the mirror is just cells.
frame(W, H, Max, NB) ->
    Dx = 3.5 / (W - 1),
    Dy = 2.0 / (H - 1),
    Crs = crs(0, W, Dx, []),
    Top = (H + 1) div 2,
    %% 4 bands per scheduler for load balance near the set boundary.
    Chunk = max(1, (Top + NB * 4 - 1) div (NB * 4)),
    Bands = bands(0, Top, Chunk),
    Parent = self(),
    lists:foreach(
      fun({R0, R1}) ->
              spawn_opt(fun() -> Parent ! {R0, rows(R0, R1, Crs, Dy, Max)} end,
                        [link, {min_heap_size, 65536}])
      end, Bands),
    TopRows = lists:append([receive {R0, Rs} -> Rs end || {R0, _} <- Bands]),
    [TopRows, lists:reverse(lists:sublist(TopRows, H - Top))].

bands(R0, Top, Chunk) when R0 < Top ->
    [{R0, min(R0 + Chunk, Top)} | bands(R0 + Chunk, Top, Chunk)];
bands(_, _, _) ->
    [].

%% Descending-x list of cr values; prepending pixels restores ascending x.
crs(X, W, Dx, Acc) when X < W -> crs(X + 1, W, Dx, [-2.5 + X * Dx | Acc]);
crs(_, _, _, Acc) -> Acc.

rows(R, R1, Crs, Dy, Max) when R < R1 ->
    Ci = -1.0 + R * Dy,
    [row(Crs, Ci, Max, []) | rows(R + 1, R1, Crs, Dy, Max)];
rows(_, _, _, _, _) ->
    [].

row([Cr | Rest], Ci, Max, Acc) ->
    row(Rest, Ci, Max, [<<(pixel(Cr, Ci, Max)):16/little>> | Acc]);
row([], _, _, Acc) ->
    iolist_to_binary(Acc).

pixel(Cr, Ci, Max) ->
    Crm = Cr - 0.25,
    Ci2 = Ci * Ci,
    Q = Crm * Crm + Ci2,
    if
        %% Main cardioid and period-2 bulb: in the set, skip the loop.
        Q * (Q + Crm) =< 0.25 * Ci2 -> Max;
        (Cr + 1.0) * (Cr + 1.0) + Ci2 =< 0.0625 -> Max;
        true -> escape(0.0, 0.0, Cr, Ci, 0, Max)
    end.

escape(Zr, Zi, Cr, Ci, It, Max) when It < Max ->
    Zr2 = Zr * Zr,
    Zi2 = Zi * Zi,
    if
        Zr2 + Zi2 > 4.0 -> It;
        true -> escape(Zr2 - Zi2 + Cr, 2.0 * Zr * Zi + Ci, Cr, Ci, It + 1, Max)
    end;
escape(_, _, _, _, It, _) ->
    It.
