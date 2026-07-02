% PrologBrot - SWI-Prolog Mandelbrot persistent worker.
%
% Protocol: reads "W H MAX_ITER\n" lines on stdin; for each frame writes the
% TOP HALF ((H+1)//2 rows) as raw uint16 little-endian, row-major, on stdout
% (binary). The Python wrapper mirrors the bottom half (y-axis symmetry).
%
% Parallelism: real SWI-Prolog threads. Persistent worker threads pull one-row
% jobs from a message queue (dynamic load balancing; rows near the real axis
% are much heavier) and send back per-row byte strings; the main thread
% reassembles them in order and writes.
%
% Hot loop: deterministic tail-recursive escape iteration (if-then-else, no
% choicepoints), compiled arithmetic (-O / optimise flag). Iteration counts
% down so the loop carries one argument less; survived = Max - Left.
%
% Value semantics: iterations survived before zr^2+zi^2 > 4 (check-then-
% iterate), capped at Max (in-set pixels = Max). Cardioid and period-2 bulb
% interiors are resolved analytically.

:- set_prolog_flag(optimise, true).

% iter(Zr, Zi, Cr, Ci, K, Left): K iterations remaining; Left = K at escape.
iter(Zr, Zi, Cr, Ci, K, Left) :-
    (   K =:= 0
    ->  Left = 0
    ;   Zr2 is Zr*Zr,
        Zi2 is Zi*Zi,
        (   Zr2 + Zi2 > 4.0
        ->  Left = K
        ;   Nzr is Zr2 - Zi2 + Cr,
            Nzi is 2.0*Zr*Zi + Ci,
            K1 is K - 1,
            iter(Nzr, Nzi, Cr, Ci, K1, Left)
        )
    ).

% pixel(Cr, Ci, Ci2, Max, N): escape count with cardioid/bulb early-out.
pixel(Cr, Ci, Ci2, Max, N) :-
    Crm is Cr - 0.25,
    Q is Crm*Crm + Ci2,
    (   Q*(Q+Crm) =< 0.25*Ci2
    ->  N = Max
    ;   Cr1 is Cr + 1.0,
        (   Cr1*Cr1 + Ci2 =< 0.0625
        ->  N = Max
        ;   iter(0.0, 0.0, Cr, Ci, Max, Left),
            N is Max - Left
        )
    ).

% row_codes(X, Dx, Ci, Ci2, Max, Acc, Codes): bytes for pixels 0..X, uint16 LE.
% X counts down and prepends, so Codes come out in forward order.
row_codes(X, Dx, Ci, Ci2, Max, Acc, Codes) :-
    (   X < 0
    ->  Codes = Acc
    ;   Cr is -2.5 + X*Dx,
        pixel(Cr, Ci, Ci2, Max, N),
        Low is N /\ 0xff,
        High is N >> 8,
        X1 is X - 1,
        row_codes(X1, Dx, Ci, Ci2, Max, [Low, High|Acc], Codes)
    ).

worker(Jobs, Results) :-
    thread_get_message(Jobs, job(R, W1, Dx, Dy, Max)),
    Ci is -1.0 + R*Dy,
    Ci2 is Ci*Ci,
    row_codes(W1, Dx, Ci, Ci2, Max, [], Codes),
    string_codes(Str, Codes),
    thread_send_message(Results, row(R, Str)),
    worker(Jobs, Results).

frame(W, H, Max, Jobs, Results) :-
    Top is (H + 1) // 2,
    W1 is W - 1,
    Dx is 3.5 / max(1, W1),
    Dy is 2.0 / max(1, H - 1),
    Top1 is Top - 1,
    forall(between(0, Top1, R),
           thread_send_message(Jobs, job(R, W1, Dx, Dy, Max))),
    functor(Arr, rows, Top),
    forall(between(1, Top, _),
           (   thread_get_message(Results, row(R, Str)),
               I is R + 1,
               nb_setarg(I, Arr, Str)
           )),
    forall(between(1, Top, I),
           (   arg(I, Arr, Str),
               format(user_output, '~s', [Str])
           )),
    flush_output(user_output).

serve(Jobs, Results) :-
    read_line_to_string(user_input, Line),
    (   Line == end_of_file
    ->  true
    ;   split_string(Line, " ", " \r\t", Parts0),
        exclude(==(""), Parts0, [Ws, Hs, Is]),
        number_string(W, Ws),
        number_string(H, Hs),
        number_string(Max, Is),
        frame(W, H, Max, Jobs, Results),
        serve(Jobs, Results)
    ).

run :-
    set_stream(user_output, type(binary)),
    message_queue_create(Jobs),
    message_queue_create(Results),
    current_prolog_flag(cpu_count, NC0),
    NC is min(NC0, 16),
    forall(between(1, NC, _),
           thread_create(worker(Jobs, Results), _, [detached(true)])),
    serve(Jobs, Results).

main :-
    catch(run, E, (print_message(error, E), halt(1))),
    halt(0).
