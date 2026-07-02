%% Minimal FFI for gleambrot: binary stdio + parallel band map.
%% All compute stays in Gleam; this only moves bytes and spawns processes.
-module(gleambrot_ffi).
-export([set_binary_io/0, read_request/0, write_frame/1, halt_now/0,
         schedulers/0, pmap/2]).

set_binary_io() ->
    %% Raw byte mode: unicode mode would UTF-8-expand bytes >= 16#80.
    ok = io:setopts(standard_io, [binary, {encoding, latin1}]),
    nil.

read_request() ->
    case io:get_line(standard_io, "") of
        eof -> {error, nil};
        {error, _} -> {error, nil};
        Line ->
            [W, H, It | _] =
                [binary_to_integer(T)
                 || T <- binary:split(Line, [<<" ">>, <<"\n">>, <<"\r">>],
                                      [global, trim_all])],
            {ok, {W, H, It}}
    end.

write_frame(Rows) ->
    ok = file:write(standard_io, Rows),
    nil.

halt_now() ->
    erlang:halt(0).

schedulers() ->
    erlang:system_info(schedulers_online).

%% Order-preserving parallel map; one linked process per item.
pmap(Items, F) ->
    Parent = self(),
    Refs = [begin
                Ref = make_ref(),
                spawn_opt(fun() -> Parent ! {Ref, F(Item)} end,
                          [link, {min_heap_size, 65536}]),
                Ref
            end || Item <- Items],
    [receive {Ref, R} -> R end || Ref <- Refs].
