(* OcamlBrot - persistent Mandelbrot worker, OCaml 5 multicore Domains.
   Protocol: read "w h iters" line on stdin, reply with w*h uint16 LE
   pixels row-major on stdout. Loops until EOF. *)

type job = {
  w : int;
  iter : int;
  half : int;
  dx : float;
  dy : float;
  buf : Bytes.t;
}

let job = ref { w = 0; iter = 0; half = 0; dx = 0.0; dy = 0.0; buf = Bytes.empty }
let next_row = Atomic.make 0
let start_sem = Semaphore.Counting.make 0
let done_sem = Semaphore.Counting.make 0
let quit = Atomic.make false

let compute_row j row =
  let w = j.w and max_iter = j.iter and buf = j.buf in
  let ci = -1.0 +. float_of_int row *. j.dy in
  let ci2 = ci *. ci in
  let base = row * w * 2 in
  for x = 0 to w - 1 do
    let cr = -2.5 +. float_of_int x *. j.dx in
    let crm = cr -. 0.25 in
    let q = crm *. crm +. ci2 in
    let cr1 = cr +. 1.0 in
    let it =
      (* cardioid + period-2 bulb early-out *)
      if q *. (q +. crm) <= 0.25 *. ci2 || cr1 *. cr1 +. ci2 <= 0.0625 then
        max_iter
      else begin
        let zr = ref 0.0 and zi = ref 0.0 in
        let it = ref 0 and live = ref true in
        while !live && !it < max_iter do
          let zr2 = !zr *. !zr and zi2 = !zi *. !zi in
          if zr2 +. zi2 > 4.0 then live := false
          else begin
            zi := 2.0 *. !zr *. !zi +. ci;
            zr := zr2 -. zi2 +. cr;
            incr it
          end
        done;
        !it
      end
    in
    Bytes.set_uint16_le buf (base + x * 2) it
  done

(* Each domain waits for a start token, steals rows via the atomic counter,
   then posts a done token. Tokens balance per frame, so it does not matter
   which physical domain services which frame. *)
let worker () =
  let running = ref true in
  while !running do
    Semaphore.Counting.acquire start_sem;
    if Atomic.get quit then running := false
    else begin
      let j = !job in
      let more = ref true in
      while !more do
        let row = Atomic.fetch_and_add next_row 1 in
        if row < j.half then compute_row j row else more := false
      done;
      Semaphore.Counting.release done_sem
    end
  done

let () =
  let n = max 1 (Domain.recommended_domain_count ()) in
  let domains = List.init n (fun _ -> Domain.spawn worker) in
  (try
     while true do
       let line = input_line stdin in
       match String.split_on_char ' ' (String.trim line) with
       | [ ws; hs; is ] ->
           let w = int_of_string ws
           and h = int_of_string hs
           and iter = int_of_string is in
           let buf = Bytes.create (w * h * 2) in
           let half = (h + 1) / 2 in
           job :=
             { w; iter; half;
               dx = 3.5 /. float_of_int (w - 1);
               dy = 2.0 /. float_of_int (h - 1);
               buf };
           Atomic.set next_row 0;
           for _ = 1 to n do Semaphore.Counting.release start_sem done;
           for _ = 1 to n do Semaphore.Counting.acquire done_sem done;
           (* y-axis symmetry: mirror computed top rows onto the bottom *)
           let rw = w * 2 in
           for r = 0 to half - 1 do
             let dst = h - 1 - r in
             if dst <> r then Bytes.blit buf (r * rw) buf (dst * rw) rw
           done;
           output_bytes stdout buf;
           flush stdout
       | _ -> ()
     done
   with End_of_file -> ());
  Atomic.set quit true;
  for _ = 1 to n do Semaphore.Counting.release start_sem done;
  List.iter Domain.join domains
