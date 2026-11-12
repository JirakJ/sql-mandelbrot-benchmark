{ PascalBrot - float32 scalar Mandelbrot kernel for the benchmark suite.
  Persistent thread pool (one per logical CPU) pulls rows off an atomic
  counter; y-axis symmetry halves the work; cardioid/bulb early-outs.
  Built as a dylib, called from Python via ctypes. }
library pascalbrot;

{$mode objfpc}
{$POINTERMATH ON}
{$inline on}

uses
  cthreads;

const
  MAXTHREADS = 64;

function sysctlbyname(name: PAnsiChar; oldp: Pointer; var oldlen: PtrUInt;
  newp: Pointer; newlen: PtrUInt): LongInt; cdecl; external 'c' name 'sysctlbyname';

var
  gW, gH, gIt, gHalfRows: LongInt;
  gDx, gDy: Single;
  gOut: PWord;
  gRowCounter: LongInt;
  gActive: LongInt;
  gStart: array[0..MAXTHREADS - 1] of PRTLEvent;
  gDone: PRTLEvent;
  gWorkers: LongInt = -1; { -1 = pool not created yet }

procedure ComputeRow(row: LongInt);
var
  x, it: LongInt;
  cr, ci, zr, zi, zr2, zi2, crm, q, t: Single;
  p, m: PWord;
begin
  ci := -1.0 + row * gDy;
  p := gOut + PtrUInt(row) * PtrUInt(gW);
  for x := 0 to gW - 1 do
  begin
    cr := -2.5 + x * gDx;
    { cardioid + period-2 bulb early-outs }
    crm := cr - 0.25;
    q := crm * crm + ci * ci;
    t := cr + 1.0;
    if (q * (q + crm) <= 0.25 * (ci * ci)) or (t * t + ci * ci <= 0.0625) then
      p[x] := Word(gIt)
    else
    begin
      zr := 0.0; zi := 0.0; it := 0;
      while it < gIt do
      begin
        zr2 := zr * zr;
        zi2 := zi * zi;
        if zr2 + zi2 > 4.0 then break;
        zi := 2.0 * zr * zi + ci;
        zr := zr2 - zi2 + cr;
        Inc(it);
      end;
      p[x] := Word(it);
    end;
  end;
  { mirror across the y axis }
  m := gOut + PtrUInt(gH - 1 - row) * PtrUInt(gW);
  if m <> p then
    Move(p^, m^, gW * SizeOf(Word));
end;

procedure DrainRows;
var
  r: LongInt;
begin
  repeat
    r := InterlockedIncrement(gRowCounter);
    if r >= gHalfRows then
      break;
    ComputeRow(r);
  until False;
end;

function WorkerMain(param: Pointer): PtrInt;
var
  idx: LongInt;
begin
  idx := LongInt(PtrInt(param));
  repeat
    RTLEventWaitFor(gStart[idx]);
    DrainRows;
    if InterlockedDecrement(gActive) = 0 then
      RTLEventSetEvent(gDone);
  until False;
  Result := 0;
end;

procedure EnsurePool;
var
  n, i: LongInt;
  len: PtrUInt;
begin
  if gWorkers >= 0 then
    exit;
  n := 0;
  len := SizeOf(n);
  if (sysctlbyname('hw.logicalcpu', @n, len, nil, 0) <> 0) or (n < 1) then
    n := 8;
  if n > MAXTHREADS then
    n := MAXTHREADS;
  gWorkers := n - 1; { calling thread participates too }
  gDone := RTLEventCreate;
  for i := 0 to gWorkers - 1 do
  begin
    gStart[i] := RTLEventCreate;
    BeginThread(@WorkerMain, Pointer(PtrInt(i)));
  end;
end;

procedure mandelbrot_pascal(w, h, it: LongInt; out_: PWord); cdecl;
var
  i: LongInt;
begin
  if (w < 1) or (h < 1) or (out_ = nil) then
    exit;
  EnsurePool;
  gW := w; gH := h; gIt := it; gOut := out_;
  if w > 1 then gDx := 3.5 / (w - 1) else gDx := 0.0;
  if h > 1 then gDy := 2.0 / (h - 1) else gDy := 0.0;
  gHalfRows := (h + 1) div 2;
  gRowCounter := -1;
  if gWorkers > 0 then
  begin
    gActive := gWorkers;
    for i := 0 to gWorkers - 1 do
      RTLEventSetEvent(gStart[i]);
    DrainRows;
    RTLEventWaitFor(gDone);
  end
  else
    DrainRows;
end;

exports
  mandelbrot_pascal;

begin
end.
