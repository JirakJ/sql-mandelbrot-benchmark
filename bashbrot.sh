#!/bin/bash
# bashbrot.sh - persistent Mandelbrot worker in pure Bash (3.2+) integer math.
#
# Q26 fixed point: bash arithmetic is 64-bit signed; |zr| <= 6.5 entering the
# multiply, so zr*zr <= 42.25*2^52 < 2^63. Right shift of the (possibly
# negative) cross term is arithmetic (floor) - consistent fixed-point rounding.
#
# The escape loop lives entirely in one (( )) while-condition with single-char
# variables and literal shift counts: bash re-parses arithmetic text on every
# evaluation (even short-circuited parts), so fewer characters = faster, and
# unrolling does not help. Measured: literals beat variable lookups.
#
# Protocol (one band per line on stdin, loops until EOF):
#   "w h m start stride top"
# Computes rows start, start+stride, ... < top of the full image and prints
# one text line per row: space-separated decimal survived-iteration counts
# (in-set pixels = m). stdout only carries pixel data.
#
# Vars: a=zr b=zi c=ci d=cr p=zr2 q=zi2 i=it m=max_iter.

F=$((4 << 26))    # escape bound 4.0
Q=$((1 << 26))    # 1.0
QC=$((1 << 24))   # 0.25
B=$((1 << 22))    # 0.0625 (period-2 bulb radius^2)
C7=$((7 << 26))   # cr = (7x + w-1)/(2(w-1)) - 2.5, rounded to nearest
C25=$((5 << 25))  # 2.5

while read -r w h m start stride top; do
  wm1=$((w - 1)); hm1=$((h - 1)); den=$((2 * wm1)); dci=$((2 * hm1))
  r=$start
  while ((r < top)); do
    # ci = -1 + r*2/(h-1), rounded to nearest Q26 (r<<28 == r*2*2^26*2)
    ((c = ((r << 28) + hm1) / dci - Q, c2 = c * c >> 26, c2q = c2 >> 2))
    orow=""
    x=0
    while ((x < w)); do
      # cr, then cardioid (q(q+crm) <= ci^2/4) and period-2 bulb early-out
      if ((d = (C7 * x + wm1) / den - C25, u = d - QC, k = (u * u >> 26) + c2, t = d + Q,
           (k * (k + u) >> 26) <= c2q || (t * t >> 26) + c2 <= B)); then
        orow+=" $m"
      else
        ((a = 0, b = 0, i = 0))
        while ((p=a*a>>26,q=b*b>>26,p+q<=F&&i<m&&(b=(a*b>>25)+c,a=p-q+d,++i))); do :; done
        orow+=" $i"
      fi
      ((++x))
    done
    printf '%s\n' "${orow# }"
    ((r += stride))
  done
done
