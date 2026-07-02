⍝ aplbrot.apl - GNU APL Mandelbrot worker (persistent, raw uint16 LE over stdout)
⍝ Protocol: read a request line "w h mi a b" from stdin (rows a..b-1 of the
⍝ top half), compute the whole (b-a, w) band with vectorized array ops, write
⍝ it back as raw little-endian uint16 bytes, flush, repeat until EOF.
⍝ Whole-grid escape-time: no per-pixel loop, only a loop over the <=mi steps,
⍝ each step operating on the full complex matrix under a boolean "alive" mask.

⎕IO←1

∇BAND←BANDFN ARGS;W;H;MI;A;B;NB;DX;DY;CR;CI;ZR;ZI;COUNT;ALIVE;K;ZR2;ZI2;AL2;NZI;CRM;Q;C1;C2;INSET;VALS;LO;HI
 W←ARGS[1] ⋄ H←ARGS[2] ⋄ MI←ARGS[3] ⋄ A←ARGS[4] ⋄ B←ARGS[5]
 NB←B-A
 DX←3.5÷W-1 ⋄ DY←2.0÷H-1
 ⍝ CR[j;x]=cr[x] ; CI[j;x]=ci[j]
 CR←(NB,W)⍴¯2.5+DX×¯1+⍳W
 CI←(¯1.0+DY×(A-1)+⍳NB)∘.+W⍴0
 ZR←(NB,W)⍴0 ⋄ ZI←(NB,W)⍴0
 COUNT←(NB,W)⍴0
 ALIVE←(NB,W)⍴1
 K←0
LOOP:→(K≥MI)/DONE
 ZR2←ZR×ZR ⋄ ZI2←ZI×ZI
 AL2←ALIVE∧(ZR2+ZI2)≤4
 COUNT←COUNT+AL2
 NZI←(2×ZR×ZI)+CI
 ZR←AL2×((ZR2-ZI2)+CR)
 ZI←AL2×NZI
 ALIVE←AL2
 K←K+1
 →LOOP
DONE:
 ⍝ cardioid / period-2 bulb early-out: force result = MI
 CRM←CR-0.25
 Q←(CRM×CRM)+CI×CI
 C1←(Q×Q+CRM)≤0.25×CI×CI
 C2←(((CR+1)×(CR+1))+CI×CI)≤0.0625
 INSET←C1∨C2
 COUNT←(INSET×MI)+(~INSET)×COUNT
 ⍝ ravel row-major, split into LE byte pairs
 VALS←,COUNT
 LO←256|VALS
 HI←⌊VALS÷256
 BAND←,⍉(2,≢VALS)⍴LO,HI
∇

∇MAIN;REQ;R
LP:REQ←⎕FIO[8] 0
 →(0=≢REQ)/END
 R←BANDFN ⍎⎕UCS REQ
 R←R ⎕FIO[7] 1
 R←⎕FIO[16] 1
 →LP
END:
∇

MAIN
)OFF
