*> CobolBrot - persistent Mandelbrot band worker (GnuCOBOL, free format).
*> Protocol: stdin line "w h max_iter r0 r1", stdout raw uint16 LE rows r0..r1-1.
*>
*> GnuCOBOL routes every float multiply through its GMP decimal runtime with
*> costly double<->decimal conversions, so the core uses Q28 fixed point on
*> native BINARY-DOUBLE fields instead: integer set_llint conversions are
*> cheap, IF field compares are native cob_cmp_s64, and Q28 resolution
*> (3.7e-9) is ~30x finer than the float32 other entries use.
*> Scale S = 2^28: 3.5S=939524096  2.5S=671088640  2S=536870912
*> 1S=268435456  4S=1073741824  0.25S=67108864  0.0625S=16777216  2^27=134217728
*> Cardioid + period-2 bulb early-outs; binary output via libc write(2) on fd 1.
IDENTIFICATION DIVISION.
PROGRAM-ID. cobolbrot.

ENVIRONMENT DIVISION.
INPUT-OUTPUT SECTION.
FILE-CONTROL.
    SELECT SIN ASSIGN TO "/dev/stdin"
        ORGANIZATION IS LINE SEQUENTIAL
        FILE STATUS IS WS-SIN-STAT.

DATA DIVISION.
FILE SECTION.
FD  SIN.
01  IN-REC               PIC X(100).

WORKING-STORAGE SECTION.
01  WS-SIN-STAT          PIC XX VALUE "00".
01  TOK1                 PIC X(16).
01  TOK2                 PIC X(16).
01  TOK3                 PIC X(16).
01  TOK4                 PIC X(16).
01  TOK5                 PIC X(16).
01  W                    USAGE BINARY-LONG.
01  H                    USAGE BINARY-LONG.
01  W1                   USAGE BINARY-LONG.
01  H1                   USAGE BINARY-LONG.
01  MAXIT                USAGE BINARY-LONG.
01  R0                   USAGE BINARY-LONG.
01  R1                   USAGE BINARY-LONG.
01  ROWI                 USAGE BINARY-LONG.
01  COLI                 USAGE BINARY-LONG.
01  IT                   USAGE BINARY-LONG.
*> Q28 fixed-point state (signed 64-bit native).
01  CRF                  USAGE BINARY-DOUBLE.
01  CIF                  USAGE BINARY-DOUBLE.
01  CI2F                 USAGE BINARY-DOUBLE.
01  RQF                  USAGE BINARY-DOUBLE.
01  ZR                   USAGE BINARY-DOUBLE.
01  ZI                   USAGE BINARY-DOUBLE.
01  ZR2                  USAGE BINARY-DOUBLE.
01  ZI2                  USAGE BINARY-DOUBLE.
01  NZR                  USAGE BINARY-DOUBLE.
01  CRMF                 USAGE BINARY-DOUBLE.
01  QF                   USAGE BINARY-DOUBLE.
01  LHF                  USAGE BINARY-DOUBLE.
01  T1F                  USAGE BINARY-DOUBLE.
01  TF                   USAGE BINARY-DOUBLE.
01  ROW-BUF.
    05 PX OCCURS 16384 TIMES USAGE BINARY-SHORT UNSIGNED.
01  ROW-BYTES REDEFINES ROW-BUF PIC X(32768).
01  NBYTES               USAGE BINARY-C-LONG.
01  WOFF                 USAGE BINARY-C-LONG.
01  WREM                 USAGE BINARY-C-LONG.
01  WRET                 USAGE BINARY-C-LONG.
01  FD-OUT               USAGE BINARY-LONG VALUE 1.

PROCEDURE DIVISION.
MAIN-LOOP.
    OPEN INPUT SIN
    PERFORM FOREVER
        READ SIN
            AT END EXIT PERFORM
        END-READ
        PERFORM PARSE-REQUEST
        PERFORM DO-BAND
    END-PERFORM
    CLOSE SIN
    GOBACK.

PARSE-REQUEST.
    MOVE SPACES TO TOK1 TOK2 TOK3 TOK4 TOK5
    UNSTRING IN-REC DELIMITED BY ALL " "
        INTO TOK1 TOK2 TOK3 TOK4 TOK5
    END-UNSTRING
    COMPUTE W     = FUNCTION NUMVAL(TOK1)
    COMPUTE H     = FUNCTION NUMVAL(TOK2)
    COMPUTE MAXIT = FUNCTION NUMVAL(TOK3)
    COMPUTE R0    = FUNCTION NUMVAL(TOK4)
    COMPUTE R1    = FUNCTION NUMVAL(TOK5).

DO-BAND.
    COMPUTE W1 = W - 1
    COMPUTE H1 = H - 1
    COMPUTE NBYTES = W * 2
    PERFORM VARYING ROWI FROM R0 BY 1 UNTIL ROWI >= R1
        *> ci = -1 + row * 2/(h-1)
        IF H1 > 0
            COMPUTE CIF = ROWI * 536870912 / H1 - 268435456
        ELSE
            MOVE -268435456 TO CIF
        END-IF
        COMPUTE CI2F = CIF * CIF / 268435456
        COMPUTE RQF = CI2F / 4
        PERFORM VARYING COLI FROM 0 BY 1 UNTIL COLI >= W
            *> cr = -2.5 + x * 3.5/(w-1)
            IF W1 > 0
                COMPUTE CRF = COLI * 939524096 / W1 - 671088640
            ELSE
                MOVE -671088640 TO CRF
            END-IF
            *> cardioid: q*(q+crm) <= 0.25*ci*ci
            COMPUTE CRMF = CRF - 67108864
            COMPUTE QF = CRMF * CRMF / 268435456 + CI2F
            COMPUTE LHF = QF * (QF + CRMF) / 268435456
            IF LHF <= RQF
                MOVE MAXIT TO IT
            ELSE
                *> period-2 bulb: (cr+1)^2 + ci*ci <= 0.0625
                COMPUTE T1F = CRF + 268435456
                COMPUTE TF = T1F * T1F / 268435456 + CI2F
                IF TF <= 16777216
                    MOVE MAXIT TO IT
                ELSE
                    PERFORM ESCAPE-LOOP
                END-IF
            END-IF
            MOVE IT TO PX(COLI + 1)
        END-PERFORM
        PERFORM WRITE-ROW
    END-PERFORM.

ESCAPE-LOOP.
    MOVE 0 TO IT
    MOVE 0 TO ZR
    MOVE 0 TO ZI
    PERFORM UNTIL IT >= MAXIT
        COMPUTE ZR2 = ZR * ZR / 268435456
        COMPUTE ZI2 = ZI * ZI / 268435456
        IF ZR2 + ZI2 > 1073741824
            EXIT PERFORM
        END-IF
        COMPUTE NZR = ZR2 - ZI2 + CRF
        COMPUTE ZI = ZR * ZI / 134217728 + CIF
        MOVE NZR TO ZR
        ADD 1 TO IT
    END-PERFORM.

WRITE-ROW.
    MOVE 0 TO WOFF
    PERFORM UNTIL WOFF >= NBYTES
        COMPUTE WREM = NBYTES - WOFF
        CALL STATIC "write" USING
            BY VALUE FD-OUT
            BY REFERENCE ROW-BYTES(WOFF + 1 : WREM)
            BY VALUE WREM
            RETURNING WRET
        END-CALL
        IF WRET <= 0
            STOP RUN
        END-IF
        ADD WRET TO WOFF
    END-PERFORM.
