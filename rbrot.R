# RBrot - persistent Mandelbrot worker (GNU R, whole-grid vectorization).
# Protocol: stdin line "<width> <height> <max_iter>", stdout exactly
# width*height*2 bytes of uint16 little-endian iteration counts, row-major.
# Loops until EOF. Diagnostics (if any) go to stderr only.

compute <- function(width, height, max_iter) {
  dx <- 3.5 / (width - 1)
  dy <- 2.0 / (height - 1)
  top <- (height + 1L) %/% 2L

  # Flat vectors in image-row-major order: index = x + width*row.
  crv <- -2.5 + (0:(width - 1L)) * dx
  civ <- -1.0 + (0:(top - 1L)) * dy
  n <- width * top
  cr <- rep(crv, times = top)
  ci <- rep(civ, each = width)

  counts <- rep.int(as.integer(max_iter), n)

  # Cardioid + period-2 bulb: in-set, never iterated.
  crm <- cr - 0.25
  ci2 <- ci * ci
  q <- crm * crm + ci2
  inset <- (q * (q + crm) <= 0.25 * ci2) | ((cr + 1)^2 + ci2 <= 0.0625)

  idx <- which(!inset)
  m <- length(idx)
  if (m > 0L && max_iter > 1L) {
    cra <- cr[idx]
    cia <- ci[idx]
    zr <- numeric(m)
    zi <- numeric(m)
    alive <- rep(TRUE, m)
    for (k in 1:(max_iter - 1L)) {
      # Whole-vector update; escaped cells keep churning until compaction.
      zr2 <- zr * zr
      zi2 <- zi * zi
      zi <- 2 * zr * zi + cia
      zr <- zr2 - zi2 + cra
      esc <- alive & (zr * zr + zi * zi > 4)  # FALSE & NA == FALSE: NaN-safe
      if (any(esc)) {
        counts[idx[esc]] <- k
        alive <- alive & !esc
      }
      if (k %% 8L == 0L) {  # periodic compaction: drop escaped cells
        keep <- which(alive)
        if (length(keep) == 0L) break
        if (length(keep) < m) {
          idx <- idx[keep]
          cra <- cra[keep]
          cia <- cia[keep]
          zr <- zr[keep]
          zi <- zi[keep]
          m <- length(keep)
          alive <- rep(TRUE, m)
        }
      }
    }
  }
  counts
}

main <- function() {
  fin <- file("stdin", "r")
  fout <- file("/dev/stdout", "wb")  # file("stdout","wb") writes nothing
  repeat {
    line <- readLines(fin, n = 1L)
    if (length(line) == 0L) break  # EOF
    v <- as.integer(strsplit(trimws(line), "[[:space:]]+")[[1L]])
    w <- v[1L]
    h <- v[2L]
    it <- v[3L]
    counts <- compute(w, h, it)
    top <- (h + 1L) %/% 2L
    mat <- matrix(counts, nrow = w)  # column j = image row j-1
    if (top < h) mat <- cbind(mat, mat[, (h - top):1L, drop = FALSE])
    writeBin(as.integer(mat), fout, size = 2L, endian = "little")
    flush(fout)
  }
  close(fout)
  close(fin)
}

main()
