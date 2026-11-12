! FortranBrot - scalar real32 Mandelbrot kernel, OpenMP across rows.
! Bare subroutine (no module) so gfortran emits no .mod file.
! out is flat row-major from the C caller: index = row*w + x + 1.
subroutine mandelbrot_fortran(w, h, maxit, out) bind(c, name="mandelbrot_fortran")
  use iso_c_binding
  implicit none
  integer(c_int), value :: w, h, maxit
  integer(c_int16_t) :: out(*)

  real(c_float) :: dx, dy, cr, ci, zr, zi, zr2, zi2, crm, q
  integer :: x, row, it, half
  integer(c_int64_t) :: base, mbase

  dx = 3.5_c_float / real(w - 1, c_float)
  dy = 2.0_c_float / real(h - 1, c_float)
  half = (h + 1) / 2

  !$omp parallel do schedule(dynamic, 4) default(none) shared(w, h, maxit, out, dx, dy, half) &
  !$omp& private(x, it, cr, ci, zr, zi, zr2, zi2, crm, q, base)
  do row = 0, half - 1
    ci = -1.0_c_float + real(row, c_float) * dy
    base = int(row, c_int64_t) * int(w, c_int64_t)
    do x = 0, w - 1
      cr = -2.5_c_float + real(x, c_float) * dx
      ! cardioid + period-2 bulb early-out
      crm = cr - 0.25_c_float
      q = crm * crm + ci * ci
      if (q * (q + crm) <= 0.25_c_float * ci * ci .or. &
          (cr + 1.0_c_float)**2 + ci * ci <= 0.0625_c_float) then
        out(base + x + 1) = int(maxit, c_int16_t)
        cycle
      end if
      zr = 0.0_c_float
      zi = 0.0_c_float
      it = 0
      do while (it < maxit)
        zr2 = zr * zr
        zi2 = zi * zi
        if (zr2 + zi2 > 4.0_c_float) exit
        zi = 2.0_c_float * zr * zi + ci      ! uses old zr
        zr = zr2 - zi2 + cr
        it = it + 1
      end do
      out(base + x + 1) = int(it, c_int16_t)
    end do
  end do
  !$omp end parallel do

  ! y-axis symmetry: mirror computed rows; middle row of odd h maps to itself.
  do row = 0, half - 1
    if (h - 1 - row /= row) then
      base = int(row, c_int64_t) * int(w, c_int64_t)
      mbase = int(h - 1 - row, c_int64_t) * int(w, c_int64_t)
      out(mbase + 1 : mbase + w) = out(base + 1 : base + w)
    end if
  end do
end subroutine mandelbrot_fortran
