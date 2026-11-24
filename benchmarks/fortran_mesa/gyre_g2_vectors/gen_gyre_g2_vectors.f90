! Program : gen_gyre_g2_vectors
! Purpose : exact IEEE-754 reference vectors for the FlexMESA port of the
!           real LAPACK/BLAS subset GYRE reaches (#170, package G2), from the
!           MESA SDK x86_64-linux-26.6.1 crmath-slot liblapack/libblas that
!           MESA 26.4.1 links.
!
! Usage: gen_gyre_g2_vectors <tapped_inputs.txt> <out.tsv>
!
! <tapped_inputs.txt> holds the LAPACK inputs GYRE passed in the G0 oracle
! calls (written by extract_tapped_inputs.py), one per line:
!   kind id m n nrhs hex...      (kind = GETRF | GESVX | GESVD)
! The generator replays them with GYRE's exact arguments and adds synthetic
! edge cases from a fixed xorshift64* sequence.
!
! Output records (tab separated): section id real_in int_in real_out int_out
! (reals as 16-digit hex bit patterns, '-' for an empty list).

module gen_g2_m

   use ISO_FORTRAN_ENV, only: INT64

   implicit none

   integer, parameter    :: RD = KIND(0.D0)
   integer, save         :: out_unit
   integer(INT64), save  :: rng_state = 88172645463325252_INT64

   interface
      double precision function dlamch(c)
         character, intent(in) :: c
      end function dlamch
      integer function ilaenv(ispec, name, opts, n1, n2, n3, n4)
         integer, intent(in)      :: ispec, n1, n2, n3, n4
         character(*), intent(in) :: name, opts
      end function ilaenv
      double precision function dnrm2(n, x, incx)
         integer, intent(in)          :: n, incx
         double precision, intent(in) :: x(*)
      end function dnrm2
      double precision function dlapy2(x, y)
         double precision, intent(in) :: x, y
      end function dlapy2
   end interface

contains

   function next_u64() result(r)
      integer(INT64) :: r
      rng_state = IEOR(rng_state, SHIFTR(rng_state, 12))
      rng_state = IEOR(rng_state, SHIFTL(rng_state, 25))
      rng_state = IEOR(rng_state, SHIFTR(rng_state, 27))
      r = rng_state*2685821657736338717_INT64
   end function next_u64

   function rnd_int(lo, hi) result(i)
      integer, intent(in) :: lo, hi
      integer             :: i
      i = lo + INT(MODULO(SHIFTR(next_u64(), 11), INT(hi-lo+1, INT64)))
   end function rnd_int

   function rnd_d(emin, emax) result(x)
      integer, intent(in) :: emin, emax
      real(RD)            :: x
      integer(INT64) :: bits
      bits = IOR(SHIFTL(INT(rnd_int(emin, emax) + 1023, INT64), 52), SHIFTR(next_u64(), 12))
      if (BTEST(next_u64(), 63)) bits = IBSET(bits, 63)
      x = TRANSFER(bits, x)
   end function rnd_d

   function hx(x) result(h)
      real(RD), intent(in) :: x
      character(16)        :: h
      write(h, '(Z16.16)') TRANSFER(x, 0_INT64)
   end function hx

   function rlist(v) result(s)
      real(RD), intent(in)      :: v(:)
      character(:), allocatable :: s
      integer :: i
      if (SIZE(v) == 0) then
         s = '-'
         return
      end if
      s = hx(v(1))
      do i = 2, SIZE(v)
         s = s // ' ' // hx(v(i))
      end do
   end function rlist

   function ilist(v) result(s)
      integer, intent(in)       :: v(:)
      character(:), allocatable :: s
      character(16) :: b
      integer :: i
      if (SIZE(v) == 0) then
         s = '-'
         return
      end if
      write(b, '(I0)') v(1)
      s = TRIM(b)
      do i = 2, SIZE(v)
         write(b, '(I0)') v(i)
         s = s // ' ' // TRIM(b)
      end do
   end function ilist

   subroutine wr(sec, id, rin, iin, rout, iout)
      character(*), intent(in) :: sec
      integer, intent(in)      :: id
      real(RD), intent(in)     :: rin(:), rout(:)
      integer, intent(in)      :: iin(:), iout(:)
      character(16) :: b
      write(b, '(I0)') id
      write(out_unit, '(A)') sec // CHAR(9) // TRIM(b) // CHAR(9) // rlist(rin) // CHAR(9) // &
         ilist(iin) // CHAR(9) // rlist(rout) // CHAR(9) // ilist(iout)
   end subroutine wr

   ! ---- drivers with GYRE's argument patterns

   ! matrix/block_sysmtx_m.fypp: XGETRF(m, n, M, m, ipiv, info)
   subroutine run_getrf(sec, id, m, n, a0)
      character(*), intent(in) :: sec
      integer, intent(in)      :: id, m, n
      real(RD), intent(in)     :: a0(:)
      real(RD) :: a(m, n)
      integer  :: ipiv(MIN(m, n)), info
      a = RESHAPE(a0, [m, n])
      call dgetrf(m, n, a, m, ipiv, info)
      call wr(sec, id, a0, [m, n], RESHAPE(a, [m*n]), [info, ipiv])
   end subroutine run_getrf

   ! matrix/linalg_m.fypp linear_solve_mat: XGESVX('E', 'N', n, nrhs, ...)
   ! with work(4n), iwork(n)
   subroutine run_gesvx(sec, id, n, nrhs, a0, b0)
      character(*), intent(in) :: sec
      integer, intent(in)      :: id, n, nrhs
      real(RD), intent(in)     :: a0(:), b0(:)
      real(RD)  :: a(n, n), af(n, n), b(n, nrhs), x(n, nrhs), r(n), c(n)
      real(RD)  :: ferr(nrhs), berr(nrhs), rcond, work(4*n)
      integer   :: ipiv(n), iwork(n), info
      character :: equed
      a = RESHAPE(a0, [n, n])
      b = RESHAPE(b0, [n, nrhs])
      r = 0._RD
      c = 0._RD
      call dgesvx('E', 'N', n, nrhs, a, n, af, n, ipiv, equed, r, c, b, n, x, n, &
         rcond, ferr, berr, work, iwork, info)
      call wr(sec, id, [a0, b0], [n, nrhs], &
         [RESHAPE(a, [n*n]), RESHAPE(af, [n*n]), r, c, RESHAPE(b, [n*nrhs]), &
          RESHAPE(x, [n*nrhs]), rcond, ferr, berr, work(1)], [info, ICHAR(equed), ipiv])
   end subroutine run_gesvx

   ! matrix/linalg_m.fypp sing_decompose_r_: XGESVD('A','A', n, n, A, n,
   ! sigma, U, n, V_H, n, work, 5n, info)
   subroutine run_gesvd(sec, id, n, a0)
      character(*), intent(in) :: sec
      integer, intent(in)      :: id, n
      real(RD), intent(in)     :: a0(:)
      real(RD) :: a(n, n), s(n), u(n, n), vt(n, n), work(5*n)
      integer  :: info
      a = RESHAPE(a0, [n, n])
      call dgesvd('A', 'A', n, n, a, n, s, u, n, vt, n, work, 5*n, info)
      call wr(sec, id, a0, [n], [RESHAPE(a, [n*n]), s, RESHAPE(u, [n*n]), RESHAPE(vt, [n*n]), &
         work(1)], [info])
   end subroutine run_gesvd

end module gen_g2_m

program gen_gyre_g2_vectors

   use gen_g2_m

   implicit none

   character(4096) :: in_file, out_file, line
   character(8)    :: kind
   integer         :: unit, ios, id, m, n, nrhs, i, j, k, cnt
   real(RD), allocatable :: v(:), a0(:), b0(:)
   integer(INT64), allocatable :: bits(:)
   real(RD) :: x, y, z, f, g, h, alpha, tau, ssmin, ssmax, snr, csr, snl, csl, cs, sn, rr
   real(RD) :: xv(12)
   integer  :: e_lo, e_hi

   if (COMMAND_ARGUMENT_COUNT() /= 2) stop 'usage: gen_gyre_g2_vectors tapped_inputs.txt out.tsv'
   call GET_COMMAND_ARGUMENT(1, in_file)
   call GET_COMMAND_ARGUMENT(2, out_file)
   open(NEWUNIT=out_unit, FILE=TRIM(out_file), STATUS='REPLACE', ACTION='WRITE')

   ! ---- constants

   call wr('dlamch', 1, [real(RD)::], [integer::], [dlamch('E'), dlamch('S'), dlamch('B'), &
      dlamch('P'), dlamch('O'), dlamch('U')], [integer::])
   call wr('ilaenv', 1, [real(RD)::], [integer::], [real(RD)::], &
      [ilaenv(1, 'DGETRF', ' ', 8, 8, -1, -1), ilaenv(1, 'DGEQRF', ' ', 8, 8, -1, -1), &
       ilaenv(1, 'DGEBRD', ' ', 8, 8, -1, -1), ilaenv(1, 'DORGQR', ' ', 8, 8, 8, -1), &
       ilaenv(1, 'DORGLQ', ' ', 8, 8, 8, -1), (ilaenv(6, 'DGESVD', 'AA', k, k, 0, 0), k = 1, 12)])

   ! ---- leaf routines with edge cases

   do i = 1, 300
      n = rnd_int(1, 12)
      e_lo = MERGE(-1074, -60, MOD(i, 7) == 0)
      e_hi = MERGE(1020, 60, MOD(i, 11) == 0)
      do j = 1, n
         xv(j) = rnd_d(MAX(e_lo, -1022), e_hi)
      end do
      if (MOD(i, 13) == 0) xv(1:n) = xv(1:n)*2._RD**(-1060)    ! subnormal range
      call wr('dnrm2', i, xv(1:n), [n], [dnrm2(n, xv, 1)], [integer::])

      f = rnd_d(-300, 300); g = rnd_d(-300, 300); h = rnd_d(-300, 300)
      if (i <= 6) then
         select case (i)
         case (1); g = 0._RD
         case (2); f = 0._RD
         case (3); h = 0._RD
         case (4); f = 0._RD; h = 0._RD
         case (5); g = f*2._RD**60
         case (6); f = h
         end select
      end if
      if (MOD(i, 5) == 0) g = f*2._RD**(-rnd_int(1, 80))
      if (MOD(i, 9) == 0) h = f*2._RD**(-rnd_int(1, 80))
      call dlartg(f, g, cs, sn, rr)
      call wr('dlartg', i, [f, g], [integer::], [cs, sn, rr], [integer::])
      call dlas2(f, g, h, ssmin, ssmax)
      call wr('dlas2', i, [f, g, h], [integer::], [ssmin, ssmax], [integer::])
      call dlasv2(f, g, h, ssmin, ssmax, snr, csr, snl, csl)
      call wr('dlasv2', i, [f, g, h], [integer::], [ssmin, ssmax, snr, csr, snl, csl], [integer::])
      call wr('dlapy2', i, [f, g], [integer::], [dlapy2(f, g)], [integer::])

      n = rnd_int(1, 12)
      alpha = rnd_d(-30, 30)
      do j = 1, n - 1
         xv(j) = rnd_d(-30, 30)
      end do
      if (MOD(i, 8) == 0) then        ! tiny vector: exercises the rescaling loop
         alpha = alpha*2._RD**(-1040)
         xv(1:n-1) = xv(1:n-1)*2._RD**(-1040)
      end if
      if (MOD(i, 10) == 0) xv(1:n-1) = 0._RD
      allocate(a0(n))
      a0 = [alpha, xv(1:n-1)]
      call dlarfg(n, alpha, xv, 1, tau)
      call wr('dlarfg', i, a0, [n], [alpha, tau, xv(1:n-1)], [integer::])
      deallocate(a0)
   end do

   ! ---- replay of GYRE's tapped calls

   open(NEWUNIT=unit, FILE=TRIM(in_file), STATUS='OLD', ACTION='READ')
   do
      read(unit, '(A)', IOSTAT=ios) line
      if (ios /= 0) exit
      read(line, *) kind, id, m, n, nrhs
      cnt = m*n + MERGE(n*nrhs, 0, TRIM(kind) == 'GESVX')
      allocate(bits(cnt), v(cnt))
      read(line, *) kind, id, m, n, nrhs, (bits(k), k = 1, cnt)
      v = TRANSFER(bits, v)
      select case (TRIM(kind))
      case ('GETRF')
         call run_getrf('getrf_tap', id, m, n, v)
      case ('GESVX')
         call run_gesvx('gesvx_tap', id, n, nrhs, v(1:n*n), v(n*n+1:))
      case ('GESVD')
         call run_gesvd('gesvd_tap', id, n, v)
      end select
      deallocate(bits, v)
   end do
   close(unit)

   ! ---- synthetic cases

   do i = 1, 200
      m = rnd_int(1, 12); n = rnd_int(1, 12)
      if (MOD(i, 3) == 0) m = 2*n          ! GYRE's cyclic-reduction shape
      allocate(a0(m*n))
      do k = 1, m*n
         a0(k) = rnd_d(-20, 20)
      end do
      if (MOD(i, 10) == 0) a0(1:m) = 0._RD                         ! zero column: INFO > 0
      if (MOD(i, 17) == 0) a0 = a0*2._RD**(-1030)                   ! pivots below SFMIN
      if (MOD(i, 7) == 0 .AND. n >= 2) a0(m+1:2*m) = a0(1:m)        ! rank deficient
      call run_getrf('getrf_syn', i, m, n, a0)
      deallocate(a0)
   end do

   do i = 1, 200
      n = rnd_int(1, 12); nrhs = rnd_int(1, 6)
      allocate(a0(n*n), b0(n*nrhs))
      do k = 1, n*n
         a0(k) = rnd_d(-6, 6)
      end do
      do k = 1, n*nrhs
         b0(k) = rnd_d(-6, 6)
      end do
      select case (MOD(i, 8))
      case (1)       ! badly scaled rows -> EQUED = 'R'
         do j = 1, n
            a0(j:n*n:n) = a0(j:n*n:n)*2._RD**rnd_int(-40, 40)
         end do
      case (2)       ! badly scaled columns -> EQUED = 'C'
         do j = 1, n
            a0((j-1)*n+1:j*n) = a0((j-1)*n+1:j*n)*2._RD**rnd_int(-40, 40)
         end do
      case (3)       ! both -> EQUED = 'B'
         do j = 1, n
            a0(j:n*n:n) = a0(j:n*n:n)*2._RD**rnd_int(-40, 40)
            a0((j-1)*n+1:j*n) = a0((j-1)*n+1:j*n)*2._RD**rnd_int(-40, 40)
         end do
      case (4)       ! nearly singular -> RCOND < EPS (INFO = N+1)
         if (n >= 2) a0(n+1:2*n) = a0(1:n)*(1._RD + 2._RD**(-60))
      case (5)       ! exactly singular after elimination -> INFO > 0
         if (n >= 2) a0(n+1:2*n) = a0(1:n)
      end select
      call run_gesvx('gesvx_syn', i, n, nrhs, a0, b0)
      deallocate(a0, b0)
   end do

   do i = 1, 200
      n = rnd_int(2, 12)
      allocate(a0(n*n))
      do k = 1, n*n
         a0(k) = rnd_d(-20, 20)
      end do
      select case (MOD(i, 6))
      case (1); if (n >= 2) a0(n+1:2*n) = a0(1:n)                ! rank deficient
      case (2)                                                     ! diagonal
         a0 = 0._RD
         do j = 1, n
            a0((j-1)*n+j) = rnd_d(-20, 20)
         end do
      case (3)                                                     ! repeated singular values
         a0 = 0._RD
         do j = 1, n
            a0((j-1)*n+j) = MERGE(2._RD, -2._RD, MOD(j, 2) == 0)
         end do
         a0(n) = 1e-3_RD
      case (4); a0 = a0*2._RD**rnd_int(-200, 200)                  ! scaled within range
      end select
      call run_gesvd('gesvd_syn', i, n, a0)
      deallocate(a0)
   end do

   close(out_unit)

end program gen_gyre_g2_vectors
