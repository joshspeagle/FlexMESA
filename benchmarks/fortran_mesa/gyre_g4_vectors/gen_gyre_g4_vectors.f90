! Program : gen_gyre_g4_vectors
! Purpose : exact IEEE-754 reference vectors for the FlexMESA port of the
!           complex LAPACK/BLAS subset GYRE's nonadiabatic path reaches
!           (#170, package G4), from the MESA SDK x86_64-linux-26.6.1
!           crmath-slot liblapack/libblas that MESA 26.4.1 links.
!
! Usage: gen_gyre_g4_vectors <tapped_inputs.txt> <out.tsv>
!
! <tapped_inputs.txt> holds the complex inputs extracted from the G0 oracle
! calls (extract_tapped_inputs.py), one per line:
!   kind id m n bits...      (kind = GESVD | FACTOR | GEMV)
! with complex matrices column-major as (re, im) pairs. The generator
! replays them with GYRE's exact argument patterns and adds synthetic edge
! cases from a fixed xorshift64* sequence.
!
! Output records (tab separated): section id real_in int_in real_out int_out
! (reals as 16-digit hex bit patterns, complex values as (re, im) pairs,
! '-' for an empty list).

module gen_g4_m

   use ISO_FORTRAN_ENV, only: INT64

   implicit none

   integer, parameter    :: RD = KIND(0.D0)
   complex(RD), parameter :: CZERO = (0._RD, 0._RD)
   complex(RD), parameter :: CONE = (1._RD, 0._RD)
   integer, save         :: out_unit
   integer(INT64), save  :: rng_state = 88172645463325252_INT64

   interface
      double precision function dlamch(c)
         character, intent(in) :: c
      end function dlamch
      double precision function dznrm2(n, x, incx)
         integer, intent(in)     :: n, incx
         complex(KIND(0.D0)), intent(in) :: x(*)
      end function dznrm2
      double precision function dlapy3(x, y, z)
         double precision, intent(in) :: x, y, z
      end function dlapy3
      complex(KIND(0.D0)) function zladiv(x, y)
         complex(KIND(0.D0)), intent(in) :: x, y
      end function zladiv
      integer function izamax(n, x, incx)
         integer, intent(in)     :: n, incx
         complex(KIND(0.D0)), intent(in) :: x(*)
      end function izamax
      double precision function zlange(norm, m, n, a, lda, work)
         character, intent(in)   :: norm
         integer, intent(in)     :: m, n, lda
         complex(KIND(0.D0)), intent(in) :: a(lda, *)
         double precision, intent(inout) :: work(*)
      end function zlange
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

   function rnd_c(emin, emax) result(z)
      integer, intent(in) :: emin, emax
      complex(RD)         :: z
      real(RD) :: re, im
      re = rnd_d(emin, emax)
      im = rnd_d(emin, emax)
      z = CMPLX(re, im, RD)
   end function rnd_c

   subroutine rnd_mat(a, emin, emax)
      complex(RD), intent(out) :: a(:)
      integer, intent(in)      :: emin, emax
      integer :: k
      do k = 1, SIZE(a)
         a(k) = rnd_c(emin, emax)
      end do
   end subroutine rnd_mat

   pure function c2r(z) result(r)
      complex(RD), intent(in) :: z(:)
      real(RD)                :: r(2*SIZE(z))
      r(1::2) = REAL(z, RD)
      r(2::2) = AIMAG(z)
   end function c2r

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

   ! block_sysmtx_m.fypp: XGETRF(2n_e, n_e, M_U, 2n_e, ...) and
   ! XGETRF(2n_e, 2n_e, M, 2n_e, ...); synthetic shapes use lda = m
   subroutine run_getrf(sec, id, m, n, a0)
      character(*), intent(in) :: sec
      integer, intent(in)      :: id, m, n
      complex(RD), intent(in)  :: a0(:)
      complex(RD) :: a(m, n)
      integer     :: ipiv(MIN(m, n)), info
      a = RESHAPE(a0, [m, n])
      call zgetrf(m, n, a, m, ipiv, info)
      call wr(sec, id, c2r(a0), [m, n], c2r(RESHAPE(a, [m*n])), [info, ipiv])
   end subroutine run_getrf

   ! matrix/linalg_m.fypp sing_decompose_c_: XGESVD('A','A', n, n, A, n,
   ! sigma, U, n, V_H, n, work(3n), 3n, rwork(5n), info); for m > n the
   ! minimal workspace 2n+m
   subroutine run_gesvd(sec, id, m, n, a0)
      character(*), intent(in) :: sec
      integer, intent(in)      :: id, m, n
      complex(RD), intent(in)  :: a0(:)
      complex(RD) :: a(m, n), u(m, m), vt(n, n), work(2*n+m)
      real(RD)    :: s(n), rwork(5*n)
      integer     :: info
      a = RESHAPE(a0, [m, n])
      call zgesvd('A', 'A', m, n, a, m, s, u, m, vt, n, work, 2*n+m, rwork, info)
      call wr(sec, id, c2r(a0), [m, n], [c2r(RESHAPE(a, [m*n])), s, c2r(RESHAPE(u, [m*m])), &
         c2r(RESHAPE(vt, [n*n])), c2r(work(1:1))], [info])
   end subroutine run_gesvd

   ! block_sysmtx_m.fypp factor_c_ reduction step (Wright 1994)
   subroutine run_factor(sec, id, ne, v)
      character(*), intent(in) :: sec
      integer, intent(in)      :: id, ne
      complex(RD), intent(in)  :: v(:)
      complex(RD) :: el1(ne, ne), er1(ne, ne), el2(ne, ne), er2(ne, ne)
      complex(RD) :: m_g(2*ne, ne), m_u(2*ne, ne), m_e(2*ne, ne)
      integer     :: ipiv(ne), info, q
      q = ne*ne
      el1 = RESHAPE(v(1:q), [ne, ne])
      er1 = RESHAPE(v(q+1:2*q), [ne, ne])
      el2 = RESHAPE(v(2*q+1:3*q), [ne, ne])
      er2 = RESHAPE(v(3*q+1:4*q), [ne, ne])
      m_g(:ne, :) = el1
      m_g(ne+1:, :) = 0._RD
      m_u(:ne, :) = er1
      m_u(ne+1:, :) = el2
      m_e(:ne, :) = 0._RD
      m_e(ne+1:, :) = er2
      call zgetrf(2*ne, ne, m_u, 2*ne, ipiv, info)
      call zlaswp(ne, m_e(1,1), 2*ne, 1, ne, ipiv, 1)
      call ztrsm('L', 'L', 'N', 'U', ne, ne, CONE, m_u(1,1), 2*ne, m_e(1,1), 2*ne)
      call zgemm('N', 'N', ne, ne, ne, -CONE, m_u(ne+1,1), 2*ne, m_e(1,1), 2*ne, CONE, &
         m_e(ne+1,1), 2*ne)
      call ztrsm('L', 'U', 'N', 'N', ne, ne, CONE, m_u(1,1), 2*ne, m_e(1,1), 2*ne)
      call zlaswp(ne, m_g(1,1), 2*ne, 1, ne, ipiv, 1)
      call ztrsm('L', 'L', 'N', 'U', ne, ne, CONE, m_u(1,1), 2*ne, m_g(1,1), 2*ne)
      call zgemm('N', 'N', ne, ne, ne, -CONE, m_u(ne+1,1), 2*ne, m_g(1,1), 2*ne, CONE, &
         m_g(ne+1,1), 2*ne)
      call ztrsm('L', 'U', 'N', 'N', ne, ne, CONE, m_u(1,1), 2*ne, m_g(1,1), 2*ne)
      call wr(sec, id, c2r(v), [ne], [c2r(RESHAPE(m_u, [2*q])), c2r(RESHAPE(m_e, [2*q])), &
         c2r(RESHAPE(m_g, [2*q]))], [info, ipiv])
   end subroutine run_factor

   ! block_sysmtx_m.fypp backsub_c_: y = -E_l x1 (beta ZERO), y = -E_r x2 + y
   subroutine run_gemv(sec, id, ne, v, x1, x2, y0)
      character(*), intent(in) :: sec
      integer, intent(in)      :: id, ne
      complex(RD), intent(in)  :: v(:), x1(:), x2(:), y0(:)
      complex(RD) :: el(ne, ne), er(ne, ne), y(ne)
      el = RESHAPE(v(1:ne*ne), [ne, ne])
      er = RESHAPE(v(ne*ne+1:2*ne*ne), [ne, ne])
      y = y0
      call zgemv('N', ne, ne, -CONE, el, ne, x1, 1, CZERO, y, 1)
      call zgemv('N', ne, ne, -CONE, er, ne, x2, 1, CONE, y, 1)
      call wr(sec, id, c2r([v, x1, x2, y0]), [ne], c2r(y), [integer::])
   end subroutine run_gemv

end module gen_g4_m

program gen_gyre_g4_vectors

   use gen_g4_m

   implicit none

   character(65536) :: line
   character(4096)  :: in_file, out_file
   character(8)     :: kind
   integer          :: unit, ios, id, m, n, i, j, k, cnt, info, ncv, nru
   integer          :: incx, incy, e_lo, e_hi, itr, iul, idg
   integer(INT64), allocatable :: bits(:)
   real(RD), allocatable    :: rv(:), d(:), e(:), d0(:), e0(:), rw(:)
   complex(RD), allocatable :: v(:), a0(:), b0(:), x(:), y(:), vt(:,:), u(:,:)
   complex(RD) :: alpha, beta, tau, z1, cdum(1)
   real(RD)    :: f, g, h, p, q, xr(4)
   character   :: ch(3) = ['N', 'T', 'C'], ul(2) = ['U', 'L'], dg(2) = ['U', 'N'], nm(4) = ['M', 'O', '1', 'I']

   if (COMMAND_ARGUMENT_COUNT() /= 2) stop 'usage: gen_gyre_g4_vectors tapped_inputs.txt out.tsv'
   call GET_COMMAND_ARGUMENT(1, in_file)
   call GET_COMMAND_ARGUMENT(2, out_file)
   open(NEWUNIT=out_unit, FILE=TRIM(out_file), STATUS='REPLACE', ACTION='WRITE')

   ! ---- leaf routines with edge cases

   do i = 1, 34
      ! DZNRM2: complex vectors with tiny, huge and subnormal components
      n = rnd_int(0, 12)
      e_lo = MERGE(-1074, -60, MOD(i, 7) == 0)
      e_hi = MERGE(1020, 60, MOD(i, 11) == 0)
      allocate(x(n))
      do j = 1, n
         x(j) = CMPLX(rnd_d(MAX(e_lo, -1022), e_hi), rnd_d(MAX(e_lo, -1022), e_hi), RD)
      end do
      if (MOD(i, 13) == 0) x = x*2._RD**(-1060)
      if (MOD(i, 17) == 0 .AND. n > 0) x(1) = CMPLX(REAL(x(1)), 0._RD, RD)
      call wr('dznrm2', i, c2r(x), [n], [dznrm2(n, x, 1)], [integer::])
      call wr('izamax', i, c2r(x), [n], [real(RD)::], [izamax(n, x, 1)])
      deallocate(x)

      ! DLAPY3
      f = rnd_d(-300, 300); g = rnd_d(-300, 300); h = rnd_d(-300, 300)
      select case (MOD(i, 8))
      case (1); f = 0._RD; g = 0._RD; h = 0._RD
      case (2); g = 0._RD
      case (3); f = HUGE(f); g = HUGE(g)
      case (4); h = f*2._RD**(-60)
      end select
      call wr('dlapy3', i, [f, g, h], [integer::], [dlapy3(f, g, h)], [integer::])

      ! DLADIV / ZLADIV, including the overflow and underflow rescalings
      xr(1) = rnd_d(-60, 60); xr(2) = rnd_d(-60, 60); xr(3) = rnd_d(-60, 60); xr(4) = rnd_d(-60, 60)
      select case (MOD(i, 10))
      case (1); xr(1:2) = xr(1:2)*2._RD**960               ! AB >= OV/2
      case (2); xr(3:4) = xr(3:4)*2._RD**960               ! CD >= OV/2
      case (3); xr(1:2) = xr(1:2)*2._RD**(-1000)           ! AB tiny
      case (4); xr(3:4) = xr(3:4)*2._RD**(-1000)           ! CD tiny
      case (5); xr(4) = 0._RD                               ! R = 0
      case (6); xr(2) = 0._RD; xr(4) = xr(3)*2._RD**(-600) ! BR = 0
      case (7); xr(3) = 0._RD                               ! |D| > |C|
      case (8); xr(1) = 0._RD; xr(2) = 0._RD
      end select
      call dladiv(xr(1), xr(2), xr(3), xr(4), p, q)
      call wr('dladiv', i, xr, [integer::], [p, q], [integer::])
      z1 = zladiv(CMPLX(xr(1), xr(2), RD), CMPLX(xr(3), xr(4), RD))
      call wr('zladiv', i, xr, [integer::], c2r([z1]), [integer::])

      ! ZLARFG, including the tiny-beta rescaling loop
      n = rnd_int(0, 12)
      alpha = rnd_c(-30, 30)
      allocate(x(MAX(n-1, 0)), a0(MAX(n-1, 0)))
      call rnd_mat(x, -30, 30)
      select case (MOD(i, 9))
      case (1)                                   ! tiny vector: rescaling loop
         alpha = alpha*2._RD**(-1040)
         x = x*2._RD**(-1040)
      case (2); x = CZERO                        ! XNORM = 0, ALPHI /= 0
      case (3); x = CZERO; alpha = CMPLX(REAL(alpha), 0._RD, RD)   ! H = I
      case (4); alpha = CMPLX(REAL(alpha), 0._RD, RD)
      case (5)                                   ! very tiny: KNT > 1
         alpha = alpha*2._RD**(-1070)
         x = x*2._RD**(-1070)
      end select
      a0 = x
      z1 = alpha
      call zlarfg(n, alpha, x, 1, tau)
      call wr('zlarfg', i, c2r([z1, a0]), [n], c2r([alpha, tau, x]), [integer::])
      deallocate(x, a0)
   end do

   ! ---- ZLANGE norms

   do i = 1, 10
      m = rnd_int(1, 8); n = rnd_int(1, 8)
      allocate(a0(m*n))
      call rnd_mat(a0, -40, 40)
      if (MOD(i, 5) == 0) a0 = a0*2._RD**(-1040)
      allocate(rv(m))
      call wr('zlange', i, c2r(a0), [m, n, ICHAR(nm(MOD(i, 4)+1))], &
         [zlange(nm(MOD(i, 4)+1), m, n, a0, m, rv)], [integer::])
      deallocate(a0, rv)
   end do

   ! ---- ZGEMV, ZGEMM, ZTRSM argument patterns beyond GYRE's

   do i = 1, 24
      itr = MOD(i, 3) + 1
      m = rnd_int(1, 7); n = rnd_int(1, 7)
      incx = rnd_int(1, 2); incy = rnd_int(1, 2)
      alpha = rnd_c(-4, 4); beta = rnd_c(-4, 4)
      select case (MOD(i/3, 4))
      case (1); beta = CZERO
      case (2); beta = CONE
      case (3); alpha = CZERO
      end select
      allocate(a0(m*n))
      call rnd_mat(a0, -10, 10)
      if (itr == 1) then
         allocate(x(n*incx), y(m*incy))
      else
         allocate(x(m*incx), y(n*incy))
      end if
      call rnd_mat(x, -10, 10)
      call rnd_mat(y, -10, 10)
      b0 = y
      call zgemv(ch(itr), m, n, alpha, a0, m, x, incx, beta, y, incy)
      call wr('zgemv', i, c2r([alpha, beta, a0, x, b0]), [ICHAR(ch(itr)), m, n, incx, incy], &
         c2r(y), [integer::])
      deallocate(a0, x, y, b0)
   end do

   do i = 1, 8
      m = rnd_int(1, 7); n = rnd_int(1, 7); k = rnd_int(0, 7)
      alpha = rnd_c(-4, 4); beta = rnd_c(-4, 4)
      select case (MOD(i, 4))
      case (1); beta = CZERO
      case (2); beta = CONE
      case (3); alpha = CZERO
      end select
      allocate(a0(m*MAX(k, 1)), b0(MAX(k, 1)*n), y(m*n))
      call rnd_mat(a0, -10, 10)
      call rnd_mat(b0, -10, 10)
      call rnd_mat(y, -10, 10)
      x = y
      call zgemm('N', 'N', m, n, k, alpha, a0, m, b0, MAX(k, 1), beta, y, m)
      call wr('zgemm', i, c2r([alpha, beta, a0, b0, x]), [m, n, k], c2r(y), [integer::])
      deallocate(a0, b0, y, x)
   end do

   do i = 1, 24
      itr = MOD(i, 3) + 1; iul = MOD(i/3, 2) + 1; idg = MOD(i/6, 2) + 1
      m = rnd_int(1, 7); n = rnd_int(1, 5)
      alpha = rnd_c(-3, 3)
      if (MOD(i, 8) == 0) alpha = CONE
      if (MOD(i, 23) == 0) alpha = CZERO
      allocate(a0(m*m), b0(m*n))
      call rnd_mat(a0, -3, 3)
      do j = 1, m
         a0((j-1)*m+j) = a0((j-1)*m+j)*4._RD
      end do
      call rnd_mat(b0, -6, 6)
      if (MOD(i, 5) == 0) b0(1:m) = CZERO                   ! zero right-hand side column
      y = b0
      call ztrsm('L', ul(iul), ch(itr), dg(idg), m, n, alpha, a0, m, y, m)
      call wr('ztrsm', i, c2r([alpha, a0, b0]), [ICHAR(ul(iul)), ICHAR(ch(itr)), ICHAR(dg(idg)), m, n], &
         c2r(y), [integer::])
      deallocate(a0, b0, y)
   end do

   ! ---- ZBDSQR directly ('U' and 'L', with VT and U)

   do i = 1, 18
      n = rnd_int(1, 10)
      ncv = n; nru = n
      if (MOD(i, 6) == 0) ncv = rnd_int(1, 4)
      if (MOD(i, 7) == 0) nru = rnd_int(1, 4)
      allocate(d(n), e(MAX(n-1, 1)), d0(n), e0(MAX(n-1, 1)), vt(n, ncv), u(nru, n), rw(4*n))
      do j = 1, n
         d(j) = rnd_d(-10, 10)
      end do
      do j = 1, n - 1
         e(j) = rnd_d(-10, 10)
      end do
      if (MOD(i, 4) == 0 .AND. n > 2) e(2) = 0._RD
      if (MOD(i, 5) == 0 .AND. n > 1) d(n) = 0._RD
      if (MOD(i, 9) == 0 .AND. n > 3) d(2) = d(1)*2._RD**(-40)
      allocate(a0(n*ncv), b0(nru*n))
      call rnd_mat(a0, -2, 2)
      call rnd_mat(b0, -2, 2)
      vt = RESHAPE(a0, [n, ncv]); u = RESHAPE(b0, [nru, n])
      d0 = d; e0 = e
      call zbdsqr(ul(MOD(i, 2)+1), n, ncv, nru, 0, d, e, vt, n, u, nru, cdum, 1, rw, info)
      call wr('zbdsqr', i, [d0, e0(1:n-1), c2r(a0), c2r(b0)], [ICHAR(ul(MOD(i, 2)+1)), n, ncv, nru], &
         [d, e(1:n-1), c2r(RESHAPE(vt, [n*ncv])), c2r(RESHAPE(u, [nru*n]))], [info])
      deallocate(d, e, d0, e0, vt, u, rw, a0, b0)
   end do

   ! ---- replay of GYRE's tapped calls

   open(NEWUNIT=unit, FILE=TRIM(in_file), STATUS='OLD', ACTION='READ')
   do
      read(unit, '(A)', IOSTAT=ios) line
      if (ios /= 0) exit
      read(line, *) kind, id, m, n
      select case (TRIM(kind))
      case ('GESVD'); cnt = 2*m*n
      case ('FACTOR'); cnt = 8*m*n
      case ('GEMV'); cnt = 4*m*n
      case default; stop 'unknown tapped kind'
      end select
      allocate(bits(cnt), rv(cnt))
      read(line, *) kind, id, m, n, (bits(k), k = 1, cnt)
      rv = TRANSFER(bits, rv)
      v = CMPLX(rv(1::2), rv(2::2), RD)
      select case (TRIM(kind))
      case ('GESVD')
         call run_gesvd('gesvd_tap', id, m, n, v)
         call run_getrf('getrf_det_tap', id, m, n, v)        ! det_c_: ZGETRF(2n_e, 2n_e)
      case ('FACTOR')
         call run_factor('factor_tap', id, m, v)
      case ('GEMV')
         allocate(x(3*m))
         call rnd_mat(x, -4, 4)
         call run_gemv('gemv_tap', id, m, v, x(1:m), x(m+1:2*m), x(2*m+1:))
         deallocate(x)
      end select
      deallocate(bits, rv)
   end do
   close(unit)

   ! ---- synthetic LU cases

   ! i = 1..38 cover every selector below; i = 119 adds tiny pivots in a
   ! rank-deficient 12 x 10 matrix, the case that separates ZGETRF2's
   ! recursive order from the classical right-looking order
   do j = 1, 39
      i = MERGE(119, j, j == 39)
      m = rnd_int(1, 12); n = rnd_int(1, 12)
      if (i == 119) then
         m = 12; n = 10
      end if
      if (MOD(i, 3) == 0) then                 ! GYRE's cyclic-reduction shape
         n = rnd_int(1, 6); m = 2*n
      end if
      if (MOD(i, 4) == 0) then                 ! GYRE's det shape
         n = 2*rnd_int(1, 6); m = n
      end if
      allocate(a0(m*n))
      call rnd_mat(a0, -20, 20)
      if (MOD(i, 10) == 0) a0(1:m) = CZERO                                  ! zero column: INFO > 0
      if (MOD(i, 17) == 0) a0 = a0*2._RD**(-1030)                            ! pivots below SFMIN
      if (MOD(i, 7) == 0 .AND. n >= 2) a0(m+1:2*m) = a0(1:m)                 ! rank deficient
      if (MOD(i, 11) == 0) a0 = CMPLX(REAL(a0), 0._RD, RD)                   ! real-valued
      if (MOD(i, 13) == 0) a0 = CMPLX(0._RD, AIMAG(a0), RD)                  ! imaginary
      if (MOD(i, 19) == 0) a0(m+1:) = CZERO                                  ! zero trailing block
      call run_getrf('getrf_syn', i, m, n, a0)
      deallocate(a0)
   end do

   ! ---- synthetic SVD cases (path 10: n <= m < 1.6 n)

   do i = 1, 14
      n = rnd_int(2, 12)   ! n = 1 has MNTHR = 1 <= m: paths 1-9, not GYRE's
      m = n
      if (MOD(i, 5) == 0) m = MAX(n, INT(REAL(n)*1.6) - 1)
      allocate(a0(m*n))
      call rnd_mat(a0, -20, 20)
      select case (MOD(i, 7))
      case (1); if (n >= 2) a0(m+1:2*m) = a0(1:m)*(0._RD, 1._RD)              ! rank deficient
      case (2)                                                                ! diagonal
         a0 = CZERO
         do j = 1, n
            a0((j-1)*m+j) = rnd_c(-20, 20)
         end do
      case (3)                                                                ! repeated values
         a0 = CZERO
         do j = 1, n
            a0((j-1)*m+j) = MERGE((2._RD, 0._RD), (0._RD, -2._RD), MOD(j, 2) == 0)
         end do
         a0(m) = (1e-3_RD, 1e-3_RD)
      case (4); a0 = a0*2._RD**rnd_int(-200, 200)                              ! scaled within range
      case (5); a0 = CMPLX(REAL(a0), 0._RD, RD)                                ! real-valued
      case (6); a0(1:m) = CZERO                                                ! zero column
      end select
      if (i == 1) a0 = CZERO                                                   ! zero matrix
      call run_gesvd('gesvd_syn', i, m, n, a0)
      deallocate(a0)
   end do

   close(out_unit)

end program gen_gyre_g4_vectors
