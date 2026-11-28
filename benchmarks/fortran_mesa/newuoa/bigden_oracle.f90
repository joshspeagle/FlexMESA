! SPDX-License-Identifier: GPL-3.0-or-later
!
! BIGDEN unit oracle (Stream G, package G2).
!
! No full NEWUOA run found so far reaches BIGDEN (gcov, see README.md), so
! this driver calls MESA's compiled BIGDEN (num/private/mod_newuoa.f:703-1056,
! the object in $(MESA_DIR)/lib/libnum.a) directly on deterministic inputs
! and records inputs and outputs as IEEE-754 binary64 hexadecimal words.
!
! Inputs use g(i) = mod(37*i+11, 101)/101 - 1/2, which is exact to evaluate.
!   case A: N = 3, NPT = 7, IDZ = 1, KOPT = 2, KNEW = 5, generic D;
!   case B: N = 4, NPT = 9, IDZ = 2, KOPT = 1, KNEW = 3, D nearly parallel
!           to XPT(KNEW,:) - XOPT (exercises mod_newuoa.f:773-796).
!
! Record format: "CASE name N NPT IDZ KOPT KNEW", then "IN <array> <hex...>"
! and "OUT <array> <hex...>" lines (arrays column-major), then "END".

module bigden_oracle_mod
   use const_def, only: dp
   use iso_fortran_env, only: int64
   implicit none
   integer, parameter :: iout = 20
contains
   real(dp) function g(i)
      integer, intent(in) :: i
      g = dble(mod(37*i + 11, 101))/101d0 - 0.5d0
   end function g

   subroutine put(tag, name, a, m)
      character(len=*), intent(in) :: tag, name
      integer, intent(in) :: m
      real(dp), intent(in) :: a(m)
      integer :: i
      write (iout, '(A,1X,A)', advance='no') tag, name
      do i = 1, m
         write (iout, '(1X,Z16.16)', advance='no') transfer(a(i), 0_int64)
      end do
      write (iout, '(A)') ''
   end subroutine put

   subroutine run_case(name, n, npt, idz, kopt, knew, parallel)
      use mod_newuoa, only: bigden
      character(len=*), intent(in) :: name
      integer, intent(in) :: n, npt, idz, kopt, knew
      logical, intent(in) :: parallel
      integer :: ndim, nptm, i, j, k
      real(dp), allocatable :: xopt(:), xpt(:, :), bmat(:, :), zmat(:, :), &
         d(:), w(:), vlag(:), s(:), wvec(:, :), prod(:, :)
      real(dp) :: beta(1)
      ndim = npt + n
      nptm = npt - n - 1
      allocate (xopt(n), xpt(npt, n), bmat(ndim, n), zmat(npt, nptm), d(n), &
                w(ndim), vlag(ndim), s(n), wvec(ndim, 5), prod(ndim, 5))
      do j = 1, n
         do k = 1, npt
            xpt(k, j) = g(k + 10*j)
         end do
      end do
      do j = 1, n
         xopt(j) = xpt(kopt, j)
      end do
      do j = 1, n
         do i = 1, ndim
            bmat(i, j) = 0.1d0*g(3*i + 7*j + 200)
         end do
      end do
      do j = 1, nptm
         do k = 1, npt
            zmat(k, j) = g(5*k + 11*j + 400)
         end do
      end do
      do i = 1, n
         if (parallel) then
            d(i) = 0.5d0*(xpt(knew, i) - xopt(i)) + 1d-3*g(i + 700)
         else
            d(i) = 0.3d0*g(i + 600)
         end if
      end do
      w = 0; vlag = 0; s = 0; wvec = 0; prod = 0; beta = 0
      write (iout, '(A,1X,A,5(1X,I0))') 'CASE', name, n, npt, idz, kopt, knew
      call put('IN', 'XOPT', xopt, n)
      call put('IN', 'XPT', reshape(xpt, [npt*n]), npt*n)
      call put('IN', 'BMAT', reshape(bmat, [ndim*n]), ndim*n)
      call put('IN', 'ZMAT', reshape(zmat, [npt*nptm]), npt*nptm)
      call put('IN', 'D', d, n)
      call bigden(n, npt, xopt, xpt, bmat, zmat, idz, ndim, kopt, knew, d, w, &
                  vlag, beta(1), s, wvec, prod)
      call put('OUT', 'D', d, n)
      call put('OUT', 'W', w, ndim)
      call put('OUT', 'VLAG', vlag, ndim)
      call put('OUT', 'BETA', beta, 1)
      call put('OUT', 'S', s, n)
      call put('OUT', 'WVEC', reshape(wvec, [ndim*5]), ndim*5)
      call put('OUT', 'PROD', reshape(prod, [ndim*5]), ndim*5)
      write (iout, '(A)') 'END'
   end subroutine run_case
end module bigden_oracle_mod

program bigden_oracle
   use iso_fortran_env, only: compiler_version
   use math_lib, only: math_init, MATH_BACKEND
   use bigden_oracle_mod
   implicit none
   character(len=256) :: outfile, mesa_version, sdk_version
   if (command_argument_count() /= 2) stop 'usage: bigden_oracle <outfile> <mesa_version_file>'
   call get_command_argument(1, outfile)
   call get_command_argument(2, mesa_version)
   open (unit=21, file=trim(mesa_version), status='old', action='read')
   read (21, '(A)') mesa_version
   close (21)
   call get_environment_variable('MESASDK_VERSION', sdk_version)
   call math_init()
   open (unit=iout, file=trim(outfile), status='replace', action='write')
   write (iout, '(A)') '# NEWUOA BIGDEN unit oracle: inputs and outputs (IEEE-754 binary64 hex)'
   write (iout, '(A)') '# MESA version: '//trim(mesa_version)
   write (iout, '(A)') '# compiler: '//trim(compiler_version())
   write (iout, '(A)') '# MESA SDK: '//trim(sdk_version)
   write (iout, '(A)') '# math backend: '//trim(MATH_BACKEND)
   call run_case('A', 3, 7, 1, 2, 5, .false.)
   call run_case('B', 4, 9, 2, 1, 3, .true.)
   close (iout)
   write (*, '(A)') 'bigden oracle done'
end program bigden_oracle
