! SPDX-License-Identifier: GPL-3.0-or-later
!
! NEWUOA oracle driver (Stream G, package G2).
!
! Adapted from MESA 26.4.1 num/test/src/test_newuoa.f90. Calls MESA's
! num_lib NEWUOA (num/public/num_newuoa.dek, num/private/mod_newuoa.f)
! and records every CALFUN call (X(1:N) and F) and the returned X as
! IEEE-754 binary64 hexadecimal words.
!
! Runs (one output file each, named on the command line):
!   chebyquad_n2, chebyquad_n4, chebyquad_n6:
!       test_newuoa.f90 do_test_newuoa (N = 2, 4, 6; NPT = 2N+1;
!       X(I) = I/(N+1); RHOBEG = 0.2*X(1); RHOEND = 1d-6; MAXFUN = 5000;
!       max_valid_value = 1d99).
!   chebyquad_n4_npt6:
!       the same problem with N = 4 and NPT = 6 < 2N+1, which takes the
!       NPT < NF+N initialization branch (mod_newuoa.f:208-212).
!   quad_rhoend1:
!       astero do_bobyqa_or_newuoa shape (astero_run_support.f90:663-691)
!       with three varied parameters as in fast_newuoa: N = 3, NPT = 7,
!       X = 0, RHOBEG = 0.45, RHOEND = 1.0 (inlist_astero_search_controls
!       newuoa_rhoend), MAXFUN = 1000, max_valid_value = 1d6, and a
!       closed-form quadratic objective (quad_f below).
!   quad_rhoend1em6:
!       the same with RHOEND = 1d-6.
!
! Record format (whitespace separated):
!   # <header lines>
!   CALL <ncall> <X(1)> ... <X(N)> <F>
!   XOUT <X(1)> ... <X(N)>
! where every real is 16 hexadecimal digits of transfer(value, 0_int64).

module newuoa_oracle_mod
   use const_def, only: dp
   use iso_fortran_env, only: int64
   implicit none
   integer :: ncall = 0
   integer :: iout = 20
   integer :: which = 0   ! 1 = Chebyquad, 2 = quadratic
contains

   subroutine put_hex(v)
      real(dp), intent(in) :: v
      write (iout, '(1X,Z16.16)', advance='no') transfer(v, 0_int64)
   end subroutine put_hex

   subroutine calfun(n, x, f)
      integer, intent(in) :: n
      real(dp), intent(in) :: x(*)
      real(dp), intent(out) :: f
      integer :: i
      if (which == 1) then
         call chebyquad(n, x, f)
      else
         call quad_f(n, x, f)
      end if
      ncall = ncall + 1
      write (iout, '(A,1X,I0)', advance='no') 'CALL', ncall
      do i = 1, n
         call put_hex(x(i))
      end do
      call put_hex(f)
      write (iout, '(A)') ''
   end subroutine calfun

   ! Chebyquad, verbatim arithmetic of test_newuoa.f90 calfun.
   subroutine chebyquad(n, x, f)
      integer, intent(in) :: n
      real(dp), intent(in) :: x(*)
      real(dp), intent(out) :: f
      integer :: I, J, IW, NP
      real(dp) :: Y(10, 10), sum
      do J = 1, N
         Y(1, J) = 1.0D0
         Y(2, J) = 2.0D0*X(J) - 1.0D0
      end do
      do I = 2, N
      do J = 1, N
         Y(I + 1, J) = 2.0D0*Y(2, J)*Y(I, J) - Y(I - 1, J)
      end do
      end do
      F = 0.0D0
      NP = N + 1
      IW = 1
      do I = 1, NP
         SUM = 0.0D0
         do J = 1, N
            SUM = SUM + Y(I, J)
         end do
         SUM = SUM/DBLE(N)
         IF (IW > 0) SUM = SUM + 1.0D0/DBLE(I*I - 2*I)
         IW = -IW
         F = F + SUM*SUM
      end do
   end subroutine chebyquad

   ! Closed-form quadratic in three variables with a coupled Hessian:
   !   f = (x1 - 0.3)^2 + 2 (x2 + 0.2)^2 + 0.5 (x3 - 0.1)^2
   !       + 0.25 x1 x2 - 0.125 x2 x3 + 1.5
   ! evaluated left to right exactly as written.
   subroutine quad_f(n, x, f)
      integer, intent(in) :: n
      real(dp), intent(in) :: x(*)
      real(dp), intent(out) :: f
      real(dp) :: a, b, c
      if (n /= 3) stop 'quad_f needs n = 3'
      a = x(1) - 0.3d0
      b = x(2) + 0.2d0
      c = x(3) - 0.1d0
      f = a*a + 2.0d0*(b*b) + 0.5d0*(c*c) + 0.25d0*(x(1)*x(2)) &
          - 0.125d0*(x(2)*x(3)) + 1.5d0
   end subroutine quad_f

end module newuoa_oracle_mod

program newuoa_oracle
   use const_def, only: dp
   use num_lib, only: newuoa
   use math_lib, only: math_init, MATH_BACKEND
   use newuoa_oracle_mod
   use iso_fortran_env, only: compiler_version
   implicit none
   real(dp) :: x(10), w(10000), rhobeg, rhoend, max_valid_value
   integer :: n, npt, maxfun, i, ierr, iprint, nargs
   character(len=256) :: run, outfile, mesa_version, sdk_version
   character(len=512) :: compiler

   nargs = command_argument_count()
   if (nargs /= 3) stop 'usage: newuoa_oracle <run> <outfile> <mesa_version_file>'
   call get_command_argument(1, run)
   call get_command_argument(2, outfile)
   call get_command_argument(3, mesa_version)
   open (unit=21, file=trim(mesa_version), status='old', action='read')
   read (21, '(A)') mesa_version
   close (21)
   call get_environment_variable('MESASDK_VERSION', sdk_version)
   compiler = compiler_version()

   call math_init()
   iprint = 0
   x = 0
   w = 0
   select case (trim(run))
   case ('chebyquad_n2', 'chebyquad_n4', 'chebyquad_n6', 'chebyquad_n4_npt6')
      which = 1
      read (run(12:12), '(I1)') n
      npt = 2*n + 1
      if (len_trim(run) > 12) read (run(17:), *) npt
      maxfun = 5000
      rhoend = 1.0D-6
      max_valid_value = 1d99
      do i = 1, n
         x(i) = DBLE(i)/DBLE(n + 1)
      end do
      rhobeg = 0.2D0*x(1)
   case ('quad_rhoend1', 'quad_rhoend1em6')
      which = 2
      n = 3
      npt = 2*n + 1
      maxfun = 1000
      max_valid_value = 1d6
      rhobeg = 0.45d0
      if (trim(run) == 'quad_rhoend1') then
         rhoend = 1.0d0
      else
         rhoend = 1d-6
      end if
      x(1:n) = 0
   case default
      stop 'unknown run'
   end select

   open (unit=iout, file=trim(outfile), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'cannot open output'
   write (iout, '(A)') '# NEWUOA oracle: CALFUN sequence and returned X (IEEE-754 binary64 hex)'
   write (iout, '(A)') '# MESA version: '//trim(mesa_version)
   write (iout, '(A)') '# compiler: '//trim(compiler)
   write (iout, '(A)') '# MESA SDK: '//trim(sdk_version)
   write (iout, '(A)') '# math backend: '//trim(MATH_BACKEND)
   write (iout, '(A)') '# run: '//trim(run)
   write (iout, '(A,I0,A,I0,A,I0)') '# N = ', n, ', NPT = ', npt, ', MAXFUN = ', maxfun
   write (iout, '(A)', advance='no') '# RHOBEG RHOEND max_valid_value:'
   call put_hex(rhobeg)
   call put_hex(rhoend)
   call put_hex(max_valid_value)
   write (iout, '(A)') ''
   write (iout, '(A)', advance='no') '# X0:'
   do i = 1, n
      call put_hex(x(i))
   end do
   write (iout, '(A)') ''

   ncall = 0
   call newuoa(n, npt, x, rhobeg, rhoend, iprint, maxfun, w, calfun, max_valid_value)

   write (iout, '(A)', advance='no') 'XOUT'
   do i = 1, n
      call put_hex(x(i))
   end do
   write (iout, '(A)') ''
   close (iout)
   write (*, '(A,1X,A,1X,I0,A)') trim(run), 'done:', ncall, ' CALFUN calls'
end program newuoa_oracle
