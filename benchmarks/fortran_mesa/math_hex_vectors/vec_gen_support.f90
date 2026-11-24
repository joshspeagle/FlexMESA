! SPDX-License-Identifier: GPL-3.0-or-later
! Isolated Fortran platform-intrinsic wrappers (`#196`, `#198`) and
! deterministic IEEE-754 edge-case table builders for `gen_math_vectors.f90` (`#165`).

module mesa_platform_intrinsics
   use const_def, only: dp
   implicit none
   private
   public :: f90_dlog, f90_dlog10, f90_log10, f90_exp, f90_atan2, f90_datan2, f90_pow_r
contains
   pure real(dp) function f90_dlog(x) result(y)
      intrinsic :: dlog
      real(dp), intent(in) :: x
      y = dlog(x)
   end function f90_dlog

   pure real(dp) function f90_dlog10(x) result(y)
      intrinsic :: dlog10
      real(dp), intent(in) :: x
      y = dlog10(x)
   end function f90_dlog10

   pure real(dp) function f90_log10(x) result(y)
      intrinsic :: log10
      real(dp), intent(in) :: x
      y = log10(x)
   end function f90_log10

   pure real(dp) function f90_exp(x) result(y)
      intrinsic :: exp
      real(dp), intent(in) :: x
      y = exp(x)
   end function f90_exp

   pure real(dp) function f90_atan2(y, x) result(z)
      intrinsic :: atan2
      real(dp), intent(in) :: y, x
      z = atan2(y, x)
   end function f90_atan2

   pure real(dp) function f90_datan2(y, x) result(z)
      intrinsic :: datan2
      real(dp), intent(in) :: y, x
      z = datan2(y, x)
   end function f90_datan2

   pure real(dp) function f90_pow_r(x, y) result(z)
      real(dp), intent(in) :: x, y
      z = x ** y
   end function f90_pow_r
end module mesa_platform_intrinsics

module vec_gen_support
   use, intrinsic :: iso_fortran_env, only: int64
   use const_def, only: dp
   implicit none
   private
   public :: read_mesa_version, current_iso_date, build_unary_edges

contains

   subroutine read_mesa_version(mdir, ver)
      character(len=*), intent(in)  :: mdir
      character(len=*), intent(out) :: ver
      integer :: u, io
      ver = 'unknown'
      if (len_trim(mdir) == 0) return
      open(newunit=u, file=trim(mdir)//'/data/version_number', status='old', action='read', iostat=io)
      if (io == 0) then
         read(u, '(A)', iostat=io) ver
         close(u)
      end if
   end subroutine read_mesa_version

   subroutine current_iso_date(dstr)
      character(len=*), intent(out) :: dstr
      character(len=8) :: d8
      call date_and_time(date=d8)
      dstr = d8(1:4) // '-' // d8(5:6) // '-' // d8(7:8)
   end subroutine current_iso_date

   pure integer(int64) function dp_bits(x)
      real(dp), intent(in) :: x
      dp_bits = transfer(x, 0_int64)
   end function dp_bits

   subroutine push_edge(arr, n, val)
      integer(int64), intent(inout) :: arr(:)
      integer, intent(inout) :: n
      integer(int64), intent(in) :: val
      n = n + 1
      arr(n) = val
   end subroutine push_edge

   subroutine build_unary_edges(edges, n)
      integer(int64), intent(out) :: edges(:)
      integer, intent(out) :: n
      integer :: k
      n = 0
      call push_edge(edges, n, int(Z'0000000000000000', int64)) ! +0.0
      call push_edge(edges, n, int(Z'8000000000000000', int64)) ! -0.0
      call push_edge(edges, n, int(Z'0000000000000001', int64)) ! +min subnormal
      call push_edge(edges, n, int(Z'8000000000000001', int64)) ! -min subnormal
      call push_edge(edges, n, int(Z'000FFFFFFFFFFFFF', int64)) ! +max subnormal
      call push_edge(edges, n, int(Z'800FFFFFFFFFFFFF', int64)) ! -max subnormal
      call push_edge(edges, n, int(Z'0010000000000000', int64)) ! +min normal
      call push_edge(edges, n, int(Z'8010000000000000', int64)) ! -min normal
      call push_edge(edges, n, int(Z'7FEFFFFFFFFFFFFF', int64)) ! +max finite
      call push_edge(edges, n, int(Z'FFEFFFFFFFFFFFFF', int64)) ! -max finite
      call push_edge(edges, n, int(Z'7FF0000000000000', int64)) ! +Inf
      call push_edge(edges, n, int(Z'FFF0000000000000', int64)) ! -Inf
      call push_edge(edges, n, int(Z'7FF8000000000000', int64)) ! canonical qNaN
      call push_edge(edges, n, int(Z'FFF8000000000013', int64)) ! negative payload qNaN
      call push_edge(edges, n, int(Z'3FF0000000000000', int64)) ! +1.0
      call push_edge(edges, n, int(Z'BFF0000000000000', int64)) ! -1.0
      call push_edge(edges, n, int(Z'3FEFFFFFFFFFFFFF', int64)) ! 1 - eps/2
      call push_edge(edges, n, int(Z'3FF0000000000001', int64)) ! 1 + eps
      call push_edge(edges, n, int(Z'BFEFFFFFFFFFFFFF', int64)) ! -(1 - eps/2)
      call push_edge(edges, n, int(Z'BFF0000000000001', int64)) ! -(1 + eps)
      call push_edge(edges, n, dp_bits(0.5_dp))
      call push_edge(edges, n, dp_bits(-0.5_dp))
      call push_edge(edges, n, dp_bits(2.0_dp))
      call push_edge(edges, n, dp_bits(10.0_dp))
      call push_edge(edges, n, dp_bits(1.0e-99_dp))
      call push_edge(edges, n, dp_bits(0.9999999999999999_dp * 1.0e-99_dp))
      call push_edge(edges, n, dp_bits(1.0e-16_dp))
      call push_edge(edges, n, dp_bits(-1.0e-16_dp))
      call push_edge(edges, n, int(Z'40862E42FEFA39EF', int64))
      call push_edge(edges, n, int(Z'40862E42FEFA39F0', int64))
      call push_edge(edges, n, int(Z'408633CE8FB9F87D', int64))
      call push_edge(edges, n, int(Z'408633CE8FB9F87E', int64))
      call push_edge(edges, n, int(Z'C086232BDD7ABCD2', int64))
      call push_edge(edges, n, int(Z'C0874910D52D3051', int64))
      call push_edge(edges, n, int(Z'C0874910D52D3052', int64))
      call push_edge(edges, n, dp_bits(1.0e6_dp))
      call push_edge(edges, n, dp_bits(-1.0e6_dp))
      call push_edge(edges, n, dp_bits(1.0e15_dp))
      call push_edge(edges, n, dp_bits(-1.0e15_dp))
      call push_edge(edges, n, dp_bits(1.0e50_dp))
      call push_edge(edges, n, dp_bits(1.0e100_dp))
      call push_edge(edges, n, dp_bits(1.0e200_dp))
      call push_edge(edges, n, dp_bits(1.0e300_dp))
      call push_edge(edges, n, dp_bits(-1.0e300_dp))
      do k = -5, 5
         call push_edge(edges, n, dp_bits(real(k, dp)))
      end do
      call push_edge(edges, n, dp_bits(307.0_dp))
      call push_edge(edges, n, dp_bits(308.0_dp))
      call push_edge(edges, n, dp_bits(309.0_dp))
      call push_edge(edges, n, dp_bits(-307.0_dp))
      call push_edge(edges, n, dp_bits(-308.0_dp))
      call push_edge(edges, n, dp_bits(-309.0_dp))
      call push_edge(edges, n, dp_bits(-323.0_dp))
      call push_edge(edges, n, dp_bits(-324.0_dp))
      call push_edge(edges, n, dp_bits(308.2547155599167_dp))
      call push_edge(edges, n, dp_bits(-323.3062153431158_dp))
      call push_edge(edges, n, dp_bits(0.3010299956639812_dp))
      call push_edge(edges, n, dp_bits(-2.5_dp))
   end subroutine build_unary_edges

end module vec_gen_support
