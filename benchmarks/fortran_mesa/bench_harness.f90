! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Module: bench_harness
!
! Shared timing, relative-error verification, and deterministic stellar fixture
! generators for the Wave 1 four-stage Fortran MESA benchmark suite.
!
! Provides:
! - LatencyStats and BenchRecord derived types matching flexmesa-bench (Rust)
!   and flexmesa.bench (Python/NumPy/JAX).
! - High-resolution wall-clock timing (now_us, compute_latency_stats).
! - NaN-propagating relative discrepancy helpers (max_rel_diff, max_err3).
! - Deterministic ThermoGrid, BicubTable, and block-tridiagonal fixtures.
! -----------------------------------------------------------------------------
module bench_harness
   use const_def, only: dp
   use math_lib, only: exp10, sinpi, cospi
   use interp_2d_lib_db, only: interp_mkbicub_db
   use omp_lib, only: omp_get_wtime
   implicit none
   private

   ! Number of OpenMP worker threads used in Stage 3 benchmarks.
   integer, parameter, public :: STAGE3_THREADS = 4

   !> Summary latency statistics (median and interquartile range in us).
   type, public :: LatencyStats
      real(dp) :: median_us = 0.0_dp
      real(dp) :: iqr_us = 0.0_dp
   end type LatencyStats

   !> Complete four-stage benchmark record for a single workload and N_zones.
   type, public :: BenchRecord
      character(len=16) :: workload = ''
      integer :: n_zones = 0
      type(LatencyStats) :: stage0
      type(LatencyStats) :: stage1
      type(LatencyStats) :: stage2
      type(LatencyStats) :: stage3
      real(dp) :: max_rel_err = 0.0_dp
      real(dp) :: checksum = 0.0_dp
   end type BenchRecord

   !> Deterministic stellar thermodynamic and composition profile across
   !! n_zones radial shells:
   !! - log10_t(n_zones), t(n_zones): Temperature in log10(K) and K.
   !! - log10_rho(n_zones), rho(n_zones): Density in log10(g cm^-3) and g cm^-3.
   !! - abar(n_zones), zbar(n_zones): Mean atomic mass (amu) and charge (e).
   type, public :: ThermoGrid
      integer :: n_zones = 0
      real(dp), allocatable :: log10_t(:)
      real(dp), allocatable :: t(:)
      real(dp), allocatable :: log10_rho(:)
      real(dp), allocatable :: rho(:)
      real(dp), allocatable :: abar(:)
      real(dp), allocatable :: zbar(:)
   end type ThermoGrid

   !> Precomputed 64 x 64 2D bicubic spline table over (log10_rho, log10_T)
   !! with packed PSPLINE coefficients f1(4 * nx * ny).
   type, public :: BicubTable
      integer :: nx = 64
      integer :: ny = 64
      real(dp), allocatable :: x(:)
      real(dp), allocatable :: y(:)
      real(dp), pointer :: f1(:) => null()
      integer :: ilinx = 0
      integer :: iliny = 0
   end type BicubTable

   public :: now_us
   public :: compute_latency_stats
   public :: max_rel_diff
   public :: max_err3
   public :: init_thermo_grid
   public :: free_thermo_grid
   public :: init_bicub_table
   public :: free_bicub_table

contains

   !> Returns current wall-clock time in microseconds (us) from omp_get_wtime.
   real(dp) function now_us()
      now_us = omp_get_wtime() * 1.0e6_dp
   end function now_us

   !> Sorts an array of timing samples(1:n_samples) in ascending order via
   !! insertion sort and computes the median and interquartile range
   !! (Q75 - Q25) in microseconds (us).
   subroutine compute_latency_stats(samples, n_samples, stats)
      integer, intent(in) :: n_samples
      real(dp), intent(inout) :: samples(n_samples)
      type(LatencyStats), intent(out) :: stats
      integer :: i, j, mid
      real(dp) :: key, q25, q75

      ! Sort timing samples in ascending order.
      do i = 2, n_samples
         key = samples(i)
         j = i - 1
         do while (j >= 1)
            if (samples(j) <= key) exit
            samples(j + 1) = samples(j)
            j = j - 1
         end do
         samples(j + 1) = key
      end do

      ! Extract median and interquartile range matching Rust and Python.
      mid = n_samples / 2
      if (mod(n_samples, 2) == 1) then
         stats%median_us = samples(mid + 1)
      else
         stats%median_us = 0.5_dp * (samples(mid) + samples(mid + 1))
      end if
      q25 = samples(n_samples / 4 + 1)
      q75 = samples((3 * n_samples) / 4 + 1)
      stats%iqr_us = q75 - q25
   end subroutine compute_latency_stats

   !> Returns .true. if x is finite (neither IEEE NaN nor +/-Infinity).
   pure logical function is_finite_dp(x)
      real(dp), intent(in) :: x
      is_finite_dp = (x == x) .and. (abs(x) <= huge(1.0_dp))
   end function is_finite_dp

   !> Returns an IEEE-754 quiet NaN in double precision.
   pure real(dp) function quiet_nan()
      real(dp) :: z
      z = 0.0_dp
      quiet_nan = z / z
   end function quiet_nan

   !> Computes the maximum elementwise relative difference between two flat
   !! arrays a(1:n) and b(1:n):
   !!   max_i |a(i) - b(i)| / max(|a(i)|, |b(i)|, 1e-30)
   !! Propagates IEEE NaN immediately if any entry of a or b is non-finite.
   pure real(dp) function max_rel_diff(n, a, b)
      integer, intent(in) :: n
      real(dp), intent(in) :: a(n), b(n)
      integer :: i
      real(dp) :: denom, rel

      max_rel_diff = 0.0_dp
      do i = 1, n
         if ((.not. is_finite_dp(a(i))) .or. (.not. is_finite_dp(b(i)))) then
            max_rel_diff = quiet_nan()
            return
         end if
         denom = max(abs(a(i)), abs(b(i)), 1.0e-30_dp)
         rel = abs(a(i) - b(i)) / denom
         if ((rel /= rel) .or. (rel > max_rel_diff)) max_rel_diff = rel
      end do
   end function max_rel_diff

   !> Combines three stage relative errors max(d1, d2, d3) while propagating
   !! IEEE NaN if any input is NaN.
   pure real(dp) function max_err3(d1, d2, d3)
      real(dp), intent(in) :: d1, d2, d3
      if ((d1 /= d1) .or. (d2 /= d2) .or. (d3 /= d3)) then
         max_err3 = quiet_nan()
      else
         max_err3 = max(d1, d2, d3)
      end if
   end function max_err3

   !> Initializes the deterministic stellar thermodynamic profile across
   !! n_zones radial shells spanning:
   !! - log10(T/K) in [7.2, 9.8]
   !! - log10(rho / g cm^-3) in [4.0, 9.0]
   !! - abar in [4.0, 28.0] amu, zbar = 0.5 * abar
   subroutine init_thermo_grid(n_zones, grid)
      integer, intent(in) :: n_zones
      type(ThermoGrid), intent(out) :: grid
      integer :: k
      real(dp) :: denom, s, lt, lrho, a

      grid%n_zones = n_zones
      allocate(grid%log10_t(n_zones), grid%t(n_zones))
      allocate(grid%log10_rho(n_zones), grid%rho(n_zones))
      allocate(grid%abar(n_zones), grid%zbar(n_zones))

      denom = real(max(n_zones - 1, 1), dp)
      do k = 1, n_zones
         s = real(k - 1, dp) / denom
         lt = 7.2_dp + 2.6_dp * s
         lrho = 4.0_dp + 5.0_dp * s
         grid%log10_t(k) = lt; grid%t(k) = exp10(lt)
         grid%log10_rho(k) = lrho; grid%rho(k) = exp10(lrho)
         a = 4.0_dp + 24.0_dp * s
         grid%abar(k) = a; grid%zbar(k) = 0.5_dp * a
      end do
   end subroutine init_thermo_grid

   !> Deallocates ThermoGrid arrays.
   subroutine free_thermo_grid(grid)
      type(ThermoGrid), intent(inout) :: grid
      if (allocated(grid%log10_t)) deallocate(grid%log10_t, grid%t)
      if (allocated(grid%log10_rho)) deallocate(grid%log10_rho, grid%rho)
      if (allocated(grid%abar)) deallocate(grid%abar, grid%zbar)
   end subroutine free_thermo_grid

   !> Builds the deterministic 64 x 64 bicubic spline table using MESA's
   !! interp_mkbicub_db routine with not-a-knot (ibc = 0) boundary conditions.
   subroutine init_bicub_table(table)
      type(BicubTable), intent(out) :: table
      integer :: nx, ny, i, j, idx, ierr
      real(dp) :: xi, yj
      real(dp), allocatable :: bcxmin(:), bcxmax(:), bcymin(:), bcymax(:)

      nx = 64; ny = 64
      table%nx = nx; table%ny = ny
      allocate(table%x(nx), table%y(ny), table%f1(4 * nx * ny))
      allocate(bcxmin(ny), bcxmax(ny), bcymin(nx), bcymax(nx))
      bcxmin = 0.0_dp; bcxmax = 0.0_dp; bcymin = 0.0_dp; bcymax = 0.0_dp

      do i = 1, nx
         table%x(i) = 3.5_dp + 6.0_dp * real(i - 1, dp) / real(nx - 1, dp)
      end do
      do j = 1, ny
         table%y(j) = 7.0_dp + 3.0_dp * real(j - 1, dp) / real(ny - 1, dp)
      end do

      table%f1 = 0.0_dp
      do j = 1, ny
         yj = table%y(j)
         do i = 1, nx
            xi = table%x(i)
            idx = 4 * ((j - 1) * nx + (i - 1)) + 1
            table%f1(idx) = 0.25_dp * xi * xi + 0.15_dp * xi * yj &
               + 0.35_dp * yj * yj &
               + 0.1_dp * sinpi(0.2_dp * xi) * cospi(0.15_dp * yj)
         end do
      end do

      call interp_mkbicub_db( &
         table%x, nx, table%y, ny, table%f1, nx, &
         0, bcxmin, 0, bcxmax, 0, bcymin, 0, bcymax, &
         table%ilinx, table%iliny, ierr)
      if (ierr /= 0) stop 'interp_mkbicub_db failed in init_bicub_table'
      deallocate(bcxmin, bcxmax, bcymin, bcymax)
   end subroutine init_bicub_table

   !> Deallocates BicubTable storage.
   subroutine free_bicub_table(table)
      type(BicubTable), intent(inout) :: table
      if (allocated(table%x)) deallocate(table%x, table%y)
      if (associated(table%f1)) deallocate(table%f1)
      nullify(table%f1)
   end subroutine free_bicub_table

end module bench_harness
