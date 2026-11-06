! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Module: bench_kernels_micro
!
! Four-stage Fortran MESA microphysics and leaf utility benchmarks:
! 1. bench_autodiff: Two-variable second-order forward-mode hyperdual
!    evaluation (auto_diff_real_2var_order2) across Stages 0..3.
! 2. bench_interp2d: 2D bicubic spline evaluation (interp_evbicub_db and
!    pre-validated Hermite evaluation fvbicub_db) across Stages 0..3.
! 3. bench_neu: Itoh et al. (1996) pair + plasmon thermal neutrino cooling
!    (neu_get) across Stages 0..3.
! -----------------------------------------------------------------------------
module bench_kernels_micro
   use const_def, only: dp
   use math_lib, only: safe_log10
   use auto_diff, only: auto_diff_real_2var_order2, assignment(=), &
      operator(+), operator(*), operator(/), exp, pow
   use interp_2d_lib_db, only: interp_evbicub_db
   use bicub_db, only: fvbicub_db
   use neu_def, only: num_neu_types, num_neu_rvs, pair_neu_type, plas_neu_type
   use neu_lib, only: neu_get
   use omp_lib, only: omp_set_num_threads
   use bench_harness, only: STAGE3_THREADS, BenchRecord, ThermoGrid, &
      BicubTable, now_us, compute_latency_stats, max_rel_diff, max_err3
   implicit none
   private

   public :: bench_autodiff
   public :: bench_interp2d
   public :: bench_neu

contains

   !> Evaluates f(x, y) = exp(0.1*x*y) + x^1.5 / (y + 2) using MESA's
   !! operator-overloaded auto_diff_real_2var_order2 derived type.
   !!
   !! Arguments:
   !! - x_val: First independent variable x (dimensionless).
   !! - y_val: Second independent variable y (dimensionless).
   !! - out6(6): Output array [val, df/dx, df/dy, d2f/dx2, d2f/dxdy, d2f/dy2].
   subroutine eval_autodiff_zone(x_val, y_val, out6)
      real(dp), intent(in) :: x_val, y_val
      real(dp), intent(out) :: out6(6)
      type(auto_diff_real_2var_order2) :: x, y, z

      x = x_val; x%d1val1 = 1.0_dp
      y = y_val; y%d1val2 = 1.0_dp
      z = exp(0.1_dp * x * y) + pow(x, 1.5_dp) / (y + 2.0_dp)
      out6 = [z%val, z%d1val1, z%d1val2, z%d2val1, z%d1val1_d1val2, z%d2val2]
   end subroutine eval_autodiff_zone

   !> Evaluates the fused analytical value, gradient, and Hessian of
   !! f(x, y) = exp(0.1*x*y) + x^1.5 / (y + 2) in out6(1:6).
   !!
   !! Arguments:
   !! - x: First independent variable x (dimensionless).
   !! - y: Second independent variable y (dimensionless).
   !! - out6(6): Output array [val, df/dx, df/dy, d2f/dx2, d2f/dxdy, d2f/dy2].
   pure subroutine eval_autodiff_fused(x, y, out6)
      real(dp), intent(in) :: x, y
      real(dp), intent(out) :: out6(6)
      real(dp) :: e_xy, inv_y2, inv_y2_2, inv_y2_3, sx, x15, q

      e_xy = exp(0.1_dp * x * y)
      inv_y2 = 1.0_dp / (y + 2.0_dp)
      inv_y2_2 = inv_y2 * inv_y2
      inv_y2_3 = inv_y2_2 * inv_y2
      sx = sqrt(x)
      x15 = x * sx
      q = x15 * inv_y2

      out6(1) = e_xy + q
      out6(2) = (0.1_dp * y) * e_xy + 1.5_dp * sx * inv_y2
      out6(3) = (0.1_dp * x) * e_xy - x15 * inv_y2_2
      out6(4) = (0.01_dp * y * y) * e_xy + (0.75_dp / sx) * inv_y2
      out6(5) = (0.1_dp + 0.01_dp * x * y) * e_xy - 1.5_dp * sx * inv_y2_2
      out6(6) = (0.01_dp * x * x) * e_xy + 2.0_dp * x15 * inv_y2_3
   end subroutine eval_autodiff_fused

   !> Benchmarks two-variable 2nd-order automatic differentiation (auto_diff)
   !! across Stages 0..3 over grid%n_zones radial shells:
   !! - Stage 0: Scalar zone loop with per-step buffer allocation (1 thread).
   !! - Stage 1: Contiguous pre-allocated SoA buffer (1 thread).
   !! - Stage 2: Fused analytical derivative evaluation (1 thread).
   !! - Stage 3: 4-thread OpenMP parallel execution across zones.
   subroutine bench_autodiff(grid, n_warmup, n_repeats, rec)
      type(ThermoGrid), intent(in) :: grid
      integer, intent(in) :: n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: nz, rep, reps, k
      real(dp) :: t0
      real(dp), allocatable :: samples(:), tmp(:,:)
      real(dp), allocatable :: out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      nz = grid%n_zones
      reps = max(n_repeats, 3)
      allocate(samples(reps))
      allocate(out0(6, nz), out1(6, nz), out2(6, nz), out3(6, nz))

      ! Stage 0: Scalar loop allocating temporary buffer per step (1 thread).
      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(6, nz))
         do k = 1, nz
            call eval_autodiff_zone( &
               grid%log10_t(k), grid%log10_rho(k), tmp(:, k))
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      ! Stage 1: Contiguous pre-allocated SoA buffer (1 thread).
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call eval_autodiff_zone( &
               grid%log10_t(k), grid%log10_rho(k), out1(:, k))
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      ! Stage 2: Fused derivative evaluation over contiguous array (1 thread).
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call eval_autodiff_fused( &
               grid%log10_t(k), grid%log10_rho(k), out2(:, k))
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      ! Stage 3: 4-thread OpenMP parallel loop across stellar zones.
      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k) schedule(static)
         do k = 1, nz
            call eval_autodiff_fused( &
               grid%log10_t(k), grid%log10_rho(k), out3(:, k))
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'auto_diff'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(6 * nz, out0, out1), &
         max_rel_diff(6 * nz, out0, out2), &
         max_rel_diff(6 * nz, out0, out3))
      rec%checksum = sum(out2(1, :))
      deallocate(samples, out0, out1, out2, out3)
   end subroutine bench_autodiff

   !> Pre-validated uniform-grid bicubic spline evaluation for a single zone
   !! calling MESA's fvbicub_db kernel directly without range/spacing checks.
   !!
   !! Arguments:
   !! - xq: Query log10(rho / g cm^-3) coordinate.
   !! - yq: Query log10(T / K) coordinate.
   !! - table: Precomputed 64 x 64 BicubTable.
   !! - out6(6): Output [f, df/dx, df/dy, d2f/dx2, d2f/dy2, d2f/dxdy].
   subroutine eval_bicub_prevalidated_zone(xq, yq, table, out6)
      real(dp), intent(in) :: xq, yq
      type(BicubTable), intent(in) :: table
      real(dp), intent(out) :: out6(6)
      integer, parameter :: ict(6) = [1, 1, 1, 1, 1, 1]
      integer :: nx, ny, ii(1), jj(1)
      real(dp) :: dx, dy, xp(1), yp(1), hx(1), hxi(1), hy(1), hyi(1)

      nx = table%nx; ny = table%ny
      dx = (table%x(nx) - table%x(1)) / real(nx - 1, dp)
      dy = (table%y(ny) - table%y(1)) / real(ny - 1, dp)
      ii(1) = min(max(int((xq - table%x(1)) / dx) + 1, 1), nx - 1)
      jj(1) = min(max(int((yq - table%y(1)) / dy) + 1, 1), ny - 1)
      hx(1) = table%x(ii(1) + 1) - table%x(ii(1))
      hxi(1) = 1.0_dp / hx(1)
      xp(1) = (xq - table%x(ii(1))) * hxi(1)
      hy(1) = table%y(jj(1) + 1) - table%y(jj(1))
      hyi(1) = 1.0_dp / hy(1)
      yp(1) = (yq - table%y(jj(1))) * hyi(1)
      call fvbicub_db( &
         ict, 1, 1, out6, ii, jj, xp, yp, hx, hxi, hy, hyi, &
         table%f1, nx, ny)
   end subroutine eval_bicub_prevalidated_zone

   !> Benchmarks 2D bicubic spline evaluation (interp_2d) across Stages 0..3:
   !! - Stage 0: Per-step buffer allocation calling interp_evbicub_db (1t).
   !! - Stage 1: Contiguous pre-allocated buffer calling interp_evbicub_db (1t).
   !! - Stage 2: Direct pre-validated uniform-grid fvbicub_db evaluation (1t).
   !! - Stage 3: 4-thread OpenMP parallel execution across zones.
   subroutine bench_interp2d(grid, table, n_warmup, n_repeats, rec)
      type(ThermoGrid), intent(in) :: grid
      type(BicubTable), intent(inout) :: table
      integer, intent(in) :: n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer, parameter :: ict(6) = [1, 1, 1, 1, 1, 1]
      integer :: nz, rep, reps, k, ierr
      real(dp) :: t0
      real(dp), allocatable :: samples(:), tmp(:,:)
      real(dp), allocatable :: out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      nz = grid%n_zones
      reps = max(n_repeats, 3)
      allocate(samples(reps))
      allocate(out0(6, nz), out1(6, nz), out2(6, nz), out3(6, nz))

      ! Stage 0: Per-step buffer allocation calling interp_evbicub_db.
      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(6, nz))
         do k = 1, nz
            call interp_evbicub_db( &
               grid%log10_rho(k), grid%log10_t(k), table%x, table%nx, &
               table%y, table%ny, table%ilinx, table%iliny, table%f1, &
               table%nx, ict, tmp(:, k), ierr)
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      ! Stage 1: Contiguous pre-allocated output calling interp_evbicub_db.
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call interp_evbicub_db( &
               grid%log10_rho(k), grid%log10_t(k), table%x, table%nx, &
               table%y, table%ny, table%ilinx, table%iliny, table%f1, &
               table%nx, ict, out1(:, k), ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      ! Stage 2: Direct pre-validated uniform-grid fvbicub_db evaluation.
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call eval_bicub_prevalidated_zone( &
               grid%log10_rho(k), grid%log10_t(k), table, out2(:, k))
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      ! Stage 3: 4-thread OpenMP parallel execution across zones.
      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k) schedule(static)
         do k = 1, nz
            call eval_bicub_prevalidated_zone( &
               grid%log10_rho(k), grid%log10_t(k), table, out3(:, k))
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'interp_2d'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(6 * nz, out0, out1), &
         max_rel_diff(6 * nz, out0, out2), &
         max_rel_diff(6 * nz, out0, out3))
      rec%checksum = sum(out2(1, :))
      deallocate(samples, out0, out1, out2, out3)
   end subroutine bench_interp2d

   !> Benchmarks Itoh et al. (1996) pair + plasmon thermal neutrino cooling
   !! (neu_get) across Stages 0..3 over grid%n_zones radial shells:
   !! - Stage 0: Per-step buffer allocation recomputing log10(T)/log10(Rho).
   !! - Stage 1: Contiguous buffer recomputing log10(T)/log10(Rho) (1 thread).
   !! - Stage 2: Precomputed log10_T and log10_Rho over contiguous slice (1t).
   !! - Stage 3: 4-thread OpenMP parallel execution across zones.
   subroutine bench_neu(grid, n_warmup, n_repeats, rec)
      type(ThermoGrid), intent(in) :: grid
      integer, intent(in) :: n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: nz, rep, reps, k, info
      logical :: flags(num_neu_types)
      real(dp) :: t0, lt, lrho, sources(num_neu_types, num_neu_rvs)
      real(dp), allocatable :: samples(:), tmp(:,:)
      real(dp), allocatable :: out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      nz = grid%n_zones
      reps = max(n_repeats, 3)
      flags = .false.
      flags(pair_neu_type) = .true.; flags(plas_neu_type) = .true.
      allocate(samples(reps))
      allocate(out0(5, nz), out1(5, nz), out2(5, nz), out3(5, nz))

      ! Stage 0: Recompute log10(T)/log10(Rho) and allocate output per step.
      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(5, nz))
         do k = 1, nz
            lt = safe_log10(grid%t(k)); lrho = safe_log10(grid%rho(k))
            call neu_get( &
               grid%t(k), lt, grid%rho(k), lrho, grid%abar(k), grid%zbar(k), &
               7.5_dp, flags, tmp(:, k), sources, info)
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      ! Stage 1: Contiguous output buffer recomputing log10(T)/log10(Rho).
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            lt = safe_log10(grid%t(k)); lrho = safe_log10(grid%rho(k))
            call neu_get( &
               grid%t(k), lt, grid%rho(k), lrho, grid%abar(k), grid%zbar(k), &
               7.5_dp, flags, out1(:, k), sources, info)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      ! Stage 2: Precomputed log10_T and log10_Rho over contiguous slice.
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call neu_get( &
               grid%t(k), grid%log10_t(k), grid%rho(k), grid%log10_rho(k), &
               grid%abar(k), grid%zbar(k), 7.5_dp, flags, out2(:, k), &
               sources, info)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      ! Stage 3: 4-thread OpenMP parallel execution across zones.
      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k, sources, info) &
         !$omp schedule(static)
         do k = 1, nz
            call neu_get( &
               grid%t(k), grid%log10_t(k), grid%rho(k), grid%log10_rho(k), &
               grid%abar(k), grid%zbar(k), 7.5_dp, flags, out3(:, k), &
               sources, info)
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'neu'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(5 * nz, out0, out1), &
         max_rel_diff(5 * nz, out0, out2), &
         max_rel_diff(5 * nz, out0, out3))
      rec%checksum = sum(out2(1, :))
      deallocate(samples, out0, out1, out2, out3)
   end subroutine bench_neu

end module bench_kernels_micro
