! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Module: bench_kernels_solver
!
! Four-stage Fortran MESA linear solver and convective transport benchmarks:
! 1. bench_mtx_block_tridiag: 5 x 5 block-tridiagonal Henyey system solve
!    comparing MESA's bcyclic_dble_decsolblk (1-thread and 4-thread OpenMP
!    odd-even cyclic reduction) and pre-allocated block-Thomas across
!    Stages 0..3.
! 2. bench_turb_mlt: Cox (1968) Mixing-Length Theory (set_MLT) with 33-tangent
!    auto_diff_real_star_order1 sensitivities across Stages 0..3.
! 3. bench_turb_tdc: Kuhfuss (1986) 1D Time-Dependent Convection (set_TDC)
!    with auto_diff_real_star_order1 sensitivities across Stages 0..3.
! -----------------------------------------------------------------------------
module bench_kernels_solver
   use const_def, only: dp, kerg, mp, msun, rsun, lsun, standard_cgrav
   use auto_diff, only: auto_diff_real_star_order1, assignment(=)
   use mtx_lib, only: bcyclic_dble_decsolblk, bcyclic_dble_work_sizes
   use turb, only: set_MLT, set_TDC
   use omp_lib, only: omp_set_num_threads
   use bench_harness, only: STAGE3_THREADS, LatencyStats, BenchRecord, &
      ThermoGrid, now_us, compute_latency_stats, max_rel_diff, max_err3
   use bench_tridiag, only: make_block_tridiag_fixture, fast_block_thomas_solve
   implicit none
   private

   ! Solar surface gravity and ideal monatomic gas constants in CGS units.
   real(dp), parameter :: GRAV_SOLAR = msun * standard_cgrav / (rsun * rsun)
   real(dp), parameter :: RGAS_IDEAL = kerg / mp
   real(dp), parameter :: CP_IDEAL = 2.5_dp * RGAS_IDEAL

   public :: bench_mtx_block_tridiag
   public :: bench_turb_mlt
   public :: bench_turb_tdc

contains

   !> Benchmarks 5 x 5 block-tridiagonal Henyey solves across Stages 0..3
   !! and records MESA's 1-thread and 4-thread bcyclic_dble_decsolblk timings:
   !! - Stage 0: bcyclic_dble_decsolblk with per-step factor + solve + dealloc.
   !! - Stage 1: bcyclic_dble_decsolblk reusing persistent odd_storage (1t).
   !! - Stage 2: Pre-allocated sequential block-Thomas (1 thread).
   !! - Stage 3: 4-thread concurrent block-Thomas (amortized per system).
   !!
   !! Arguments:
   !! - nz: Number of radial zones N_zones.
   !! - n_warmup, n_repeats: Untimed warmup and timed iteration counts.
   !! - rec: Output BenchRecord with Stages 0..3 latencies and checksum.
   !! - bcyclic_1t_us, bcyclic_4t_us: 1-thread and 4-thread BCYCLIC medians.
   subroutine bench_mtx_block_tridiag( &
         nz, n_warmup, n_repeats, rec, bcyclic_1t_us, bcyclic_4t_us)
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      real(dp), intent(out) :: bcyclic_1t_us, bcyclic_4t_us
      integer, parameter :: nvar = 5
      integer :: rep, reps, t, lrd, lid, ierr
      real(dp) :: t0, d3, err_t
      type(LatencyStats) :: bcyclic_4t_stats
      real(dp), allocatable :: samples(:)
      real(dp), allocatable :: lblk(:,:,:), dblk(:,:,:), ublk(:,:,:), rhs(:,:)
      real(dp), allocatable :: out0(:,:), out1(:,:), out2(:,:), out3(:,:,:)
      real(dp), allocatable :: dw(:,:,:,:), uw(:,:,:,:)
      integer, allocatable :: ipiv2(:,:), ipiv3(:,:,:)
      real(dp), pointer :: l1(:), d1(:), u1(:), b1(:), rpar(:)
      integer, pointer :: ipiv1(:), ipar(:)

      ! Allocate block-tridiagonal fixtures and per-stage workspaces.
      reps = max(n_repeats, 3)
      allocate(samples(reps), lblk(5, 5, nz), dblk(5, 5, nz), ublk(5, 5, nz))
      allocate(rhs(5, nz), out0(5, nz), out1(5, nz), out2(5, nz))
      allocate(out3(5, nz, STAGE3_THREADS), ipiv2(5, nz))
      allocate(dw(5, 5, nz, STAGE3_THREADS), uw(5, 5, nz, STAGE3_THREADS))
      allocate(ipiv3(5, nz, STAGE3_THREADS))
      call make_block_tridiag_fixture(nz, lblk, dblk, ublk, rhs)

      call bcyclic_dble_work_sizes(nvar, nz, lrd, lid)
      allocate(l1(25 * nz), d1(25 * nz), u1(25 * nz), b1(5 * nz))
      allocate(ipiv1(5 * nz), rpar(max(lrd, 1)), ipar(max(lid, 1)))

      ! Stage 0: bcyclic_dble_decsolblk with per-step factor + solve + dealloc.
      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         l1 = reshape(lblk, [25 * nz]); d1 = reshape(dblk, [25 * nz])
         u1 = reshape(ublk, [25 * nz]); b1 = reshape(rhs, [5 * nz])
         call bcyclic_dble_decsolblk( &
            0, 0, nvar, nz, l1, d1, u1, b1, ipiv1, lrd, rpar, lid, ipar, ierr)
         call bcyclic_dble_decsolblk( &
            1, 0, nvar, nz, l1, d1, u1, b1, ipiv1, lrd, rpar, lid, ipar, ierr)
         call bcyclic_dble_decsolblk( &
            2, 0, nvar, nz, l1, d1, u1, b1, ipiv1, lrd, rpar, lid, ipar, ierr)
         out0 = reshape(b1, [5, nz])
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      ! Stage 1: bcyclic_dble_decsolblk reusing persistent odd_storage (1t).
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         l1 = reshape(lblk, [25 * nz]); d1 = reshape(dblk, [25 * nz])
         u1 = reshape(ublk, [25 * nz]); b1 = reshape(rhs, [5 * nz])
         call bcyclic_dble_decsolblk( &
            0, 0, nvar, nz, l1, d1, u1, b1, ipiv1, lrd, rpar, lid, ipar, ierr)
         call bcyclic_dble_decsolblk( &
            1, 0, nvar, nz, l1, d1, u1, b1, ipiv1, lrd, rpar, lid, ipar, ierr)
         out1 = reshape(b1, [5, nz])
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)
      bcyclic_1t_us = rec%stage1%median_us

      ! Measure 4-thread OpenMP single-matrix bcyclic_dble_decsolblk.
      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         l1 = reshape(lblk, [25 * nz]); d1 = reshape(dblk, [25 * nz])
         u1 = reshape(ublk, [25 * nz]); b1 = reshape(rhs, [5 * nz])
         call bcyclic_dble_decsolblk( &
            0, 0, nvar, nz, l1, d1, u1, b1, ipiv1, lrd, rpar, lid, ipar, ierr)
         call bcyclic_dble_decsolblk( &
            1, 0, nvar, nz, l1, d1, u1, b1, ipiv1, lrd, rpar, lid, ipar, ierr)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call bcyclic_dble_decsolblk( &
         2, 0, nvar, nz, l1, d1, u1, b1, ipiv1, lrd, rpar, lid, ipar, ierr)
      call compute_latency_stats(samples, reps, bcyclic_4t_stats)
      bcyclic_4t_us = bcyclic_4t_stats%median_us

      ! Stage 2: Pre-allocated sequential block-Thomas (1 thread).
      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         dw(:, :, :, 1) = dblk; uw(:, :, :, 1) = ublk; out2 = rhs
         call fast_block_thomas_solve( &
            nz, lblk, dw(:, :, :, 1), uw(:, :, :, 1), ipiv2, out2)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      ! Stage 3: 4-thread concurrent block-Thomas (amortized per system).
      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(t) schedule(static, 1)
         do t = 1, STAGE3_THREADS
            dw(:, :, :, t) = dblk; uw(:, :, :, t) = ublk; out3(:, :, t) = rhs
            call fast_block_thomas_solve( &
               nz, lblk, dw(:, :, :, t), uw(:, :, :, t), &
               ipiv3(:, :, t), out3(:, :, t))
         end do
         !$omp end parallel do
         if (rep > n_warmup) then
            samples(rep - n_warmup) = (now_us() - t0) / real(STAGE3_THREADS, dp)
         end if
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      ! Verify solution parity across all stages and threads.
      d3 = 0.0_dp
      do t = 1, STAGE3_THREADS
         err_t = max_rel_diff(5 * nz, out0, out3(:, :, t))
         d3 = max_err3(d3, err_t, 0.0_dp)
      end do
      rec%workload = 'mtx_tridiag'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(5 * nz, out0, out1), &
         max_rel_diff(5 * nz, out0, out2), d3)
      rec%checksum = sum(out2)
      deallocate(l1, d1, u1, b1, ipiv1, rpar, ipar)
      deallocate(samples, lblk, dblk, ublk, rhs, out0, out1, out2, out3)
      deallocate(dw, uw, ipiv2, ipiv3)
   end subroutine bench_mtx_block_tridiag

   !> Evaluates Cox (1968) MLT (set_MLT, if is_tdc=.false.) or Kuhfuss (1986)
   !! TDC (set_TDC, if is_tdc=.true.) for shell coordinate s_frac in [0, 1].
   !! Propagates 6 independent sensitivities through MESA's 33-component
   !! auto_diff_real_star_order1 derived type:
   !! - d1Array(1): gradr
   !! - d1Array(2): grada
   !! - d1Array(3): gradL
   !! - d1Array(4): Lambda (MLT) or scale_height (TDC)
   !! - d1Array(5): T (K)
   !! - d1Array(6): rho (g cm^-3)
   !! Returns [gradT, D, conv_vel, d(gradT)/d(gradr)] in out4(1:4).
   subroutine eval_conv_zone(s_frac, is_tdc, out4)
      real(dp), intent(in) :: s_frac
      logical, intent(in) :: is_tdc
      real(dp), intent(out) :: out4(4)
      real(dp) :: t_val, rho_val, p_val, hp_val, gradr_val, scale_val, mult
      type(auto_diff_real_star_order1) :: chi_t, chi_rho, gradr, r, p, t, rho
      type(auto_diff_real_star_order1) :: dv, cp, opacity, scale_height, lambda
      type(auto_diff_real_star_order1) :: gradl, grada, eq_div_w, grav, energy
      type(auto_diff_real_star_order1) :: gamma, conv_vel, d, y_face, grad_t
      integer :: tdc_num_iters, mixing_type, ierr

      mult = merge(0.2_dp, 0.5_dp, is_tdc)
      t_val = 1.0e5_dp * (1.0_dp + mult * s_frac)
      rho_val = 1.0e-5_dp * (1.0_dp + mult * s_frac)
      p_val = rho_val * t_val * RGAS_IDEAL
      hp_val = p_val / (rho_val * GRAV_SOLAR)
      gradr_val = merge(0.42_dp + 0.08_dp * s_frac, &
         0.45_dp + 0.25_dp * s_frac, is_tdc)
      scale_val = 70.0_dp * lsun * 1.0e-3_dp

      chi_t = 1.0_dp; chi_rho = 1.0_dp; r = rsun; p = p_val; dv = 0.0_dp
      cp = CP_IDEAL; opacity = 1.0_dp; eq_div_w = 0.0_dp
      grav = GRAV_SOLAR; energy = 0.0_dp
      gradr = gradr_val; gradr%d1Array(1) = 1.0_dp
      grada = 0.40_dp; grada%d1Array(2) = 1.0_dp
      gradl = 0.40_dp; gradl%d1Array(3) = 1.0_dp
      scale_height = hp_val; scale_height%d1Array(4) = 1.0_dp
      lambda = 2.0_dp * hp_val; lambda%d1Array(4) = 1.0_dp
      t = t_val; t%d1Array(5) = 1.0_dp
      rho = rho_val; rho%d1Array(6) = 1.0_dp

      if (is_tdc) then
         call set_TDC( &
            1.0e5_dp, 2.0_dp, 1.0_dp, 0.0_dp, 0.0_dp, 1.0e4_dp, &
            standard_cgrav, msun, .false., mixing_type, scale_val, &
            chi_t, chi_rho, gradr, r, p, t, rho, dv, cp, opacity, &
            scale_height, gradl, grada, conv_vel, d, y_face, grad_t, &
            tdc_num_iters, 1.0e99_dp, eq_div_w, grav, .true., 1.0_dp, &
            1.0_dp, .false., energy, 0.0_dp, ierr)
      else
         call set_MLT( &
            'Cox', 2.0_dp, 0.0_dp, 0.0_dp, chi_t, chi_rho, cp, grav, lambda, &
            rho, p, t, opacity, gradr, grada, gradl, gamma, grad_t, y_face, &
            conv_vel, d, mixing_type, 1.0e99_dp, ierr)
      end if
      out4 = [grad_t%val, d%val, conv_vel%val, grad_t%d1Array(1)]
   end subroutine eval_conv_zone

   !> Shared four-stage benchmark runner for MLT (is_tdc=.false.) and TDC
   !! (is_tdc=.true.) across n_zones radial shells:
   !! - Stage 0: Per-step output allocation over scalar zone loop (1 thread).
   !! - Stage 1: Contiguous pre-allocated SoA buffer (1 thread).
   !! - Stage 2: Precomputed radial coordinate array over contiguous slice.
   !! - Stage 3: 4-thread OpenMP parallel execution across radial zones.
   subroutine run_conv_bench(workload, is_tdc, grid, n_warmup, n_repeats, rec)
      character(len=*), intent(in) :: workload
      logical, intent(in) :: is_tdc
      type(ThermoGrid), intent(in) :: grid
      integer, intent(in) :: n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: nz, rep, reps, k
      real(dp) :: denom, t0
      real(dp), allocatable :: samples(:), s_arr(:), tmp(:,:)
      real(dp), allocatable :: out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      nz = grid%n_zones
      reps = max(n_repeats, 3)
      denom = real(max(nz - 1, 1), dp)
      allocate(samples(reps), s_arr(nz))
      allocate(out0(4, nz), out1(4, nz), out2(4, nz), out3(4, nz))
      do k = 1, nz
         s_arr(k) = real(k - 1, dp) / denom
      end do

      ! Stage 0: Per-step output allocation over scalar zone loop (1 thread).
      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(4, nz))
         do k = 1, nz
            call eval_conv_zone(real(k - 1, dp) / denom, is_tdc, tmp(:, k))
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      ! Stage 1: Contiguous pre-allocated SoA buffer (1 thread).
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call eval_conv_zone(real(k - 1, dp) / denom, is_tdc, out1(:, k))
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      ! Stage 2: Precomputed radial coordinate array over contiguous slice.
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call eval_conv_zone(s_arr(k), is_tdc, out2(:, k))
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      ! Stage 3: 4-thread OpenMP parallel execution across radial zones.
      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k) schedule(dynamic, 4)
         do k = 1, nz
            call eval_conv_zone(s_arr(k), is_tdc, out3(:, k))
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = workload; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(4 * nz, out0, out1), &
         max_rel_diff(4 * nz, out0, out2), &
         max_rel_diff(4 * nz, out0, out3))
      rec%checksum = sum(out2(1, :))
      deallocate(samples, s_arr, out0, out1, out2, out3)
   end subroutine run_conv_bench

   !> Benchmarks Cox Mixing-Length Theory (set_MLT) across Stages 0..3.
   subroutine bench_turb_mlt(grid, n_warmup, n_repeats, rec)
      type(ThermoGrid), intent(in) :: grid
      integer, intent(in) :: n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      call run_conv_bench('turb_mlt', .false., grid, n_warmup, n_repeats, rec)
   end subroutine bench_turb_mlt

   !> Benchmarks Kuhfuss 1D Time-Dependent Convection (set_TDC) across
   !! Stages 0..3.
   subroutine bench_turb_tdc(grid, n_warmup, n_repeats, rec)
      type(ThermoGrid), intent(in) :: grid
      integer, intent(in) :: n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      call run_conv_bench('turb_tdc', .true., grid, n_warmup, n_repeats, rec)
   end subroutine bench_turb_tdc

end module bench_kernels_solver
