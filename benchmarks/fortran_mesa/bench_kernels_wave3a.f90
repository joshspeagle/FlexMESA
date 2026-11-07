! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Module: bench_kernels_wave3a
!
! Four-stage Fortran MESA Wave 3a equation-of-state benchmarks
! (bench_eos_ideal, bench_eos_helm, bench_eos_pc, bench_eos_skye).
! -----------------------------------------------------------------------------
module bench_kernels_wave3a
   use const_def, only: dp, arg_not_provided
   use math_lib, only: exp10
   use chem_def, only: iso_name_length, num_chem_isos
   use chem_lib, only: chem_get_iso_id, basic_composition_info
   use eos_def, only: EoS_General_Info, num_eos_basic_results, &
      num_eos_d_dxa_results, i_lnPgas, i_eos_HELM, i_eos_PC, i_eos_Skye, &
      i_eos_ideal
   use eos_lib, only: eos_ptr, eosDT_get_component
   use eosDT_eval, only: Test_one_eosDT_component
   use omp_lib, only: omp_set_num_threads
   use bench_harness, only: STAGE3_THREADS, BenchRecord, now_us, &
      compute_latency_stats, max_rel_diff, max_err3
   implicit none
   private

   public :: bench_eos_ideal, bench_eos_helm, bench_eos_pc, bench_eos_skye

   character(len=iso_name_length), parameter :: WAVE2_ISO_NAMES(8) = [ &
      'h1      ', 'he3     ', 'he4     ', 'c12     ', &
      'n14     ', 'o16     ', 'ne20    ', 'mg24    ']

contains

   !> Executes Stages 0..3 for a single MESA EOS component (which_eos) across
   !! nz radial zones along log10(rho) = lrho0 + dlrho*s, log10(T) = lt0 + dlt*s.
   subroutine bench_eos_component_profile( &
         eos_handle, which_eos, wname, n_sp, chem_ids_in, xa_in, &
         lrho0, dlrho, lt0, dlt, nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: eos_handle, which_eos, n_sp, nz, n_warmup, n_repeats
      character(len=*), intent(in) :: wname
      integer, intent(in) :: chem_ids_in(n_sp)
      real(dp), intent(in) :: xa_in(n_sp), lrho0, dlrho, lt0, dlt
      type(BenchRecord), intent(out) :: rec

      type(EoS_General_Info), pointer :: rq
      integer, pointer :: chem_id(:), net_iso(:)
      integer :: i, k, rep, reps, ierr
      real(dp) :: denom, s, t0, xh, xhe, xz, abar, zbar, z2bar, z53bar, ye
      real(dp) :: mass_correction, sumx
      real(dp) :: dlnd(num_eos_basic_results), dlnt(num_eos_basic_results)
      real(dp) :: d_dxa_pub(num_eos_d_dxa_results, n_sp)
      real(dp) :: d_dxa_eos(num_eos_basic_results, n_sp)
      real(dp), allocatable :: lrhov(:), rhov(:), ltv(:), tv(:), samples(:)
      real(dp), allocatable :: tmp(:,:), out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      reps = max(n_repeats, 3)
      call eos_ptr(eos_handle, rq, ierr)
      allocate(chem_id(n_sp), net_iso(num_chem_isos))
      net_iso = 0
      do i = 1, n_sp
         chem_id(i) = chem_ids_in(i)
         net_iso(chem_ids_in(i)) = i
      end do
      call basic_composition_info( &
         n_sp, chem_id, xa_in, xh, xhe, xz, abar, zbar, z2bar, z53bar, &
         ye, mass_correction, sumx)

      allocate(lrhov(nz), rhov(nz), ltv(nz), tv(nz), samples(reps))
      allocate(out0(num_eos_basic_results, nz), out1(num_eos_basic_results, nz))
      allocate(out2(num_eos_basic_results, nz), out3(num_eos_basic_results, nz))
      denom = real(max(nz - 1, 1), dp)
      do k = 1, nz
         s = real(k - 1, dp) / denom
         lrhov(k) = lrho0 + dlrho * s; rhov(k) = exp10(lrhov(k))
         ltv(k) = lt0 + dlt * s; tv(k) = exp10(ltv(k))
      end do

      ! Warm up internal tables once before timing
      call Test_one_eosDT_component( &
         rq, which_eos, xz, xh, abar, zbar, n_sp, chem_id, net_iso, xa_in, &
         rhov(1), lrhov(1), tv(1), ltv(1), out0(:, 1), dlnd, dlnt, d_dxa_eos, ierr)

      ! Stage 0: Public eosDT_get_component with per-call output allocation
      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(num_eos_basic_results, nz))
         do k = 1, nz
            call eosDT_get_component( &
               eos_handle, which_eos, n_sp, chem_id, net_iso, xa_in, &
               rhov(k), arg_not_provided, tv(k), arg_not_provided, &
               tmp(:, k), dlnd, dlnt, d_dxa_pub, ierr)
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      ! Stage 1: Contiguous pre-allocated SoA output slice via eosDT_get_component
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call eosDT_get_component( &
               eos_handle, which_eos, n_sp, chem_id, net_iso, xa_in, &
               rhov(k), lrhov(k), tv(k), ltv(k), &
               out1(:, k), dlnd, dlnt, d_dxa_pub, ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      ! Stage 2: Hoisted composition moments & pre-allocated workspace
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call Test_one_eosDT_component( &
               rq, which_eos, xz, xh, abar, zbar, n_sp, chem_id, net_iso, xa_in, &
               rhov(k), lrhov(k), tv(k), ltv(k), &
               out2(:, k), dlnd, dlnt, d_dxa_eos, ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      ! Stage 3: 4-thread OpenMP data-parallel zone loop
      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) &
         !$omp private(k, dlnd, dlnt, d_dxa_eos, ierr) schedule(static)
         do k = 1, nz
            call Test_one_eosDT_component( &
               rq, which_eos, xz, xh, abar, zbar, n_sp, chem_id, net_iso, xa_in, &
               rhov(k), lrhov(k), tv(k), ltv(k), &
               out3(:, k), dlnd, dlnt, d_dxa_eos, ierr)
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = wname; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(num_eos_basic_results * nz, out0, out1), &
         max_rel_diff(num_eos_basic_results * nz, out0, out2), &
         max_rel_diff(num_eos_basic_results * nz, out0, out3))
      rec%checksum = sum(out2(i_lnPgas, :))
      deallocate(chem_id, net_iso, lrhov, rhov, ltv, tv, samples)
      deallocate(out0, out1, out2, out3)
   end subroutine bench_eos_component_profile

   !> Benchmarks neutral ideal-gas + radiation EOS (eos_ideal).
   subroutine bench_eos_ideal(eos_handle, nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: eos_handle, nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: chem_ids(8), i
      real(dp) :: z, x, y, xa(8)

      do i = 1, 8
         chem_ids(i) = chem_get_iso_id(trim(WAVE2_ISO_NAMES(i)))
      end do
      z = 0.014_dp; x = 0.70_dp; y = 1.0_dp - x - z
      xa = [x, 1.0e-4_dp * y, 0.9999_dp * y, 0.18_dp * z, &
         0.06_dp * z, 0.48_dp * z, 0.14_dp * z, 0.14_dp * z]
      call bench_eos_component_profile( &
         eos_handle, i_eos_ideal, 'eos_ideal', 8, chem_ids, xa, &
         -6.0_dp, 4.0_dp, 3.5_dp, 1.0_dp, nz, n_warmup, n_repeats, rec)
   end subroutine bench_eos_ideal

   !> Benchmarks Timmes & Swesty (2000) HELM EOS (eos_helm).
   subroutine bench_eos_helm(eos_handle, nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: eos_handle, nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: chem_ids(2)
      real(dp), parameter :: xa(2) = [0.5_dp, 0.5_dp]

      chem_ids = [chem_get_iso_id('c12'), chem_get_iso_id('o16')]
      call bench_eos_component_profile( &
         eos_handle, i_eos_HELM, 'eos_helm', 2, chem_ids, xa, &
         1.0_dp, 5.0_dp, 8.0_dp, 0.008_dp, nz, n_warmup, n_repeats, rec)
   end subroutine bench_eos_helm

   !> Benchmarks Potekhin & Chabrier (2010, 2013) PC EOS (eos_pc).
   subroutine bench_eos_pc(eos_handle, nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: eos_handle, nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: chem_ids(2)
      real(dp), parameter :: xa(2) = [0.5_dp, 0.5_dp]

      chem_ids = [chem_get_iso_id('c12'), chem_get_iso_id('o16')]
      call bench_eos_component_profile( &
         eos_handle, i_eos_PC, 'eos_pc', 2, chem_ids, xa, &
         4.5_dp, 3.0_dp, 6.2_dp, 1.2_dp, nz, n_warmup, n_repeats, rec)
   end subroutine bench_eos_pc

   !> Benchmarks Jermyn et al. (2021) third-order Skye EOS (eos_skye).
   subroutine bench_eos_skye(eos_handle, nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: eos_handle, nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: chem_ids(2)
      real(dp), parameter :: xa(2) = [0.5_dp, 0.5_dp]

      chem_ids = [chem_get_iso_id('c12'), chem_get_iso_id('o16')]
      call bench_eos_component_profile( &
         eos_handle, i_eos_Skye, 'eos_skye', 2, chem_ids, xa, &
         1.0_dp, 5.0_dp, 8.0_dp, 0.008_dp, nz, n_warmup, n_repeats, rec)
   end subroutine bench_eos_skye

end module bench_kernels_wave3a
