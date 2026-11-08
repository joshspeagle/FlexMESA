! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Module: bench_kernels_wave3b
!
! Four-stage Fortran MESA Wave 3b stellar microphysics benchmarks
! (bench_eos_4d, bench_kap_fixed, bench_rates_screen, bench_net_eval,
! bench_atm_table).
! -----------------------------------------------------------------------------
module bench_kernels_wave3b
   use const_def, only: dp, pi, crad, clight, standard_cgrav, Msun, Rsun
   use math_lib, only: exp10, safe_log10
   use chem_def, only: num_categories, chem_isos, num_chem_isos
   use chem_lib, only: chem_get_iso_id, basic_composition_info
   use eos_def, only: i_eos_OPAL_SCVH
   use kap_def, only: Kap_General_Info, Kap_gn93, num_kap_fracs
   use kap_lib, only: kap_get, kap_ptr, kap_setup_tables
   use kap_eval, only: Get_kap_Results
   use rates_def, only: reaclib_rates, extended_screening, Screen_Info, &
      std_reaction_Qs, std_reaction_neuQs
   use rates_lib, only: reaclib_indices_for_reaction, reaclib_reaction_rates, &
      screen_init_AZ_info, screen_set_context, screen_pair
   use net_def, only: Net_General_Info, Net_Info, get_net_ptr
   use net_lib, only: net_get
   use atm_def, only: ATM_TABLE_PHOTOSPHERE
   use atm_lib, only: atm_eval_table
   use omp_lib, only: omp_set_num_threads
   use bench_harness, only: STAGE3_THREADS, BenchRecord, now_us, &
      compute_latency_stats, max_rel_diff, max_err3
   use bench_kernels_wave3a, only: bench_eos_component_profile
   implicit none
   private

   public :: bench_eos_4d, bench_kap_fixed, bench_rates_screen, &
      bench_net_eval, bench_atm_table

contains

   !> Benchmarks 4D OPAL/SCVH tabulated equation of state (eos_4d).
   subroutine bench_eos_4d(eos_handle, nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: eos_handle, nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: chem_ids(7)
      real(dp), parameter :: xa(7) = [ &
         0.70_dp, 0.28_dp, 0.004_dp, 0.001_dp, 0.009_dp, 0.003_dp, 0.003_dp]

      chem_ids = [ &
         chem_get_iso_id('h1'), chem_get_iso_id('he4'), &
         chem_get_iso_id('c12'), chem_get_iso_id('n14'), &
         chem_get_iso_id('o16'), chem_get_iso_id('ne20'), &
         chem_get_iso_id('mg24')]
      call bench_eos_component_profile( &
         eos_handle, i_eos_OPAL_SCVH, 'eos_4d', 7, chem_ids, xa, &
         -2.0_dp, 0.01_dp, 5.0_dp, 0.01_dp, nz, n_warmup, n_repeats, rec)
   end subroutine bench_eos_4d

   !> Benchmarks Type 1 fixed-metal + conductive + Compton opacity (kap_fixed).
   subroutine bench_kap_fixed(kap_handle, nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: kap_handle, nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec

      type(Kap_General_Info), pointer :: rq
      integer, pointer :: chem_id(:), net_iso(:)
      integer :: k, rep, reps, ierr
      real(dp) :: denom, s, t0, xh, xhe, z, abar, zbar, z2bar, z53bar, ye
      real(dp) :: mass_correction, sumx, kap, dlnkap_dlnd, dlnkap_dlnt
      real(dp) :: kap_fracs(num_kap_fracs), dlnkap_dxa(7)
      real(dp), parameter :: xa(7) = [ &
         0.70_dp, 0.28_dp, 0.004_dp, 0.001_dp, 0.009_dp, 0.003_dp, 0.003_dp]
      real(dp), parameter :: lnfree_e = -0.1604428_dp, eta = -15.72947_dp
      real(dp), parameter :: d_eta_dlnd = 1.0_dp, d_eta_dlnt = -1.5_dp
      real(dp), allocatable :: lrhov(:), ltv(:), samples(:)
      real(dp), allocatable :: tmp(:,:), out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      reps = max(n_repeats, 3)
      call kap_ptr(kap_handle, rq, ierr)
      rq%Zbase = 0.02_dp
      allocate(chem_id(7), net_iso(num_chem_isos))
      chem_id = [ &
         chem_get_iso_id('h1'), chem_get_iso_id('he4'), &
         chem_get_iso_id('c12'), chem_get_iso_id('n14'), &
         chem_get_iso_id('o16'), chem_get_iso_id('ne20'), &
         chem_get_iso_id('mg24')]
      net_iso = 0
      do k = 1, 7
         net_iso(chem_id(k)) = k
      end do
      call basic_composition_info( &
         7, chem_id, xa, xh, xhe, z, abar, zbar, z2bar, z53bar, &
         ye, mass_correction, sumx)

      allocate(lrhov(nz), ltv(nz), samples(reps))
      allocate(out0(3, nz), out1(3, nz), out2(3, nz), out3(3, nz))
      denom = real(max(nz - 1, 1), dp)
      do k = 1, nz
         s = real(k - 1, dp) / denom
         lrhov(k) = -6.0_dp + 0.04_dp * s
         ltv(k) = 6.0_dp + 0.04_dp * s
      end do

      call Get_kap_Results( &
         rq, zbar, xh, z, 0.004_dp, 0.001_dp, 0.009_dp, 0.003_dp, &
         lrhov(1), ltv(1), lnfree_e, 0.0_dp, 0.0_dp, &
         eta, d_eta_dlnd, d_eta_dlnt, kap_fracs, kap, dlnkap_dlnd, dlnkap_dlnt, ierr)

      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(3, nz))
         do k = 1, nz
            call kap_get( &
               kap_handle, 7, chem_id, net_iso, xa, lrhov(k), ltv(k), &
               lnfree_e, 0.0_dp, 0.0_dp, eta, d_eta_dlnd, d_eta_dlnt, &
               kap_fracs, tmp(1, k), tmp(2, k), tmp(3, k), dlnkap_dxa, ierr)
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call kap_get( &
               kap_handle, 7, chem_id, net_iso, xa, lrhov(k), ltv(k), &
               lnfree_e, 0.0_dp, 0.0_dp, eta, d_eta_dlnd, d_eta_dlnt, &
               kap_fracs, out1(1, k), out1(2, k), out1(3, k), dlnkap_dxa, ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call Get_kap_Results( &
               rq, zbar, xh, z, 0.004_dp, 0.001_dp, 0.009_dp, 0.003_dp, &
               lrhov(k), ltv(k), lnfree_e, 0.0_dp, 0.0_dp, &
               eta, d_eta_dlnd, d_eta_dlnt, kap_fracs, &
               out2(1, k), out2(2, k), out2(3, k), ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k, kap_fracs, ierr) schedule(static)
         do k = 1, nz
            call Get_kap_Results( &
               rq, zbar, xh, z, 0.004_dp, 0.001_dp, 0.009_dp, 0.003_dp, &
               lrhov(k), ltv(k), lnfree_e, 0.0_dp, 0.0_dp, &
               eta, d_eta_dlnd, d_eta_dlnt, kap_fracs, &
               out3(1, k), out3(2, k), out3(3, k), ierr)
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'kap_fixed'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(3 * nz, out0, out1), &
         max_rel_diff(3 * nz, out0, out2), &
         max_rel_diff(3 * nz, out0, out3))
      rec%checksum = sum(out2(1, :))
      deallocate(chem_id, net_iso, lrhov, ltv, samples, out0, out1, out2, out3)
   end subroutine bench_kap_fixed

   !> Benchmarks REACLIB r_c12_ag_o16 rate + extended screen5 (rates_screen).
   subroutine bench_rates_screen(nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec

      type(Screen_Info) :: sc
      integer :: k, rep, reps, ierr, lo, hi
      real(dp) :: denom, s, t0, t9, temp, den
      real(dp) :: lambda, dlambda_dlnt, rlambda, drlambda_dlnt
      real(dp) :: scor, dscor_dt, dscor_dd
      real(dp) :: zs13, zhat, zhat2, lzav, aznut, zs13inv
      real(dp), parameter :: zbar = 6.857142857142857_dp
      real(dp), parameter :: abar = 13.714285714285714_dp
      real(dp), parameter :: z2bar = 47.42857142857143_dp
      real(dp), parameter :: y_co(2) = [0.5_dp / 12.0_dp, 0.5_dp / 16.0_dp]
      real(dp) :: iso_z158(2)
      real(dp), allocatable :: lrhov(:), rhov(:), ltv(:), tv(:), samples(:)
      real(dp), allocatable :: tmp(:,:), out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      reps = max(n_repeats, 3)
      ierr = 0
      iso_z158 = [6.0_dp**1.58_dp, 8.0_dp**1.58_dp]
      call reaclib_indices_for_reaction('r_c12_ag_o16', reaclib_rates, lo, hi, ierr)
      call screen_init_AZ_info( &
         12.0_dp, 6.0_dp, 4.0_dp, 2.0_dp, &
         zs13, zhat, zhat2, lzav, aznut, zs13inv, ierr)

      allocate(lrhov(nz), rhov(nz), ltv(nz), tv(nz), samples(reps))
      allocate(out0(3, nz), out1(3, nz), out2(3, nz), out3(3, nz))
      denom = real(max(nz - 1, 1), dp)
      do k = 1, nz
         s = real(k - 1, dp) / denom
         lrhov(k) = 2.0_dp + 3.0_dp * s; rhov(k) = exp10(lrhov(k))
         ltv(k) = 8.1_dp + 0.4_dp * s; tv(k) = exp10(ltv(k))
      end do

      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(3, nz))
         do k = 1, nz
            temp = tv(k); den = rhov(k); t9 = temp * 1.0e-9_dp
            call reaclib_reaction_rates( &
               lo, hi, t9, reaclib_rates, chem_isos, .true., &
               lambda, dlambda_dlnt, rlambda, drlambda_dlnt, ierr)
            call screen_init_AZ_info( &
               12.0_dp, 6.0_dp, 4.0_dp, 2.0_dp, &
               zs13, zhat, zhat2, lzav, aznut, zs13inv, ierr)
            call screen_set_context( &
               sc, temp, den, ltv(k), lrhov(k), zbar, abar, z2bar, &
               extended_screening, 2, y_co, iso_z158)
            call screen_pair( &
               sc, 12.0_dp, 6.0_dp, 4.0_dp, 2.0_dp, extended_screening, &
               zs13, zhat, zhat2, lzav, aznut, zs13inv, 0.0_dp, &
               scor, dscor_dt, dscor_dd, ierr)
            tmp(1, k) = safe_log10(lambda * scor)
            tmp(2, k) = lambda; tmp(3, k) = scor
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            temp = tv(k); den = rhov(k); t9 = temp * 1.0e-9_dp
            call reaclib_reaction_rates( &
               lo, hi, t9, reaclib_rates, chem_isos, .true., &
               lambda, dlambda_dlnt, rlambda, drlambda_dlnt, ierr)
            call screen_init_AZ_info( &
               12.0_dp, 6.0_dp, 4.0_dp, 2.0_dp, &
               zs13, zhat, zhat2, lzav, aznut, zs13inv, ierr)
            call screen_set_context( &
               sc, temp, den, ltv(k), lrhov(k), zbar, abar, z2bar, &
               extended_screening, 2, y_co, iso_z158)
            call screen_pair( &
               sc, 12.0_dp, 6.0_dp, 4.0_dp, 2.0_dp, extended_screening, &
               zs13, zhat, zhat2, lzav, aznut, zs13inv, 0.0_dp, &
               scor, dscor_dt, dscor_dd, ierr)
            out1(1, k) = safe_log10(lambda * scor)
            out1(2, k) = lambda; out1(3, k) = scor
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            temp = tv(k); den = rhov(k); t9 = temp * 1.0e-9_dp
            call reaclib_reaction_rates( &
               lo, hi, t9, reaclib_rates, chem_isos, .true., &
               lambda, dlambda_dlnt, rlambda, drlambda_dlnt, ierr)
            call screen_set_context( &
               sc, temp, den, ltv(k), lrhov(k), zbar, abar, z2bar, &
               extended_screening, 2, y_co, iso_z158)
            call screen_pair( &
               sc, 12.0_dp, 6.0_dp, 4.0_dp, 2.0_dp, extended_screening, &
               zs13, zhat, zhat2, lzav, aznut, zs13inv, 0.0_dp, &
               scor, dscor_dt, dscor_dd, ierr)
            out2(1, k) = safe_log10(lambda * scor)
            out2(2, k) = lambda; out2(3, k) = scor
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) &
         !$omp private(k, sc, temp, den, t9, lambda, dlambda_dlnt, rlambda, &
         !$omp    drlambda_dlnt, scor, dscor_dt, dscor_dd, ierr) schedule(static)
         do k = 1, nz
            temp = tv(k); den = rhov(k); t9 = temp * 1.0e-9_dp
            call reaclib_reaction_rates( &
               lo, hi, t9, reaclib_rates, chem_isos, .true., &
               lambda, dlambda_dlnt, rlambda, drlambda_dlnt, ierr)
            call screen_set_context( &
               sc, temp, den, ltv(k), lrhov(k), zbar, abar, z2bar, &
               extended_screening, 2, y_co, iso_z158)
            call screen_pair( &
               sc, 12.0_dp, 6.0_dp, 4.0_dp, 2.0_dp, extended_screening, &
               zs13, zhat, zhat2, lzav, aznut, zs13inv, 0.0_dp, &
               scor, dscor_dt, dscor_dd, ierr)
            out3(1, k) = safe_log10(lambda * scor)
            out3(2, k) = lambda; out3(3, k) = scor
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'rates_screen'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(3 * nz, out0, out1), &
         max_rel_diff(3 * nz, out0, out2), &
         max_rel_diff(3 * nz, out0, out3))
      rec%checksum = sum(out2(1, :))
      deallocate(lrhov, rhov, ltv, tv, samples, out0, out1, out2, out3)
   end subroutine bench_rates_screen

   !> Benchmarks basic.net 8-isotope reaction network RHS & Jacobian (net_eval).
   subroutine bench_net_eval(net_handle, nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: net_handle, nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec

      type(Net_General_Info), pointer :: g
      type(Net_Info) :: n_work
      integer :: k, rep, reps, ierr, n_sp, n_rxn
      real(dp) :: denom, s, t0, xh, xhe, z, abar, zbar, z2bar, z53bar, ye
      real(dp) :: mass_correction, sumx, eps_nuc, deps_dr, deps_dt, eps_neu
      real(dp) :: xa(8), d_eps_dx(8), dxdt(8), d_dxdt_dr(8), d_dxdt_dt(8)
      real(dp) :: d_dxdt_dx(8, 8), eps_cat(num_categories)
      real(dp), pointer :: rate_factors(:)
      real(dp), allocatable :: lrhov(:), rhov(:), ltv(:), tv(:), samples(:)
      real(dp), allocatable :: tmp(:,:), out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      reps = max(n_repeats, 3)
      call get_net_ptr(net_handle, g, ierr)
      n_sp = g%num_isos
      n_rxn = g%num_reactions
      allocate(rate_factors(n_rxn))
      rate_factors = 1.0_dp
      xa(1:7) = [0.70_dp, 2.98e-5_dp, 0.28_dp, 0.0034_dp, 0.001_dp, 0.0096_dp, 0.002_dp]
      xa(8) = 1.0_dp - sum(xa(1:7))
      call basic_composition_info( &
         n_sp, g%chem_id, xa, xh, xhe, z, abar, zbar, z2bar, z53bar, &
         ye, mass_correction, sumx)

      allocate(lrhov(nz), rhov(nz), ltv(nz), tv(nz), samples(reps))
      allocate(out0(4, nz), out1(4, nz), out2(4, nz), out3(4, nz))
      denom = real(max(nz - 1, 1), dp)
      do k = 1, nz
         s = real(k - 1, dp) / denom
         lrhov(k) = 2.0_dp + 0.1_dp * s; rhov(k) = exp10(lrhov(k))
         ltv(k) = 7.9_dp + 0.05_dp * s; tv(k) = exp10(ltv(k))
      end do

      call net_get( &
         net_handle, .false., n_work, n_sp, n_rxn, xa, tv(1), ltv(1), &
         rhov(1), lrhov(1), abar, zbar, z2bar, ye, 0.0_dp, 0.0_dp, 0.0_dp, &
         rate_factors, 1.0_dp, std_reaction_Qs, std_reaction_neuQs, &
         eps_nuc, deps_dr, deps_dt, d_eps_dx, dxdt, d_dxdt_dr, d_dxdt_dt, &
         d_dxdt_dx, extended_screening, eps_cat, eps_neu, ierr)

      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(4, nz))
         do k = 1, nz
            call net_get( &
               net_handle, .false., n_work, n_sp, n_rxn, xa, tv(k), ltv(k), &
               rhov(k), lrhov(k), abar, zbar, z2bar, ye, 0.0_dp, 0.0_dp, 0.0_dp, &
               rate_factors, 1.0_dp, std_reaction_Qs, std_reaction_neuQs, &
               eps_nuc, deps_dr, deps_dt, d_eps_dx, dxdt, d_dxdt_dr, d_dxdt_dt, &
               d_dxdt_dx, extended_screening, eps_cat, eps_neu, ierr)
            tmp(1, k) = safe_log10(eps_nuc); tmp(2, k) = eps_nuc
            tmp(3, k) = deps_dt; tmp(4, k) = deps_dr
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call net_get( &
               net_handle, .false., n_work, n_sp, n_rxn, xa, tv(k), ltv(k), &
               rhov(k), lrhov(k), abar, zbar, z2bar, ye, 0.0_dp, 0.0_dp, 0.0_dp, &
               rate_factors, 1.0_dp, std_reaction_Qs, std_reaction_neuQs, &
               eps_nuc, deps_dr, deps_dt, d_eps_dx, dxdt, d_dxdt_dr, d_dxdt_dt, &
               d_dxdt_dx, extended_screening, eps_cat, eps_neu, ierr)
            out1(1, k) = safe_log10(eps_nuc); out1(2, k) = eps_nuc
            out1(3, k) = deps_dt; out1(4, k) = deps_dr
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call net_get( &
               net_handle, .false., n_work, n_sp, n_rxn, xa, tv(k), ltv(k), &
               rhov(k), lrhov(k), abar, zbar, z2bar, ye, 0.0_dp, 0.0_dp, 0.0_dp, &
               rate_factors, 1.0_dp, std_reaction_Qs, std_reaction_neuQs, &
               eps_nuc, deps_dr, deps_dt, d_eps_dx, dxdt, d_dxdt_dr, d_dxdt_dt, &
               d_dxdt_dx, extended_screening, eps_cat, eps_neu, ierr)
            out2(1, k) = safe_log10(eps_nuc); out2(2, k) = eps_nuc
            out2(3, k) = deps_dt; out2(4, k) = deps_dr
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) &
         !$omp private(k, n_work, eps_nuc, deps_dr, deps_dt, d_eps_dx, dxdt, &
         !$omp    d_dxdt_dr, d_dxdt_dt, d_dxdt_dx, eps_cat, eps_neu, ierr) schedule(static)
         do k = 1, nz
            call net_get( &
               net_handle, .false., n_work, n_sp, n_rxn, xa, tv(k), ltv(k), &
               rhov(k), lrhov(k), abar, zbar, z2bar, ye, 0.0_dp, 0.0_dp, 0.0_dp, &
               rate_factors, 1.0_dp, std_reaction_Qs, std_reaction_neuQs, &
               eps_nuc, deps_dr, deps_dt, d_eps_dx, dxdt, d_dxdt_dr, d_dxdt_dt, &
               d_dxdt_dx, extended_screening, eps_cat, eps_neu, ierr)
            out3(1, k) = safe_log10(eps_nuc); out3(2, k) = eps_nuc
            out3(3, k) = deps_dt; out3(4, k) = deps_dr
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'net_eval'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(4 * nz, out0, out1), &
         max_rel_diff(4 * nz, out0, out2), &
         max_rel_diff(4 * nz, out0, out3))
      rec%checksum = sum(out2(1, :))
      deallocate(rate_factors, lrhov, rhov, ltv, tv, samples, out0, out1, out2, out3)
   end subroutine bench_net_eval

   !> Benchmarks ATM_TABLE_PHOTOSPHERE boundary conditions (atm_table).
   subroutine bench_atm_table(nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec

      integer :: k, rep, reps, ierr
      real(dp) :: denom, s, t0, lnt_dl, lnt_dr, lnt_dm, lnt_dk
      real(dp) :: lnp_dl, lnp_dr, lnp_dm, lnp_dk
      real(dp), allocatable :: lv(:), rv(:), mv(:), teffv(:), samples(:)
      real(dp), allocatable :: tmp(:,:), out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      reps = max(n_repeats, 3)
      allocate(lv(nz), rv(nz), mv(nz), teffv(nz), samples(reps))
      allocate(out0(8, nz), out1(8, nz), out2(8, nz), out3(8, nz))
      denom = real(max(nz - 1, 1), dp)
      do k = 1, nz
         s = real(k - 1, dp) / denom
         mv(k) = Msun
         rv(k) = Rsun * (0.8_dp + 0.6_dp * s)
         teffv(k) = 5200.0_dp + 1200.0_dp * s
         lv(k) = pi * crad * clight * rv(k) * rv(k) * (teffv(k)**4)
      end do

      call atm_eval_table( &
         lv(1), rv(1), mv(1), standard_cgrav, ATM_TABLE_PHOTOSPHERE, 0.0179_dp, &
         .false., teffv(1), out0(1, 1), lnt_dl, lnt_dr, lnt_dm, lnt_dk, &
         out0(5, 1), lnp_dl, lnp_dr, lnp_dm, lnp_dk, ierr)

      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(8, nz))
         do k = 1, nz
            call atm_eval_table( &
               lv(k), rv(k), mv(k), standard_cgrav, ATM_TABLE_PHOTOSPHERE, 0.0179_dp, &
               .false., teffv(k), tmp(1, k), tmp(2, k), tmp(3, k), tmp(4, k), lnt_dk, &
               tmp(5, k), tmp(6, k), tmp(7, k), tmp(8, k), lnp_dk, ierr)
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call atm_eval_table( &
               lv(k), rv(k), mv(k), standard_cgrav, ATM_TABLE_PHOTOSPHERE, 0.0179_dp, &
               .false., teffv(k), out1(1, k), out1(2, k), out1(3, k), out1(4, k), lnt_dk, &
               out1(5, k), out1(6, k), out1(7, k), out1(8, k), lnp_dk, ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call atm_eval_table( &
               lv(k), rv(k), mv(k), standard_cgrav, ATM_TABLE_PHOTOSPHERE, 0.0179_dp, &
               .false., teffv(k), out2(1, k), out2(2, k), out2(3, k), out2(4, k), lnt_dk, &
               out2(5, k), out2(6, k), out2(7, k), out2(8, k), lnp_dk, ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k, lnt_dk, lnp_dk, ierr) schedule(static)
         do k = 1, nz
            call atm_eval_table( &
               lv(k), rv(k), mv(k), standard_cgrav, ATM_TABLE_PHOTOSPHERE, 0.0179_dp, &
               .false., teffv(k), out3(1, k), out3(2, k), out3(3, k), out3(4, k), lnt_dk, &
               out3(5, k), out3(6, k), out3(7, k), out3(8, k), lnp_dk, ierr)
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'atm_table'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(8 * nz, out0, out1), &
         max_rel_diff(8 * nz, out0, out2), &
         max_rel_diff(8 * nz, out0, out3))
      rec%checksum = sum(out2(5, :))
      deallocate(lv, rv, mv, teffv, samples, out0, out1, out2, out3)
   end subroutine bench_atm_table

end module bench_kernels_wave3b
