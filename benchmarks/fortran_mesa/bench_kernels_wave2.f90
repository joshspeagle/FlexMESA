! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Module: bench_kernels_wave2
!
! Four-stage Fortran MESA Wave 2 microphysics and synthetic photometry
! benchmarks (bench_chem_comp, bench_ion_fe56, bench_ion_4d, bench_colors_sed).
! -----------------------------------------------------------------------------
module bench_kernels_wave2
   use const_def, only: dp, rsun, arg_not_provided
   use math_lib, only: exp10
   use chem_def, only: chem_isos, iso_name_length
   use chem_lib, only: chem_get_iso_id, basic_composition_info
   use ionization_def, only: num_ion_vals, fe_he_ptr
   use ionization_lib, only: eval_ionization, eval_charge_of_Fe56_in_He4
   use interp_2d_lib_db, only: interp_evbicub_db
   use colors_def, only: Colors_General_Info, num_color_filters, &
      color_filter_names
   use colors_lib, only: colors_ptr, data_for_colors_history_columns, &
      calculate_bolometric, calculate_synthetic
   use colors_utils, only: resolve_path
   use omp_lib, only: omp_set_num_threads
   use bench_harness, only: STAGE3_THREADS, BenchRecord, now_us, &
      compute_latency_stats, max_rel_diff, max_err3
   implicit none
   private

   public :: bench_chem_comp, bench_ion_fe56, bench_ion_4d, bench_colors_sed

   character(len=iso_name_length), parameter :: WAVE2_ISO_NAMES(8) = [ &
      'h1      ', 'he3     ', 'he4     ', 'c12     ', &
      'n14     ', 'o16     ', 'ne20    ', 'mg24    ']

contains

   !> Evaluates the 10 composition moment columns for xa(1:8) using hoisted
   !! species properties (a_sp, z_sp, w_sp, z53_sp).
   pure subroutine eval_comp_hoisted( &
         xa, iz_sp, a_sp, z_sp, w_sp, z53_sp, out10)
      real(dp), intent(in) :: xa(8), a_sp(8), z_sp(8), w_sp(8), z53_sp(8)
      integer, intent(in) :: iz_sp(8)
      real(dp), intent(out) :: out10(10)
      real(dp) :: y(8), xh, xhe, xz, sumx, sumy, abar, zbar, z2bar, z53bar
      integer :: i

      xh = 0.0_dp; xhe = 0.0_dp
      do i = 1, 8
         y(i) = xa(i) / a_sp(i)
         if (iz_sp(i) == 1) xh = xh + xa(i)
         if (iz_sp(i) == 2) xhe = xhe + xa(i)
      end do
      xz = max(0.0_dp, min(1.0_dp, 1.0_dp - (xh + xhe)))
      sumx = sum(xa); sumy = sum(y); abar = sumx / sumy
      zbar = sum(y * z_sp) * abar; z2bar = sum(y * z_sp * z_sp) * abar
      z53bar = sum(y * z53_sp) * abar
      out10 = [xh, xhe, xz, abar, zbar, z2bar, z53bar, zbar / abar, &
         sum(y * w_sp), sumx]
   end subroutine eval_comp_hoisted

   !> Benchmarks 8-isotope composition moments (chem_comp) across Stages 0..3.
   subroutine bench_chem_comp(nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: chem_ids(8), iz_sp(8), i, k, rep, reps, cid
      real(dp) :: a_sp(8), z_sp(8), w_sp(8), z53_sp(8), denom, s, z, x, y, t0
      real(dp), allocatable :: xa(:,:), samples(:), tmp(:,:)
      real(dp), allocatable :: out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      reps = max(n_repeats, 3)
      do i = 1, 8
         cid = chem_get_iso_id(trim(WAVE2_ISO_NAMES(i)))
         chem_ids(i) = cid; iz_sp(i) = chem_isos%Z(cid)
         z_sp(i) = real(iz_sp(i), dp)
         a_sp(i) = real(chem_isos%Z_plus_N(cid), dp)
         w_sp(i) = chem_isos%W(cid); z53_sp(i) = chem_isos%Z53(cid)
      end do
      allocate(xa(8, nz), samples(reps))
      allocate(out0(10, nz), out1(10, nz), out2(10, nz), out3(10, nz))
      denom = real(max(nz - 1, 1), dp)
      do k = 1, nz
         s = real(k - 1, dp) / denom; z = 0.014_dp + 0.016_dp * s
         x = 0.70_dp * (1.0_dp - s) + 0.05_dp * s; y = 1.0_dp - x - z
         xa(:, k) = [x, 1.0e-4_dp * y, 0.9999_dp * y, 0.18_dp * z, &
            0.06_dp * z, 0.48_dp * z, 0.14_dp * z, 0.14_dp * z]
      end do

      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(10, nz))
         do k = 1, nz
            call basic_composition_info(8, chem_ids, xa(:, k), tmp(1, k), &
               tmp(2, k), tmp(3, k), tmp(4, k), tmp(5, k), tmp(6, k), &
               tmp(7, k), tmp(8, k), tmp(9, k), tmp(10, k))
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call basic_composition_info(8, chem_ids, xa(:, k), out1(1, k), &
               out1(2, k), out1(3, k), out1(4, k), out1(5, k), out1(6, k), &
               out1(7, k), out1(8, k), out1(9, k), out1(10, k))
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call eval_comp_hoisted( &
               xa(:, k), iz_sp, a_sp, z_sp, w_sp, z53_sp, out2(:, k))
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k) schedule(static)
         do k = 1, nz
            call eval_comp_hoisted( &
               xa(:, k), iz_sp, a_sp, z_sp, w_sp, z53_sp, out3(:, k))
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'chem_comp'; rec%n_zones = nz
      rec%max_rel_err = max_err3(max_rel_diff(10 * nz, out0, out1), &
         max_rel_diff(10 * nz, out0, out2), max_rel_diff(10 * nz, out0, out3))
      rec%checksum = sum(out2(4, :))
      deallocate(xa, samples, out0, out1, out2, out3)
   end subroutine bench_chem_comp

   !> Benchmarks Potekhin 56Fe-in-4He effective charge (ion_fe56).
   subroutine bench_ion_fe56(nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer, parameter :: ict(6) = [1, 1, 1, 0, 0, 0]
      integer :: k, rep, reps, ierr
      real(dp) :: denom, s, t0, z_warm, fval(6)
      real(dp), allocatable :: lne(:), lt(:), samples(:), tmp(:)
      real(dp), allocatable :: out0(:), out1(:), out2(:), out3(:)

      reps = max(n_repeats, 3)
      allocate(lne(nz), lt(nz), samples(reps))
      allocate(out0(nz), out1(nz), out2(nz), out3(nz))
      denom = real(max(nz - 1, 1), dp)
      do k = 1, nz
         s = real(k - 1, dp) / denom
         lne(k) = 24.5_dp + 5.5_dp * s; lt(k) = 6.2_dp + 1.6_dp * s
      end do
      z_warm = eval_charge_of_Fe56_in_He4(lne(1), lt(1), ierr)

      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(nz))
         do k = 1, nz
            tmp(k) = eval_charge_of_Fe56_in_He4(lne(k), lt(k), ierr)
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            out1(k) = eval_charge_of_Fe56_in_He4(lne(k), lt(k), ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call interp_evbicub_db(lne(k), lt(k), fe_he_ptr%log_ne, &
               fe_he_ptr%num_log_ne, fe_he_ptr%logT, fe_he_ptr%num_logT, &
               fe_he_ptr%ilinx, fe_he_ptr%iliny, fe_he_ptr%f1, &
               fe_he_ptr%num_log_ne, ict, fval, ierr)
            out2(k) = fval(1)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k, fval, ierr) &
         !$omp schedule(static)
         do k = 1, nz
            call interp_evbicub_db(lne(k), lt(k), fe_he_ptr%log_ne, &
               fe_he_ptr%num_log_ne, fe_he_ptr%logT, fe_he_ptr%num_logT, &
               fe_he_ptr%ilinx, fe_he_ptr%iliny, fe_he_ptr%f1, &
               fe_he_ptr%num_log_ne, ict, fval, ierr)
            out3(k) = fval(1)
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'ion_fe56'; rec%n_zones = nz
      rec%max_rel_err = max_err3(max_rel_diff(nz, out0, out1), &
         max_rel_diff(nz, out0, out2), max_rel_diff(nz, out0, out3))
      rec%checksum = sum(out2)
      deallocate(lne, lt, samples, out0, out1, out2, out3)
   end subroutine bench_ion_fe56

   !> Benchmarks 4D ionization table evaluation (ion_4d, 28 channels).
   subroutine bench_ion_4d(nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer :: k, rep, reps, ierr
      real(dp) :: denom, s, lq, t0, warm_res(num_ion_vals)
      real(dp), allocatable :: zv(:), xv(:), lrhov(:), ltv(:), rhov(:), tv(:)
      real(dp), allocatable :: samples(:), tmp(:,:)
      real(dp), allocatable :: out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      reps = max(n_repeats, 3)
      allocate(zv(nz), xv(nz), lrhov(nz), ltv(nz), rhov(nz), tv(nz))
      allocate(samples(reps))
      allocate(out0(num_ion_vals, nz), out1(num_ion_vals, nz))
      allocate(out2(num_ion_vals, nz), out3(num_ion_vals, nz))
      denom = real(max(nz - 1, 1), dp)
      do k = 1, nz
         s = real(k - 1, dp) / denom
         zv(k) = 0.005_dp + 0.030_dp * s; xv(k) = 0.62_dp + 0.16_dp * s
         ltv(k) = 4.505_dp + 0.050_dp * s; lq = -2.075_dp + 0.080_dp * s
         lrhov(k) = lq + 2.0_dp * ltv(k) - 12.0_dp
         rhov(k) = exp10(lrhov(k)); tv(k) = exp10(ltv(k))
      end do
      call eval_ionization( &
         zv(1), xv(1), rhov(1), lrhov(1), tv(1), ltv(1), warm_res, ierr)

      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(num_ion_vals, nz))
         do k = 1, nz
            call eval_ionization(zv(k), xv(k), arg_not_provided, lrhov(k), &
               arg_not_provided, ltv(k), tmp(:, k), ierr)
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call eval_ionization(zv(k), xv(k), arg_not_provided, lrhov(k), &
               arg_not_provided, ltv(k), out1(:, k), ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call eval_ionization( &
               zv(k), xv(k), rhov(k), lrhov(k), tv(k), ltv(k), out2(:, k), ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k, ierr) schedule(static)
         do k = 1, nz
            call eval_ionization( &
               zv(k), xv(k), rhov(k), lrhov(k), tv(k), ltv(k), out3(:, k), ierr)
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'ion_4d'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(num_ion_vals * nz, out0, out1), &
         max_rel_diff(num_ion_vals * nz, out0, out2), &
         max_rel_diff(num_ion_vals * nz, out0, out3))
      rec%checksum = sum(out2(1, :))
      deallocate(zv, xv, lrhov, ltv, rhov, tv, samples, out0, out1, out2, out3)
   end subroutine bench_ion_4d

   !> Benchmarks 3D Kurucz2003all SED synthesis + Johnson photometry.
   subroutine bench_colors_sed(colors_handle, nz, n_warmup, n_repeats, rec)
      integer, intent(in) :: colors_handle, nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec
      integer, parameter :: n_cols = 10
      type(Colors_General_Info), pointer :: cs
      character(len=80) :: col_names(n_cols)
      character(len=256) :: sed_dir
      integer :: k, f_idx, rep, reps, ierr
      real(dp) :: denom, s, t0, d
      real(dp), allocatable :: teff(:), logg(:), meta(:), rad(:)
      real(dp), allocatable :: wav(:), flx(:), samples(:), tmp(:,:)
      real(dp), allocatable :: out0(:,:), out1(:,:), out2(:,:), out3(:,:)

      reps = max(n_repeats, 3)
      call colors_ptr(colors_handle, cs, ierr)
      d = cs%distance; sed_dir = trim(resolve_path(cs%stellar_atm))
      allocate(teff(nz), logg(nz), meta(nz), rad(nz), samples(reps))
      allocate(out0(n_cols, nz), out1(n_cols, nz))
      allocate(out2(n_cols, nz), out3(n_cols, nz))
      denom = real(max(nz - 1, 1), dp)
      do k = 1, nz
         s = real(k - 1, dp) / denom
         teff(k) = 5760.0_dp + 230.0_dp * s; logg(k) = 4.05_dp + 0.40_dp * s
         meta(k) = -0.45_dp + 0.60_dp * s; rad(k) = rsun * (0.9_dp + 0.4_dp * s)
      end do

      call omp_set_num_threads(1)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         allocate(tmp(n_cols, nz))
         do k = 1, nz
            call data_for_colors_history_columns(teff(k), logg(k), rad(k), &
               meta(k), k, colors_handle, n_cols, col_names, tmp(:, k), ierr)
         end do
         out0 = tmp; deallocate(tmp)
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage0)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call data_for_colors_history_columns(teff(k), logg(k), rad(k), &
               meta(k), k, colors_handle, n_cols, col_names, out1(:, k), ierr)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage1)

      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         do k = 1, nz
            call calculate_bolometric(cs, teff(k), logg(k), meta(k), rad(k), &
               d, out2(1, k), out2(2, k), wav, flx, sed_dir, out2(3, k))
            do f_idx = 1, num_color_filters
               out2(3 + f_idx, k) = calculate_synthetic(teff(k), logg(k), &
                  meta(k), ierr, wav, flx, cs%filters(f_idx)%wavelengths, &
                  cs%filters(f_idx)%transmission, &
                  cs%filters(f_idx)%vega_zero_point, &
                  color_filter_names(f_idx), .false., .false., '', k)
            end do
            deallocate(wav, flx)
         end do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage2)

      call omp_set_num_threads(STAGE3_THREADS)
      do rep = 1, n_warmup + reps
         if (rep > n_warmup) t0 = now_us()
         !$omp parallel do default(shared) private(k, f_idx, wav, flx, ierr) &
         !$omp schedule(static)
         do k = 1, nz
            call calculate_bolometric(cs, teff(k), logg(k), meta(k), rad(k), &
               d, out3(1, k), out3(2, k), wav, flx, sed_dir, out3(3, k))
            do f_idx = 1, num_color_filters
               out3(3 + f_idx, k) = calculate_synthetic(teff(k), logg(k), &
                  meta(k), ierr, wav, flx, cs%filters(f_idx)%wavelengths, &
                  cs%filters(f_idx)%transmission, &
                  cs%filters(f_idx)%vega_zero_point, &
                  color_filter_names(f_idx), .false., .false., '', k)
            end do
            deallocate(wav, flx)
         end do
         !$omp end parallel do
         if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
      end do
      call compute_latency_stats(samples, reps, rec%stage3)

      rec%workload = 'colors_sed'; rec%n_zones = nz
      rec%max_rel_err = max_err3(max_rel_diff(n_cols * nz, out0, out1), &
         max_rel_diff(n_cols * nz, out0, out2), &
         max_rel_diff(n_cols * nz, out0, out3))
      rec%checksum = sum(out2(10, :))
      deallocate(teff, logg, meta, rad, samples, out0, out1, out2, out3)
   end subroutine bench_colors_sed

end module bench_kernels_wave2
