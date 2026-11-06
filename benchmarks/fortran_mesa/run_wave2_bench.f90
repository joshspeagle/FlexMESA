! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Program: run_wave2_bench
!
! Executes the Wave 2 four-stage Fortran MESA benchmark and cross-language
! parity suite across all four Wave 2 atomic and photometric microphysics
! workloads:
!   1. chem_comp  (8-isotope basic_composition_info & hoisted moments)
!   2. ion_fe56   (Potekhin 56Fe-in-4He eval_charge_of_Fe56_in_He4 & bicub)
!   3. ion_4d     (28-channel 4D eval_ionization over Z, X, log10_rho, log10_T)
!   4. colors_sed (Kurucz2003all 3D Hermite SED + 7 Johnson Vega bands)
!
! First verifies internal stage parity (Stages 1..3 vs. Stage 0 <= 1e-12)
! across N_zones in {8, 32} and checks N_zones = 8 checksums against the
! pinned cross-language reference values from flexmesa-bench (Rust) and
! flexmesa.bench (Python/NumPy/JAX). Then runs the full timing suite across
! N_zones in {128, 1024, 4096} and prints the Markdown summary table.
! -----------------------------------------------------------------------------
program run_wave2_bench
   use const_def, only: dp
   use const_lib, only: const_init
   use math_lib, only: math_init
   use chem_lib, only: chem_init, chem_shutdown
   use ionization_lib, only: ionization_init
   use colors_def, only: Colors_General_Info
   use colors_lib, only: colors_init, colors_shutdown, &
      alloc_colors_handle_using_inlist, free_colors_handle, colors_ptr, &
      colors_setup_tables, colors_setup_hooks
   use bench_harness, only: BenchRecord
   use bench_kernels_wave2, only: bench_chem_comp, bench_ion_fe56, &
      bench_ion_4d, bench_colors_sed
   implicit none

   ! Pinned N_zones = 8 cross-language checksums matching flexmesa-bench (Rust)
   ! and flexmesa.bench (Python/NumPy/JAX).
   real(dp), parameter :: EXPECTED_WAVE2_CHECKSUMS_N8(4) = [ &
      1.687001493521061e1_dp, &
      1.683709392958418e2_dp, &
      6.118846146172181e1_dp, &
      3.608691732539108e1_dp]

   integer, parameter :: PARITY_ZONES(2) = [8, 32]
   integer, parameter :: BENCH_ZONES(3) = [128, 1024, 4096]
   integer, parameter :: N_WARMUP = 3
   integer, parameter :: N_REPEATS = 9

   character(len=512) :: mesa_dir_arg
   integer :: ierr, idx, w, colors_handle
   real(dp) :: rel_chk, tol_chk
   type(Colors_General_Info), pointer :: cs
   type(BenchRecord) :: recs(4)

   call get_environment_variable('MESA_DIR', mesa_dir_arg)
   if (len_trim(mesa_dir_arg) == 0) mesa_dir_arg = '../../..'
   call const_init(mesa_dir_arg, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()

   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'

   call ionization_init('ion', '_CO_1', '', .false., ierr)
   if (ierr /= 0) stop 'ionization_init failed'

   call colors_init(.false., '', ierr)
   if (ierr /= 0) stop 'colors_init failed'
   colors_handle = alloc_colors_handle_using_inlist('', ierr)
   if (ierr /= 0) stop 'alloc_colors_handle_using_inlist failed'
   call colors_ptr(colors_handle, cs, ierr)
   if (ierr /= 0) stop 'colors_ptr failed'
   cs%use_colors = .true.
   cs%mag_system = 'Vega'
   call colors_setup_tables(colors_handle, ierr)
   if (ierr /= 0) stop 'colors_setup_tables failed'
   call colors_setup_hooks(colors_handle, ierr)
   if (ierr /= 0) stop 'colors_setup_hooks failed'

   ! Step 1: Verify internal stage parity and cross-language N_zones = 8
   ! reference checksums before running the timing suite.
   do idx = 1, size(PARITY_ZONES)
      call run_all_wave2_workloads( &
         colors_handle, PARITY_ZONES(idx), 1, 3, recs)
      do w = 1, 4
         if ((recs(w)%max_rel_err /= recs(w)%max_rel_err) .or. &
             (recs(w)%max_rel_err > 1.0e-12_dp)) then
            write (*, '(a,a,a,es12.4)') 'Parity failure in ', &
               trim(recs(w)%workload), ': max_rel_err = ', recs(w)%max_rel_err
            stop 1
         end if
         if (PARITY_ZONES(idx) == 8) then
            rel_chk = abs(recs(w)%checksum - EXPECTED_WAVE2_CHECKSUMS_N8(w)) &
               / abs(EXPECTED_WAVE2_CHECKSUMS_N8(w))
            ! Upstream MESA fvbicub_db (used by ion_fe56, w == 2) has implicit
            ! 32-bit REAL locals (xp, yp, cx, cy, hx2, hy2, z36th), giving
            ! ~4.8e-9 relative difference vs true IEEE-754 float64.
            tol_chk = merge(1.0e-8_dp, 1.0e-11_dp, w == 2)
            if ((rel_chk /= rel_chk) .or. (rel_chk > tol_chk)) then
               write (*, '(a,a,a,es12.4,a,es22.15,a,es22.15)') &
                  'Cross-language checksum mismatch ', &
                  trim(recs(w)%workload), ': rel_err = ', rel_chk, &
                  ' got ', recs(w)%checksum, &
                  ' expected ', EXPECTED_WAVE2_CHECKSUMS_N8(w)
               stop 1
            end if
         end if
      end do
   end do

   ! Step 2: Execute the four-stage benchmark suite across N_zones in
   ! {128, 1024, 4096} and print the Markdown table.
   write (*, '(a)') '| Workload | N_zones | Stage 0 (us) | Stage 1 (us) | ' // &
      'Stage 2 (us) | Stage 3 (us) | Max Rel Err | Checksum |'
   write (*, '(a)') '| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'

   do idx = 1, size(BENCH_ZONES)
      call run_all_wave2_workloads( &
         colors_handle, BENCH_ZONES(idx), N_WARMUP, N_REPEATS, recs)
      do w = 1, 4
         call print_record_row(recs(w))
      end do
   end do

   call free_colors_handle(colors_handle)
   call colors_shutdown()
   call chem_shutdown()

contains

   !> Runs all four Wave 2 workloads for a single n_zones value.
   subroutine run_all_wave2_workloads(ch, nz, n_warm, n_rep, out_recs)
      integer, intent(in) :: ch, nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: out_recs(4)

      call bench_chem_comp(nz, n_warm, n_rep, out_recs(1))
      call bench_ion_fe56(nz, n_warm, n_rep, out_recs(2))
      call bench_ion_4d(nz, n_warm, n_rep, out_recs(3))
      call bench_colors_sed(ch, nz, n_warm, n_rep, out_recs(4))
   end subroutine run_all_wave2_workloads

   !> Formats and prints a single Markdown table row for a BenchRecord.
   subroutine print_record_row(rec)
      type(BenchRecord), intent(in) :: rec
      character(len=*), parameter :: FMT_ROW = &
         '(a,a12,a,i5,a,f9.2,a,f8.2,a,f9.2,a,f8.2,a,f9.2,a,f8.2,a,' // &
         'f9.2,a,f8.2,a,es9.2,a,es16.8,a)'
      write (*, FMT_ROW) &
         '| `', rec%workload, '` | ', rec%n_zones, ' | ', &
         rec%stage0%median_us, ' +/- ', rec%stage0%iqr_us, ' | ', &
         rec%stage1%median_us, ' +/- ', rec%stage1%iqr_us, ' | ', &
         rec%stage2%median_us, ' +/- ', rec%stage2%iqr_us, ' | ', &
         rec%stage3%median_us, ' +/- ', rec%stage3%iqr_us, ' | ', &
         rec%max_rel_err, ' | ', rec%checksum, ' |'
   end subroutine print_record_row

end program run_wave2_bench
