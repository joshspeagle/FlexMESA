! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Program: run_wave3a_bench
!
! Executes the Wave 3a four-stage Fortran MESA benchmark and cross-language
! parity suite across all four Wave 3a equation-of-state (eos) workloads:
!   1. eos_ideal (Neutral ideal-gas + Planckian radiation EOS)
!   2. eos_helm  (Timmes & Swesty 2000 biquintic/bicubic HELM EOS)
!   3. eos_pc    (Potekhin & Chabrier 2010/2013 Coulomb liquid/solid PC EOS)
!   4. eos_skye  (Jermyn et al. 2021 3rd-order free-energy minimization Skye EOS)
!
! First verifies internal stage parity (Stages 1..3 vs. Stage 0 <= 1e-12)
! across N_zones in {8, 32} and checks N_zones = 8 checksums against the
! pinned cross-language reference values from flexmesa-bench (Rust) and
! flexmesa.bench (Python/NumPy/JAX). Then runs the full timing suite across
! N_zones in {128, 1024, 4096} and prints the Markdown summary table.
! -----------------------------------------------------------------------------
program run_wave3a_bench
   use const_def, only: dp
   use const_lib, only: const_init
   use math_lib, only: math_init
   use chem_lib, only: chem_init, chem_shutdown
   use eos_lib, only: eos_init, eos_shutdown, alloc_eos_handle, free_eos_handle
   use bench_harness, only: BenchRecord
   use bench_kernels_wave3a, only: bench_eos_ideal, bench_eos_helm, &
      bench_eos_pc, bench_eos_skye
   implicit none

   ! Pinned N_zones = 8 cross-language checksums matching flexmesa-bench (Rust)
   ! and flexmesa.bench (Python/NumPy/JAX).
   real(dp), parameter :: EXPECTED_WAVE3A_CHECKSUMS_N8(4) = [ &
      1.438224257481198e2_dp, &
      3.561688957866044e2_dp, &
      4.117886894420108e2_dp, &
      3.561331836722772e2_dp]

   integer, parameter :: PARITY_ZONES(2) = [8, 32]
   integer, parameter :: BENCH_ZONES(3) = [128, 1024, 4096]
   integer, parameter :: N_WARMUP = 3
   integer, parameter :: N_REPEATS = 9

   character(len=512) :: mesa_dir_arg
   integer :: ierr, idx, w, eos_handle
   real(dp) :: rel_chk
   type(BenchRecord) :: recs(4)

   call get_environment_variable('MESA_DIR', mesa_dir_arg)
   if (len_trim(mesa_dir_arg) == 0) mesa_dir_arg = '../../..'
   call const_init(mesa_dir_arg, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()

   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'

   call eos_init(' ', .true., ierr)
   if (ierr /= 0) stop 'eos_init failed'
   eos_handle = alloc_eos_handle(ierr)
   if (ierr /= 0) stop 'alloc_eos_handle failed'

   ! Step 1: Verify internal stage parity and cross-language N_zones = 8
   ! reference checksums before running the timing suite.
   do idx = 1, size(PARITY_ZONES)
      call run_all_wave3a_workloads( &
         eos_handle, PARITY_ZONES(idx), 1, 3, recs)
      do w = 1, 4
         if ((recs(w)%max_rel_err /= recs(w)%max_rel_err) .or. &
             (recs(w)%max_rel_err > 1.0e-11_dp)) then
            write (*, '(a,a,a,es12.4)') 'Parity failure in ', &
               trim(recs(w)%workload), ': max_rel_err = ', recs(w)%max_rel_err
            stop 1
         end if
         if (PARITY_ZONES(idx) == 8) then
            rel_chk = abs(recs(w)%checksum - EXPECTED_WAVE3A_CHECKSUMS_N8(w)) &
               / abs(EXPECTED_WAVE3A_CHECKSUMS_N8(w))
            if ((rel_chk /= rel_chk) .or. (rel_chk > 1.0e-11_dp)) then
               write (*, '(a,a,a,es12.4,a,es22.15,a,es22.15)') &
                  'Cross-language checksum mismatch ', &
                  trim(recs(w)%workload), ': rel_err = ', rel_chk, &
                  ' got ', recs(w)%checksum, &
                  ' expected ', EXPECTED_WAVE3A_CHECKSUMS_N8(w)
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
      call run_all_wave3a_workloads( &
         eos_handle, BENCH_ZONES(idx), N_WARMUP, N_REPEATS, recs)
      do w = 1, 4
         call print_record_row(recs(w))
      end do
   end do

   call free_eos_handle(eos_handle)
   call eos_shutdown()
   call chem_shutdown()

contains

   subroutine run_all_wave3a_workloads( &
         eh, nz, n_warm, n_rep, out_recs)
      integer, intent(in) :: eh, nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: out_recs(4)

      call bench_eos_ideal(eh, nz, n_warm, n_rep, out_recs(1))
      call bench_eos_helm(eh, nz, n_warm, n_rep, out_recs(2))
      call bench_eos_pc(eh, nz, n_warm, n_rep, out_recs(3))
      call bench_eos_skye(eh, nz, n_warm, n_rep, out_recs(4))
   end subroutine run_all_wave3a_workloads

   subroutine print_record_row(rec)
      type(BenchRecord), intent(in) :: rec

      write (*, '(a,a,a,i5,a,f9.2,a,f7.2,a,f9.2,a,f7.2,a,' // &
                'f9.2,a,f7.2,a,f9.2,a,f7.2,a,es9.2,a,es15.8,a)') &
         '| `', trim(rec%workload), '` | ', rec%n_zones, &
         ' | ', rec%stage0%median_us, ' +/- ', rec%stage0%iqr_us, &
         ' | ', rec%stage1%median_us, ' +/- ', rec%stage1%iqr_us, &
         ' | ', rec%stage2%median_us, ' +/- ', rec%stage2%iqr_us, &
         ' | ', rec%stage3%median_us, ' +/- ', rec%stage3%iqr_us, &
         ' | ', rec%max_rel_err, ' | ', rec%checksum, ' |'
   end subroutine print_record_row

end program run_wave3a_bench
