! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Program: run_wave5_bench
!
! Executes the Wave 5 four-stage Fortran MESA benchmark suite across all four
! Wave 5 advanced stellar, interacting binary, and asteroseismology workloads:
!   1. w5_diff_rot     (Wave 5.1: Roche distortion, Caplan OCP & Spruit dynamo)
!   2. w5_riemann_rsp2 (Wave 5.2: Lagrangian HLLC, RSP2 & Dziembowski variables)
!   3. w5_binary_rlof  (Wave 5.3: Eggleton Roche lobe, RLOF, BH, Jdot & tides)
!   4. w5_astero_seism (Wave 5.4: Acoustic profile, scaling, BG14 & r01/r10)
! -----------------------------------------------------------------------------
program run_wave5_bench
   use const_def, only: dp, do_const_init
   use math_lib, only: math_init
   use bench_harness, only: BenchRecord
   use bench_kernels_wave5, only: bench_w5_diff_rot, bench_w5_riemann_rsp2, &
      bench_w5_binary_rlof, bench_w5_astero_seism
   implicit none

   integer, parameter :: PARITY_ZONES(1) = [8]
   integer, parameter :: BENCH_ZONES(3) = [128, 512, 1024]
   integer, parameter :: N_WARMUP = 3
   integer, parameter :: N_REPEATS = 11

   character(len=512) :: mesa_dir_arg
   integer :: ierr, idx, w
   type(BenchRecord) :: recs(4)

   call get_environment_variable('MESA_DIR', mesa_dir_arg)
   if (len_trim(mesa_dir_arg) == 0) mesa_dir_arg = '../../..'
   call do_const_init(mesa_dir_arg, ierr)
   if (ierr /= 0) stop 'do_const_init failed'
   call math_init()

   ! Step 1: Verify internal stage parity at N_zones = 8 before timing.
   do idx = 1, size(PARITY_ZONES)
      call run_all_wave5_workloads(PARITY_ZONES(idx), 1, 3, recs)
      do w = 1, 4
         if ((recs(w)%max_rel_err /= recs(w)%max_rel_err) .or. &
             (recs(w)%max_rel_err > 1.0e-12_dp)) then
            write (*, '(a,a,a,es12.4)') 'Parity failure in ', &
               trim(recs(w)%workload), ': max_rel_err = ', recs(w)%max_rel_err
            stop 1
         end if
      end do
   end do

   ! Step 2: Execute the four-stage benchmark suite across N_zones in
   ! {128, 512, 1024} and print the Markdown table.
   write (*, '(a)') '| Workload | N_zones | Stage 0 (us) | Stage 1 (us) | ' // &
      'Stage 2 (us) | Stage 3 (us) | Max Rel Err | Checksum |'
   write (*, '(a)') '| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'

   do idx = 1, size(BENCH_ZONES)
      call run_all_wave5_workloads(BENCH_ZONES(idx), N_WARMUP, N_REPEATS, recs)
      do w = 1, 4
         call print_record_row(recs(w))
      end do
   end do

contains

   subroutine run_all_wave5_workloads(nz, n_warm, n_rep, out_recs)
      integer, intent(in) :: nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: out_recs(4)

      call bench_w5_diff_rot(nz, n_warm, n_rep, out_recs(1))
      call bench_w5_riemann_rsp2(nz, n_warm, n_rep, out_recs(2))
      call bench_w5_binary_rlof(nz, n_warm, n_rep, out_recs(3))
      call bench_w5_astero_seism(nz, n_warm, n_rep, out_recs(4))
   end subroutine run_all_wave5_workloads

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

end program run_wave5_bench
