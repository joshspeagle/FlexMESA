! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Program: run_wave4_bench
!
! Executes the Wave 4 four-stage Fortran MESA benchmark suite across all five
! Wave 4 1D stellar structure and evolution workloads:
!   1. star_micro_vars  (Zone EOS, kap, neu, net, and face MLT + Brunt)
!   2. star_hydro_eqns  (Coupled 1D Henyey residual & 4x4 block Jacobian)
!   3. star_newton_step (Single linearize-and-solve bcyclic Henyey iteration)
!   4. star_remesh      (Adaptive mesh spacing function g_k evaluation)
!   5. star_evolve_step (Full coupled structure + burn + mix evolution step)
! -----------------------------------------------------------------------------
program run_wave4_bench
   use const_def, only: dp
   use star_def, only: star_info
   use star_lib, only: starlib_shutdown, free_star
   use bench_harness, only: BenchRecord
   use bench_kernels_wave4, only: init_base_zams_star, &
      bench_star_micro_vars, bench_star_hydro_eqns, &
      bench_star_newton_step, bench_star_remesh, bench_star_evolve_step
   implicit none

   integer, parameter :: PARITY_ZONES(1) = [8]
   integer, parameter :: BENCH_ZONES(3) = [128, 512, 1024]
   integer, parameter :: N_WARMUP = 2
   integer, parameter :: N_REPEATS = 7

   character(len=512) :: mesa_dir_arg
   integer :: ierr, idx, w, id_base
   type(star_info), pointer :: s_base
   type(BenchRecord) :: recs(5)

   call get_environment_variable('MESA_DIR', mesa_dir_arg)
   if (len_trim(mesa_dir_arg) == 0) mesa_dir_arg = '../../..'

   call init_base_zams_star(mesa_dir_arg, id_base, s_base, ierr)
   if (ierr /= 0) stop 'init_base_zams_star failed'

   ! Step 1: Verify internal stage parity at N_zones = 8 before timing.
   do idx = 1, size(PARITY_ZONES)
      call run_all_wave4_workloads(s_base, PARITY_ZONES(idx), 1, 3, recs)
      do w = 1, 5
         if ((recs(w)%max_rel_err /= recs(w)%max_rel_err) .or. &
             (recs(w)%max_rel_err > 1.0e-10_dp)) then
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
      call run_all_wave4_workloads( &
         s_base, BENCH_ZONES(idx), N_WARMUP, N_REPEATS, recs)
      do w = 1, 5
         call print_record_row(recs(w))
      end do
   end do

   call free_star(id_base, ierr)
   call starlib_shutdown()

contains

   subroutine run_all_wave4_workloads(sb, nz, n_warm, n_rep, out_recs)
      type(star_info), pointer, intent(in) :: sb
      integer, intent(in) :: nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: out_recs(5)

      call bench_star_micro_vars(sb, nz, n_warm, n_rep, out_recs(1))
      call bench_star_hydro_eqns(sb, nz, n_warm, n_rep, out_recs(2))
      call bench_star_newton_step(sb, nz, n_warm, n_rep, out_recs(3))
      call bench_star_remesh(sb, nz, n_warm, n_rep, out_recs(4))
      call bench_star_evolve_step(sb, nz, n_warm, n_rep, out_recs(5))
   end subroutine run_all_wave4_workloads

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

end program run_wave4_bench
