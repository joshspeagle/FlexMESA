! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Program: run_wave3b_bench
!
! Executes the Wave 3b four-stage Fortran MESA benchmark and cross-language
! parity suite across all five Wave 3b stellar microphysics workloads:
!   1. eos_4d       (4D OPAL/SCVH tabulated equation of state)
!   2. kap_fixed    (Type 1 fixed-metal + conductive + Compton opacity)
!   3. rates_screen (REACLIB r_c12_ag_o16 rate + extended screen5 screening)
!   4. net_eval     (basic.net 8-isotope reaction network RHS & Jacobian)
!   5. atm_table    (ATM_TABLE_PHOTOSPHERE bicubic/Steffen atmosphere table)
! -----------------------------------------------------------------------------
program run_wave3b_bench
   use const_def, only: dp
   use const_lib, only: const_init
   use math_lib, only: math_init
   use chem_lib, only: chem_init, chem_shutdown
   use eos_lib, only: eos_init, eos_shutdown, alloc_eos_handle, free_eos_handle
   use kap_lib, only: kap_init, kap_shutdown, alloc_kap_handle, free_kap_handle
   use rates_lib, only: rates_init, rates_shutdown, rates_warning_init
   use net_lib, only: net_init, net_shutdown, alloc_net_handle, &
      free_net_handle, net_start_def, read_net_file, net_finish_def, &
      net_setup_tables
   use atm_lib, only: atm_init, atm_shutdown
   use bench_harness, only: BenchRecord
   use bench_kernels_wave3b, only: bench_eos_4d, bench_kap_fixed, &
      bench_rates_screen, bench_net_eval, bench_atm_table
   implicit none

   ! Pinned N_zones = 8 cross-language checksums matching flexmesa-bench (Rust)
   ! and flexmesa.bench (Python/NumPy/JAX).
   real(dp), parameter :: EXPECTED_WAVE3B_CHECKSUMS_N8(5) = [ &
      2.037364564689326e2_dp, &
      2.909195757111935e0_dp, &
      -1.139188914724813e2_dp, &
      9.619649848151354e1_dp, &
      8.985411175716764e1_dp]

   integer, parameter :: PARITY_ZONES(2) = [8, 32]
   integer, parameter :: BENCH_ZONES(3) = [128, 1024, 4096]
   integer, parameter :: N_WARMUP = 3
   integer, parameter :: N_REPEATS = 9

   character(len=512) :: mesa_dir_arg
   integer :: ierr, idx, w, eos_handle, kap_handle, net_handle
   real(dp) :: rel_chk
   type(BenchRecord) :: recs(5)

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

   call kap_init(.false., ' ', ierr)
   if (ierr /= 0) stop 'kap_init failed'
   kap_handle = alloc_kap_handle(ierr)
   if (ierr /= 0) stop 'alloc_kap_handle failed'

   call rates_init('reactions.list', '', 'rate_tables', .false., &
      .false., '', '', '', ierr)
   if (ierr /= 0) stop 'rates_init failed'
   call rates_warning_init(.true., 10.0_dp)

   call net_init(ierr)
   if (ierr /= 0) stop 'net_init failed'
   net_handle = alloc_net_handle(ierr)
   if (ierr /= 0) stop 'alloc_net_handle failed'
   call net_start_def(net_handle, ierr)
   call read_net_file('basic.net', net_handle, ierr)
   call net_finish_def(net_handle, ierr)
   call net_setup_tables(net_handle, 'rate_cache', ierr)
   if (ierr /= 0) stop 'net_setup_tables failed'

   call atm_init(.false., ierr)
   if (ierr /= 0) stop 'atm_init failed'

   ! Step 1: Verify internal stage parity and cross-language N_zones = 8
   ! reference checksums before running the timing suite.
   do idx = 1, size(PARITY_ZONES)
      call run_all_wave3b_workloads( &
         eos_handle, kap_handle, net_handle, PARITY_ZONES(idx), 1, 3, recs)
      do w = 1, 5
         if ((recs(w)%max_rel_err /= recs(w)%max_rel_err) .or. &
             (recs(w)%max_rel_err > 1.0e-11_dp)) then
            write (*, '(a,a,a,es12.4)') 'Parity failure in ', &
               trim(recs(w)%workload), ': max_rel_err = ', recs(w)%max_rel_err
            stop 1
         end if
         if (PARITY_ZONES(idx) == 8) then
            rel_chk = abs(recs(w)%checksum - EXPECTED_WAVE3B_CHECKSUMS_N8(w)) &
               / abs(EXPECTED_WAVE3B_CHECKSUMS_N8(w))
            if ((rel_chk /= rel_chk) .or. (rel_chk > 1.0e-9_dp)) then
               write (*, '(a,a,a,es12.4,a,es22.15,a,es22.15)') &
                  'Cross-language checksum mismatch ', &
                  trim(recs(w)%workload), ': rel_err = ', rel_chk, &
                  ' got ', recs(w)%checksum, &
                  ' expected ', EXPECTED_WAVE3B_CHECKSUMS_N8(w)
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
      call run_all_wave3b_workloads( &
         eos_handle, kap_handle, net_handle, &
         BENCH_ZONES(idx), N_WARMUP, N_REPEATS, recs)
      do w = 1, 5
         call print_record_row(recs(w))
      end do
   end do

   call atm_shutdown()
   call free_net_handle(net_handle)
   call net_shutdown()
   call rates_shutdown()
   call free_kap_handle(kap_handle)
   call kap_shutdown()
   call free_eos_handle(eos_handle)
   call eos_shutdown()
   call chem_shutdown()

contains

   subroutine run_all_wave3b_workloads( &
         eh, kh, nh, nz, n_warm, n_rep, out_recs)
      integer, intent(in) :: eh, kh, nh, nz, n_warm, n_rep
      type(BenchRecord), intent(out) :: out_recs(5)

      call bench_eos_4d(eh, nz, n_warm, n_rep, out_recs(1))
      call bench_kap_fixed(kh, nz, n_warm, n_rep, out_recs(2))
      call bench_rates_screen(nz, n_warm, n_rep, out_recs(3))
      call bench_net_eval(nh, nz, n_warm, n_rep, out_recs(4))
      call bench_atm_table(nz, n_warm, n_rep, out_recs(5))
   end subroutine run_all_wave3b_workloads

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

end program run_wave3b_bench
