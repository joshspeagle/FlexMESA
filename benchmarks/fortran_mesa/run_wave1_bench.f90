! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Program: run_wave1_bench
!
! Executes the Wave 1 four-stage Fortran MESA benchmark and cross-language
! parity suite across all six Wave 1 workloads:
!   1. auto_diff     (auto_diff_real_2var_order2)
!   2. interp_2d     (interp_mkbicub_db / interp_evbicub_db / fvbicub_db)
!   3. neu           (Itoh et al. 1996 pair + plasmon neu_get)
!   4. mtx_tridiag   (5 x 5 block-tridiagonal bcyclic_dble_decsolblk & Thomas)
!   5. turb_mlt      (Cox 1968 set_MLT with auto_diff_real_star_order1)
!   6. turb_tdc      (Kuhfuss 1986 set_TDC with auto_diff_real_star_order1)
!
! First verifies internal stage parity (Stages 1..3 vs. Stage 0 <= 1e-12)
! across N_zones in {8, 32} and checks N_zones = 8 checksums against the
! pinned cross-language reference values from flexmesa-bench (Rust) and
! flexmesa.bench (Python/NumPy/JAX). Then runs the full timing suite across
! N_zones in {128, 1024, 4096} and prints the Markdown summary table.
! -----------------------------------------------------------------------------
program run_wave1_bench
   use const_def, only: dp
   use const_lib, only: const_init
   use math_lib, only: math_init
   use bench_harness, only: BenchRecord, ThermoGrid, BicubTable, &
      init_thermo_grid, free_thermo_grid, init_bicub_table, free_bicub_table
   use bench_kernels_micro, only: bench_autodiff, bench_interp2d, bench_neu
   use bench_kernels_solver, only: bench_mtx_block_tridiag, bench_turb_mlt, &
      bench_turb_tdc
   implicit none

   ! Pinned N_zones = 8 cross-language checksums matching flexmesa-bench (Rust)
   ! and flexmesa.bench (Python/NumPy/JAX).
   real(dp), parameter :: EXPECTED_CHECKSUMS_N8(6) = [ &
      1.089274155553244e4_dp, &
      3.622775349153983e2_dp, &
      9.868900412911318e12_dp, &
      1.065650530823759e1_dp, &
      3.419048546150960_dp, &
      3.3187342241650115_dp]

   integer, parameter :: PARITY_ZONES(2) = [8, 32]
   integer, parameter :: BENCH_ZONES(3) = [128, 1024, 4096]
   integer, parameter :: N_WARMUP = 3
   integer, parameter :: N_REPEATS = 9

   character(len=256) :: mesa_dir_arg
   integer :: ierr, idx, w
   real(dp) :: bc_1t(3), bc_1t_iqr(3), bc_4t(3), bc_4t_iqr(3)
   real(dp) :: d1, d2, d3, d4, rel_chk, tol_chk
   type(BicubTable) :: table
   type(BenchRecord) :: recs(6)

   ! Initialize MESA constants and math lookup/CRLIBM tables.
   mesa_dir_arg = ''
   call const_init(mesa_dir_arg, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call init_bicub_table(table)

   ! Step 1: Verify internal stage parity and cross-language N_zones = 8
   ! reference checksums before running the timing suite.
   do idx = 1, size(PARITY_ZONES)
      call run_all_workloads( &
         PARITY_ZONES(idx), table, 1, 3, recs, d1, d2, d3, d4)
      do w = 1, 6
         if ((recs(w)%max_rel_err /= recs(w)%max_rel_err) .or. &
             (recs(w)%max_rel_err > 1.0e-12_dp)) then
            write (*, '(a,a,a,es12.4)') 'Parity failure in ', &
               trim(recs(w)%workload), ': max_rel_err = ', recs(w)%max_rel_err
            stop 1
         end if
         if (PARITY_ZONES(idx) == 8) then
            rel_chk = abs(recs(w)%checksum - EXPECTED_CHECKSUMS_N8(w)) &
               / abs(EXPECTED_CHECKSUMS_N8(w))
            ! Upstream MESA fvbicub_db (w == 2) uses implicit 32-bit REAL
            ! locals (xp, yp, cx, cy, hx2, hy2, z36th), giving ~7.2e-9 vs f64.
            tol_chk = merge(1.0e-8_dp, 1.0e-11_dp, w == 2)
            if ((rel_chk /= rel_chk) .or. (rel_chk > tol_chk)) then
               write (*, '(a,a,a,es12.4)') &
                  'Cross-language checksum mismatch ', &
                  trim(recs(w)%workload), ': rel_err = ', rel_chk
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
      call run_all_workloads( &
         BENCH_ZONES(idx), table, N_WARMUP, N_REPEATS, recs, &
         bc_1t(idx), bc_1t_iqr(idx), bc_4t(idx), bc_4t_iqr(idx))
      do w = 1, 6
         call print_record_row(recs(w))
      end do
   end do

   write (*, '(a)') ''
   write (*, '(a)') '# Single-Matrix BCYCLIC (bcyclic_dble_decsolblk) Latency:'
   do idx = 1, size(BENCH_ZONES)
      write (*, '(a,i5,a,f8.2,a,f6.2,a,f8.2,a,f6.2,a)') &
         '#   N_zones = ', BENCH_ZONES(idx), &
         ': 1-thread = ', bc_1t(idx), ' +/- ', bc_1t_iqr(idx), &
         ' us, 4-thread OpenMP = ', bc_4t(idx), ' +/- ', bc_4t_iqr(idx), ' us'
   end do

   call free_bicub_table(table)

contains

   !> Runs all six Wave 1 workloads for a single n_zones value.
   subroutine run_all_workloads( &
         nz, tbl, n_warm, n_rep, out_recs, bc1_med, bc1_iqr, bc4_med, bc4_iqr)
      integer, intent(in) :: nz, n_warm, n_rep
      type(BicubTable), intent(inout) :: tbl
      type(BenchRecord), intent(out) :: out_recs(6)
      real(dp), intent(out) :: bc1_med, bc1_iqr, bc4_med, bc4_iqr
      type(ThermoGrid) :: grid

      call init_thermo_grid(nz, grid)
      call bench_autodiff(grid, n_warm, n_rep, out_recs(1))
      call bench_interp2d(grid, tbl, n_warm, n_rep, out_recs(2))
      call bench_neu(grid, n_warm, n_rep, out_recs(3))
      call bench_mtx_block_tridiag( &
         nz, n_warm, n_rep, out_recs(4), bc1_med, bc4_med)
      bc1_iqr = out_recs(4)%stage1%iqr_us
      bc4_iqr = 0.0_dp
      call bench_turb_mlt(grid, n_warm, n_rep, out_recs(5))
      call bench_turb_tdc(grid, n_warm, n_rep, out_recs(6))
      call free_thermo_grid(grid)
   end subroutine run_all_workloads

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

end program run_wave1_bench
