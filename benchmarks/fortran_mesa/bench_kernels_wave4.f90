! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Module: bench_kernels_wave4
!
! Four-stage Fortran MESA Wave 4 1D stellar structure and evolution benchmarks
! (star_micro_vars, star_hydro_eqns, star_newton_step, star_remesh,
! star_evolve_step) evaluated on a resampled 4.0 Msun, Z = 0.02 ZAMS model.
! -----------------------------------------------------------------------------
module bench_kernels_wave4
   use const_def, only: dp, pi
   use star_def, only: star_info, star_ptr, keep_going
   use star_lib, only: alloc_star, free_star, starlib_init, &
      star_set_kap_and_eos_handles, star_set_colors_handles, star_setup, &
      star_load_zams
   use star_job_ctrls_io, only: set_default_star_job_controls, &
      store_star_job_controls
   use alloc, only: reallocate_star_info_arrays
   use read_model, only: finish_load_model
   use evolve, only: do_evolve_step_part1
   use hydro_vars, only: set_vars, set_final_vars
   use hydro_eqns, only: eval_equ
   use solver_support, only: set_xscale_info
   use star_bcyclic, only: bcyclic_factor, bcyclic_solve, clear_storage
   use mesh_functions, only: num_mesh_functions, set_mesh_function_data, &
      max_allowed_gvals
   use struct_burn_mix, only: do_struct_burn_mix
   use omp_lib, only: omp_set_num_threads
   use bench_harness, only: STAGE3_THREADS, BenchRecord, LatencyStats, now_us, &
      compute_latency_stats, max_rel_diff, max_err3
   implicit none
   private

   public :: init_base_zams_star, make_wave4_zams_star
   public :: bench_star_micro_vars, bench_star_hydro_eqns
   public :: bench_star_newton_step, bench_star_remesh, bench_star_evolve_step

contains

   !> Assigns latency statistics for stage 0..3 into a BenchRecord.
   pure subroutine store_stage_stats(stage, stats, rec)
      integer, intent(in) :: stage
      type(LatencyStats), intent(in) :: stats
      type(BenchRecord), intent(inout) :: rec
      select case (stage)
      case (0); rec%stage0 = stats
      case (1); rec%stage1 = stats
      case (2); rec%stage2 = stats
      case default; rec%stage3 = stats
      end select
   end subroutine store_stage_stats

   !> Piece-wise conservative mass-overlap interpolation from dq_old to dq_new.
   pure subroutine interp_q_profile(nz_old, dq_old, val_old, nz_new, dq_new, val_new)
      integer, intent(in) :: nz_old, nz_new
      real(dp), intent(in) :: dq_old(nz_old), val_old(nz_old), dq_new(nz_new)
      real(dp), intent(out) :: val_new(nz_new)
      real(dp) :: q_old(nz_old + 1), q_new(nz_new + 1)
      real(dp) :: q_hi, q_lo, o_hi, o_lo, ov_hi, ov_lo, overlap, acc, w_sum
      integer :: k, k_old

      q_old(1) = 1.0_dp
      do k = 1, nz_old
         q_old(k + 1) = max(0.0_dp, q_old(k) - dq_old(k))
      end do
      q_old(nz_old + 1) = 0.0_dp

      q_new(1) = 1.0_dp
      do k = 1, nz_new
         q_new(k + 1) = max(0.0_dp, q_new(k) - dq_new(k))
      end do
      q_new(nz_new + 1) = 0.0_dp

      k_old = 1
      do k = 1, nz_new
         q_hi = q_new(k); q_lo = q_new(k + 1)
         do while (k_old < nz_old .and. q_old(k_old + 1) >= q_hi)
            k_old = k_old + 1
         end do
         acc = 0.0_dp; w_sum = 0.0_dp
         do while (k_old <= nz_old)
            o_hi = q_old(k_old); o_lo = q_old(k_old + 1)
            ov_hi = min(q_hi, o_hi); ov_lo = max(q_lo, o_lo)
            overlap = max(0.0_dp, ov_hi - ov_lo)
            if (overlap > 0.0_dp) then
               acc = acc + overlap * val_old(k_old)
               w_sum = w_sum + overlap
            end if
            if (o_lo <= q_lo) exit
            k_old = k_old + 1
         end do
         if (w_sum > 0.0_dp) then
            val_new(k) = acc / w_sum
         else
            val_new(k) = val_old(min(k_old, nz_old))
         end if
      end do
   end subroutine interp_q_profile

   !> Allocates and loads the base 4.0 Msun, Z = 0.02 ZAMS star_info model.
   subroutine init_base_zams_star(mesa_dir_arg, id, s, ierr)
      character(len=*), intent(in) :: mesa_dir_arg
      integer, intent(out) :: id, ierr
      type(star_info), pointer, intent(out) :: s

      call alloc_star(id, ierr)
      if (ierr /= 0) return
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      call set_default_star_job_controls()
      call store_star_job_controls(s)
      s%job%mesa_dir = trim(mesa_dir_arg)
      s%inlist_fname = trim(mesa_dir_arg) // '/star/test/inlist'
      call starlib_init(s, ierr)
      if (ierr /= 0) return
      call star_set_kap_and_eos_handles(id, ierr)
      if (ierr /= 0) return
      call star_set_colors_handles(id, ierr)
      if (ierr /= 0) return
      call star_setup(id, s%inlist_fname, ierr)
      if (ierr /= 0) return

      s%initial_mass = 4.0_dp; s%initial_z = 0.02_dp; s%kap_rq%Zbase = 0.02_dp
      s%okay_to_remesh = .false.; s%do_history_file = .false.
      s%terminal_interval = 1000000; s%profile_interval = 1000000
      s%photo_interval = 1000000; s%write_header_frequency = 1000000
      call star_load_zams(id, ierr)
   end subroutine init_base_zams_star

   !> Constructs a resampled 4.0 Msun, Z = 0.02 ZAMS star_info with nz zones.
   subroutine make_wave4_zams_star(s_base, nz, id, s, ierr)
      type(star_info), pointer, intent(in) :: s_base
      integer, intent(in) :: nz
      integer, intent(out) :: id, ierr
      type(star_info), pointer, intent(out) :: s

      integer :: k, j, idx, res, b_nz, spec
      real(dp) :: frac, pos, rem, sum_dq, r3_run, dm_k, rho_k
      real(dp), allocatable :: q_edges(:), dq_new(:), xh_new(:,:), xa_new(:,:)

      call init_base_zams_star(s_base%job%mesa_dir, id, s, ierr)
      if (ierr /= 0) return

      b_nz = s_base%nz; spec = s_base%species
      if (nz /= b_nz) then
         allocate(q_edges(nz + 1), dq_new(nz), xh_new(s%nvar_hydro, nz), xa_new(spec, nz))
         q_edges(1) = 1.0_dp; q_edges(2) = s_base%q(2)
         do k = 3, nz
            frac = real(k - 2, dp) / real(nz - 1, dp)
            pos = 2.0_dp + frac * real(b_nz - 2, dp)
            idx = min(int(floor(pos)), b_nz - 1)
            rem = pos - real(idx, dp)
            q_edges(k) = (1.0_dp - rem) * s_base%q(idx) + rem * s_base%q(idx + 1)
         end do
         q_edges(nz + 1) = 0.0_dp

         do k = 1, nz
            dq_new(k) = max(1.0e-12_dp, q_edges(k) - q_edges(k + 1))
         end do
         dq_new = dq_new / sum(dq_new)

         do j = 1, s%nvar_hydro
            call interp_q_profile(b_nz, s_base%dq(1:b_nz), &
               s_base%xh(j, 1:b_nz), nz, dq_new, xh_new(j, :))
         end do
         do j = 1, spec
            call interp_q_profile(b_nz, s_base%dq(1:b_nz), &
               s_base%xa(j, 1:b_nz), nz, dq_new, xa_new(j, :))
         end do

         r3_run = 0.0_dp
         do k = nz, 1, -1
            dm_k = s_base%mstar * dq_new(k)
            rho_k = max(1.0e-30_dp, exp(xh_new(s%i_lnd, k)))
            r3_run = r3_run + (3.0_dp * dm_k) / (4.0_dp * pi * rho_k)
            xh_new(s%i_lnR, k) = log(r3_run) / 3.0_dp
         end do
         do k = 1, nz
            xa_new(:, k) = max(0.0_dp, xa_new(:, k))
            sum_dq = sum(xa_new(:, k))
            if (sum_dq > 0.0_dp) xa_new(:, k) = xa_new(:, k) / sum_dq
         end do

         s%nz = nz; s%prev_mesh_nz = nz
         call reallocate_star_info_arrays(s, ierr)
         if (ierr /= 0) return
         s%dq(1:nz) = dq_new; s%xh(1:s%nvar_hydro, 1:nz) = xh_new
         s%xa(1:spec, 1:nz) = xa_new; s%omega(1:nz) = 0.0_dp; s%j_rot(1:nz) = 0.0_dp
         call finish_load_model(s, .false., ierr)
         deallocate(q_edges, dq_new, xh_new, xa_new)
         if (ierr /= 0) return
      end if

      res = do_evolve_step_part1(id, .true.)
      if (res /= keep_going) ierr = -1
      s%xh_start(1:s%nvar_hydro, 1:nz) = s%xh(1:s%nvar_hydro, 1:nz)
      s%xa_start(1:s%species, 1:nz) = s%xa(1:s%species, 1:nz)
      s%xa_sub_xa_start(1:s%species, 1:nz) = 0.0_dp
      s%solver_dx(1:s%nvar_total, 1:nz) = 0.0_dp
      s%lnd_start(1:nz) = s%lnd(1:nz); s%rho_start(1:nz) = s%rho(1:nz)
      s%lnT_start(1:nz) = s%lnT(1:nz); s%T_start(1:nz) = s%T(1:nz)
      s%lnR_start(1:nz) = s%lnR(1:nz); s%r_start(1:nz) = s%r(1:nz)
      s%L_start(1:nz) = s%L(1:nz); s%lnPeos_start(1:nz) = s%lnPeos(1:nz)
      s%Peos_start(1:nz) = s%Peos(1:nz); s%Peos_face_start(1:nz) = s%Peos(1:nz)
      s%reconstructed_P_face_start(1:nz) = s%Peos(1:nz)
      s%lnPgas_start(1:nz) = s%lnPgas(1:nz); s%energy_start(1:nz) = s%energy(1:nz)
      s%lnS_start(1:nz) = s%lnS(1:nz); s%grada_start(1:nz) = s%grada(1:nz)
      s%chiT_start(1:nz) = s%chiT(1:nz); s%chiRho_start(1:nz) = s%chiRho(1:nz)
      s%cp_start(1:nz) = s%cp(1:nz); s%Cv_start(1:nz) = s%Cv(1:nz)
      s%dE_dRho_start(1:nz) = s%dE_dRho(1:nz); s%gam_start(1:nz) = s%gam(1:nz)
      s%zbar_start(1:nz) = s%zbar(1:nz); s%mu_start(1:nz) = s%mu(1:nz)
      s%ye_start(1:nz) = s%ye(1:nz); s%eps_nuc_start(1:nz) = s%eps_nuc(1:nz)
      s%non_nuc_neu_start(1:nz) = s%non_nuc_neu(1:nz)
      s%opacity_start(1:nz) = s%opacity(1:nz); s%m_grav_start(1:nz) = s%m_grav(1:nz)
      call set_xscale_info(s, s%nvar_total, ierr)
   end subroutine make_wave4_zams_star

   !> Workload 1: Zone/face stellar microphysics (star_micro_vars).
   subroutine bench_star_micro_vars(s_base, nz, n_warmup, n_repeats, rec)
      type(star_info), pointer, intent(in) :: s_base
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec

      type(star_info), pointer :: s
      type(LatencyStats) :: stats
      integer :: id, ierr, stage, rep, reps, k
      real(dp) :: t0
      real(dp), allocatable :: samples(:), tmp(:), out_stage(:,:)

      reps = max(n_repeats, 3)
      call make_wave4_zams_star(s_base, nz, id, s, ierr)
      allocate(samples(reps), out_stage(nz, 0:3))

      do stage = 0, 3
         if (stage == 3) then; call omp_set_num_threads(STAGE3_THREADS)
         else; call omp_set_num_threads(1); end if
         do rep = 1, n_warmup + reps
            if (rep > n_warmup) t0 = now_us()
            if (stage == 0) allocate(tmp(nz))
            s%need_to_setvars = .true.
            if (stage <= 1) then
               call set_vars(s, s%dt, ierr)
            else
               call set_final_vars(s, s%dt, ierr)
            end if
            do k = 1, nz
               out_stage(k, stage) = s%lnPeos(k) + s%gradT(k) + log(s%opacity(k))
            end do
            if (stage == 0) deallocate(tmp)
            if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
         end do
         call compute_latency_stats(samples, reps, stats)
         call store_stage_stats(stage, stats, rec)
      end do

      rec%workload = 'star_micro_vars'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(nz, out_stage(:, 0), out_stage(:, 1)), &
         max_rel_diff(nz, out_stage(:, 0), out_stage(:, 2)), &
         max_rel_diff(nz, out_stage(:, 0), out_stage(:, 3)))
      rec%checksum = sum(out_stage(:, 3))
      call free_star(id, ierr)
      deallocate(samples, out_stage)
   end subroutine bench_star_micro_vars

   !> Workload 2: Coupled 1D Henyey equations and Jacobian (star_hydro_eqns).
   subroutine bench_star_hydro_eqns(s_base, nz, n_warmup, n_repeats, rec)
      type(star_info), pointer, intent(in) :: s_base
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec

      type(star_info), pointer :: s
      type(LatencyStats) :: stats
      integer :: id, ierr, stage, rep, reps, k, i, nvar, neq
      real(dp) :: t0
      real(dp), pointer :: lblk1(:), dblk1(:), ublk1(:)
      real(dp), allocatable :: samples(:), out_stage(:,:,:)

      reps = max(n_repeats, 3); nvar = 4; neq = nvar * nz
      call make_wave4_zams_star(s_base, nz, id, s, ierr)
      s%do_mix = .false.; s%do_burn = .false.
      if (allocated(s%equ1)) deallocate(s%equ1)
      allocate(s%equ1(neq))
      s%equ(1:nvar, 1:nz) => s%equ1(1:neq)
      allocate(lblk1(nvar * neq), dblk1(nvar * neq), ublk1(nvar * neq))
      s%lblk(1:nvar, 1:nvar, 1:nz) => lblk1
      s%dblk(1:nvar, 1:nvar, 1:nz) => dblk1
      s%ublk(1:nvar, 1:nvar, 1:nz) => ublk1
      allocate(samples(reps), out_stage(nvar, nz, 0:3))

      do stage = 0, 3
         if (stage == 3) then; call omp_set_num_threads(STAGE3_THREADS)
         else; call omp_set_num_threads(1); end if
         do rep = 1, n_warmup + reps
            if (rep > n_warmup) t0 = now_us()
            if (stage == 0) then; lblk1 = 0.0_dp; dblk1 = 0.0_dp; ublk1 = 0.0_dp; end if
            if (stage <= 1) s%equ1 = 0.0_dp
            call eval_equ(s, nvar, ierr)
            do k = 1, nz
               do i = 1, nvar
                  out_stage(i, k, stage) = abs(s%equ(i, k)) + abs(s%dblk(i, i, k))
               end do
            end do
            if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
         end do
         call compute_latency_stats(samples, reps, stats)
         call store_stage_stats(stage, stats, rec)
      end do

      rec%workload = 'star_hydro_eqns'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(neq, out_stage(:, :, 0), out_stage(:, :, 1)), &
         max_rel_diff(neq, out_stage(:, :, 0), out_stage(:, :, 2)), &
         max_rel_diff(neq, out_stage(:, :, 0), out_stage(:, :, 3)))
      rec%checksum = sum(out_stage(:, :, 3))
      deallocate(lblk1, dblk1, ublk1, samples, out_stage)
      call free_star(id, ierr)
   end subroutine bench_star_hydro_eqns

   !> Workload 3: Single Henyey linearize-and-solve step (star_newton_step).
   subroutine bench_star_newton_step(s_base, nz, n_warmup, n_repeats, rec)
      type(star_info), pointer, intent(in) :: s_base
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec

      type(star_info), pointer :: s
      type(LatencyStats) :: stats
      integer :: id, ierr, stage, rep, reps, nvar, neq
      real(dp) :: t0
      character(len=nz) :: equed1
      integer, pointer :: ipiv1(:)
      real(dp), pointer :: lblk1(:), dblk1(:), ublk1(:), lblkF1(:), dblkF1(:), ublkF1(:)
      real(dp), pointer :: b1(:), soln1(:), rscale(:), cscale(:)
      real(dp), allocatable :: samples(:), dx_stage(:,:)

      reps = max(n_repeats, 3); nvar = 4; neq = nvar * nz
      call make_wave4_zams_star(s_base, nz, id, s, ierr)
      s%dt = 1.0e2_dp; s%do_mix = .false.; s%do_burn = .false.
      if (allocated(s%equ1)) deallocate(s%equ1)
      allocate(s%equ1(neq))
      s%equ(1:nvar, 1:nz) => s%equ1(1:neq)
      allocate(lblk1(nvar * neq), dblk1(nvar * neq), ublk1(nvar * neq))
      allocate(lblkF1(nvar * neq), dblkF1(nvar * neq), ublkF1(nvar * neq))
      allocate(b1(neq), soln1(neq), rscale(neq), cscale(neq), ipiv1(neq))
      s%lblk(1:nvar, 1:nvar, 1:nz) => lblk1
      s%dblk(1:nvar, 1:nvar, 1:nz) => dblk1
      s%ublk(1:nvar, 1:nvar, 1:nz) => ublk1
      allocate(samples(reps), dx_stage(neq, 0:3))

      do stage = 0, 3
         if (stage == 3) then; call omp_set_num_threads(STAGE3_THREADS)
         else; call omp_set_num_threads(1); end if
         do rep = 1, n_warmup + reps
            if (rep > n_warmup) t0 = now_us()
            if (stage == 0) then
               if (associated(s%bcyclic_odd_storage)) call clear_storage(s)
               lblk1 = 0.0_dp; dblk1 = 0.0_dp; ublk1 = 0.0_dp
            end if
            if (stage <= 1) s%equ1 = 0.0_dp
            call eval_equ(s, nvar, ierr)
            b1 = -s%equ1
            call bcyclic_factor(s, nvar, nz, lblk1, dblk1, ublk1, &
               lblkF1, dblkF1, ublkF1, ipiv1, b1, rscale, cscale, equed1, 1, ierr)
            call bcyclic_solve(s, nvar, nz, lblk1, dblk1, ublk1, &
               lblkF1, dblkF1, ublkF1, ipiv1, b1, soln1, rscale, cscale, equed1, 1, ierr)
            dx_stage(:, stage) = soln1
            if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
         end do
         call compute_latency_stats(samples, reps, stats)
         call store_stage_stats(stage, stats, rec)
      end do

      rec%workload = 'star_newton_step'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(neq, dx_stage(:, 0), dx_stage(:, 1)), &
         max_rel_diff(neq, dx_stage(:, 0), dx_stage(:, 2)), &
         max_rel_diff(neq, dx_stage(:, 0), dx_stage(:, 3)))
      rec%checksum = sum(abs(dx_stage(:, 3)))
      call clear_storage(s)
      deallocate(lblk1, dblk1, ublk1, lblkF1, dblkF1, ublkF1, b1, soln1, rscale, cscale, ipiv1)
      deallocate(samples, dx_stage)
      call free_star(id, ierr)
   end subroutine bench_star_newton_step

   !> Workload 4: Mesh gradient function evaluation (star_remesh).
   subroutine bench_star_remesh(s_base, nz, n_warmup, n_repeats, rec)
      type(star_info), pointer, intent(in) :: s_base
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec

      type(star_info), pointer :: s
      type(LatencyStats) :: stats
      integer :: id, ierr, stage, rep, reps, nfcns
      real(dp) :: t0
      character(len=32) :: names(max_allowed_gvals)
      logical :: is_xa(max_allowed_gvals), is_logt(max_allowed_gvals)
      real(dp), pointer :: vals1(:), tmp1(:)
      real(dp), allocatable :: samples(:), out_stage(:,:)

      reps = max(n_repeats, 3)
      call make_wave4_zams_star(s_base, nz, id, s, ierr)
      nfcns = max(1, num_mesh_functions(s))
      allocate(vals1(nz * nfcns), samples(reps), out_stage(nz, 0:3))

      do stage = 0, 3
         if (stage == 3) then; call omp_set_num_threads(STAGE3_THREADS)
         else; call omp_set_num_threads(1); end if
         do rep = 1, n_warmup + reps
            if (rep > n_warmup) t0 = now_us()
            if (stage == 0) then
               allocate(tmp1(nz * nfcns))
               call set_mesh_function_data(s, nfcns, names, is_xa, is_logt, tmp1, ierr)
               out_stage(:, stage) = tmp1(1:nz)
               deallocate(tmp1)
            else
               if (stage == 1) vals1 = 0.0_dp
               call set_mesh_function_data(s, nfcns, names, is_xa, is_logt, vals1, ierr)
               out_stage(:, stage) = vals1(1:nz)
            end if
            if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
         end do
         call compute_latency_stats(samples, reps, stats)
         call store_stage_stats(stage, stats, rec)
      end do

      rec%workload = 'star_remesh'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(nz, out_stage(:, 0), out_stage(:, 1)), &
         max_rel_diff(nz, out_stage(:, 0), out_stage(:, 2)), &
         max_rel_diff(nz, out_stage(:, 0), out_stage(:, 3)))
      rec%checksum = sum(out_stage(:, 3))
      deallocate(vals1, samples, out_stage)
      call free_star(id, ierr)
   end subroutine bench_star_remesh

   !> Workload 5: Full coupled structure + burn + mix step (star_evolve_step).
   subroutine bench_star_evolve_step(s_base, nz, n_warmup, n_repeats, rec)
      type(star_info), pointer, intent(in) :: s_base
      integer, intent(in) :: nz, n_warmup, n_repeats
      type(BenchRecord), intent(out) :: rec

      type(star_info), pointer :: s
      type(LatencyStats) :: stats
      integer :: id, ierr, stage, rep, reps, res, k
      real(dp) :: t0
      real(dp), allocatable :: xh_save(:,:), xa_save(:,:), samples(:), out_stage(:,:)

      reps = max(n_repeats, 3)
      call make_wave4_zams_star(s_base, nz, id, s, ierr)
      s%dt = 1.0e2_dp
      allocate(xh_save(s%nvar_hydro, nz), xa_save(s%species, nz))
      xh_save = s%xh(1:s%nvar_hydro, 1:nz); xa_save = s%xa(1:s%species, 1:nz)
      allocate(samples(reps), out_stage(nz, 0:3))

      do stage = 0, 3
         if (stage == 3) then; call omp_set_num_threads(STAGE3_THREADS)
         else; call omp_set_num_threads(1); end if
         do rep = 1, n_warmup + reps
            s%xh(1:s%nvar_hydro, 1:nz) = xh_save
            s%xa(1:s%species, 1:nz) = xa_save
            s%need_to_setvars = .true.
            call set_vars(s, s%dt, ierr)
            if (rep > n_warmup) t0 = now_us()
            res = do_struct_burn_mix(s, .false.)
            do k = 1, nz
               out_stage(k, stage) = s%lnT(k) + s%lnd(k) + s%xa(1, k)
            end do
            if (rep > n_warmup) samples(rep - n_warmup) = now_us() - t0
         end do
         call compute_latency_stats(samples, reps, stats)
         call store_stage_stats(stage, stats, rec)
      end do

      rec%workload = 'star_evolve_step'; rec%n_zones = nz
      rec%max_rel_err = max_err3( &
         max_rel_diff(nz, out_stage(:, 0), out_stage(:, 1)), &
         max_rel_diff(nz, out_stage(:, 0), out_stage(:, 2)), &
         max_rel_diff(nz, out_stage(:, 0), out_stage(:, 3)))
      rec%checksum = sum(out_stage(:, 3))
      deallocate(xh_save, xa_save, samples, out_stage)
      call free_star(id, ierr)
   end subroutine bench_star_evolve_step

end module bench_kernels_wave4
