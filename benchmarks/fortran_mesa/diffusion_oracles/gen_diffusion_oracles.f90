! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex + decimal reference vectors against compiled MESA 26.4.1
! (`libstar.a` + `CRMATH`) for element diffusion and phase separation (#156).
!
! Sections emitted:
!   1. paquette      : paquette_coeffs::paquette_coefficients (4 regimes)
!   2. matrix_coeffs : diffusion_support::get_matrix_coeffs (8 solver/transport regimes)
!   3. solve_diff    : diffusion::do_solve_diffusion (BE_solve Thoul, BE_solve CGS WD,
!                      and isolve with all 7 Rosenbrock solvers)
!   4. phase_sep     : phase_separation::do_phase_separation (CO, ONe, and CO with
!                      phase_separation_mixing_use_brunt)
!   5. kernels       : private MESA kernels get_Dstar_OCP (with bitsafe_erf_fit),
!                      blouin_delta_xo, and blouin_delta_xne, compiled from source text
!                      extracted verbatim from the MESA tree by the Makefile

! The MESA 26.4.1 routines below are private to their modules and are not callable
! through libstar.a. The Makefile copies their source text verbatim from
! star/private/diffusion_support.f90 and star/private/phase_separation.f90 into
! mesa_private_kernels.inc in the build directory, and this module compiles that
! text with the same compiler flags as the rest of the generator.
module gen_diffusion_private_kernels
   use const_def, only: dp, pi
   use math_lib
   implicit none
   private
   public :: get_Dstar_OCP, blouin_delta_xo, blouin_delta_xne
contains
   include 'mesa_private_kernels.inc'
end module gen_diffusion_private_kernels

program gen_diffusion_oracles
   use, intrinsic :: iso_fortran_env, only: int64, compiler_version
   use const_def, only: dp, pi, pi4, amu, me, msun, rsun, lsun, standard_cgrav
   use const_lib, only: const_init
   use math_lib, only: math_init, exp10
   use chem_def, only: chem_isos, ih1, ihe4, ic12, io16, ine20, img24, ife56
   use star_def, only: star_info, star_ptr
   use star_lib, only: alloc_star, free_star, starlib_init, &
      star_set_kap_and_eos_handles, star_set_colors_handles, star_setup, star_load_zams
   use star_job_ctrls_io, only: set_default_star_job_controls, store_star_job_controls
   use alloc, only: reallocate_star_info_arrays
   use hydro_vars, only: set_vars
   use paquette_coeffs, only: initialise_collision_integrals, &
      free_collision_integrals, paquette_coefficients
   use diffusion_support, only: get_matrix_coeffs
   use diffusion, only: do_solve_diffusion
   use phase_separation, only: do_phase_separation
   use gen_diffusion_private_kernels, only: get_Dstar_OCP, blouin_delta_xo, blouin_delta_xne
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver
   integer :: ierr, u, id
   type(star_info), pointer :: s

   call get_environment_variable('MESA_DIR', mesa_dir)
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   if (command_argument_count() >= 1) then
      call get_command_argument(1, out_path)
   else
      out_path = 'mesa_26_4_1_diffusion_oracles.tsv'
   end if

   ierr = 0
   call alloc_star(id, ierr)
   if (ierr /= 0) stop 'alloc_star failed'
   call star_ptr(id, s, ierr)
   if (ierr /= 0) stop 'star_ptr failed'
   call set_default_star_job_controls()
   call store_star_job_controls(s, ierr)
   if (ierr /= 0) stop 'store_star_job_controls failed'
   s%job%mesa_dir = trim(mesa_dir)
   s%inlist_fname = trim(mesa_dir) // '/star/test/inlist'
   call starlib_init(s, ierr)
   if (ierr /= 0) stop 'starlib_init failed'
   call star_set_kap_and_eos_handles(id, ierr)
   if (ierr /= 0) stop 'star_set_kap_and_eos_handles failed'
   call star_set_colors_handles(id, ierr)
   if (ierr /= 0) stop 'star_set_colors_handles failed'
   call star_setup(id, s%inlist_fname, ierr)
   if (ierr /= 0) stop 'star_setup failed'

   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output file'

   write(u, '(A)') '# FlexMESA MESA 26.4.1 element diffusion and phase separation oracle vectors (#156)'
   write(u, '(A)') '# mesa_version: ' // trim(mesa_ver)
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(sdk_ver)
   write(u, '(A)') '# math_backend: CRMATH'
   write(u, '(A)') '# compiler: ' // compiler_version()
   write(u, '(A)') 'section	case_id	field	i	j	raw_bits_hex	decimal'

   call emit_paquette_oracles(u)
   call emit_matrix_coeffs_oracles(u, s)
   call emit_solve_diffusion_oracles(u, s)
   call emit_phase_sep_oracles(u, mesa_dir)
   call emit_kernel_oracles(u)

   close(u)
   call free_star(id, ierr)

contains

   subroutine read_ver(mdir, ver)
      character(len=*), intent(in) :: mdir
      character(len=*), intent(out) :: ver
      integer :: vu, ios
      ver = 'unknown'
      open(newunit=vu, file=trim(mdir)//'/data/version_number', status='old', action='read', iostat=ios)
      if (ios == 0) then
         read(vu, '(A)', iostat=ios) ver
         close(vu)
      end if
      ver = adjustl(ver)
   end subroutine read_ver

   subroutine emit_row(unit_no, sec, cid, fld, ii, jj, val)
      integer, intent(in) :: unit_no, ii, jj
      character(len=*), intent(in) :: sec, cid, fld
      real(dp), intent(in) :: val
      integer(int64) :: bits
      character(len=20) :: hbuf
      character(len=32) :: dbuf
      bits = transfer(val, 0_int64)
      write(hbuf, '(A,Z16.16)') '0x', bits
      write(dbuf, '(ES25.17E3)') val
      write(unit_no, '(A,A,A,A,A,A,I0,A,I0,A,A,A,A)') &
         trim(sec), achar(9), trim(cid), achar(9), trim(fld), achar(9), &
         ii, achar(9), jj, achar(9), trim(hbuf), achar(9), trim(adjustl(dbuf))
   end subroutine emit_row

   ! --------------------------------------------------------------------------
   ! Section 1: Paquette et al. (1986) collision integrals across psi regimes
   ! --------------------------------------------------------------------------
   subroutine emit_paquette_oracles(unit_no)
      integer, intent(in) :: unit_no
      integer, parameter :: nn = 5
      real(dp) :: can1(nn), kzn1(nn), na1(nn)
      real(dp) :: ddiff(nn, nn), kdiff(nn, nn), zdiff(nn, nn), &
         zdiff1(nn, nn), zdiff2(nn, nn), ath(nn, nn)
      integer :: icase, i, j
      real(dp) :: rho, temp, xs(4), ni_sum, ne_sum
      character(len=32) :: cid

      call initialise_collision_integrals()
      can1 = [1.00794_dp, 4.002602_dp, 12.011_dp, 15.9994_dp, me / amu]
      kzn1 = [1.0_dp, 2.0_dp, 6.0_dp, 8.0_dp, -1.0_dp]

      do icase = 1, 4
         select case (icase)
         case (1)
            cid = 'solar_core_low_psi'
            rho = 1.5e2_dp; temp = 1.5e7_dp
            xs = [0.40_dp, 0.58_dp, 0.01_dp, 0.01_dp]
         case (2)
            cid = 'solar_env_neg_psi'
            rho = 2.0e-1_dp; temp = 2.0e6_dp
            xs = [0.70_dp, 0.28_dp, 0.01_dp, 0.01_dp]
         case (3)
            cid = 'he_shell_mod_psi'
            rho = 1.0e3_dp; temp = 1.0e7_dp
            xs = [0.01_dp, 0.79_dp, 0.15_dp, 0.05_dp]
         case default
            cid = 'co_wd_high_psi'
            rho = 2.0e5_dp; temp = 1.5e7_dp
            xs = [1.0e-4_dp, 0.0999_dp, 0.45_dp, 0.45_dp]
         end select

         ne_sum = 0.0_dp
         do i = 1, 4
            na1(i) = rho * xs(i) / (can1(i) * amu)
            ne_sum = ne_sum + kzn1(i) * na1(i)
         end do
         na1(5) = ne_sum

         call paquette_coefficients(rho, temp, nn, can1, kzn1, na1, &
            ddiff, kdiff, zdiff, zdiff1, zdiff2, ath)

         do i = 1, nn
            do j = 1, nn
               call emit_row(unit_no, 'paquette', cid, 'Ddiff', i, j, ddiff(i, j))
               call emit_row(unit_no, 'paquette', cid, 'Kdiff', i, j, kdiff(i, j))
               call emit_row(unit_no, 'paquette', cid, 'Zdiff', i, j, zdiff(i, j))
               call emit_row(unit_no, 'paquette', cid, 'Zdiff1', i, j, zdiff1(i, j))
               call emit_row(unit_no, 'paquette', cid, 'Zdiff2', i, j, zdiff2(i, j))
               call emit_row(unit_no, 'paquette', cid, 'Ath', i, j, ath(i, j))
            end do
         end do
      end do
      call free_collision_integrals()
   end subroutine emit_paquette_oracles

   ! --------------------------------------------------------------------------
   ! Section 2: get_matrix_coeffs across all 8 transport / Burgers regimes
   ! --------------------------------------------------------------------------
   subroutine emit_matrix_coeffs_oracles(unit_no, s)
      integer, intent(in) :: unit_no
      type(star_info), pointer, intent(inout) :: s
      integer, parameter :: nz = 5, nc = 4, m = 5
      integer :: icase, k, i, j, ierr
      logical :: pure_coulomb
      character(len=32) :: cid
      real(dp) :: dt, v_adv_max, tiny_c, min_t_rad, max_t_rad
      real(dp) :: diff_fac(nc), a_vec(m)
      real(dp) :: rho_face(nz), t_face(nz), fpr2r(nz), xm_face(nz), cell_dm(nz), dm_bar(nz)
      real(dp) :: dlnp_dr(nz), dlnt_dr(nz), dlnrho_dr(nz), r_face(nz), r_mid(nz)
      real(dp) :: gam_t_lim(nz), alfa_face(nz)
      real(dp) :: z_mat(m, nz), x_init(m, nz), x_work(m, nz), rad_acc(nc, nz)
      real(dp) :: log10_grad(nc, nz), g_rad(nc, nz)
      real(dp) :: ad_face(nz), e_ap(nz), e_at(nz), e_ar(nz), e_fld(nz)
      real(dp) :: g_ap(nz), g_at(nz), g_ar(nz), g_fld(nz)
      real(dp) :: x_face(m, nz), c_mat(m, nz), c_div_x(m, nz), c_div_x_f(m, nz)
      real(dp) :: v_adv(nc, nz), v_tot(nc, nz), vlnp(nc, nz), vlnt(nc, nz), vrad(nc, nz)
      real(dp) :: gt_face(nc, nz), d_self(nc, nz), e_ax(m, nz), g_ax(m, nz)
      real(dp) :: sig_face(nc, nc, nz), sigma_lnc(nc, nc, nz)

      call initialise_collision_integrals()
      s%nz = nz
      if (associated(s%eta)) deallocate(s%eta)
      if (associated(s%gam)) deallocate(s%gam)
      if (associated(s%grav)) deallocate(s%grav)
      if (associated(s%extra_diffusion_factor)) deallocate(s%extra_diffusion_factor)
      allocate(s%eta(nz), s%gam(nz), s%grav(nz), s%extra_diffusion_factor(nc, nz))

      ! Reset relevant star controls to exact MESA 26.4.1 defaults
      s%cgs_thermal_diffusion_eta_full_on = 10.0_dp
      s%cgs_thermal_diffusion_eta_full_off = 40.0_dp
      s%diffusion_AD_dm_full_on = -1.0_dp
      s%diffusion_AD_dm_full_off = -1.0_dp
      s%diffusion_AD_boost_factor = 1.0_dp
      s%diffusion_Vlimit_dm_full_on = -1.0_dp
      s%diffusion_Vlimit_dm_full_off = -1.0_dp
      s%diffusion_Vlimit = 0.0_dp
      s%diffusion_SIG_factor = 1.0_dp
      s%diffusion_GT_factor = 1.0_dp
      s%use_other_diffusion_factor = .false.
      s%use_other_diffusion_coefficients = .false.
      s%extra_diffusion_factor(:, :) = 1.0_dp

      dt = 1.0e11_dp
      s%dt = dt
      v_adv_max = 1.0e99_dp
      tiny_c = 1.0e-50_dp
      min_t_rad = 1.0e99_dp
      max_t_rad = -1.0_dp
      diff_fac = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]
      a_vec = [1.00794_dp, 4.002602_dp, 12.011_dp, 15.9994_dp, me / amu]

      rad_acc = 0.0_dp; log10_grad = 0.0_dp; g_rad = 0.0_dp
      gam_t_lim = 1.0_dp
      alfa_face = 0.5_dp
      do k = 1, nz
         r_face(k) = 5.0e10_dp * (1.0_dp - 0.08_dp * real(k - 1, dp))
         r_mid(k) = 5.0e10_dp * (1.0_dp - 0.08_dp * (real(k, dp) - 0.5_dp))
         cell_dm(k) = 1.0e30_dp
         dm_bar(k) = 1.0e30_dp
         xm_face(k) = real(k - 1, dp) * 1.0e30_dp
         s%grav(k) = 2.7e4_dp * (1.0_dp + 0.1_dp * real(k - 1, dp))
         dlnp_dr(k) = -2.0e-10_dp * (1.0_dp + 0.05_dp * real(k - 1, dp))
         dlnt_dr(k) = -6.0e-11_dp * (1.0_dp + 0.04_dp * real(k - 1, dp))
         dlnrho_dr(k) = -1.4e-10_dp * (1.0_dp + 0.05_dp * real(k - 1, dp))
         z_mat(1, k) = 1.0_dp
         z_mat(2, k) = 2.0_dp
         z_mat(3, k) = 6.0_dp
         z_mat(4, k) = 8.0_dp
         z_mat(5, k) = -1.0_dp
         x_init(1, k) = 0.70_dp - 0.03_dp * real(k - 1, dp)
         x_init(2, k) = 0.26_dp + 0.025_dp * real(k - 1, dp)
         x_init(3, k) = 0.02_dp + 0.003_dp * real(k - 1, dp)
         x_init(4, k) = 0.02_dp + 0.002_dp * real(k - 1, dp)
         x_init(5, k) = 0.0_dp
      end do

      do icase = 1, 8
         s%diffusion_use_paquette = .false.
         s%diffusion_use_cgs_solver = .false.
         s%diffusion_use_caplan = .false.
         pure_coulomb = .false.
         rho_face(:) = 10.0_dp
         t_face(:) = 5.0e6_dp
         s%eta(:) = -1.0_dp
         s%gam(:) = 0.05_dp

         select case (icase)
         case (1)
            cid = 'pure_coulomb_thoul'
            pure_coulomb = .true.
         case (2)
            cid = 'paquette_thoul'
            s%diffusion_use_paquette = .true.
         case (3)
            cid = 'sm_thoul'
         case (4)
            cid = 'sm_cgs_low_eta'
            s%diffusion_use_cgs_solver = .true.
            s%eta(:) = 2.0_dp
         case (5)
            cid = 'sm_cgs_blend_eta'
            s%diffusion_use_cgs_solver = .true.
            s%eta(:) = 25.0_dp
         case (6)
            cid = 'sm_cgs_high_eta'
            s%diffusion_use_cgs_solver = .true.
            s%eta(:) = 60.0_dp
            rho_face(:) = 1.0e5_dp; t_face(:) = 1.0e7_dp; s%gam(:) = 1.5_dp
         case (7)
            cid = 'caplan_blend_cgs'
            s%diffusion_use_cgs_solver = .true.
            s%diffusion_use_caplan = .true.
            s%eta(:) = 60.0_dp
            rho_face(:) = 5.0e5_dp; t_face(:) = 8.0e6_dp; s%gam(:) = 6.0_dp
         case default
            cid = 'caplan_pure_cgs'
            s%diffusion_use_cgs_solver = .true.
            s%diffusion_use_caplan = .true.
            s%eta(:) = 80.0_dp
            rho_face(:) = 2.0e6_dp; t_face(:) = 6.0e6_dp; s%gam(:) = 50.0_dp
         end select

         do k = 1, nz
            fpr2r(k) = pi4 * r_face(k) * r_face(k) * rho_face(k)
         end do
         x_work = x_init

         call get_matrix_coeffs( &
            s, nz, nc, m, 1, nz, 1, 2, pure_coulomb, &
            dt, v_adv_max, tiny_c, diff_fac, &
            a_vec, x_work, z_mat, rho_face, t_face, fpr2r, &
            xm_face, cell_dm, dm_bar, dlnp_dr, dlnt_dr, dlnrho_dr, &
            r_face, r_mid, gam_t_lim, alfa_face, &
            rad_acc, log10_grad, g_rad, &
            min_t_rad, max_t_rad, &
            x_init, x_face, c_mat, c_div_x, c_div_x_f, &
            e_ap, e_at, e_ar, e_ax, e_fld, &
            g_ap, g_at, g_ar, g_ax, g_fld, &
            v_adv, v_tot, vlnp, vlnt, vrad, &
            gt_face, d_self, ad_face, sig_face, sigma_lnc, ierr)
         if (ierr /= 0) stop 'get_matrix_coeffs failed'

         ! Emit face k=3 (interior face) full vectors and matrices
         k = 3
         call emit_row(unit_no, 'matrix_coeffs', cid, 'AD_face', k, 0, ad_face(k))
         call emit_row(unit_no, 'matrix_coeffs', cid, 'E_field_face', k, 0, e_fld(k))
         call emit_row(unit_no, 'matrix_coeffs', cid, 'g_field_face', k, 0, g_fld(k))
         do i = 1, nc
            call emit_row(unit_no, 'matrix_coeffs', cid, 'v_advection', i, k, v_adv(i, k))
            call emit_row(unit_no, 'matrix_coeffs', cid, 'v_total', i, k, v_tot(i, k))
            call emit_row(unit_no, 'matrix_coeffs', cid, 'vlnP', i, k, vlnp(i, k))
            call emit_row(unit_no, 'matrix_coeffs', cid, 'vlnT', i, k, vlnt(i, k))
            call emit_row(unit_no, 'matrix_coeffs', cid, 'GT_face', i, k, gt_face(i, k))
            call emit_row(unit_no, 'matrix_coeffs', cid, 'D_self_face', i, k, d_self(i, k))
            do j = 1, nc
               call emit_row(unit_no, 'matrix_coeffs', cid, 'SIG_face', i, j, sig_face(i, j, k))
               call emit_row(unit_no, 'matrix_coeffs', cid, 'sigma_lnC', i, j, sigma_lnc(i, j, k))
            end do
         end do
      end do
      call free_collision_integrals()
   end subroutine emit_matrix_coeffs_oracles

   ! --------------------------------------------------------------------------
   ! Section 3: do_solve_diffusion (BE_solve Thoul, BE_solve CGS WD, 7 isolve solvers)
   ! --------------------------------------------------------------------------
   subroutine emit_solve_diffusion_oracles(unit_no, s)
      integer, intent(in) :: unit_no
      type(star_info), pointer, intent(inout) :: s
      integer, parameter :: nz = 12, species = 4, nc = 4, m = 5
      integer :: icase, k, j, ierr, nzlo, nzhi, steps_used, tot_iters, tot_retries
      integer :: cls(species), class_cid(nc), chem_id(species), net_iso(2000)
      real(dp) :: abar(nz), ye(nz), free_e(nz), gam_arr(nz), dm_bar(nz), dm_in(nz)
      real(dp) :: t_arr(nz), lnt(nz), rho_arr(nz), lnd(nz), r_mid(nz)
      real(dp) :: dlnp_dm(nz), dlnt_dm(nz), dlnrho_dm(nz)
      real(dp) :: l_face(nz), r_face(nz), dlnp_dm_f(nz), dlnt_dm_f(nz), dlnrho_dm_f(nz)
      real(dp) :: typ_q(nc, nz), xa(species, nz), x_init(nc, nz), x_final(nc, nz)
      real(dp) :: d_self(nc, nz), v_adv(nc, nz), v_tot(nc, nz), vlnp(nc, nz), vlnt(nc, nz)
      real(dp) :: vrad(nc, nz), g_rad_f(nc, nz), e_fld(nz), g_fld(nz), diff_fac(nc)
      real(dp) :: total_time, m_tot, r_out, dr
      character(len=32) :: cid

      call initialise_collision_integrals()
      s%nz = nz; s%species = species; s%model_number = 10
      if (associated(s%eta)) deallocate(s%eta)
      if (associated(s%gam)) deallocate(s%gam)
      if (associated(s%grav)) deallocate(s%grav)
      if (associated(s%q)) deallocate(s%q)
      if (associated(s%rho)) deallocate(s%rho)
      if (associated(s%phase)) deallocate(s%phase)
      if (associated(s%extra_diffusion_factor)) deallocate(s%extra_diffusion_factor)
      allocate(s%eta(nz), s%gam(nz), s%grav(nz), s%q(nz), s%rho(nz), s%phase(nz))
      allocate(s%extra_diffusion_factor(nc, nz))

      s%extra_diffusion_factor(:, :) = 1.0_dp
      s%cgs_thermal_diffusion_eta_full_on = 10.0_dp
      s%cgs_thermal_diffusion_eta_full_off = 40.0_dp
      s%diffusion_AD_dm_full_on = -1.0_dp
      s%diffusion_AD_dm_full_off = -1.0_dp
      s%diffusion_AD_boost_factor = 1.0_dp
      s%diffusion_Vlimit_dm_full_on = -1.0_dp
      s%diffusion_Vlimit_dm_full_off = -1.0_dp
      s%diffusion_Vlimit = 0.0_dp
      s%diffusion_SIG_factor = 1.0_dp
      s%diffusion_GT_factor = 1.0_dp
      s%use_other_diffusion_factor = .false.
      s%use_other_diffusion_coefficients = .false.
      s%diffusion_tol_correction_max = 3.0e-1_dp
      s%diffusion_tol_correction_norm = 3.0e-3_dp
      s%diffusion_min_X_hard_limit = -1.0e-3_dp
      s%diffusion_X_total_rtol = 1.0e-6_dp
      s%diffusion_X_total_atol = 1.0e-6_dp
      s%diffusion_upwind_abs_v_limit = 1.0e99_dp
      s%diffusion_max_retries_per_substep = 20
      s%diffusion_min_num_substeps = 1
      s%diffusion_rtol_for_isolve = 1.0e-4_dp
      s%diffusion_atol_for_isolve = 1.0e-5_dp
      s%diffusion_maxsteps_for_isolve = 1000
      s%diffusion_isolve_solver = 'ros2_solver'
      s%show_diffusion_substep_info = .false.
      s%show_diffusion_timing = .false.
      s%report_ierr = .false.
      s%phase(:) = 0.0_dp

      chem_id = [ih1, ihe4, ic12, ife56]
      class_cid = chem_id
      cls = [1, 2, 3, 4]
      net_iso = 0
      do j = 1, species
         net_iso(chem_id(j)) = j
      end do
      diff_fac = [1.0_dp, 1.0_dp, 1.0_dp, 1.0_dp]

      do icase = 1, 9
         select case (icase)
         case (1)
            cid = 'be_thoul_solar'
            s%diffusion_use_isolve = .false.
            s%diffusion_use_cgs_solver = .false.
            s%diffusion_use_paquette = .false.
            s%diffusion_use_caplan = .false.
            total_time = 1.0e14_dp
            s%eta(:) = -1.0_dp; gam_arr(:) = 0.05_dp; s%gam(:) = 0.05_dp
         case (2, 4:9)
            ! Rosenbrock isolve on the solar profile; only the solver differs
            select case (icase)
            case (2)
               cid = 'isolve_ros2_solar'
               s%diffusion_isolve_solver = 'ros2_solver'
            case (4)
               cid = 'isolve_rose2_solar'
               s%diffusion_isolve_solver = 'rose2_solver'
            case (5)
               cid = 'isolve_ros3p_solar'
               s%diffusion_isolve_solver = 'ros3p_solver'
            case (6)
               cid = 'isolve_ros3pl_solar'
               s%diffusion_isolve_solver = 'ros3pl_solver'
            case (7)
               cid = 'isolve_rodas3_solar'
               s%diffusion_isolve_solver = 'rodas3_solver'
            case (8)
               cid = 'isolve_rodas4_solar'
               s%diffusion_isolve_solver = 'rodas4_solver'
            case default
               cid = 'isolve_rodasp_solar'
               s%diffusion_isolve_solver = 'rodasp_solver'
            end select
            s%diffusion_use_isolve = .true.
            s%diffusion_use_cgs_solver = .false.
            s%diffusion_use_paquette = .false.
            s%diffusion_use_caplan = .false.
            total_time = 1.0e14_dp
            s%eta(:) = -1.0_dp; gam_arr(:) = 0.05_dp; s%gam(:) = 0.05_dp
         case default
            cid = 'be_cgs_caplan_wd'
            s%diffusion_use_isolve = .false.
            s%diffusion_use_cgs_solver = .true.
            s%diffusion_use_paquette = .false.
            s%diffusion_use_caplan = .true.
            total_time = 1.0e12_dp
            s%eta(:) = 50.0_dp; gam_arr(:) = 20.0_dp; s%gam(:) = 20.0_dp
         end select
         s%dt = total_time

         m_tot = 1.0_dp * msun
         r_out = 6.0e10_dp
         if (icase == 3) then
            m_tot = 0.6_dp * msun
            r_out = 8.0e8_dp
         end if
         dr = 0.5_dp * r_out / real(nz, dp)
         do k = 1, nz
            r_face(k) = r_out - real(k - 1, dp) * dr
            r_mid(k) = r_out - (real(k, dp) - 0.5_dp) * dr
            dm_in(k) = (0.05_dp * m_tot) / real(nz, dp)
            dm_bar(k) = dm_in(k)
            s%q(k) = 1.0_dp - real(k - 1, dp) * (0.05_dp / real(nz, dp))
            if (icase == 3) then
               rho_arr(k) = 1.0e5_dp * (1.0_dp + 0.15_dp * real(k - 1, dp))
               t_arr(k) = 1.0e7_dp * (1.0_dp + 0.03_dp * real(k - 1, dp))
               s%grav(k) = 1.0e8_dp
               xa(1, k) = 1.0e-5_dp
               xa(2, k) = 0.20_dp - 0.015_dp * real(k - 1, dp)
               xa(3, k) = 0.78_dp + 0.015_dp * real(k - 1, dp)
               xa(4, k) = 0.02_dp - 1.0e-5_dp
               abar(k) = 10.0_dp; ye(k) = 0.5_dp; free_e(k) = 0.5_dp
            else
               rho_arr(k) = 1.0_dp * (1.0_dp + 0.35_dp * real(k - 1, dp))
               t_arr(k) = 3.0e6_dp * (1.0_dp + 0.12_dp * real(k - 1, dp))
               s%grav(k) = 2.7e4_dp
               xa(1, k) = 0.72_dp - 0.012_dp * real(k - 1, dp)
               xa(2, k) = 0.26_dp + 0.010_dp * real(k - 1, dp)
               xa(3, k) = 0.015_dp + 0.0015_dp * real(k - 1, dp)
               xa(4, k) = 0.005_dp + 0.0005_dp * real(k - 1, dp)
               abar(k) = 1.30_dp; ye(k) = 0.85_dp; free_e(k) = 0.85_dp
            end if
            s%rho(k) = rho_arr(k)
            lnd(k) = log(rho_arr(k))
            lnt(k) = log(t_arr(k))
            l_face(k) = lsun
            dlnp_dm(k) = -2.0e-10_dp / (pi4 * r_mid(k) * r_mid(k) * rho_arr(k))
            dlnt_dm(k) = -6.0e-11_dp / (pi4 * r_mid(k) * r_mid(k) * rho_arr(k))
            dlnrho_dm(k) = -1.4e-10_dp / (pi4 * r_mid(k) * r_mid(k) * rho_arr(k))
            if (icase == 3) then
               dlnp_dm(k) = -1.5e-8_dp / (pi4 * r_mid(k) * r_mid(k) * rho_arr(k))
               dlnt_dm(k) = -1.0e-9_dp / (pi4 * r_mid(k) * r_mid(k) * rho_arr(k))
               dlnrho_dm(k) = -1.2e-8_dp / (pi4 * r_mid(k) * r_mid(k) * rho_arr(k))
            end if
            dlnp_dm_f(k) = dlnp_dm(k)
            dlnt_dm_f(k) = dlnt_dm(k)
            dlnrho_dm_f(k) = dlnrho_dm(k)
            typ_q(1, k) = 1.0_dp
            typ_q(2, k) = 2.0_dp
            typ_q(3, k) = 6.0_dp
            typ_q(4, k) = 26.0_dp
         end do

         nzlo = 1; nzhi = nz
         ! Zero intent(out) buffers so untouched boundary slot k=1 (nzlo) is
         ! deterministic 0.0 rather than uninitialized stack memory.
         x_init = 0.0_dp; x_final = 0.0_dp
         d_self = 0.0_dp; v_adv = 0.0_dp; v_tot = 0.0_dp
         vlnp = 0.0_dp; vlnt = 0.0_dp; vrad = 0.0_dp; g_rad_f = 0.0_dp
         e_fld = 0.0_dp; g_fld = 0.0_dp
         call do_solve_diffusion( &
            s, nz, species, nc, m, cls, class_cid, net_iso, chem_id, &
            abar, ye, free_e, dm_bar, dm_in, &
            t_arr, lnt, rho_arr, lnd, r_mid, dlnp_dm, dlnt_dm, dlnrho_dm, &
            l_face, r_face, dlnp_dm_f, dlnt_dm_f, dlnrho_dm_f, &
            .false., total_time, 1.0_dp, &
            500, 1000, 20, &
            .false., typ_q, 0, &
            1.0e99_dp, -1.0_dp, &
            1, 30, &
            .false., '', '', &
            1.0e99_dp, 0.0_dp, &
            gam_arr, 40.0_dp, 50.0_dp, &
            -1.0_dp, -1.0_dp, diff_fac, &
            xa, steps_used, tot_iters, tot_retries, &
            nzlo, nzhi, x_init, x_final, &
            d_self, v_adv, v_tot, &
            vlnp, vlnt, vrad, g_rad_f, &
            e_fld, g_fld, &
            ierr)
         if (ierr /= 0) then
            ! A failing solver is recorded in the table rather than hidden
            write(*, '(A,A,A,I0)') 'do_solve_diffusion failed for ', trim(cid), ': ierr = ', ierr
            if (icase <= 3) stop 'do_solve_diffusion failed'
            call emit_row(unit_no, 'solve_diff', cid, 'ierr', 0, 0, real(ierr, dp))
            cycle
         end if

         call emit_row(unit_no, 'solve_diff', cid, 'steps_used', 0, 0, real(steps_used, dp))
         call emit_row(unit_no, 'solve_diff', cid, 'total_iters', 0, 0, real(tot_iters, dp))
         do k = 1, nz
            call emit_row(unit_no, 'solve_diff', cid, 'E_field', k, 0, e_fld(k))
            call emit_row(unit_no, 'solve_diff', cid, 'g_field', k, 0, g_fld(k))
            do j = 1, nc
               call emit_row(unit_no, 'solve_diff', cid, 'X_final', j, k, x_final(j, k))
               call emit_row(unit_no, 'solve_diff', cid, 'v_total', j, k, v_tot(j, k))
            end do
         end do
      end do
      call free_collision_integrals()
   end subroutine emit_solve_diffusion_oracles

   ! --------------------------------------------------------------------------
   ! Section 4: phase_separation::do_phase_separation (CO, ONe, CO with Brunt B)
   ! --------------------------------------------------------------------------
   subroutine emit_phase_sep_oracles(unit_no, mdir)
      integer, intent(in) :: unit_no
      character(len=*), intent(in) :: mdir
      integer :: id_ps, ierr, k, icase, nz_ps, ic12_i, io16_i, ine20_i
      type(star_info), pointer :: sp
      real(dp) :: dt_ps, r3_run, dm_k, rho_k
      character(len=16) :: cid

      do icase = 1, 3
         call alloc_star(id_ps, ierr)
         if (ierr /= 0) stop 'alloc_star ps failed'
         call star_ptr(id_ps, sp, ierr)
         if (ierr /= 0) stop 'star_ptr ps failed'
         call set_default_star_job_controls()
         call store_star_job_controls(sp, ierr)
         if (ierr /= 0) stop 'store_star_job_controls ps failed'
         sp%job%mesa_dir = trim(mdir)
         sp%inlist_fname = trim(mdir) // '/star/test/inlist'
         call starlib_init(sp, ierr)
         if (ierr /= 0) stop 'starlib_init ps failed'
         call star_set_kap_and_eos_handles(id_ps, ierr)
         if (ierr /= 0) stop 'star_set_kap_and_eos_handles ps failed'
         call star_set_colors_handles(id_ps, ierr)
         if (ierr /= 0) stop 'star_set_colors_handles ps failed'
         call star_setup(id_ps, sp%inlist_fname, ierr)
         if (ierr /= 0) stop 'star_setup ps failed'
         sp%initial_mass = 1.0_dp
         sp%initial_z = 0.02_dp
         sp%kap_rq%Zbase = 0.02_dp
         sp%kap_rq%use_Type2_opacities = .true.
         sp%eos_rq%use_Skye = .true.
         sp%eos_rq%use_PC = .false.
         sp%eos_rq%Skye_solid_mixing_rule = 'Ogata'
         sp%report_ierr = .true.
         sp%okay_to_remesh = .false.
         sp%do_history_file = .false.
         call star_load_zams(id_ps, ierr)
         if (ierr /= 0) stop 'star_load_zams ps failed'

         ! Construct an 8-zone WD profile using the active basic.net (which has c12, o16, ne20)
         nz_ps = 8
         sp%nz = nz_ps
         call reallocate_star_info_arrays(sp, ierr)
         if (ierr /= 0) stop 'reallocate ps failed'
         sp%mstar = 0.6_dp * msun
         sp%xmstar = sp%mstar
         sp%M_center = 0.0_dp
         sp%R_center = 0.0_dp
         sp%L_center = 0.0_dp
         sp%calculate_Brunt_B = .true.
         sp%use_Ledoux_criterion = .true.
         sp%do_phase_separation = .true.
         sp%do_phase_separation_heating = .true.
         sp%phase_separation_mixing_use_brunt = .false.
         sp%crystal_core_boundary_mass = 0.0_dp
         sp%min_dq = 1.0e-14_dp

         ic12_i = sp%net_iso(ic12)
         io16_i = sp%net_iso(io16)
         ine20_i = sp%net_iso(ine20)
         if (ic12_i <= 0 .or. io16_i <= 0 .or. ine20_i <= 0) then
            stop 'basic.net missing c12/o16/ne20'
         end if

         sp%xa(:, 1:nz_ps) = 0.0_dp
         do k = 1, nz_ps
            sp%dq(k) = 1.0_dp / real(nz_ps, dp)
         end do
         sp%q(1) = 1.0_dp
         do k = 2, nz_ps
            sp%q(k) = sp%q(k - 1) - sp%dq(k - 1)
         end do
         do k = 1, nz_ps
            sp%dm(k) = sp%mstar * sp%dq(k)
            sp%m(k) = sp%mstar * sp%q(k)
         end do

         if (icase == 1 .or. icase == 3) then
            if (icase == 1) then
               cid = 'blouin_co'
            else
               ! Same CO profile with Brunt B in the mix_outward stability test
               cid = 'blouin_co_brunt'
               sp%phase_separation_mixing_use_brunt = .true.
            end if
            sp%phase_separation_option = 'CO'
            ! Outer zones k=1,2 have stably lower mu (higher C12) so mix_outward
            ! stabilizes before reaching k=1; inner zones k=5..8 are solid (phase > 0.9)
            do k = 1, nz_ps
               sp%xh(sp%i_lnd, k) = log(1.5e6_dp * (1.0_dp + 1.2_dp * real(k - 1, dp)))
               sp%xh(sp%i_lnT, k) = log(5.5e6_dp - 4.0e5_dp * real(k - 1, dp))
               if (k <= 2) then
                  sp%xa(ic12_i, k) = 0.85_dp
                  sp%xa(io16_i, k) = 0.15_dp
               else
                  sp%xa(ic12_i, k) = 0.40_dp
                  sp%xa(io16_i, k) = 0.60_dp
               end if
            end do
         else
            cid = 'blouin_one'
            sp%phase_separation_option = 'ONe'
            ! Outer zones k=1,2 have stably lower mu (higher O16); inner zones solid
            do k = 1, nz_ps
               sp%xh(sp%i_lnd, k) = log(3.0e6_dp * (1.0_dp + 1.2_dp * real(k - 1, dp)))
               sp%xh(sp%i_lnT, k) = log(9.5e6_dp - 6.0e5_dp * real(k - 1, dp))
               if (k <= 2) then
                  sp%xa(io16_i, k) = 0.90_dp
                  sp%xa(ine20_i, k) = 0.10_dp
               else
                  sp%xa(io16_i, k) = 0.55_dp
                  sp%xa(ine20_i, k) = 0.45_dp
               end if
            end do
         end if

         r3_run = 0.0_dp
         do k = nz_ps, 1, -1
            dm_k = sp%dm(k)
            rho_k = exp(sp%xh(sp%i_lnd, k))
            r3_run = r3_run + (3.0_dp * dm_k) / (pi4 * rho_k)
            sp%xh(sp%i_lnR, k) = log(r3_run**(1.0_dp / 3.0_dp))
            sp%xh(sp%i_lum, k) = 1.0e-3_dp * lsun * sp%q(k)
         end do
         sp%cgrav(1:nz_ps) = standard_cgrav
         sp%xa_start(:, 1:nz_ps) = sp%xa(:, 1:nz_ps)

         dt_ps = 1.0e13_dp
         sp%dt = dt_ps
         call set_vars(sp, dt_ps, ierr)
         if (ierr /= 0) stop 'set_vars ps failed'

         ! Emit pre-phase-sep eos phase & lnPgas so Rust can check identical boundary
         do k = 1, nz_ps
            call emit_row(unit_no, 'phase_sep', cid, 'phase_init', k, 0, sp%phase(k))
            call emit_row(unit_no, 'phase_sep', cid, 'lnPgas_init', k, 0, sp%lnPgas(k))
         end do

         call do_phase_separation(sp, dt_ps, ierr)
         if (ierr /= 0) stop 'do_phase_separation failed'

         call emit_row(unit_no, 'phase_sep', cid, 'crystal_core_boundary_mass', 0, 0, sp%crystal_core_boundary_mass)
         call emit_row(unit_no, 'phase_sep', cid, 'phase_sep_mixing_mass', 0, 0, sp%phase_sep_mixing_mass)
         do k = 1, nz_ps
            call emit_row(unit_no, 'phase_sep', cid, 'X_c12', k, 0, sp%xa(ic12_i, k))
            call emit_row(unit_no, 'phase_sep', cid, 'X_o16', k, 0, sp%xa(io16_i, k))
            call emit_row(unit_no, 'phase_sep', cid, 'X_ne20', k, 0, sp%xa(ine20_i, k))
            call emit_row(unit_no, 'phase_sep', cid, 'eps_phase_sep', k, 0, sp%eps_phase_separation(k))
            call emit_row(unit_no, 'phase_sep', cid, 'lnd_final', k, 0, sp%lnd(k))
         end do
         do k = 2, nz_ps
            call emit_row(unit_no, 'phase_sep', cid, 'gradr_final', k, 0, sp%gradr(k))
            call emit_row(unit_no, 'phase_sep', cid, 'grada_face_final', k, 0, sp%grada_face(k))
         end do
         do k = 1, nz_ps
            call emit_row(unit_no, 'phase_sep', cid, 'lnPgas_final', k, 0, sp%lnPgas(k))
         end do

         call free_star(id_ps, ierr)
      end do
   end subroutine emit_phase_sep_oracles

   ! --------------------------------------------------------------------------
   ! Section 5: private kernels compiled from verbatim MESA 26.4.1 source text
   ! --------------------------------------------------------------------------
   subroutine emit_kernel_oracles(unit_no)
      integer, intent(in) :: unit_no
      real(dp), parameter :: gammas(3) = [5.0_dp, 20.0_dp, 100.0_dp]
      real(dp), parameter :: kappas(3) = [0.5_dp, 1.2_dp, 0.8_dp]
      real(dp), parameter :: x_ins(5) = [0.1_dp, 0.3_dp, 0.5_dp, 0.7_dp, 0.85_dp]
      real(dp) :: dstar
      integer :: i

      do i = 1, size(gammas)
         call get_Dstar_OCP(gammas(i), kappas(i), dstar)
         call emit_row(unit_no, 'kernels', 'dstar_ocp', 'Gamma', i, 0, gammas(i))
         call emit_row(unit_no, 'kernels', 'dstar_ocp', 'kappa', i, 0, kappas(i))
         call emit_row(unit_no, 'kernels', 'dstar_ocp', 'Dstar', i, 0, dstar)
      end do
      do i = 1, size(x_ins)
         call emit_row(unit_no, 'kernels', 'blouin_xo', 'X_in', i, 0, x_ins(i))
         call emit_row(unit_no, 'kernels', 'blouin_xo', 'dX', i, 0, blouin_delta_xo(x_ins(i)))
      end do
      do i = 1, size(x_ins)
         call emit_row(unit_no, 'kernels', 'blouin_xne', 'X_in', i, 0, x_ins(i))
         call emit_row(unit_no, 'kernels', 'blouin_xne', 'dX', i, 0, blouin_delta_xne(x_ins(i)))
      end do
   end subroutine emit_kernel_oracles

end program gen_diffusion_oracles
