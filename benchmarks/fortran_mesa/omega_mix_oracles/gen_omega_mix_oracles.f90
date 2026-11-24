! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex + decimal reference vectors against compiled MESA 26.4.1
! (`libstar.a` + `CRMATH`) for `hydro_rotation` and `solve_omega_mix` (#158).
!
! Sections emitted:
!   1. eval_i_rot_fp_ft : eval_i_rot / set1_i_rot / set_rotation_info (eval_fp_ft)
!                         across w_div_w_crit_roche in [0.0..0.99] + retrograde -0.60,
!                         for both simple_i_rot_flag = .false. and .true.
!   2. roche_factors    : hydro_rotation::a and hydro_rotation::c across w in [0.0..0.99]
!   3. w_div_w_roche    : w_div_w_roche_omega and w_div_w_roche_jrot across slow,
!                         intermediate, near-critical, supercritical, and retrograde
!                         inputs for both w_div_wc_flag = .false. and .true.
!   4. rotation_sigmas  : get_rotation_sigmas (am_sig_j, am_sig_omega) in normal
!                         and am_sig_term_limit-capped regimes
!   5. solve_omega_mix  : do_solve_omega_mix (real(qp)) across 11 cases covering
!                         diffusive omega shear, pure-j diffusion, mixed omega+j,
!                         stiff 1e13 am_sig_term_limit, explicit extra_jdot,
!                         explicit extra_omegadot, nonlinear 3-point implicit
!                         torque with all 6 derivatives, adaptive substepping
!                         (steps_used > min_steps), max_iters_per_substep retry,
!                         conservation-error retry triggered vs passed, and
!                         simple_i_rot_flag = .true.

module omega_mix_oracle_hooks
   use const_def, only: dp
   use star_def, only: star_info, star_ptr
   implicit none
   private
   public :: explicit_jdot_torque, explicit_omegadot_torque, &
      implicit_tridiag_torque, implicit_nonconverging_torque, &
      implicit_hook_calls

   integer :: implicit_hook_calls = 0

contains

   subroutine explicit_jdot_torque(id, ierr)
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type(star_info), pointer :: s
      integer :: k
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      do k = 1, s%nz
         s%extra_jdot(k) = -2.0e6_dp * (1.0_dp - 0.025_dp * real(k - 1, dp))
         s%extra_omegadot(k) = 0.0_dp
      end do
   end subroutine explicit_jdot_torque

   subroutine explicit_omegadot_torque(id, ierr)
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type(star_info), pointer :: s
      integer :: k
      ierr = 0
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      do k = 1, s%nz
         s%extra_jdot(k) = 0.0_dp
         s%extra_omegadot(k) = -1.5e-16_dp * (1.0_dp + 0.08_dp * real(k - 1, dp))
      end do
   end subroutine explicit_omegadot_torque

   subroutine implicit_tridiag_torque(id, ierr)
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type(star_info), pointer :: s
      integer :: k, nz
      real(dp), parameter :: om_eq = 3.0e-5_dp
      real(dp), parameter :: c_j_drag = 2.0e-11_dp
      real(dp), parameter :: c_j_cpl = 5.0e-12_dp
      real(dp), parameter :: c_om_drag = 1.5e-11_dp
      real(dp), parameter :: c_om_cpl = 4.0e-12_dp
      real(dp) :: dw, irot_k

      ierr = 0
      implicit_hook_calls = implicit_hook_calls + 1
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      nz = s%nz

      do k = 1, nz
         s%extra_jdot(k) = 0.0_dp
         s%extra_omegadot(k) = 0.0_dp
         s%d_extra_jdot_domega_m1(k) = 0.0_dp
         s%d_extra_jdot_domega_00(k) = 0.0_dp
         s%d_extra_jdot_domega_p1(k) = 0.0_dp
         s%d_extra_omegadot_domega_m1(k) = 0.0_dp
         s%d_extra_omegadot_domega_00(k) = 0.0_dp
         s%d_extra_omegadot_domega_p1(k) = 0.0_dp
      end do

      do k = 2, nz
         dw = s%omega(k) - om_eq
         irot_k = s%i_rot(k)%val

         s%extra_jdot(k) = -c_j_drag * irot_k * (dw + 1.0e8_dp * dw * dw * dw)
         s%d_extra_jdot_domega_00(k) = -c_j_drag * irot_k * (1.0_dp + 3.0e8_dp * dw * dw)

         s%extra_jdot(k) = s%extra_jdot(k) + c_j_cpl * irot_k * (s%omega(k - 1) - s%omega(k))
         s%d_extra_jdot_domega_m1(k) = c_j_cpl * irot_k
         s%d_extra_jdot_domega_00(k) = s%d_extra_jdot_domega_00(k) - c_j_cpl * irot_k
         if (k < nz) then
            s%extra_jdot(k) = s%extra_jdot(k) + c_j_cpl * irot_k * (s%omega(k + 1) - s%omega(k))
            s%d_extra_jdot_domega_p1(k) = c_j_cpl * irot_k
            s%d_extra_jdot_domega_00(k) = s%d_extra_jdot_domega_00(k) - c_j_cpl * irot_k
         end if

         s%extra_omegadot(k) = -c_om_drag * (dw + 2.0e4_dp * dw * dw)
         s%d_extra_omegadot_domega_00(k) = -c_om_drag * (1.0_dp + 4.0e4_dp * dw)

         s%extra_omegadot(k) = s%extra_omegadot(k) + c_om_cpl * (s%omega(k - 1) - s%omega(k))
         s%d_extra_omegadot_domega_m1(k) = c_om_cpl
         s%d_extra_omegadot_domega_00(k) = s%d_extra_omegadot_domega_00(k) - c_om_cpl
         if (k < nz) then
            s%extra_omegadot(k) = s%extra_omegadot(k) + c_om_cpl * (s%omega(k + 1) - s%omega(k))
            s%d_extra_omegadot_domega_p1(k) = c_om_cpl
            s%d_extra_omegadot_domega_00(k) = s%d_extra_omegadot_domega_00(k) - c_om_cpl
         end if
      end do
   end subroutine implicit_tridiag_torque

   subroutine implicit_nonconverging_torque(id, ierr)
      ! Deliberately strongly nonlinear torque with zero Jacobian derivatives
      ! so Picard/Newton iterations fail to converge within max_iters_per_substep = 4
      ! and trigger do_solve_omega_mix == retry.
      integer, intent(in) :: id
      integer, intent(out) :: ierr
      type(star_info), pointer :: s
      integer :: k
      ierr = 0
      implicit_hook_calls = implicit_hook_calls + 1
      call star_ptr(id, s, ierr)
      if (ierr /= 0) return
      do k = 1, s%nz
         s%extra_jdot(k) = 0.0_dp
         s%extra_omegadot(k) = -1.0e-5_dp * (s%omega(k) - 1.0e-5_dp)
         s%d_extra_jdot_domega_m1(k) = 0.0_dp
         s%d_extra_jdot_domega_00(k) = 0.0_dp
         s%d_extra_jdot_domega_p1(k) = 0.0_dp
         s%d_extra_omegadot_domega_m1(k) = 0.0_dp
         s%d_extra_omegadot_domega_00(k) = 0.0_dp
         s%d_extra_omegadot_domega_p1(k) = 0.0_dp
      end do
   end subroutine implicit_nonconverging_torque

end module omega_mix_oracle_hooks

program gen_omega_mix_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use const_def, only: dp, pi, pi4, msun, rsun, standard_cgrav, secyer
   use math_lib, only: MATH_BACKEND, pow, pow2, pow4, pow6, log
   use auto_diff, only: auto_diff_real_star_order1
   use star_def, only: star_info, star_ptr, keep_going, retry, terminate, &
      i_lnR_00, i_w_div_wc_00
   use star_lib, only: alloc_star, free_star, starlib_init, &
      star_set_kap_and_eos_handles, star_set_colors_handles, star_setup, star_load_zams
   use star_job_ctrls_io, only: set_default_star_job_controls, store_star_job_controls
   use alloc, only: reallocate_star_info_arrays
   use star_utils, only: total_angular_momentum
   use hydro_rotation, only: log_term_power, roche_a => a, roche_c => c, &
      w_div_w_roche_omega, w_div_w_roche_jrot, eval_i_rot, set_i_rot, &
      set_omega, set_rotation_info, use_xh_to_update_i_rot_and_j_rot, &
      get_rotation_sigmas
   use solve_omega_mix, only: do_solve_omega_mix
   use omega_mix_oracle_hooks, only: explicit_jdot_torque, explicit_omegadot_torque, &
      implicit_tridiag_torque, implicit_nonconverging_torque, implicit_hook_calls
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver, fflags_str
   integer :: ierr, u, id
   type(star_info), pointer :: s

   call get_environment_variable('MESA_DIR', mesa_dir)
   if (len_trim(mesa_dir) == 0) stop 'MESA_DIR must be set'
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   call get_environment_variable('ORACLE_FFLAGS', fflags_str)
   if (len_trim(fflags_str) == 0) then
      fflags_str = '-march=x86-64 -mtune=generic -mfpmath=sse -O2 -fopenmp -ffp-contract=off -fprotect-parens -fno-sign-zero -fno-range-check -ffree-line-length-none'
   end if

   if (command_argument_count() >= 1) then
      call get_command_argument(1, out_path)
   else
      out_path = 'mesa_26_4_1_omega_mix_oracles.tsv'
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
   s%initial_mass = 1.0_dp
   s%initial_z = 0.02_dp
   s%kap_rq%Zbase = 0.02_dp
   s%okay_to_remesh = .false.
   s%do_history_file = .false.
   call star_load_zams(id, ierr)
   if (ierr /= 0) stop 'star_load_zams failed'

   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'

   write(u, '(A)') '# FlexMESA MESA 26.4.1 hydro_rotation and solve_omega_mix oracle vectors (#158)'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') 'section	case_id	field	i	j	raw_bits_hex	decimal'

   call emit_eval_i_rot_fp_ft_oracles(u, s)
   call emit_roche_factors_oracles(u)
   call emit_w_div_w_roche_oracles(u)
   call emit_rotation_sigmas_oracles(u, s)
   call emit_solve_omega_mix_oracles(u, s)

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
   ! Section 1: eval_i_rot, set1_i_rot, and set_rotation_info (eval_fp_ft)
   ! --------------------------------------------------------------------------
   subroutine emit_eval_i_rot_fp_ft_oracles(unit_no, s)
      integer, intent(in) :: unit_no
      type(star_info), pointer, intent(inout) :: s
      integer, parameter :: nz = 13
      real(dp), parameter :: w_grid(nz) = [ &
         0.0_dp, 0.05_dp, 0.15_dp, 0.30_dp, 0.50_dp, 0.60_dp, &
         0.75_dp, 0.85_dp, 0.89_dp, 0.90_dp, 0.95_dp, 0.99_dp, -0.60_dp ]
      integer :: icase, k, ierr
      character(len=24) :: cid
      type(auto_diff_real_star_order1) :: i_rot_direct

      s%nz = nz
      call reallocate_star_info_arrays(s, ierr)
      if (ierr /= 0) stop 'reallocate failed in emit_eval_i_rot_fp_ft_oracles'

      s%rotation_flag = .true.
      s%use_other_eval_i_rot = .false.
      s%use_other_eval_fp_ft = .false.
      s%binary_other_irot => null()
      s%binary_other_fp_ft => null()
      s%report_ierr = .false.

      do icase = 1, 2
         if (icase == 1) then
            cid = 'roche_paxton'
            s%simple_i_rot_flag = .false.
            s%w_div_wc_flag = .true.
         else
            cid = 'roche_simple'
            s%simple_i_rot_flag = .true.
            s%w_div_wc_flag = .false.
         end if

         do k = 1, nz
            s%r(k) = rsun * (0.25_dp + 0.125_dp * real(k - 1, dp))
            s%w_div_w_crit_roche(k) = w_grid(k)
            s%j_rot(k) = 1.0e18_dp
            s%m(k) = 4.0_dp * msun
            s%cgrav(k) = standard_cgrav
            s%rho(k) = 1.0_dp
            s%r_polar(k) = 0.0_dp
            s%r_equatorial(k) = 0.0_dp
         end do

         call set_rotation_info(s, .true., ierr)
         if (ierr /= 0) stop 'set_rotation_info failed'

         do k = 1, nz
            call eval_i_rot(s%id, s%r(k), s%w_div_w_crit_roche(k), i_rot_direct)
            if (transfer(i_rot_direct%val, 0_int64) /= transfer(s%i_rot(k)%val, 0_int64)) then
               stop 'eval_i_rot disagreed with set_rotation_info'
            end if

            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'r', k, 0, s%r(k))
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'w_div_wc', k, 0, s%w_div_w_crit_roche(k))
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'i_rot', k, 0, s%i_rot(k)%val)
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'd_i_rot_dlnR', k, 0, s%i_rot(k)%d1Array(i_lnR_00))
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'd_i_rot_dw_div_wc', k, 0, s%i_rot(k)%d1Array(i_w_div_wc_00))
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'fp_rot', k, 0, s%fp_rot(k)%val)
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'ft_rot', k, 0, s%ft_rot(k)%val)
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'd_fp_dw_div_wc', k, 0, s%fp_rot(k)%d1Array(i_w_div_wc_00))
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'd_ft_dw_div_wc', k, 0, s%ft_rot(k)%d1Array(i_w_div_wc_00))
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'r_polar', k, 0, s%r_polar(k))
            call emit_row(unit_no, 'eval_i_rot_fp_ft', cid, 'r_equatorial', k, 0, s%r_equatorial(k))
         end do
      end do
   end subroutine emit_eval_i_rot_fp_ft_oracles

   ! --------------------------------------------------------------------------
   ! Section 2: public hydro_rotation::a and hydro_rotation::c
   ! --------------------------------------------------------------------------
   subroutine emit_roche_factors_oracles(unit_no)
      integer, intent(in) :: unit_no
      integer, parameter :: npts = 12
      real(dp), parameter :: w_pos(npts) = [ &
         0.0_dp, 0.05_dp, 0.15_dp, 0.30_dp, 0.50_dp, 0.60_dp, &
         0.75_dp, 0.85_dp, 0.89_dp, 0.90_dp, 0.95_dp, 0.99_dp ]
      integer :: k
      real(dp) :: w, w2, w4, w6, w_log

      do k = 1, npts
         w = w_pos(k)
         w2 = pow2(w)
         w4 = pow4(w)
         w6 = pow6(w)
         w_log = log(1.0_dp - pow(w, log_term_power))
         call emit_row(unit_no, 'roche_factors', 'fabry2022', 'w', k, 0, w)
         call emit_row(unit_no, 'roche_factors', 'fabry2022', 'roche_A', k, 0, roche_a(w4, w6, w_log))
         call emit_row(unit_no, 'roche_factors', 'fabry2022', 'roche_C', k, 0, roche_c(w2, w4, w6, w_log))
      end do
   end subroutine emit_roche_factors_oracles

   ! --------------------------------------------------------------------------
   ! Section 3: w_div_w_roche_omega and w_div_w_roche_jrot
   ! --------------------------------------------------------------------------
   subroutine emit_w_div_w_roche_oracles(unit_no)
      integer, intent(in) :: unit_no
      integer, parameter :: npts = 10
      real(dp), parameter :: omegas(npts) = [ &
         0.0_dp, 1.0e-6_dp, 1.0e-5_dp, 5.0e-5_dp, 1.2e-4_dp, &
         1.55e-4_dp, 1.60e-4_dp, 1.0e-3_dp, -1.0e-4_dp, -1.0e-3_dp ]
      real(dp), parameter :: jrots(npts) = [ &
         0.0_dp, 1.0e16_dp, 1.0e17_dp, 5.0e17_dp, 1.5e18_dp, &
         2.5e18_dp, 4.0e18_dp, 1.0e20_dp, -1.5e18_dp, -1.0e20_dp ]
      real(dp) :: rpsi, m_phi, max_w, max_w2, w_om, w_jr
      integer :: icase, k
      logical :: w_flag
      character(len=24) :: cid

      rpsi = 2.5_dp * rsun
      m_phi = 4.0_dp * msun
      max_w = 0.89_dp
      max_w2 = 0.90_dp

      do icase = 1, 2
         if (icase == 1) then
            cid = 'hard_cap'
            w_flag = .false.
         else
            cid = 'sigmoid_cap'
            w_flag = .true.
         end if

         call emit_row(unit_no, 'w_div_w_roche', cid, 'rpsi', 0, 0, rpsi)
         call emit_row(unit_no, 'w_div_w_roche', cid, 'm_phi', 0, 0, m_phi)
         call emit_row(unit_no, 'w_div_w_roche', cid, 'cgrav', 0, 0, standard_cgrav)
         call emit_row(unit_no, 'w_div_w_roche', cid, 'max_w', 0, 0, max_w)
         call emit_row(unit_no, 'w_div_w_roche', cid, 'max_w2', 0, 0, max_w2)

         do k = 1, npts
            w_om = w_div_w_roche_omega(rpsi, m_phi, omegas(k), standard_cgrav, max_w, max_w2, w_flag)
            w_jr = w_div_w_roche_jrot(rpsi, m_phi, jrots(k), standard_cgrav, max_w, max_w2, w_flag)
            call emit_row(unit_no, 'w_div_w_roche', cid, 'omega_in', k, 0, omegas(k))
            call emit_row(unit_no, 'w_div_w_roche', cid, 'w_from_omega', k, 0, w_om)
            call emit_row(unit_no, 'w_div_w_roche', cid, 'jrot_in', k, 0, jrots(k))
            call emit_row(unit_no, 'w_div_w_roche', cid, 'w_from_jrot', k, 0, w_jr)
         end do
      end do
   end subroutine emit_w_div_w_roche_oracles

   ! --------------------------------------------------------------------------
   ! Section 4: get_rotation_sigmas (normal and am_sig_term_limit-capped)
   ! --------------------------------------------------------------------------
   subroutine emit_rotation_sigmas_oracles(unit_no, s)
      integer, intent(in) :: unit_no
      type(star_info), pointer, intent(inout) :: s
      integer, parameter :: nz = 12
      integer :: icase, k, ierr
      real(dp) :: dt, dq_raw(nz), dq_sum
      character(len=24) :: cid

      s%nz = nz
      call reallocate_star_info_arrays(s, ierr)
      if (ierr /= 0) stop 'reallocate failed in emit_rotation_sigmas_oracles'

      s%mstar = 4.0_dp * msun
      s%xmstar = s%mstar
      dt = 1.0e11_dp

      dq_sum = 0.0_dp
      do k = 1, nz
         dq_raw(k) = 1.5_dp**(k - 1)
         dq_sum = dq_sum + dq_raw(k)
      end do
      do k = 1, nz
         s%dq(k) = dq_raw(k) / dq_sum
         s%dm(k) = s%xmstar * s%dq(k)
         s%r(k) = 2.4_dp * rsun * (1.0_dp - 0.075_dp * real(k - 1, dp))
         s%rho(k) = 1.0e-3_dp * (2.5_dp**(k - 1))
      end do

      do icase = 1, 2
         if (icase == 1) then
            cid = 'sigmas_normal'
            s%am_sig_term_limit = 1.0e13_dp
            do k = 1, nz
               s%am_nu_omega(k) = 1.0e12_dp * real(k, dp)
               s%am_nu_j(k) = 5.0e11_dp * real(nz - k + 1, dp)
            end do
            s%am_nu_omega(4) = 0.0_dp
            s%am_nu_j(7) = -1.0e8_dp
         else
            cid = 'sigmas_capped'
            s%am_sig_term_limit = 1.0e3_dp
            do k = 1, nz
               s%am_nu_omega(k) = 1.0e20_dp * real(k, dp)
               s%am_nu_j(k) = 2.0e19_dp * real(nz - k + 1, dp)
            end do
         end if

         s%am_sig_j(1:nz) = 0.0_dp
         s%am_sig_omega(1:nz) = 0.0_dp
         call get_rotation_sigmas(s, 1, nz, dt, ierr)
         if (ierr /= 0) stop 'get_rotation_sigmas failed'

         call emit_row(unit_no, 'rotation_sigmas', cid, 'xmstar', 0, 0, s%xmstar)
         call emit_row(unit_no, 'rotation_sigmas', cid, 'dt', 0, 0, dt)
         call emit_row(unit_no, 'rotation_sigmas', cid, 'am_sig_term_limit', 0, 0, s%am_sig_term_limit)
         do k = 1, nz
            call emit_row(unit_no, 'rotation_sigmas', cid, 'r', k, 0, s%r(k))
            call emit_row(unit_no, 'rotation_sigmas', cid, 'rho', k, 0, s%rho(k))
            call emit_row(unit_no, 'rotation_sigmas', cid, 'dm', k, 0, s%dm(k))
            call emit_row(unit_no, 'rotation_sigmas', cid, 'dq', k, 0, s%dq(k))
            call emit_row(unit_no, 'rotation_sigmas', cid, 'am_nu_omega', k, 0, s%am_nu_omega(k))
            call emit_row(unit_no, 'rotation_sigmas', cid, 'am_nu_j', k, 0, s%am_nu_j(k))
            call emit_row(unit_no, 'rotation_sigmas', cid, 'am_sig_j', k, 0, s%am_sig_j(k))
            call emit_row(unit_no, 'rotation_sigmas', cid, 'am_sig_omega', k, 0, s%am_sig_omega(k))
         end do
      end do
   end subroutine emit_rotation_sigmas_oracles

   ! --------------------------------------------------------------------------
   ! Section 5: do_solve_omega_mix (real(qp) solver across 11 regimes)
   ! --------------------------------------------------------------------------
   subroutine init_omega_mix_profile(s, nz, omega_init, simple_irot)
      type(star_info), pointer, intent(inout) :: s
      integer, intent(in) :: nz
      real(dp), intent(in) :: omega_init(nz)
      logical, intent(in) :: simple_irot
      integer :: k, ierr
      real(dp) :: dq_raw(nz), dq_sum, r_in, vol

      s%nz = nz
      call reallocate_star_info_arrays(s, ierr)
      if (ierr /= 0) stop 'reallocate failed in init_omega_mix_profile'

      s%model_number = 10
      s%rotation_flag = .true.
      s%doing_timing = .false.
      s%report_ierr = .false.
      s%stop_for_bad_nums = .false.
      s%recalc_mixing_info_each_substep = .false.
      s%simple_i_rot_flag = simple_irot
      s%w_div_wcrit_max = 0.89_dp
      s%w_div_wcrit_max2 = 0.90_dp
      s%w_div_wc_flag = .false.
      s%use_other_eval_i_rot = .false.
      s%use_other_eval_fp_ft = .false.
      s%use_other_torque = .false.
      s%use_other_torque_implicit = .false.
      s%other_torque => null()
      s%other_torque_implicit => null()
      s%binary_other_torque => null()
      s%binary_other_torque_implicit => null()
      s%binary_other_irot => null()
      s%binary_other_fp_ft => null()
      s%am_sig_term_limit = 1.0e13_dp
      s%angular_momentum_error_retry = 1.0e-6_dp
      s%angular_momentum_error_warn = 5.0e-6_dp
      s%num_rotation_solver_steps = 0

      s%mstar = 4.0_dp * msun
      s%xmstar = s%mstar
      s%M_center = 0.0_dp
      s%R_center = 0.0_dp

      ! Stellar-like mass zoning: thin outer surface shell (dq(1) = 1e-10)
      ! matching realistic MESA surface resolution, with geometric interior cells.
      s%dq(1) = 1.0e-10_dp
      dq_sum = 0.0_dp
      do k = 2, nz
         dq_raw(k) = 1.4_dp**(k - 2)
         dq_sum = dq_sum + dq_raw(k)
      end do
      do k = 2, nz
         s%dq(k) = (1.0_dp - s%dq(1)) * dq_raw(k) / dq_sum
      end do
      do k = 1, nz
         s%dm(k) = s%xmstar * s%dq(k)
      end do
      s%q(1) = 1.0_dp
      do k = 2, nz
         s%q(k) = s%q(k - 1) - s%dq(k - 1)
      end do
      do k = 1, nz
         s%m(k) = s%xmstar * s%q(k)
         s%cgrav(k) = standard_cgrav
         if (k == 1) then
            s%r(k) = 2.4_dp * rsun
         else
            s%r(k) = 2.4_dp * rsun * (1.0_dp - 1.0e-4_dp - 0.90_dp * real(k - 2, dp) / real(nz - 2, dp))
         end if
         s%xh(s%i_lnR, k) = log(s%r(k))
      end do
      s%dm_bar(1) = 0.5_dp * s%dm(1)
      do k = 2, nz - 1
         s%dm_bar(k) = 0.5_dp * (s%dm(k - 1) + s%dm(k))
      end do
      s%dm_bar(nz) = 0.5_dp * s%dm(nz - 1) + s%dm(nz)

      do k = 1, nz
         if (k < nz) then
            r_in = s%r(k + 1)
         else
            r_in = 0.0_dp
         end if
         vol = (pi4 / 3.0_dp) * (s%r(k)**3 - r_in**3)
         s%rho(k) = s%dm(k) / vol
         s%omega(k) = omega_init(k)
      end do

      ! Initialize consistent Roche w_div_wc, i_rot, and j_rot from omega
      call use_xh_to_update_i_rot_and_j_rot(s)
      ! Recompute i_rot and omega from j_rot exactly as do_solve_omega_mix does on entry
      call set_i_rot(s, .false.)
      call set_omega(s, 'oracle_init')
   end subroutine init_omega_mix_profile

   subroutine emit_solve_omega_mix_oracles(unit_no, s)
      integer, intent(in) :: unit_no
      type(star_info), pointer, intent(inout) :: s
      integer, parameter :: nz = 24
      integer :: icase, k, res
      real(dp) :: omega_init(nz), dt_total, j_tot0, j_tot1, surf_rel_err
      logical :: simple_irot
      character(len=32) :: cid

      do icase = 1, 11
         simple_irot = .false.
         do k = 2, nz
            omega_init(k) = 3.0e-5_dp * (1.0_dp + 5.0_dp * real(k - 2, dp) / real(nz - 2, dp))
         end do
         omega_init(1) = omega_init(2)

         select case (icase)
         case (1)
            cid = 'diffusive_omega_shear'
            dt_total = 1.0e11_dp
         case (2)
            cid = 'diffusive_pure_j'
            omega_init(:) = 4.0e-5_dp
            dt_total = 1.0e10_dp
         case (3)
            cid = 'diffusive_mixed_omega_j'
            dt_total = 5.0e10_dp
         case (4)
            cid = 'diffusive_stiff_1e13'
            do k = 2, nz
               omega_init(k) = 1.0e-7_dp * (1.0_dp + 0.5_dp * real(k - 2, dp) / real(nz - 2, dp))
            end do
            omega_init(1) = omega_init(2)
            dt_total = 1.0e12_dp
         case (5)
            cid = 'explicit_extra_jdot'
            dt_total = 5.0e10_dp
         case (6)
            cid = 'explicit_extra_omegadot'
            dt_total = 5.0e10_dp
         case (7)
            cid = 'implicit_torque_tridiag'
            dt_total = 5.0e10_dp
         case (8)
            cid = 'adaptive_substepping'
            ! Sharp localized shear around k=8..14 with fast mixing timescale
            do k = 1, nz
               if (k <= 8) then
                  omega_init(k) = 1.0e-5_dp
               else if (k <= 14) then
                  omega_init(k) = 1.0e-5_dp + 1.5e-4_dp * real(k - 8, dp) / 6.0_dp
               else
                  omega_init(k) = 1.6e-4_dp
               end if
            end do
            dt_total = 2.0e10_dp
         case (9)
            cid = 'max_iters_retry'
            dt_total = 1.0e8_dp
         case (10)
            cid = 'conservation_retry_triggered'
            omega_init(:) = 2.0e-5_dp
            dt_total = 1.0e5_dp
         case default
            cid = 'conservation_retry_passed'
            omega_init(:) = 2.0e-5_dp
            dt_total = 1.0e5_dp
         end select

         call init_omega_mix_profile(s, nz, omega_init, simple_irot)

         s%am_nu_omega(1:nz) = 0.0_dp
         s%am_nu_j(1:nz) = 0.0_dp

         select case (icase)
         case (1)
            do k = 1, nz
               s%am_nu_omega(k) = 1.0e12_dp * (1.0_dp + 0.5_dp * real(k - 1, dp))
            end do
         case (2)
            s%am_nu_j(1:nz) = 5.0e13_dp
         case (3)
            do k = 1, nz
               s%am_nu_omega(k) = 3.0e12_dp * (1.0_dp + 0.2_dp * real(k - 1, dp))
               s%am_nu_j(k) = 8.0e12_dp * (1.0_dp + 0.1_dp * real(nz - k, dp))
            end do
         case (4)
            s%am_nu_omega(1:nz) = 1.0e26_dp
            s%am_sig_term_limit = 1.0e13_dp
         case (5)
            s%am_nu_omega(1:nz) = 2.0e12_dp
            s%use_other_torque = .true.
            s%other_torque => explicit_jdot_torque
         case (6)
            s%am_nu_omega(1:nz) = 2.0e12_dp
            s%am_nu_j(1:nz) = 1.0e12_dp
            s%use_other_torque = .true.
            s%other_torque => explicit_omegadot_torque
         case (7)
            s%am_nu_omega(1:nz) = 5.0e11_dp
            s%am_nu_j(1:nz) = 2.0e11_dp
            s%use_other_torque_implicit = .true.
            s%other_torque_implicit => implicit_tridiag_torque
         case (8)
            s%am_nu_omega(1:nz) = 5.0e14_dp
            s%am_nu_j(1:nz) = 5.0e13_dp
         case (9)
            s%am_nu_omega(1:nz) = 0.0_dp
            s%use_other_torque_implicit = .true.
            s%other_torque_implicit => implicit_nonconverging_torque
         case (10, 11)
            ! Perturb surface j_rot(1) so s% omega(1) = s% omega(2) produces a
            ! controlled conservation discrepancy |J1 - J0| / |J0|.
            s%j_rot(1) = s%i_rot(1)%val * 8.0e-5_dp
            call set_i_rot(s, .false.)
            call set_omega(s, 'conservation_setup')
            j_tot0 = total_angular_momentum(s)
            surf_rel_err = abs(s%dm_bar(1) * s%i_rot(1)%val * (s%omega(2) - s%omega(1))) / abs(j_tot0)
            if (icase == 10) then
               s%angular_momentum_error_retry = 0.5_dp * surf_rel_err
            else
               s%angular_momentum_error_retry = 2.0_dp * surf_rel_err
            end if
         end select

         ! Emit exact inputs before calling do_solve_omega_mix
         j_tot0 = total_angular_momentum(s)
         call emit_row(unit_no, 'solve_omega_mix', cid, 'xmstar', 0, 0, s%xmstar)
         call emit_row(unit_no, 'solve_omega_mix', cid, 'dt_total', 0, 0, dt_total)
         call emit_row(unit_no, 'solve_omega_mix', cid, 'am_sig_term_limit', 0, 0, s%am_sig_term_limit)
         call emit_row(unit_no, 'solve_omega_mix', cid, 'angular_momentum_error_retry', 0, 0, s%angular_momentum_error_retry)
         call emit_row(unit_no, 'solve_omega_mix', cid, 'J_tot0', 0, 0, j_tot0)
         do k = 1, nz
            call emit_row(unit_no, 'solve_omega_mix', cid, 'r', k, 0, s%r(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'm', k, 0, s%m(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'dm', k, 0, s%dm(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'dq', k, 0, s%dq(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'dm_bar', k, 0, s%dm_bar(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'rho', k, 0, s%rho(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'j_rot_init', k, 0, s%j_rot(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'am_nu_omega', k, 0, s%am_nu_omega(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'am_nu_j', k, 0, s%am_nu_j(k))
         end do

         s%num_rotation_solver_steps = 0
         implicit_hook_calls = 0
         res = do_solve_omega_mix(s, dt_total)
         j_tot1 = total_angular_momentum(s)

         write(*, '(A,A,I0,A,I0,A,I0,A,ES12.4)') &
            'solve_omega_mix ', trim(cid) // ': outcome=', res, &
            ' steps=', s%num_rotation_solver_steps, &
            ' imp_calls=', implicit_hook_calls, &
            ' rel_dJ=', abs(j_tot1 - j_tot0) / max(1.0e-99_dp, abs(j_tot0))

         call emit_row(unit_no, 'solve_omega_mix', cid, 'outcome', 0, 0, real(res, dp))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'steps_used', 0, 0, real(s%num_rotation_solver_steps, dp))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'implicit_calls', 0, 0, real(implicit_hook_calls, dp))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'J_tot1', 0, 0, j_tot1)
         do k = 1, nz
            call emit_row(unit_no, 'solve_omega_mix', cid, 'i_rot_final', k, 0, s%i_rot(k)%val)
            call emit_row(unit_no, 'solve_omega_mix', cid, 'omega_final', k, 0, s%omega(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'j_rot_final', k, 0, s%j_rot(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'am_sig_omega', k, 0, s%am_sig_omega(k))
            call emit_row(unit_no, 'solve_omega_mix', cid, 'am_sig_j', k, 0, s%am_sig_j(k))
         end do
      end do

      ! Case 12: simple_i_rot_flag = .true. shear transport
      cid = 'simple_i_rot_shear'
      do k = 2, nz
         omega_init(k) = 3.0e-5_dp * (1.0_dp + 5.0_dp * real(k - 2, dp) / real(nz - 2, dp))
      end do
      omega_init(1) = omega_init(2)
      dt_total = 1.0e11_dp
      call init_omega_mix_profile(s, nz, omega_init, .true.)
      do k = 1, nz
         s%am_nu_omega(k) = 1.0e12_dp * (1.0_dp + 0.5_dp * real(k - 1, dp))
         s%am_nu_j(k) = 0.0_dp
      end do
      j_tot0 = total_angular_momentum(s)
      call emit_row(unit_no, 'solve_omega_mix', cid, 'xmstar', 0, 0, s%xmstar)
      call emit_row(unit_no, 'solve_omega_mix', cid, 'dt_total', 0, 0, dt_total)
      call emit_row(unit_no, 'solve_omega_mix', cid, 'am_sig_term_limit', 0, 0, s%am_sig_term_limit)
      call emit_row(unit_no, 'solve_omega_mix', cid, 'angular_momentum_error_retry', 0, 0, s%angular_momentum_error_retry)
      call emit_row(unit_no, 'solve_omega_mix', cid, 'J_tot0', 0, 0, j_tot0)
      do k = 1, nz
         call emit_row(unit_no, 'solve_omega_mix', cid, 'r', k, 0, s%r(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'm', k, 0, s%m(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'dm', k, 0, s%dm(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'dq', k, 0, s%dq(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'dm_bar', k, 0, s%dm_bar(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'rho', k, 0, s%rho(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'j_rot_init', k, 0, s%j_rot(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'am_nu_omega', k, 0, s%am_nu_omega(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'am_nu_j', k, 0, s%am_nu_j(k))
      end do
      s%num_rotation_solver_steps = 0
      res = do_solve_omega_mix(s, dt_total)
      j_tot1 = total_angular_momentum(s)
      write(*, '(A,A,I0,A,I0,A,ES12.4)') &
         'solve_omega_mix ', trim(cid) // ': outcome=', res, &
         ' steps=', s%num_rotation_solver_steps, &
         ' rel_dJ=', abs(j_tot1 - j_tot0) / max(1.0e-99_dp, abs(j_tot0))
      call emit_row(unit_no, 'solve_omega_mix', cid, 'outcome', 0, 0, real(res, dp))
      call emit_row(unit_no, 'solve_omega_mix', cid, 'steps_used', 0, 0, real(s%num_rotation_solver_steps, dp))
      call emit_row(unit_no, 'solve_omega_mix', cid, 'implicit_calls', 0, 0, 0.0_dp)
      call emit_row(unit_no, 'solve_omega_mix', cid, 'J_tot1', 0, 0, j_tot1)
      do k = 1, nz
         call emit_row(unit_no, 'solve_omega_mix', cid, 'i_rot_final', k, 0, s%i_rot(k)%val)
         call emit_row(unit_no, 'solve_omega_mix', cid, 'omega_final', k, 0, s%omega(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'j_rot_final', k, 0, s%j_rot(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'am_sig_omega', k, 0, s%am_sig_omega(k))
         call emit_row(unit_no, 'solve_omega_mix', cid, 'am_sig_j', k, 0, s%am_sig_j(k))
      end do
   end subroutine emit_solve_omega_mix_oracles

end program gen_omega_mix_oracles
