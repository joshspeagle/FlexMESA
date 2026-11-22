! SPDX-License-Identifier: LGPL-3.0-or-later
! Generate IEEE-754 hex + decimal reference vectors against compiled MESA 26.4.1
! (`libturb.a` + `libauto_diff.a` + `CRMATH`) for `turb::set_TDC` values and all
! 16 input `d1Array` partials across the 32 operating points in upstream
! `turb/test/src/test_turb.f90` (#328, #321, DECISIONS #50).

program gen_turb_tdc_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use math_lib, only: MATH_BACKEND, math_init, pow, pow2, pow4
   use auto_diff, only: auto_diff_real_star_order1
   use const_def, only: dp, pi, rsun, lsun, msun, kerg, mp, boltz_sigma, standard_cgrav
   use const_lib, only: const_init
   use turb, only: set_TDC
   implicit none

   integer, parameter :: NUM_DIFF_INPUTS = 16
   character(len=16), parameter :: SLOT_NAMES(NUM_DIFF_INPUTS) = [ character(len=16) :: &
      'chiT', 'chiRho', 'gradr', 'r', 'P', 'T', 'rho', 'dV', &
      'Cp', 'opacity', 'scale_height', 'gradL', 'grada', 'Eq_div_w', 'grav', 'energy' ]

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver, fflags_str
   integer :: ierr, u, row_count

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
      out_path = 'mesa_26_4_1_turb_tdc_oracles.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'

   write(u, '(A)') '# FlexMESA MESA 26.4.1 turb::set_TDC value and d1Array partial oracle vectors (#328, #321)'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') 'section	case_id	role	field	slot	slot_name	raw_bits_hex	decimal'

   row_count = 0
   call emit_check_tdc_section(u, row_count)
   call emit_compare_tdc_cox_section(u, row_count)

   close(u)
   write(*, '(A,I0,A)') 'Wrote ', row_count, ' turb set_TDC oracle rows.'

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

   subroutine init_seed(val, slot, ad)
      real(dp), intent(in) :: val
      integer, intent(in) :: slot
      type(auto_diff_real_star_order1), intent(out) :: ad
      ad%val = val
      ad%d1Array(:) = 0.0_dp
      if (slot >= 1 .and. slot <= 33) then
         ad%d1Array(slot) = 1.0_dp
      end if
   end subroutine init_seed

   subroutine emit_f64(unit_no, rc, sec, cid, role, fld, slot, sname, val)
      integer, intent(in) :: unit_no, slot
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: sec, cid, role, fld, sname
      real(dp), intent(in) :: val
      integer(int64) :: bits
      character(len=20) :: hbuf
      character(len=32) :: dbuf
      bits = transfer(val, 0_int64)
      write(hbuf, '(A,Z16.16)') '0x', bits
      write(dbuf, '(ES25.17E3)') val
      write(unit_no, '(A,A,A,A,A,A,A,A,I0,A,A,A,A,A,A)') &
         trim(sec), char(9), trim(cid), char(9), trim(role), char(9), &
         trim(fld), char(9), slot, char(9), trim(sname), char(9), &
         trim(hbuf), char(9), trim(adjustl(dbuf))
      rc = rc + 1
   end subroutine emit_f64

   subroutine emit_int(unit_no, rc, sec, cid, role, fld, slot, sname, ival)
      integer, intent(in) :: unit_no, slot, ival
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: sec, cid, role, fld, sname
      call emit_f64(unit_no, rc, sec, cid, role, fld, slot, sname, real(ival, dp))
   end subroutine emit_int

   subroutine run_and_emit_case( &
      unit_no, rc, sec, cid, &
      conv_vel_start, mixing_length_alpha, tdc_alpha_d, tdc_alpha_r, tdc_alpha_pt, &
      dt, cgrav, m, scale, max_conv_vel, include_mlt_corr, tdc_alpha_c, tdc_alpha_s, &
      use_limiter, in_vals)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: sec, cid
      real(dp), intent(in) :: conv_vel_start, mixing_length_alpha, tdc_alpha_d, tdc_alpha_r, tdc_alpha_pt
      real(dp), intent(in) :: dt, cgrav, m, scale, max_conv_vel, tdc_alpha_c, tdc_alpha_s
      logical, intent(in) :: include_mlt_corr, use_limiter
      real(dp), intent(in) :: in_vals(NUM_DIFF_INPUTS)

      type(auto_diff_real_star_order1) :: chiT, chiRho, gradr, r, P, T, rho, dV, Cp, opacity
      type(auto_diff_real_star_order1) :: scale_height, gradL, grada, Eq_div_w, grav, energy
      type(auto_diff_real_star_order1) :: conv_vel, D, Y_face, gradT
      integer :: mixing_type, tdc_num_iters, ierr, slot

      call init_seed(in_vals(1), 1, chiT)
      call init_seed(in_vals(2), 2, chiRho)
      call init_seed(in_vals(3), 3, gradr)
      call init_seed(in_vals(4), 4, r)
      call init_seed(in_vals(5), 5, P)
      call init_seed(in_vals(6), 6, T)
      call init_seed(in_vals(7), 7, rho)
      call init_seed(in_vals(8), 8, dV)
      call init_seed(in_vals(9), 9, Cp)
      call init_seed(in_vals(10), 10, opacity)
      call init_seed(in_vals(11), 11, scale_height)
      call init_seed(in_vals(12), 12, gradL)
      call init_seed(in_vals(13), 13, grada)
      call init_seed(in_vals(14), 14, Eq_div_w)
      call init_seed(in_vals(15), 15, grav)
      call init_seed(in_vals(16), 16, energy)

      ierr = 0
      call set_TDC( &
         conv_vel_start, mixing_length_alpha, tdc_alpha_d, tdc_alpha_r, tdc_alpha_pt, &
         dt, cgrav, m, .false., mixing_type, scale, &
         chiT, chiRho, gradr, r, P, T, rho, dV, Cp, opacity, scale_height, gradL, grada, &
         conv_vel, D, Y_face, gradT, tdc_num_iters, max_conv_vel, Eq_div_w, grav, &
         include_mlt_corr, tdc_alpha_c, tdc_alpha_s, use_limiter, energy, ierr)
      if (ierr /= 0) then
         write(*, '(A,A,A,I0)') 'set_TDC failed for ', trim(cid), ': ierr = ', ierr
         stop 'set_TDC failed'
      end if

      if (any(Y_face%d1Array(NUM_DIFF_INPUTS + 1:33) /= 0.0_dp)) then
         stop 'unexpected non-zero trailing Y_face%d1Array entry'
      end if
      if (any(conv_vel%d1Array(NUM_DIFF_INPUTS + 1:33) /= 0.0_dp)) then
         stop 'unexpected non-zero trailing conv_vel%d1Array entry'
      end if

      ! Emit scalar parameters
      call emit_f64(unit_no, rc, sec, cid, 'param', 'conv_vel_start', 0, '-', conv_vel_start)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'mixing_length_alpha', 0, '-', mixing_length_alpha)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'TDC_alpha_D', 0, '-', tdc_alpha_d)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'TDC_alpha_R', 0, '-', tdc_alpha_r)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'TDC_alpha_Pt', 0, '-', tdc_alpha_pt)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'dt', 0, '-', dt)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'cgrav', 0, '-', cgrav)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'm', 0, '-', m)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'scale', 0, '-', scale)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'max_conv_vel', 0, '-', max_conv_vel)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'TDC_alpha_C', 0, '-', tdc_alpha_c)
      call emit_f64(unit_no, rc, sec, cid, 'param', 'TDC_alpha_S', 0, '-', tdc_alpha_s)

      ! Emit all 16 differentiable inputs in FlexMESA TdcInputs order
      do slot = 1, NUM_DIFF_INPUTS
         call emit_f64(unit_no, rc, sec, cid, 'input', trim(SLOT_NAMES(slot)), slot, trim(SLOT_NAMES(slot)), in_vals(slot))
      end do

      ! Emit primary scalar and integer outputs
      call emit_int(unit_no, rc, sec, cid, 'output', 'tdc_num_iters', 0, '-', tdc_num_iters)
      call emit_int(unit_no, rc, sec, cid, 'output', 'mixing_type', 0, '-', mixing_type)
      call emit_f64(unit_no, rc, sec, cid, 'output', 'Y_face_val', 0, '-', Y_face%val)
      call emit_f64(unit_no, rc, sec, cid, 'output', 'conv_vel_val', 0, '-', conv_vel%val)
      call emit_f64(unit_no, rc, sec, cid, 'output', 'gradT_val', 0, '-', gradT%val)
      call emit_f64(unit_no, rc, sec, cid, 'output', 'D_val', 0, '-', D%val)

      ! Emit all 16 d1Array partials of Y_face and conv_vel
      do slot = 1, NUM_DIFF_INPUTS
         call emit_f64(unit_no, rc, sec, cid, 'output', 'Y_face_d1', slot, trim(SLOT_NAMES(slot)), Y_face%d1Array(slot))
      end do
      do slot = 1, NUM_DIFF_INPUTS
         call emit_f64(unit_no, rc, sec, cid, 'output', 'conv_vel_d1', slot, trim(SLOT_NAMES(slot)), conv_vel%d1Array(slot))
      end do
   end subroutine run_and_emit_case

   subroutine emit_check_tdc_section(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      real(dp) :: conv_vel_start, mixing_length_alpha, tdc_alpha_d, tdc_alpha_r, tdc_alpha_pt
      real(dp) :: dt, cgrav, m, scale, max_conv_vel, L_val, tdc_alpha_c, tdc_alpha_s
      real(dp) :: r_v, p_v, t_v, opacity_v, rho_v, dv_v, chit_v, chirho_v, cp_v, gradr_v
      real(dp) :: grada_v, hp_v, gradl_v, eq_v, grav_v, en_v
      real(dp) :: in_vals(NUM_DIFF_INPUTS)
      integer :: j
      character(len=32) :: cid

      ! Matches `turb/test/src/test_turb.f90:182-229`
      max_conv_vel = 1d99
      conv_vel_start = 52320587.415154047d0
      mixing_length_alpha = 2.0d0
      tdc_alpha_d = 1.0d0
      tdc_alpha_r = 0.0d0
      tdc_alpha_pt = 0.0d0
      tdc_alpha_c = 1.0d0
      tdc_alpha_s = 1.0d0
      cgrav = 6.6743000000000004d-8
      m = 5.8707400456875664d34
      scale = 5.0386519362246294d45
      L_val = 1.0941528815883500015d0 * (-5.0386519362246294d45)
      r_v = 10314294541.567163d0
      p_v = 5.0581587249808894d20
      t_v = 613193666.51783681d0
      rho_v = 5204.5732574745753d0
      dv_v = 3.8256494463482604d-7
      cp_v = 6628075118.4606590d0
      opacity_v = 9.0750171231469945d-2
      hp_v = 2638686602.0063782d0
      gradl_v = 0.25207587267343501d0
      grada_v = 0.25204697256872738d0
      chit_v = 1.0_dp
      chirho_v = 1.0_dp
      gradr_v = 3d0 * p_v * opacity_v * L_val / (64 * pi * boltz_sigma * pow4(t_v) * cgrav * m)
      grav_v = m * cgrav / pow2(r_v)
      eq_v = 0.0_dp
      en_v = 0.0_dp

      in_vals = [ chit_v, chirho_v, gradr_v, r_v, p_v, t_v, rho_v, dv_v, &
                  cp_v, opacity_v, hp_v, gradl_v, grada_v, eq_v, grav_v, en_v ]

      do j = 0, 30
         dt = 500d0 * pow(1.02d0, j)
         write(cid, '(A,I2.2)') 'check_tdc_j', j
         call run_and_emit_case( &
            unit_no, rc, 'check_tdc', cid, &
            conv_vel_start, mixing_length_alpha, tdc_alpha_d, tdc_alpha_r, tdc_alpha_pt, &
            dt, cgrav, m, scale, max_conv_vel, .true., tdc_alpha_c, tdc_alpha_s, .false., in_vals)
      end do
   end subroutine emit_check_tdc_section

   subroutine emit_compare_tdc_cox_section(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      real(dp) :: mixing_length_alpha, conv_vel_start, tdc_alpha_d, tdc_alpha_r, tdc_alpha_pt
      real(dp) :: dt, cgrav, m, scale, tdc_alpha_c, tdc_alpha_s, max_conv_vel, L_val
      real(dp) :: r_v, t_v, p_v, opacity_v, rho_v, dv_v, chirho_v, chit_v, cp_v
      real(dp) :: gradr_v, grada_v, hp_v, gradl_v, grav_v, eq_v, en_v
      real(dp) :: in_vals(NUM_DIFF_INPUTS)

      ! Matches `turb/test/src/test_turb.f90:99-145`
      max_conv_vel = 1d99
      mixing_length_alpha = 2.0d0
      chit_v = 1d0
      chirho_v = 1d0
      t_v = 1d5
      rho_v = 1d-5
      r_v = rsun
      m = msun
      cgrav = standard_cgrav
      grav_v = m * cgrav / pow2(r_v)
      cp_v = 2.5d0 * kerg / mp
      p_v = rho_v * t_v * kerg / mp
      hp_v = p_v / (rho_v * grav_v)
      opacity_v = 1d0
      grada_v = 0.4d0
      gradl_v = grada_v

      L_val = 70 * lsun
      gradr_v = 3d0 * p_v * opacity_v * L_val / (64 * pi * boltz_sigma * pow4(t_v) * grav_v * pow2(r_v))
      L_val = L_val * (1d0 + 1d-5) * (grada_v / gradr_v)
      gradr_v = 3d0 * p_v * opacity_v * L_val / (64 * pi * boltz_sigma * pow4(t_v) * grav_v * pow2(r_v))

      tdc_alpha_d = 1.0d0
      tdc_alpha_r = 0.0d0
      tdc_alpha_pt = 0.0d0
      tdc_alpha_c = 1.0d0
      tdc_alpha_s = 1.0d0
      dv_v = 0d0
      en_v = 0d0
      conv_vel_start = 0d0
      scale = L_val * 1d-3
      dt = 1d40
      eq_v = 0d0

      in_vals = [ chit_v, chirho_v, gradr_v, r_v, p_v, t_v, rho_v, dv_v, &
                  cp_v, opacity_v, hp_v, gradl_v, grada_v, eq_v, grav_v, en_v ]

      call run_and_emit_case( &
         unit_no, rc, 'compare_tdc_cox', 'compare_tdc_cox_0', &
         conv_vel_start, mixing_length_alpha, tdc_alpha_d, tdc_alpha_r, tdc_alpha_pt, &
         dt, cgrav, m, scale, max_conv_vel, .true., tdc_alpha_c, tdc_alpha_s, .false., in_vals)
   end subroutine emit_compare_tdc_cox_section

end program gen_turb_tdc_oracles
