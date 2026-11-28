! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex reference vectors against compiled MESA 26.4.1
! (`libturb.a` + `libauto_diff.a` + `CRMATH`) for `turb::set_semiconvection`
! (through `semiconvection::calc_semiconvection`) and `turb::set_thermohaline`
! at the operating points of `crates/flexmesa-star/tests/turb_155_bd_parity_test.rs`
! (#155, DECISIONS #80).
!
! Every differentiable input is an `auto_diff_real_star_order1` seeded in
! `d1Array(1)` only, as the Rust test seeds one partial. The TSV records the
! value and slot-1 partial of every input and output.

program gen_turb_mix_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use math_lib, only: MATH_BACKEND, math_init
   use auto_diff, only: auto_diff_real_star_order1, operator(+), operator(-), operator(*), operator(/), assignment(=)
   use const_def, only: dp, crad, standard_cgrav
   use const_lib, only: const_init
   use chem_lib, only: chem_init, chem_get_iso_id
   use turb, only: set_semiconvection, set_thermohaline
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver, fflags_str
   integer :: ierr, u, rc

   call get_environment_variable('MESA_DIR', mesa_dir)
   if (len_trim(mesa_dir) == 0) stop 'MESA_DIR must be set'
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   call get_environment_variable('ORACLE_FFLAGS', fflags_str)
   if (command_argument_count() >= 1) then
      call get_command_argument(1, out_path)
   else
      out_path = 'mesa_26_4_1_turb_mix_oracles.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'
   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'
   write(u, '(A)') '# FlexMESA MESA 26.4.1 turb set_semiconvection / set_thermohaline oracle vectors (#155, DECISIONS #80)'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') 'case_id' // char(9) // 'role' // char(9) // 'field' // char(9) // &
      'part' // char(9) // 'raw_bits_hex' // char(9) // 'decimal'

   rc = 0
   call emit_semiconvection(u, rc)
   call emit_thermohaline(u, rc)
   close(u)
   write(*, '(A,I0,A)') 'Wrote ', rc, ' turb mixing oracle rows.'

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

   type(auto_diff_real_star_order1) function seed(val, d0) result(ad)
      real(dp), intent(in) :: val, d0
      ad%val = val
      ad%d1Array(:) = 0.0_dp
      ad%d1Array(1) = d0
   end function seed

   subroutine emit(unit_no, rc, cid, role, fld, part, val)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: cid, role, fld, part
      real(dp), intent(in) :: val
      character(len=20) :: hbuf
      character(len=32) :: dbuf
      write(hbuf, '(A,Z16.16)') '0x', transfer(val, 0_int64)
      write(dbuf, '(ES25.17E3)') val
      write(unit_no, '(A)') trim(cid) // char(9) // trim(role) // char(9) // trim(fld) // &
         char(9) // trim(part) // char(9) // trim(hbuf) // char(9) // trim(adjustl(dbuf))
      rc = rc + 1
   end subroutine emit

   subroutine emit_ad(unit_no, rc, cid, role, fld, x)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: cid, role, fld
      type(auto_diff_real_star_order1), intent(in) :: x
      call emit(unit_no, rc, cid, role, fld, 'val', x%val)
      call emit(unit_no, rc, cid, role, fld, 'd1', x%d1Array(1))
   end subroutine emit_ad

   subroutine emit_outputs(unit_no, rc, cid, gradT, Y_face, conv_vel, D, mixing_type, ierr)
      integer, intent(in) :: unit_no, mixing_type, ierr
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: cid
      type(auto_diff_real_star_order1), intent(in) :: gradT, Y_face, conv_vel, D
      call emit_ad(unit_no, rc, cid, 'out', 'gradT', gradT)
      call emit_ad(unit_no, rc, cid, 'out', 'Y_face', Y_face)
      call emit_ad(unit_no, rc, cid, 'out', 'conv_vel', conv_vel)
      call emit_ad(unit_no, rc, cid, 'out', 'D', D)
      call emit(unit_no, rc, cid, 'out', 'mixing_type', 'val', real(mixing_type, dp))
      call emit(unit_no, rc, cid, 'out', 'ierr', 'val', real(ierr, dp))
   end subroutine emit_outputs

   ! `test_semiconvection_langer85_0_ulp_scalar_and_ad`.
   subroutine emit_semiconvection(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      type(auto_diff_real_star_order1) :: L, Lambda, T, P, Pr, beta, opacity, rho, Cp
      type(auto_diff_real_star_order1) :: gradr, grada, gradL, gradT, Y_face, conv_vel, D
      real(dp), parameter :: gradL_comp = 0.22_dp, m = 2.0e33_dp
      integer :: mixing_type, ierr

      L = seed(4.0e37_dp, 4.0e37_dp)
      Lambda = seed(3.0e9_dp, -1.5e9_dp)
      T = seed(1.8e7_dp, 1.8e7_dp)
      P = seed(1.2e16_dp, 2.4e16_dp)
      Pr = (crad/3.0_dp)*(T*T*T*T)
      beta = (P - Pr)/P
      opacity = seed(0.38_dp, -0.76_dp)
      rho = seed(5.5_dp, 5.5_dp)
      Cp = seed(4.2e8_dp, -1.0e8_dp)
      grada = seed(0.28_dp, 0.02_dp)
      gradL = grada + gradL_comp
      gradr = seed(0.41_dp, 0.15_dp)

      call emit_ad(unit_no, rc, 'sc_inputs', 'in', 'Pr', Pr)
      call emit_ad(unit_no, rc, 'sc_inputs', 'in', 'beta', beta)
      call emit_ad(unit_no, rc, 'sc_inputs', 'in', 'gradL', gradL)
      call emit(unit_no, rc, 'sc_inputs', 'in', 'cgrav', 'val', standard_cgrav)

      call set_semiconvection(L, Lambda, m, T, P, Pr, beta, opacity, rho, 0.1_dp, &
         'Langer_85 mixing; gradT = gradr', standard_cgrav, Cp, gradr, grada, gradL, &
         gradL_comp, gradT, Y_face, conv_vel, D, mixing_type, ierr)
      call emit_outputs(unit_no, rc, 'sc_langer85_gradt_is_gradr', gradT, Y_face, conv_vel, D, mixing_type, ierr)

      call set_semiconvection(L, Lambda, m, T, P, Pr, beta, opacity, rho, 0.1_dp, &
         'Langer_85', standard_cgrav, Cp, gradr, grada, gradL, &
         gradL_comp, gradT, Y_face, conv_vel, D, mixing_type, ierr)
      call emit_outputs(unit_no, rc, 'sc_langer85', gradT, Y_face, conv_vel, D, mixing_type, ierr)

      ! alpha_semiconvection = 0 with gradr == gradL.
      call set_semiconvection(L, Lambda, m, T, P, Pr, beta, opacity, rho, 0.0_dp, &
         'Langer_85', standard_cgrav, Cp, gradL, grada, gradL, &
         gradL_comp, gradT, Y_face, conv_vel, D, mixing_type, ierr)
      call emit_outputs(unit_no, rc, 'sc_langer85_alpha0_gradr_eq_gradl', gradT, Y_face, conv_vel, D, mixing_type, ierr)
   end subroutine emit_semiconvection

   ! `test_thermohaline_prescriptions_0_ulp_scalar_and_ad`; the driving
   ! species has Z = 2, A = 3 (he3).
   subroutine emit_thermohaline(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      type(auto_diff_real_star_order1) :: Lambda, grada, gradr, T, opacity, rho, Cp
      type(auto_diff_real_star_order1) :: gradT, Y_face, conv_vel, D
      real(dp), parameter :: gradL_comp = -0.05_dp, XH1 = 0.68_dp, coeff = 2.0_dp
      integer :: iso, mixing_type, ierr

      iso = chem_get_iso_id('he3')
      if (iso <= 0) stop 'he3 not found'
      Lambda = seed(4.5e9_dp, -1.25e9_dp)
      grada = seed(0.39_dp, 0.03_dp)
      gradr = seed(0.24_dp, 0.11_dp)
      T = seed(2.5e7_dp, 2.5e7_dp)
      opacity = seed(0.42_dp, -0.2_dp)
      rho = seed(15.0_dp, 15.0_dp)
      Cp = seed(3.8e8_dp, -0.5e8_dp)

      call set_thermohaline('Kippenhahn', Lambda, grada, gradr, T, opacity, rho, Cp, gradL_comp, &
         iso, XH1, coeff, D, gradT, Y_face, conv_vel, mixing_type, ierr)
      call emit_outputs(unit_no, rc, 'thrm_kippenhahn', gradT, Y_face, conv_vel, D, mixing_type, ierr)

      call set_thermohaline('Traxler_Garaud_Stellmach_11', Lambda, grada, gradr, T, opacity, rho, Cp, &
         gradL_comp, iso, XH1, coeff, D, gradT, Y_face, conv_vel, mixing_type, ierr)
      call emit_outputs(unit_no, rc, 'thrm_tgs11', gradT, Y_face, conv_vel, D, mixing_type, ierr)

      call set_thermohaline('Brown_Garaud_Stellmach_13', Lambda, grada, gradr, T, opacity, rho, Cp, &
         gradL_comp, iso, XH1, coeff, D, gradT, Y_face, conv_vel, mixing_type, ierr)
      call emit_outputs(unit_no, rc, 'thrm_bgs13', gradT, Y_face, conv_vel, D, mixing_type, ierr)
   end subroutine emit_thermohaline

end program gen_turb_mix_oracles
