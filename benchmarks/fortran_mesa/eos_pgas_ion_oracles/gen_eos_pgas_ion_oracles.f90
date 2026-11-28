! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex reference vectors against compiled MESA 26.4.1
! (`libeos.a` + `CRMATH`) for:
! - `ion_offset::compute_ion_offset` at the operating points of
!   `crates/flexmesa-eos/tests/wave3a1_golden_tests.rs`
! - `eosPT_eval::get_Pgas` at the operating point of
!   `crates/flexmesa-eos/tests/wave3b1_golden_tests.rs`
! (#161, DECISIONS #80).

program gen_eos_pgas_ion_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use math_lib, only: MATH_BACKEND, math_init, exp, exp10, log, log10
   use const_def, only: dp, ln10, arg_not_provided
   use const_lib, only: const_init
   use chem_def, only: num_chem_isos
   use chem_lib, only: chem_init, chem_shutdown, chem_get_iso_id, basic_composition_info
   use eos_def, only: num_eos_basic_results, num_eos_d_dxa_results, i_lnPgas, i_lnE
   use eos_lib, only: eos_init, eos_shutdown, alloc_eos_handle, free_eos_handle, eosPT_get
   use eosPT_eval, only: get_Pgas
   use ideal, only: get_ideal_for_eosdt
   use ion_offset, only: compute_ion_offset
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
      out_path = 'mesa_26_4_1_eos_pgas_ion_oracles.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'
   call eos_init(' ', .true., ierr)
   if (ierr /= 0) stop 'eos_init failed'
   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'
   write(u, '(A)') '# FlexMESA MESA 26.4.1 eos exact oracle vectors (#161, DECISIONS #80)'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') 'case_id' // char(9) // 'role' // char(9) // 'field' // char(9) // &
      'part' // char(9) // 'raw_bits_hex' // char(9) // 'decimal'

   rc = 0
   call emit_ion_offset_cases(u, rc)
   call emit_get_pgas_case(u, rc)
   call emit_ideal_case(u, rc)
   close(u)
   call eos_shutdown()
   call chem_shutdown()
   write(*, '(A,I0,A)') 'Wrote ', rc, ' eos_pgas_ion_oracles rows.'

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

   subroutine emit_ion_offset_cases(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      integer, pointer :: cid1(:), cid3(:)
      real(dp) :: xa1(1), xa3(3), off
      integer :: k
      character(len=8) :: fld

      allocate(cid1(1), cid3(3))
      cid1(1) = chem_get_iso_id('h1      ')
      if (cid1(1) <= 0) stop 'h1 not found'
      xa1 = [1.0_dp]
      call emit(unit_no, rc, 'ion_offset_pure_h', 'in', 'xa_0', 'val', xa1(1))
      off = compute_ion_offset(1, xa1, cid1)
      call emit(unit_no, rc, 'ion_offset_pure_h', 'out', 'offset', 'val', off)

      cid3(1) = chem_get_iso_id('h1      ')
      cid3(2) = chem_get_iso_id('he4     ')
      cid3(3) = chem_get_iso_id('c12     ')
      if (any(cid3 <= 0)) stop '3sp isotope not found'
      xa3 = [0.70_dp, 0.28_dp, 0.02_dp]
      do k = 1, 3
         write(fld, '(A,I0)') 'xa_', k - 1
         call emit(unit_no, rc, 'ion_offset_solar_3sp', 'in', trim(fld), 'val', xa3(k))
      end do
      off = compute_ion_offset(3, xa3, cid3)
      call emit(unit_no, rc, 'ion_offset_solar_3sp', 'out', 'offset', 'val', off)

      deallocate(cid1, cid3)
   end subroutine emit_ion_offset_cases

   subroutine emit_get_pgas_case(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      character(len=8), parameter :: names(7) = [ &
         'h1      ', 'he4     ', 'c12     ', 'n14     ', &
         'o16     ', 'ne20    ', 'mg24    ']
      real(dp), parameter :: zfrac_c = 0.173312_dp
      real(dp), parameter :: zfrac_n = 0.053177_dp
      real(dp), parameter :: zfrac_o = 0.482398_dp
      real(dp), parameter :: zfrac_ne = 0.098675_dp
      real(dp), parameter :: x_in = 0.70_dp, zinit_in = 0.02_dp
      real(dp), parameter :: log_pgas_ref = 15.0_dp, log_t_in = 7.2_dp
      real(dp), parameter :: log_pgas_guess = 14.8_dp, log_pgas_tol = 1.0e-6_dp

      integer, pointer :: cid7(:), niso7(:)
      real(dp) :: xa7(7), x_c, y_c, z_c, abar_c, zbar_c, z2bar_c, z53bar_c, ye_c, mc_c, sx_c
      real(dp) :: fwd_res(num_eos_basic_results), res(num_eos_basic_results)
      real(dp) :: dlnd(num_eos_basic_results), dlnt(num_eos_basic_results)
      real(dp) :: ddxa_star(num_eos_d_dxa_results, 7)
      real(dp) :: ddxa_eos(num_eos_basic_results, 7)
      real(dp) :: rho_fwd, log10_rho_fwd, dlnr_dlnp_fwd, dlnr_dlnt_fwd
      real(dp) :: target_lne, other_tol
      real(dp) :: log_pgas_res, rho_pt, log10_rho_pt, dlnr_dlnp, dlnr_dlnt
      integer :: eh, k, eos_calls, ierr_loc
      character(len=8) :: fld

      allocate(cid7(7), niso7(num_chem_isos))
      niso7 = 0
      do k = 1, 7
         cid7(k) = chem_get_iso_id(names(k))
         if (cid7(k) <= 0) stop '7sp isotope not found'
         niso7(cid7(k)) = k
      end do

      xa7(1) = x_in
      xa7(2) = 1.0_dp - (x_in + zinit_in)
      xa7(3) = zinit_in * zfrac_c
      xa7(4) = zinit_in * zfrac_n
      xa7(5) = zinit_in * zfrac_o
      xa7(6) = zinit_in * zfrac_ne
      xa7(7) = 1.0_dp - (((((xa7(1) + xa7(2)) + xa7(3)) + xa7(4)) + xa7(5)) + xa7(6))

      call basic_composition_info(7, cid7, xa7, x_c, y_c, z_c, &
         abar_c, zbar_c, z2bar_c, z53bar_c, ye_c, mc_c, sx_c)

      eh = alloc_eos_handle(ierr_loc)
      if (ierr_loc /= 0) stop 'alloc_eos_handle failed'

      fwd_res = 0.0_dp; dlnd = 0.0_dp; dlnt = 0.0_dp; ddxa_star = 0.0_dp
      call eosPT_get(eh, 7, cid7, niso7, xa7, &
         arg_not_provided, log_pgas_ref, arg_not_provided, log_t_in, &
         rho_fwd, log10_rho_fwd, dlnr_dlnp_fwd, dlnr_dlnt_fwd, &
         fwd_res, dlnd, dlnt, ddxa_star, ierr_loc)
      if (ierr_loc /= 0) stop 'fwd eosPT_get failed'

      target_lne = fwd_res(i_lnE)
      other_tol = ln10 * log_pgas_tol

      res = 0.0_dp; dlnd = 0.0_dp; dlnt = 0.0_dp; ddxa_eos = 0.0_dp
      call get_Pgas(eh, z_c, x_c, abar_c, zbar_c, &
         7, cid7, niso7, xa7, &
         log_t_in, i_lnE, target_lne, &
         log_pgas_tol, other_tol, 100, log_pgas_guess, &
         arg_not_provided, arg_not_provided, arg_not_provided, arg_not_provided, &
         log_pgas_res, rho_pt, log10_rho_pt, dlnr_dlnp, dlnr_dlnt, &
         res, dlnd, dlnt, ddxa_eos, eos_calls, ierr_loc)
      if (ierr_loc /= 0) stop 'get_Pgas failed'

      do k = 1, 7
         write(fld, '(A,I0)') 'xa_', k - 1
         call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'in', trim(fld), 'val', xa7(k))
      end do
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'in', 'logT', 'val', log_t_in)
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'in', 'target_lne', 'val', target_lne)
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'in', 'logPgas_guess', 'val', log_pgas_guess)
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'in', 'logPgas_tol', 'val', log_pgas_tol)
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'in', 'other_tol', 'val', other_tol)

      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'out', 'logPgas_result', 'val', log_pgas_res)
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'out', 'Rho', 'val', rho_pt)
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'out', 'logRho', 'val', log10_rho_pt)
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'out', 'dlnRho_dlnPgas_c_T', 'val', dlnr_dlnp)
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'out', 'dlnRho_dlnT_c_Pgas', 'val', dlnr_dlnt)
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'out', 'res_lnE', 'val', res(i_lnE))
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'out', 'res_lnPgas', 'val', res(i_lnPgas))
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'out', 'eos_calls', 'val', real(eos_calls, dp))
      call emit(unit_no, rc, 'eospt_get_pgas_solar_15_7p2', 'out', 'ierr', 'val', real(ierr_loc, dp))

      call free_eos_handle(eh)
      deallocate(cid7, niso7)
   end subroutine emit_get_pgas_case

   subroutine emit_ideal_case(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      character(len=8), parameter :: names(3) = ['h1      ', 'he4     ', 'c12     ']
      integer, pointer :: cid3(:), niso3(:)
      real(dp) :: xa3(3), x_c, y_c, z_c, abar_c, zbar_c, z2bar_c, z53bar_c, ye_c, mc_c, sx_c
      real(dp) :: res(num_eos_basic_results), dlnd(num_eos_basic_results), dlnt(num_eos_basic_results)
      real(dp) :: ddxa_eos(num_eos_basic_results, 3)
      real(dp) :: temp_in, rho_in
      integer :: eh, k, ierr_loc
      logical :: skip
      character(len=8) :: fld

      allocate(cid3(3), niso3(num_chem_isos))
      niso3 = 0
      do k = 1, 3
         cid3(k) = chem_get_iso_id(names(k))
         if (cid3(k) <= 0) stop 'ideal 3sp isotope not found'
         niso3(cid3(k)) = k
      end do
      xa3 = [0.70_dp, 0.28_dp, 0.02_dp]
      call basic_composition_info(3, cid3, xa3, x_c, y_c, z_c, &
         abar_c, zbar_c, z2bar_c, z53bar_c, ye_c, mc_c, sx_c)

      temp_in = 1.0e5_dp
      rho_in = 1.0e-2_dp
      eh = alloc_eos_handle(ierr_loc)
      if (ierr_loc /= 0) stop 'ideal alloc_eos_handle failed'

      res = 0.0_dp; dlnd = 0.0_dp; dlnt = 0.0_dp; ddxa_eos = 0.0_dp
      call get_ideal_for_eosdt(eh, .false., z_c, x_c, abar_c, zbar_c, &
         3, cid3, niso3, xa3, rho_in, log10(rho_in), temp_in, log10(temp_in), 1.0_dp, &
         res, dlnd, dlnt, ddxa_eos, skip, ierr_loc)
      if (ierr_loc /= 0 .or. skip) stop 'get_ideal_for_eosdt failed'

      call emit(unit_no, rc, 'ideal_solar_1e5_1em2', 'in', 'T', 'val', temp_in)
      call emit(unit_no, rc, 'ideal_solar_1e5_1em2', 'in', 'Rho', 'val', rho_in)
      call emit(unit_no, rc, 'ideal_solar_1e5_1em2', 'in', 'abar', 'val', abar_c)
      call emit(unit_no, rc, 'ideal_solar_1e5_1em2', 'in', 'zbar', 'val', zbar_c)
      do k = 1, 3
         write(fld, '(A,I0)') 'xa_', k - 1
         call emit(unit_no, rc, 'ideal_solar_1e5_1em2', 'in', trim(fld), 'val', xa3(k))
      end do

      do k = 1, num_eos_basic_results
         write(fld, '(A,I0)') 'res_', k - 1
         call emit(unit_no, rc, 'ideal_solar_1e5_1em2', 'out', trim(fld), 'val', res(k))
         call emit(unit_no, rc, 'ideal_solar_1e5_1em2', 'out', trim(fld), 'd_dlnd', dlnd(k))
         call emit(unit_no, rc, 'ideal_solar_1e5_1em2', 'out', trim(fld), 'd_dlnT', dlnt(k))
      end do

      call free_eos_handle(eh)
      deallocate(cid3, niso3)
   end subroutine emit_ideal_case

end program gen_eos_pgas_ion_oracles
