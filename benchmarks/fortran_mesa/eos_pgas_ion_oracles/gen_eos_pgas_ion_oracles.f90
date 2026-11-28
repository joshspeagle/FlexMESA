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
   use eos_def, only: EoS_General_Info, num_eos_basic_results, num_eos_d_dxa_results, num_helm_results, &
      i_lnPgas, i_lnE, eosdt_OPAL_SCVH, eosdt_max_FreeEOS
   use eos_lib, only: eos_init, eos_shutdown, alloc_eos_handle, free_eos_handle, eos_ptr, eosPT_get, helmeos2_eval
   use eosPT_eval, only: get_Pgas
   use eos_helm_eval, only: Get_HELMEOS_Results
   use eosdt_eval, only: get1_for_eosdt
   use eospc_eval, only: Get_PC_alfa
   use ideal, only: get_ideal_for_eosdt
   use ion_offset, only: compute_ion_offset
   use skye, only: Get_Skye_alfa_simple
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
   call emit_helm_cases(u, rc)
   call emit_pc_alfa_cases(u, rc)
   call emit_skye_alfa_simple_cases(u, rc)
   call emit_get1_eosdt_cases(u, rc)
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

   subroutine emit_helm_one(unit_no, rc, rq, label, t_in, rho_in, ab_in, zb_in, emit_raw)
      use eos_def
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      type (EoS_General_Info), pointer :: rq
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: t_in, rho_in, ab_in, zb_in
      logical, intent(in) :: emit_raw

      real(dp) :: lt_in, lr_in
      real(dp) :: res(num_eos_basic_results), dlnd(num_eos_basic_results), dlnt(num_eos_basic_results)
      real(dp) :: dabar(num_eos_basic_results), dzbar(num_eos_basic_results)
      real(dp) :: helm_res(num_helm_results)
      logical :: skip, off_table
      integer :: ierr_loc, k
      character(len=16) :: fld

      integer, parameter :: helm_slots(93) = [ &
         h_ptot, h_dpt, h_dpd, h_dpa, h_dpz, &
         h_etot, h_det, h_ded, h_dea, h_dez, h_dedd, h_dedt, h_deda, h_dedz, &
         h_stot, h_dst, h_dsd, h_dsa, h_dsz, h_dsdd, h_dsdt, h_dsda, h_dsdz, h_dstt, h_dsta, h_dstz, &
         h_prad, h_dpradt, h_dpradd, h_erad, h_srad, &
         h_pgas, h_dpgast, h_dpgasd, h_dpgasa, h_dpgasz, h_dpgasdd, h_dpgasdt, h_egas, h_sgas, &
         h_xni, h_etaele, h_detat, h_detad, h_detaa, h_detaz, &
         h_xne, h_dxnet, h_dxned, h_dxnea, h_dxnez, &
         h_plasg, &
         h_cp, h_dcpdd, h_dcpdt, h_dcpda, h_dcpdz, &
         h_cv, h_dcvdd, h_dcvdt, h_dcvda, h_dcvdz, &
         h_gam1, h_dgam1dd, h_dgam1dt, h_dgam1da, h_dgam1dz, &
         h_gam2, h_dgam2dd, h_dgam2dt, h_dgam2da, h_dgam2dz, &
         h_gam3, h_dgam3dd, h_dgam3dt, h_dgam3da, h_dgam3dz, &
         h_nabad, h_dnabdd, h_dnabdt, h_dnabda, h_dnabdz, &
         h_cs, &
         h_chit, h_dchitdd, h_dchitdt, h_dchitda, h_dchitdz, &
         h_chid, h_dchiddd, h_dchiddt, h_dchidda, h_dchiddz ]

      lt_in = log10(t_in)
      lr_in = log10(rho_in)
      res = 0.0_dp; dlnd = 0.0_dp; dlnt = 0.0_dp; dabar = 0.0_dp; dzbar = 0.0_dp
      helm_res = 0.0_dp

      call Get_HELMEOS_Results(rq, 0.0_dp, ab_in, zb_in, rho_in, lr_in, t_in, lt_in, &
         res, dlnd, dlnt, dabar, dzbar, helm_res, skip, ierr_loc)
      if (ierr_loc /= 0 .or. skip) stop 'Get_HELMEOS_Results failed'

      call emit(unit_no, rc, trim(label), 'in', 'T', 'val', t_in)
      call emit(unit_no, rc, trim(label), 'in', 'logT', 'val', lt_in)
      call emit(unit_no, rc, trim(label), 'in', 'Rho', 'val', rho_in)
      call emit(unit_no, rc, trim(label), 'in', 'logRho', 'val', lr_in)
      call emit(unit_no, rc, trim(label), 'in', 'abar', 'val', ab_in)
      call emit(unit_no, rc, trim(label), 'in', 'zbar', 'val', zb_in)

      do k = 1, 16
         write(fld, '(A,I0)') 'res_', k - 1
         call emit(unit_no, rc, trim(label), 'out', trim(fld), 'val', res(k))
         call emit(unit_no, rc, trim(label), 'out', trim(fld), 'd_dlnd', dlnd(k))
         call emit(unit_no, rc, trim(label), 'out', trim(fld), 'd_dlnT', dlnt(k))
         call emit(unit_no, rc, trim(label), 'out', trim(fld), 'd_dabar', dabar(k))
         call emit(unit_no, rc, trim(label), 'out', trim(fld), 'd_dzbar', dzbar(k))
      end do

      if (emit_raw) then
         helm_res = 0.0_dp
         call helmeos2_eval(t_in, lt_in, rho_in, lr_in, ab_in, zb_in, &
            rq% coulomb_temp_cut_HELM, rq% coulomb_den_cut_HELM, &
            helm_res, .false., .true., .true., off_table, ierr_loc)
         if (ierr_loc /= 0 .or. off_table) stop 'helmeos2_eval failed'
         do k = 1, 93
            write(fld, '(A,I0)') 'helm_', helm_slots(k)
            call emit(unit_no, rc, trim(label), 'out', trim(fld), 'val', helm_res(helm_slots(k)))
         end do
      end if
   end subroutine emit_helm_one

   subroutine emit_helm_cases(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      type (EoS_General_Info), pointer :: rq
      integer :: eh, ierr_loc

      eh = alloc_eos_handle(ierr_loc)
      if (ierr_loc /= 0) stop 'helm alloc_eos_handle failed'
      call eos_ptr(eh, rq, ierr_loc)
      if (ierr_loc /= 0) stop 'helm eos_ptr failed'

      call emit_helm_one(unit_no, rc, rq, 'helm_test_components', &
         1.0e6_dp, exp10(3.5_dp), 2.990964711670598_dp, 1.6749402385355348_dp, .false.)
      call emit_helm_one(unit_no, rc, rq, 'helm_sample_eos_c12', &
         1.0e9_dp, 1.0e4_dp, 12.0_dp, 6.0_dp, .false.)
      call emit_helm_one(unit_no, rc, rq, 'helm_pure_he_2e8_100', &
         2.0e8_dp, 100.0_dp, 4.0_dp, 2.0_dp, .true.)
      call emit_helm_one(unit_no, rc, rq, 'helm_pure_h_2e8_100', &
         2.0e8_dp, 100.0_dp, 1.0_dp, 1.0_dp, .false.)

      call free_eos_handle(eh)
   end subroutine emit_helm_cases

   subroutine emit_pc_alfa_one(unit_no, rc, rq, label, lr, lt, z_in, ab_in, zb_in)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      type (EoS_General_Info), pointer :: rq
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: lr, lt, z_in, ab_in, zb_in
      real(dp) :: alfa, dt_a, dr_a
      integer :: ierr_loc

      call Get_PC_alfa(rq, lr, lt, z_in, ab_in, zb_in, alfa, dt_a, dr_a, ierr_loc)
      if (ierr_loc /= 0) stop 'Get_PC_alfa failed'

      call emit(unit_no, rc, trim(label), 'in', 'logRho', 'val', lr)
      call emit(unit_no, rc, trim(label), 'in', 'logT', 'val', lt)
      call emit(unit_no, rc, trim(label), 'in', 'abar', 'val', ab_in)
      call emit(unit_no, rc, trim(label), 'in', 'zbar', 'val', zb_in)
      call emit(unit_no, rc, trim(label), 'out', 'alfa', 'val', alfa)
      call emit(unit_no, rc, trim(label), 'out', 'd_alfa_dlogT', 'val', dt_a)
      call emit(unit_no, rc, trim(label), 'out', 'd_alfa_dlogRho', 'val', dr_a)
   end subroutine emit_pc_alfa_one

   subroutine emit_pc_alfa_cases(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      type (EoS_General_Info), pointer :: rq
      integer :: eh, ierr_loc

      eh = alloc_eos_handle(ierr_loc)
      if (ierr_loc /= 0) stop 'pc_alfa alloc_eos_handle failed'
      call eos_ptr(eh, rq, ierr_loc)
      if (ierr_loc /= 0) stop 'pc_alfa eos_ptr failed'

      rq% PC_use_Gamma_limit_instead_of_T = .true.
      call emit_pc_alfa_one(unit_no, rc, rq, 'pc_alfa_gamma_c12_2p9_6p5', &
         2.9_dp, 6.5_dp, 1.0_dp, 12.0_dp, 6.0_dp)
      call emit_pc_alfa_one(unit_no, rc, rq, 'pc_alfa_gamma_c12_3p5_8p35', &
         3.5_dp, 8.35_dp, 1.0_dp, 12.0_dp, 6.0_dp)

      rq% PC_use_Gamma_limit_instead_of_T = .false.
      call emit_pc_alfa_one(unit_no, rc, rq, 'pc_alfa_tlim_c12_3p3_7p65', &
         3.3_dp, 7.65_dp, 1.0_dp, 12.0_dp, 6.0_dp)

      call free_eos_handle(eh)
   end subroutine emit_pc_alfa_cases

   subroutine emit_skye_alfa_simple_one(unit_no, rc, rq, label, lr, lt, z_in, ab_in, zb_in)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      type (EoS_General_Info), pointer :: rq
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: lr, lt, z_in, ab_in, zb_in
      real(dp) :: alfa, dt_a, dr_a
      integer :: ierr_loc

      call Get_Skye_alfa_simple(rq, lr, lt, z_in, ab_in, zb_in, alfa, dt_a, dr_a, ierr_loc)
      if (ierr_loc /= 0) stop 'Get_Skye_alfa_simple failed'

      call emit(unit_no, rc, trim(label), 'in', 'logRho', 'val', lr)
      call emit(unit_no, rc, trim(label), 'in', 'logT', 'val', lt)
      call emit(unit_no, rc, trim(label), 'in', 'logRho_any', 'val', rq% logRho_min_for_any_Skye)
      call emit(unit_no, rc, trim(label), 'in', 'logRho_all', 'val', rq% logRho_min_for_all_Skye)
      call emit(unit_no, rc, trim(label), 'in', 'logT_any', 'val', rq% logT_min_for_any_Skye)
      call emit(unit_no, rc, trim(label), 'in', 'logT_all', 'val', rq% logT_min_for_all_Skye)
      call emit(unit_no, rc, trim(label), 'out', 'alfa', 'val', alfa)
      call emit(unit_no, rc, trim(label), 'out', 'd_alfa_dlogT', 'val', dt_a)
      call emit(unit_no, rc, trim(label), 'out', 'd_alfa_dlogRho', 'val', dr_a)
   end subroutine emit_skye_alfa_simple_one

   subroutine emit_skye_alfa_simple_cases(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      type (EoS_General_Info), pointer :: rq
      integer :: eh, ierr_loc

      eh = alloc_eos_handle(ierr_loc)
      if (ierr_loc /= 0) stop 'skye_alfa alloc_eos_handle failed'
      call eos_ptr(eh, rq, ierr_loc)
      if (ierr_loc /= 0) stop 'skye_alfa eos_ptr failed'

      call emit_skye_alfa_simple_one(unit_no, rc, rq, &
         'skye_alfa_simple_2p35_7p3', 2.35_dp, 7.3_dp, 1.0_dp, 12.0_dp, 6.0_dp)

      rq% logRho_min_for_any_Skye = 2.3_dp
      rq% logRho_min_for_all_Skye = 2.4_dp
      rq% logT_min_for_any_Skye = 7.2_dp
      rq% logT_min_for_all_Skye = 7.4_dp
      call emit_skye_alfa_simple_one(unit_no, rc, rq, &
         'skye_alfa_simple_f64_2p35_7p3', 2.35_dp, 7.3_dp, 1.0_dp, 12.0_dp, 6.0_dp)

      call free_eos_handle(eh)
   end subroutine emit_skye_alfa_simple_cases

   subroutine emit_get1_eosdt_one(unit_no, rc, eh, which_eosdt, label, &
         z_c, x_c, abar_c, zbar_c, cid7, niso7, xa7, rho_in, temp_in)
      integer, intent(in) :: unit_no, eh, which_eosdt
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: z_c, x_c, abar_c, zbar_c, xa7(7), rho_in, temp_in
      integer, pointer :: cid7(:), niso7(:)

      real(dp) :: res(num_eos_basic_results), dlnd(num_eos_basic_results), dlnt(num_eos_basic_results)
      real(dp) :: ddxa_eos(num_eos_basic_results, 7)
      real(dp) :: lr_in, lt_in
      logical :: skip
      integer :: ierr_loc, k
      character(len=16) :: fld

      lr_in = log10(rho_in)
      lt_in = log10(temp_in)
      res = 0.0_dp; dlnd = 0.0_dp; dlnt = 0.0_dp; ddxa_eos = 0.0_dp
      call get1_for_eosdt(eh, which_eosdt, .false., z_c, x_c, abar_c, zbar_c, &
         7, cid7, niso7, xa7, rho_in, lr_in, temp_in, lt_in, 1.0_dp, &
         res, dlnd, dlnt, ddxa_eos, skip, ierr_loc)
      if (ierr_loc /= 0 .or. skip) stop 'get1_for_eosdt failed'

      call emit(unit_no, rc, trim(label), 'in', 'X', 'val', x_c)
      call emit(unit_no, rc, trim(label), 'in', 'Z', 'val', z_c)
      call emit(unit_no, rc, trim(label), 'in', 'rho', 'val', rho_in)
      call emit(unit_no, rc, trim(label), 'in', 'logRho', 'val', lr_in)
      call emit(unit_no, rc, trim(label), 'in', 'T', 'val', temp_in)
      call emit(unit_no, rc, trim(label), 'in', 'logT', 'val', lt_in)
      do k = 1, 16
         write(fld, '(A,I0)') 'res_', k - 1
         call emit(unit_no, rc, trim(label), 'out', trim(fld), 'val', res(k))
         call emit(unit_no, rc, trim(label), 'out', trim(fld), 'd_dlnd', dlnd(k))
         call emit(unit_no, rc, trim(label), 'out', trim(fld), 'd_dlnT', dlnt(k))
      end do
   end subroutine emit_get1_eosdt_one

   subroutine emit_get1_eosdt_cases(unit_no, rc)
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

      integer, pointer :: cid7(:), niso7(:)
      real(dp) :: xa7(7), x_c, y_c, z_c, abar_c, zbar_c, z2bar_c, z53bar_c, ye_c, mc_c, sx_c
      integer :: eh, k, ierr_loc

      allocate(cid7(7), niso7(num_chem_isos))
      niso7 = 0
      do k = 1, 7
         cid7(k) = chem_get_iso_id(names(k))
         if (cid7(k) <= 0) stop 'get1 7sp isotope not found'
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
      if (ierr_loc /= 0) stop 'get1 alloc_eos_handle failed'

      call emit_get1_eosdt_one(unit_no, rc, eh, eosdt_OPAL_SCVH, &
         'get1_opal_scvh_solar_1em2_1e5', z_c, x_c, abar_c, zbar_c, cid7, niso7, xa7, &
         1.0e-2_dp, 1.0e5_dp)
      call emit_get1_eosdt_one(unit_no, rc, eh, eosdt_max_FreeEOS, &
         'get1_free_eos_solar_1em2_1e6', z_c, x_c, abar_c, zbar_c, cid7, niso7, xa7, &
         1.0e-2_dp, 1.0e6_dp)

      call free_eos_handle(eh)
      deallocate(cid7, niso7)
   end subroutine emit_get1_eosdt_cases

end program gen_eos_pgas_ion_oracles
