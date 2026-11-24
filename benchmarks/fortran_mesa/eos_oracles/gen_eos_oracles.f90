! SPDX-License-Identifier: GPL-3.0-or-later
! -----------------------------------------------------------------------------
! Program: gen_eos_oracles
!
! Generates `benchmarks/fortran_mesa/eos_oracles/mesa_26_4_1_eos_oracles.tsv`
! directly from compiled upstream Fortran MESA release 26.4.1 (`libeos.a`,
! `libchem.a`, `libconst.a`, `libmath.a` / `CRMATH`, `libinterp_2d.a`,
! `libnum.a`, `libauto_diff.a`) for Issue #161 (audit findings T2-15 & T1-11).
!
! Build & run (with MESA SDK 26.6.1 initialized and MESA_DIR set to 26.4.1):
!   make -C benchmarks/fortran_mesa/eos_oracles generate
! -----------------------------------------------------------------------------
program gen_eos_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use const_def, only: dp, arg_not_provided, kerg, me, clight, ln10
   use const_lib, only: const_init
   use math_lib, only: MATH_BACKEND, math_init, exp, exp10, log, log10
   use chem_def, only: num_chem_isos
   use chem_lib, only: chem_init, chem_shutdown, chem_get_iso_id, basic_composition_info
   use eos_def, only: EoS_General_Info, num_eos_basic_results, &
      num_eos_d_dxa_results, num_helm_results, i_eos_OPAL_SCVH, &
      i_eos_FreeEOS, i_eos_PC, i_eos_Skye, i_eos_CMS, i_eos_ideal, &
      i_lnPgas, i_lnE, i_mu, i_lnfree_e, i_eta
   use eos_lib, only: eos_init, eos_shutdown, alloc_eos_handle, &
      free_eos_handle, eos_ptr, eos_get_helm_results, &
      eos_convert_helm_results, eosDT_get_component, eosDT_get, &
      eosPT_get, eosDT_get_T, eosDT_get_Rho, eosPT_get_T, &
      eos_gamma_DE_get, eos_fermi_dirac_integral, helmeos2_eval
   use eospc_eval, only: Get_PC_alfa
   use skye, only: Get_Skye_alfa, Get_Skye_alfa_simple
   implicit none

   character(len=512) :: mesa_dir_arg, out_path, sdk_ver
   character(len=64) :: ver_str
   integer :: ierr, eos_handle, iunit, out_unit
   type(EoS_General_Info), pointer :: rq
   integer, pointer :: chem_id(:), net_iso(:), cid3(:), niso3(:)
   real(dp) :: xa(7), xa3(3)
   real(dp) :: x_c, y_c, z_c, abar_co, zbar_co, z2bar_c, z53bar_c, ye_c, mc_c, sx_c
   real(dp) :: abar_c12, zbar_c12
   real(dp) :: eta_d, temp_d, theta_d, dk_list(4), eta_list(6), theta_list(4)
   real(dp) :: dom_lr(8), dom_lt(8), dom_ab(8), dom_zb(8)
   integer :: ik, ie, it, g_idx
   character(len=32) :: dom_label

   call get_environment_variable('MESA_DIR', mesa_dir_arg)
   if (len_trim(mesa_dir_arg) == 0) stop 'MESA_DIR must be set'
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   call get_command_argument(1, out_path)
   if (len_trim(out_path) == 0) out_path = 'mesa_26_4_1_eos_oracles.tsv'

   call const_init(mesa_dir_arg, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'
   call eos_init(' ', .true., ierr)
   if (ierr /= 0) stop 'eos_init failed'

   allocate(chem_id(7), net_iso(num_chem_isos))
   call setup_7sp_network(chem_id, net_iso)
   allocate(cid3(3), niso3(num_chem_isos))
   call setup_3sp_network(cid3, niso3)

   ver_str = 'unknown'
   open(newunit=iunit, file=trim(mesa_dir_arg)//'/data/version_number', &
      status='old', action='read', iostat=ierr)
   if (ierr == 0) then
      read(iunit, '(a)', iostat=ierr) ver_str
      close(iunit)
   end if

   open(newunit=out_unit, file=trim(out_path), status='replace', &
      action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'

   write (out_unit, '(a)') '# FlexMESA upstream Fortran MESA 26.4.1 EOS module oracle table (#161, T2-15, T1-11)'
   write (out_unit, '(a,a)') '# mesa_version_number=', trim(adjustl(ver_str))
   write (out_unit, '(a)') '# mesa_git_tag=26.4.1'
   write (out_unit, '(a)') '# mesa_git_commit=c8c202662922a22b3efefa824473ad615c623afd'
   write (out_unit, '(a,a)') '# mesasdk_version=', trim(adjustl(sdk_ver))
   write (out_unit, '(a,a)') '# math_backend=', trim(MATH_BACKEND)
   write (out_unit, '(a,a)') '# compiler=', trim(f90_compiler_version())
   write (out_unit, '(a)') '# fflags=-march=x86-64 -mtune=generic -mfpmath=sse -O2 -ffp-contract=off -fprotect-parens -fno-sign-zero -fno-range-check'
   write (out_unit, '(a)') '# schema_dfermi: DFERMI	case_id	dk	eta	theta	fd	fdeta	fdtheta'
   write (out_unit, '(a)') '# schema_alfa:   ALFA	case_id	alfa	d_alfa_dlogT	d_alfa_dlogRho'
   write (out_unit, '(a)') '# schema_helm:   HELM	case_id	idx	val	d_dlnd	d_dlnT	d_dabar	d_dzbar'
   write (out_unit, '(a)') '# schema_eos:    EOS	case_id	idx	val	d_dlnd	d_dlnT'
   write (out_unit, '(a)') '# schema_inv:    INV	case_id	v0	v1	v2	v3	v4	v5'
   write (out_unit, '(a)') '# schema_ierr:   IERR	case_id	ierr_code	off_table_int'

   ! ------------------------------------------------------------------
   ! 1. Fermi-Dirac integrals (`eos_fermi_dirac_integral` / `gauss_fermi.f90`)
   ! ------------------------------------------------------------------
   eta_d = 1.46722890948893d0
   temp_d = 11327678.5183021d0
   theta_d = (kerg * temp_d) / (me * clight * clight)
   call emit_dfermi_case('dirac_test_m0p5', -0.5d0, eta_d, theta_d)
   call emit_dfermi_case('dirac_test_p0p5',  0.5d0, eta_d, theta_d)
   call emit_dfermi_case('dirac_test_p1p5',  1.5d0, eta_d, theta_d)
   call emit_dfermi_case('dirac_test_p2p5',  2.5d0, eta_d, theta_d)

   dk_list = [-0.5d0, 0.5d0, 1.5d0, 2.5d0]
   eta_list = [-4.0d0, -0.5d0, 0.0d0, 2.5d0, 12.0d0, 35.0d0]
   theta_list = [1.0d-5, 1.0d-2, 0.2d0, 2.0d0]
   g_idx = 0
   do ik = 1, 4
      do ie = 1, 6
         do it = 1, 4
            write (dom_label, '(a,i0)') 'grid_', g_idx
            call emit_dfermi_case(trim(dom_label), dk_list(ik), eta_list(ie), theta_list(it))
            g_idx = g_idx + 1
         end do
      end do
   end do

   ! ------------------------------------------------------------------
   ! 2. Ideal gas + radiation component (`i_eos_ideal`)
   ! ------------------------------------------------------------------
   eos_handle = alloc_eos_handle(ierr)
   xa3 = [0.70d0, 0.28d0, 0.02d0]
   call emit_comp_case_n(eos_handle, i_eos_ideal, 'ideal_solar_1e5_1em2', &
      3, cid3, niso3, xa3, 1.0d-2, 1.0d5)
   call free_eos_handle(eos_handle)

   ! ------------------------------------------------------------------
   ! 3. HELM 16-channel x 5-column cases + off-table / crystallized ierr checks
   ! ------------------------------------------------------------------
   eos_handle = alloc_eos_handle(ierr)
   call eos_ptr(eos_handle, rq, ierr)
   call emit_helm_case(rq, 'pure_he_2e8_100', 2.0d8, 100.0d0, 4.0d0, 2.0d0)
   call emit_helm_case(rq, 'pure_h_2e8_100', 2.0d8, 100.0d0, 1.0d0, 1.0d0)
   call emit_helm_case(rq, 'solar_2e8_100', 2.0d8, 100.0d0, &
      1.2966353410255222d0, 1.1021400398716938d0)
   call emit_helm_case(rq, 'test_components', 1.0d6, exp10(3.5d0), &
      2.9909647116705980d0, 1.6749402385355348d0)
   call emit_helm_case(rq, 'sample_eos_c12', 1.0d9, 1.0d4, 12.0d0, 6.0d0)

   ! Additional Python `test_eos_wave3a2.py` sample and full-domain states
   call emit_helm_case(rq, 'mixed_z_2e8_100', 2.0d8, 100.0d0, 1.72d0, 1.15d0)
   call emit_helm_case(rq, 'solar_1e6_100', 1.0d6, 100.0d0, &
      1.29818593d0, 1.10539606d0)
   call emit_helm_case(rq, 'py_test_components', exp10(6.1d0), exp10(0.1d0), &
      1.29666341d0, 1.10668478d0)
   call emit_helm_case(rq, 'py_sample_eos_c12', 1.0d9, 1.0d6, 12.0d0, 6.0d0)

   dom_lr = [-7.35d0, -3.42d0, 0.73d0, 2.15d0, 4.62d0, 6.48d0, 8.35d0, 9.40d0]
   dom_lt = [ 4.15d0,  5.68d0, 6.84d0, 7.18d0, 7.85d0, 8.12d0, 8.95d0, 9.30d0]
   dom_ab = [ 1.29d0,  1.29d0, 1.65d0, 1.30d0, 4.00d0, 13.71d0, 16.00d0, 28.00d0]
   dom_zb = [ 1.10d0,  1.10d0, 1.18d0, 1.10d0, 2.00d0,  6.86d0,  8.00d0, 14.00d0]
   do ik = 1, 8
      write (dom_label, '(a,i0)') 'helm_dom_', ik - 1
      call emit_helm_case(rq, trim(dom_label), exp10(dom_lt(ik)), exp10(dom_lr(ik)), &
         dom_ab(ik), dom_zb(ik))
   end do

   call emit_helm_ierr_checks(rq)
   call free_eos_handle(eos_handle)

   ! ------------------------------------------------------------------
   ! 4. Blend weight (`Get_PC_alfa`, `Get_Skye_alfa`, `Get_Skye_alfa_simple`)
   ! ------------------------------------------------------------------
   call make_xa(0.0d0, 0.5d0, 0.5d0, xa)
   call basic_composition_info(7, chem_id, xa, x_c, y_c, z_c, &
      abar_co, zbar_co, z2bar_c, z53bar_c, ye_c, mc_c, sx_c)
   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call basic_composition_info(7, chem_id, xa, x_c, y_c, z_c, &
      abar_c12, zbar_c12, z2bar_c, z53bar_c, ye_c, mc_c, sx_c)

   eos_handle = alloc_eos_handle(ierr)
   call eos_ptr(eos_handle, rq, ierr)
   rq% PC_use_Gamma_limit_instead_of_T = .true.
   call emit_pc_alfa_case(rq, 'pc_alfa_gamma_c12_2p9_6p5', 2.9d0, 6.5d0, 1.0d0, 12.0d0, 6.0d0)
   call emit_pc_alfa_case(rq, 'pc_alfa_gamma_c12_3p5_8p35', 3.5d0, 8.35d0, 1.0d0, 12.0d0, 6.0d0)
   call emit_pc_alfa_case(rq, 'pc_alfa_co_1p0_7p0', 1.0d0, 7.0d0, 1.0d0, abar_co, zbar_co)
   call emit_pc_alfa_case(rq, 'pc_alfa_co_2p9_7p0', 2.9d0, 7.0d0, 1.0d0, abar_co, zbar_co)
   call emit_pc_alfa_case(rq, 'pc_alfa_co_5p0_6p2', 5.0d0, 6.2d0, 1.0d0, abar_co, zbar_co)
   call emit_pc_alfa_case(rq, 'pc_alfa_co_6p0_7p65', 6.0d0, 7.65d0, 1.0d0, abar_co, zbar_co)

   rq% PC_use_Gamma_limit_instead_of_T = .false.
   call emit_pc_alfa_case(rq, 'pc_alfa_tlim_c12_3p3_7p65', 3.3d0, 7.65d0, 1.0d0, 12.0d0, 6.0d0)

   call emit_skye_alfa_case(rq, 'skye_alfa_c12_4p0_7p5', 4.0d0, 7.5d0, 1.0d0, 12.0d0, 6.0d0)
   call emit_skye_alfa_case(rq, 'skye_alfa_c12_3p45_7p0', 3.45d0, 7.0d0, 1.0d0, 12.0d0, 6.0d0)
   call emit_skye_alfa_case(rq, 'skye_alfa_co_6p0_7p0', 6.0d0, 7.0d0, 1.0d0, abar_co, zbar_co)
   call emit_skye_alfa_case(rq, 'skye_alfa_co_m10_4p0', -10.0d0, 4.0d0, 1.0d0, abar_co, zbar_co)

   call emit_skye_alfa_simple_case(rq, 'skye_alfa_simple_2p0_6p0', 2.0d0, 6.0d0, 1.0d0, 12.0d0, 6.0d0)
   call emit_skye_alfa_simple_case(rq, 'skye_alfa_simple_2p35_7p3', 2.35d0, 7.3d0, 1.0d0, 12.0d0, 6.0d0)
   call emit_skye_alfa_simple_case(rq, 'skye_alfa_simple_4p0_8p0', 4.0d0, 8.0d0, 1.0d0, 12.0d0, 6.0d0)
   rq% logRho_min_for_any_Skye = 2.3d0
   rq% logRho_min_for_all_Skye = 2.4d0
   rq% logT_min_for_any_Skye = 7.2d0
   rq% logT_min_for_all_Skye = 7.4d0
   call emit_skye_alfa_simple_case(rq, 'skye_alfa_simple_f64_2p35_7p3', 2.35d0, 7.3d0, 1.0d0, 12.0d0, 6.0d0)
   call free_eos_handle(eos_handle)

   ! ------------------------------------------------------------------
   ! 5. Seven PC component cases + PC/HELM & PC/FreeEOS blends
   ! ------------------------------------------------------------------
   eos_handle = alloc_eos_handle(ierr)
   call eos_ptr(eos_handle, rq, ierr)

   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_comp_case_n(eos_handle, i_eos_PC, 'pc_c12_liq', &
      7, chem_id, net_iso, xa, 1.0d6, 1.0d8)

   call make_xa(0.0d0, 0.5d0, 0.5d0, xa)
   call emit_comp_case_n(eos_handle, i_eos_PC, 'pc_co_liq', &
      7, chem_id, net_iso, xa, 1.0d6, 1.0d7)

   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_comp_case_n(eos_handle, i_eos_PC, 'pc_c12_trans_eosfi', &
      7, chem_id, net_iso, xa, 1.0d6, 3.1d6)
   call emit_comp_case_n(eos_handle, i_eos_PC, 'pc_c12_trans_phase', &
      7, chem_id, net_iso, xa, 1.0d6, 2.05d6)
   call emit_comp_case_n(eos_handle, i_eos_PC, 'pc_c12_sol', &
      7, chem_id, net_iso, xa, 1.0d6, 1.0d6)

   call make_xa(0.0d0, 0.5d0, 0.5d0, xa)
   call emit_comp_case_n(eos_handle, i_eos_PC, 'pc_co_sol', &
      7, chem_id, net_iso, xa, 1.0d6, 1.0d6)

   call make_xa(1.0d0, 0.0d0, 0.0d0, xa)
   call emit_comp_case_n(eos_handle, i_eos_PC, 'pc_he4_qmelt', &
      7, chem_id, net_iso, xa, 1.0d8, 1.0d5)

   rq% use_Skye = .false.
   rq% use_PC = .true.
   rq% use_FreeEOS = .false.
   rq% use_OPAL_SCVH = .false.
   rq% PC_use_Gamma_limit_instead_of_T = .true.
   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_dt_case(eos_handle, 'eosdt_pc_helm_blend', &
      7, chem_id, net_iso, xa, exp10(2.9d0), exp10(6.5d0))

   rq% use_FreeEOS = .true.
   rq% okay_to_convert_ierr_to_skip = .false.
   call emit_dt_case(eos_handle, 'eosdt_pc_freeeos_blend', &
      7, chem_id, net_iso, xa, exp10(2.9d0), exp10(6.5d0))
   call free_eos_handle(eos_handle)

   ! ------------------------------------------------------------------
   ! 6. Six Skye component cases + Skye/HELM blend
   ! ------------------------------------------------------------------
   eos_handle = alloc_eos_handle(ierr)
   call eos_ptr(eos_handle, rq, ierr)

   rq% Skye_use_ion_offsets = .false.
   rq% Skye_solid_mixing_rule = 'Ogata'
   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_comp_case_n(eos_handle, i_eos_Skye, 'skye_c12_liq', &
      7, chem_id, net_iso, xa, 1.0d6, 1.0d8)

   rq% Skye_use_ion_offsets = .true.
   call emit_comp_case_n(eos_handle, i_eos_Skye, 'skye_c12_liq_ion', &
      7, chem_id, net_iso, xa, 1.0d6, 1.0d8)

   rq% Skye_use_ion_offsets = .false.
   call make_xa(0.0d0, 0.5d0, 0.5d0, xa)
   call emit_comp_case_n(eos_handle, i_eos_Skye, 'skye_co_liq', &
      7, chem_id, net_iso, xa, 1.0d6, 1.0d7)
   call emit_comp_case_n(eos_handle, i_eos_Skye, 'skye_co_sol_ogata', &
      7, chem_id, net_iso, xa, 1.0d6, 2.0d6)

   rq% Skye_solid_mixing_rule = 'PC'
   call emit_comp_case_n(eos_handle, i_eos_Skye, 'skye_co_sol_pc', &
      7, chem_id, net_iso, xa, 1.0d6, 2.0d6)

   rq% Skye_solid_mixing_rule = 'Ogata'
   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_comp_case_n(eos_handle, i_eos_Skye, 'skye_c12_phase', &
      7, chem_id, net_iso, xa, 1.0d6, 2.65d6)

   rq% use_Skye = .true.
   rq% use_PC = .false.
   rq% use_FreeEOS = .false.
   rq% use_OPAL_SCVH = .false.
   rq% Skye_use_ion_offsets = .false.
   call emit_dt_case(eos_handle, 'eosdt_skye_helm_blend', &
      7, chem_id, net_iso, xa, exp10(3.45d0), 1.0d7)
   rq% Skye_use_ion_offsets = .true.
   call emit_dt_case(eos_handle, 'eosdt_skye_helm_blend_ion', &
      7, chem_id, net_iso, xa, exp10(3.45d0), 1.0d7)
   call free_eos_handle(eos_handle)

   ! ------------------------------------------------------------------
   ! 7. Standalone 4D table components (`OPAL/SCVH`, `FreeEOS`, `CMS`)
   ! ------------------------------------------------------------------
   eos_handle = alloc_eos_handle(ierr)
   call eos_ptr(eos_handle, rq, ierr)
   call make_solar_xa(0.70d0, 0.02d0, xa)
   call emit_comp_case_n(eos_handle, i_eos_OPAL_SCVH, 'opal_scvh_solar_1em2_1e5', &
      7, chem_id, net_iso, xa, 1.0d-2, 1.0d5)
   call emit_comp_case_n(eos_handle, i_eos_FreeEOS, 'free_eos_solar_1em2_1e6', &
      7, chem_id, net_iso, xa, 1.0d-2, 1.0d6)
   rq% use_CMS = .true.
   call emit_comp_case_n(eos_handle, i_eos_CMS, 'cms_solar_1em2_1e5', &
      7, chem_id, net_iso, xa, 1.0d-2, 1.0d5)
   rq% include_radiation = .false.
   call emit_comp_case_n(eos_handle, i_eos_CMS, 'cms_solar_1em2_1e5_norad', &
      7, chem_id, net_iso, xa, 1.0d-2, 1.0d5)
   call free_eos_handle(eos_handle)

   ! ------------------------------------------------------------------
   ! 7b. Level-6 HELM/ideal edge blends (`eosdt_eval.f90:728,749-750`, #176)
   ! ------------------------------------------------------------------
   eos_handle = alloc_eos_handle(ierr)
   call eos_ptr(eos_handle, rq, ierr)
   rq% use_CMS = .false.
   rq% use_Skye = .false.
   rq% use_PC = .false.
   rq% use_FreeEOS = .false.
   rq% use_OPAL_SCVH = .false.
   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_dt_case(eos_handle, 'eosdt_helm_ideal_hi_rho_blend', &
      7, chem_id, net_iso, xa, exp10(15.05d0), exp10(12.5d0))
   rq% include_elec_pos = .false.
   call make_solar_xa(0.70d0, 0.02d0, xa)
   call emit_dt_case(eos_handle, 'eosdt_helm_ideal_lo_t_noep_blend', &
      7, chem_id, net_iso, xa, 1.0d-8, exp10(2.95d0))
   call free_eos_handle(eos_handle)

   ! ------------------------------------------------------------------
   ! 8. Inverse EOS solvers (`eosPT_get`, `eosDT_get_T`, `eosDT_get_Rho`, `eos_gamma_DE_get`, `eosPT_get_T`)
   ! ------------------------------------------------------------------
   call emit_inverse_cases(chem_id, net_iso, abar_c12)

   close(out_unit)
   deallocate(chem_id, net_iso, cid3, niso3)
   call eos_shutdown()
   call chem_shutdown()

contains

   subroutine setup_7sp_network(cid, niso)
      integer, pointer :: cid(:), niso(:)
      character(len=8), parameter :: names(7) = [ &
         'h1      ', 'he4     ', 'c12     ', 'n14     ', &
         'o16     ', 'ne20    ', 'mg24    ']
      integer :: k
      niso = 0
      do k = 1, 7
         cid(k) = chem_get_iso_id(names(k))
         niso(cid(k)) = k
      end do
   end subroutine setup_7sp_network

   subroutine setup_3sp_network(cid, niso)
      integer, pointer :: cid(:), niso(:)
      character(len=8), parameter :: names(3) = [ &
         'h1      ', 'he4     ', 'c12     ']
      integer :: k
      niso = 0
      do k = 1, 3
         cid(k) = chem_get_iso_id(names(k))
         niso(cid(k)) = k
      end do
   end subroutine setup_3sp_network

   subroutine make_xa(y_in, xc_in, xo_in, xa_out)
      real(dp), intent(in) :: y_in, xc_in, xo_in
      real(dp), intent(out) :: xa_out(7)
      xa_out = [0.0d0, y_in, xc_in, 0.0d0, xo_in, 0.0d0, 0.0d0]
   end subroutine make_xa

   subroutine make_solar_xa(x_in, zinit_in, xa_out)
      real(dp), intent(in) :: x_in, zinit_in
      real(dp), intent(out) :: xa_out(7)
      real(dp), parameter :: zfrac_c = 0.173312d0
      real(dp), parameter :: zfrac_n = 0.053177d0
      real(dp), parameter :: zfrac_o = 0.482398d0
      real(dp), parameter :: zfrac_ne = 0.098675d0
      xa_out = 0.0d0
      xa_out(1) = x_in
      xa_out(2) = 1.0d0 - (x_in + zinit_in)
      xa_out(3) = zinit_in * zfrac_c
      xa_out(4) = zinit_in * zfrac_n
      xa_out(5) = zinit_in * zfrac_o
      xa_out(6) = zinit_in * zfrac_ne
      xa_out(7) = 1.0d0 - sum(xa_out(1:6))
   end subroutine make_solar_xa

   subroutine emit_dfermi_case(label, dk_in, eta_in, theta_in)
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: dk_in, eta_in, theta_in
      real(dp) :: fd_v, fdeta_v, fdtheta_v
      call eos_fermi_dirac_integral(dk_in, eta_in, theta_in, fd_v, fdeta_v, fdtheta_v)
      write (out_unit, '(a,6(a,es25.17e3))') &
         'DFERMI' // achar(9) // trim(label), &
         achar(9), dk_in, achar(9), eta_in, achar(9), theta_in, &
         achar(9), fd_v, achar(9), fdeta_v, achar(9), fdtheta_v
   end subroutine emit_dfermi_case

   subroutine emit_pc_alfa_case(rq_in, label, lr, lt, z_val, ab, zb)
      type(EoS_General_Info), pointer :: rq_in
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: lr, lt, z_val, ab, zb
      real(dp) :: alfa, dt_a, dr_a
      integer :: ierr_loc
      call Get_PC_alfa(rq_in, lr, lt, z_val, ab, zb, alfa, dt_a, dr_a, ierr_loc)
      if (ierr_loc /= 0) stop 'Get_PC_alfa failed'
      write (out_unit, '(a,3(a,es25.17e3))') &
         'ALFA' // achar(9) // trim(label), &
         achar(9), alfa, achar(9), dt_a, achar(9), dr_a
   end subroutine emit_pc_alfa_case

   subroutine emit_skye_alfa_case(rq_in, label, lr, lt, z_val, ab, zb)
      type(EoS_General_Info), pointer :: rq_in
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: lr, lt, z_val, ab, zb
      real(dp) :: alfa, dt_a, dr_a
      integer :: ierr_loc
      call Get_Skye_alfa(rq_in, lr, lt, z_val, ab, zb, alfa, dt_a, dr_a, ierr_loc)
      if (ierr_loc /= 0) stop 'Get_Skye_alfa failed'
      write (out_unit, '(a,3(a,es25.17e3))') &
         'ALFA' // achar(9) // trim(label), &
         achar(9), alfa, achar(9), dt_a, achar(9), dr_a
   end subroutine emit_skye_alfa_case

   subroutine emit_skye_alfa_simple_case(rq_in, label, lr, lt, z_val, ab, zb)
      type(EoS_General_Info), pointer :: rq_in
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: lr, lt, z_val, ab, zb
      real(dp) :: alfa, dt_a, dr_a
      integer :: ierr_loc
      call Get_Skye_alfa_simple(rq_in, lr, lt, z_val, ab, zb, alfa, dt_a, dr_a, ierr_loc)
      if (ierr_loc /= 0) stop 'Get_Skye_alfa_simple failed'
      write (out_unit, '(a,3(a,es25.17e3))') &
         'ALFA' // achar(9) // trim(label), &
         achar(9), alfa, achar(9), dt_a, achar(9), dr_a
   end subroutine emit_skye_alfa_simple_case

   subroutine emit_helm_case(rq_in, label, temp, den, abar, zbar)
      type(EoS_General_Info), pointer :: rq_in
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: temp, den, abar, zbar
      real(dp) :: hres(num_helm_results)
      real(dp) :: res(num_eos_basic_results), dlnd(num_eos_basic_results)
      real(dp) :: dlnt(num_eos_basic_results), dabar(num_eos_basic_results)
      real(dp) :: dzbar(num_eos_basic_results)
      integer :: ierr_loc, j

      res = 0.0d0; dlnd = 0.0d0; dlnt = 0.0d0; dabar = 0.0d0; dzbar = 0.0d0
      call eos_get_helm_results( &
         0.0d0, abar, zbar, den, log10(den), temp, log10(temp), &
         rq_in% coulomb_temp_cut_HELM, rq_in% coulomb_den_cut_HELM, &
         .true., .true., hres, ierr_loc)
      if (ierr_loc /= 0) stop 'eos_get_helm_results failed'

      call eos_convert_helm_results( &
         hres, 0.0d0, 0.0d0, abar, zbar, den, temp, .true., &
         res, dlnd, dlnt, dabar, dzbar, ierr_loc)
      if (ierr_loc /= 0) stop 'eos_convert_helm_results failed'

      do j = 1, 16
         write (out_unit, '(a,a,i0,5(a,es25.17e3))') &
            'HELM' // achar(9) // trim(label), achar(9), j - 1, &
            achar(9), res(j), achar(9), dlnd(j), achar(9), dlnt(j), &
            achar(9), dabar(j), achar(9), dzbar(j)
      end do
   end subroutine emit_helm_case

   subroutine emit_helm_ierr_checks(rq_in)
      type(EoS_General_Info), pointer :: rq_in
      real(dp) :: hres(num_helm_results)
      real(dp) :: res(num_eos_basic_results), dlnd(num_eos_basic_results)
      real(dp) :: dlnt(num_eos_basic_results), dabar(num_eos_basic_results)
      real(dp) :: dzbar(num_eos_basic_results)
      logical :: off_table
      integer :: ierr_loc, off_int

      hres = 0.0d0
      call eos_get_helm_results( &
         0.0d0, 13.714285714285715d0, 6.8571428571428577d0, &
         exp10(6.5d0), 6.5d0, 1.0d6, 6.0d0, &
         rq_in% coulomb_temp_cut_HELM, rq_in% coulomb_den_cut_HELM, &
         .true., .true., hres, ierr_loc)
      if (ierr_loc == 0) then
         call eos_convert_helm_results( &
            hres, 0.0d0, 0.0d0, 13.714285714285715d0, 6.8571428571428577d0, &
            exp10(6.5d0), 1.0d6, .true., res, dlnd, dlnt, dabar, dzbar, ierr_loc)
      end if
      write (out_unit, '(a,a,i0,a,i0)') &
         'IERR' // achar(9) // 'crystallized_wd_1e6_6p5', achar(9), ierr_loc, achar(9), 0

      hres = 0.0d0
      call helmeos2_eval( &
         100.0d0, 2.0d0, 100.0d0, 2.0d0, 4.0d0, 2.0d0, &
         rq_in% coulomb_temp_cut_HELM, rq_in% coulomb_den_cut_HELM, &
         hres, .false., .true., .true., off_table, ierr_loc)
      off_int = merge(1, 0, off_table)
      write (out_unit, '(a,a,i0,a,i0)') &
         'IERR' // achar(9) // 'helmeos2_low_t_100_100', achar(9), ierr_loc, achar(9), off_int
   end subroutine emit_helm_ierr_checks

   subroutine emit_comp_case_n(eh, which_eos, label, nsp, cid, niso, xa_in, den, temp)
      integer, intent(in) :: eh, which_eos, nsp
      character(len=*), intent(in) :: label
      integer, pointer :: cid(:), niso(:)
      real(dp), intent(in) :: xa_in(nsp), den, temp
      real(dp) :: res(num_eos_basic_results), dlnd(num_eos_basic_results)
      real(dp) :: dlnt(num_eos_basic_results)
      real(dp), allocatable :: ddxa(:, :)
      integer :: ierr_loc, j

      allocate(ddxa(num_eos_d_dxa_results, nsp))
      res = 0.0d0; dlnd = 0.0d0; dlnt = 0.0d0
      call eosDT_get_component( &
         eh, which_eos, nsp, cid, niso, xa_in, &
         den, arg_not_provided, temp, arg_not_provided, &
         res, dlnd, dlnt, ddxa, ierr_loc)
      if (ierr_loc /= 0) stop 'eosDT_get_component failed'

      do j = 1, num_eos_basic_results
         write (out_unit, '(a,a,i0,3(a,es25.17e3))') &
            'EOS' // achar(9) // trim(label), achar(9), j - 1, &
            achar(9), res(j), achar(9), dlnd(j), achar(9), dlnt(j)
      end do
      deallocate(ddxa)
   end subroutine emit_comp_case_n

   subroutine emit_dt_case(eh, label, nsp, cid, niso, xa_in, den, temp)
      integer, intent(in) :: eh, nsp
      character(len=*), intent(in) :: label
      integer, pointer :: cid(:), niso(:)
      real(dp), intent(in) :: xa_in(nsp), den, temp
      real(dp) :: res(num_eos_basic_results), dlnd(num_eos_basic_results)
      real(dp) :: dlnt(num_eos_basic_results)
      real(dp), allocatable :: ddxa(:, :)
      integer :: ierr_loc, j

      allocate(ddxa(num_eos_d_dxa_results, nsp))
      res = 0.0d0; dlnd = 0.0d0; dlnt = 0.0d0
      call eosDT_get( &
         eh, nsp, cid, niso, xa_in, &
         den, arg_not_provided, temp, arg_not_provided, &
         res, dlnd, dlnt, ddxa, ierr_loc)
      if (ierr_loc /= 0) stop 'eosDT_get failed'

      do j = 1, num_eos_basic_results
         write (out_unit, '(a,a,i0,3(a,es25.17e3))') &
            'EOS' // achar(9) // trim(label), achar(9), j - 1, &
            achar(9), res(j), achar(9), dlnd(j), achar(9), dlnt(j)
      end do
      deallocate(ddxa)
   end subroutine emit_dt_case

   subroutine emit_inverse_cases(cid, niso, abar_c)
      integer, pointer :: cid(:), niso(:)
      real(dp), intent(in) :: abar_c
      integer :: eh, ierr_loc, eos_calls
      type(EoS_General_Info), pointer :: rq_loc
      real(dp) :: xa_c(7), xa_sol(7)
      real(dp) :: fwd_res(num_eos_basic_results), res(num_eos_basic_results)
      real(dp) :: dlnd(num_eos_basic_results), dlnt(num_eos_basic_results)
      real(dp) :: ddxa(num_eos_d_dxa_results, 7)
      real(dp) :: log_pgas, ln_e, e_val
      real(dp) :: rho_pt, log10_rho_pt, dlnr_dlnp, dlnr_dlnt
      real(dp) :: logt_res, logrho_res
      real(dp) :: t_gam, logt_gam, dt_de, dt_dd, dp_de, dp_dd

      eh = alloc_eos_handle(ierr_loc)
      call eos_ptr(eh, rq_loc, ierr_loc)
      rq_loc% use_Skye = .false.
      rq_loc% use_PC = .false.
      rq_loc% use_FreeEOS = .false.
      rq_loc% use_OPAL_SCVH = .false.
      rq_loc% use_CMS = .false.

      call make_xa(0.0d0, 1.0d0, 0.0d0, xa_c)
      call eosDT_get(eh, 7, cid, niso, xa_c, &
         1.0d4, arg_not_provided, 1.0d8, arg_not_provided, &
         fwd_res, dlnd, dlnt, ddxa, ierr_loc)
      if (ierr_loc /= 0) stop 'fwd eosDT_get failed'

      log_pgas = fwd_res(i_lnPgas) / ln10
      ln_e = fwd_res(i_lnE)
      e_val = exp(ln_e)

      call eosPT_get(eh, 7, cid, niso, xa_c, &
         arg_not_provided, log_pgas, arg_not_provided, 8.0d0, &
         rho_pt, log10_rho_pt, dlnr_dlnp, dlnr_dlnt, &
         res, dlnd, dlnt, ddxa, ierr_loc)
      if (ierr_loc /= 0) stop 'helm_c12_pt_4_8 failed'
      write (out_unit, '(a,6(a,es25.17e3))') &
         'INV' // achar(9) // 'helm_c12_pt_4_8', &
         achar(9), rho_pt, achar(9), log10_rho_pt, &
         achar(9), dlnr_dlnp, achar(9), dlnr_dlnt, &
         achar(9), res(i_lnPgas), achar(9), res(i_lnE)

      call eosDT_get_T(eh, 7, cid, niso, xa_c, &
         4.0d0, i_lnE, ln_e, 1.0d-12, 1.0d-12, 100, 7.5d0, &
         arg_not_provided, arg_not_provided, arg_not_provided, arg_not_provided, &
         logt_res, res, dlnd, dlnt, ddxa, eos_calls, ierr_loc)
      if (ierr_loc /= 0) stop 'helm_c12_get_t_4_8 failed'
      write (out_unit, '(a,6(a,es25.17e3))') &
         'INV' // achar(9) // 'helm_c12_get_t_4_8', &
         achar(9), logt_res, achar(9), res(i_lnPgas), &
         achar(9), res(i_lnE), achar(9), fwd_res(i_lnPgas), &
         achar(9), fwd_res(i_lnE), achar(9), real(eos_calls, dp)

      call eosDT_get_Rho(eh, 7, cid, niso, xa_c, &
         8.0d0, i_lnPgas, fwd_res(i_lnPgas), 1.0d-12, 1.0d-12, 100, 3.5d0, &
         arg_not_provided, arg_not_provided, arg_not_provided, arg_not_provided, &
         logrho_res, res, dlnd, dlnt, ddxa, eos_calls, ierr_loc)
      if (ierr_loc /= 0) stop 'helm_c12_get_rho_4_8 failed'
      write (out_unit, '(a,6(a,es25.17e3))') &
         'INV' // achar(9) // 'helm_c12_get_rho_4_8', &
         achar(9), logrho_res, achar(9), res(i_lnE), &
         achar(9), res(i_lnPgas), achar(9), fwd_res(i_lnE), &
         achar(9), fwd_res(i_lnPgas), achar(9), real(eos_calls, dp)

      call eos_gamma_DE_get(eh, abar_c, e_val, log10(e_val), 1.0d4, 4.0d0, 5.0d0 / 3.0d0, &
         t_gam, logt_gam, res, dlnd, dlnt, dt_de, dt_dd, dp_de, dp_dd, ierr_loc)
      if (ierr_loc /= 0) stop 'gamma_de_c12_1e4 failed'
      write (out_unit, '(a,6(a,es25.17e3))') &
         'INV' // achar(9) // 'gamma_de_c12_1e4', &
         achar(9), t_gam, achar(9), logt_gam, &
         achar(9), dt_de, achar(9), dt_dd, &
         achar(9), dp_de, achar(9), dp_dd
      write (out_unit, '(a,6(a,es25.17e3))') &
         'INV' // achar(9) // 'gamma_de_c12_1e4_res', &
         achar(9), res(i_mu), achar(9), res(i_lnfree_e), &
         achar(9), res(i_eta), achar(9), res(i_lnPgas), &
         achar(9), res(i_lnE), achar(9), 0.0d0
      call free_eos_handle(eh)

      ! Full default-cascade `eosPT_get_T` (`wave3b1_golden_tests.rs:462-480`)
      eh = alloc_eos_handle(ierr_loc)
      call make_solar_xa(0.70d0, 0.02d0, xa_sol)
      call eosPT_get(eh, 7, cid, niso, xa_sol, &
         arg_not_provided, 15.0d0, arg_not_provided, 7.2d0, &
         rho_pt, log10_rho_pt, dlnr_dlnp, dlnr_dlnt, &
         fwd_res, dlnd, dlnt, ddxa, ierr_loc)
      if (ierr_loc /= 0) stop 'solar_pt_15_7p2 failed'
      call eosPT_get_T(eh, 7, cid, niso, xa_sol, &
         15.0d0, i_lnE, fwd_res(i_lnE), 1.0d-6, ln10 * 1.0d-6, 100, 7.0d0, &
         arg_not_provided, arg_not_provided, arg_not_provided, arg_not_provided, &
         logt_res, rho_pt, log10_rho_pt, dlnr_dlnp, dlnr_dlnt, &
         res, dlnd, dlnt, ddxa, eos_calls, ierr_loc)
      if (ierr_loc /= 0) stop 'solar_pt_get_t_15_7p2 failed'
      write (out_unit, '(a,6(a,es25.17e3))') &
         'INV' // achar(9) // 'solar_pt_get_t_15_7p2', &
         achar(9), logt_res, achar(9), log10_rho_pt, &
         achar(9), res(i_lnE), achar(9), fwd_res(i_lnE), &
         achar(9), dlnr_dlnp, achar(9), dlnr_dlnt
      call free_eos_handle(eh)
   end subroutine emit_inverse_cases

end program gen_eos_oracles
