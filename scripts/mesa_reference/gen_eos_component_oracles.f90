! SPDX-License-Identifier: LGPL-3.0-or-later
! -----------------------------------------------------------------------------
! Program: gen_eos_component_oracles
!
! Generates `tests/fixtures/golden/eos/component_oracles_26_4_1.tsv` directly
! from compiled upstream Fortran MESA release 26.4.1 (`libeos.a`, `libchem.a`,
! `libconst.a`, `libmath.a`, `libutils.a`, `libinterp_2d.a`, `libnum.a`,
! `libauto_diff.a`) for audit finding T2-15 and Issue #141.
!
! Build & run (with MESA SDK 26.6.1 active and $MESA_DIR pointing to MESA 26.4.1):
!   gfortran -O2 -fopenmp -ffree-form -ffree-line-length-none \
!     -I"$MESA_DIR/include" scripts/mesa_reference/gen_eos_component_oracles.f90 \
!     "$MESA_DIR/lib/libeos.a" "$MESA_DIR/lib/libinterp_2d.a" \
!     "$MESA_DIR/lib/libnum.a" "$MESA_DIR/lib/libmtx.a" \
!     "$MESA_DIR/lib/libinterp_1d.a" "$MESA_DIR/lib/libauto_diff.a" \
!     "$MESA_DIR/lib/libchem.a" "$MESA_DIR/lib/libmath.a" \
!     "$MESA_DIR/lib/libutils.a" "$MESA_DIR/lib/libconst.a" \
!     `mesasdk_lapack_link` `mesasdk_blas_link` `mesasdk_crmath_link` \
!     `mesasdk_hdf5_link` -o gen_eos_component_oracles
!   ./gen_eos_component_oracles tests/fixtures/golden/eos/component_oracles_26_4_1.tsv
! -----------------------------------------------------------------------------
program gen_eos_component_oracles
   use const_def, only: dp, arg_not_provided
   use const_lib, only: const_init
   use math_lib, only: math_init, exp10, log10
   use chem_def, only: num_chem_isos
   use chem_lib, only: chem_init, chem_shutdown, chem_get_iso_id
   use eos_def, only: EoS_General_Info, num_eos_basic_results, &
      num_eos_d_dxa_results, num_helm_results, i_eos_PC, i_eos_Skye
   use eos_lib, only: eos_init, eos_shutdown, alloc_eos_handle, &
      free_eos_handle, eos_ptr, eos_get_helm_results, &
      eos_convert_helm_results, eosDT_get_component, eosDT_get
   implicit none

   character(len=512) :: mesa_dir_arg, out_path
   character(len=64) :: ver_str
   integer :: ierr, eos_handle, iunit, out_unit
   type(EoS_General_Info), pointer :: rq
   integer, pointer :: chem_id(:), net_iso(:)
   real(dp) :: xa(7)

   call get_environment_variable('MESA_DIR', mesa_dir_arg)
   if (len_trim(mesa_dir_arg) == 0) stop 'MESA_DIR must be set'
   call get_command_argument(1, out_path)
   if (len_trim(out_path) == 0) stop 'Usage: gen_eos_component_oracles <out.tsv>'

   call const_init(mesa_dir_arg, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'
   call eos_init(' ', .true., ierr)
   if (ierr /= 0) stop 'eos_init failed'

   allocate(chem_id(7), net_iso(num_chem_isos))
   call setup_7sp_network(chem_id, net_iso)

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

   write (out_unit, '(a)') '# FlexMESA upstream Fortran MESA EOS component oracle table'
   write (out_unit, '(a,a)') '# mesa_version_number=', trim(adjustl(ver_str))
   write (out_unit, '(a)') '# mesa_git_tag=26.4.1'
   write (out_unit, '(a)') '# mesa_git_commit=c8c202662922a22b3efefa824473ad615c623afd'
   write (out_unit, '(a)') '# compiler=' // compiler_version()
   write (out_unit, '(a)') '# columns_helm: case_id idx val d_dlnd d_dlnT d_dabar d_dzbar'
   write (out_unit, '(a)') '# columns_eos:  case_id idx val d_dlnd d_dlnT'

   ! ------------------------------------------------------------------
   ! 1. Five HELM 16-channel x 5-column cases (wave3a2_helm*_tests.rs)
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
   call free_eos_handle(eos_handle)

   ! ------------------------------------------------------------------
   ! 2. Seven PC component cases + PC/HELM blend (wave3a3_pc*_tests.rs)
   ! ------------------------------------------------------------------
   eos_handle = alloc_eos_handle(ierr)
   call eos_ptr(eos_handle, rq, ierr)

   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_comp_case(eos_handle, i_eos_PC, 'pc_c12_liq', &
      chem_id, net_iso, xa, 1.0d6, 1.0d8)

   call make_xa(0.0d0, 0.5d0, 0.5d0, xa)
   call emit_comp_case(eos_handle, i_eos_PC, 'pc_co_liq', &
      chem_id, net_iso, xa, 1.0d6, 1.0d7)

   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_comp_case(eos_handle, i_eos_PC, 'pc_c12_trans_eosfi', &
      chem_id, net_iso, xa, 1.0d6, 3.1d6)
   call emit_comp_case(eos_handle, i_eos_PC, 'pc_c12_trans_phase', &
      chem_id, net_iso, xa, 1.0d6, 2.05d6)
   call emit_comp_case(eos_handle, i_eos_PC, 'pc_c12_sol', &
      chem_id, net_iso, xa, 1.0d6, 1.0d6)

   call make_xa(0.0d0, 0.5d0, 0.5d0, xa)
   call emit_comp_case(eos_handle, i_eos_PC, 'pc_co_sol', &
      chem_id, net_iso, xa, 1.0d6, 1.0d6)

   call make_xa(1.0d0, 0.0d0, 0.0d0, xa)
   call emit_comp_case(eos_handle, i_eos_PC, 'pc_he4_qmelt', &
      chem_id, net_iso, xa, 1.0d8, 1.0d5)

   ! PC + HELM blend (use_Skye=F, use_PC=T, use_FreeEOS=F, use_OPAL_SCVH=F)
   rq% use_Skye = .false.
   rq% use_PC = .true.
   rq% use_FreeEOS = .false.
   rq% use_OPAL_SCVH = .false.
   rq% PC_use_Gamma_limit_instead_of_T = .true.
   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_dt_case(eos_handle, 'eosdt_pc_helm_blend', &
      chem_id, net_iso, xa, 10.0d0**2.9d0, 10.0d0**6.5d0)
   call free_eos_handle(eos_handle)

   ! ------------------------------------------------------------------
   ! 3. Six Skye component cases + Skye/HELM blend (wave3a3_skye_inverse_tests.rs)
   ! ------------------------------------------------------------------
   eos_handle = alloc_eos_handle(ierr)
   call eos_ptr(eos_handle, rq, ierr)

   rq% Skye_use_ion_offsets = .false.
   rq% Skye_solid_mixing_rule = 'Ogata'
   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_comp_case(eos_handle, i_eos_Skye, 'skye_c12_liq', &
      chem_id, net_iso, xa, 1.0d6, 1.0d8)

   rq% Skye_use_ion_offsets = .true.
   call emit_comp_case(eos_handle, i_eos_Skye, 'skye_c12_liq_ion', &
      chem_id, net_iso, xa, 1.0d6, 1.0d8)

   rq% Skye_use_ion_offsets = .false.
   call make_xa(0.0d0, 0.5d0, 0.5d0, xa)
   call emit_comp_case(eos_handle, i_eos_Skye, 'skye_co_liq', &
      chem_id, net_iso, xa, 1.0d6, 1.0d7)
   call emit_comp_case(eos_handle, i_eos_Skye, 'skye_co_sol_ogata', &
      chem_id, net_iso, xa, 1.0d6, 2.0d6)

   rq% Skye_solid_mixing_rule = 'PC'
   call emit_comp_case(eos_handle, i_eos_Skye, 'skye_co_sol_pc', &
      chem_id, net_iso, xa, 1.0d6, 2.0d6)

   rq% Skye_solid_mixing_rule = 'Ogata'
   call make_xa(0.0d0, 1.0d0, 0.0d0, xa)
   call emit_comp_case(eos_handle, i_eos_Skye, 'skye_c12_phase', &
      chem_id, net_iso, xa, 1.0d6, 2.65d6)

   rq% use_Skye = .true.
   rq% use_PC = .false.
   rq% use_FreeEOS = .false.
   rq% use_OPAL_SCVH = .false.
   rq% Skye_use_ion_offsets = .false.
   call emit_dt_case(eos_handle, 'eosdt_skye_helm_blend', &
      chem_id, net_iso, xa, exp10(3.45d0), 1.0d7)
   call free_eos_handle(eos_handle)

   close(out_unit)
   deallocate(chem_id, net_iso)
   call eos_shutdown()
   call chem_shutdown()

contains

   function compiler_version() result(v)
      use iso_fortran_env, only: cv => compiler_version
      character(len=:), allocatable :: v
      v = cv()
   end function compiler_version

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

   subroutine make_xa(y_in, xc_in, xo_in, xa_out)
      real(dp), intent(in) :: y_in, xc_in, xo_in
      real(dp), intent(out) :: xa_out(7)
      xa_out = [0.0d0, y_in, xc_in, 0.0d0, xo_in, 0.0d0, 0.0d0]
   end subroutine make_xa

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

   subroutine emit_comp_case(eh, which_eos, label, cid, niso, xa_in, den, temp)
      integer, intent(in) :: eh, which_eos
      character(len=*), intent(in) :: label
      integer, pointer :: cid(:), niso(:)
      real(dp), intent(in) :: xa_in(7), den, temp
      real(dp) :: res(num_eos_basic_results), dlnd(num_eos_basic_results)
      real(dp) :: dlnt(num_eos_basic_results)
      real(dp) :: ddxa(num_eos_d_dxa_results, 7)
      integer :: ierr_loc, j

      res = 0.0d0; dlnd = 0.0d0; dlnt = 0.0d0
      call eosDT_get_component( &
         eh, which_eos, 7, cid, niso, xa_in, &
         den, arg_not_provided, temp, arg_not_provided, &
         res, dlnd, dlnt, ddxa, ierr_loc)
      if (ierr_loc /= 0) stop 'eosDT_get_component failed'

      do j = 1, num_eos_basic_results
         write (out_unit, '(a,a,i0,3(a,es25.17e3))') &
            'EOS' // achar(9) // trim(label), achar(9), j - 1, &
            achar(9), res(j), achar(9), dlnd(j), achar(9), dlnt(j)
      end do
   end subroutine emit_comp_case

   subroutine emit_dt_case(eh, label, cid, niso, xa_in, den, temp)
      integer, intent(in) :: eh
      character(len=*), intent(in) :: label
      integer, pointer :: cid(:), niso(:)
      real(dp), intent(in) :: xa_in(7), den, temp
      real(dp) :: res(num_eos_basic_results), dlnd(num_eos_basic_results)
      real(dp) :: dlnt(num_eos_basic_results)
      real(dp) :: ddxa(num_eos_d_dxa_results, 7)
      integer :: ierr_loc, j

      res = 0.0d0; dlnd = 0.0d0; dlnt = 0.0d0
      call eosDT_get( &
         eh, 7, cid, niso, xa_in, &
         den, arg_not_provided, temp, arg_not_provided, &
         res, dlnd, dlnt, ddxa, ierr_loc)
      if (ierr_loc /= 0) stop 'eosDT_get failed'

      do j = 1, num_eos_basic_results
         write (out_unit, '(a,a,i0,3(a,es25.17e3))') &
            'EOS' // achar(9) // trim(label), achar(9), j - 1, &
            achar(9), res(j), achar(9), dlnd(j), achar(9), dlnt(j)
      end do
   end subroutine emit_dt_case

end program gen_eos_component_oracles
