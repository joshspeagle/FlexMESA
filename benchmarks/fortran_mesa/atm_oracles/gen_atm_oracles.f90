! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex + decimal surface atmosphere (`atm` + `F2-23` `atm_support`)
! reference vectors with `skip_partials = .false.` against compiled MESA 26.4.1.

module gen_atm_cb_mod
   use chem_def, only: num_chem_isos, ih1, ihe4, ic12, in14, io16, ine20, img24
   use const_def, only: dp, ln10
   use eos_def, only: num_eos_basic_results, num_eos_d_dxa_results, &
      i_lnfree_e, i_eta, i_chiRho, i_chiT
   use eos_lib, only: eosPT_get, radiation_pressure
   use kap_def, only: num_kap_fracs, Kap_General_Info, kap_lowT_Freedman11, kap_lowT_fa05_gs98
   use kap_lib, only: kap_get, kap_ptr, kap_setup_tables
   use math_lib, only: exp, log10
   implicit none

   integer, parameter :: N_SPEC = 7
   integer :: g_eos_handle = 0, g_kap_handle = 0
   integer, pointer :: g_chem_id(:) => null()
   integer, pointer :: g_net_iso(:) => null()
   real(dp) :: g_xa(N_SPEC)

contains

   subroutine set_cb_comp(x_h, z_met, use_freedman)
      real(dp), intent(in) :: x_h, z_met
      logical, intent(in) :: use_freedman
      type(Kap_General_Info), pointer :: rq
      integer :: i, ierr
      real(dp) :: y_he, xc, xn, xo

      if (.not. associated(g_chem_id)) allocate(g_chem_id(N_SPEC))
      if (.not. associated(g_net_iso)) allocate(g_net_iso(num_chem_isos))
      g_chem_id = [ih1, ihe4, ic12, in14, io16, ine20, img24]
      g_net_iso = 0
      do i = 1, N_SPEC
         g_net_iso(g_chem_id(i)) = i
      end do
      y_he = 1.0_dp - (x_h + z_met)
      xc = 3.2724592105263235e-3_dp
      xn = 9.5023842105263292e-4_dp
      xo = 8.8218000000000601e-3_dp
      g_xa = [x_h, y_he, xc, xn, xo, 0.0_dp, 0.0_dp]
      g_xa(N_SPEC) = 1.0_dp - sum(g_xa)

      ierr = 0
      call kap_ptr(g_kap_handle, rq, ierr)
      if (ierr /= 0) stop 'kap_ptr failed'
      rq%Zbase = z_met
      if (use_freedman) then
         rq%kap_lowT_option = kap_lowT_Freedman11
      else
         rq%kap_lowT_option = kap_lowT_fa05_gs98
      end if
      call kap_setup_tables(g_kap_handle, ierr)
      if (ierr /= 0) stop 'kap_setup_tables failed'
   end subroutine set_cb_comp

   subroutine eos_cb(lnP, lnT, lnRho, res, dres_dlnRho, dres_dlnT, ierr)
      real(dp), intent(in) :: lnP, lnT
      real(dp), intent(out) :: lnRho, res(:), dres_dlnRho(:), dres_dlnT(:)
      integer, intent(out) :: ierr
      real(dp) :: T, P, Prad, Pgas, logPgas, Rho, logRho, dlnRho_dlnPgas, dlnRho_dlnT
      real(dp) :: dres_dxa(num_eos_d_dxa_results, N_SPEC)

      T = exp(lnT)
      P = exp(lnP)
      Prad = radiation_pressure(T)
      Pgas = max(1.0e-99_dp, P - Prad)
      logPgas = log10(Pgas)
      call eosPT_get(g_eos_handle, N_SPEC, g_chem_id, g_net_iso, g_xa, &
         Pgas, logPgas, T, lnT / ln10, Rho, logRho, dlnRho_dlnPgas, dlnRho_dlnT, &
         res, dres_dlnRho, dres_dlnT, dres_dxa, ierr)
      lnRho = logRho * ln10
   end subroutine eos_cb

   subroutine kap_cb(lnRho, lnT, res, dres_dlnRho, dres_dlnT, kap, dlnkap_dlnRho, dlnkap_dlnT, ierr)
      real(dp), intent(in) :: lnRho, lnT, res(:), dres_dlnRho(:), dres_dlnT(:)
      real(dp), intent(out) :: kap, dlnkap_dlnRho, dlnkap_dlnT
      integer, intent(out) :: ierr
      real(dp) :: kap_fracs(num_kap_fracs), dlnkap_dxa(N_SPEC)

      call kap_get(g_kap_handle, N_SPEC, g_chem_id, g_net_iso, g_xa, &
         lnRho / ln10, lnT / ln10, res(i_lnfree_e), dres_dlnRho(i_lnfree_e), dres_dlnT(i_lnfree_e), &
         res(i_eta), dres_dlnRho(i_eta), dres_dlnT(i_eta), &
         kap_fracs, kap, dlnkap_dlnRho, dlnkap_dlnT, dlnkap_dxa, ierr)
   end subroutine kap_cb

end module gen_atm_cb_mod

program gen_atm_oracles
   use, intrinsic :: iso_fortran_env, only: int64
   use atm_def, only: ATM_T_TAU_EDDINGTON, ATM_T_TAU_SOLAR_HOPF, &
      ATM_T_TAU_KRISHNA_SWAMY, ATM_T_TAU_TRAMPEDACH_SOLAR, &
      ATM_TABLE_TAU_100, ATM_TABLE_TAU_10, ATM_TABLE_TAU_1, ATM_TABLE_TAU_1M1, &
      ATM_TABLE_PHOTOSPHERE, ATM_TABLE_WD_TAU_25, ATM_TABLE_DB_WD_TAU_25
   use atm_lib, only: atm_init, atm_Teff, atm_get_T_tau_base, atm_eval_T_tau_dq_dtau, &
      atm_eval_T_tau_uniform, atm_eval_T_tau_varying, atm_eval_table, &
      atm_get_table_alfa_beta, atm_get_table_base, atm_eval_irradiated
   use atm_t_tau_relations, only: eval_T_tau
   use atm_utils, only: eval_E2, eval_Paczynski_gradr
   use chem_lib, only: chem_init
   use const_def, only: dp, standard_cgrav, Msun, Rsun, Lsun, M_jupiter, pi, crad, clight
   use const_lib, only: const_init
   use eos_def, only: num_eos_basic_results, i_chiRho, i_chiT
   use eos_lib, only: eos_init, alloc_eos_handle, radiation_pressure
   use gen_atm_cb_mod, only: g_eos_handle, g_kap_handle, set_cb_comp, eos_cb, kap_cb
   use kap_lib, only: kap_init, alloc_kap_handle
   use math_lib, only: math_init, exp, exp10, safe_sqrt
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver
   integer :: ierr, u, row_id
   real(dp) :: cgrav

   call get_environment_variable('MESA_DIR', mesa_dir)
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   if (command_argument_count() >= 1) then
      call get_command_argument(1, out_path)
   else
      out_path = 'mesa_26_4_1_atm_oracles.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   cgrav = standard_cgrav
   call math_init()
   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'
   call eos_init(' ', .true., ierr)
   if (ierr /= 0) stop 'eos_init failed'
   g_eos_handle = alloc_eos_handle(ierr)
   if (ierr /= 0) stop 'alloc_eos_handle failed'
   call kap_init(.true., ' ', ierr)
   if (ierr /= 0) stop 'kap_init failed'
   g_kap_handle = alloc_kap_handle(ierr)
   if (ierr /= 0) stop 'alloc_kap_handle failed'
   call atm_init(.false., ierr)
   if (ierr /= 0) stop 'atm_init failed'
   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output file'

   write(u, '(A)') '# FlexMESA MESA 26.4.1 surface atmosphere (atm + F2-23 atm_support) oracle vectors (#161)'
   write(u, '(A)') '# mesa_version: ' // trim(mesa_ver)
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(sdk_ver)
   write(u, '(A)') '# math_backend: CRMATH'
   write(u, '(A)') 'row_id	section	case_label	in1_hex	in2_hex	in3_hex	in4_hex	in5_hex	out_name	raw_bits_hex	decimal'

   row_id = 0
   call emit_relations_and_e2(u, row_id, cgrav)
   call emit_f2_23_t_tau_atm(u, row_id, cgrav)
   call emit_tables_and_blending(u, row_id, cgrav)
   call emit_irradiated_cases(u, row_id, cgrav)

   close(u)
   write(*, '(A,I0,A)') 'Wrote ', row_id, ' atm oracle rows.'

contains

   subroutine read_ver(mdir, ver)
      character(len=*), intent(in) :: mdir
      character(len=*), intent(out) :: ver
      integer :: iu, io
      ver = 'unknown'
      open(newunit=iu, file=trim(mdir)//'/data/version_number', status='old', action='read', iostat=io)
      if (io == 0) then
         read(iu, '(A)', iostat=io) ver
         close(iu)
      end if
   end subroutine read_ver

   pure function to_hex(x) result(h)
      real(dp), intent(in) :: x
      character(len=18) :: h
      integer(int64) :: bits
      bits = transfer(x, 0_int64)
      write(h, '("0x", Z16.16)') bits
   end function to_hex

   subroutine write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, oname, val)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      character(len=*), intent(in) :: sec, lbl, oname
      real(dp), intent(in) :: x1, x2, x3, x4, x5, val
      row_id = row_id + 1
      write(u, '(I0,A,A,A,A,5(A,A),A,A,A,A,A,ES25.17E3)') &
         row_id, char(9), trim(sec), char(9), trim(lbl), &
         char(9), to_hex(x1), char(9), to_hex(x2), char(9), to_hex(x3), &
         char(9), to_hex(x4), char(9), to_hex(x5), &
         char(9), trim(oname), char(9), to_hex(val), char(9), val
   end subroutine write_row

   subroutine emit_atm11(u, row_id, sec, lbl, x1, x2, x3, x4, x5, &
         lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
         lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, kap)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      character(len=*), intent(in) :: sec, lbl
      real(dp), intent(in) :: x1, x2, x3, x4, x5
      real(dp), intent(in) :: lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap
      real(dp), intent(in) :: lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, kap
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'lnT', lnT)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'dlnT_dL', dlnT_dL)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'dlnT_dlnR', dlnT_dlnR)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'dlnT_dlnM', dlnT_dlnM)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'dlnT_dlnkap', dlnT_dlnkap)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'lnP', lnP)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'dlnP_dL', dlnP_dL)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'dlnP_dlnR', dlnP_dlnR)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'dlnP_dlnM', dlnP_dlnM)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'dlnP_dlnkap', dlnP_dlnkap)
      call write_row(u, row_id, sec, lbl, x1, x2, x3, x4, x5, 'kap', kap)
   end subroutine emit_atm11

   subroutine emit_relations_and_e2(u, row_id, cgrav)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      real(dp), intent(in) :: cgrav
      integer, parameter :: rel_ids(4) = [ &
         ATM_T_TAU_EDDINGTON, ATM_T_TAU_SOLAR_HOPF, &
         ATM_T_TAU_KRISHNA_SWAMY, ATM_T_TAU_TRAMPEDACH_SOLAR]
      character(len=24), parameter :: rel_names(4) = [ &
         character(len=24) :: 'Eddington', 'solar_Hopf', 'Krishna_Swamy', 'Trampedach_solar']
      real(dp), parameter :: taus(4) = [1.0e-3_dp, 0.10_dp, 2.0_dp / 3.0_dp, 25.0_dp]
      real(dp), parameter :: xs_e2(5) = [0.0_dp, 0.05_dp, 0.75_dp, 4.20_dp, 12.0_dp]
      integer :: r, k, ierr
      real(dp) :: tbase, lnt, dq, e2v, de2v, gradr

      do r = 1, 4
         call atm_get_T_tau_base(rel_ids(r), tbase, ierr)
         call write_row(u, row_id, 't_tau_rel', trim(rel_names(r)), dble(rel_ids(r)), 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 'tau_base', tbase)
         do k = 1, 4
            call eval_T_tau(rel_ids(r), taus(k), 5777.0_dp, lnt, ierr)
            call atm_eval_T_tau_dq_dtau(rel_ids(r), taus(k), dq, ierr)
            call write_row(u, row_id, 't_tau_rel', trim(rel_names(r)), dble(rel_ids(r)), taus(k), 5777.0_dp, 0.0_dp, 0.0_dp, 'lnT', lnt)
            call write_row(u, row_id, 't_tau_rel', trim(rel_names(r)), dble(rel_ids(r)), taus(k), 5777.0_dp, 0.0_dp, 0.0_dp, 'dq_dtau', dq)
         end do
      end do

      do k = 1, 5
         call eval_E2(xs_e2(k), e2v, de2v, ierr)
         call write_row(u, row_id, 'e2', 'eval_e2', xs_e2(k), 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 'e2', e2v)
         call write_row(u, row_id, 'e2', 'eval_e2', xs_e2(k), 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 'de2_dx', de2v)
      end do

      gradr = eval_Paczynski_gradr(5800.0_dp, 1.0e5_dp, 2.5e-7_dp, 0.15_dp, 0.35_dp, Lsun, Msun, Rsun, cgrav)
      call write_row(u, row_id, 'paczynski', 'optically_thin', 5800.0_dp, 1.0e5_dp, 2.5e-7_dp, 0.15_dp, 0.35_dp, 'gradr', gradr)
      gradr = eval_Paczynski_gradr(9500.0_dp, 3.0e5_dp, 4.0e-7_dp, 2.00_dp, 0.80_dp, Lsun, Msun, Rsun, cgrav)
      call write_row(u, row_id, 'paczynski', 'optically_thick', 9500.0_dp, 3.0e5_dp, 4.0e-7_dp, 2.00_dp, 0.80_dp, 'gradr', gradr)
   end subroutine emit_relations_and_e2

   subroutine run_one_t_tau_atm(u, row_id, lbl, rel_name, rel_id, op_mode, tau_surf, pextra, l, r, m, cgrav, kap_guess)
      integer, intent(in) :: u, rel_id
      integer, intent(inout) :: row_id
      character(len=*), intent(in) :: lbl, rel_name, op_mode
      real(dp), intent(in) :: tau_surf, pextra, l, r, m, cgrav, kap_guess
      real(dp) :: teff, kap, lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap
      real(dp) :: lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap
      real(dp) :: lnRho, res(num_eos_basic_results), dres_dlnRho(num_eos_basic_results), dres_dlnT(num_eos_basic_results)
      real(dp) :: kap_chk, dlnkap_dlnRho, dlnkap_dlnT, t_val, p_val, prad, pgas
      real(dp) :: dlnPgas_dlnP, dlnRho_dlnP_cT, dlnkap_dlnP_cT, dlnPgas_dlnT_cP, dlnRho_dlnT_cP, dlnkap_dlnT_cP
      integer :: ierr

      teff = atm_Teff(l, r)
      if (trim(op_mode) == 'fixed') then
         call atm_eval_T_tau_uniform(tau_surf, l, r, m, cgrav, kap_guess, pextra, rel_id, &
            eos_cb, kap_cb, 1.0e-6_dp, 0, .false., teff, kap, &
            lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
            lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, ierr)
         if (ierr /= 0) stop 'atm_eval_T_tau_uniform fixed failed'
      else if (trim(op_mode) == 'iterated') then
         call atm_eval_T_tau_uniform(tau_surf, l, r, m, cgrav, kap_guess, pextra, rel_id, &
            eos_cb, kap_cb, 1.0e-6_dp, 30, .false., teff, kap, &
            lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
            lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, ierr)
         if (ierr /= 0) stop 'atm_eval_T_tau_uniform iterated failed'
         ! Apply star/private/atm_support.f90:312-322 implicit opacity partial fixup
         call eos_cb(lnP, lnT, lnRho, res, dres_dlnRho, dres_dlnT, ierr)
         call kap_cb(lnRho, lnT, res, dres_dlnRho, dres_dlnT, kap_chk, dlnkap_dlnRho, dlnkap_dlnT, ierr)
         t_val = exp(lnT)
         p_val = exp(lnP)
         prad = radiation_pressure(t_val)
         pgas = max(1.0e-99_dp, p_val - prad)
         dlnPgas_dlnP = p_val / pgas
         dlnRho_dlnP_cT = dlnPgas_dlnP / res(i_chiRho)
         dlnkap_dlnP_cT = dlnkap_dlnRho * dlnRho_dlnP_cT
         dlnPgas_dlnT_cP = -4.0_dp * prad / pgas
         dlnRho_dlnT_cP = (dlnPgas_dlnT_cP - res(i_chiT)) / res(i_chiRho)
         dlnkap_dlnT_cP = dlnkap_dlnT + dlnkap_dlnRho * dlnRho_dlnT_cP
         dlnP_dL = dlnP_dL + dlnP_dlnkap * (dlnkap_dlnP_cT * dlnP_dL + dlnkap_dlnT_cP * dlnT_dL)
         dlnP_dlnR = dlnP_dlnR + dlnP_dlnkap * (dlnkap_dlnP_cT * dlnP_dlnR + dlnkap_dlnT_cP * dlnT_dlnR)
         dlnP_dlnM = dlnP_dlnM + dlnP_dlnkap * (dlnkap_dlnP_cT * dlnP_dlnM + dlnkap_dlnT_cP * dlnT_dlnM)
         dlnP_dlnkap = 0.0_dp
      else if (trim(op_mode) == 'varying') then
         call atm_eval_T_tau_varying(tau_surf, l, r, m, cgrav, rel_id, eos_cb, kap_cb, &
            1.0e-9_dp, 500, .false., teff, &
            lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
            lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, ierr)
         if (ierr /= 0) stop 'atm_eval_T_tau_varying failed'
         kap = 0.0_dp
      end if

      call emit_atm11(u, row_id, 't_tau_atm', trim(lbl)//':'//trim(rel_name)//':'//trim(op_mode), &
         tau_surf, pextra, l, r, m, &
         lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
         lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, kap)
   end subroutine run_one_t_tau_atm

   subroutine emit_f2_23_t_tau_atm(u, row_id, cgrav)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      real(dp), intent(in) :: cgrav
      real(dp) :: tb_edd, tb_hopf, tb_ks, tb_tr
      integer :: ierr

      call set_cb_comp(0.70_dp, 0.02_dp, .false.)
      call atm_get_T_tau_base(ATM_T_TAU_EDDINGTON, tb_edd, ierr)
      call atm_get_T_tau_base(ATM_T_TAU_SOLAR_HOPF, tb_hopf, ierr)
      call atm_get_T_tau_base(ATM_T_TAU_KRISHNA_SWAMY, tb_ks, ierr)
      call atm_get_T_tau_base(ATM_T_TAU_TRAMPEDACH_SOLAR, tb_tr, ierr)

      ! Fixed & iterated across Pextra_factor = {0.0, 1.0, 1.5} and relations
      call run_one_t_tau_atm(u, row_id, 'solar_pex0', 'Eddington', ATM_T_TAU_EDDINGTON, 'fixed', tb_edd, 0.0_dp, Lsun, Rsun, Msun, cgrav, 0.35_dp)
      call run_one_t_tau_atm(u, row_id, 'solar_pex1', 'Eddington', ATM_T_TAU_EDDINGTON, 'fixed', tb_edd, 1.0_dp, Lsun, Rsun, Msun, cgrav, 0.35_dp)
      call run_one_t_tau_atm(u, row_id, 'hot_pex1p5', 'Eddington', ATM_T_TAU_EDDINGTON, 'fixed', tb_edd, 1.5_dp, 2000.0_dp * Lsun, 8.0_dp * Rsun, 10.0_dp * Msun, cgrav, 0.34_dp)
      call run_one_t_tau_atm(u, row_id, 'solar_iter_pex0', 'Eddington', ATM_T_TAU_EDDINGTON, 'iterated', tb_edd, 0.0_dp, Lsun, Rsun, Msun, cgrav, 0.50_dp)
      call run_one_t_tau_atm(u, row_id, 'solar_iter_pex1', 'Eddington', ATM_T_TAU_EDDINGTON, 'iterated', tb_edd, 1.0_dp, Lsun, Rsun, Msun, cgrav, 0.50_dp)
      call run_one_t_tau_atm(u, row_id, 'solar_iter_pex1p5', 'solar_Hopf', ATM_T_TAU_SOLAR_HOPF, 'iterated', tb_hopf, 1.5_dp, Lsun, Rsun, Msun, cgrav, 0.50_dp)
      call run_one_t_tau_atm(u, row_id, 'tau10_iter_ks', 'Krishna_Swamy', ATM_T_TAU_KRISHNA_SWAMY, 'iterated', 10.0_dp, 1.0_dp, Lsun, Rsun, Msun, cgrav, 0.50_dp)

      ! Varying across all 4 relations
      call run_one_t_tau_atm(u, row_id, 'solar_var', 'Eddington', ATM_T_TAU_EDDINGTON, 'varying', tb_edd, 1.0_dp, Lsun, Rsun, Msun, cgrav, 0.40_dp)
      call run_one_t_tau_atm(u, row_id, 'solar_var', 'solar_Hopf', ATM_T_TAU_SOLAR_HOPF, 'varying', tb_hopf, 1.0_dp, Lsun, Rsun, Msun, cgrav, 0.40_dp)
      call run_one_t_tau_atm(u, row_id, 'solar_var', 'Krishna_Swamy', ATM_T_TAU_KRISHNA_SWAMY, 'varying', tb_ks, 1.0_dp, Lsun, Rsun, Msun, cgrav, 0.40_dp)
      call run_one_t_tau_atm(u, row_id, 'solar_var', 'Trampedach_solar', ATM_T_TAU_TRAMPEDACH_SOLAR, 'varying', tb_tr, 1.0_dp, Lsun, Rsun, Msun, cgrav, 0.40_dp)
   end subroutine emit_f2_23_t_tau_atm

   subroutine emit_tables_and_blending(u, row_id, cgrav)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      real(dp), intent(in) :: cgrav
      integer :: ierr
      real(dp) :: m, r, l, teff, alfa, beta, tbase
      real(dp) :: lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap
      real(dp) :: lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap

      ! WD_tau_25 and DB_WD_tau_25 on exact grid knots + partials
      m = 0.8_dp * Msun
      r = safe_sqrt(cgrav * m / exp10(7.1_dp))
      teff = 5000.0_dp
      l = pi * crad * clight * r * r * teff * teff * teff * teff
      call atm_eval_table(l, r, m, cgrav, ATM_TABLE_WD_TAU_25, 0.02_dp, .false., teff, &
         lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
         lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, ierr)
      call emit_atm11(u, row_id, 'table', 'WD_tau_25', l, r, m, 0.02_dp, teff, &
         lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
         lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, 0.0_dp)

      r = safe_sqrt(cgrav * m / exp10(8.0_dp))
      teff = 25000.0_dp
      l = pi * crad * clight * r * r * teff * teff * teff * teff
      call atm_eval_table(l, r, m, cgrav, ATM_TABLE_DB_WD_TAU_25, 0.02_dp, .false., teff, &
         lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
         lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, ierr)
      call emit_atm11(u, row_id, 'table', 'DB_WD_tau_25', l, r, m, 0.02_dp, teff, &
         lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
         lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, 0.0_dp)

      ! alfa/beta weights across interior, partial blend, and off-table states
      call atm_get_table_alfa_beta(Lsun, atm_Teff(Lsun, Rsun), Rsun, Msun, cgrav, ATM_TABLE_PHOTOSPHERE, alfa, beta, ierr)
      call atm_get_table_base(ATM_TABLE_PHOTOSPHERE, tbase, ierr)
      call write_row(u, row_id, 'table_ab', 'photosphere_in', Lsun, Rsun, Msun, 0.0_dp, 0.0_dp, 'alfa', alfa)
      call write_row(u, row_id, 'table_ab', 'photosphere_in', Lsun, Rsun, Msun, 0.0_dp, 0.0_dp, 'beta', beta)
      call write_row(u, row_id, 'table_ab', 'photosphere_in', Lsun, Rsun, Msun, 0.0_dp, 0.0_dp, 'tau_base', tbase)

      ! Partial margin blend on photosphere (high Teff ~ 49500 K near upper table edge)
      teff = 49500.0_dp
      r = 5.0_dp * Rsun
      m = 25.0_dp * Msun
      l = pi * crad * clight * r * r * teff * teff * teff * teff
      call atm_get_table_alfa_beta(l, teff, r, m, cgrav, ATM_TABLE_TAU_100, alfa, beta, ierr)
      call write_row(u, row_id, 'table_ab', 'tau_100_edge_blend', l, r, m, teff, 0.0_dp, 'alfa', alfa)
      call write_row(u, row_id, 'table_ab', 'tau_100_edge_blend', l, r, m, teff, 0.0_dp, 'beta', beta)

      ! Pure off-table (Teff = 75000 K > 50000 K) -> alfa = 1, beta = 0
      teff = 75000.0_dp
      l = pi * crad * clight * r * r * teff * teff * teff * teff
      call atm_get_table_alfa_beta(l, teff, r, m, cgrav, ATM_TABLE_TAU_100, alfa, beta, ierr)
      call write_row(u, row_id, 'table_ab', 'tau_100_off_table', l, r, m, teff, 0.0_dp, 'alfa', alfa)
      call write_row(u, row_id, 'table_ab', 'tau_100_off_table', l, r, m, teff, 0.0_dp, 'beta', beta)

      ! Full star/private/atm_support.f90:572-689 off-table blending on WD_tau_25:
      ! (a) Partial blend (Teff = 39500 K, logg = 7.1 -> 0 < alfa < 1, tau_base = table base)
      m = 0.8_dp * Msun
      r = safe_sqrt(cgrav * m / exp10(7.1_dp))
      call emit_table_blend_case(u, row_id, 'wd25_edge_blend', ATM_TABLE_WD_TAU_25, &
         ATM_T_TAU_EDDINGTON, 39500.0_dp, r, m, cgrav, 0.02_dp, 0.40_dp)
      ! (b) Pure off-table (Teff = 50000 K > 40000 K -> alfa = 1, beta = 0, tau_base = T_tau base)
      call emit_table_blend_case(u, row_id, 'wd25_pure_off_table', ATM_TABLE_WD_TAU_25, &
         ATM_T_TAU_SOLAR_HOPF, 50000.0_dp, r, m, cgrav, 0.02_dp, 0.40_dp)
   end subroutine emit_tables_and_blending

   subroutine emit_table_blend_case(u, row_id, lbl, tab_id, rel_id, teff_target, r, m, cgrav, z_met, kap_g)
      integer, intent(in) :: u, tab_id, rel_id
      integer, intent(inout) :: row_id
      character(len=*), intent(in) :: lbl
      real(dp), intent(in) :: teff_target, r, m, cgrav, z_met, kap_g
      real(dp) :: l, teff, alfa, beta, tau_base, kap_u
      real(dp) :: lnT_a, dlnT_dL_a, dlnT_dlnR_a, dlnT_dlnM_a, dlnT_dlnkap_a
      real(dp) :: lnP_a, dlnP_dL_a, dlnP_dlnR_a, dlnP_dlnM_a, dlnP_dlnkap_a
      real(dp) :: lnT_b, dlnT_dL_b, dlnT_dlnR_b, dlnT_dlnM_b, dlnT_dlnkap_b
      real(dp) :: lnP_b, dlnP_dL_b, dlnP_dlnR_b, dlnP_dlnM_b, dlnP_dlnkap_b
      real(dp) :: lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap
      real(dp) :: lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap
      integer :: ierr

      call set_cb_comp(0.70_dp, z_met, .false.)
      l = pi * crad * clight * r * r * teff_target * teff_target * teff_target * teff_target
      teff = atm_Teff(l, r)
      call atm_get_table_alfa_beta(l, teff, r, m, cgrav, tab_id, alfa, beta, ierr)
      if (ierr /= 0) stop 'atm_get_table_alfa_beta failed'
      ! Match star/private/atm_support.f90:531-567 tau_base selection before get_T_tau(:610)
      if (beta == 0.0_dp) then
         call atm_get_T_tau_base(rel_id, tau_base, ierr)
      else
         call atm_get_table_base(tab_id, tau_base, ierr)
      end if
      if (ierr /= 0) stop 'tau_base lookup failed'
      if (beta /= 0.0_dp) then
         call atm_eval_table(l, r, m, cgrav, tab_id, z_met, .false., teff, &
            lnT_b, dlnT_dL_b, dlnT_dlnR_b, dlnT_dlnM_b, dlnT_dlnkap_b, &
            lnP_b, dlnP_dL_b, dlnP_dlnR_b, dlnP_dlnM_b, dlnP_dlnkap_b, ierr)
         if (ierr /= 0) stop 'table eval failed'
      else
         lnT_b = 0.0_dp; dlnT_dL_b = 0.0_dp; dlnT_dlnR_b = 0.0_dp; dlnT_dlnM_b = 0.0_dp; dlnT_dlnkap_b = 0.0_dp
         lnP_b = 0.0_dp; dlnP_dL_b = 0.0_dp; dlnP_dlnR_b = 0.0_dp; dlnP_dlnM_b = 0.0_dp; dlnP_dlnkap_b = 0.0_dp
      end if
      if (alfa /= 0.0_dp) then
         call atm_eval_T_tau_uniform(tau_base, l, r, m, cgrav, kap_g, 1.0_dp, rel_id, &
            eos_cb, kap_cb, 1.0e-6_dp, 0, .false., teff, kap_u, &
            lnT_a, dlnT_dL_a, dlnT_dlnR_a, dlnT_dlnM_a, dlnT_dlnkap_a, &
            lnP_a, dlnP_dL_a, dlnP_dlnR_a, dlnP_dlnM_a, dlnP_dlnkap_a, ierr)
         if (ierr /= 0) stop 'off-table T_tau failed'
      else
         lnT_a = 0.0_dp; dlnT_dL_a = 0.0_dp; dlnT_dlnR_a = 0.0_dp; dlnT_dlnM_a = 0.0_dp; dlnT_dlnkap_a = 0.0_dp
         lnP_a = 0.0_dp; dlnP_dL_a = 0.0_dp; dlnP_dlnR_a = 0.0_dp; dlnP_dlnM_a = 0.0_dp; dlnP_dlnkap_a = 0.0_dp
      end if
      lnT = alfa * lnT_a + beta * lnT_b
      dlnT_dL = alfa * dlnT_dL_a + beta * dlnT_dL_b
      dlnT_dlnR = alfa * dlnT_dlnR_a + beta * dlnT_dlnR_b
      dlnT_dlnM = alfa * dlnT_dlnM_a + beta * dlnT_dlnM_b
      dlnT_dlnkap = alfa * dlnT_dlnkap_a + beta * dlnT_dlnkap_b
      lnP = alfa * lnP_a + beta * lnP_b
      dlnP_dL = alfa * dlnP_dL_a + beta * dlnP_dL_b
      dlnP_dlnR = alfa * dlnP_dlnR_a + beta * dlnP_dlnR_b
      dlnP_dlnM = alfa * dlnP_dlnM_a + beta * dlnP_dlnM_b
      dlnP_dlnkap = alfa * dlnP_dlnkap_a + beta * dlnP_dlnkap_b
      call emit_atm11(u, row_id, 'table_blend', lbl, l, r, m, z_met, kap_g, &
         lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
         lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, tau_base)
   end subroutine emit_table_blend_case

   subroutine emit_irradiated_cases(u, row_id, cgrav)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      real(dp), intent(in) :: cgrav
      real(dp) :: m, p_surf, kap_guess, r, t_int, l, teff, kap, tau
      real(dp) :: lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap
      integer :: ierr

      call set_cb_comp(0.70_dp, 0.01_dp, .true.)
      m = 1.5_dp * M_jupiter
      p_surf = 1.0e6_dp
      kap_guess = 1.5e-2_dp
      r = safe_sqrt(cgrav * m * 10.0_dp / (p_surf * kap_guess))
      t_int = 900.0_dp
      l = pi * crad * clight * r * r * t_int * t_int * t_int * t_int

      ! Case 1: gamma = 0 (uniform opacity irradiation) with partials
      call atm_eval_irradiated(l, r, m, cgrav, 1000.0_dp, p_surf, kap_guess, 4.0e-3_dp, 0.0_dp, &
         eos_cb, kap_cb, 1.0e-6_dp, 30, .false., teff, kap, tau, &
         lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, ierr)
      if (ierr /= 0) stop 'atm_eval_irradiated gamma=0 failed'
      call emit_irr7(u, row_id, 'hot_jup_gamma0', l, r, m, 1000.0_dp, 0.0_dp, &
         kap, tau, lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap)

      ! Case 2: gamma = 0.25 (E2 visible irradiation active) with partials
      call atm_eval_irradiated(l, r, m, cgrav, 1250.0_dp, p_surf, kap_guess, 4.0e-3_dp, 0.25_dp, &
         eos_cb, kap_cb, 1.0e-6_dp, 30, .false., teff, kap, tau, &
         lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, ierr)
      if (ierr /= 0) stop 'atm_eval_irradiated gamma=0.25 failed'
      call emit_irr7(u, row_id, 'hot_jup_gamma0p25', l, r, m, 1250.0_dp, 0.25_dp, &
         kap, tau, lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap)
   end subroutine emit_irradiated_cases

   subroutine emit_irr7(u, row_id, lbl, l, r, m, teq, gam, kap, tau, lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      character(len=*), intent(in) :: lbl
      real(dp), intent(in) :: l, r, m, teq, gam, kap, tau, lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap
      call write_row(u, row_id, 'irradiated', lbl, l, r, m, teq, gam, 'kap', kap)
      call write_row(u, row_id, 'irradiated', lbl, l, r, m, teq, gam, 'tau', tau)
      call write_row(u, row_id, 'irradiated', lbl, l, r, m, teq, gam, 'lnT', lnT)
      call write_row(u, row_id, 'irradiated', lbl, l, r, m, teq, gam, 'dlnT_dL', dlnT_dL)
      call write_row(u, row_id, 'irradiated', lbl, l, r, m, teq, gam, 'dlnT_dlnR', dlnT_dlnR)
      call write_row(u, row_id, 'irradiated', lbl, l, r, m, teq, gam, 'dlnT_dlnM', dlnT_dlnM)
      call write_row(u, row_id, 'irradiated', lbl, l, r, m, teq, gam, 'dlnT_dlnkap', dlnT_dlnkap)
   end subroutine emit_irr7

end program gen_atm_oracles
