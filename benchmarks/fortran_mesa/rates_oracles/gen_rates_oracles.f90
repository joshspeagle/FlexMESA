! SPDX-License-Identifier: LGPL-3.0-or-later
! Generate IEEE-754 hex + decimal nuclear reaction rates (`rates`) reference vectors
! against compiled MESA 26.4.1 (`librates.a` + `CRMATH`) for #161.

program gen_rates_oracles
   use, intrinsic :: iso_fortran_env, only: int64
   use auto_diff, only: auto_diff_real_2var_order1
   use chem_def, only: chem_isos, ih1, ihe4, ic12, in14, io16, ine20, img24, isi28, ineut
   use chem_lib, only: chem_init
   use const_def, only: dp, kev
   use const_lib, only: const_init
   use eval_coulomb, only: None, DGC1973, I1993, PCR2009, ThomasFermi, Itoh2002, &
      do_mui_coulomb, do_vs_coulomb
   use math_lib, only: math_init, exp10, pow
   use rates_def, only: Screen_Info, Coulomb_Info, T_Factors, &
      salpeter_screening, extended_screening, chugunov_screening, no_screening, &
      rattab_tlo, which_mui_coulomb, which_vs_coulomb, use_suzuki_tables, do_ecapture, &
      reaclib_rates, T9_weaklib_full_on, T9_weaklib_full_off, &
      T9_weaklib_full_on_hi_Z, T9_weaklib_full_off_hi_Z
   use rates_lib, only: rates_init, rates_warning_init, read_raw_rates_records, &
      screen_set_context, screen_pair, coulomb_set_context, &
      eval_tfactors, get_raw_rate, rates_reaction_id, eval_FL_epsnuc_3alf, &
      get_weak_rate_id, eval_weak_reaction_info, reaclib_indices_for_reaction
   use reaclib_eval, only: reaction_rates
   use screen5, only: screen5_init_AZ_info
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver
   integer :: ierr, u, row_id

   call get_environment_variable('MESA_DIR', mesa_dir)
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   if (command_argument_count() >= 1) then
      call get_command_argument(1, out_path)
   else
      out_path = 'mesa_26_4_1_rates_oracles.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'
   call rates_init('reactions.list', '', 'rate_tables', .true., .false., '', '', '', ierr)
   if (ierr /= 0) stop 'rates_init failed'
   call rates_warning_init(.false., 10.0_dp)
   call read_raw_rates_records(ierr)
   if (ierr /= 0) stop 'read_raw_rates_records failed'
   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output file'

   write(u, '(A)') '# FlexMESA MESA 26.4.1 nuclear reaction rates (rates) oracle vectors (#161)'
   write(u, '(A)') '# mesa_version: ' // trim(mesa_ver)
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(sdk_ver)
   write(u, '(A)') '# math_backend: CRMATH'
   write(u, '(A)') 'row_id	section	case_label	in1_hex	in2_hex	in3_hex	in4_hex	in5_hex	out_name	raw_bits_hex	decimal'

   row_id = 0
   call emit_screening_section(u, row_id)
   call emit_coulomb_section(u, row_id)
   call emit_reaclib_section(u, row_id)
   call emit_raw_and_pycno_section(u, row_id)
   call emit_weak_section(u, row_id)

   close(u)
   write(*, '(A,I0,A)') 'Wrote ', row_id, ' rates oracle rows.'

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

   subroutine emit_one_screen(u, row_id, lbl, sc_mode, lt, lr, zb, ab, z2b, i1, i2)
      integer, intent(in) :: u, sc_mode, i1, i2
      integer, intent(inout) :: row_id
      character(len=*), intent(in) :: lbl
      real(dp), intent(in) :: lt, lr, zb, ab, z2b
      type(Screen_Info) :: sc
      real(dp) :: t, d, a1, z1, a2, z2, zs13, zhat, zhat2, lzav, aznut, zs13inv
      real(dp) :: scor, scordt, scordd, y_dummy(1), z158_dummy(1)
      integer :: ierr

      t = exp10(lt)
      d = exp10(lr)
      a1 = chem_isos%W(i1)
      z1 = dble(chem_isos%Z(i1))
      a2 = chem_isos%W(i2)
      z2 = dble(chem_isos%Z(i2))
      y_dummy(1) = 1.0_dp / ab
      z158_dummy(1) = pow(zb, 1.58_dp)
      ierr = 0
      call screen5_init_AZ_info(zs13, zhat, zhat2, lzav, aznut, zs13inv, a1, z1, a2, z2, ierr)
      if (ierr /= 0) stop 'screen5_init_AZ_info failed'
      call screen_set_context(sc, t, d, lt, lr, zb, ab, z2b, sc_mode, 1, y_dummy, z158_dummy)
      call screen_pair(sc, a1, z1, a2, z2, sc_mode, zs13, zhat, zhat2, lzav, aznut, zs13inv, &
         rattab_tlo, scor, scordt, scordd, ierr)
      if (ierr /= 0) stop 'screen_pair failed'

      call write_row(u, row_id, 'screen', lbl, t, d, zb, ab, z2b, 'scor', scor)
      call write_row(u, row_id, 'screen', lbl, t, d, zb, ab, z2b, 'dscor_dt', scordt)
      call write_row(u, row_id, 'screen', lbl, t, d, zb, ab, z2b, 'dscor_dd', scordd)
   end subroutine emit_one_screen

   subroutine emit_screening_section(u, row_id)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      ! Weak, intermediate (xlgfac >= 0.77), strong, Chugunov degenerate, and neutral cases
      call emit_one_screen(u, row_id, 'salpeter:solar:h1_h1', salpeter_screening, &
         7.18_dp, 2.18_dp, 1.25_dp, 1.35_dp, 1.75_dp, ih1, ih1)
      call emit_one_screen(u, row_id, 'salpeter:he_burn:he4_c12', salpeter_screening, &
         8.10_dp, 4.20_dp, 2.10_dp, 4.20_dp, 4.60_dp, ihe4, ic12)
      call emit_one_screen(u, row_id, 'extended:weak:h1_n14', extended_screening, &
         7.45_dp, 1.90_dp, 1.30_dp, 1.45_dp, 1.95_dp, ih1, in14)
      call emit_one_screen(u, row_id, 'extended:interm:he4_c12', extended_screening, &
         7.85_dp, 5.60_dp, 3.00_dp, 6.00_dp, 10.50_dp, ihe4, ic12)
      call emit_one_screen(u, row_id, 'extended:xlgfac_hi:c12_c12', extended_screening, &
         8.20_dp, 8.80_dp, 6.50_dp, 13.0_dp, 43.0_dp, ic12, ic12)
      call emit_one_screen(u, row_id, 'extended:strong:o16_o16', extended_screening, &
         8.40_dp, 9.50_dp, 8.00_dp, 16.0_dp, 64.0_dp, io16, io16)
      call emit_one_screen(u, row_id, 'extended:neutral:neut_fe', extended_screening, &
         8.50_dp, 6.00_dp, 12.0_dp, 24.0_dp, 144.0_dp, ineut, isi28)
      call emit_one_screen(u, row_id, 'chugunov:interm:he4_c12', chugunov_screening, &
         7.90_dp, 6.20_dp, 4.00_dp, 8.00_dp, 18.0_dp, ihe4, ic12)
      call emit_one_screen(u, row_id, 'chugunov:strong:c12_c12', chugunov_screening, &
         8.10_dp, 9.20_dp, 6.857142857142857_dp, 13.714285714285714_dp, 48.0_dp, ic12, ic12)
      call emit_one_screen(u, row_id, 'chugunov:crust:mg24_mg24', chugunov_screening, &
         8.30_dp, 9.80_dp, 12.0_dp, 24.0_dp, 144.0_dp, img24, img24)
      call emit_one_screen(u, row_id, 'no_screening:c12_o16', no_screening, &
         8.50_dp, 7.00_dp, 7.00_dp, 14.0_dp, 50.0_dp, ic12, io16)
   end subroutine emit_screening_section

   subroutine emit_coulomb_section(u, row_id)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      type(Coulomb_Info), pointer :: cc
      type(auto_diff_real_2var_order1) :: mu1, mu2, vs
      integer :: k
      integer, parameter :: mui_opts(4) = [None, DGC1973, I1993, PCR2009]
      integer, parameter :: vs_opts(4) = [None, ThomasFermi, Itoh2002, Itoh2002]
      character(len=32), parameter :: lbls(4) = [ &
         character(len=32) :: 'none:none', 'dgc1973:thomasfermi', 'i1993:itoh2002', 'pcr2009:itoh2002']
      real(dp) :: lt, lr, t, d, zb, ab, z2b, z1, z2

      allocate(cc)
      lt = 8.45_dp
      lr = 9.35_dp
      t = exp10(lt)
      d = exp10(lr)
      zb = 10.909090909090908_dp
      ab = 21.818181818181817_dp
      z2b = 119.81818181818181_dp
      z1 = 12.0_dp
      z2 = 11.0_dp
      call coulomb_set_context(cc, t, d, lt, lr, zb, ab, z2b)

      do k = 1, 4
         which_mui_coulomb = mui_opts(k)
         which_vs_coulomb = vs_opts(k)
         mu1 = do_mui_coulomb(cc, z1)
         mu2 = do_mui_coulomb(cc, z2)
         vs = do_vs_coulomb(cc, z1)
         call write_row(u, row_id, 'coulomb', trim(lbls(k)), t, d, zb, ab, z1, 'mu1', mu1%val)
         call write_row(u, row_id, 'coulomb', trim(lbls(k)), t, d, zb, ab, z2, 'mu2', mu2%val)
         call write_row(u, row_id, 'coulomb', trim(lbls(k)), t, d, zb, ab, z1, 'dmu_ev', abs(mu1%val - mu2%val) * kev * t)
         call write_row(u, row_id, 'coulomb', trim(lbls(k)), t, d, zb, ab, z1, 'vs', vs%val)
      end do
      deallocate(cc)
   end subroutine emit_coulomb_section

   subroutine emit_reaclib_section(u, row_id)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      character(len=32), parameter :: handles(5) = [ &
         character(len=32) :: 'r_h2_pg_he3', 'r_c12_ag_o16', 'r_he4_he4_he4_to_c12', &
         'r_fe54_pg_co55', 'r_p31_pa_si28']
      real(dp), parameter :: t9s(4) = [0.05_dp, 0.25_dp, 2.50_dp, 11.50_dp]
      integer :: h, k, lo, hi, num_l, ierr
      real(dp) :: lam, dlam, rlam, drlam

      do h = 1, size(handles)
         call reaclib_indices_for_reaction(trim(handles(h)), reaclib_rates, lo, hi, ierr)
         if (ierr /= 0 .or. lo <= 0) stop 'reaclib_indices_for_reaction failed'
         num_l = hi - lo + 1
         do k = 1, size(t9s)
            call reaction_rates(num_l, lo, hi, t9s(k), reaclib_rates, reaclib_rates%nuclides, &
               .false., lam, dlam, rlam, drlam, ierr)
            if (ierr /= 0) stop 'reaction_rates failed'
            call write_row(u, row_id, 'reaclib', trim(handles(h)), t9s(k), 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 'lambda', lam)
            call write_row(u, row_id, 'reaclib', trim(handles(h)), t9s(k), 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 'dlambda_dlnT', dlam)
            call write_row(u, row_id, 'reaclib', trim(handles(h)), t9s(k), 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 'rlambda', rlam)
            call write_row(u, row_id, 'reaclib', trim(handles(h)), t9s(k), 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 'drlambda_dlnT', drlam)
         end do
      end do
   end subroutine emit_reaclib_section

   subroutine emit_raw_and_pycno_section(u, row_id)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      type(T_Factors), target :: tf_rec
      type(T_Factors), pointer :: tf
      character(len=32), parameter :: rxns(6) = [ &
         character(len=32) :: 'rpp_to_he3', 'rc12ap_to_o16', 'r_c12_to_he4_he4_he4', &
         'r1212', 'r1616', 'ro16gp_aux']
      real(dp), parameter :: logts(3) = [7.20_dp, 8.35_dp, 9.45_dp]
      integer :: r, k, ir, ierr
      real(dp) :: lt, temp, raw_r, eps, deps_dt, deps_dr

      tf => tf_rec
      do r = 1, size(rxns)
         ir = rates_reaction_id(trim(rxns(r)))
         if (ir <= 0) stop 'unknown reaction in emit_raw_and_pycno_section'
         do k = 1, size(logts)
            lt = logts(k)
            temp = exp10(lt)
            call eval_tfactors(tf, lt, temp)
            call get_raw_rate(ir, temp, tf, raw_r, ierr)
            if (ierr /= 0) stop 'get_raw_rate failed'
            call write_row(u, row_id, 'raw_rate', trim(rxns(r)), lt, temp, 0.0_dp, 0.0_dp, 0.0_dp, 'rate', raw_r)
         end do
      end do

      ! Pycnonuclear 3-alpha across strong-screening (U >= 1) and pycnonuclear (U < 1) regimes
      call eval_FL_epsnuc_3alf(1.0e7_dp, 1.0e10_dp, 1.0_dp, 2.0_dp, eps, deps_dt, deps_dr)
      call write_row(u, row_id, 'pycno', 'pycno_regime', 1.0e7_dp, 1.0e10_dp, 1.0_dp, 2.0_dp, 0.0_dp, 'eps_nuc', eps)
      call eval_FL_epsnuc_3alf(8.0e7_dp, 3.0e9_dp, 0.8_dp, 2.0_dp, eps, deps_dt, deps_dr)
      call write_row(u, row_id, 'pycno', 'strong_screen_regime', 8.0e7_dp, 3.0e9_dp, 0.8_dp, 2.0_dp, 0.0_dp, 'eps_nuc', eps)
   end subroutine emit_raw_and_pycno_section

   subroutine emit_weak_section(u, row_id)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      type(Coulomb_Info), pointer :: cc
      integer, pointer :: ids(:), rxn_ids(:)
      real(dp), pointer :: lam(:), dlam_dt(:), dlam_dr(:), q(:), dq_dt(:), dq_dr(:), qn(:), dqn_dt(:), dqn_dr(:)
      integer :: ierr
      real(dp) :: save_on, save_off, save_on_hi, save_off_hi

      nullify(cc)
      allocate(ids(1), rxn_ids(1), lam(1), dlam_dt(1), dlam_dr(1), &
         q(1), dq_dt(1), dq_dr(1), qn(1), dqn_dt(1), dqn_dr(1))
      rxn_ids(1) = 0
      do_ecapture = .false.
      save_on = T9_weaklib_full_on; save_off = T9_weaklib_full_off
      save_on_hi = T9_weaklib_full_on_hi_Z; save_off_hi = T9_weaklib_full_off_hi_Z
      T9_weaklib_full_on = 0.0_dp; T9_weaklib_full_off = 0.0_dp
      T9_weaklib_full_on_hi_Z = 0.0_dp; T9_weaklib_full_off_hi_Z = 0.0_dp

      ! 1. weaklib interior on mg24 -> na24 (use_suzuki_tables = .false.)
      ids(1) = get_weak_rate_id('mg24', 'na24')
      use_suzuki_tables = .false.
      call eval_weak_reaction_info(1, ids, rxn_ids, cc, 0.45_dp, 2.5e9_dp, &
         10.0_dp, 0.0_dp, 0.0_dp, lam, dlam_dt, dlam_dr, q, dq_dt, dq_dr, qn, dqn_dt, dqn_dr, ierr)
      call emit_weak9(u, row_id, 'weaklib:mg24_na24', 0.45_dp, 2.5e9_dp, 0.0_dp, &
         lam(1), dlam_dt(1), dlam_dr(1), q(1), dq_dt(1), dq_dr(1), qn(1), dqn_dt(1), dqn_dr(1))

      ! 2. Suzuki2016 interior on mg24 -> na24 (use_suzuki_tables = .true., lYeRho ~ 9.4 in bounds)
      use_suzuki_tables = .true.
      call eval_weak_reaction_info(1, ids, rxn_ids, cc, 0.45_dp, 2.5e9_dp, &
         10.0_dp, 0.0_dp, 0.0_dp, lam, dlam_dt, dlam_dr, q, dq_dt, dq_dr, qn, dqn_dt, dqn_dr, ierr)
      call emit_weak9(u, row_id, 'suzuki_in_bounds:mg24_na24', 0.45_dp, 2.5e9_dp, 1.0_dp, &
         lam(1), dlam_dt(1), dlam_dr(1), q(1), dq_dt(1), dq_dr(1), qn(1), dqn_dt(1), dqn_dr(1))

      ! 3. Suzuki2016 out-of-bounds fallback to weaklib on na24 -> mg24 (lYeRho = 3.0 < Suzuki min)
      ids(1) = get_weak_rate_id('na24', 'mg24')
      call eval_weak_reaction_info(1, ids, rxn_ids, cc, 0.90_dp, 1.0e3_dp, &
         0.0_dp, 0.0_dp, 0.0_dp, lam, dlam_dt, dlam_dr, q, dq_dt, dq_dr, qn, dqn_dt, dqn_dr, ierr)
      call emit_weak9(u, row_id, 'suzuki_fallback_low_yerho:na24_mg24', 0.90_dp, 1.0e3_dp, 1.0_dp, &
         lam(1), dlam_dt(1), dlam_dr(1), q(1), dq_dt(1), dq_dr(1), qn(1), dqn_dt(1), dqn_dr(1))

      ! 4. Fe-peak weaklib on fe56 -> mn56
      ids(1) = get_weak_rate_id('fe56', 'mn56')
      use_suzuki_tables = .false.
      call eval_weak_reaction_info(1, ids, rxn_ids, cc, 3.20_dp, 8.0e8_dp, &
         10.0_dp, 0.0_dp, 0.0_dp, lam, dlam_dt, dlam_dr, q, dq_dt, dq_dr, qn, dqn_dt, dqn_dr, ierr)
      call emit_weak9(u, row_id, 'weaklib:fe56_mn56', 3.20_dp, 8.0e8_dp, 0.0_dp, &
         lam(1), dlam_dt(1), dlam_dr(1), q(1), dq_dt(1), dq_dr(1), qn(1), dqn_dt(1), dqn_dr(1))

      T9_weaklib_full_on = save_on; T9_weaklib_full_off = save_off
      T9_weaklib_full_on_hi_Z = save_on_hi; T9_weaklib_full_off_hi_Z = save_off_hi
      deallocate(ids, rxn_ids, lam, dlam_dt, dlam_dr, q, dq_dt, dq_dr, qn, dqn_dt, dqn_dr)
   end subroutine emit_weak_section

   subroutine emit_weak9(u, row_id, lbl, t9, yerho, suz_flag, l, dl_dt, dl_dr, q, dq_dt, dq_dr, qn, dqn_dt, dqn_dr)
      integer, intent(in) :: u
      integer, intent(inout) :: row_id
      character(len=*), intent(in) :: lbl
      real(dp), intent(in) :: t9, yerho, suz_flag, l, dl_dt, dl_dr, q, dq_dt, dq_dr, qn, dqn_dt, dqn_dr
      call write_row(u, row_id, 'weak', lbl, t9, yerho, suz_flag, 0.0_dp, 0.0_dp, 'lambda', l)
      call write_row(u, row_id, 'weak', lbl, t9, yerho, suz_flag, 0.0_dp, 0.0_dp, 'dlambda_dlnT', dl_dt)
      call write_row(u, row_id, 'weak', lbl, t9, yerho, suz_flag, 0.0_dp, 0.0_dp, 'dlambda_dlnRho', dl_dr)
      call write_row(u, row_id, 'weak', lbl, t9, yerho, suz_flag, 0.0_dp, 0.0_dp, 'Q', q)
      call write_row(u, row_id, 'weak', lbl, t9, yerho, suz_flag, 0.0_dp, 0.0_dp, 'dQ_dlnT', dq_dt)
      call write_row(u, row_id, 'weak', lbl, t9, yerho, suz_flag, 0.0_dp, 0.0_dp, 'dQ_dlnRho', dq_dr)
      call write_row(u, row_id, 'weak', lbl, t9, yerho, suz_flag, 0.0_dp, 0.0_dp, 'Qneu', qn)
      call write_row(u, row_id, 'weak', lbl, t9, yerho, suz_flag, 0.0_dp, 0.0_dp, 'dQneu_dlnT', dqn_dt)
      call write_row(u, row_id, 'weak', lbl, t9, yerho, suz_flag, 0.0_dp, 0.0_dp, 'dQneu_dlnRho', dqn_dr)
   end subroutine emit_weak9

end program gen_rates_oracles
