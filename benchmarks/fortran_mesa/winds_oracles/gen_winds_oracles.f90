! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex reference vectors against compiled MESA 26.4.1
! (`libstar.a`) for `winds::set_mdot` (which calls the module-private
! `do_set_mdot`, `eval_super_eddington_wind`, `eval_rlo_wind` and the
! contained wind-scheme evaluators) at the operating points of
! `crates/flexmesa-star/tests/winds_159_parity_test.rs` (#159).
!
! `set_mdot` is the only public entry point of `winds`, so each case builds a
! `star_info` holding only the fields `do_set_mdot` reads, calls
! `set_mdot(s, L_phot, M_phot, T_phot, ierr)`, and records `s% mstar_dot`,
! `s% explicit_mstar_dot`, `s% super_eddington_wind_mdot` (the output of
! `eval_super_eddington_wind`) and `s% doing_rlo_wind` (whether
! `eval_rlo_wind` returned a nonzero rate). Individual wind schemes are
! exercised by selecting them as the hot wind with the hot/cool switch
! temperatures below the photospheric temperature, so that
! `wind = scheme(L, M, R, T, ...)*scaling` is the only contribution.
program gen_winds_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use math_lib, only: MATH_BACKEND, math_init
   use const_def, only: dp, pi, crad, clight, standard_cgrav, msun, rsun, lsun, secyer
   use chem_def, only: ih1, ihe4, num_chem_isos
   use chem_lib, only: chem_get_iso_id
   use star_def, only: star_info, star_ptr
   use star_lib, only: alloc_star, free_star, starlib_init, star_setup, &
      star_set_kap_and_eos_handles, star_set_colors_handles
   use star_job_ctrls_io, only: set_default_star_job_controls, store_star_job_controls
   use alloc, only: allocate_star_info_arrays
   use winds, only: set_mdot
   use report, only: get_core_info
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver, fflags_str
   integer :: ierr, u, id, rc
   type(star_info), pointer :: s
   logical :: arrays_ready = .false.

   call get_environment_variable('MESA_DIR', mesa_dir)
   if (len_trim(mesa_dir) == 0) stop 'MESA_DIR must be set'
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   call get_environment_variable('ORACLE_FFLAGS', fflags_str)
   if (command_argument_count() >= 1) then
      call get_command_argument(1, out_path)
   else
      out_path = 'mesa_26_4_1_winds_oracles.tsv'
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
   if (ierr /= 0) stop 'failed to open output TSV'
   write(u, '(A)') '# FlexMESA MESA 26.4.1 winds set_mdot oracle vectors (#159)'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') 'case_id' // char(9) // 'role' // char(9) // 'field' // char(9) // &
      'part' // char(9) // 'raw_bits_hex' // char(9) // 'decimal'

   rc = 0
   call emit_all_cases()
   close(u)
   write(*, '(A,I0,A)') 'Wrote ', rc, ' winds oracle rows.'
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

   real(dp) function hexd(bits)
      integer(int64), intent(in) :: bits
      hexd = transfer(bits, 0d0)
   end function hexd

   subroutine emit(cid, role, fld, val)
      character(len=*), intent(in) :: cid, role, fld
      real(dp), intent(in) :: val
      character(len=20) :: hbuf
      character(len=32) :: dbuf
      write(hbuf, '(A,Z16.16)') '0x', transfer(val, 0_int64)
      write(dbuf, '(ES25.17E3)') val
      write(u, '(A)') trim(cid) // char(9) // trim(role) // char(9) // trim(fld) // &
         char(9) // 'val' // char(9) // trim(hbuf) // char(9) // trim(adjustl(dbuf))
      rc = rc + 1
   end subroutine emit

   subroutine emit_str(cid, fld, str)
      character(len=*), intent(in) :: cid, fld, str
      write(u, '(A)') trim(cid) // char(9) // 'in_str' // char(9) // trim(fld) // &
         char(9) // 'str' // char(9) // '-' // char(9) // '"' // trim(str) // '"'
      rc = rc + 1
   end subroutine emit_str

   subroutine emit_bool(cid, role, fld, b)
      character(len=*), intent(in) :: cid, role, fld
      logical, intent(in) :: b
      real(dp) :: v
      v = 0d0
      if (b) v = 1d0
      call emit(cid, role, fld, v)
   end subroutine emit_bool

   ! Reset the star to the state `make_wind_model` in the Rust test builds
   ! (every control it sets, plus the defaults the test leaves untouched).
   subroutine base_model(nz, l_phot, m_phot, t_phot, z_base, surf_h1, surf_he4, cen_h1, cen_he4)
      integer, intent(in) :: nz
      real(dp), intent(in) :: l_phot, m_phot, t_phot, z_base, surf_h1, surf_he4, cen_h1, cen_he4
      real(dp) :: r_bb
      integer :: k
      s%nz = nz
      s%species = 2
      s%nvar_hydro = 4
      s%nvar_chem = 2
      s%nvar_total = 6
      s%num_reactions = 0
      if (.not. arrays_ready) then
         s%nz = 3
         s%prev_mesh_nz = 3
         s%zero_when_allocate = .true.
         call allocate_star_info_arrays(s, ierr)
         if (ierr /= 0) stop 'allocate_star_info_arrays failed'
         arrays_ready = .true.
         s%nz = nz
      end if
      s%xa(:, :) = 0d0
      s%L(:) = 0d0
      if (.not. associated(s%chem_id)) allocate(s%chem_id(2))
      if (.not. associated(s%net_iso)) allocate(s%net_iso(num_chem_isos))
      s%net_iso(:) = 0
      s%chem_id(1) = chem_get_iso_id('h1')
      s%chem_id(2) = chem_get_iso_id('he4')
      s%net_iso(ih1) = 1
      s%net_iso(ihe4) = 2

      s%use_other_wind = .false.
      s%use_other_adjust_mdot = .false.
      s%report_ierr = .false.
      s%stop_for_bad_nums = .false.
      s%trace_dt_control_mass_change = .false.
      s%trace_super_eddington_wind_boost = .false.
      s%terminal_interval = 1000000
      s%model_number = 1
      s%doing_rlo_wind = .false.
      s%rotation_flag = .false.
      s%mdot_omega_power = 0d0

      s%kap_rq%Zbase = z_base
      s%dt = 1d10
      s%mass_change = 0d0
      s%cool_wind_full_on_T = 8d3
      s%hot_wind_full_on_T = 1.2d4
      s%hot_wind_scheme = ''
      s%cool_wind_RGB_scheme = ''
      s%cool_wind_AGB_scheme = ''
      s%RGB_to_AGB_wind_switch = 0d0
      s%Reimers_scaling_factor = 0d0
      s%Blocker_scaling_factor = 0d0
      s%Vink_scaling_factor = 0d0
      s%Dutch_scaling_factor = 0d0
      s%de_Jager_scaling_factor = 0d0
      s%van_Loon_scaling_factor = 0d0
      s%Nieuwenhuijzen_scaling_factor = 0d0
      s%Bjorklund_scaling_factor = 0d0
      s%Dutch_wind_lowT_scheme = 'de Jager'
      s%super_eddington_scaling_factor = 0d0
      s%super_eddington_wind_Ledd_factor = 1d0
      s%prev_Ledd = 1d40
      s%wind_boost_full_off_L_div_Ledd = 1d0
      s%wind_boost_full_on_L_div_Ledd = 2d0
      s%super_eddington_wind_max_boost = 1d0
      s%rlo_scaling_factor = 0d0
      s%rlo_wind_min_L = 0d0
      s%rlo_wind_max_Teff = 1d99
      s%rlo_wind_scale_height = 0d0
      s%rlo_wind_roche_lobe_radius = 1d99
      s%rlo_wind_base_mdot = 0d0
      s%roche_lobe_xfer_full_on = 0d0
      s%roche_lobe_xfer_full_off = 1d0
      s%no_wind_if_no_rotation = .false.
      s%mass_change_full_on_dt = 0d0
      s%mass_change_full_off_dt = 0d0
      s%min_wind = 0d0
      s%max_wind = 0d0
      s%starting_T_center = 1d7
      s%max_T_center_for_any_mass_loss = 1d99
      s%max_T_center_for_full_mass_loss = 1d99
      s%wind_H_envelope_limit = 0d0
      s%wind_H_He_envelope_limit = 0d0
      s%wind_He_layer_limit = 0d0
      s%star_mass = m_phot/msun
      s%he_core_mass = 0d0
      s%co_core_mass = 0d0
      s%mstar = m_phot
      s%xmstar = m_phot
      s%M_center = 0d0
      s%mstar_dot = 0d0

      r_bb = sqrt(l_phot/(pi*crad*clight*t_phot*t_phot*t_phot*t_phot))
      s%L(1:nz) = l_phot
      s%m(1) = m_phot
      s%r(1) = r_bb
      s%cgrav(1:nz) = standard_cgrav
      s%rho(1) = 1d-9
      s%Peos(1) = 1d3
      s%xa(:, :) = 0d0
      s%xa(1, 1) = surf_h1
      s%xa(2, 1) = surf_he4
      s%xa(1, nz) = cen_h1
      s%xa(2, nz) = cen_he4
      if (nz == 2) then
         s%m(2) = 0.5d0*m_phot
         s%r(2) = 0.5d0*r_bb
         s%rho(2) = 1d0
         s%Peos(2) = 1d15
      end if
   end subroutine base_model

   ! Record every input set above, call set_mdot, and record the outputs.
   subroutine run_case(cid, l_phot, m_phot, t_phot)
      character(len=*), intent(in) :: cid
      real(dp), intent(in) :: l_phot, m_phot, t_phot
      real(dp) :: l_star
      integer :: ierr_c
      s%mstar_dot = 0d0
      s%explicit_mstar_dot = 0d0
      s%super_eddington_wind_mdot = 0d0
      call emit(cid, 'in', 'L_phot', l_phot)
      call emit(cid, 'in', 'M_phot', m_phot)
      call emit(cid, 'in', 'T_phot', t_phot)
      call emit(cid, 'in', 'Zbase', s%kap_rq%Zbase)
      call emit(cid, 'in', 'surface_h1', s%xa(1, 1))
      call emit(cid, 'in', 'surface_he4', s%xa(2, 1))
      call emit(cid, 'in', 'center_h1', s%xa(1, s%nz))
      call emit(cid, 'in', 'center_he4', s%xa(2, s%nz))
      call emit(cid, 'in', 'dt', s%dt)
      call emit(cid, 'in', 'mass_change', s%mass_change)
      call emit(cid, 'in', 'cool_wind_full_on_T', s%cool_wind_full_on_T)
      call emit(cid, 'in', 'hot_wind_full_on_T', s%hot_wind_full_on_T)
      call emit_str(cid, 'hot_wind_scheme', s%hot_wind_scheme)
      call emit_str(cid, 'cool_wind_RGB_scheme', s%cool_wind_RGB_scheme)
      call emit_str(cid, 'cool_wind_AGB_scheme', s%cool_wind_AGB_scheme)
      call emit_str(cid, 'Dutch_wind_lowT_scheme', s%Dutch_wind_lowT_scheme)
      call emit(cid, 'in', 'RGB_to_AGB_wind_switch', s%RGB_to_AGB_wind_switch)
      call emit(cid, 'in', 'Reimers_scaling_factor', s%Reimers_scaling_factor)
      call emit(cid, 'in', 'Blocker_scaling_factor', s%Blocker_scaling_factor)
      call emit(cid, 'in', 'Vink_scaling_factor', s%Vink_scaling_factor)
      call emit(cid, 'in', 'Dutch_scaling_factor', s%Dutch_scaling_factor)
      call emit(cid, 'in', 'de_Jager_scaling_factor', s%de_Jager_scaling_factor)
      call emit(cid, 'in', 'van_Loon_scaling_factor', s%van_Loon_scaling_factor)
      call emit(cid, 'in', 'Nieuwenhuijzen_scaling_factor', s%Nieuwenhuijzen_scaling_factor)
      call emit(cid, 'in', 'Bjorklund_scaling_factor', s%Bjorklund_scaling_factor)
      call emit(cid, 'in', 'super_eddington_scaling_factor', s%super_eddington_scaling_factor)
      call emit(cid, 'in', 'super_eddington_wind_Ledd_factor', s%super_eddington_wind_Ledd_factor)
      call emit(cid, 'in', 'prev_Ledd', s%prev_Ledd)
      call emit(cid, 'in', 'wind_boost_full_off_L_div_Ledd', s%wind_boost_full_off_L_div_Ledd)
      call emit(cid, 'in', 'wind_boost_full_on_L_div_Ledd', s%wind_boost_full_on_L_div_Ledd)
      call emit(cid, 'in', 'super_eddington_wind_max_boost', s%super_eddington_wind_max_boost)
      call emit(cid, 'in', 'rlo_scaling_factor', s%rlo_scaling_factor)
      call emit(cid, 'in', 'rlo_wind_min_L', s%rlo_wind_min_L)
      call emit(cid, 'in', 'rlo_wind_max_Teff', s%rlo_wind_max_Teff)
      call emit(cid, 'in', 'rlo_wind_scale_height', s%rlo_wind_scale_height)
      call emit(cid, 'in', 'rlo_wind_roche_lobe_radius', s%rlo_wind_roche_lobe_radius)
      call emit(cid, 'in', 'rlo_wind_base_mdot', s%rlo_wind_base_mdot)
      call emit(cid, 'in', 'roche_lobe_xfer_full_on', s%roche_lobe_xfer_full_on)
      call emit(cid, 'in', 'roche_lobe_xfer_full_off', s%roche_lobe_xfer_full_off)
      call emit_bool(cid, 'in', 'no_wind_if_no_rotation', s%no_wind_if_no_rotation)
      call emit_bool(cid, 'in', 'rotation_flag', s%rotation_flag)
      call emit(cid, 'in', 'mass_change_full_on_dt', s%mass_change_full_on_dt)
      call emit(cid, 'in', 'mass_change_full_off_dt', s%mass_change_full_off_dt)
      call emit(cid, 'in', 'min_wind', s%min_wind)
      call emit(cid, 'in', 'max_wind', s%max_wind)
      call emit(cid, 'in', 'starting_T_center', s%starting_T_center)
      call emit(cid, 'in', 'max_T_center_for_any_mass_loss', s%max_T_center_for_any_mass_loss)
      call emit(cid, 'in', 'max_T_center_for_full_mass_loss', s%max_T_center_for_full_mass_loss)
      call emit(cid, 'in', 'wind_H_envelope_limit', s%wind_H_envelope_limit)
      call emit(cid, 'in', 'wind_H_He_envelope_limit', s%wind_H_He_envelope_limit)
      call emit(cid, 'in', 'wind_He_layer_limit', s%wind_He_layer_limit)
      call emit(cid, 'in', 'star_mass', s%star_mass)
      call emit(cid, 'in', 'he_core_mass', s%he_core_mass)
      call emit(cid, 'in', 'co_core_mass', s%co_core_mass)
      call emit(cid, 'in', 'r1', s%r(1))
      call emit(cid, 'in', 'm1', s%m(1))
      call emit(cid, 'in', 'cgrav1', s%cgrav(1))
      call emit(cid, 'in', 'rho1', s%rho(1))
      call emit(cid, 'in', 'Peos1', s%Peos(1))

      ! `star/private/evolve.f90:427` calls `set_mdot(s, s% L_phot*Lsun, ...)`
      ! where `s% L_phot` is stored in Lsun units (`L_erg / Lsun`) (#679).
      s%L_phot = l_phot / lsun
      l_star = s%L_phot * lsun
      ierr_c = 0
      call set_mdot(s, l_star, m_phot, t_phot, ierr_c)
      call emit(cid, 'out', 'ierr', real(ierr_c, dp))
      call emit(cid, 'out', 'mstar_dot', s%mstar_dot)
      call emit(cid, 'out', 'explicit_mstar_dot', s%explicit_mstar_dot)
      call emit(cid, 'out', 'super_eddington_wind_mdot', s%super_eddington_wind_mdot)
      call emit_bool(cid, 'out', 'doing_rlo_wind', s%doing_rlo_wind)
   end subroutine run_case

   ! Single-scheme case: scheme selected as the hot wind with both switch
   ! temperatures far below T so wind = scheme*scaling is the only term.
   subroutine scheme_case(cid, scheme, l, m, t, zb, sh1, she4, ch1, che4)
      character(len=*), intent(in) :: cid, scheme
      real(dp), intent(in) :: l, m, t, zb, sh1, she4, ch1, che4
      call base_model(2, l, m, t, zb, sh1, she4, ch1, che4)
      s%cool_wind_full_on_T = 1d3
      s%hot_wind_full_on_T = 1d3
      s%hot_wind_scheme = scheme
   end subroutine scheme_case

   real(dp) function r_of(l, t)
      real(dp), intent(in) :: l, t
      r_of = sqrt(l/(pi*crad*clight*t*t*t*t))
   end function r_of

   subroutine emit_all_cases()
      real(dp) :: l1, m30, m25, m15, m20, m5, m7, t25k, t4k, t10500, t15k, t30k
      real(dp) :: l_bj, m_bj, t_bj, l_se, l_bst, l_rlo, l_xfer, l_env, t_env
      real(dp) :: lc, mc, tc
      integer :: k

      ! ---- test_mesa_26_4_1_target_cases_step1_and_blends_zero_ulp ----
      ! 1. star/hot_cool_wind step 1.
      lc = hexd(int(z'4795BD6A8B756743', int64)); mc = hexd(int(z'470572069D3FA0C4', int64))
      tc = hexd(int(z'40D4769F25A30032', int64))
      call base_model(2, lc, mc, tc, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      call hot_cool_controls()
      call run_case('hot_cool_wind_step1', lc, mc, tc)
      ! 2. hot/cool blend midpoint T = 10000 K.
      lc = hexd(int(z'47D932E7005377AA', int64)); mc = hexd(int(z'47056A2ED5580A3B', int64))
      tc = hexd(int(z'40C3880000000000', int64))
      call base_model(2, lc, mc, tc, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      call hot_cool_controls()
      call run_case('hot_cool_wind_blend', lc, mc, tc)
      ! 3. star/high_z Zbase = 0.07 (controls of case 1, kap Zbase = 0.07).
      lc = hexd(int(z'47A118695D5CED92', int64)); mc = hexd(int(z'470572069D3FA0C4', int64))
      tc = hexd(int(z'40D202C000000000', int64))
      call base_model(2, lc, mc, tc, 0.07d0, 0.55d0, 0.38d0, 0.55d0, 0.38d0)
      call hot_cool_controls()
      call run_case('high_z_z07', lc, mc, tc)
      ! 4. star/7M_prems_to_AGB pre-MS Reimers.
      lc = hexd(int(z'479A1614A8ED5870', int64)); mc = hexd(int(z'470572069D3FA0C4', int64))
      tc = hexd(int(z'40B2809431B74921', int64))
      call base_model(2, lc, mc, tc, 0.001d0, 0.757d0, 0.242d0, 0.757d0, 0.242d0)
      call prems_controls()
      call run_case('prems_7m_reimers', lc, mc, tc)
      ! 5. AGB Blocker switch.
      lc = hexd(int(z'47CCCC75B73AD1E7', int64)); mc = hexd(int(z'470533485E02EC76', int64))
      tc = hexd(int(z'40B0040000000000', int64))
      call base_model(2, lc, mc, tc, 0.001d0, 0.757d0, 0.242d0, 0.0d0, 1d-5)
      call prems_controls()
      call run_case('agb_7m_blocker', lc, mc, tc)

      ! ---- test_mesa_26_4_1_vink_and_bjorklund_zero_ulp ----
      l1 = hexd(int(z'47F1FFC99284C330', int64))
      m30 = hexd(int(z'4726FA2BA87B07AE', int64))
      call vink_case('vink_hot_35k', hexd(int(z'40E1170000000000', int64)), 0.005d0, 0.9d0)
      call vink_case('vink_hot_30k', hexd(int(z'40DD4C0000000000', int64)), 0.005d0, 0.9d0)
      call vink_case('vink_cool_18k', hexd(int(z'40D1940000000000', int64)), 0.005d0, 0.9d0)
      call vink_case('vink_jump_mid', hexd(int(z'40D946B0A3D70A3D', int64)), 0.019d0, 1d0)
      call vink_case('vink_jump_25950', hexd(int(z'40D9578000000000', int64)), 0.019d0, 1d0)
      call vink_case('vink_lowz_1e_12', hexd(int(z'40E1170000000000', int64)), 1d-12, 1d0)

      l_bj = hexd(int(z'48267FBBF725F3FC', int64))
      m_bj = hexd(int(z'47313BA0BE5C45C2', int64))
      t_bj = hexd(int(z'40E5F90000000000', int64))
      call scheme_case('bjorklund_std', 'Bjorklund', l_bj, m_bj, t_bj, 0.014d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%Bjorklund_scaling_factor = 0.8d0
      call run_case('bjorklund_std', l_bj, m_bj, t_bj)
      call scheme_case('bjorklund_lowz', 'Bjorklund', l_bj, m_bj, t_bj, 1d-13, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%Bjorklund_scaling_factor = 0.8d0
      call run_case('bjorklund_lowz', l_bj, m_bj, t_bj)

      ! ---- test_mesa_26_4_1_dutch_cool_sedd_and_rlo_zero_ulp ----
      m25 = hexd(int(z'472325CF0C668666', int64))
      m15 = hexd(int(z'4716FA2BA87B07AE', int64))
      m20 = hexd(int(z'471EA2E4E0A40A3D', int64))
      m5 = hexd(int(z'46FEA2E4E0A40A3D', int64))
      t25k = hexd(int(z'40D86A0000000000', int64))
      t4k = hexd(int(z'40AF400000000000', int64))
      t10500 = hexd(int(z'40C4820000000000', int64))
      t15k = hexd(int(z'40CD4C0000000000', int64))
      t30k = hexd(int(z'40DD4C0000000000', int64))

      call dutch_case('dutch_vink', l1, m30, t25k, 0.70d0, 0.28d0, 'de Jager')
      call dutch_case('dutch_wr_nl', l1, m15, t25k, 0.20d0, 0.78d0, 'de Jager')
      call dutch_case('dutch_low_dejager', l1, m25, t4k, 0.70d0, 0.28d0, 'de Jager')
      call dutch_case('dutch_low_vanloon', l1, m25, t4k, 0.70d0, 0.28d0, 'van Loon')
      call dutch_case('dutch_low_nieuwenhuijzen', l1, m25, t4k, 0.70d0, 0.28d0, 'Nieuwenhuijzen')
      call dutch_case('dutch_blend_10500', l1, m25, t10500, 0.70d0, 0.28d0, 'van Loon')

      call scheme_case('direct_dejager', 'de Jager', l1, m30, t15k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%de_Jager_scaling_factor = 0.7d0
      call run_case('direct_dejager', l1, m30, t15k)
      call scheme_case('direct_vanloon', 'van Loon', l1, m30, t15k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%van_Loon_scaling_factor = 0.6d0
      call run_case('direct_vanloon', l1, m30, t15k)
      call scheme_case('direct_nieuwenhuijzen', 'Nieuwenhuijzen', l1, m25, t15k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%Nieuwenhuijzen_scaling_factor = 0.5d0
      call run_case('direct_nieuwenhuijzen', l1, m25, t15k)
      call scheme_case('direct_reimers', 'Reimers', l1, m25, t4k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%Reimers_scaling_factor = 0.5d0
      call run_case('direct_reimers', l1, m25, t4k)
      call scheme_case('direct_blocker', 'Blocker', l1, m25, t4k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%Reimers_scaling_factor = 1d0
      s%Blocker_scaling_factor = 0.5d0
      call run_case('direct_blocker', l1, m25, t4k)

      ! Super-Eddington wind.
      l_se = hexd(int(z'48267FBBF725F3FC', int64))
      call base_model(2, l_se, m20, t30k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%super_eddington_scaling_factor = 0.7d0
      s%super_eddington_wind_Ledd_factor = 1.2d0
      s%prev_Ledd = 2d5*lsun
      call run_case('super_edd_wind', l_se, m20, t30k)

      ! Super-Eddington boost to Vink wind.
      l_bst = hexd(int(z'4801FFC99284C330', int64))
      call base_model(2, l_bst, m20, t30k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%cool_wind_full_on_T = 8d3
      s%hot_wind_full_on_T = 1.2d4
      s%hot_wind_scheme = 'Vink'
      s%Vink_scaling_factor = 1d0
      s%super_eddington_scaling_factor = 0d0
      s%wind_boost_full_off_L_div_Ledd = 1d0
      s%wind_boost_full_on_L_div_Ledd = 3d0
      s%super_eddington_wind_max_boost = 5d0
      s%prev_Ledd = 1d5*lsun
      call run_case('super_edd_boost', l_bst, m20, t30k)

      ! RLO overflow.
      l_rlo = hexd(int(z'479D4AAD50AAAA71', int64))
      call base_model(2, l_rlo, m5, t4k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      call rlo_controls()
      s%r(1) = 105d0*rsun
      call run_case('rlo_overflow', l_rlo, m5, t4k)

      ! RLO transfer-ratio scaling (ratio < 1).
      l_xfer = hexd(int(z'4797FA5F6F4FEA26', int64))
      call base_model(2, l_xfer, m5, t4k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      call rlo_controls()
      s%roche_lobe_xfer_full_on = 0.90d0
      s%roche_lobe_xfer_full_off = 1.00d0
      s%mass_change = 1d-6
      s%r(1) = 95d0*rsun
      call run_case('rlo_xfer_ratio', l_xfer, m5, t4k)

      ! Timestep ramp.
      call base_model(2, l1, m30, hexd(int(z'40E1170000000000', int64)), 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      call vink_hot_controls()
      s%mass_change_full_off_dt = 1d6
      s%mass_change_full_on_dt = 5d6
      s%dt = 2.5d6
      call run_case('dt_ramp_mid', l1, m30, hexd(int(z'40E1170000000000', int64)))

      ! Central-temperature taper and Zbase < 0.
      call base_model(2, l1, m30, hexd(int(z'40E1170000000000', int64)), -1d0, 0.68d0, 0.29d0, 0.68d0, 0.29d0)
      call vink_hot_controls()
      s%starting_T_center = 1.5d9
      s%max_T_center_for_full_mass_loss = 1d9
      s%max_T_center_for_any_mass_loss = 2d9
      call run_case('tcenter_taper_zneg', l1, m30, hexd(int(z'40E1170000000000', int64)))

      ! hot_wind_full_on_T == cool_wind_full_on_T tie.
      tc = hexd(int(z'40C3880000000000', int64))
      call base_model(2, l1, m20, tc, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%cool_wind_full_on_T = 1d4
      s%hot_wind_full_on_T = 1d4
      s%hot_wind_scheme = 'Vink'
      s%Vink_scaling_factor = 1d0
      s%cool_wind_RGB_scheme = 'Reimers'
      s%Reimers_scaling_factor = 1d0
      call run_case('hot_cool_tie', l1, m20, tc)

      ! min_wind floor and max_wind cap.
      call base_model(2, l1, m20, t15k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%hot_wind_scheme = 'Vink'
      s%Vink_scaling_factor = 1d0
      s%min_wind = 1d-4
      call run_case('min_wind_floor', l1, m20, t15k)
      tc = hexd(int(z'40E1170000000000', int64))
      call base_model(2, l1, m20, tc, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%hot_wind_scheme = 'Vink'
      s%Vink_scaling_factor = 1d0
      s%max_wind = 1d-9
      call run_case('max_wind_cap', l1, m20, tc)

      ! Envelope-mass wind cutoff with a three-zone model.
      m7 = hexd(int(z'470572069D3FA0C4', int64))
      t_env = hexd(int(z'40D3880000000000', int64))
      l_env = l1
      call base_model(3, l_env, m7, t_env, 0.02d0, 0.70d0, 0.28d0, 0.0d0, 0.98d0)
      s%xa(1:2, 1) = [0.70d0, 0.28d0]
      s%xa(1:2, 2) = [0.0d0, 0.98d0]
      s%xa(1:2, 3) = [0.0d0, 0.98d0]
      s%hot_wind_scheme = 'Vink'
      s%Vink_scaling_factor = 1d0
      s%wind_H_envelope_limit = 0.1d0
      s%r(1:3) = [r_of(l_env, t_env), 0.99d0*r_of(l_env, t_env), 0.5d0*r_of(l_env, t_env)]
      s%q(1:3) = [1d0, 6.99d0/7d0, 6.98d0/7d0]
      s%dq(1:3) = [0.01d0/7d0, 0.01d0/7d0, 6.98d0/7d0]
      s%dm(1:3) = [0.01d0*msun, 0.01d0*msun, 6.98d0*msun]
      s%m(1:3) = [7d0*msun, 6.99d0*msun, 6.98d0*msun]
      s%rho(1:3) = [1d-9, 1d-3, 1d0]
      s%Peos(1:3) = [1d3, 1d9, 1d15]
      s%lnT(1:3) = log([t_env, 1d6, 1d7])
      s%lnd(1:3) = log(s%rho(1:3))
      s%lnPeos(1:3) = log(s%Peos(1:3))
      s%L(1:3) = l_env
      s%Ye(1:3) = 0.5d0
      s%X(1:3) = s%xa(1, 1:3)
      s%Y(1:3) = s%xa(2, 1:3)
      s%rmid(1:3) = s%r(1:3)
      s%grav(1:3) = 0d0
      s%scale_height(1:3) = 0d0
      s%omega(1:3) = 0d0
      s%mstar = m7
      s%xmstar = m7
      s%star_mass = m7/msun
      call get_core_info(s)
      call run_case('h_env_limit_zero', l_env, m7, t_env)

      ! no_wind_if_no_rotation and dt <= mass_change_full_off_dt gates.
      call base_model(2, l1, m30, t15k, 0.02d0, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      call vink_hot_controls_only()
      s%no_wind_if_no_rotation = .true.
      call run_case('no_wind_if_no_rotation', l1, m30, t15k)
      s%no_wind_if_no_rotation = .false.
      s%mass_change_full_off_dt = 1d6
      s%mass_change_full_on_dt = 5d6
      s%dt = 1d6
      call run_case('dt_full_off', l1, m30, t15k)
   end subroutine emit_all_cases

   subroutine hot_cool_controls()
      s%cool_wind_full_on_T = 8d3
      s%hot_wind_full_on_T = 1.2d4
      s%cool_wind_RGB_scheme = 'Reimers'
      s%RGB_to_AGB_wind_switch = -1d0
      s%Reimers_scaling_factor = 1d0
      s%hot_wind_scheme = 'Vink'
      s%Vink_scaling_factor = 1d0
   end subroutine hot_cool_controls

   subroutine prems_controls()
      s%cool_wind_full_on_T = 9.99d9
      s%hot_wind_full_on_T = 1d10
      s%cool_wind_RGB_scheme = 'Reimers'
      s%cool_wind_AGB_scheme = 'Blocker'
      s%RGB_to_AGB_wind_switch = 1d-4
      s%Reimers_scaling_factor = 0.5d0
      s%Blocker_scaling_factor = 0.0003d0
   end subroutine prems_controls

   subroutine vink_hot_controls()
      s%cool_wind_full_on_T = 8d3
      s%hot_wind_full_on_T = 1.2d4
      s%hot_wind_scheme = 'Vink'
      s%Vink_scaling_factor = 1d0
   end subroutine vink_hot_controls

   subroutine vink_hot_controls_only()
      s%hot_wind_scheme = 'Vink'
      s%Vink_scaling_factor = 1d0
   end subroutine vink_hot_controls_only

   subroutine rlo_controls()
      s%rlo_scaling_factor = 0.8d0
      s%rlo_wind_min_L = 10d0
      s%rlo_wind_max_Teff = 1d4
      s%rlo_wind_scale_height = 0d0
      s%rlo_wind_roche_lobe_radius = 100d0
      s%rlo_wind_base_mdot = 1d-8
   end subroutine rlo_controls

   subroutine vink_case(cid, t, z, eta)
      character(len=*), intent(in) :: cid
      real(dp), intent(in) :: t, z, eta
      real(dp) :: l, m
      l = hexd(int(z'47F1FFC99284C330', int64))
      m = hexd(int(z'4726FA2BA87B07AE', int64))
      call scheme_case(cid, 'Vink', l, m, t, z, 0.70d0, 0.28d0, 0.70d0, 0.28d0)
      s%Vink_scaling_factor = eta
      call run_case(cid, l, m, t)
   end subroutine vink_case

   subroutine dutch_case(cid, l, m, t, sh1, she4, lowt)
      character(len=*), intent(in) :: cid, lowt
      real(dp), intent(in) :: l, m, t, sh1, she4
      call scheme_case(cid, 'Dutch', l, m, t, 0.02d0, sh1, she4, sh1, she4)
      s%Dutch_scaling_factor = 0.8d0
      s%Dutch_wind_lowT_scheme = lowt
      call run_case(cid, l, m, t)
   end subroutine dutch_case

end program gen_winds_oracles
