! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex + decimal reference vectors against compiled MESA 26.4.1
! (`libstar.a` + `CRMATH`) for `eps_mdot::calculate_eps_mdot` and surface mass-change
! energy rates across one accreting and one mass-losing model (#328, #318, PR #319).

program gen_eps_mdot_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use const_def, only: dp, msun, secyer
   use math_lib, only: MATH_BACKEND
   use star_def, only: star_info, star_ptr
   use star_lib, only: alloc_star, free_star, starlib_init, &
      star_set_kap_and_eos_handles, star_set_colors_handles, star_setup, star_load_zams
   use star_job_ctrls_io, only: set_default_star_job_controls, store_star_job_controls
   use hydro_vars, only: set_vars
   use adjust_mass, only: do_adjust_mass
   use eps_mdot, only: calculate_eps_mdot
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver, fflags_str
   integer :: ierr, u

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
      out_path = 'mesa_26_4_1_eps_mdot_oracles.tsv'
   end if

   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'

   write(u, '(A)') '# FlexMESA MESA 26.4.1 eps_mdot(k) and surface rate oracle vectors (#328, #318)'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') '# schema_scalar: SCALAR	case_id	0	field	raw_bits_hex	decimal'
   write(u, '(A)') '# schema_zone: ZONE	case_id	k	dm_before	te_profile_before	dm	rho	Peos	T	cp	L	gradT	grada	energy	m	m_grav	r	cgrav	te_profile_after	eps_mdot	eps_mdot_decimal'

   call run_eps_mdot_case(u, mesa_dir, 'accreting_1m_zams', 3.0e-3_dp * msun / secyer, 1.0e7_dp)
   call run_eps_mdot_case(u, mesa_dir, 'mass_losing_1m_zams', -3.0e-3_dp * msun / secyer, 1.0e7_dp)

   close(u)

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

   pure function to_hex(val) result(hbuf)
      real(dp), intent(in) :: val
      character(len=18) :: hbuf
      integer(int64) :: bits
      bits = transfer(val, 0_int64)
      write(hbuf, '(A,Z16.16)') '0x', bits
   end function to_hex

   subroutine emit_scalar(unit_no, cid, fld, val)
      integer, intent(in) :: unit_no
      character(len=*), intent(in) :: cid, fld
      real(dp), intent(in) :: val
      character(len=32) :: dbuf
      write(dbuf, '(ES25.17E3)') val
      write(unit_no, '(A,A,A,A,A,A,A,A,A,A,A)') &
         'SCALAR', char(9), trim(cid), char(9), '0', char(9), &
         trim(fld), char(9), to_hex(val), char(9), trim(adjustl(dbuf))
   end subroutine emit_scalar

   subroutine run_eps_mdot_case(unit_no, mdir, cid, mdot_cgs, dt_val)
      integer, intent(in) :: unit_no
      character(len=*), intent(in) :: mdir, cid
      real(dp), intent(in) :: mdot_cgs, dt_val

      integer :: id, ierr, k, nz
      character(len=32) :: dbuf
      type(star_info), pointer :: s

      ierr = 0
      call alloc_star(id, ierr)
      if (ierr /= 0) stop 'alloc_star failed'
      call star_ptr(id, s, ierr)
      if (ierr /= 0) stop 'star_ptr failed'
      call set_default_star_job_controls()
      call store_star_job_controls(s, ierr)
      if (ierr /= 0) stop 'store_star_job_controls failed'

      s%job%mesa_dir = trim(mdir)
      s%inlist_fname = trim(mdir) // '/star/test/inlist'
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
      s%report_ierr = .true.
      call star_load_zams(id, ierr)
      if (ierr /= 0) stop 'star_load_zams failed'

      if (s%v_flag .or. s%u_flag .or. s%RSP_flag .or. s%RSP2_flag .or. s%use_mass_corrections) then
         stop 'unexpected baseline velocity/RSP/mass_correction flags on ZAMS model'
      end if

      s%dt = dt_val
      s%okay_to_set_mixing_info = .true.
      call set_vars(s, dt_val, ierr)
      if (ierr /= 0) stop 'initial set_vars failed'

      s%mstar_dot = mdot_cgs
      s%super_eddington_wind_mdot = 0.0_dp
      s%min_star_mass_for_loss = 0.0_dp
      s%max_star_mass_for_gain = 0.0_dp
      s%use_other_accreting_state = .false.

      call do_adjust_mass(s, s%species, ierr)
      if (ierr /= 0) stop 'do_adjust_mass failed'

      call set_vars(s, dt_val, ierr)
      if (ierr /= 0) stop 'post-adjust_mass set_vars failed'

      call calculate_eps_mdot(s, dt_val, ierr)
      if (ierr /= 0) stop 'calculate_eps_mdot failed'

      nz = s%nz
      if (s%k_below_const_q <= 1 .or. s%k_below_const_q >= s%k_const_mass .or. s%k_const_mass >= nz) then
         stop 'unexpected k_below_const_q / k_const_mass region bounds'
      end if
      if (s%mdot_acoustic_surface == 0.0_dp) stop 'expected nonzero mdot_acoustic_surface'
      if (mdot_cgs < 0.0_dp .and. s%mdot_adiabatic_surface == 0.0_dp) then
         stop 'expected nonzero mdot_adiabatic_surface on mass-losing model'
      end if

      ! Emit scalar configuration, adjust_mass record fields, and surface outputs
      call emit_scalar(unit_no, cid, 'nz', real(nz, dp))
      call emit_scalar(unit_no, cid, 'dt', dt_val)
      call emit_scalar(unit_no, cid, 'mstar_dot', s%mstar_dot)
      call emit_scalar(unit_no, cid, 'eps_mdot_leak_frac_factor', s%eps_mdot_leak_frac_factor)
      call emit_scalar(unit_no, cid, 'eps_mdot_factor', s%eps_mdot_factor)
      call emit_scalar(unit_no, cid, 'k_below_const_q', real(s%k_below_const_q, dp))
      call emit_scalar(unit_no, cid, 'k_const_mass', real(s%k_const_mass, dp))
      call emit_scalar(unit_no, cid, 'outer_frac_sub1', s%adjust_mass_outer_frac_sub1)
      call emit_scalar(unit_no, cid, 'mid_frac_sub1', s%adjust_mass_mid_frac_sub1)
      call emit_scalar(unit_no, cid, 'inner_frac_sub1', s%adjust_mass_inner_frac_sub1)
      call emit_scalar(unit_no, cid, 'm_center', s%M_center)
      call emit_scalar(unit_no, cid, 'r_center', s%R_center)
      call emit_scalar(unit_no, cid, 'mdot_acoustic_surface', s%mdot_acoustic_surface)
      call emit_scalar(unit_no, cid, 'mdot_adiabatic_surface', s%mdot_adiabatic_surface)

      ! Emit one compact row per zone covering all inputs and outputs
      do k = 1, nz
         write(dbuf, '(ES25.17E3)') s%eps_mdot(k)
         write(unit_no, '(A,A,A,A,I0,17(A,A),A,A)') &
            'ZONE', char(9), trim(cid), char(9), k, &
            char(9), to_hex(s%dm_before_adjust_mass(k)), &
            char(9), to_hex(s%total_energy_profile_before_adjust_mass(k)), &
            char(9), to_hex(s%dm(k)), &
            char(9), to_hex(s%rho(k)), &
            char(9), to_hex(s%Peos(k)), &
            char(9), to_hex(s%T(k)), &
            char(9), to_hex(s%cp(k)), &
            char(9), to_hex(s%L(k)), &
            char(9), to_hex(s%gradT(k)), &
            char(9), to_hex(s%grada(k)), &
            char(9), to_hex(s%energy(k)), &
            char(9), to_hex(s%m(k)), &
            char(9), to_hex(s%m_grav(k)), &
            char(9), to_hex(s%r(k)), &
            char(9), to_hex(s%cgrav(k)), &
            char(9), to_hex(s%total_energy_profile_after_adjust_mass(k)), &
            char(9), to_hex(s%eps_mdot(k)), &
            char(9), trim(adjustl(dbuf))
      end do

      write(*, '(A,A,A,I0,A,I0,A,I0,A,ES14.6,A,ES14.6)') &
         'Case ', trim(cid), ': nz=', nz, ', kA=', s%k_below_const_q, ', kB=', s%k_const_mass, &
         ', mdot_acoustic=', s%mdot_acoustic_surface, ', mdot_adiabatic=', s%mdot_adiabatic_surface

      call free_star(id, ierr)
   end subroutine run_eps_mdot_case

end program gen_eps_mdot_oracles
