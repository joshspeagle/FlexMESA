! SPDX-License-Identifier: LGPL-3.0-or-later
! Generate IEEE-754 hex + decimal thermal neutrino (`neu_get`) reference vectors
! against compiled MESA 26.4.1 (`libneu.a` + `CRMATH`) for #161.

program gen_neu_oracles
   use, intrinsic :: iso_fortran_env, only: int64
   use const_def, only: dp
   use const_lib, only: const_init
   use math_lib, only: math_init, exp10
   use neu_def, only: num_neu_types, num_neu_rvs
   use neu_lib, only: neu_get
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver
   integer :: ierr, u, idx
   logical :: all_on(num_neu_types), one_ch(num_neu_types)

   call get_environment_variable('MESA_DIR', mesa_dir)
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   if (command_argument_count() >= 1) then
      call get_command_argument(1, out_path)
   else
      out_path = 'mesa_26_4_1_neu_oracles.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output file'

   write(u, '(A)') '# FlexMESA MESA 26.4.1 thermal neutrino (neu) oracle vectors (#161)'
   write(u, '(A)') '# mesa_version: ' // trim(mesa_ver)
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(sdk_ver)
   write(u, '(A)') '# math_backend: CRMATH'
   write(u, '(A)') 'case_id	label	flags_mask	log10_T_hex	log10_Rho_hex	abar_hex	zbar_hex	log10_Tlim_hex	out_idx	out_name	raw_bits_hex	decimal'

   all_on = .true.
   idx = 0

   ! 1. Low-T cutoff and smooth cospi/sinpi ramp regimes (default log10_Tlim = 7.5 and custom 6.8)
   call emit_case(u, idx, 'below_cutoff_default', 7.40_dp, 4.0_dp, 4.0_dp, 2.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'at_cutoff_edge', 7.50_dp, 4.0_dp, 4.0_dp, 2.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'ramp_q1_7p55', 7.55_dp, 5.0_dp, 4.0_dp, 2.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'ramp_mid_7p70', 7.70_dp, 5.5_dp, 12.0_dp, 6.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'ramp_q3_7p85', 7.85_dp, 6.0_dp, 16.0_dp, 8.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'ramp_top_7p90', 7.90_dp, 6.0_dp, 16.0_dp, 8.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'custom_tlim_6p8_ramp', 7.05_dp, 5.0_dp, 4.0_dp, 2.0_dp, 6.8_dp, all_on)
   call emit_case(u, idx, 'custom_tlim_6p8_full', 7.30_dp, 6.5_dp, 12.0_dp, 6.0_dp, 6.8_dp, all_on)

   ! 2. Astrophysical regimes across (logT, logRho, abar, zbar): H, He, C/O, O/Ne/Mg, Si, Fe
   call emit_case(u, idx, 'solar_h_he_mix', 7.62_dp, 2.15_dp, 1.30_dp, 1.10_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'he_core_flash', 8.05_dp, 5.80_dp, 4.00_dp, 2.00_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'agb_thermal_pulse', 8.35_dp, 4.20_dp, 12.80_dp, 6.40_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'co_wd_warm_liq', 7.95_dp, 6.80_dp, 13.714285714285714_dp, 6.857142857142857_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'co_wd_brem_blend', 7.92_dp, 9.65_dp, 12.0_dp, 6.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'fe_crystal_brem_solid', 7.95_dp, 9.20_dp, 56.0_dp, 26.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'fe_brem_gamma_mid', 8.22_dp, 9.95_dp, 56.0_dp, 26.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'c_burn_ign', 8.85_dp, 5.90_dp, 13.5_dp, 6.75_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'onemg_degenerate', 9.05_dp, 9.40_dp, 21.818181818181817_dp, 10.909090909090908_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 't9_eq_1_boundary_lo', 8.999_dp, 6.50_dp, 16.0_dp, 8.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 't9_eq_1_boundary_hi', 9.001_dp, 6.50_dp, 16.0_dp, 8.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'ne_o_burning', 9.25_dp, 6.80_dp, 18.5_dp, 9.25_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'si_burning_pair_dom', 9.55_dp, 7.40_dp, 28.0_dp, 14.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'fe_core_collapse_pre', 9.85_dp, 9.30_dp, 54.5_dp, 25.2_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'pair_instability_lo_rho', 9.45_dp, 2.50_dp, 16.0_dp, 8.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'ultra_rel_pair_10p2', 10.20_dp, 8.10_dp, 56.0_dp, 26.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'ultra_hi_rho_plas_10p5', 9.10_dp, 10.50_dp, 56.0_dp, 26.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'reco_dom_hi_z_cool', 7.92_dp, 3.80_dp, 56.0_dp, 26.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'reco_degen_crossover', 8.15_dp, 6.20_dp, 28.0_dp, 14.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'phot_peak_regime', 8.65_dp, 4.80_dp, 4.0_dp, 2.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'pure_h_hot_low_d', 8.50_dp, 1.50_dp, 1.0_dp, 1.0_dp, 7.5_dp, all_on)
   call emit_case(u, idx, 'neutron_rich_crust', 8.75_dp, 10.20_dp, 62.0_dp, 26.0_dp, 7.5_dp, all_on)

   ! 3. Single-channel flag isolation + all-off check at representative stellar interior points
   one_ch = [.true., .false., .false., .false., .false.]
   call emit_case(u, idx, 'only_pair_t9gt1', 9.40_dp, 6.50_dp, 28.0_dp, 14.0_dp, 7.5_dp, one_ch)
   one_ch = [.false., .true., .false., .false., .false.]
   call emit_case(u, idx, 'only_plas_degen', 8.80_dp, 8.80_dp, 12.0_dp, 6.0_dp, 7.5_dp, one_ch)
   one_ch = [.false., .false., .true., .false., .false.]
   call emit_case(u, idx, 'only_phot_mid_t', 8.60_dp, 5.20_dp, 16.0_dp, 8.0_dp, 7.5_dp, one_ch)
   one_ch = [.false., .false., .false., .true., .false.]
   call emit_case(u, idx, 'only_brem_hi_rho', 8.40_dp, 8.50_dp, 24.0_dp, 12.0_dp, 7.5_dp, one_ch)
   one_ch = [.false., .false., .false., .false., .true.]
   call emit_case(u, idx, 'only_reco_hi_z', 8.10_dp, 4.50_dp, 56.0_dp, 26.0_dp, 7.5_dp, one_ch)
   one_ch = [.false., .false., .false., .false., .false.]
   call emit_case(u, idx, 'all_flags_off', 9.20_dp, 7.00_dp, 16.0_dp, 8.0_dp, 7.5_dp, one_ch)

   close(u)
   write(*, '(A,I0,A)') 'Wrote ', idx, ' neu oracle cases.'

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

   subroutine emit_case(u, idx, label, lt, lrho, abar, zbar, ltlim, flags)
      integer, intent(in) :: u
      integer, intent(inout) :: idx
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: lt, lrho, abar, zbar, ltlim
      logical, intent(in) :: flags(num_neu_types)

      real(dp) :: t, rho, loss(num_neu_rvs), sources(num_neu_types, num_neu_rvs)
      integer :: info, j, c, d, mask_int
      character(len=5) :: mask_str
      character(len=24), parameter :: rv_names(5) = [ &
         character(len=24) :: 'val', 'd_dT', 'd_dRho', 'd_dabar', 'd_dzbar']
      character(len=8), parameter :: ch_names(5) = [ &
         character(len=8) :: 'pair', 'plas', 'phot', 'brem', 'reco']
      character(len=32) :: oname

      idx = idx + 1
      t = exp10(lt)
      rho = exp10(lrho)
      info = 0
      call neu_get(t, lt, rho, lrho, abar, zbar, ltlim, flags, loss, sources, info)
      if (info /= 0) stop 'neu_get returned nonzero info'

      do mask_int = 1, 5
         mask_str(mask_int:mask_int) = merge('1', '0', flags(mask_int))
      end do

      do d = 1, num_neu_rvs
         oname = 'total.' // trim(rv_names(d))
         write(u, '(I0,A,A,A,A,5(A,A),A,I0,A,A,A,A,A,ES25.17E3)') &
            idx, char(9), trim(label), char(9), mask_str, &
            char(9), to_hex(lt), char(9), to_hex(lrho), char(9), to_hex(abar), &
            char(9), to_hex(zbar), char(9), to_hex(ltlim), &
            char(9), d - 1, char(9), trim(oname), char(9), to_hex(loss(d)), char(9), loss(d)
      end do

      j = 5
      do c = 1, num_neu_types
         do d = 1, num_neu_rvs
            oname = trim(ch_names(c)) // '.' // trim(rv_names(d))
            write(u, '(I0,A,A,A,A,5(A,A),A,I0,A,A,A,A,A,ES25.17E3)') &
               idx, char(9), trim(label), char(9), mask_str, &
               char(9), to_hex(lt), char(9), to_hex(lrho), char(9), to_hex(abar), &
               char(9), to_hex(zbar), char(9), to_hex(ltlim), &
               char(9), j, char(9), trim(oname), char(9), to_hex(sources(c, d)), char(9), sources(c, d)
            j = j + 1
         end do
      end do
   end subroutine emit_case

end program gen_neu_oracles
