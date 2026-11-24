! SPDX-License-Identifier: GPL-3.0-or-later
! One-zone MESA 26.4.1 `net::net_get` capture for `approx21_cr60_plus_co56.net`
! at a hydrogen-burning state taken bit for bit from a MESA 26.4.1 step dump
! (`star/20M_z2m2_high_rotation`, model 1, zone 650), with the star's default
! controls (#516).
!
! The call reproduces `do1_net` in `star/private/net.f90:184-256`:
!   - `T = exp(lnT)` and `rho = exp(lnd)` (`star/private/hydro_vars.f90:299,305`);
!   - `log10_T = lnT/ln10` and `log10_rho = lnd/ln10` (`net.f90:184-186`);
!   - `abar`, `zbar`, `z2bar` and `ye` from `basic_composition_info`;
!   - `eta`, `d_eta_dlnT` and `d_eta_dlnd` from `eosDT_get` with the default
!     `&eos` controls (the case's `&eos` namelist is empty);
!   - `screening_mode = 'chugunov'`, `rate_factors = 1`, `weak_rate_factor = 1`,
!     `reaction_neuQs_factor = 1` (`star/defaults/controls.defaults:6779,
!     6835, 6845`);
!   - `fe56ec_fake_factor = 1d-7` and `min_T_for_fe56ec_fake_factor = 3d9`
!     (`controls.defaults:8245-8246`), set as `net_tables` does
!     (`net.f90:747-748`); `net_logTcut_lo/lim = -1` keep the net defaults.
! The mass fractions are passed unnormalised, as the star passes `s% xa`.

program gen_net_516
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use chem_def, only: chem_isos, category_name, num_categories
   use chem_lib, only: chem_init, chem_get_iso_id, basic_composition_info
   use const_def, only: dp, ln10
   use const_lib, only: const_init
   use math_lib, only: MATH_BACKEND, math_init
   use eos_def, only: num_eos_basic_results, num_eos_d_dxa_results, i_eta
   use eos_lib, only: eos_init, alloc_eos_handle, eosDT_get
   use net_def, only: Net_General_Info, Net_Info, get_net_ptr
   use net_lib, only: net_init, alloc_net_handle, free_net_handle, &
      net_start_def, read_net_file, net_finish_def, net_setup_tables, net_get, &
      net_set_fe56ec_fake_factor
   use rates_def, only: chugunov_screening, reaction_Name, &
      std_reaction_Qs, std_reaction_neuQs
   use rates_lib, only: rates_init, rates_warning_init
   implicit none

   character(len=*), parameter :: net_name = 'approx21_cr60_plus_co56.net'
   integer, parameter :: nspec_in = 22
   ! Species and IEEE-754 bits of `xh`/`xa` at model 1, zone 650 of the MESA
   ! 26.4.1 step dump for `star/20M_z2m2_high_rotation`.
   character(len=5), parameter :: iso_names(nspec_in) = [character(len=5) :: &
      'neut', 'h1', 'prot', 'he3', 'he4', 'c12', 'n14', 'o16', 'ne20', &
      'mg24', 'si28', 's32', 'ar36', 'ca40', 'ti44', 'cr48', 'cr60', &
      'fe52', 'fe54', 'fe56', 'co56', 'ni56']
   integer(int64), parameter :: x_bits(nspec_in) = [ &
      int(z'2B714F4A252C8053', int64), int(z'3FE7210DABFE3E67', int64), &
      int(z'2B714F4A252C8053', int64), int(z'3EFCB4E5559AEEB6', int64), &
      int(z'3FD07949ED2E9B39', int64), int(z'3F6BED83934800E5', int64), &
      int(z'3F506338AA533068', int64), int(z'3F83012DBF9BDF6D', int64), &
      int(z'3F60E59033AC7B21', int64), int(z'3F708546429D9AAD', int64), &
      int(z'2B714F4A252C8053', int64), int(z'2B714F4A252C8053', int64), &
      int(z'2B714F4A252C8053', int64), int(z'2B714F4A252C8053', int64), &
      int(z'2B714F4A252C8053', int64), int(z'2B714F4A252C8053', int64), &
      int(z'2B714F4A252C8053', int64), int(z'2B714F4A252C8053', int64), &
      int(z'2B714F4A252C8053', int64), int(z'2B714F4A252C8053', int64), &
      int(z'2B714F4A252C8053', int64), int(z'2B714F4A252C8053', int64)]
   integer(int64), parameter :: lnd_bits = int(z'BFB86928C537341A', int64)
   integer(int64), parameter :: lnT_bits = int(z'40309892B542D309', int64)

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
      out_path = 'mesa_26_4_1_net_approx21_zone650.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'
   call eos_init('', .true., ierr)
   if (ierr /= 0) stop 'eos_init failed'
   call rates_init('reactions.list', '', 'rate_tables', .false., .false., '', '', '', ierr)
   if (ierr /= 0) stop 'rates_init failed'
   call rates_warning_init(.true., 10.0_dp)
   call net_init(ierr)
   if (ierr /= 0) stop 'net_init failed'
   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'
   write(u, '(A)') '# FlexMESA MESA 26.4.1 one-zone net::net_get capture, approx21_cr60_plus_co56.net (#516)'
   write(u, '(A)') '# state: star/20M_z2m2_high_rotation, MESA 26.4.1 step dump, model 1, zone 650'
   write(u, '(A)') '# controls: star defaults (chugunov screening, rate_factors=1, weak_rate_factor=1, fe56ec_fake_factor=1d-7)'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') 'row_kind	net_name	key1	key2	key3	raw_bits_hex	decimal'
   rc = 0
   call emit_case(u, rc)
   close(u)
   write(*, '(A,I0,A)') 'Wrote ', rc, ' net oracle rows.'

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

   subroutine emit_row(unit_no, rc, kind_str, k1, k2, k3, val)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: kind_str, k1, k2, k3
      real(dp), intent(in) :: val
      character(len=18) :: hbuf
      character(len=32) :: dbuf
      write(hbuf, '("0x", Z16.16)') transfer(val, 0_int64)
      write(dbuf, '(ES26.17E3)') val
      rc = rc + 1
      write(unit_no, '(A,6(A,A))') trim(kind_str), achar(9), net_name, &
         achar(9), trim(k1), achar(9), trim(k2), achar(9), trim(k3), &
         achar(9), trim(hbuf), achar(9), trim(adjustl(dbuf))
   end subroutine emit_row

   subroutine emit_case(unit_no, rc)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      type(Net_General_Info), pointer :: g
      type(Net_Info) :: n
      integer :: handle, eos_handle, info, species, num_reactions, i, j, c, r, cid
      integer, pointer :: chem_id(:), net_iso(:)
      real(dp) :: lnd, lnT, T, rho, logT, logRho
      real(dp) :: xh, xhe, z, abar, zbar, z2bar, z53bar, ye, mass_correction, sumx
      real(dp) :: eta, d_eta_dlnT, d_eta_dlnd, weak_rate_factor
      real(dp) :: eps_nuc, d_eps_nuc_dRho, d_eps_nuc_dT, eps_neu_total
      real(dp) :: eps_nuc_categories(num_categories)
      real(dp) :: res(num_eos_basic_results), d_dlnd(num_eos_basic_results), d_dlnT(num_eos_basic_results)
      real(dp), allocatable :: xa(:), d_eps_nuc_dx(:), dxdt(:), d_dxdt_dRho(:), d_dxdt_dT(:), &
         d_dxdt_dx(:,:), d_dxa(:,:)
      real(dp), pointer :: rate_factors(:)
      character(len=16) :: idx_buf

      info = 0
      handle = alloc_net_handle(info)
      if (info /= 0) stop 'alloc_net_handle failed'
      call net_start_def(handle, info)
      call read_net_file(net_name, handle, info)
      if (info /= 0) stop 'read_net_file failed'
      call net_finish_def(handle, info)
      if (info /= 0) stop 'net_finish_def failed'
      call net_set_fe56ec_fake_factor(handle, 1d-7, 3d9, info)
      if (info /= 0) stop 'net_set_fe56ec_fake_factor failed'
      call net_setup_tables(handle, '', info)
      if (info /= 0) stop 'net_setup_tables failed'
      call get_net_ptr(handle, g, info)
      if (info /= 0) stop 'get_net_ptr failed'
      species = g% num_isos
      num_reactions = g% num_reactions
      if (species /= nspec_in) stop 'unexpected species count'
      chem_id => g% chem_id
      net_iso => g% net_iso

      allocate(xa(species), d_eps_nuc_dx(species), dxdt(species), d_dxdt_dRho(species), &
         d_dxdt_dT(species), d_dxdt_dx(species, species), &
         d_dxa(num_eos_d_dxa_results, species), rate_factors(num_reactions))
      do i = 1, nspec_in
         cid = chem_get_iso_id(trim(iso_names(i)))
         if (cid <= 0) stop 'unknown isotope name'
         if (net_iso(cid) <= 0) stop 'isotope not in net'
         xa(net_iso(cid)) = transfer(x_bits(i), 0.0_dp)
      end do

      lnd = transfer(lnd_bits, 0.0_dp)
      lnT = transfer(lnT_bits, 0.0_dp)
      rho = exp(lnd)
      T = exp(lnT)
      logRho = lnd/ln10
      logT = lnT/ln10
      call basic_composition_info(species, chem_id, xa, xh, xhe, z, &
         abar, zbar, z2bar, z53bar, ye, mass_correction, sumx)

      eos_handle = alloc_eos_handle(info)
      if (info /= 0) stop 'alloc_eos_handle failed'
      call eosDT_get(eos_handle, species, chem_id, net_iso, xa, rho, logRho, T, logT, &
         res, d_dlnd, d_dlnT, d_dxa, info)
      if (info /= 0) stop 'eosDT_get failed'
      eta = res(i_eta)
      d_eta_dlnT = d_dlnT(i_eta)
      d_eta_dlnd = d_dlnd(i_eta)

      rate_factors(:) = 1.0_dp
      weak_rate_factor = 1.0_dp
      call net_get( &
         handle, .false., n, species, num_reactions, &
         xa, T, logT, rho, logRho, &
         abar, zbar, z2bar, ye, eta, d_eta_dlnT, d_eta_dlnd, &
         rate_factors, weak_rate_factor, &
         std_reaction_Qs, std_reaction_neuQs, &
         eps_nuc, d_eps_nuc_dRho, d_eps_nuc_dT, d_eps_nuc_dx, &
         dxdt, d_dxdt_dRho, d_dxdt_dT, d_dxdt_dx, &
         chugunov_screening, eps_nuc_categories, eps_neu_total, info)
      if (info /= 0) stop 'net_get failed'

      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'lnd', '-', lnd)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'lnT', '-', lnT)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'rho', '-', rho)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'temp', '-', T)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'logRho', '-', logRho)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'logT', '-', logT)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'abar', '-', abar)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'zbar', '-', zbar)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'z2bar', '-', z2bar)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'ye', '-', ye)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'eta', '-', eta)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'd_eta_dlnT', '-', d_eta_dlnT)
      call emit_row(unit_no, rc, 'INPUT', 'scalar', 'd_eta_dlnd', '-', d_eta_dlnd)
      do i = 1, species
         call emit_row(unit_no, rc, 'INPUT', 'x', chem_isos% name(chem_id(i)), '-', xa(i))
      end do
      call emit_row(unit_no, rc, 'SCALAR', 'eps_nuc', '-', '-', eps_nuc)
      call emit_row(unit_no, rc, 'SCALAR', 'd_eps_nuc_dRho', '-', '-', d_eps_nuc_dRho)
      call emit_row(unit_no, rc, 'SCALAR', 'd_eps_nuc_dT', '-', '-', d_eps_nuc_dT)
      call emit_row(unit_no, rc, 'SCALAR', 'eps_neu_total', '-', '-', eps_neu_total)
      do i = 1, species
         call emit_row(unit_no, rc, 'ISO', 'dxdt', chem_isos% name(chem_id(i)), '-', dxdt(i))
         call emit_row(unit_no, rc, 'ISO', 'd_dxdt_dRho', chem_isos% name(chem_id(i)), '-', d_dxdt_dRho(i))
         call emit_row(unit_no, rc, 'ISO', 'd_dxdt_dT', chem_isos% name(chem_id(i)), '-', d_dxdt_dT(i))
         call emit_row(unit_no, rc, 'ISO', 'd_eps_nuc_dx', chem_isos% name(chem_id(i)), '-', d_eps_nuc_dx(i))
      end do
      do i = 1, species
         do j = 1, species
            call emit_row(unit_no, rc, 'JAC', 'd_dxdt_dx', chem_isos% name(chem_id(i)), &
               chem_isos% name(chem_id(j)), d_dxdt_dx(i, j))
         end do
      end do
      do c = 1, num_categories
         write(idx_buf, '(I0)') c
         call emit_row(unit_no, rc, 'CAT', 'eps_nuc_categories', trim(idx_buf), &
            trim(category_name(c)), eps_nuc_categories(c))
      end do
      do r = 1, num_reactions
         write(idx_buf, '(I0)') r
         call emit_row(unit_no, rc, 'RATE', 'rate_raw', trim(reaction_Name(g% reaction_id(r))), &
            trim(idx_buf), n% rate_raw(r))
         call emit_row(unit_no, rc, 'RATE', 'rate_screened', trim(reaction_Name(g% reaction_id(r))), &
            trim(idx_buf), n% rate_screened(r))
      end do
      call free_net_handle(handle)
   end subroutine emit_case

end program gen_net_516
