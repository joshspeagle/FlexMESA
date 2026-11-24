! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex + decimal nuclear reaction network (`net::net_get`)
! oracle vectors against compiled MESA 26.4.1 (`libnet.a` + `librates.a` +
! `CRMATH`) for #328 and #161.

program gen_net_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use chem_def, only: chem_isos, category_name, num_categories, &
      ih1, ih2, ihe3, ihe4, ili7, ibe7, ic12, ic13, in13, in14, in15, &
      io14, io15, io16, io18, if17, if18, ine18, ine19, ine20, ine22, &
      img22, img24, isi28, is32, iar36, ica40, iti44, icr48, icr56, icr60, &
      ife52, ife54, ife56, ico56, ini56, iprot, ineut
   use chem_lib, only: chem_init, composition_info
   use const_def, only: dp
   use const_lib, only: const_init
   use math_lib, only: MATH_BACKEND, math_init, exp10
   use net_def, only: Net_General_Info, Net_Info, get_net_ptr
   use net_lib, only: net_init, alloc_net_handle, free_net_handle, &
      net_start_def, read_net_file, net_finish_def, net_setup_tables, net_get
   use rates_def, only: extended_screening, reaction_Name, &
      std_reaction_Qs, std_reaction_neuQs
   use rates_lib, only: rates_init, rates_warning_init
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver, fflags_str
   integer :: ierr, u, row_count

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
      out_path = 'mesa_26_4_1_net_oracles.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'
   call rates_init('reactions.list', '', 'rate_tables', .false., .false., '', '', '', ierr)
   if (ierr /= 0) stop 'rates_init failed'
   call rates_warning_init(.true., 10.0_dp)
   call net_init(ierr)
   if (ierr /= 0) stop 'net_init failed'
   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'

   write(u, '(A)') '# FlexMESA MESA 26.4.1 net::net_get nuclear reaction network oracle vectors (#328, #161)'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') 'row_kind	net_name	key1	key2	key3	raw_bits_hex	decimal'

   row_count = 0
   call emit_net_case(u, row_count, 'basic.net', 7.833_dp, 2.0_dp, 0.0_dp)
   call emit_net_case(u, row_count, 'o18_and_ne22.net', 7.833_dp, 2.0_dp, 0.0_dp)
   call emit_net_case(u, row_count, 'pp_extras.net', 7.833_dp, 2.0_dp, 0.0_dp)
   call emit_net_case(u, row_count, 'cno_extras.net', 7.833_dp, 2.0_dp, 0.0_dp)
   call emit_net_case(u, row_count, 'approx21.net', 9.6_dp, 6.0_dp, 0.0_dp)
   call emit_net_case( &
      u, row_count, 'approx21_cr60_plus_co56.net', &
      4.6233007922659333_dp, -10.746410107891649_dp, -22.590260158215202_dp)

   close(u)
   write(*, '(A,I0,A)') 'Wrote ', row_count, ' net oracle rows.'

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

   subroutine emit_row(unit_no, rc, kind_str, net_name, k1, k2, k3, val)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: kind_str, net_name, k1, k2, k3
      real(dp), intent(in) :: val
      integer(int64) :: bits
      character(len=18) :: hbuf
      character(len=32) :: dbuf
      bits = transfer(val, 0_int64)
      write(hbuf, '("0x", Z16.16)') bits
      write(dbuf, '(ES26.17E3)') val
      rc = rc + 1
      write(unit_no, '(A,6(A,A))') &
         trim(kind_str), &
         achar(9), trim(net_name), &
         achar(9), trim(k1), &
         achar(9), trim(k2), &
         achar(9), trim(k3), &
         achar(9), trim(hbuf), &
         achar(9), trim(adjustl(dbuf))
   end subroutine emit_row

   subroutine set_case_composition(net_name, net_iso, species, xin)
      character(len=*), intent(in) :: net_name
      integer, intent(in) :: net_iso(:)
      integer, intent(in) :: species
      real(dp), intent(out) :: xin(species)
      integer :: i, adjustment_iso
      real(dp) :: xsum

      xin(:) = 0.0_dp
      adjustment_iso = net_iso(img24)

      select case (trim(net_name))
      case ('basic.net')
         ! Exact upstream single-precision literal for ih1 from net/test/src/test_net_support.f90:514
         xin(net_iso(ih1)) = 0.655186E+00
         xin(net_iso(ihe4)) = 0.31002164_dp
         xin(net_iso(ic12)) = 0.002725e-1_dp
         xin(net_iso(in14)) = 0.203101e-1_dp + 0.612124e-6_dp + 0.109305e-2_dp + 0.356004e-4_dp
         xin(net_iso(io16)) = 0.094000e-1_dp
         xin(net_iso(ine20)) = 0.162163e-2_dp
         xin(net_iso(img24)) = 0.658226e-25_dp
         xin(net_iso(ihe3)) = 0.201852e-2_dp
      case ('o18_and_ne22.net')
         xin(net_iso(ih1)) = 0.655186E+00
         xin(net_iso(ihe4)) = 0.31002164_dp
         xin(net_iso(ic12)) = 0.002725e-1_dp
         xin(net_iso(in14)) = 0.203101e-1_dp + 0.612124e-6_dp + 0.109305e-2_dp
         xin(net_iso(io16)) = 0.094000e-1_dp
         xin(net_iso(io18)) = 1.0e-20_dp
         xin(net_iso(ine20)) = 0.162163e-2_dp
         xin(net_iso(img24)) = 0.658226e-25_dp
         xin(net_iso(ine22)) = 0.201852e-2_dp
      case ('pp_extras.net')
         xin(net_iso(ih1)) = 0.655186E+00
         xin(net_iso(ihe4)) = 0.31002164_dp
         xin(net_iso(ic12)) = 0.002725e-1_dp
         xin(net_iso(in14)) = 0.203101e-1_dp + 0.612124e-6_dp + 0.109305e-2_dp + 0.356004e-4_dp
         xin(net_iso(io16)) = 0.094000e-1_dp
         xin(net_iso(ine20)) = 0.162163e-2_dp
         xin(net_iso(img24)) = 0.658226e-25_dp
         xin(net_iso(ih2)) = 0.632956e-17_dp
         xin(net_iso(ihe3)) = 0.201852e-2_dp
         xin(net_iso(ili7)) = 0.664160e-15_dp
         xin(net_iso(ibe7)) = 0.103866e-15_dp
      case ('cno_extras.net')
         xin(net_iso(ih1)) = 0.173891680788e-1_dp
         xin(net_iso(ihe4)) = 0.963245225401_dp
         xin(net_iso(ic12)) = 0.238935745993e-3_dp
         xin(net_iso(in14)) = 0.134050688300e-1_dp
         xin(net_iso(io16)) = 0.268791618452e-3_dp
         xin(net_iso(ine20)) = 0.180001692845e-2_dp
         xin(net_iso(img24)) = 0.353667702698e-2_dp
         xin(net_iso(ic13)) = 0.717642727071e-4_dp
         xin(net_iso(in13)) = 0.370732258156e-9_dp
         xin(net_iso(in15)) = 0.450484708137e-6_dp
         xin(net_iso(io14)) = 0.100000000000e-49_dp
         xin(net_iso(io15)) = 0.874815374966e-10_dp
         xin(net_iso(if17)) = 0.100000000000e-49_dp
         xin(net_iso(if18)) = 0.100000000000e-49_dp
         xin(net_iso(ine18)) = 0.100000000000e-49_dp
         xin(net_iso(ine19)) = 0.100000000000e-49_dp
         xin(net_iso(img22)) = 0.439011547696e-4_dp
      case ('approx21.net')
         xin(net_iso(ife56)) = 8.0387021484318166e-1_dp
         xin(net_iso(ife54)) = 1.6096648736760832e-1_dp
         xin(net_iso(icr56)) = 2.9480945535920525e-2_dp
         xin(net_iso(ihe4)) = 4.8624637161320565e-3_dp
         xin(net_iso(ini56)) = 2.8376270731890360e-4_dp
         xin(net_iso(isi28)) = 1.5018628906135739e-4_dp
         xin(net_iso(is32)) = 1.1613271635573457e-4_dp
         xin(net_iso(iprot)) = 1.1139431633673653e-4_dp
         xin(net_iso(ica40)) = 5.3688377473494185e-5_dp
         xin(net_iso(iar36)) = 5.2702831822567062e-5_dp
         xin(net_iso(ife52)) = 3.7866504131935185e-5_dp
         xin(net_iso(icr48)) = 5.8401123037667974e-6_dp
         xin(net_iso(ineut)) = 4.9141227703118397e-6_dp
         xin(net_iso(iti44)) = 1.5085038561154746e-6_dp
         xin(net_iso(io16)) = 9.5384019609049255e-7_dp
         xin(net_iso(img24)) = 6.3808207717725580e-7_dp
         xin(net_iso(ic12)) = 2.9048656673991868e-7_dp
         xin(net_iso(ine20)) = 9.6468865023609427e-9_dp
         xin(net_iso(ihe3)) = 4.6203603862263096e-80_dp
         xin(net_iso(in14)) = 7.5867472225841235e-99_dp
      case ('approx21_cr60_plus_co56.net')
         xin(net_iso(ihe4)) = 3.4555392534813939e-1_dp
         xin(net_iso(io16)) = 1.9367778420430937e-1_dp
         xin(net_iso(ih1)) = 1.9052931367172501e-1_dp
         xin(net_iso(isi28)) = 9.1023936032064601e-2_dp
         xin(net_iso(ini56)) = 5.8492459653295005e-2_dp
         xin(net_iso(is32)) = 4.1416642476999908e-2_dp
         xin(net_iso(ine20)) = 2.0927782332477558e-2_dp
         xin(net_iso(ic12)) = 2.0589617104448312e-2_dp
         xin(net_iso(img24)) = 1.8975914108406104e-2_dp
         xin(net_iso(iar36)) = 7.0600338785939956e-3_dp
         xin(net_iso(ica40)) = 4.9182171981353726e-3_dp
         xin(net_iso(ife52)) = 3.1618820806005280e-3_dp
         xin(net_iso(in14)) = 1.9878460082760952e-3_dp
         xin(net_iso(ife56)) = 7.1668776621130190e-4_dp
         xin(net_iso(ico56)) = 4.4974971099921181e-4_dp
         xin(net_iso(ife54)) = 3.3432423890988422e-4_dp
         xin(net_iso(icr48)) = 1.7602626907769762e-4_dp
         xin(net_iso(ihe3)) = 5.1650097191631545e-6_dp
         xin(net_iso(iti44)) = 2.6452422203389906e-6_dp
         xin(net_iso(icr60)) = 4.7653353095937982e-8_dp
         xin(net_iso(iprot)) = 1.2739022407246250e-11_dp
         xin(net_iso(ineut)) = 0.0_dp
      case default
         stop 'unrecognized net_name'
      end select

      xsum = 0.0_dp
      do i = 1, species
         if (xin(i) < 1.0e-99_dp) xin(i) = 1.0e-99_dp
         xsum = xsum + xin(i)
      end do
      xin(adjustment_iso) = xin(adjustment_iso) + (1.0_dp - xsum)
      if (xin(adjustment_iso) < 0.0_dp) stop 'negative adjustment_iso mass fraction'
   end subroutine set_case_composition

   subroutine emit_net_case(unit_no, rc, net_name, logT_in, logRho_in, eta_in)
      integer, intent(in) :: unit_no
      integer, intent(inout) :: rc
      character(len=*), intent(in) :: net_name
      real(dp), intent(in) :: logT_in, logRho_in, eta_in

      type(Net_General_Info), pointer :: g
      type(Net_Info) :: n
      integer :: handle, info, species, num_reactions, i, j, c, r
      real(dp) :: logT, temp, logRho, rho, eta, d_eta_dlnT, d_eta_dlnRho
      real(dp) :: xh, xhe, z, abar, zbar, z2bar, z53bar, ye, mass_correction, xsum
      real(dp) :: weak_rate_factor, eps_nuc, d_eps_nuc_dRho, d_eps_nuc_dT, eps_neu_total
      real(dp) :: eps_nuc_categories(num_categories)
      real(dp), allocatable :: xin(:), d_eps_nuc_dx(:), dxdt(:)
      real(dp), allocatable :: d_dxdt_dRho(:), d_dxdt_dT(:), d_dxdt_dx(:,:)
      real(dp), allocatable :: dabar_dx(:), dzbar_dx(:), dmc_dx(:)
      real(dp), pointer :: rate_factors(:)
      character(len=16) :: idx_buf

      info = 0
      handle = alloc_net_handle(info)
      if (info /= 0) stop 'alloc_net_handle failed'
      call net_start_def(handle, info)
      if (info /= 0) stop 'net_start_def failed'
      call read_net_file(trim(net_name), handle, info)
      if (info /= 0) stop 'read_net_file failed'
      call net_finish_def(handle, info)
      if (info /= 0) stop 'net_finish_def failed'
      call net_setup_tables(handle, '', info)
      if (info /= 0) stop 'net_setup_tables failed'
      call get_net_ptr(handle, g, info)
      if (info /= 0) stop 'get_net_ptr failed'

      species = g%num_isos
      num_reactions = g%num_reactions
      allocate(xin(species), d_eps_nuc_dx(species), dxdt(species), &
         d_dxdt_dRho(species), d_dxdt_dT(species), d_dxdt_dx(species, species), &
         dabar_dx(species), dzbar_dx(species), dmc_dx(species), &
         rate_factors(num_reactions))

      call set_case_composition(net_name, g%net_iso, species, xin)
      call composition_info( &
         species, g%chem_id, xin, xh, xhe, z, abar, zbar, z2bar, z53bar, ye, &
         mass_correction, xsum, dabar_dx, dzbar_dx, dmc_dx)

      logT = logT_in
      logRho = logRho_in
      temp = exp10(logT)
      rho = exp10(logRho)
      eta = eta_in
      d_eta_dlnT = 0.0_dp
      d_eta_dlnRho = 0.0_dp
      rate_factors(:) = 1.0_dp
      weak_rate_factor = 1.0_dp

      call net_get( &
         handle, .false., n, species, num_reactions, &
         xin, temp, logT, rho, logRho, &
         abar, zbar, z2bar, ye, eta, d_eta_dlnT, d_eta_dlnRho, &
         rate_factors, weak_rate_factor, &
         std_reaction_Qs, std_reaction_neuQs, &
         eps_nuc, d_eps_nuc_dRho, d_eps_nuc_dT, d_eps_nuc_dx, &
         dxdt, d_dxdt_dRho, d_dxdt_dT, d_dxdt_dx, &
         extended_screening, &
         eps_nuc_categories, eps_neu_total, info)
      if (info /= 0) stop 'net_get failed'

      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'temp', '-', temp)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'logT', '-', logT)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'rho', '-', rho)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'logRho', '-', logRho)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'abar', '-', abar)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'zbar', '-', zbar)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'z2bar', '-', z2bar)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'ye', '-', ye)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'eta', '-', eta)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'd_eta_dlnT', '-', d_eta_dlnT)
      call emit_row(unit_no, rc, 'INPUT', net_name, 'scalar', 'd_eta_dlnRho', '-', d_eta_dlnRho)
      do i = 1, species
         call emit_row(unit_no, rc, 'INPUT', net_name, 'x', chem_isos%name(g%chem_id(i)), '-', xin(i))
      end do

      call emit_row(unit_no, rc, 'SCALAR', net_name, 'eps_nuc', '-', '-', eps_nuc)
      call emit_row(unit_no, rc, 'SCALAR', net_name, 'd_eps_nuc_dRho', '-', '-', d_eps_nuc_dRho)
      call emit_row(unit_no, rc, 'SCALAR', net_name, 'd_eps_nuc_dT', '-', '-', d_eps_nuc_dT)
      call emit_row(unit_no, rc, 'SCALAR', net_name, 'eps_neu_total', '-', '-', eps_neu_total)

      do i = 1, species
         call emit_row(unit_no, rc, 'ISO', net_name, 'dxdt', chem_isos%name(g%chem_id(i)), '-', dxdt(i))
         call emit_row(unit_no, rc, 'ISO', net_name, 'd_dxdt_dRho', chem_isos%name(g%chem_id(i)), '-', d_dxdt_dRho(i))
         call emit_row(unit_no, rc, 'ISO', net_name, 'd_dxdt_dT', chem_isos%name(g%chem_id(i)), '-', d_dxdt_dT(i))
         call emit_row(unit_no, rc, 'ISO', net_name, 'd_eps_nuc_dx', chem_isos%name(g%chem_id(i)), '-', d_eps_nuc_dx(i))
      end do

      do i = 1, species
         do j = 1, species
            call emit_row( &
               unit_no, rc, 'JAC', net_name, 'd_dxdt_dx', &
               chem_isos%name(g%chem_id(i)), chem_isos%name(g%chem_id(j)), d_dxdt_dx(i, j))
         end do
      end do

      do c = 1, num_categories
         write(idx_buf, '(I0)') c
         call emit_row( &
            unit_no, rc, 'CAT', net_name, 'eps_nuc_categories', &
            trim(idx_buf), trim(category_name(c)), eps_nuc_categories(c))
      end do

      do r = 1, num_reactions
         write(idx_buf, '(I0)') r
         call emit_row( &
            unit_no, rc, 'RATE', net_name, 'rate_raw', &
            trim(reaction_Name(g%reaction_id(r))), trim(idx_buf), n%rate_raw(r))
         call emit_row( &
            unit_no, rc, 'RATE', net_name, 'rate_screened', &
            trim(reaction_Name(g%reaction_id(r))), trim(idx_buf), n%rate_screened(r))
      end do

      deallocate(xin, d_eps_nuc_dx, dxdt, d_dxdt_dRho, d_dxdt_dT, d_dxdt_dx, &
         dabar_dx, dzbar_dx, dmc_dx, rate_factors)
      call free_net_handle(handle)
   end subroutine emit_net_case

end program gen_net_oracles
