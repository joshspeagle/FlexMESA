! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex reference vectors against compiled MESA 26.4.1
! (`libstar.a` + `libneu.a` + `CRMATH`) for `neu::do_neu_for_cell`
! (`star/private/neu.f90`) at the mid-run probe points of
! `crates/flexmesa-star/tests/micro_161_parity_test.rs` (#161).
!
! The one-zone star state is set as `set_basic_vars` and `do_eos_for_cell`
! leave it: `T = exp(lnT)`, `rho = exp(lnd)`, and `abar`, `zbar` from
! `basic_composition_info` on the raw `xa` (`star/private/micro.f90:321`).

program gen_micro_neu_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use math_lib, only: MATH_BACKEND, math_init, exp
   use const_def, only: dp, ln10
   use const_lib, only: const_init
   use chem_def
   use chem_lib, only: chem_init, chem_get_iso_id, basic_composition_info
   use star_def, only: star_info
   use neu, only: do_neu_for_cell
   implicit none

   character(len=512) :: mesa_dir, out_path, sdk_ver, mesa_ver, fflags_str
   integer :: ierr, u, rc
   type(star_info), pointer :: s

   call get_environment_variable('MESA_DIR', mesa_dir)
   if (len_trim(mesa_dir) == 0) stop 'MESA_DIR must be set'
   call get_environment_variable('MESASDK_VERSION_STR', sdk_ver)
   if (len_trim(sdk_ver) == 0) sdk_ver = 'unknown'
   call get_environment_variable('ORACLE_FFLAGS', fflags_str)
   if (command_argument_count() >= 1) then
      call get_command_argument(1, out_path)
   else
      out_path = 'mesa_26_4_1_micro_neu_oracles.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call chem_init('isotopes.data', ierr)
   if (ierr /= 0) stop 'chem_init failed'
   call read_ver(mesa_dir, mesa_ver)

   allocate(s)
   allocate(s% lnd(1), s% lnT(1), s% T(1), s% rho(1), s% abar(1), s% zbar(1))
   allocate(s% non_nuc_neu(1), s% d_nonnucneu_dlnd(1), s% d_nonnucneu_dlnT(1))
   allocate(s% nonnucneu_plas(1), s% nonnucneu_brem(1), s% nonnucneu_phot(1))
   allocate(s% nonnucneu_pair(1), s% nonnucneu_reco(1))
   s% non_nuc_neu_factor = 1d0
   s% use_other_neu = .false.
   s% report_ierr = .false.
   s% stop_for_bad_nums = .false.

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'
   write(u, '(A)') '# FlexMESA MESA 26.4.1 star neu do_neu_for_cell oracle vectors (#161)'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') 'case_id' // char(9) // 'role' // char(9) // 'field' // char(9) // &
      'part' // char(9) // 'raw_bits_hex' // char(9) // 'decimal'

   rc = 0
   block
      real(dp) :: xa(10), tail
      ! HIGHZ_M340
      xa = [0d0, 0d0, 0.803521d0, 0.108974d0, 0d0, 0.010230d0, 0d0, 0.006897d0, 0d0, 0d0]
      call set_tail(xa, 10)
      call run_case('HIGHZ_M340', ['h1  ','he3 ','he4 ','c12 ','n14 ','o16 ','o18 ','ne20','ne22','mg24'], &
         xa, 3.4921578275053786d0 * ln10, 8.1690433015297d0 * ln10)
   end block
   block
      real(dp) :: xa(5)
      xa = [0d0, 0.350611d0, 0.622309d0, 0.022977d0, 0d0]
      call set_tail(xa, 5)
      call run_case('WD_M370', ['he4 ','c12 ','o16 ','ne20','si28'], xa, 9.893157d0 * ln10, 8.191366d0 * ln10)
      xa = [0d0, 0.419117d0, 0.549867d0, 0.025894d0, 0d0]
      call set_tail(xa, 5)
      call run_case('WD_M668', ['he4 ','c12 ','o16 ','ne20','si28'], xa, 9.806680d0 * ln10, 8.595580d0 * ln10)
   end block
   call run_case('NS_SKYE_HE4', ['he4 ','fe56'], [1d0, 0d0], 4.95d0 * ln10, 7.30d0 * ln10)
   call run_case('NS_SKYE_FE56', ['he4 ','fe56'], [0d0, 1d0], 6.99d0 * ln10, 7.30d0 * ln10)
   close(u)
   write(*, '(A,I0,A)') 'Wrote ', rc, ' neu oracle rows.'

contains

   ! xa(n) = 1 - sequential sum of xa(1:n-1), as the test builds it.
   subroutine set_tail(xa, n)
      real(dp), intent(inout) :: xa(:)
      integer, intent(in) :: n
      real(dp) :: sm
      integer :: i
      sm = 0d0
      do i = 1, n - 1
         sm = sm + xa(i)
      end do
      xa(n) = 1d0 - sm
   end subroutine set_tail

   subroutine run_case(cid, names, xa, lnd, lnT)
      character(len=*), intent(in) :: cid
      character(len=4), intent(in) :: names(:)
      real(dp), intent(in) :: xa(:), lnd, lnT
      integer :: n, i, chem_id(size(names)), k
      real(dp) :: xh, xhe, z, z2bar, z53bar, ye, mc, sumx
      n = size(names)
      do i = 1, n
         chem_id(i) = chem_get_iso_id(trim(names(i)))
         if (chem_id(i) <= 0) stop 'unknown isotope'
         call emit(cid, 'in', 'xa_' // trim(names(i)), 'val', xa(i))
      end do
      s% lnd(1) = lnd
      s% lnT(1) = lnT
      s% rho(1) = exp(lnd)
      s% T(1) = exp(lnT)
      call basic_composition_info(n, chem_id, xa, xh, xhe, z, s% abar(1), s% zbar(1), z2bar, z53bar, ye, mc, sumx)
      call emit(cid, 'in', 'lnd', 'val', s% lnd(1))
      call emit(cid, 'in', 'lnT', 'val', s% lnT(1))
      call emit(cid, 'in', 'rho', 'val', s% rho(1))
      call emit(cid, 'in', 'T', 'val', s% T(1))
      call emit(cid, 'in', 'abar', 'val', s% abar(1))
      call emit(cid, 'in', 'zbar', 'val', s% zbar(1))
      k = 1
      call do_neu_for_cell(s, k, ierr)
      if (ierr /= 0) stop 'do_neu_for_cell failed'
      call emit(cid, 'out', 'non_nuc_neu', 'val', s% non_nuc_neu(1))
      call emit(cid, 'out', 'd_nonnucneu_dlnd', 'val', s% d_nonnucneu_dlnd(1))
      call emit(cid, 'out', 'd_nonnucneu_dlnT', 'val', s% d_nonnucneu_dlnT(1))
      call emit(cid, 'out', 'nonnucneu_plas', 'val', s% nonnucneu_plas(1))
      call emit(cid, 'out', 'nonnucneu_brem', 'val', s% nonnucneu_brem(1))
      call emit(cid, 'out', 'nonnucneu_phot', 'val', s% nonnucneu_phot(1))
      call emit(cid, 'out', 'nonnucneu_pair', 'val', s% nonnucneu_pair(1))
      call emit(cid, 'out', 'nonnucneu_reco', 'val', s% nonnucneu_reco(1))
   end subroutine run_case

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

   subroutine emit(cid, role, fld, part, val)
      character(len=*), intent(in) :: cid, role, fld, part
      real(dp), intent(in) :: val
      character(len=20) :: hbuf
      character(len=32) :: dbuf
      write(hbuf, '(A,Z16.16)') '0x', transfer(val, 0_int64)
      write(dbuf, '(ES25.17E3)') val
      write(u, '(A)') trim(cid) // char(9) // trim(role) // char(9) // trim(fld) // &
         char(9) // trim(part) // char(9) // trim(hbuf) // char(9) // trim(adjustl(dbuf))
      rc = rc + 1
   end subroutine emit

end program gen_micro_neu_oracles
