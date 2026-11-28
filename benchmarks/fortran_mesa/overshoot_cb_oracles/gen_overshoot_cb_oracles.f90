! SPDX-License-Identifier: GPL-3.0-or-later
! Generate IEEE-754 hex reference vectors against compiled MESA 26.4.1 for
! `overshoot_utils::eval_conv_bdy_r` and `eval_conv_bdy_Hp` at the three
! convective-boundary stencils of
! `crates/flexmesa-star/tests/overshoot_hp_cb_mesa_hex_test.rs`.
!
! A five-cell `star_info` is built holding only the fields those routines
! read. The boundary cell is k = 3 (1-based); cells 2, 3, 4 hold the recorded
! stencil, and a top boundary has conv_bdy_loc = k while a bottom boundary has
! conv_bdy_loc = k + 1 (`eval_conv_bdy_k`).

program gen_overshoot_cb_oracles
   use, intrinsic :: iso_fortran_env, only: int64, f90_compiler_version => compiler_version
   use math_lib, only: MATH_BACKEND, math_init
   use const_def, only: dp
   use const_lib, only: const_init
   use star_private_def, only: star_info
   use overshoot_utils, only: eval_conv_bdy_r, eval_conv_bdy_Hp
   implicit none

   integer, parameter :: nz = 5, kb = 3
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
      out_path = 'mesa_26_4_1_overshoot_cb_oracles.tsv'
   end if

   ierr = 0
   call const_init(mesa_dir, ierr)
   if (ierr /= 0) stop 'const_init failed'
   call math_init()
   call read_ver(mesa_dir, mesa_ver)

   open(newunit=u, file=trim(out_path), status='replace', action='write', iostat=ierr)
   if (ierr /= 0) stop 'failed to open output TSV'
   write(u, '(A)') '# FlexMESA MESA 26.4.1 overshoot eval_conv_bdy_r / eval_conv_bdy_Hp oracle vectors'
   write(u, '(A)') '# mesa_version: ' // trim(adjustl(mesa_ver))
   write(u, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
   write(u, '(A)') '# mesasdk_version: ' // trim(adjustl(sdk_ver))
   write(u, '(A)') '# math_backend: ' // trim(adjustl(MATH_BACKEND))
   write(u, '(A)') '# compiler: ' // trim(f90_compiler_version())
   write(u, '(A)') '# fflags: ' // trim(adjustl(fflags_str))
   write(u, '(A)') 'case_id' // char(9) // 'role' // char(9) // 'field' // char(9) // &
      'part' // char(9) // 'raw_bits_hex' // char(9) // 'decimal'

   rc = 0
   call run_case('cepheid_2112_bdy2', .false., &
      int(z'3F4E5DBF584F5D84', int64), int(z'3FDA49D7F81A64CD', int64), &
      [int(z'3F785B3F7F9ECD80', int64), int(z'3F787F86C53A5E80', int64), int(z'3F78A348EA18E600', int64)], &
      [int(z'403753E0BAE85760', int64), int(z'403760D0C7DDD404', int64), int(z'40376E2EC9FA3E39', int64)], &
      [int(z'C021E7D753E6FC93', int64), int(z'C021D7A69A375330', int64), int(z'C021C688E323C791', int64)], &
      int(z'3E71EA8B45EDF98C', int64), int(z'0000000000000000', int64), int(z'46FEA2E4E0A40A3D', int64), &
      int(z'4270F201E6EE4522', int64), int(z'42708E61700ECEC8', int64))
   call run_case('high_z_1_bdy1', .true., &
      int(z'3F41DF208EA52C9C', int64), int(z'3FD0878AEA0B2A03', int64), &
      [int(z'3F5AD4AF36CC1FFC', int64), int(z'3F6AD4AF36CC1EFD', int64), int(z'3F6AE00C40DA78FD', int64)], &
      [int(z'404238C8CB4BAEEA', int64), int(z'404239D83BD3A28E', int64), int(z'40423B424F116DEA', int64)], &
      [int(z'3FF325DB81F33ABF', int64), int(z'3FF33ABFC757BAB5', int64), int(z'3FF356A0FC40024E', int64)], &
      int(z'3E71EA8B45EDF98C', int64), int(z'0000000000000000', int64), int(z'470572069D3FA0C4', int64), &
      int(z'422B3A4360C8EBED', int64), int(z'422B1406540622F9', int64))
   call run_case('high_z_2_bdy3', .true., &
      int(z'3E11F4391BDD032E', int64), int(z'3FEFFFFFDD1E5B66', int64), &
      [int(z'3E26DF0910F54AE0', int64), int(z'3E26DF0910F54AE0', int64), int(z'3E212746CCB7F800', int64)], &
      [int(z'402E450C3B51FAE8', int64), int(z'402E5ADD1AB05D93', int64), int(z'402E6D3DA2D8B6D6', int64)], &
      [int(z'C02ED868E0909AB8', int64), int(z'C02EC9DE2A190AA3', int64), int(z'C02EBDBA4729BAD3', int64)], &
      int(z'3E71EA8B45EDF98C', int64), int(z'0000000000000000', int64), int(z'470572063D08C39D', int64), &
      int(z'4254F6A360DD07FC', int64), int(z'4254F4FEAE3C97A8', int64))
   close(u)
   write(*, '(A,I0,A)') 'Wrote ', rc, ' overshoot oracle rows.'

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

   real(dp) function d(bits)
      integer(int64), intent(in) :: bits
      d = transfer(bits, 0.0_dp)
   end function d

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

   subroutine run_case(cid, top, b_cz, b_q, b_dq, b_lnp, b_lnd, b_cg, b_mc, b_xm, b_rk, b_rkp1)
      character(len=*), intent(in) :: cid
      logical, intent(in) :: top
      integer(int64), intent(in) :: b_cz, b_q, b_cg, b_mc, b_xm, b_rk, b_rkp1
      integer(int64), intent(in) :: b_dq(3), b_lnp(3), b_lnd(3)
      type(star_info), pointer :: s
      real(dp) :: r_cb, hp_cb
      integer :: j, ierr1, ierr2
      character(len=1) :: jc

      allocate(s)
      s%nz = nz
      allocate(s%dq(nz), s%r(nz), s%lnPeos(nz), s%lnd(nz), s%cgrav(nz), s%cz_bdy_dq(nz))
      allocate(s%conv_bdy_q(1), s%top_conv_bdy(1), s%conv_bdy_loc(1))
      s%dq = 0d0; s%lnPeos = 0d0; s%lnd = 0d0; s%cgrav = 0d0; s%cz_bdy_dq = 0d0
      do j = 1, 3
         s%dq(kb-2+j) = d(b_dq(j))
         s%lnPeos(kb-2+j) = d(b_lnp(j))
         s%lnd(kb-2+j) = d(b_lnd(j))
         s%cgrav(kb-2+j) = d(b_cg)
      end do
      s%r = [2d0*d(b_rk), 1.5d0*d(b_rk), d(b_rk), d(b_rkp1), 0.5d0*d(b_rkp1)]
      s%M_center = d(b_mc)
      s%xmstar = d(b_xm)
      s%cz_bdy_dq(kb) = d(b_cz)
      s%conv_bdy_q(1) = d(b_q)
      s%top_conv_bdy(1) = top
      if (top) then
         s%conv_bdy_loc(1) = kb
      else
         s%conv_bdy_loc(1) = kb + 1
      end if
      s%limit_overshoot_Hp_using_size_of_convection_zone = .false.

      call emit(cid, 'in', 'top', 'val', merge(1d0, 0d0, top))
      call emit(cid, 'in', 'cz_bdy_dq', 'val', s%cz_bdy_dq(kb))
      call emit(cid, 'in', 'conv_bdy_q', 'val', s%conv_bdy_q(1))
      do j = 1, 3
         write(jc, '(I1)') j - 1
         call emit(cid, 'in', 'dq' // jc, 'val', s%dq(kb-2+j))
         call emit(cid, 'in', 'lnPeos' // jc, 'val', s%lnPeos(kb-2+j))
         call emit(cid, 'in', 'lnd' // jc, 'val', s%lnd(kb-2+j))
      end do
      call emit(cid, 'in', 'cgrav', 'val', s%cgrav(kb))
      call emit(cid, 'in', 'M_center', 'val', s%M_center)
      call emit(cid, 'in', 'xmstar', 'val', s%xmstar)
      call emit(cid, 'in', 'r_k', 'val', s%r(kb))
      call emit(cid, 'in', 'r_kp1', 'val', s%r(kb+1))

      call eval_conv_bdy_r(s, 1, r_cb, ierr1)
      call eval_conv_bdy_Hp(s, 1, hp_cb, ierr2)
      call emit(cid, 'out', 'r_cb', 'val', r_cb)
      call emit(cid, 'out', 'Hp_cb', 'val', hp_cb)
      call emit(cid, 'out', 'ierr_r', 'val', real(ierr1, dp))
      call emit(cid, 'out', 'ierr_Hp', 'val', real(ierr2, dp))
      deallocate(s%dq, s%r, s%lnPeos, s%lnd, s%cgrav, s%cz_bdy_dq, s%conv_bdy_q, s%top_conv_bdy, s%conv_bdy_loc)
      deallocate(s)
   end subroutine run_case

end program gen_overshoot_cb_oracles
