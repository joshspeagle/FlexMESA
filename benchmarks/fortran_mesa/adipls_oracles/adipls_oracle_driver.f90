! Stand-alone replay of MESA 26.4.1's ADIPLS calls (#171, package A0).
!
! Reproduces the call sequence of astero/private/adipls_support.f90 in MESA
! release 26.4.1 (r26.4.1, commit c8c2026) against the ADIPLS library MESA
! builds (lib/libadipls.a), from inputs captured in a MESA run:
!
!   first call (run_adipls with first_time = .true.):
!     setup_redist (only with a redistrb file): read nn_redist, rewind,
!       istdin = unit, i_inout = 0, i_paramset = 1, isprtp = 0, call srdist
!       on empty arrays to read the redistribution parameters;
!     setup_adipls: call setups_adi, istdin = unit, call adipls on an empty
!       model to read adipls.c.pruned.in;
!   then, for every captured call (run_adipls with first_time = .false.):
!     open adipls.stdout and route istdou and istdpr to it;
!     if MESA stored a new model before the call and the case redistributes
!       the mesh, call srdist on the stored AMDL arrays (store_model_for_adipls,
!       redist_amdl);
!     set the controls exactly as set_adipls_controls does;
!     call adipls(1, ierr_param, 0, x, aa, data, nn, 6, 10).
!   spcout_adi below repeats MESA's spcout_adi (adipls_support_procs.f90)
!   and records the mode.
!
! Usage:
!   adipls_oracle_driver <adipls.c.pruned.in> <redistrb.c.pruned.in or -> \
!                        <call list> <output file> [eig_stride]
!
! Call list: one call per line, tab separated:
!   call_id  stored(T/F)  amdl_file  redist_input_file(or -)
!   para_el para_els1 para_dels para_dfsig1 para_dfsig2 para_sig1 para_sig2
!   para_dfsig para_eltrw1 para_eltrw2 para_sgtrw1 para_sgtrw2   (hex bits)
!   ipara_nsel ipara_nsig1 ipara_nsig2 ipara_itrsig ipara_nsig ipara_istsig
!   ipara_inomd1 ipara_iscan irotkr nprtkr igm1kr npgmkr
! amdl_file is the (x, aa, data) MESA passed to adipls (layout of
! astero_glue_tap.inc). With a redist input file and stored = T, the arrays are
! instead produced by srdist from that file (the store_amdl output) and also
! written to <output file>.redist_<call_id>.bin; with a redist input file and
! stored = F the arrays of the previous call are reused, as in MESA.
! The second argument (redistrb file or -) only controls the start-up
! srdist parameter read, which MESA makes whenever the case's inlist sets
! do_redistribute_mesh, independently of the per-call redistribution flag.
!
! Output records (tab separated, reals as IEEE-754 bit patterns in hex):
!   CALL  call_id ierr_param ctrl_mismatches nn
!   MODE  call_id k el order em cyclic_freq_uHz inertia
!   OBS   call_id k obs_st(1:10,nobs_st)
!   CSUM  call_id k csummm(1:50)
!   EIG   call_id k n x y(1:8,n)     every eig_stride points plus the last
!
! Environment: ADIPLS_A0_POISON_ROWS (see poison_rows) for sensitivity checks.
!
! The run writes ADIPLS's own files (adipls.stdout, ttt.adipls.*, agsm.mesa,
! adipls-status.log) into the current directory; run it in a scratch dir.

module a0drv
   implicit none
   integer, parameter :: dp = kind(1d0), i8 = selected_int_kind(18)
   integer :: out_unit = -1, cur_call = 0, eig_stride = 0, mode_k = 0
contains
   character(len=16) function hx(v)
      real(dp), intent(in) :: v
      write(hx, '(z16.16)') transfer(v, 0_i8)
   end function hx

   real(dp) function unhx(s)
      character(len=*), intent(in) :: s
      integer(i8) :: b
      read(s, '(z16)') b
      unhx = transfer(b, 1.0_dp)
   end function unhx

   subroutine read_xaa(fname, nn, data, x, aa)
      character(len=*), intent(in) :: fname
      integer, intent(out) :: nn
      real(dp), intent(out) :: data(8)
      real(dp), allocatable, intent(inout) :: x(:), aa(:,:)
      integer :: u, n4, i4
      integer(4) :: hdr(2)
      open(newunit=u, file=trim(fname), access='stream', form='unformatted', &
         status='old', action='read')
      read(u) hdr
      n4 = hdr(1); i4 = hdr(2)
      if (i4 /= 10) stop 'read_xaa: expected iaa = 10'
      if (allocated(x)) deallocate(x, aa)
      allocate(x(n4), aa(10, n4))
      read(u) data, x, aa
      close(u)
      nn = n4
      call poison_rows(aa)
   end subroutine read_xaa

   ! Sensitivity check: with ADIPLS_A0_POISON_ROWS=6,7,8,9 the listed rows of
   ! aa are overwritten with a quiet NaN after reading. MESA leaves rows 6-9
   ! of the arrays it passes uninitialised (store_amdl reads aa1(7,:), which
   ! fgong_amdl never sets), so an unchanged result shows ADIPLS does not
   ! read them on this path.
   subroutine poison_rows(aa)
      real(dp), intent(inout) :: aa(:,:)
      character(len=64) :: spec
      integer :: st, r, p, q
      call get_environment_variable('ADIPLS_A0_POISON_ROWS', spec, status=st)
      if (st /= 0 .or. len_trim(spec) == 0) return
      p = 1
      do
         q = index(spec(p:), ',')
         if (q == 0) then
            read(spec(p:), *) r
         else
            read(spec(p:p+q-2), *) r
         end if
         aa(r, :) = transfer(int(z'7FF8DEADBEEF0000', i8), 1.0_dp)
         if (q == 0) exit
         p = p + q
      end do
   end subroutine poison_rows

   subroutine write_xaa(fname, nn, data, x, aa)
      character(len=*), intent(in) :: fname
      integer, intent(in) :: nn
      real(dp), intent(in) :: data(8), x(:), aa(:,:)
      integer :: u
      open(newunit=u, file=trim(fname), access='stream', form='unformatted', &
         status='replace', action='write')
      write(u) int(nn, 4), 10_4
      write(u) data, x(1:nn), aa(:, 1:nn)
      close(u)
   end subroutine write_xaa
end module a0drv


subroutine spcout_adi(x, y, aa, data, nn, iy, iaa, ispcpr)
   ! MESA 26.4.1 astero/private/adipls_support_procs.f90:23-62, without the
   ! astero_def bookkeeping (store_new_oscillation_results only appends).
   use a0drv
   implicit none
   integer :: nn, iy, iaa, ispcpr
   real(dp) :: x(1:nn), y(1:iy,1:nn), aa(1:iaa,1:nn), data(8)
   common/cobs_param/ icobs_st, nobs_st, obs_st
   real(dp) :: csummm(50)
   common/csumma/ csummm
   integer :: icobs_st, nobs_st
   real(dp) :: obs_st(10,100000)
   real(dp), parameter :: pi = 3.1415926535897932384626433832795028841971693993751d0
   real(dp), parameter :: pi4 = 4*pi
   integer :: new_el, new_order, new_em, i, n
   real(dp) :: new_inertia, new_cyclic_freq

   new_el = int(obs_st(1,nobs_st) + 0.5_dp)
   new_order = int(obs_st(2,nobs_st) + 0.5_dp)
   new_em = csummm(38)
   new_inertia = obs_st(4,nobs_st)*pi4
   new_cyclic_freq = obs_st(3,nobs_st)

   mode_k = mode_k + 1
   write(out_unit, '(a,i0,a,i0,3(a,i0),4a)') 'MODE'//char(9), cur_call, char(9), mode_k, &
      char(9), new_el, char(9), new_order, char(9), new_em, &
      char(9), hx(new_cyclic_freq), char(9), hx(new_inertia)
   write(out_unit, '(a,i0,a,i0)', advance='no') 'OBS'//char(9), cur_call, char(9), mode_k
   do i = 1, 10
      write(out_unit, '(2a)', advance='no') char(9), hx(obs_st(i,nobs_st))
   end do
   write(out_unit, '(a)') ''
   write(out_unit, '(a,i0,a,i0)', advance='no') 'CSUM'//char(9), cur_call, char(9), mode_k
   do i = 1, 50
      write(out_unit, '(2a)', advance='no') char(9), hx(csummm(i))
   end do
   write(out_unit, '(a)') ''
   if (eig_stride > 0) then
      do n = 1, nn
         if (mod(n-1, eig_stride) /= 0 .and. n /= nn) cycle
         write(out_unit, '(a,i0,a,i0,a,i0,2a)', advance='no') 'EIG'//char(9), cur_call, &
            char(9), mode_k, char(9), n, char(9), hx(x(n))
         do i = 1, iy
            write(out_unit, '(2a)', advance='no') char(9), hx(y(i,n))
         end do
         write(out_unit, '(a)') ''
      end do
   end if
   if (.false.) print *, aa(1,1), data(1), ispcpr
end subroutine spcout_adi


subroutine modmod(x, aa, data, nn, ivarmd, iaa, imdmod)
   ! MESA 26.4.1 adipls_support_procs.f90:65-69 (empty)
   implicit none
   integer :: nn, ivarmd, iaa, imdmod
   double precision :: x(nn), aa(iaa,nn), data(8)
end subroutine modmod


subroutine resdif
   ! MESA 26.4.1 adipls_support_procs.f90:72-74 (empty)
end subroutine resdif


program adipls_oracle_driver
   use a0drv
   implicit none
   integer, parameter :: ivarmd = 6, iaa_arg = 10
   common/cadr_param/ para(12)
   real(dp) :: para
   common/cadi_param/ ipara(9)
   integer :: ipara
   common/coutpt/ nout, nprcen, iper, irotkr, nprtkr, igm1kr, npgmkr, &
      nfmode, nfmesh, ispcpr, npout, nobs_stmx, nfmscn
   integer :: nout, nprcen, iper, irotkr, nprtkr, igm1kr, npgmkr, &
      nfmode, nfmesh, ispcpr, npout, nobs_stmx, nfmscn
   common/cstdio/ istdin, istdou, istdpr, istder
   integer :: istdin, istdou, istdpr, istder
   common/comgrp/ isprtp
   integer :: isprtp

   character(len=1024) :: f_adipls, f_redist, f_calls, f_out, arg, line
   character(len=512) :: tok(40)
   integer :: i_paramset, ierr_param, i_inout, nn, nn_arg_0, nn_redist, nn_new
   integer :: u_in, u_calls, u_null, ios, ntok, i, nmis, call_id
   real(dp), allocatable :: x(:), aa(:,:), x_new(:), aa_new(:,:)
   real(dp) :: data(8), data_arg(8), x_arg0(0), aa_arg0(0,0)
   real(dp) :: want_r(12)
   integer :: want_i(12)
   logical :: do_redist, stored

   call get_command_argument(1, f_adipls)
   call get_command_argument(2, f_redist)
   call get_command_argument(3, f_calls)
   call get_command_argument(4, f_out)
   if (command_argument_count() >= 5) then
      call get_command_argument(5, arg)
      read(arg, *) eig_stride
   end if
   do_redist = trim(f_redist) /= '-'

   open(newunit=out_unit, file=trim(f_out), status='replace', action='write')

   data_arg = 0
   nn_arg_0 = 0
   nn_redist = 0
   i_inout = 0
   i_paramset = 1
   ierr_param = 0

   ! run_adipls(first_time): setup_redist, then setup_adipls
   if (do_redist) then
      open(newunit=u_in, file=trim(f_redist), action='read', status='old')
      read(u_in, *) nn_redist
      rewind(u_in)
      istdin = u_in
      i_inout = 0
      i_paramset = 1
      ierr_param = 0
      isprtp = 0
      call srdist(i_paramset, ierr_param, i_inout, &
         x_arg0, aa_arg0, data_arg, x_arg0, aa_arg0, &
         nn_arg_0, nn_arg_0, ivarmd, iaa_arg, iaa_arg)
      close(u_in)
      if (ierr_param < 0) stop 'first call to srdist failed'
   end if

   open(newunit=u_in, file=trim(f_adipls), action='read', status='old')
   call setups_adi
   nn_arg_0 = 0
   istdin = u_in
   call adipls(i_paramset, ierr_param, i_inout, &
      x_arg0, aa_arg0, data_arg, nn_arg_0, ivarmd, iaa_arg)
   close(u_in)
   if (ierr_param < 0) stop 'first call to adipls failed'

   u_null = -1
   open(newunit=u_calls, file=trim(f_calls), action='read', status='old')
   do
      read(u_calls, '(a)', iostat=ios) line
      if (ios /= 0) exit
      if (len_trim(line) == 0 .or. line(1:1) == '#') cycle
      call split_tabs(line, tok, ntok)
      if (ntok /= 28) stop 'call list: expected 28 fields'
      read(tok(1), *) call_id
      stored = tok(2)(1:1) == 'T'
      do i = 1, 12
         want_r(i) = unhx(tok(4+i))
      end do
      do i = 1, 12
         read(tok(16+i), *) want_i(i)
      end do

      ! run_adipls(first_time = .false.)
      i_inout = 0
      i_paramset = 1
      ierr_param = 0
      if (u_null > 0) then
         close(u_null)
      else
         u_null = 99
      end if
      open(unit=u_null, file='adipls.stdout')
      istdou = u_null
      istdpr = u_null

      if (trim(tok(4)) /= '-' .and. stored) then
         ! store_model_for_adipls: store_amdl output, then redist_amdl
         call read_xaa(tok(4), nn, data, x, aa)
         nn_new = nn_redist
         if (allocated(x_new)) deallocate(x_new, aa_new)
         allocate(aa_new(iaa_arg, nn_new), x_new(nn_new))
         ierr_param = 0
         call srdist(i_paramset, ierr_param, i_inout, &
            x, aa, data, x_new, aa_new, &
            nn, nn_new, ivarmd, iaa_arg, iaa_arg)
         call move_alloc(aa_new, aa)
         call move_alloc(x_new, x)
         nn = nn_new
         write(arg, '(a,".redist_",i6.6,".bin")') trim(f_out), call_id
         call write_xaa(arg, nn, data, x, aa)
         if (ierr_param < 0) stop 'srdist failed'
      else if (trim(tok(4)) == '-') then
         call read_xaa(tok(3), nn, data, x, aa)
      end if
      ! (redist input given and no new store: MESA reuses the arrays in memory)

      ! set_adipls_controls
      para(1) = want_r(1)    ! para_el = dble(el)
      ipara(1) = want_i(1)   ! nsel
      para(2) = want_r(2)    ! els1
      para(3) = want_r(3)    ! dels
      ipara(4) = want_i(4)   ! itrsig
      ipara(8) = want_i(8)   ! iscan
      para(6) = want_r(6)    ! sig1
      para(7) = want_r(7)    ! sig2
      para(8) = want_r(8)    ! dfsig
      ipara(5) = want_i(5)   ! nsig
      irotkr = want_i(9)
      nprtkr = want_i(10)
      igm1kr = want_i(11)
      npgmkr = want_i(12)

      ! every control variable MESA's commons hold must equal the capture
      nmis = 0
      do i = 1, 12
         if (transfer(para(i), 0_i8) /= transfer(want_r(i), 0_i8)) nmis = nmis + 1
      end do
      do i = 1, 8
         if (ipara(i) /= want_i(i)) nmis = nmis + 1
      end do

      cur_call = call_id
      mode_k = 0
      call adipls(i_paramset, ierr_param, i_inout, &
         x, aa, data, nn, ivarmd, iaa_arg)
      write(out_unit, '(a,i0,a,i0,a,i0,a,i0)') 'CALL'//char(9), call_id, char(9), ierr_param, &
         char(9), nmis, char(9), nn
      flush(out_unit)
      if (ierr_param < 0) exit
   end do
   close(u_calls)
   close(out_unit)

contains

   subroutine split_tabs(s, t, n)
      character(len=*), intent(in) :: s
      character(len=*), intent(out) :: t(:)
      integer, intent(out) :: n
      integer :: p, q
      n = 0
      p = 1
      do
         q = index(s(p:), char(9))
         n = n + 1
         if (q == 0) then
            t(n) = s(p:len_trim(s))
            exit
         end if
         t(n) = s(p:p+q-2)
         p = p + q
      end do
   end subroutine split_tabs

end program adipls_oracle_driver
