      ! Read-only capture taps for the MESA 26.4.1 ADIPLS glue (#171, package A0).
      !
      ! apply_astero_taps.py includes this file in the contains section of an
      ! overlay copy of astero/private/adipls_support.f90 and adds one call per
      ! hook. The taps only read their arguments and write files; they change no
      ! value that MESA or ADIPLS uses. They are active only when the
      ! environment variable ADIPLS_A0_TAP_DIR names an (existing or creatable)
      ! directory, relative to the case work directory.
      !
      ! Files written (all binary files little-endian, int32 counts, float64 data):
      !   calls.tsv              one line per per-model ADIPLS call (see a0tap_call)
      !   modes.tsv              one line per mode returned through spcout_adi
      !   store_NNNNNN.fgong.bin FGONG arrays from star_get_pulse_data:
      !                          int32 iconst, ivar, nn_in; glob(iconst); var(ivar,nn_in)
      !   store_NNNNNN.amdl_fgong.bin  after fgong_amdl: int32 nn, 6; data(8); aa(1:6,1:nn)
      !   store_NNNNNN.amdl_store.bin  after store_amdl: int32 nn, 10; data(8); x(nn); aa(10,nn)
      !   store_NNNNNN.amdl_redist.bin after redist_amdl (do_redistribute_mesh only),
      !                          same layout as amdl_store
      !   call_NNNNNN.amdl.bin   the (x, aa, data) passed to adipls, written whenever
      !                          it differs bytewise from the previous call's arrays;
      !                          same layout as amdl_store

      subroutine a0tap_init()
         integer :: st, ln
         if (a0tap_state /= 0) return
         a0tap_state = -1
         a0tap_dir = ''
         call get_environment_variable('ADIPLS_A0_TAP_DIR', a0tap_dir, length=ln, status=st)
         if (st /= 0 .or. ln <= 0) return
         call execute_command_line('mkdir -p ' // trim(a0tap_dir), exitstat=st)
         if (st /= 0) return
         a0tap_state = 1
      end subroutine a0tap_init


      character(len=16) function a0tap_hex(v)
         real(dp), intent(in) :: v
         write(a0tap_hex, '(z16.16)') transfer(v, 0_i8)
      end function a0tap_hex


      subroutine a0tap_store_fgong(glob, var)
         real(dp), intent(in) :: glob(:), var(:,:)
         integer :: u
         call a0tap_init()
         a0tap_store_id = a0tap_store_id + 1
         a0tap_stored_this_call = .true.
         if (a0tap_state /= 1) return
         write(a0tap_name, '(a,"/store_",i6.6,".fgong.bin")') trim(a0tap_dir), a0tap_store_id
         open(newunit=u, file=trim(a0tap_name), access='stream', form='unformatted', &
            status='replace', action='write')
         write(u) int(size(glob), kind=4), int(size(var,1), kind=4), int(size(var,2), kind=4)
         write(u) glob, var
         close(u)
      end subroutine a0tap_store_fgong


      subroutine a0tap_store_amdl6(nn_out, data_out, aa_out)
         integer, intent(in) :: nn_out
         real(dp), intent(in) :: data_out(8)
         real(dp), intent(in) :: aa_out(:,:)
         integer :: u
         if (a0tap_state /= 1) return
         write(a0tap_name, '(a,"/store_",i6.6,".amdl_fgong.bin")') trim(a0tap_dir), a0tap_store_id
         open(newunit=u, file=trim(a0tap_name), access='stream', form='unformatted', &
            status='replace', action='write')
         write(u) int(nn_out, kind=4), 6_4
         write(u) data_out, aa_out(1:6, 1:nn_out)
         close(u)
      end subroutine a0tap_store_amdl6


      subroutine a0tap_write_xaa(fname, nn_out, x_out, aa_out, data_out)
         character(len=*), intent(in) :: fname
         integer, intent(in) :: nn_out
         real(dp), intent(in) :: x_out(:), aa_out(:,:), data_out(8)
         integer :: u
         open(newunit=u, file=trim(fname), access='stream', form='unformatted', &
            status='replace', action='write')
         write(u) int(nn_out, kind=4), int(size(aa_out,1), kind=4)
         write(u) data_out, x_out(1:nn_out), aa_out(:, 1:nn_out)
         close(u)
      end subroutine a0tap_write_xaa


      subroutine a0tap_store_xaa(stage)
         character(len=*), intent(in) :: stage
         if (a0tap_state /= 1) return
         write(a0tap_name, '(a,"/store_",i6.6,".",a,".bin")') &
            trim(a0tap_dir), a0tap_store_id, stage
         call a0tap_write_xaa(a0tap_name, nn, x, aa, data)
      end subroutine a0tap_store_xaa


      ! Called in run_adipls immediately before the per-model call to adipls.
      ! Records the controls MESA passes through common blocks (exactly the
      ! variables set_adipls_controls writes) and the model arrays.
      subroutine a0tap_call(s)
         type (star_info), pointer :: s
         common/cadr_param/ &
            para_el, para_els1, para_dels, para_dfsig1, para_dfsig2, &
            para_sig1, para_sig2, para_dfsig, para_eltrw1, para_eltrw2, &
            para_sgtrw1, para_sgtrw2
         real(dp) :: &
            para_el, para_els1, para_dels, para_dfsig1, para_dfsig2, &
            para_sig1, para_sig2, para_dfsig, para_eltrw1, para_eltrw2, &
            para_sgtrw1, para_sgtrw2
         common/cadi_param/ &
            ipara_nsel, ipara_nsig1, ipara_nsig2, ipara_itrsig, ipara_nsig, &
            ipara_istsig, ipara_inomd1, ipara_iscan
         integer :: &
            ipara_nsel, ipara_nsig1, ipara_nsig2, ipara_itrsig, ipara_nsig, &
            ipara_istsig, ipara_inomd1, ipara_iscan
         common/coutpt/ &
            ipara_nout, ipara_nprcen, ipara_iper, ipara_irotkr, ipara_nprtkr, &
            ipara_igm1kr, ipara_npgmkr, ipara_nfmode, ipara_nfmesh, ipara_ispcpr, &
            ipara_npout, ipara_nobs_stmx, ipara_nfmscn
         integer :: &
            ipara_nout, ipara_nprcen, ipara_iper, ipara_irotkr, ipara_nprtkr, &
            ipara_igm1kr, ipara_npgmkr, ipara_nfmode, ipara_nfmesh, ipara_ispcpr, &
            ipara_npout, ipara_nobs_stmx, ipara_nfmscn
         real(dp) :: rp(12)
         integer :: u, i, mnum
         logical :: same
         character(len=64) :: amdl_file

         call a0tap_init()
         a0tap_call_id = a0tap_call_id + 1
         if (a0tap_state /= 1) then
            a0tap_stored_this_call = .false.
            return
         end if

         same = allocated(a0tap_last_x) .and. a0tap_last_nn == nn
         if (same) then
            same = all(transfer(a0tap_last_x, 0_i8, nn) == transfer(x(1:nn), 0_i8, nn)) &
               .and. all(transfer(a0tap_last_aa, 0_i8, iaa_arg*nn) == &
                         transfer(aa(:, 1:nn), 0_i8, iaa_arg*nn)) &
               .and. all(transfer(a0tap_last_data, 0_i8, 8) == transfer(data, 0_i8, 8))
         end if
         if (.not. same) then
            if (allocated(a0tap_last_x)) deallocate(a0tap_last_x, a0tap_last_aa)
            allocate(a0tap_last_x(nn), a0tap_last_aa(iaa_arg, nn))
            a0tap_last_x = x(1:nn)
            a0tap_last_aa = aa(:, 1:nn)
            a0tap_last_data = data
            a0tap_last_nn = nn
            write(a0tap_last_file, '("call_",i6.6,".amdl.bin")') a0tap_call_id
            write(a0tap_name, '(a,"/",a)') trim(a0tap_dir), trim(a0tap_last_file)
            call a0tap_write_xaa(a0tap_name, nn, x, aa, data)
         end if
         amdl_file = a0tap_last_file

         mnum = -1
         if (associated(s)) mnum = s% model_number
         rp = [para_el, para_els1, para_dels, para_dfsig1, para_dfsig2, &
            para_sig1, para_sig2, para_dfsig, para_eltrw1, para_eltrw2, &
            para_sgtrw1, para_sgtrw2]

         write(a0tap_name, '(a,"/calls.tsv")') trim(a0tap_dir)
         if (.not. a0tap_calls_open) then
            open(newunit=a0tap_calls_unit, file=trim(a0tap_name), status='replace', action='write')
            write(a0tap_calls_unit, '(a)') &
               'call_id'//char(9)//'model_number'//char(9)//'store_id'//char(9)// &
               'stored_before_call'//char(9)//'nn'//char(9)//'amdl_file'//char(9)// &
               'i_paramset'//char(9)//'i_inout'//char(9)//'ivarmd'//char(9)//'iaa_arg'//char(9)// &
               'para_el'//char(9)//'para_els1'//char(9)//'para_dels'//char(9)// &
               'para_dfsig1'//char(9)//'para_dfsig2'//char(9)//'para_sig1'//char(9)// &
               'para_sig2'//char(9)//'para_dfsig'//char(9)//'para_eltrw1'//char(9)// &
               'para_eltrw2'//char(9)//'para_sgtrw1'//char(9)//'para_sgtrw2'//char(9)// &
               'ipara_nsel'//char(9)//'ipara_nsig1'//char(9)//'ipara_nsig2'//char(9)// &
               'ipara_itrsig'//char(9)//'ipara_nsig'//char(9)//'ipara_istsig'//char(9)// &
               'ipara_inomd1'//char(9)//'ipara_iscan'//char(9)// &
               'irotkr'//char(9)//'nprtkr'//char(9)//'igm1kr'//char(9)//'npgmkr'
            a0tap_calls_open = .true.
         end if
         u = a0tap_calls_unit
         write(u, '(i0,a,i0,a,i0,a,l1,a,i0,a,a)', advance='no') &
            a0tap_call_id, char(9), mnum, char(9), a0tap_store_id, char(9), &
            a0tap_stored_this_call, char(9), nn, char(9), trim(amdl_file)
         write(u, '(4(a,i0))', advance='no') &
            char(9), i_paramset, char(9), i_inout, char(9), ivarmd, char(9), iaa_arg
         do i = 1, 12
            write(u, '(a,a)', advance='no') char(9), a0tap_hex(rp(i))
         end do
         write(u, '(12(a,i0))') &
            char(9), ipara_nsel, char(9), ipara_nsig1, char(9), ipara_nsig2, &
            char(9), ipara_itrsig, char(9), ipara_nsig, char(9), ipara_istsig, &
            char(9), ipara_inomd1, char(9), ipara_iscan, &
            char(9), ipara_irotkr, char(9), ipara_nprtkr, char(9), ipara_igm1kr, &
            char(9), ipara_npgmkr
         flush(a0tap_calls_unit)
         a0tap_stored_this_call = .false.
      end subroutine a0tap_call


      ! Called from spcout_adi after store_new_oscillation_results.
      subroutine a0tap_mode(obs, csummm, new_el, new_order, new_em, &
            new_inertia, new_cyclic_freq, nobs_st)
         real(dp), intent(in) :: obs(10), csummm(50), new_inertia, new_cyclic_freq
         integer, intent(in) :: new_el, new_order, new_em, nobs_st
         integer :: i, u
         call a0tap_init()
         if (a0tap_state /= 1) return
         write(a0tap_name, '(a,"/modes.tsv")') trim(a0tap_dir)
         if (.not. a0tap_modes_open) then
            open(newunit=a0tap_modes_unit, file=trim(a0tap_name), status='replace', action='write')
            write(a0tap_modes_unit, '(a)') 'call_id'//char(9)//'num_results'//char(9)// &
               'nobs_st'//char(9)//'el'//char(9)//'order'//char(9)//'em'//char(9)// &
               'cyclic_freq_uHz'//char(9)//'inertia'//char(9)// &
               'obs_st(1:10)'//char(9)//'csummm(1:50)'
            a0tap_modes_open = .true.
         end if
         u = a0tap_modes_unit
         write(u, '(i0,a,i0,a,i0,3(a,i0),4a)', advance='no') &
            a0tap_call_id, char(9), num_results, char(9), nobs_st, &
            char(9), new_el, char(9), new_order, char(9), new_em, &
            char(9), a0tap_hex(new_cyclic_freq), char(9), a0tap_hex(new_inertia)
         do i = 1, 10
            if (i == 1) then
               write(u, '(2a)', advance='no') char(9), a0tap_hex(obs(i))
            else
               write(u, '(2a)', advance='no') ' ', a0tap_hex(obs(i))
            end if
         end do
         do i = 1, 50
            if (i == 1) then
               write(u, '(2a)', advance='no') char(9), a0tap_hex(csummm(i))
            else
               write(u, '(2a)', advance='no') ' ', a0tap_hex(csummm(i))
            end if
         end do
         write(u, '(a)') ''
         flush(a0tap_modes_unit)
      end subroutine a0tap_mode
