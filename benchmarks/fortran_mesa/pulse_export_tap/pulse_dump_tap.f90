! Pulse-export state tap for MESA 26.4.1 (#491). Linked with
! -Wl,--wrap=__pulse_MOD_export_pulse_data through pulse_dump_wrap.c, it
! writes, before each export_pulse_data call (star/private/pulse.f90:43-86),
! every star_info field the GYRE, FGONG and OSC producers read, as IEEE-754
! hexadecimal, to pulse_dump/call_NNNN.dump; the wrapper then copies the
! file MESA wrote to pulse_dump/call_NNNN.<format>. No MESA source is
! modified and no arithmetic is changed.
module pulse_dump_tap
  use star_lib, only: star_ptr
  use star_def, only: star_info
  use const_def, only: dp
  use chem_def, only: chem_isos
  implicit none
  private
  public :: dump_pulse_state
  integer, save :: n_calls = 0
  integer, parameter :: max_calls = 12
contains

  subroutine put(iu, name, v)
    integer, intent(in) :: iu
    character(*), intent(in) :: name
    real(dp), intent(in) :: v(:)
    integer :: i
    write(iu, '(a,1x,i0)') name, size(v)
    do i = 1, size(v)
       write(iu, '(z16.16)') v(i)
    end do
  end subroutine put

  subroutine dump_pulse_state(id, data_format, call_no) bind(C, name='pulse_dump_tap_state')
    use iso_c_binding, only: c_int, c_char
    integer(c_int), intent(in) :: id
    character(kind=c_char), intent(in) :: data_format(8)
    integer(c_int), intent(out) :: call_no
    type(star_info), pointer :: s
    integer :: ierr, iu, k, j, nz
    character(len=64) :: fname
    real(dp), allocatable :: tmp(:)
    call_no = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
    n_calls = n_calls + 1
    if (n_calls > max_calls) return
    call_no = n_calls
    nz = s% nz
    call execute_command_line('mkdir -p pulse_dump')
    write(fname, '(a,i4.4,a)') 'pulse_dump/call_', n_calls, '.dump'
    open(newunit=iu, file=trim(fname), status='replace', action='write')
    write(iu, '(a,8a1)') 'format ', data_format
    write(iu, '(a,1x,i0)') 'nz', nz
    write(iu, '(a,1x,i0)') 'species', s% species
    do j = 1, s% species
       write(iu, '(a,1x,a)') 'iso', trim(chem_isos% name(s% chem_id(j)))
    end do
    write(iu, '(a,1x,i0)') 'model_number', s% model_number
    write(iu, '(a,1x,i0)') 'gyre_data_schema', s% gyre_data_schema
    write(iu, '(a,1x,i0)') 'fgong_ivers', s% fgong_ivers
    write(iu, '(a,1x,l1)') 'rotation_flag', s% rotation_flag
    write(iu, '(a,1x,l1)') 'interpolate_rho_for_pulse_data', s% interpolate_rho_for_pulse_data
    write(iu, '(a,1x,l1)') 'add_double_points_to_pulse_data', s% add_double_points_to_pulse_data
    write(iu, '(a,1x,i0)') 'max_number_of_double_points', s% max_number_of_double_points
    write(iu, '(a,1x,l1)') 'calculate_Brunt_N2', s% calculate_Brunt_N2
    if (s% largest_conv_mixing_region /= 0) then
       write(iu, '(a,1x,i0)') 'largest_conv_mixing_region_bottom', &
          s% mixing_region_bottom(s% largest_conv_mixing_region)
    else
       write(iu, '(a,1x,i0)') 'largest_conv_mixing_region_bottom', 0
    end if
    write(iu, '(a)') 'mixing_type'
    write(iu, '(*(i0,1x))') (s% mixing_type(k), k=1,nz)
    call put(iu, 'photosphere_r', [s% photosphere_r])
    call put(iu, 'Teff', [s% Teff])
    call put(iu, 'star_age', [s% star_age])
    call put(iu, 'initial_z', [s% initial_z])
    call put(iu, 'initial_y', [s% initial_y])
    call put(iu, 'mixing_length_alpha', [s% mixing_length_alpha])
    call put(iu, 'threshold_grad_mu_for_double_point', [s% threshold_grad_mu_for_double_point])
    call put(iu, 'fgong_zero_A_inside_r', [s% fgong_zero_A_inside_r])
    call put(iu, 'r', s% r(1:nz))
    call put(iu, 'm', s% m(1:nz))
    call put(iu, 'm_grav', s% m_grav(1:nz))
    call put(iu, 'dm', s% dm(1:nz))
    call put(iu, 'dq', s% dq(1:nz))
    call put(iu, 'rmid', s% rmid(1:nz))
    call put(iu, 'L', s% L(1:nz))
    call put(iu, 'Peos', s% Peos(1:nz))
    call put(iu, 'rho', s% rho(1:nz))
    call put(iu, 'T', s% T(1:nz))
    call put(iu, 'mu', s% mu(1:nz))
    call put(iu, 'brunt_N2', s% brunt_N2(1:nz))
    call put(iu, 'grav', s% grav(1:nz))
    call put(iu, 'cgrav', s% cgrav(1:nz))
    call put(iu, 'gamma1', s% gamma1(1:nz))
    call put(iu, 'grada', s% grada(1:nz))
    call put(iu, 'chiT', s% chiT(1:nz))
    call put(iu, 'chiRho', s% chiRho(1:nz))
    call put(iu, 'gradT', s% gradT(1:nz))
    call put(iu, 'opacity', s% opacity(1:nz))
    call put(iu, 'd_opacity_dlnT', s% d_opacity_dlnT(1:nz))
    call put(iu, 'd_opacity_dlnd', s% d_opacity_dlnd(1:nz))
    call put(iu, 'eps_nuc', s% eps_nuc(1:nz))
    call put(iu, 'd_epsnuc_dlnT', s% d_epsnuc_dlnT(1:nz))
    call put(iu, 'd_epsnuc_dlnd', s% d_epsnuc_dlnd(1:nz))
    call put(iu, 'omega', s% omega(1:nz))
    call put(iu, 'X', s% X(1:nz))
    call put(iu, 'Y', s% Y(1:nz))
    call put(iu, 'cp', s% cp(1:nz))
    call put(iu, 'lnfree_e', s% lnfree_e(1:nz))
    allocate(tmp(nz))
    do k = 1, nz
       tmp(k) = s% eps_grav_ad(k)% val
    end do
    call put(iu, 'eps_grav', tmp)
    call put(iu, 'Pgas', s% Pgas(1:nz))
    call put(iu, 'gradr', s% gradr(1:nz))
    do j = 1, s% species
       do k = 1, nz
          tmp(k) = s% dxdt_nuc(j, k)
       end do
       call put(iu, 'dxdt_nuc:' // trim(chem_isos% name(s% chem_id(j))), tmp)
       do k = 1, nz
          tmp(k) = s% xa(j, k)
       end do
       call put(iu, 'xa:' // trim(chem_isos% name(s% chem_id(j))), tmp)
    end do
    close(iu)
  end subroutine dump_pulse_state

end module pulse_dump_tap
