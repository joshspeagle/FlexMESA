! Module  : pulse_state_tap_m
! Purpose : capture, at each GYRE set_model call of an astero run, the
!           star state that MESA 26.4.1's get_gyre_data (star/private/
!           pulse_gyre.f90:38-416) reads, and the global/point arrays it
!           produced (#170, package G3b). Linked into an unmodified test
!           case with `-Wl,--wrap=__gyre_mesa_m_MOD_set_model`; the MESA
!           libraries are not rebuilt.
!
! Output: pulse_state/gyre_eval_NNNNNN.tsv, one record per line,
!   name <TAB> count <TAB> values
! with reals as IEEE-754 binary64 bit patterns in hex and integers or
! logicals (T/F) in decimal.

module pulse_state_tap_m

   use const_def, only: dp
   use star_def, only: star_info, star_ptr
   use astero_def, only: add_center_point, keep_surface_point, add_atmosphere
   use ISO_FORTRAN_ENV, only: INT64

   implicit none

   private
   public :: pstate_on_set_model

   integer, save :: eval_id = 0
   integer, save :: unit = -1

contains

   subroutine wr_r(name, v)
      character(*), intent(in) :: name
      real(dp), intent(in)     :: v(:)
      character(17), allocatable :: buf(:)
      integer :: i
      allocate(buf(SIZE(v)))
      do i = 1, SIZE(v)
         write(buf(i), '(1X,Z16.16)') TRANSFER(v(i), 0_INT64)
      end do
      write(unit, '(A,A,I0,A,*(A))') name, CHAR(9), SIZE(v), CHAR(9), (TRIM(ADJUSTL(buf(i)))//' ', i = 1, SIZE(v))
   end subroutine wr_r

   subroutine wr_i(name, v)
      character(*), intent(in) :: name
      integer, intent(in)      :: v(:)
      write(unit, '(A,A,I0,A,*(I0,:,1X))') name, CHAR(9), SIZE(v), CHAR(9), v
   end subroutine wr_i

   subroutine wr_l(name, v)
      character(*), intent(in) :: name
      logical, intent(in)      :: v
      write(unit, '(A,A,I0,A,L1)') name, CHAR(9), 1, CHAR(9), v
   end subroutine wr_l

   subroutine pstate_on_set_model(global_data, point_data, version)
      real(dp), intent(in) :: global_data(:)
      real(dp), intent(in) :: point_data(:,:)
      integer, intent(in)  :: version
      type(star_info), pointer :: s
      integer :: ierr, nz
      character(64) :: file

      eval_id = eval_id + 1
      call execute_command_line('mkdir -p pulse_state')
      write(file, '("pulse_state/gyre_eval_",I6.6,".tsv")') eval_id
      open(NEWUNIT=unit, FILE=TRIM(file), STATUS='REPLACE', ACTION='WRITE')

      ierr = 0
      nullify(s)
      call star_ptr(1, s, ierr)
      if (ierr /= 0 .OR. .NOT. ASSOCIATED(s)) stop 'pulse_state_tap: no star'
      nz = s%nz

      write(unit, '(A)') '# MESA 26.4.1 get_gyre_data inputs and outputs at a GYRE set_model call (#170, G3b)'
      write(unit, '(A)') '# mesa_version: r26.4.1'
      write(unit, '(A)') '# mesa_git_commit: c8c202662922a22b3efefa824473ad615c623afd'
      write(unit, '(A)') '# mesa_zenodo_record: 19722306'
      write(unit, '(A)') '# mesasdk_version: x86_64-linux-26.6.1'
      write(unit, '(A)') '# compiler: GCC version 15.2.0 (gfortran)'
      write(unit, '(A)') '# case: astero/astero_gyre (test_suite inlists unmodified)'
      write(unit, '(A)') '# tap: benchmarks/fortran_mesa/gyre_pulse_data/pulse_state_tap.f90 (link wrap of gyre_mesa_m::set_model)'
      write(unit, '(A)') '# values: reals as IEEE-754 binary64 bit patterns in hex; integers decimal; logicals T/F'
      call wr_i('model_number', [s%model_number])
      call wr_i('nz', [nz])
      call wr_i('gyre_data_schema', [s%gyre_data_schema])
      call wr_i('version', [version])
      call wr_l('add_center_point', add_center_point)
      call wr_l('keep_surface_point', keep_surface_point)
      call wr_l('add_atmosphere', add_atmosphere)
      call wr_l('rotation_flag', s%rotation_flag)
      call wr_l('interpolate_rho_for_pulse_data', s%interpolate_rho_for_pulse_data)
      call wr_l('add_double_points_to_pulse_data', s%add_double_points_to_pulse_data)
      call wr_l('calculate_Brunt_N2', s%calculate_Brunt_N2)
      call wr_r('threshold_grad_mu_for_double_point', [s%threshold_grad_mu_for_double_point])
      call wr_i('max_number_of_double_points', [s%max_number_of_double_points])
      call wr_r('photosphere_r', [s%photosphere_r])
      call wr_r('Teff', [s%Teff])

      call wr_r('r', s%r(1:nz))
      call wr_r('m', s%m(1:nz))
      call wr_r('m_grav', s%m_grav(1:nz))
      call wr_r('dm', s%dm(1:nz))
      call wr_r('dq', s%dq(1:nz))
      call wr_r('rmid', s%rmid(1:nz))
      call wr_r('L', s%L(1:nz))
      call wr_r('Peos', s%Peos(1:nz))
      call wr_r('rho', s%rho(1:nz))
      call wr_r('T', s%T(1:nz))
      call wr_r('mu', s%mu(1:nz))
      call wr_r('brunt_N2', s%brunt_N2(1:nz))
      call wr_r('grav', s%grav(1:nz))
      call wr_r('cgrav', s%cgrav(1:nz))
      call wr_r('gamma1', s%gamma1(1:nz))
      call wr_r('grada', s%grada(1:nz))
      call wr_r('chiT', s%chiT(1:nz))
      call wr_r('chiRho', s%chiRho(1:nz))
      call wr_r('gradT', s%gradT(1:nz))
      call wr_r('opacity', s%opacity(1:nz))
      call wr_r('d_opacity_dlnT', s%d_opacity_dlnT(1:nz))
      call wr_r('d_opacity_dlnd', s%d_opacity_dlnd(1:nz))
      call wr_r('eps_nuc', s%eps_nuc(1:nz))
      call wr_r('d_epsnuc_dlnT', s%d_epsnuc_dlnT(1:nz))
      call wr_r('d_epsnuc_dlnd', s%d_epsnuc_dlnd(1:nz))
      call wr_r('omega', s%omega(1:nz))

      call wr_r('global_data', global_data)
      call wr_i('point_data_shape', [SIZE(point_data, 1), SIZE(point_data, 2)])
      call wr_r('point_data', RESHAPE(point_data, [SIZE(point_data)]))

      close(unit)
   end subroutine pstate_on_set_model

end module pulse_state_tap_m
