! Program : gyre_oracle_driver
! Purpose : run MESA 26.4.1's bundled GYRE 8.1 (through its MESA interface
!           gyre_mesa_m) on a captured input model and write every mode's
!           results, eigenfunction samples and model coefficients as exact
!           IEEE-754 hex, for the FlexMESA GYRE port (#170, package G0).
!
! Usage:
!   gyre_oracle_driver <gyre.in> <model.gyre> <l list, e.g. 0,1,2> <out.tsv> <eig_stride>
!
! <model.gyre> is a pulse_tap capture: the exact global_data/point_data that
! MESA passed to set_model, written by star_write_pulse_data in format
! 1PE26.16 (17 significant digits, so the list-directed read below recovers
! the same bits). The driver mirrors what MESA does around GYRE:
! init, the set_constant calls of astero/private/gyre_support.f90:54-66 and
! star/test_suite/gyre_in_mesa_*/src/run_star_extras.f90:82-94, set_model with
! the file's schema version, and one get_modes call per requested degree in
! the given order. <eig_stride> = 0 writes no eigenfunctions, k > 0 writes
! every k-th grid point plus the last one.

module gyre_oracle_driver_m

   use const_def, only: dp, standard_cgrav, clight, crad, Msun, Rsun, Lsun
   use gyre_mesa_m
   use gyre_m, only: model_t, point_t, &
      I_V_2, I_AS, I_U, I_C_1, I_GAMMA_1, I_UPS_T, I_NABLA_AD, I_NABLA, &
      I_BETA_RAD, I_C_LUM, I_C_RAD, I_C_THN, I_C_THK, I_C_EPS, I_C_EGV, &
      I_EPS_RHO, I_EPS_T, I_KAP_RHO, I_KAP_T, I_OMEGA_ROT
   use ISO_FORTRAN_ENV, only: INT64

   implicit none

   integer, save :: out_unit = -1
   integer, save :: eig_stride = 0
   integer, save :: mode_count = 0

contains

   function hx(x) result(h)
      real(dp), intent(in) :: x
      character(16)        :: h
      write(h, '(Z16.16)') TRANSFER(x, 0_INT64)
   end function hx

   function chx(z) result(h)
      complex(dp), intent(in) :: z
      character(33)           :: h
      h = hx(REAL(z, dp)) // ' ' // hx(AIMAG(z))
   end function chx

   subroutine load_model(file, global_data, point_data, version)

      character(*), intent(in)           :: file
      real(dp), allocatable, intent(out) :: global_data(:)
      real(dp), allocatable, intent(out) :: point_data(:,:)
      integer, intent(out)               :: version

      integer :: unit, n, k, k_chk, n_col

      open(NEWUNIT=unit, FILE=file, STATUS='OLD', ACTION='READ')
      allocate(global_data(3))
      read(unit, *) n, global_data, version
      select case (version)
      case (101)
         n_col = 18
      case (120)
         n_col = 19
      case default
         stop 'gyre_oracle_driver: unsupported GYRE schema version'
      end select
      allocate(point_data(n_col, n))
      do k = 1, n
         read(unit, *) k_chk, point_data(:, k)
         if (k_chk /= k) stop 'gyre_oracle_driver: index mismatch'
      end do
      close(unit)

   end subroutine load_model

   subroutine user_sub(md, ipar, rpar, retcode)

      type(mode_t), intent(in) :: md
      integer, intent(inout)   :: ipar(:)
      real(dp), intent(inout)  :: rpar(:)
      integer, intent(out)     :: retcode

      integer, parameter :: COEFF_IDX(20) = [I_V_2, I_AS, I_U, I_C_1, I_GAMMA_1, &
         I_UPS_T, I_NABLA_AD, I_NABLA, I_BETA_RAD, I_C_LUM, I_C_RAD, I_C_THN, &
         I_C_THK, I_C_EPS, I_C_EGV, I_EPS_RHO, I_EPS_T, I_KAP_RHO, I_KAP_T, I_OMEGA_ROT]

      type(grid_t)            :: gr
      class(model_t), pointer :: ml
      integer                 :: k, i
      character(:), allocatable :: line
      character(32)           :: buf

      mode_count = mode_count + 1

      ! One MODE record: integers, then omega and the three frequency unit
      ! conversions used by MESA (complex, re/im hex), then E_norm

      write(out_unit, '(A,99(A,I0))') 'MODE', CHAR(9), mode_count, CHAR(9), md%l, CHAR(9), md%m, &
         CHAR(9), md%n_p, CHAR(9), md%n_g, CHAR(9), md%n_pg, CHAR(9), md%n, CHAR(9), md%id
      write(out_unit, '(A,A,I0,A,A)') 'OMEGA', CHAR(9), mode_count, CHAR(9), chx(md%omega)
      write(out_unit, '(A,A,I0,A,A)') 'FREQ_UHZ', CHAR(9), mode_count, CHAR(9), chx(md%freq('UHZ'))
      write(out_unit, '(A,A,I0,A,A)') 'FREQ_HZ', CHAR(9), mode_count, CHAR(9), chx(md%freq('HZ'))
      write(out_unit, '(A,A,I0,A,A)') 'FREQ_RAD_PER_SEC', CHAR(9), mode_count, CHAR(9), &
         chx(md%freq('RAD_PER_SEC'))
      write(out_unit, '(A,A,I0,A,A)') 'E_NORM', CHAR(9), mode_count, CHAR(9), hx(md%E_norm())

      if (eig_stride > 0) then

         gr = md%grid()
         ml => md%model()

         do k = 1, md%n
            if (MOD(k-1, eig_stride) /= 0 .AND. k /= md%n) cycle

            ! Eigenfunction sample: x, y_1..y_6 (complex), xi_r, xi_h, dW_dx

            write(buf, '(I0,A,I0,A,I0)') mode_count, CHAR(9), k, CHAR(9), gr%pt(k)%s
            line = 'EIG' // CHAR(9) // TRIM(buf) // CHAR(9) // hx(gr%pt(k)%x)
            do i = 1, 6
               line = line // ' ' // chx(md%y_i(i, k))
            end do
            line = line // ' ' // chx(md%xi_r(k)) // ' ' // chx(md%xi_h(k)) // ' ' // hx(md%dW_dx(k))
            write(out_unit, '(A)') line

            ! Model coefficients at the same point (only defined ones)

            line = 'COEFF' // CHAR(9) // TRIM(buf) // CHAR(9)
            do i = 1, SIZE(COEFF_IDX)
               if (i > 1) line = line // ' '
               if (ml%is_defined(COEFF_IDX(i))) then
                  line = line // hx(ml%coeff(COEFF_IDX(i), gr%pt(k)))
               else
                  line = line // 'undefined'
               end if
            end do
            write(out_unit, '(A)') line

         end do

      end if

      retcode = 0

   end subroutine user_sub

end module gyre_oracle_driver_m

program gyre_oracle_driver

   use gyre_oracle_driver_m

   implicit none

   character(4096)       :: gyre_in, model_file, l_list, out_file, arg, mesa_dir
   real(dp), allocatable :: global_data(:)
   real(dp), allocatable :: point_data(:,:)
   integer               :: version, ios, i, n_l
   integer               :: l_vals(16)
   integer               :: ipar(1)
   real(dp)              :: rpar(1)

   if (COMMAND_ARGUMENT_COUNT() /= 5) stop 'usage: gyre_oracle_driver gyre.in model.gyre l_list out.tsv eig_stride'
   call GET_COMMAND_ARGUMENT(1, gyre_in)
   call GET_COMMAND_ARGUMENT(2, model_file)
   call GET_COMMAND_ARGUMENT(3, l_list)
   call GET_COMMAND_ARGUMENT(4, out_file)
   call GET_COMMAND_ARGUMENT(5, arg)
   read(arg, *) eig_stride

   n_l = 1
   do i = 1, LEN_TRIM(l_list)
      if (l_list(i:i) == ',') n_l = n_l + 1
   end do
   read(l_list, *, IOSTAT=ios) l_vals(:n_l)
   if (ios /= 0) stop 'gyre_oracle_driver: bad l list'

   ! As MESA (astero/private/gyre_support.f90:54-66)

   call init(TRIM(gyre_in))

   call set_constant('G_GRAVITY', standard_cgrav)
   call set_constant('C_LIGHT', clight)
   call set_constant('A_RADIATION', crad)

   call set_constant('M_SUN', Msun)
   call set_constant('R_SUN', Rsun)
   call set_constant('L_SUN', Lsun)

   ! GYRE only asserts that GYRE_DIR is non-empty (gyre_mesa_m.fypp:284)
   call GET_ENVIRONMENT_VARIABLE('MESA_DIR', mesa_dir)
   call set_constant('GYRE_DIR', TRIM(mesa_dir)//'/gyre/gyre')

   call load_model(TRIM(model_file), global_data, point_data, version)
   call set_model(global_data, point_data, version)

   open(NEWUNIT=out_unit, FILE=TRIM(out_file), STATUS='REPLACE', ACTION='WRITE')

   do i = 1, n_l
      write(out_unit, '(A,A,I0)') 'GET_MODES', CHAR(9), l_vals(i)
      call get_modes(l_vals(i), user_sub, ipar, rpar)
   end do

   close(out_unit)

   call final()

end program gyre_oracle_driver
