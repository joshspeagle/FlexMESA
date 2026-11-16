module pulse_tap_gyre_mod
   use const_def, only: dp
   use star_def, only: star_info, star_ptr
   use star_lib, only: star_write_model, star_write_pulse_data
   use gyre_mesa_m, only: RD, mode_t
   use iso_c_binding, only: c_char, c_null_char, c_int, c_double
   implicit none
   private

   public :: ptap_on_gyre_set_model
   public :: ptap_gyre_get_modes

   abstract interface
      subroutine gyre_cb_iface(md, ipar, rpar, retcode)
         import :: mode_t, RD
         implicit none
         type(mode_t), intent(in) :: md
         integer, intent(inout)   :: ipar(:)
         real(RD), intent(inout)  :: rpar(:)
         integer, intent(out)     :: retcode
      end subroutine gyre_cb_iface
   end interface

   interface
      subroutine c_ptap_ensure_dirs() bind(C, name="ptap_ensure_dirs")
      end subroutine c_ptap_ensure_dirs

      subroutine c_ptap_write_mode(code_str, eval_id, model_number, star_age, &
            l, m, n_pg, n_p, n_g, freq_uhz, freq_hz, inertia, &
            growth_rad_s, growth_hz, mod_file, pulse_file) &
            bind(C, name="ptap_write_mode")
         import :: c_char, c_int, c_double
         character(kind=c_char), intent(in) :: code_str(*)
         integer(c_int), value :: eval_id, model_number, l, m, n_pg, n_p, n_g
         real(c_double), value :: star_age, freq_uhz, freq_hz, inertia
         real(c_double), value :: growth_rad_s, growth_hz
         character(kind=c_char), intent(in) :: mod_file(*), pulse_file(*)
      end subroutine c_ptap_write_mode

      subroutine c_call_real_gyre_get_modes(l, cb, ipar, rpar)
         import :: gyre_cb_iface, RD
         integer, intent(in)     :: l
         procedure(gyre_cb_iface) :: cb
         integer, intent(inout)  :: ipar(:)
         real(RD), intent(inout) :: rpar(:)
      end subroutine c_call_real_gyre_get_modes
   end interface

   integer, save :: gyre_eval_id = 0
   integer, save :: cur_model_number = -1
   real(dp), save :: cur_star_age = -1.0_dp
   character(len=256), save :: cur_mod_rel = "none"
   character(len=256), save :: cur_pulse_rel = "none"
   procedure(gyre_cb_iface), pointer, save :: active_user_sub => null()

contains

   subroutine ptap_on_gyre_set_model(global_data, point_data, version)
      real(RD), intent(in) :: global_data(:)
      real(RD), intent(in) :: point_data(:,:)
      integer, intent(in)  :: version

      type(star_info), pointer :: s
      integer :: ierr

      gyre_eval_id = gyre_eval_id + 1
      call c_ptap_ensure_dirs()

      cur_model_number = -1
      cur_star_age = -1.0_dp
      cur_mod_rel = "none"
      cur_pulse_rel = "none"

      ierr = 0
      nullify(s)
      call star_ptr(1, s, ierr)
      if (ierr == 0 .and. associated(s)) then
         cur_model_number = s% model_number
         cur_star_age = s% star_age
         write(cur_mod_rel, '("pulse_tap/models/gyre_eval_",I6.6,"_step_",I7.7,".mod")') &
            gyre_eval_id, max(0, cur_model_number)
         write(cur_pulse_rel, '("pulse_tap/models/gyre_eval_",I6.6,"_step_",I7.7,".gyre")') &
            gyre_eval_id, max(0, cur_model_number)
         call star_write_model(s% id, trim(cur_mod_rel), ierr)
         if (ierr /= 0) cur_mod_rel = "write_failed"
         ierr = 0
         call star_write_pulse_data(s% id, 'GYRE', trim(cur_pulse_rel), &
            global_data, point_data, ierr)
         if (ierr /= 0) cur_pulse_rel = "write_failed"
      end if
      if (version == -999999) continue
   end subroutine ptap_on_gyre_set_model

   subroutine ptap_gyre_get_modes(l, user_sub, ipar, rpar)
      integer, intent(in)     :: l
      procedure(gyre_cb_iface) :: user_sub
      integer, intent(inout)  :: ipar(:)
      real(RD), intent(inout) :: rpar(:)

      procedure(gyre_cb_iface), pointer :: prev_sub

      prev_sub => active_user_sub
      active_user_sub => user_sub
      call c_call_real_gyre_get_modes(l, ptap_gyre_cb, ipar, rpar)
      active_user_sub => prev_sub
   end subroutine ptap_gyre_get_modes

   subroutine ptap_gyre_cb(md, ipar, rpar, retcode)
      type(mode_t), intent(in) :: md
      integer, intent(inout)   :: ipar(:)
      real(RD), intent(inout)  :: rpar(:)
      integer, intent(out)     :: retcode

      complex(RD) :: cf_uhz, cf_hz, cf_rad
      real(dp) :: f_uhz, f_hz, g_rad, g_hz, inert

      cf_uhz = md% freq('UHZ')
      cf_hz  = md% freq('HZ')
      cf_rad = md% freq('RAD_PER_SEC')
      f_uhz  = real(cf_uhz, dp)
      f_hz   = real(cf_hz, dp)
      g_rad  = real(aimag(cf_rad), dp)
      g_hz   = real(aimag(cf_hz), dp)
      inert  = real(md% E_norm(), dp)

      call c_ptap_write_mode( &
         "gyre" // c_null_char, &
         int(gyre_eval_id, c_int), int(cur_model_number, c_int), &
         real(cur_star_age, c_double), &
         int(md% l, c_int), int(md% m, c_int), &
         int(md% n_pg, c_int), int(md% n_p, c_int), int(md% n_g, c_int), &
         real(f_uhz, c_double), real(f_hz, c_double), real(inert, c_double), &
         real(g_rad, c_double), real(g_hz, c_double), &
         trim(cur_mod_rel) // c_null_char, &
         trim(cur_pulse_rel) // c_null_char)

      call active_user_sub(md, ipar, rpar, retcode)
   end subroutine ptap_gyre_cb

end module pulse_tap_gyre_mod
