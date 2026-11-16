module pulse_tap_adipls_mod
   use const_def, only: dp
   use star_def, only: star_info, star_ptr
   use star_lib, only: star_write_model
   use astero_def, only: num_results, el, order, em, cyclic_freq, inertia
   use iso_c_binding, only: c_char, c_null_char, c_int, c_double
   implicit none
   private

   public :: ptap_on_adipls_model
   public :: ptap_on_adipls_mode

   interface
      subroutine c_ptap_ensure_dirs() bind(C, name="ptap_ensure_dirs")
      end subroutine c_ptap_ensure_dirs

      subroutine c_ptap_write_amdl(path_str, nn, iaa_arg, data_arr, x_arr, aa_arr) &
            bind(C, name="ptap_write_amdl")
         import :: c_char, c_int, c_double
         character(kind=c_char), intent(in) :: path_str(*)
         integer(c_int), value :: nn, iaa_arg
         real(c_double), intent(in) :: data_arr(*), x_arr(*), aa_arr(*)
      end subroutine c_ptap_write_amdl

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
   end interface

   integer, save :: adipls_eval_id = 0
   integer, save :: last_model_number = -999999
   integer, save :: last_nn = -1
   real(dp), save :: last_star_age = -1.0_dp
   real(dp), save :: last_d1 = 0.0_dp, last_d2 = 0.0_dp, last_aa_end = 0.0_dp
   character(len=256), save :: cur_mod_rel = "none"
   character(len=256), save :: cur_amdl_rel = "none"

contains

   subroutine ptap_on_adipls_model(x, aa, data, nn, iaa_arg)
      integer, intent(in) :: nn, iaa_arg
      real(dp), intent(in) :: x(nn), aa(iaa_arg, nn), data(8)

      type(star_info), pointer :: s
      integer :: ierr, mnum
      real(dp) :: sage

      if (nn <= 0) return

      mnum = -1
      sage = -1.0_dp
      ierr = 0
      nullify(s)
      call star_ptr(1, s, ierr)
      if (ierr == 0 .and. associated(s)) then
         mnum = s% model_number
         sage = s% star_age
      end if

      if (adipls_eval_id > 0 .and. mnum == last_model_number .and. &
          sage == last_star_age .and. nn == last_nn .and. &
          data(1) == last_d1 .and. data(2) == last_d2 .and. &
          aa(1, nn) == last_aa_end) then
         return
      end if

      adipls_eval_id = adipls_eval_id + 1
      last_model_number = mnum
      last_star_age = sage
      last_nn = nn
      last_d1 = data(1)
      last_d2 = data(2)
      last_aa_end = aa(1, nn)

      call c_ptap_ensure_dirs()
      write(cur_mod_rel, '("pulse_tap/models/adipls_eval_",I6.6,"_step_",I7.7,".mod")') &
         adipls_eval_id, max(0, mnum)
      write(cur_amdl_rel, '("pulse_tap/models/adipls_eval_",I6.6,"_step_",I7.7,".amdl.bin")') &
         adipls_eval_id, max(0, mnum)

      if (associated(s)) then
         call star_write_model(s% id, trim(cur_mod_rel), ierr)
         if (ierr /= 0) cur_mod_rel = "write_failed"
      else
         cur_mod_rel = "no_star_ptr"
      end if

      call c_ptap_write_amdl(trim(cur_amdl_rel) // c_null_char, &
         int(nn, c_int), int(iaa_arg, c_int), data, x, aa)
   end subroutine ptap_on_adipls_model

   subroutine ptap_on_adipls_mode()
      integer :: n
      real(dp) :: f_uhz, inert

      n = num_results
      if (n <= 0) return
      f_uhz = cyclic_freq(n)
      inert = inertia(n)

      call c_ptap_write_mode( &
         "adipls" // c_null_char, &
         int(adipls_eval_id, c_int), int(last_model_number, c_int), &
         real(last_star_age, c_double), &
         int(el(n), c_int), int(em(n), c_int), &
         int(order(n), c_int), int(-9999, c_int), int(-9999, c_int), &
         real(f_uhz, c_double), real(f_uhz * 1.0e-6_dp, c_double), &
         real(inert, c_double), 0.0_c_double, 0.0_c_double, &
         trim(cur_mod_rel) // c_null_char, &
         trim(cur_amdl_rel) // c_null_char)
   end subroutine ptap_on_adipls_mode

end module pulse_tap_adipls_mod
