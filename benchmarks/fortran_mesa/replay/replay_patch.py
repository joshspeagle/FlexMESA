# SPDX-License-Identifier: GPL-3.0-or-later
"""Inserts record/replay capture calls into MESA 26.4.1 ``star/private``.

The patch adds output only. It never changes an expression, a branch taken,
or the order of any arithmetic of MESA: after each of the four kernel calls
in ``Get_results`` (``star/private/turb_support.f90:328-334`` ``set_TDC``,
``:349-355`` ``set_TDC`` with ``gradr_scaled``, ``:366-369`` ``set_MLT``,
``:384-387`` ``set_MLT`` with ``gradr_scaled``) it inserts one call to a
writer that receives the kernel's own argument list, unchanged, and appends
one record holding every input and every output of that call. Before each
``set_TDC`` call it copies ``s% tdc_num_iters(k)`` to a local, because
``get_TDC_solution`` leaves its ``intent(out)`` iteration count unassigned on
its radiative early exits (``turb/private/tdc.f90:55,113``) and the caller's
value then passes through. After the ``Get_results`` call in
``do1_mlt_eval`` (``:188-192``) it inserts one more writer call (family
``GETR``, site 5) with that call's arguments and results: the values the
caller assembled, which ``FlexMESA``'s tap (``FLEXMESA_REPLAY_TAP``) writes
for its own run so the two can be compared call by call.

It also inserts, after the cell EOS call of ``do_eos_for_cell``
(``star/private/micro.f90:357-361``), one writer call (family ``EOS``,
site 6) with that ``get_eos`` call's inputs and results. The shared output
unit, model window and field writers are appended to ``star_utils.f90``
(``fm_replay_unit``, ``fm_put_*``), which ``star/make/makefile_base``
compiles before ``turb_support.f90`` and ``micro.f90``.

A record is written only when the environment variable ``FLEXMESA_REPLAY``
names an output file, and only for the first ``FLEXMESA_REPLAY_MODELS``
(default 1) distinct model numbers of each star, starting at model number
``FLEXMESA_REPLAY_FIRST`` when that is set. Every ``./rn`` stage is a
separate process and appends a ``RUN`` line and the ``SCHEMA`` lines.
``Get_results`` runs inside the ``!$OMP PARALLEL DO`` of ``set_mlt_vars``
(``turb_info.f90:52``), so the writer holds a named critical section and
writes each record with a single ``write`` statement; record order across
faces therefore depends on thread scheduling, and each record is
self-contained.

Line format (fields separated by one space)::

    RUN
    SCHEMA <family> <name>:<type> ...
    <family> <field> <field> ...

with ``<family>`` one of ``TDC``, ``MLT``, ``GETR`` and ``EOS`` and types

* ``i``: integer (``I0``);
* ``l``: logical (``T`` or ``F``);
* ``s``: string without blanks;
* ``r``: ``real(dp)`` as its IEEE-754 binary64 bit pattern (``Z16.16``);
* ``a``: ``auto_diff_real_star_order1``: the value's bit pattern followed by
  ``/<lane>:<bits>`` for every ``d1Array`` lane (1-based, of 33) whose bit
  pattern is not ``+0.0``; a lane that holds ``-0.0`` is written.
* ``v``: a ``real(dp)`` vector: the bit patterns of its elements joined by
  commas (``EOS`` results ``res(1:26)``, ``d_dlnd``, ``d_dlnT``, and
  ``d_dxa(1:2, 1:species)`` in column-major order).

The ``SCHEMA`` lines below are the single definition of field order; the
replay tool checks them exactly.

Usage::

    python replay_patch.py <overlay>/star/private
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "FLEXMESA_REPLAY"

# Field schema per family: context, kernel inputs (argument order), then
# kernel outputs. Names follow the upstream dummy arguments.
CONTEXT = "site:i id:i model:i solver_iter:i k:i"
TDC_SCHEMA = (
    f"{CONTEXT} tdc_num_iters_in:i "
    "conv_vel_start:r mixing_length_alpha:r TDC_alpha_D:r TDC_alpha_R:r "
    "TDC_alpha_Pt:r dt:r cgrav:r m:r scale:r chiT:a chiRho:a gradr:a r:a "
    "P:a T:a rho:a dV:a Cp:a opacity:a scale_height:a gradL:a grada:a "
    "max_conv_vel:r Eq_div_w:a grav:a include_mlt_corr_to_TDC:l "
    "TDC_alpha_C:r TDC_alpha_S:r use_TDC_enthalpy_flux_limiter:l energy:a "
    "out_mixing_type:i out_conv_vel:a out_D:a out_Y_face:a out_gradT:a "
    "out_tdc_num_iters:i out_ierr:i"
)
MLT_SCHEMA = (
    f"{CONTEXT} "
    "MLT_option:s mixing_length_alpha:r Henyey_MLT_nu_param:r "
    "Henyey_MLT_y_param:r chiT:a chiRho:a Cp:a grav:a Lambda:a rho:a P:a "
    "T:a opacity:a gradr:a grada:a gradL:a max_conv_vel:r "
    "out_Gamma:a out_gradT:a out_Y_face:a out_conv_vel:a out_D:a "
    "out_mixing_type:i out_ierr:i"
)

GETR_SCHEMA = (
    f"{CONTEXT} "
    "MLT_option:s r:a L:a T:a P:a opacity:a rho:a dV:a chiRho:a chiT:a Cp:a "
    "gradr:a grada:a scale_height:a iso:i XH1:r cgrav:r m:r "
    "gradL_composition_term:r mixing_length_alpha:r alpha_semiconvection:r "
    "thermohaline_coeff:r energy:a out_mixing_type:i out_gradT:a "
    "out_Y_face:a out_conv_vel:a out_D:a out_Gamma:a out_ierr:i"
)

EOS_SCHEMA = (
    f"{CONTEXT} "
    "chem_id:s xa:v Rho:r logRho:r T:r logT:r out_res:v out_d_dlnd:v "
    "out_d_dlnT:v out_d_dxa:v out_ierr:i"
)

# (start anchor of the call statement, last line of the statement, family,
# site id). Site ids: 1 and 2 are the plain and superad-reduction set_TDC
# calls, 3 and 4 the plain and superad-reduction set_MLT calls.
CALLS: list[tuple[str, str, str, int]] = [
    (
        "\n         call set_TDC(&\n"
        "            conv_vel_start, mixing_length_alpha,",
        "energy, ierr)\n",
        "tdc",
        1,
    ),
    (
        "\n               call set_TDC(&\n"
        "                  conv_vel_start, mixing_length_alpha,",
        "energy, ierr)\n",
        "tdc",
        2,
    ),
    (
        "\n         call set_MLT(MLT_option, mixing_length_alpha,",
        "mixing_type, max_conv_vel, ierr)\n",
        "mlt",
        3,
    ),
    (
        "\n               call set_MLT(MLT_option, mixing_length_alpha,",
        "mixing_type, max_conv_vel, ierr)\n",
        "mlt",
        4,
    ),
    (
        "\n         call Get_results(s, k, MLT_option, &\n",
        "mixing_type, gradT, Y_face, mlt_vc, D, Gamma, energy, ierr)\n",
        "getr",
        5,
    ),
]

DECL_ANCHOR = (
    "      logical ::  test_partials, using_TDC\n"
    "      logical, parameter :: report = .false.\n"
)
DECL = "      integer :: fm_iters_in\n"

# Module procedures appended before ``end module turb_support``. They read
# their arguments and write; they assign nothing MESA reads.
# Shared procedures appended to ``star_utils`` (compiled before
# ``turb_support`` and ``micro``, ``star/make/makefile_base``), so every
# capture site writes through one unit and one model window. They read
# their arguments and write; they assign nothing MESA reads.
COMMON = f"""
   ! Added by the {MARKER} overlay (FlexMESA record/replay capture):
   ! output only, no arithmetic of MESA.
   integer function fm_replay_unit(s) result(u)
      ! Output unit for a record of star s, or 0 when none is written.
      ! Called only inside the fm_replay critical section.
      type (star_info), pointer :: s
      integer, parameter :: max_ids = 16
      integer, save :: unit = 0, nmax = 1, first_model = -huge(1)
      logical, save :: initialised = .false.
      integer, save :: nseen(max_ids) = 0, last_model(max_ids) = -huge(1)
      character(len=1024) :: path
      integer :: st
      u = 0
      if (.not. initialised) then
         initialised = .true.
         call get_environment_variable('{MARKER}', path, status=st)
         if (st == 0 .and. len_trim(path) > 0) then
            open(newunit=unit, file=trim(path), status='unknown', &
               position='append', action='write')
            write(unit, '(A)') 'RUN'
            write(unit, '(A)') 'SCHEMA TDC {TDC_SCHEMA}'
            write(unit, '(A)') 'SCHEMA MLT {MLT_SCHEMA}'
            write(unit, '(A)') 'SCHEMA GETR {GETR_SCHEMA}'
            write(unit, '(A)') 'SCHEMA EOS {EOS_SCHEMA}'
            call get_environment_variable('{MARKER}_MODELS', path, status=st)
            if (st == 0 .and. len_trim(path) > 0) read(path, *) nmax
            call get_environment_variable('{MARKER}_FIRST', path, status=st)
            if (st == 0 .and. len_trim(path) > 0) read(path, *) first_model
         end if
      end if
      if (unit == 0 .or. s% id < 1 .or. s% id > max_ids) return
      if (s% model_number < first_model) return
      if (s% model_number /= last_model(s% id)) then
         if (nseen(s% id) >= nmax) return
         nseen(s% id) = nseen(s% id) + 1
         last_model(s% id) = s% model_number
      end if
      u = unit
   end function fm_replay_unit


   subroutine fm_put_i(buf, n, i)
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      integer, intent(in) :: i
      character(len=24) :: tmp
      write(tmp, '(I0)') i
      buf(n+1:n+1+len_trim(tmp)) = ' ' // trim(tmp)
      n = n + 1 + len_trim(tmp)
   end subroutine fm_put_i


   subroutine fm_put_l(buf, n, x)
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      logical, intent(in) :: x
      if (x) then
         buf(n+1:n+2) = ' T'
      else
         buf(n+1:n+2) = ' F'
      end if
      n = n + 2
   end subroutine fm_put_l


   subroutine fm_put_s(buf, n, x)
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      character(len=*), intent(in) :: x
      buf(n+1:n+1+len_trim(x)) = ' ' // trim(x)
      n = n + 1 + len_trim(x)
   end subroutine fm_put_s


   subroutine fm_put_r(buf, n, x)
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      real(dp), intent(in) :: x
      write(buf(n+1:n+17), '(1X,Z16.16)') transfer(x, 0_8)
      n = n + 17
   end subroutine fm_put_r


   subroutine fm_put_a(buf, n, x)
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      type(auto_diff_real_star_order1), intent(in) :: x
      character(len=40) :: tmp
      integer :: i
      call fm_put_r(buf, n, x% val)
      do i = 1, size(x% d1Array)
         if (transfer(x% d1Array(i), 0_8) /= 0_8) then
            write(tmp, '(A,I0,A,Z16.16)') '/', i, ':', &
               transfer(x% d1Array(i), 0_8)
            buf(n+1:n+len_trim(tmp)) = trim(tmp)
            n = n + len_trim(tmp)
         end if
      end do
   end subroutine fm_put_a


   subroutine fm_put_v(buf, n, x)
      ! A real vector: bit patterns joined by commas, one field.
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      real(dp), intent(in) :: x(:)
      integer :: i
      buf(n+1:n+1) = ' '
      n = n + 1
      do i = 1, size(x)
         if (i > 1) then
            buf(n+1:n+1) = ','
            n = n + 1
         end if
         write(buf(n+1:n+16), '(Z16.16)') transfer(x(i), 0_8)
         n = n + 16
      end do
   end subroutine fm_put_v


   subroutine fm_kv_name(buf, n, name)
      ! ' <name>=' of a CONFIG entry.
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      character(len=*), intent(in) :: name
      buf(n+1:n+2+len(name)) = ' ' // name // '='
      n = n + 2 + len(name)
   end subroutine fm_kv_name


   subroutine fm_kv_r(buf, n, name, x)
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      character(len=*), intent(in) :: name
      real(dp), intent(in) :: x
      call fm_kv_name(buf, n, name)
      write(buf(n+1:n+16), '(Z16.16)') transfer(x, 0_8)
      n = n + 16
   end subroutine fm_kv_r


   subroutine fm_kv_i(buf, n, name, i)
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      character(len=*), intent(in) :: name
      integer, intent(in) :: i
      character(len=24) :: tmp
      call fm_kv_name(buf, n, name)
      write(tmp, '(I0)') i
      buf(n+1:n+len_trim(tmp)) = trim(tmp)
      n = n + len_trim(tmp)
   end subroutine fm_kv_i


   subroutine fm_kv_l(buf, n, name, x)
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      character(len=*), intent(in) :: name
      logical, intent(in) :: x
      call fm_kv_name(buf, n, name)
      if (x) then
         buf(n+1:n+1) = 'T'
      else
         buf(n+1:n+1) = 'F'
      end if
      n = n + 1
   end subroutine fm_kv_l


   subroutine fm_kv_s(buf, n, name, x)
      ! A string control between single quotes; blanks inside it are
      ! written as '~' so that the entry stays one token.
      character(len=*), intent(inout) :: buf
      integer, intent(inout) :: n
      character(len=*), intent(in) :: name, x
      integer :: i
      call fm_kv_name(buf, n, name)
      buf(n+1:n+1) = "'"
      n = n + 1
      do i = 1, len_trim(x)
         if (x(i:i) == ' ') then
            buf(n+1:n+1) = '~'
         else
            buf(n+1:n+1) = x(i:i)
         end if
         n = n + 1
      end do
      buf(n+1:n+1) = "'"
      n = n + 1
   end subroutine fm_kv_s


"""

# Writers appended to ``turb_support``.
TURB_WRITERS = """
   subroutine fm_replay_tdc(s, k, site, iters_in, &
         conv_vel_start, mixing_length_alpha, TDC_alpha_D, TDC_alpha_R, &
         TDC_alpha_Pt, dt, cgrav, m, report, &
         mixing_type, scale, chiT, chiRho, gradr, r, P, T, rho, dV, Cp, &
         opacity, scale_height, gradL, grada, conv_vel, D, Y_face, gradT, &
         tdc_num_iters, &
         max_conv_vel, Eq_div_w, grav, include_mlt_corr_to_TDC, TDC_alpha_C, &
         TDC_alpha_S, use_TDC_enthalpy_flux_limiter, energy, ierr)
      ! Argument list of set_TDC (turb/public/turb.f90:114-128) after the
      ! context; every argument is read only.
      type (star_info), pointer :: s
      integer, intent(in) :: k, site, iters_in
      real(dp), intent(in) :: conv_vel_start, mixing_length_alpha, &
         TDC_alpha_D, TDC_alpha_R, TDC_alpha_Pt
      real(dp), intent(in) :: dt, cgrav, m, scale, max_conv_vel, &
         TDC_alpha_C, TDC_alpha_S
      type(auto_diff_real_star_order1), intent(in) :: &
         chiT, chiRho, gradr, r, P, T, rho, dV, Cp, opacity, scale_height, &
         gradL, grada, Eq_div_w, grav, energy
      logical, intent(in) :: report, include_mlt_corr_to_TDC, &
         use_TDC_enthalpy_flux_limiter
      type(auto_diff_real_star_order1), intent(in) :: conv_vel, Y_face, gradT, D
      integer, intent(in) :: tdc_num_iters, mixing_type, ierr
      character(len=65536) :: buf
      integer :: n, u
!$omp critical (fm_replay)
      u = fm_replay_unit(s)
      if (u /= 0) then
         buf = 'TDC'
         n = 3
         call fm_put_i(buf, n, site)
         call fm_put_i(buf, n, s% id)
         call fm_put_i(buf, n, s% model_number)
         call fm_put_i(buf, n, s% solver_iter)
         call fm_put_i(buf, n, k)
         call fm_put_i(buf, n, iters_in)
         call fm_put_r(buf, n, conv_vel_start)
         call fm_put_r(buf, n, mixing_length_alpha)
         call fm_put_r(buf, n, TDC_alpha_D)
         call fm_put_r(buf, n, TDC_alpha_R)
         call fm_put_r(buf, n, TDC_alpha_Pt)
         call fm_put_r(buf, n, dt)
         call fm_put_r(buf, n, cgrav)
         call fm_put_r(buf, n, m)
         call fm_put_r(buf, n, scale)
         call fm_put_a(buf, n, chiT)
         call fm_put_a(buf, n, chiRho)
         call fm_put_a(buf, n, gradr)
         call fm_put_a(buf, n, r)
         call fm_put_a(buf, n, P)
         call fm_put_a(buf, n, T)
         call fm_put_a(buf, n, rho)
         call fm_put_a(buf, n, dV)
         call fm_put_a(buf, n, Cp)
         call fm_put_a(buf, n, opacity)
         call fm_put_a(buf, n, scale_height)
         call fm_put_a(buf, n, gradL)
         call fm_put_a(buf, n, grada)
         call fm_put_r(buf, n, max_conv_vel)
         call fm_put_a(buf, n, Eq_div_w)
         call fm_put_a(buf, n, grav)
         call fm_put_l(buf, n, include_mlt_corr_to_TDC)
         call fm_put_r(buf, n, TDC_alpha_C)
         call fm_put_r(buf, n, TDC_alpha_S)
         call fm_put_l(buf, n, use_TDC_enthalpy_flux_limiter)
         call fm_put_a(buf, n, energy)
         call fm_put_i(buf, n, mixing_type)
         call fm_put_a(buf, n, conv_vel)
         call fm_put_a(buf, n, D)
         call fm_put_a(buf, n, Y_face)
         call fm_put_a(buf, n, gradT)
         call fm_put_i(buf, n, tdc_num_iters)
         call fm_put_i(buf, n, ierr)
         write(u, '(A)') buf(1:n)
      end if
!$omp end critical (fm_replay)
   end subroutine fm_replay_tdc


   subroutine fm_replay_mlt(s, k, site, &
         MLT_option, mixing_length_alpha, Henyey_MLT_nu_param, &
         Henyey_MLT_y_param, &
         chiT, chiRho, Cp, grav, Lambda, rho, P, T, opacity, &
         gradr, grada, gradL, &
         Gamma, gradT, Y_face, conv_vel, D, mixing_type, max_conv_vel, ierr)
      ! Argument list of set_MLT (turb/public/turb.f90:275-285) after the
      ! context; every argument is read only.
      type (star_info), pointer :: s
      integer, intent(in) :: k, site
      type(auto_diff_real_star_order1), intent(in) :: chiT, chiRho, Cp, &
         grav, Lambda, rho, P, T, opacity, gradr, grada, gradL
      character(len=*), intent(in) :: MLT_option
      real(dp), intent(in) :: mixing_length_alpha, Henyey_MLT_nu_param, &
         Henyey_MLT_y_param, max_conv_vel
      type(auto_diff_real_star_order1), intent(in) :: Gamma, gradT, &
         Y_face, conv_vel, D
      integer, intent(in) :: mixing_type, ierr
      character(len=65536) :: buf
      integer :: n, u
!$omp critical (fm_replay)
      u = fm_replay_unit(s)
      if (u /= 0) then
         buf = 'MLT'
         n = 3
         call fm_put_i(buf, n, site)
         call fm_put_i(buf, n, s% id)
         call fm_put_i(buf, n, s% model_number)
         call fm_put_i(buf, n, s% solver_iter)
         call fm_put_i(buf, n, k)
         call fm_put_s(buf, n, MLT_option)
         call fm_put_r(buf, n, mixing_length_alpha)
         call fm_put_r(buf, n, Henyey_MLT_nu_param)
         call fm_put_r(buf, n, Henyey_MLT_y_param)
         call fm_put_a(buf, n, chiT)
         call fm_put_a(buf, n, chiRho)
         call fm_put_a(buf, n, Cp)
         call fm_put_a(buf, n, grav)
         call fm_put_a(buf, n, Lambda)
         call fm_put_a(buf, n, rho)
         call fm_put_a(buf, n, P)
         call fm_put_a(buf, n, T)
         call fm_put_a(buf, n, opacity)
         call fm_put_a(buf, n, gradr)
         call fm_put_a(buf, n, grada)
         call fm_put_a(buf, n, gradL)
         call fm_put_r(buf, n, max_conv_vel)
         call fm_put_a(buf, n, Gamma)
         call fm_put_a(buf, n, gradT)
         call fm_put_a(buf, n, Y_face)
         call fm_put_a(buf, n, conv_vel)
         call fm_put_a(buf, n, D)
         call fm_put_i(buf, n, mixing_type)
         call fm_put_i(buf, n, ierr)
         write(u, '(A)') buf(1:n)
      end if
!$omp end critical (fm_replay)
   end subroutine fm_replay_mlt


   subroutine fm_replay_getr(s, k, site, MLT_option, &
         r, L, T, P, opacity, rho, dV, chiRho, chiT, Cp, gradr, grada, &
         scale_height, iso, XH1, cgrav, m, gradL_composition_term, &
         mixing_length_alpha, alpha_semiconvection, thermohaline_coeff, &
         mixing_type, gradT, Y_face, conv_vel, D, Gamma, energy, ierr)
      ! Argument list of Get_results (turb_support.f90:198-212) as called
      ! from do1_mlt_eval (:188-192), after the context; read only.
      type (star_info), pointer :: s
      integer, intent(in) :: k, site
      character (len=*), intent(in) :: MLT_option
      type(auto_diff_real_star_order1), intent(in) :: r, L, T, P, opacity, &
         rho, dV, chiRho, chiT, Cp, gradr, grada, scale_height, energy
      integer, intent(in) :: iso
      real(dp), intent(in) :: XH1, cgrav, m, gradL_composition_term, &
         mixing_length_alpha, alpha_semiconvection, thermohaline_coeff
      integer, intent(in) :: mixing_type, ierr
      type(auto_diff_real_star_order1), intent(in) :: gradT, Y_face, &
         conv_vel, D, Gamma
      character(len=65536) :: buf
      integer :: n, u
!$omp critical (fm_replay)
      u = fm_replay_unit(s)
      if (u /= 0) then
         buf = 'GETR'
         n = 4
         call fm_put_i(buf, n, site)
         call fm_put_i(buf, n, s% id)
         call fm_put_i(buf, n, s% model_number)
         call fm_put_i(buf, n, s% solver_iter)
         call fm_put_i(buf, n, k)
         call fm_put_s(buf, n, MLT_option)
         call fm_put_a(buf, n, r)
         call fm_put_a(buf, n, L)
         call fm_put_a(buf, n, T)
         call fm_put_a(buf, n, P)
         call fm_put_a(buf, n, opacity)
         call fm_put_a(buf, n, rho)
         call fm_put_a(buf, n, dV)
         call fm_put_a(buf, n, chiRho)
         call fm_put_a(buf, n, chiT)
         call fm_put_a(buf, n, Cp)
         call fm_put_a(buf, n, gradr)
         call fm_put_a(buf, n, grada)
         call fm_put_a(buf, n, scale_height)
         call fm_put_i(buf, n, iso)
         call fm_put_r(buf, n, XH1)
         call fm_put_r(buf, n, cgrav)
         call fm_put_r(buf, n, m)
         call fm_put_r(buf, n, gradL_composition_term)
         call fm_put_r(buf, n, mixing_length_alpha)
         call fm_put_r(buf, n, alpha_semiconvection)
         call fm_put_r(buf, n, thermohaline_coeff)
         call fm_put_a(buf, n, energy)
         call fm_put_i(buf, n, mixing_type)
         call fm_put_a(buf, n, gradT)
         call fm_put_a(buf, n, Y_face)
         call fm_put_a(buf, n, conv_vel)
         call fm_put_a(buf, n, D)
         call fm_put_a(buf, n, Gamma)
         call fm_put_i(buf, n, ierr)
         write(u, '(A)') buf(1:n)
      end if
!$omp end critical (fm_replay)
   end subroutine fm_replay_getr

"""


# Controls of the EOS handle (``EoS_General_Info``,
# ``eos/public/eos_def.f90``) written once per run as a ``CONFIG EOS`` line
# before the first ``EOS`` record: ``(name, type, count)`` with types as in
# the record schemas (``r``, ``l``, ``i``, ``s``) and ``count > 1`` for an
# array written element by element as ``name(i)``. The replay tool builds
# its ``EosConfig`` from this line and refuses a line with a control it does
# not know or without one it needs.
EOS_CONFIG_FIELDS: tuple[tuple[str, str, int], ...] = (
    *(
        (n, "r", 1)
        for n in (
            "Z_all_HELM",
            "logT_all_HELM",
            "logT_low_all_HELM",
            "coulomb_temp_cut_HELM",
            "coulomb_den_cut_HELM",
        )
    ),
    ("use_OPAL_SCVH", "l", 1),
    *(
        (n, "r", 1)
        for n in (
            "logT_low_all_SCVH",
            "logT_all_OPAL",
            "logRho1_OPAL_SCVH_limit",
            "logRho2_OPAL_SCVH_limit",
            "logRho_min_OPAL_SCVH_limit",
            "logQ_max_OPAL_SCVH",
            "logQ_min_OPAL_SCVH",
            "Z_all_OPAL",
        )
    ),
    ("use_FreeEOS", "l", 1),
    *(
        (n, "r", 1)
        for n in (
            "logQ_max_FreeEOS_hi",
            "logQ_max_FreeEOS_lo",
            "logQ_min_FreeEOS_hi",
            "logQ_min_FreeEOS_lo",
            "logRho_min_FreeEOS_hi",
            "logRho_min_FreeEOS_lo",
            "logRho_max_FreeEOS_hi",
            "logRho_max_FreeEOS_lo",
            "logT_min_FreeEOS_hi",
            "logT_min_FreeEOS_lo",
            "logT_max_FreeEOS_hi",
            "logT_max_FreeEOS_lo",
            "logQ_cut_FreeEOS_lo_Z_max",
            "logQ_cut_lo_Z_FreeEOS_hi",
            "logQ_cut_lo_Z_FreeEOS_lo",
            "logQ_cut_hi_Z_FreeEOS_hi",
            "logQ_cut_hi_Z_FreeEOS_lo",
            "logRho_cut_FreeEOS_hi",
            "logRho_cut_FreeEOS_lo",
            "logT_cut_FreeEOS_hi",
            "logT_cut_FreeEOS_lo",
        )
    ),
    ("suffix_for_FreeEOS_Z", "s", 15),
    ("use_CMS", "l", 1),
    ("CMS_use_fixed_composition", "l", 1),
    ("CMS_fixed_composition_index", "i", 1),
    *(
        (n, "r", 1)
        for n in (
            "max_Z_for_any_CMS",
            "max_Z_for_all_CMS",
            "logQ_max_for_any_CMS",
            "logQ_max_for_all_CMS",
            "logQ_min_for_any_CMS",
            "logQ_min_for_all_CMS",
            "logRho_max_for_all_CMS",
            "logRho_max_for_any_CMS",
            "logRho_min_for_all_CMS",
            "logRho_min_for_any_CMS",
            "logT_max_for_all_CMS",
            "logT_max_for_any_CMS",
            "logT_min_for_all_CMS",
            "logT_min_for_any_CMS",
            "logT_max_for_all_CMS_pure_He",
            "logT_max_for_any_CMS_pure_He",
        )
    ),
    ("use_PC", "l", 1),
    ("mass_fraction_limit_for_PC", "r", 1),
    ("logRho1_PC_limit", "r", 1),
    ("logRho2_PC_limit", "r", 1),
    ("PC_use_Gamma_limit_instead_of_T", "l", 1),
    *(
        (n, "r", 1)
        for n in (
            "logT1_PC_limit",
            "logT2_PC_limit",
            "log_Gamma_e_all_HELM",
            "Gamma_e_all_HELM",
            "log_Gamma_e_all_PC",
            "tiny_fuzz",
            "PC_Gamma_start_crystal",
            "PC_Gamma_full_crystal",
        )
    ),
    ("use_Skye", "l", 1),
    ("Skye_use_ion_offsets", "l", 1),
    ("mass_fraction_limit_for_Skye", "r", 1),
    ("Skye_min_gamma_for_solid", "r", 1),
    ("Skye_max_gamma_for_liquid", "r", 1),
    ("Skye_solid_mixing_rule", "s", 1),
    ("use_simple_Skye_blends", "l", 1),
    *(
        (n, "r", 1)
        for n in (
            "logRho_min_for_any_Skye",
            "logRho_min_for_all_Skye",
            "logT_min_for_any_Skye",
            "logT_min_for_all_Skye",
        )
    ),
    *(
        (n, "l", 1)
        for n in (
            "include_radiation",
            "include_elec_pos",
            "eosDT_use_linear_interp_for_X",
            "eosDT_use_linear_interp_to_HELM",
        )
    ),
    ("eosDT_file_prefix", "s", 1),
    ("okay_to_convert_ierr_to_skip", "l", 1),
    ("use_other_eos_component", "l", 1),
    ("use_other_eos_results", "l", 1),
    ("dbg", "l", 1),
    ("eos_ctrl", "r", 10),
    ("eos_integer_ctrl", "i", 10),
    ("eos_logical_ctrl", "l", 10),
    ("eos_character_ctrl", "s", 10),
)


def _eos_config_writer() -> str:
    """Fortran statements appending every ``EOS_CONFIG_FIELDS`` entry of
    the handle ``rq`` to ``buf``."""
    lines = []
    for name, kind, count in EOS_CONFIG_FIELDS:
        for i in range(1, count + 1):
            label = name if count == 1 else f"{name}({i})"
            lines.append(
                f"           call fm_kv_{kind}(buf, n, '{label}', rq% {label})"
            )
    return "\n".join(lines) + "\n"


COMMON_NAMES = (
    "fm_replay_unit, fm_put_i, fm_put_l, fm_put_s, fm_put_r, fm_put_a, "
    "fm_put_v, fm_kv_r, fm_kv_i, fm_kv_l, fm_kv_s"
)
STAR_UTILS_PUBLIC_ANCHOR = (
    "      private\n      public :: eval_min_cell_collapse_time\n"
)
STAR_UTILS_END = "      end module star_utils\n"

# The cell EOS call of ``do_eos_for_cell`` (``star/private/micro.f90:357-361``)
# and the writer appended to ``micro``.
EOS_CALL = (
    "    call get_eos( &\n"
    "         s, k, s% xa(:,k), &\n"
    "         s% rho(k), logRho, s% T(k), logT, &\n"
    "         res, s% d_eos_dlnd(:,k), s% d_eos_dlnT(:,k), &\n"
    "         s% d_eos_dxa(:,:,k), ierr)\n"
)
EOS_RECORD = (
    "    call fm_replay_eos(s, k, 6, s% xa(:,k), &\n"
    "         s% rho(k), logRho, s% T(k), logT, &\n"
    "         res, s% d_eos_dlnd(:,k), s% d_eos_dlnT(:,k), &\n"
    "         s% d_eos_dxa(:,:,k), ierr)\n"
)
MICRO_END = "end module micro\n"
EOS_CONFIG_WRITER = _eos_config_writer()
EOS_WRITER = f"""

  ! Added by the {MARKER} overlay: output only, no arithmetic of MESA.
  subroutine fm_replay_eos(s, k, site, xa, Rho, logRho, T, logT, &
        res, d_dlnd, d_dlnT, d_dxa, ierr)
     ! Arguments and results of the get_eos call in do_eos_for_cell
     ! (micro.f90:357-361); read only.
     use star_utils, only: {COMMON_NAMES}
     use eos_lib, only: eos_ptr
     use eos_def, only: EoS_General_Info
     type (star_info), pointer :: s
     integer, intent(in) :: k, site, ierr
     real(dp), intent(in) :: xa(:), Rho, logRho, T, logT
     real(dp), intent(in) :: res(:), d_dlnd(:), d_dlnT(:), d_dxa(:,:)
     character(len=65536) :: buf
     character(len=16) :: tmp
     integer :: n, u, j, ierr_rq
     type (EoS_General_Info), pointer :: rq
     logical, save :: config_written = .false.
!$omp critical (fm_replay)
     u = fm_replay_unit(s)
     if (u /= 0 .and. .not. config_written) then
        ! The handle's controls, once per run, before its first EOS record.
        config_written = .true.
        buf = 'CONFIG EOS'
        n = 10
        call eos_ptr(s% eos_handle, rq, ierr_rq)
        if (ierr_rq == 0) then
{EOS_CONFIG_WRITER}        end if
        write(u, '(A)') buf(1:n)
     end if
     if (u /= 0) then
        buf = 'EOS'
        n = 3
        call fm_put_i(buf, n, site)
        call fm_put_i(buf, n, s% id)
        call fm_put_i(buf, n, s% model_number)
        call fm_put_i(buf, n, s% solver_iter)
        call fm_put_i(buf, n, k)
        do j = 1, s% species
           write(tmp, '(I0)') s% chem_id(j)
           if (j == 1) then
              buf(n+1:n+1+len_trim(tmp)) = ' ' // trim(tmp)
              n = n + 1 + len_trim(tmp)
           else
              buf(n+1:n+1+len_trim(tmp)) = ',' // trim(tmp)
              n = n + 1 + len_trim(tmp)
           end if
        end do
        call fm_put_v(buf, n, xa(1:s% species))
        call fm_put_r(buf, n, Rho)
        call fm_put_r(buf, n, logRho)
        call fm_put_r(buf, n, T)
        call fm_put_r(buf, n, logT)
        call fm_put_v(buf, n, res)
        call fm_put_v(buf, n, d_dlnd)
        call fm_put_v(buf, n, d_dlnT)
        call fm_put_v(buf, n, reshape(d_dxa, [size(d_dxa)]))
        call fm_put_i(buf, n, ierr)
        write(u, '(A)') buf(1:n)
     end if
!$omp end critical (fm_replay)
  end subroutine fm_replay_eos

"""

END_ANCHOR = "end module turb_support\n"


def _insert_record(
    text: str, start: str, end: str, family: str, site: int
) -> str:
    """Returns ``text`` with a record call after the call statement.

    The record call repeats the kernel call's own argument text, so it
    receives exactly the values the kernel received and returned.
    """
    if text.count(start) != 1:
        count = text.count(start)
        raise ValueError(f"call anchor found {count} times:\n{start}")
    # Anchors start with the newline that ends the previous line.
    i = text.index(start) + 1
    j = text.index(end, i) + len(end)
    call = text[i:j]
    first = start[1:]
    indent = first[: len(first) - len(first.lstrip())]
    if family == "getr":
        head = "call Get_results(s, k, "
        repl = f"call fm_replay_getr(s, k, {site}, "
        return text[:i] + call + call.replace(head, repl, 1) + text[j:]
    kernel = "set_TDC" if family == "tdc" else "set_MLT"
    head = f"call {kernel}("
    if family == "tdc":
        repl = f"call fm_replay_tdc(s, k, {site}, fm_iters_in, "
        # Upstream's intent(out) tdc_num_iters may pass through unassigned
        # (see module docstring); keep the value the kernel received.
        before = f"{indent}fm_iters_in = s% tdc_num_iters(k)\n"
    else:
        repl = f"call fm_replay_mlt(s, k, {site}, "
        before = ""
    record = call.replace(head, repl, 1)
    return text[:i] + before + call + record + text[j:]


def _replace_once(text: str, anchor: str, new: str, what: str) -> str:
    """Returns ``text`` with the single ``anchor`` replaced by ``new``."""
    count = text.count(anchor)
    if count != 1:
        raise ValueError(f"{what}: anchor found {count} times")
    return text.replace(anchor, new)


def patch_turb_support(text: str) -> str:
    """Returns ``turb_support.f90`` with the TDC, MLT and GETR captures.

    Raises:
        ValueError: If an anchor does not occur exactly once, or the file is
            already patched.
    """
    if MARKER in text:
        raise ValueError("turb_support.f90 is already patched")
    for start, end, family, site in CALLS:
        text = _insert_record(text, start, end, family, site)
    text = _replace_once(
        text, DECL_ANCHOR, DECL_ANCHOR + DECL, "Get_results declarations"
    )
    return _replace_once(
        text, END_ANCHOR, TURB_WRITERS + END_ANCHOR, "end module turb_support"
    )


def patch_star_utils(text: str) -> str:
    """Returns ``star_utils.f90`` with the shared unit and field writers."""
    if MARKER in text:
        raise ValueError("star_utils.f90 is already patched")
    text = _replace_once(
        text,
        STAR_UTILS_PUBLIC_ANCHOR,
        STAR_UTILS_PUBLIC_ANCHOR + f"      public :: {COMMON_NAMES}\n",
        "star_utils public list",
    )
    return _replace_once(
        text, STAR_UTILS_END, COMMON + STAR_UTILS_END, "end module star_utils"
    )


def patch_micro(text: str) -> str:
    """Returns ``micro.f90`` with the cell EOS capture (family ``EOS``)."""
    if MARKER in text:
        raise ValueError("micro.f90 is already patched")
    text = _replace_once(text, EOS_CALL, EOS_CALL + EOS_RECORD, "get_eos call")
    return _replace_once(
        text, MICRO_END, EOS_WRITER + MICRO_END, "end module micro"
    )


# File (under ``star/private/``) to its patch.
PATCHES = {
    "star_utils.f90": patch_star_utils,
    "turb_support.f90": patch_turb_support,
    "micro.f90": patch_micro,
}


def main() -> None:
    """CLI entry point: patches every file of ``PATCHES`` under a
    ``star/private`` directory given as the argument."""
    private = pathlib.Path(sys.argv[1])
    for name, patch in PATCHES.items():
        path = private / name
        path.write_text(
            patch(path.read_text(encoding="utf-8")), encoding="utf-8"
        )
        print(f"patched {path}")


if __name__ == "__main__":
    main()
