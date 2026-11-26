# SPDX-License-Identifier: GPL-3.0-or-later
"""Inserts record/replay capture calls into MESA 26.4.1 ``turb_support.f90``.

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

with ``<family>`` one of ``TDC``, ``MLT`` and ``GETR`` and types

* ``i``: integer (``I0``);
* ``l``: logical (``T`` or ``F``);
* ``s``: string without blanks;
* ``r``: ``real(dp)`` as its IEEE-754 binary64 bit pattern (``Z16.16``);
* ``a``: ``auto_diff_real_star_order1``: the value's bit pattern followed by
  ``/<lane>:<bits>`` for every ``d1Array`` lane (1-based, of 33) whose bit
  pattern is not ``+0.0``; a lane that holds ``-0.0`` is written.

The ``SCHEMA`` lines below are the single definition of field order; the
replay tool checks them exactly.

Usage::

    python replay_patch.py <overlay>/star/private/turb_support.f90
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
HELPER = f"""
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


def patch_text(text: str) -> str:
    """Returns ``text`` with the record/replay capture inserted.

    Args:
        text: Contents of upstream ``star/private/turb_support.f90``.

    Returns:
        The patched source.

    Raises:
        ValueError: If an anchor does not occur exactly once, or the file is
            already patched.
    """
    if MARKER in text:
        raise ValueError("turb_support.f90 is already patched")
    for start, end, family, site in CALLS:
        text = _insert_record(text, start, end, family, site)
    if text.count(DECL_ANCHOR) != 1:
        raise ValueError("Get_results declaration anchor not unique")
    text = text.replace(DECL_ANCHOR, DECL_ANCHOR + DECL)
    if text.count(END_ANCHOR) != 1:
        raise ValueError("end module anchor not unique")
    return text.replace(END_ANCHOR, HELPER + END_ANCHOR)


def main() -> None:
    """CLI entry point."""
    path = pathlib.Path(sys.argv[1])
    path.write_text(
        patch_text(path.read_text(encoding="utf-8")), encoding="utf-8"
    )
    print(f"patched {path}")


if __name__ == "__main__":
    main()
