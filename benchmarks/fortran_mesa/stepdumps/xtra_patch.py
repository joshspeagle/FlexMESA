# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in diagnostic ``XTRA`` records for the MESA 26.4.1 step-dump overlay.

Two extra patches sit on top of ``stepdump_patch.py``. Both add ``write``
statements only: no expression, branch or order of arithmetic of MESA changes,
and neither is applied unless ``run_stepdumps.py`` is given ``--xtra``.

``star`` (``star/private/star_solver.f90``, written to the step-dump file)
    Tagged records ``XTRA <tag> model iter j k hex decimal`` (hex is the
    IEEE-754 binary64 bit pattern, decimal 17 significant digits). ``iter`` is
    the solver iteration of the matching ``CORR``/``RESID`` record, 0 for the
    initial residual evaluation (the ``CALL`` record); ``j`` is the equation
    or variable index, ``k`` the zone (1 = surface), 0 where not applicable.

    * ``START_call_index``: ``j`` is the running solver-call count of the
      process (a retry repeats a model).
    * ``START_M_center``, ``START_R_center``, ``START_L_center``,
      ``START_xmstar``, ``START_mstar``: ``s%`` values at solver entry.
    * ``RESID0_EQUSUM``/``RESID0_EQUMAX`` (initial evaluation) and
      ``RESID_EQUSUM``/``RESID_EQUMAX`` (after each accepted correction):
      ``sum_k |equ(j,k)|`` and ``max_k |equ(j,k)|`` with its zone, for every
      ``j``. The sums are serial loops over ``k``.
    * ``CORR_SOLNSUM``: ``sum_k |soln(j,k)|`` after ``sizeB`` (the scaled
      correction as returned by the solve), every ``j``.
    * With zones: ``RESID_EQU`` (``equ(j,k)``, every ``j``) and
      ``RESID_<name>`` for ``lnd lnT lnR L Peos opacity gradT gradr mlt_vc Cp
      energy`` at the zones of ``--xtra-zones``; ``CORR_SOLN`` and
      ``CORR_DX`` (``soln(j,k)`` and ``s% solver_dx(j,k)`` before the
      correction is applied) at the zones of ``--xtra-corr-zones``.

``binary-rlo`` (``binary/private/binary_mdot.f90``, written to a separate file)
    Records ``XTRA <tag> model rlo_iter name hex decimal`` for each iteration
    of the implicit mass-transfer solve (``rlo_iter`` is ``b% num_tries``).
    Masses and rates are in cgs, ``mdot`` negative for mass loss.

    * ``RLO_CALL``: ``curr_mdot`` (``b% mtransfer_rate``), ``explicit_mdot``,
      ``f`` (``function_to_solve``), ``change_factor``, ``tolerance``,
      ``starting_mdot``; and ``RLO_RESULT_in_tol`` when ``|f|`` is within the
      tolerance (the iteration that ends the solve).
    * ``RLO_BRACKET``/``RLO_FLAGS``: ``mdot_lo``, ``mdot_hi``, ``f_lo``,
      ``f_hi`` and ``have_mdot_lo/hi``, ``solver_type``, ``mod(num_tries,2)``,
      ``use_sum``, written after the ``num_tries == 1`` reset, so iteration 1
      shows the reset brackets. An iteration that ends in tolerance, or beyond
      ``max_tries_to_achieve``, never reaches this point and has none.
    * ``RLO_DO_CUBIC``: whether ``pick_mdot_for_implicit_rlo`` tries
      ``find0_quadratic`` (true) or bisects (false); only when both brackets
      exist.
    * ``RLO_QUAD``/``RLO_QUAD_FINAL``: the ``find0_quadratic`` result and its
      ``ierr`` before, and ``mdot_next`` and ``ierr`` after, the ``find0``
      fallback taken when ``ierr /= 0``.
    * ``RLO_NEXT``: ``next_mdot_raw`` (the pick, before the maximum-mdot clamp)
      with the brackets and flags after the update, and ``ierr``;
      ``RLO_REDO`` ``new_mdot_final`` (after the clamp) with
      ``RLO_RESULT_redo`` when the solve continues.

    Runtime variables: ``FLEXMESA_XTRA_BIN`` (output file),
    ``FLEXMESA_XTRA_FIRSTMODEL`` (first binary model written) and
    ``FLEXMESA_XTRA_STOPMODEL`` (the run stops when a later model starts).

Usage::

    python xtra_patch.py star <star_solver.f90> [--zones 1,262] [--corr-zones 1]
    python xtra_patch.py binary-rlo <binary_mdot.f90>

The ``star`` patch must be applied before ``stepdump_patch.patch_text``, which
supplies ``stepdump_unit`` and anchors on text this patch leaves intact.
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

STAR_FILE = "star/private/star_solver.f90"
BINARY_FILE = "binary/private/binary_mdot.f90"
KINDS = ("star", "binary-rlo")

# Fields written per requested zone at each residual evaluation.
ZONE_FIELDS = (
    "lnd",
    "lnT",
    "lnR",
    "L",
    "Peos",
    "opacity",
    "gradT",
    "gradr",
    "mlt_vc",
    "Cp",
    "energy",
)

# Each edit is (anchor, text, where); the anchor must occur exactly once.
STAR_EDITS: list[tuple[str, str, str]] = [
    (
        "            iter = 1\n            s% solver_iter = iter\n",
        "            call xtra_star(s, nvar, nz, 'RESID0', 0, soln)\n",
        "after",
    ),
    (
        "               call sizeB(s, nvar, soln, &\n"
        "                     max_correction, correction_norm, max_corr_k, "
        "max_corr_j, ierr)\n"
        "               if (ierr /= 0) then\n"
        "                  call oops('correction rejected by sizeB')\n"
        "                  exit iter_loop\n"
        "               end if\n",
        "               call xtra_star(s, nvar, nz, 'CORR', iter, soln)\n",
        "after",
    ),
    (
        "               call sizequ(s, nvar, residual_norm, max_residual, "
        "max_resid_k, max_resid_j, ierr)\n"
        "               if (ierr /= 0) then\n"
        "                  call oops('sizequ returned ierr')\n"
        "                  exit iter_loop\n"
        "               end if\n",
        "               call xtra_star(s, nvar, nz, 'RESID', iter, soln)\n",
        "after",
    ),
]

_STAR_HELPER = """
      subroutine xtra_w(u, tag, model, iter, j, k, v)
         integer, intent(in) :: u, model, iter, j, k
         character(len=*), intent(in) :: tag
         real(dp), intent(in) :: v
         write(u,'(A,1X,A,4(1X,I0),1X,Z16.16,1X,ES25.16E3)') &
            'XTRA', trim(tag), model, iter, j, k, transfer(v, 0_8), v
      end subroutine xtra_w

      subroutine xtra_star(s, nvar, nz, tag, iter, soln)
         ! Diagnostic XTRA records (write statements only). tag: RESID0
         ! (initial residual evaluation, iter 0), CORR (after sizeB of
         ! Newton iteration iter), RESID (after the residual evaluation of
         ! the accepted correction).
         type (star_info), pointer :: s
         integer, intent(in) :: nvar, nz, iter
         character(len=*), intent(in) :: tag
         real(dp), pointer :: soln(:,:)
         integer, parameter :: zc(@NC@) = [integer :: @ZC@]
         integer, parameter :: zr(@NR@) = [integer :: @ZR@]
         integer, save :: ncall = 0
         integer :: u, j, k, i, kmax, m
         logical :: nw
         real(dp) :: ssum, smax, v
         u = stepdump_unit(s, nw)
         if (u == 0) return
         m = s% model_number
         if (tag == 'RESID0') then
            ncall = ncall + 1
            call xtra_w(u, 'START_call_index', m, 0, ncall, 0, real(ncall, dp))
            call xtra_w(u, 'START_M_center', m, 0, 0, 0, s% M_center)
            call xtra_w(u, 'START_R_center', m, 0, 0, 0, s% R_center)
            call xtra_w(u, 'START_L_center', m, 0, 0, 0, s% L_center)
            call xtra_w(u, 'START_xmstar', m, 0, 0, 0, s% xmstar)
            call xtra_w(u, 'START_mstar', m, 0, 0, 0, s% mstar)
         end if
         if (tag == 'RESID0' .or. tag == 'RESID') then
            do j = 1, nvar
               ssum = 0d0
               smax = -1d0
               kmax = 0
               do k = 1, nz
                  v = abs(s% equ(j,k))
                  ssum = ssum + v
                  if (v > smax) then
                     smax = v
                     kmax = k
                  end if
               end do
               call xtra_w(u, trim(tag)//'_EQUSUM', m, iter, j, 0, ssum)
               call xtra_w(u, trim(tag)//'_EQUMAX', m, iter, j, kmax, smax)
            end do
         end if
         if (tag == 'RESID') then
            do i = 1, size(zr)
               k = zr(i)
               if (k < 1 .or. k > nz) cycle
               do j = 1, nvar
                  call xtra_w(u, 'RESID_EQU', m, iter, j, k, s% equ(j,k))
               end do
@FIELDS@
            end do
         end if
         if (tag == 'CORR') then
            do j = 1, nvar
               ssum = 0d0
               do k = 1, nz
                  ssum = ssum + abs(soln(j,k))
               end do
               call xtra_w(u, 'CORR_SOLNSUM', m, iter, j, 0, ssum)
            end do
            do i = 1, size(zc)
               k = zc(i)
               if (k < 1 .or. k > nz) cycle
               do j = 1, nvar
                  call xtra_w(u, 'CORR_SOLN', m, iter, j, k, soln(j,k))
                  call xtra_w(u, 'CORR_DX', m, iter, j, k, s% solver_dx(j,k))
               end do
            end do
         end if
      end subroutine xtra_star
"""

_STAR_END = "      end module star_solver\n"


def _fortran_ints(zones: Sequence[int]) -> str:
    """Returns ``zones`` as the items of a Fortran array constructor."""
    return ", ".join(str(int(z)) for z in zones)


def patch_star(
    text: str,
    zones: Sequence[int] = (),
    corr_zones: Sequence[int] | None = None,
) -> str:
    """Returns ``star_solver.f90`` with the solver diagnostics inserted.

    Args:
        text: Upstream ``star/private/star_solver.f90``, not yet patched by
            ``stepdump_patch``.
        zones: Zones whose per-zone residual and state values are written at
            each residual evaluation.
        corr_zones: Zones whose ``soln`` and ``dx`` are written at each
            correction; defaults to ``zones``.

    Returns:
        The patched source.

    Raises:
        ValueError: If an anchor does not occur exactly once, or the file
            already carries diagnostics.
    """
    if "xtra_star" in text:
        raise ValueError("star_solver.f90 already carries XTRA records")
    corr = zones if corr_zones is None else corr_zones
    for anchor, insert, where in STAR_EDITS:
        count = text.count(anchor)
        if count != 1:
            raise ValueError(f"anchor found {count} times:\n{anchor}")
        text = text.replace(
            anchor, anchor + insert if where == "after" else insert + anchor
        )
    fields = "\n".join(
        f"               call xtra_w(u, 'RESID_{f}', m, iter, 0, k, s% {f}(k))"
        for f in ZONE_FIELDS
    )
    helper = (
        _STAR_HELPER.replace("@NC@", str(len(corr)))
        .replace("@ZC@", _fortran_ints(corr))
        .replace("@NR@", str(len(zones)))
        .replace("@ZR@", _fortran_ints(zones))
        .replace("@FIELDS@", fields)
    )
    if text.count(_STAR_END) != 1:
        raise ValueError("end module anchor not unique")
    return text.replace(_STAR_END, helper + _STAR_END)


def _bw(tag: str, name: str, expr: str, indent: int = 12) -> str:
    """Returns one ``RLO`` record write."""
    return (
        " " * indent + f"call xtra_bw(xu, '{tag}', b% model_number, "
        f"b% num_tries, '{name}', {expr})\n"
    )


def _calls(tag: str, items: Sequence[tuple[str, str]], indent: int) -> str:
    return "".join(_bw(tag, n, e, indent) for n, e in items)


_RLO_CALL = (
    "         xu = xtra_bin_unit(b% model_number)\n"
    "         if (xu /= 0) then\n"
    + _calls(
        "RLO_CALL",
        [
            ("curr_mdot", "b% mtransfer_rate"),
            ("explicit_mdot", "explicit_mdot"),
            ("f", "function_to_solve"),
            ("change_factor", "b% change_factor"),
            ("tolerance", "b% implicit_scheme_tolerance"),
            ("starting_mdot", "b% starting_mdot"),
        ],
        12,
    )
    + "            if (abs(function_to_solve) <= "
    "b% implicit_scheme_tolerance) &\n"
    "               write(xu,'(A,2(1X,I0))') 'XTRA RLO_RESULT_in_tol', "
    "b% model_number, b% num_tries\n"
    "         end if\n"
)

_RLO_BRACKET = (
    "         if (xu /= 0) then\n"
    + _calls(
        "RLO_BRACKET",
        [
            ("mdot_lo", "b% mdot_lo"),
            ("mdot_hi", "b% mdot_hi"),
            ("f_lo", "b% implicit_function_lo"),
            ("f_hi", "b% implicit_function_hi"),
        ],
        12,
    )
    + "            write(xu,'(A,2(1X,I0),A,2(1X,L1),1X,A,1X,I0,A,L1)') "
    "'XTRA RLO_FLAGS', b% model_number, b% num_tries, &\n"
    "               ' have_lo,have_hi:', b% have_mdot_lo, b% have_mdot_hi, "
    "'solver_type=' // trim(b% solver_type), &\n"
    "               mod(b% num_tries,2), ' use_sum=', use_sum\n"
    "         end if\n"
)

_RLO_NEXT = (
    "         if (xu /= 0) then\n"
    + _calls(
        "RLO_NEXT",
        [
            ("next_mdot_raw", "new_mdot"),
            ("mdot_lo_after", "b% mdot_lo"),
            ("mdot_hi_after", "b% mdot_hi"),
            ("f_lo_after", "b% implicit_function_lo"),
            ("f_hi_after", "b% implicit_function_hi"),
            ("ierr", "real(ierr,dp)"),
        ],
        12,
    )
    + "            write(xu,'(A,2(1X,I0),A,2(1X,L1))') "
    "'XTRA RLO_FLAGS_after', b% model_number, b% num_tries, &\n"
    "               ' have_lo,have_hi:', b% have_mdot_lo, b% have_mdot_hi\n"
    "         end if\n"
)

_RLO_REDO = (
    "         if (xu /= 0) then\n"
    "            write(xu,'(A,2(1X,I0))') 'XTRA RLO_RESULT_redo', "
    "b% model_number, b% num_tries\n"
    + _bw("RLO_REDO", "new_mdot_final", "new_mdot")
    + "         end if\n"
)

_RLO_DO_CUBIC = (
    "            if (xu /= 0) write(xu,'(A,2(1X,I0),1X,L1)') "
    "'XTRA RLO_DO_CUBIC', b% model_number, b% num_tries, do_cubic\n"
)

_FALLBACK = (
    "               if (ierr /= 0) then\n"
    "                  mdot_next = find0(b% mdot_lo, b% implicit_function_lo, "
    "&\n"
    "                      b% mdot_hi, b% implicit_function_hi)\n"
    "                  ierr = 0\n"
    "               end if\n"
)

_RLO_QUAD = (
    "            if (xu /= 0) then\n"
    + _calls(
        "RLO_QUAD",
        [
            ("mdot_next_find0_quadratic", "mdot_next"),
            ("ierr", "real(ierr,dp)"),
        ],
        15,
    )
    + "            end if\n"
)

_RLO_QUAD_FINAL = (
    "            if (xu /= 0) then\n"
    + _calls(
        "RLO_QUAD_FINAL",
        [
            ("mdot_next_after_fallback", "mdot_next"),
            ("ierr", "real(ierr,dp)"),
        ],
        15,
    )
    + "            end if\n"
)

_NUM_TRIES_RESET = (
    "            b% fixed_delta_mdot = b% mtransfer_rate * "
    "(1 - b% change_factor)\n         end if\n\n"
)

BINARY_EDITS: list[tuple[str, str, str]] = [
    (
        "         integer :: ierr\n         logical :: use_sum, detached\n",
        "         integer :: xu\n",
        "after",
    ),
    (
        "         if (detached) return\n\n",
        _RLO_CALL,
        "after",
    ),
    (_NUM_TRIES_RESET, _RLO_BRACKET, "after"),
    (
        "         new_mdot = pick_mdot_for_implicit_rlo(b, function_to_solve, "
        "b% mtransfer_rate, use_sum, ierr)\n",
        _RLO_NEXT,
        "after",
    ),
    ("         check_implicit_rlo = redo\n", _RLO_REDO, "before"),
    (
        "         real(dp) :: starting_mdot, current_change_factor\n"
        "         logical :: do_cubic\n",
        "         integer :: xu\n",
        "after",
    ),
    (
        "         starting_mdot = -b% starting_mdot*Msun/secyer\n",
        "         xu = xtra_bin_unit(b% model_number)\n",
        "before",
    ),
    (
        "            if (do_cubic) then\n"
        "               mdot_next = find0_quadratic( &\n",
        _RLO_DO_CUBIC,
        "before",
    ),
    (_FALLBACK, _RLO_QUAD, "before"),
    (_FALLBACK, _RLO_QUAD_FINAL, "after"),
]

_BINARY_HELPER = """
      integer function xtra_bin_unit(model)
         ! Output unit for binary XTRA records (file named by
         ! FLEXMESA_XTRA_BIN), or 0 when none is written. The run stops
         ! when a model after FLEXMESA_XTRA_STOPMODEL starts. Output only.
         integer, intent(in) :: model
         integer, save :: unit = 0, firstmodel = -huge(1)
         integer, save :: stopmodel = huge(1)
         logical, save :: init = .false.
         character(len=1024) :: path
         integer :: st
         xtra_bin_unit = 0
         if (.not. init) then
            init = .true.
            call get_environment_variable('FLEXMESA_XTRA_BIN', path, &
               status=st)
            if (st == 0 .and. len_trim(path) > 0) then
               open(newunit=unit, file=trim(path), status='unknown', &
                  position='append', action='write')
            end if
            call get_environment_variable('FLEXMESA_XTRA_FIRSTMODEL', path, &
               status=st)
            if (st == 0 .and. len_trim(path) > 0) read(path, *) firstmodel
            call get_environment_variable('FLEXMESA_XTRA_STOPMODEL', path, &
               status=st)
            if (st == 0 .and. len_trim(path) > 0) read(path, *) stopmodel
         end if
         if (unit == 0) return
         if (model > stopmodel) then
            flush(unit)
            stop 'FLEXMESA_XTRA: requested binary models written'
         end if
         if (model < firstmodel) return
         xtra_bin_unit = unit
      end function xtra_bin_unit

      subroutine xtra_bw(u, tag, model, nt, name, v)
         integer, intent(in) :: u, model, nt
         character(len=*), intent(in) :: tag, name
         real(dp), intent(in) :: v
         write(u,'(A,1X,A,2(1X,I0),1X,A,1X,Z16.16,1X,ES25.16E3)') &
            'XTRA', trim(tag), model, nt, trim(name), transfer(v, 0_8), v
      end subroutine xtra_bw
"""

_BINARY_END = "      end module binary_mdot\n"


def patch_binary_rlo(text: str) -> str:
    """Returns ``binary_mdot.f90`` with the mass-transfer records inserted.

    Args:
        text: Upstream ``binary/private/binary_mdot.f90``.

    Returns:
        The patched source.

    Raises:
        ValueError: If an anchor does not occur exactly once, or the file
            already carries diagnostics.
    """
    if "xtra_bin_unit" in text:
        raise ValueError("binary_mdot.f90 already carries XTRA records")
    for anchor, insert, where in BINARY_EDITS:
        count = text.count(anchor)
        if count != 1:
            raise ValueError(f"anchor found {count} times:\n{anchor}")
        text = text.replace(
            anchor, anchor + insert if where == "after" else insert + anchor
        )
    if text.count(_BINARY_END) != 1:
        raise ValueError("end module anchor not unique")
    return text.replace(_BINARY_END, _BINARY_HELPER + _BINARY_END)


def parse_zones(text: str) -> list[int]:
    """Parses ``"1,262,474:481"`` into a zone list (``a:b`` is inclusive).

    Args:
        text: Comma-separated zones and inclusive ranges.

    Returns:
        The zones in the order given.

    Raises:
        ValueError: For an item that is not an integer or a range, or a
            zone below 1.
    """
    zones: list[int] = []
    for item in filter(None, (t.strip() for t in text.split(","))):
        lo, sep, hi = item.partition(":")
        first, last = int(lo), int(hi) if sep else int(lo)
        if first < 1 or last < first:
            raise ValueError(f"bad zone item {item!r}")
        zones.extend(range(first, last + 1))
    return zones


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--zones", default="")
    parser.add_argument("--corr-zones", default=None)
    args = parser.parse_args()
    text = args.path.read_text(encoding="utf-8")
    if args.kind == "star":
        corr = None if args.corr_zones is None else parse_zones(args.corr_zones)
        text = patch_star(text, parse_zones(args.zones), corr)
    else:
        text = patch_binary_rlo(text)
    args.path.write_text(text, encoding="utf-8")
    print(f"patched {args.path}")


if __name__ == "__main__":
    main()
