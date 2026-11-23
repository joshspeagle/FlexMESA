# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inserts step-dump ``write`` statements into MESA 26.4.1 ``star_solver.f90``.

The patch adds output only: it never changes an expression, a branch taken,
or the order of any arithmetic. A record is written only when the
environment variable ``FLEXMESA_STEPDUMP`` names an output file, and only for
the first ``FLEXMESA_STEPDUMP_STEPS`` (default 3) distinct model numbers of
each star, starting at model number ``FLEXMESA_STEPDUMP_FIRST`` when that is
set (a window later in the run). With ``FLEXMESA_STEPDUMP_STOP`` set, the run
stops when the next model starts, after a ``DONE`` record; nothing past that
point is compared.
Each process appends a ``RUN`` record when it opens the file, since every
``./rn`` stage is a separate process. Every floating-point value is written
as its IEEE-754 binary64 bit pattern in hexadecimal (``Z16.16``).

Records, one per line, each starting with the record type and the star id:

* ``NAMES id nvar nvar_hydro species nz name...``: once per star; the names
  of the ``nvar`` solver variables (``s% nameofvar``).
* ``EQNAMES id nvar name...``: once per star; ``s% nameofequ``.
* ``CALL id model dt residual_norm max_residual max_resid_k max_resid_j``:
  each solver call, after the initial residual evaluation
  (``star_solver.f90:274``).
* ``STATE id model k xh(1:nvar_hydro) xa(1:species)``: per zone, at the first
  solver call of each model.
* ``DM id model k dm``: per zone, the cell mass (g) at that call.
* ``EQU id model k equ(1:nvar)``: per zone, the residuals at that call.
* ``CORR id model iter correction_norm max_correction k j``: each Newton
  iteration, after ``sizeB`` (``star_solver.f90:359``).
* ``NEWTON id model iter min_corr_coeff correction_factor slope``: before
  ``adjust_correction``.
* ``FOLD id model iter fold``: start of ``adjust_correction``.
* ``TRY id model solver_iter iter alam coeff f f_target``: each trial
  coefficient of the line search.
* ``RESID id model iter coeff residual_norm max_residual k j``: after the
  residual evaluation of each accepted correction
  (``star_solver.f90:461``).

Usage::

    python stepdump_patch.py <overlay>/star/private/star_solver.f90
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "FLEXMESA_STEPDUMP"
HEX = "Z16.16"


def _h(name: str) -> str:
    """Returns the Fortran expression giving ``name``'s IEEE-754 bits."""
    return f"transfer({name}, 0_8)"


def _w(fmt_tail: str, tag: str, args: str, indent: int) -> str:
    """Returns a guarded ``write`` statement for one record.

    Args:
        fmt_tail: Fortran edit descriptors after the tag, id and model.
        tag: Record type.
        args: Comma-separated output items after ``s% id, s% model_number``.
        indent: Leading spaces.
    """
    pad = " " * indent
    return (
        f"{pad}sd_u = stepdump_unit(s, sd_new)\n"
        f"{pad}if (sd_u /= 0) write(sd_u,'(A,1X,I0,1X,I0{fmt_tail})') "
        f"'{tag}', s% id, s% model_number, {args}\n"
    )


# Each edit is (anchor, text, where): the anchor must occur exactly once in
# the file; ``where`` is "after" or "before" the anchor.
EDITS: list[tuple[str, str, str]] = [
    (
        "         integer :: nz, iter, max_tries, tiny_corr_cnt, i, &\n",
        "            sd_k, sd_j, &\n",
        "after",
    ),
    (
        # Solver entry, after the initial residuals (star_solver.f90:274-282).
        "            first_try = .true.\n            iter = 1\n",
        "            sd_u = stepdump_unit(s, sd_new)\n"
        "            if (sd_u /= 0) then\n"
        f"               write(sd_u,'(A,1X,I0,1X,I0,"
        f"3(1X,{HEX}),2(1X,I0))') 'CALL', s% id, s% model_number, "
        f"{_h('s% dt')}, {_h('residual_norm')}, {_h('max_residual')}, "
        "max_resid_k, max_resid_j\n"
        "               if (sd_new) then\n"
        "                  do sd_k = 1, nz\n"
        f"                     write(sd_u,'(A,1X,I0,1X,I0,"
        f"1X,I0,999(1X,{HEX}))') 'STATE', s% id, s% model_number, sd_k, "
        f"({_h('s% xh(sd_j,sd_k)')}, sd_j=1,s% nvar_hydro), "
        f"({_h('s% xa(sd_j,sd_k)')}, sd_j=1,s% species)\n"
        f"                     write(sd_u,'(A,1X,I0,1X,I0,"
        f"1X,I0,1X,{HEX})') 'DM', s% id, s% model_number, sd_k, "
        f"{_h('s% dm(sd_k)')}\n"
        f"                     write(sd_u,'(A,1X,I0,1X,I0,"
        f"1X,I0,999(1X,{HEX}))') 'EQU', s% id, s% model_number, sd_k, "
        f"({_h('s% equ(sd_j,sd_k)')}, sd_j=1,nvar)\n"
        "                  end do\n"
        "               end if\n"
        "            end if\n",
        "before",
    ),
    (
        # After the size of the scaled correction (star_solver.f90:359-364).
        "               correction_norm = abs(correction_norm)\n"
        "               max_abs_correction = abs(max_correction)\n",
        _w(
            f",1X,I0,2(1X,{HEX}),2(1X,I0)",
            "CORR",
            f"iter, {_h('correction_norm')}, {_h('max_correction')}, "
            "max_corr_k, max_corr_j",
            15,
        ),
        "before",
    ),
    (
        "               f = 0d0\n               call adjust_correction( &\n",
        _w(
            f",1X,I0,3(1X,{HEX})",
            "NEWTON",
            f"iter, {_h('min_corr_coeff')}, {_h('correction_factor')}, "
            f"{_h('slope')}",
            15,
        ),
        "before",
    ),
    (
        "            fold = f\n",
        _w(f",1X,I0,1X,{HEX}", "FOLD", f"s% solver_iter, {_h('f')}", 12),
        "after",
    ),
    (
        "               f_target = max(fold/2, fold + alf*coeff*slope)\n",
        _w(
            f",1X,I0,1X,I0,4(1X,{HEX})",
            "TRY",
            f"s% solver_iter, iter, {_h('alam')}, {_h('coeff')}, {_h('f')}, "
            f"{_h('f_target')}",
            15,
        ),
        "after",
    ),
    (
        # After the residuals of the accepted correction (star_solver.f90:461).
        "               call sizequ(s, nvar, residual_norm, max_residual, "
        "max_resid_k, max_resid_j, ierr)\n"
        "               if (ierr /= 0) then\n"
        "                  call oops('sizequ returned ierr')\n"
        "                  exit iter_loop\n"
        "               end if\n",
        _w(
            f",1X,I0,3(1X,{HEX}),2(1X,I0)",
            "RESID",
            f"iter, {_h('coeff')}, {_h('residual_norm')}, "
            f"{_h('max_residual')}, max_resid_k, max_resid_j",
            15,
        ),
        "after",
    ),
]

# Module procedure appended before ``end module star_solver``. It opens the
# output file on first use and decides, per star and model number, whether a
# record is written; it touches no solver state.
HELPER = f"""
      integer function stepdump_unit(s, new_model) result(u)
         ! Output unit for step-dump records, or 0 when none is written
         ! (newunit= returns negative units, never 0).
         ! new_model is true on the first call for a model of this star.
         ! Added by the {MARKER} overlay: no arithmetic of the solver.
         type (star_info), pointer :: s
         logical, intent(out) :: new_model
         integer, parameter :: max_ids = 16
         integer, save :: unit = 0, nmax = 3, first_model = -huge(1)
         logical, save :: initialised = .false., stop_when_done = .false.
         integer, save :: nseen(max_ids) = 0, last_model(max_ids) = -1
         logical, save :: named(max_ids) = .false.
         character(len=1024) :: path
         integer :: st, i
         new_model = .false.
         u = 0
         if (.not. initialised) then
            initialised = .true.
            call get_environment_variable('{MARKER}', path, status=st)
            if (st == 0 .and. len_trim(path) > 0) then
               ! Append: each ./rn stage is a separate process.
               open(newunit=unit, file=trim(path), status='unknown', &
                  position='append', action='write')
               write(unit, '(A)') 'RUN'
               call get_environment_variable('{MARKER}_STEPS', path, &
                  status=st)
               if (st == 0 .and. len_trim(path) > 0) read(path, *) nmax
               call get_environment_variable('{MARKER}_FIRST', path, &
                  status=st)
               if (st == 0 .and. len_trim(path) > 0) read(path, *) first_model
               call get_environment_variable('{MARKER}_STOP', path, &
                  status=st)
               stop_when_done = st == 0 .and. len_trim(path) > 0
            end if
         end if
         if (unit == 0 .or. s% id < 1 .or. s% id > max_ids) return
         ! Optional window: models before FLEXMESA_STEPDUMP_FIRST are skipped.
         if (s% model_number < first_model) return
         if (s% model_number /= last_model(s% id)) then
            if (nseen(s% id) >= nmax) then
               ! Optional early end once the requested models are dumped;
               ! nothing after this point is compared.
               if (stop_when_done) then
                  write(unit, '(A)') 'DONE'
                  flush(unit)
                  stop 'FLEXMESA_STEPDUMP: requested models dumped'
               end if
               return
            end if
            nseen(s% id) = nseen(s% id) + 1
            last_model(s% id) = s% model_number
            new_model = .true.
         end if
         if (.not. named(s% id)) then
            named(s% id) = .true.
            write(unit, '(A,1X,I0,4(1X,I0),999(1X,A))') 'NAMES', s% id, &
               s% nvar_total, s% nvar_hydro, s% species, s% nz, &
               (trim(s% nameofvar(i)), i=1,s% nvar_total)
            write(unit, '(A,1X,I0,1X,I0,999(1X,A))') 'EQNAMES', s% id, &
               s% nvar_total, (trim(s% nameofequ(i)), i=1,s% nvar_total)
         end if
         u = unit
      end function stepdump_unit

"""

END_ANCHOR = "      end module star_solver\n"


def patch_text(text: str) -> str:
    """Returns ``text`` with the step-dump records inserted.

    Args:
        text: Contents of upstream ``star/private/star_solver.f90``.

    Returns:
        The patched source.

    Raises:
        ValueError: If an anchor does not occur exactly once, or the file is
            already patched.
    """
    if MARKER in text:
        raise ValueError("star_solver.f90 is already patched")
    for anchor, insert, where in EDITS:
        count = text.count(anchor)
        if count != 1:
            raise ValueError(f"anchor found {count} times:\n{anchor}")
        new = anchor + insert if where == "after" else insert + anchor
        text = text.replace(anchor, new)
    # Every routine that writes a record declares the flag it passes.
    text = _declare_flags(text)
    if text.count(END_ANCHOR) != 1:
        raise ValueError("end module anchor not unique")
    return text.replace(END_ANCHOR, HELPER + END_ANCHOR)


def _declare_flags(text: str) -> str:
    """Declares ``sd_new`` and ``sd_u`` in ``do_solver``, the host of every
    record site."""
    anchor = (
        "         integer :: nz, iter, max_tries, tiny_corr_cnt, i, &\n"
        "            sd_k, sd_j, &\n"
    )
    if text.count(anchor) != 1:
        raise ValueError("do_solver declaration anchor not unique")
    return text.replace(
        anchor,
        "         logical :: sd_new\n         integer :: sd_u\n" + anchor,
    )


def main() -> None:
    """CLI entry point."""
    path = pathlib.Path(sys.argv[1])
    path.write_text(
        patch_text(path.read_text(encoding="utf-8")), encoding="utf-8"
    )
    print(f"patched {path}")


if __name__ == "__main__":
    main()
