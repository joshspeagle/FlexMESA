# SPDX-License-Identifier: GPL-3.0-or-later
"""Inserts initial-model dump ``write`` statements into MESA ``init.f90``.

The patch adds output only. When the environment variable
``FLEXMESA_RELAX154_DUMP`` names a file, ``model_builder`` appends one block
per dump point to it:

* ``BUILD``: immediately after ``build_initial_model`` or
  ``build_pre_ms_model`` returned (``init.f90``, before the ``extra_heat``
  loop), before ``finish_load_model`` runs. Only the solver unknowns in
  ``xh`` (lnd, lnT, lnR, L), ``dq`` and ``xa`` are written.
* ``LOADED``: after ``finish_load_model`` (so ``q``, ``dq``, ``m`` and the
  ``lnd``, ``lnT``, ``lnR``, ``L`` arrays are set), before the pre-main
  sequence mass relaxation and ``do_relax_num_steps`` calls.
* ``FINAL``: the end of ``model_builder``, after the pre-main-sequence mass
  relaxation, ``do_relax_num_steps`` and, for ``create_pre_main_sequence_model``
  with the default ``pre_ms_relax_to_start_radiative_core = .true.``,
  ``do_relax_to_radiative_core``. For the 15 Msun setup this changes ``nz``
  (1033 at ``BUILD`` and ``LOADED``, 680 at ``FINAL``); for the two
  ``create_initial_model`` setups all three stages agree.

With ``FLEXMESA_RELAX154_STOP`` set, the process executes ``stop 0`` right
after the ``FINAL`` block, so no evolve step runs. This is the only change to
control flow and it is taken only when that variable is set.

Records (one per line, ``dec hex`` pairs are ``ES25.16E3`` and ``Z16.16``):

* ``INT tag name value``
* ``REAL tag name dec hex``
* ``STR tag name value``
* ``SPECIES tag j chem_id name``
* ``ZONE tag k (dec hex)*``: BUILD has lnd lnT lnR L dq; LOADED and FINAL
  have lnd lnT lnR L dq q m followed by the ``s% lnd lnT lnR L`` array values
  (same order) so the two sources can be compared.
* ``XA tag k hex*``: ``xa(1:species, k)`` as bit patterns.

Usage::

    python relax154_patch.py <overlay>/star/private/init.f90
"""

from __future__ import annotations

import pathlib
import sys

MARKER = "FLEXMESA_RELAX154_DUMP"

HELPER = """\
         subroutine relax154_dump(tag, full)
            use chem_def, only: chem_isos
            character(len=*), intent(in) :: tag
            logical, intent(in) :: full
            character(len=1024) :: fname
            integer :: u, k, j, stat, nzz
            character(len=*), parameter :: rf = &
               '(A,1X,A,1X,I0,*(1X,ES25.16E3,1X,Z16.16))'
            call get_environment_variable('FLEXMESA_RELAX154_DUMP', fname, &
               status=stat)
            if (stat /= 0) return
            open(newunit=u, file=trim(fname), status='unknown', &
               position='append', action='write')
            nzz = s% nz
            write(u,'(A,1X,A,1X,A,1X,I0)') 'INT', tag, 'nz', nzz
            write(u,'(A,1X,A,1X,A,1X,I0)') 'INT', tag, 'species', s% species
            write(u,'(A,1X,A,1X,A,1X,A)') 'STR', tag, 'net_name', &
               trim(s% net_name)
            do j = 1, s% species
               write(u,'(A,1X,A,1X,I0,1X,I0,1X,A)') 'SPECIES', tag, j, &
                  s% chem_id(j), trim(chem_isos% name(s% chem_id(j)))
            end do
            write(u,'(A,1X,A,1X,A,2(1X,ES25.16E3,1X,Z16.16))') 'REAL', tag, &
               'mstar', s% mstar, transfer(s% mstar, 0_8)
            write(u,'(A,1X,A,1X,A,2(1X,ES25.16E3,1X,Z16.16))') 'REAL', tag, &
               'xmstar', s% xmstar, transfer(s% xmstar, 0_8)
            write(u,'(A,1X,A,1X,A,2(1X,ES25.16E3,1X,Z16.16))') 'REAL', tag, &
               'M_center', s% M_center, transfer(s% M_center, 0_8)
            write(u,'(A,1X,A,1X,A,2(1X,ES25.16E3,1X,Z16.16))') 'REAL', tag, &
               'R_center', s% R_center, transfer(s% R_center, 0_8)
            write(u,'(A,1X,A,1X,A,2(1X,ES25.16E3,1X,Z16.16))') 'REAL', tag, &
               'L_center', s% L_center, transfer(s% L_center, 0_8)
            write(u,'(A,1X,A,1X,A,2(1X,ES25.16E3,1X,Z16.16))') 'REAL', tag, &
               'initial_mass', s% initial_mass, transfer(s% initial_mass, 0_8)
            write(u,'(A,1X,A,1X,A,2(1X,ES25.16E3,1X,Z16.16))') 'REAL', tag, &
               'dt_next', s% dt_next, transfer(s% dt_next, 0_8)
            do k = 1, nzz
               if (full) then
                  write(u,rf) 'ZONE', tag, k, &
                     s% xh(s% i_lnd,k), transfer(s% xh(s% i_lnd,k), 0_8), &
                     s% xh(s% i_lnT,k), transfer(s% xh(s% i_lnT,k), 0_8), &
                     s% xh(s% i_lnR,k), transfer(s% xh(s% i_lnR,k), 0_8), &
                     s% xh(s% i_lum,k), transfer(s% xh(s% i_lum,k), 0_8), &
                     s% dq(k), transfer(s% dq(k), 0_8), &
                     s% q(k), transfer(s% q(k), 0_8), &
                     s% m(k), transfer(s% m(k), 0_8), &
                     s% lnd(k), transfer(s% lnd(k), 0_8), &
                     s% lnT(k), transfer(s% lnT(k), 0_8), &
                     s% lnR(k), transfer(s% lnR(k), 0_8), &
                     s% L(k), transfer(s% L(k), 0_8)
               else
                  write(u,rf) 'ZONE', tag, k, &
                     s% xh(s% i_lnd,k), transfer(s% xh(s% i_lnd,k), 0_8), &
                     s% xh(s% i_lnT,k), transfer(s% xh(s% i_lnT,k), 0_8), &
                     s% xh(s% i_lnR,k), transfer(s% xh(s% i_lnR,k), 0_8), &
                     s% xh(s% i_lum,k), transfer(s% xh(s% i_lum,k), 0_8), &
                     s% dq(k), transfer(s% dq(k), 0_8)
               end if
               write(u,'(A,1X,A,1X,I0,999(1X,Z16.16))') 'XA', tag, k, &
                  (transfer(s% xa(j,k), 0_8), j = 1, s% species)
            end do
            close(u)
         end subroutine relax154_dump

"""

# (anchor, text, where): the anchor must occur exactly once in the file.
EDITS: list[tuple[str, str, str]] = [
    (
        "         do k=1,s% nz\n            s% extra_heat(k) = 0\n"
        "         end do\n",
        "         call relax154_dump('BUILD', .false.)\n",
        "before",
    ),
    (
        "         if (s% max_years_for_timestep > 0) &\n"
        "            s% dt_next = min(s% dt_next, "
        "secyer*s% max_years_for_timestep)\n"
        "         call set_phase_of_evolution(s)\n",
        "         call relax154_dump('LOADED', .true.)\n",
        "before",
    ),
    (
        "         s% doing_first_model_of_run = .true.\n"
        "         s% num_trace_history_values = num_trace_history_values\n",
        "         call relax154_dump('FINAL', .true.)\n"
        "         block\n"
        "            character(len=8) :: stop_flag\n"
        "            integer :: stop_stat\n"
        "            call get_environment_variable("
        "'FLEXMESA_RELAX154_STOP', &\n"
        "               stop_flag, status=stop_stat)\n"
        "            if (stop_stat == 0) stop 0\n"
        "         end block\n",
        "before",
    ),
    (
        "         contains\n\n"
        "         subroutine setup_for_relax_after_create_pre_ms_model\n",
        HELPER,
        "contains",
    ),
]


def patch_text(text: str) -> str:
    """Returns ``init.f90`` text with the dump statements inserted.

    Args:
        text: Original ``star/private/init.f90``.

    Returns:
        The patched text.

    Raises:
        ValueError: If an anchor does not occur exactly once.
    """
    for anchor, new, where in EDITS:
        if text.count(anchor) != 1:
            raise ValueError(f"anchor found {text.count(anchor)}x: {anchor!r}")
        if where == "before":
            text = text.replace(anchor, new + anchor)
        else:
            head, tail = anchor.split("\n\n", 1)
            text = text.replace(anchor, head + "\n\n" + new + tail)
    return text


def main() -> None:
    """Patches the file named on the command line in place."""
    path = pathlib.Path(sys.argv[1])
    path.write_text(patch_text(path.read_text(encoding="utf-8")), "utf-8")


if __name__ == "__main__":
    main()
