#!/usr/bin/env python3
"""Insert the read-only ADIPLS glue taps into an overlay copy of MESA's astero.

Usage: apply_astero_taps.py <overlay astero dir>

Edits <dir>/private/adipls_support.f90 and adipls_support_procs.f90 of an
overlay copy (never the pristine tree) and copies astero_glue_tap.inc next to
them. Every anchor must match exactly once, so a changed upstream file fails
loudly instead of being tapped in the wrong place. See astero_glue_tap.inc for
what the taps record (#171, package A0).
"""

from __future__ import annotations

import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent

DECLS = """
      ! --- FlexMESA A0 glue taps (state) ---
      integer, save :: a0tap_state = 0, a0tap_store_id = 0, a0tap_call_id = 0
      integer, save :: a0tap_last_nn = -1, a0tap_calls_unit = -1, a0tap_modes_unit = -1
      logical, save :: a0tap_stored_this_call = .false.
      logical, save :: a0tap_calls_open = .false., a0tap_modes_open = .false.
      character(len=512), save :: a0tap_dir = '', a0tap_name = ''
      character(len=64), save :: a0tap_last_file = 'none'
      real(dp), allocatable, save :: a0tap_last_x(:), a0tap_last_aa(:,:)
      real(dp), save :: a0tap_last_data(8) = 0
"""

# (file, anchor, text inserted, where: 'after' | 'before')
EDITS: list[tuple[str, str, str, str]] = [
    (
        "adipls_support.f90",
        "      logical, parameter :: ADIPLS_IS_ENABLED = .true.\n",
        DECLS,
        "after",
    ),
    (
        "adipls_support.f90",
        "      contains\n\n"
        "      ! this can be called from user run_star_extras check model routine\n",
        "      include 'astero_glue_tap.inc'\n\n",
        "after_contains",
    ),
    (
        "adipls_support.f90",
        "         ! If necessary, write it\n",
        "         call a0tap_store_fgong(global_data, point_data)  ! FlexMESA A0 tap\n\n",
        "before",
    ),
    (
        "adipls_support.f90",
        "         deallocate(global_data, point_data)\n\n         if (ierr /= 0) then\n"
        "            write(*,*) 'failed in fgong_amdl'\n",
        "         if (ierr == 0) call a0tap_store_amdl6(nn, data, aa)  ! FlexMESA A0 tap\n",
        "before",
    ),
    (
        "adipls_support.f90",
        "            write(*,*) 'failed in store_amdl'\n"
        "            call mesa_error(__FILE__,__LINE__)\n"
        "         end if\n",
        "         call a0tap_store_xaa('amdl_store')  ! FlexMESA A0 tap\n",
        "after",
    ),
    (
        "adipls_support.f90",
        "            nn = nn_new\n",
        "            call a0tap_store_xaa('amdl_redist')  ! FlexMESA A0 tap\n",
        "after",
    ),
    (
        "adipls_support.f90",
        "         call adipls(i_paramset, ierr_param, i_inout, &\n"
        "               x, aa, data, nn, ivarmd, iaa_arg)\n",
        "         call a0tap_call(s)  ! FlexMESA A0 tap\n",
        "before",
    ),
    (
        "adipls_support_procs.f90",
        "         use adipls_support, only: adipls_mode_info\n",
        "         use adipls_support, only: adipls_mode_info, a0tap_mode  ! FlexMESA A0 tap\n",
        "replace",
    ),
    (
        "adipls_support_procs.f90",
        "         n = num_results\n         call adipls_mode_info( &\n",
        "         call a0tap_mode(obs_st(:,nobs_st), csummm, new_el, new_order, new_em, &\n"
        "            new_inertia, new_cyclic_freq, nobs_st)  ! FlexMESA A0 tap\n",
        "before",
    ),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    astero = pathlib.Path(sys.argv[1])
    priv = astero / "private"
    if not (astero / ".flexmesa_adipls_overlay").exists():
        print(f"refusing: {astero} is not marked as an overlay copy", file=sys.stderr)
        return 2
    texts = {f: (priv / f).read_text() for f in {e[0] for e in EDITS}}
    for fname, anchor, text, where in EDITS:
        s = texts[fname]
        n = s.count(anchor)
        if n != 1:
            print(f"anchor matched {n} times in {fname}: {anchor!r}", file=sys.stderr)
            return 1
        if where == "after":
            s = s.replace(anchor, anchor + text)
        elif where == "after_contains":
            s = s.replace(anchor, "      contains\n\n" + text + anchor[len("      contains\n\n") :])
        elif where == "before":
            s = s.replace(anchor, text + anchor)
        elif where == "replace":
            s = s.replace(anchor, text)
        texts[fname] = s
    for fname, s in texts.items():
        (priv / fname).write_text(s)
    shutil.copy(HERE / "astero_glue_tap.inc", priv / "astero_glue_tap.inc")
    print(f"tapped {', '.join(sorted(texts))} in {priv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
