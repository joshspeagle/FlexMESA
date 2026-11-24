#!/usr/bin/env python3
"""Patch an overlay copy of GYRE 8.1 for oracle taps or line coverage.

The overlay is a copy of `$MESA_DIR/gyre/gyre` made by `build_overlay.sh`.
The pristine MESA tree is never touched: this script refuses to run on a
directory that does not contain the marker file written by
`build_overlay.sh`.

Modes:
  --taps      insert read-only `gyre_tap_m` calls (see `gyre_tap_m.fypp`)
              and add the tap module to `libgyre_SRCS`; compiler flags stay
              exactly as in the MESA build (`-O2 -ffp-contract=off -fopenmp`).
  --coverage  only change `-O2` to `-O0 --coverage` in `build/Make.inc`.

Every edit is an exact-substring replacement that must match exactly the
stated number of times, so a changed upstream file fails loudly.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

MARKER = ".flexmesa_gyre_overlay"

USE_TAP = "   use gyre_tap_m\n\n"
MODULE_IMPLICIT = "   implicit none (type, external)\n"


def _replace(text: str, old: str, new: str, count: int, where: str) -> str:
    found = text.count(old)
    if found != count:
        raise SystemExit(f"{where}: expected {count} match(es) for anchor, found {found}:\n{old!r}")
    return text.replace(old, new)


def _add_use(text: str, where: str) -> str:
    # Insert `use gyre_tap_m` before the module-level `implicit none`
    # (the first occurrence; later ones belong to interface blocks).
    idx = text.find(MODULE_IMPLICIT)
    if idx < 0:
        raise SystemExit(f"{where}: module-level implicit none not found")
    return text[:idx] + USE_TAP + text[idx:]


def _first_only(text: str, old: str, new: str, where: str) -> str:
    idx = text.find(old)
    if idx < 0:
        raise SystemExit(f"{where}: anchor not found:\n{old!r}")
    return text[:idx] + new + text[idx + len(old) :]


EDITS: dict[str, list[tuple[str, str, int]]] = {
    "lib/gyre_mesa_m.fypp": [
        (
            "         if (md_p_m(i)%l /= l) cycle md_p_loop\n",
            "         if (md_p_m(i)%l /= l) cycle md_p_loop\n\n"
            "         call tap_begin_modepar(md_p_m(i)%l, i)\n",
            1,
        ),
        (
            "         call refine_grid(gr, cx, omega_re, gr_p_sel, os_p_sel)\n",
            "         call refine_grid(gr, cx, omega_re, gr_p_sel, os_p_sel)\n\n"
            "         call tap_val('GRID', 'x', gr%pt%x)\n"
            "         call tap_val('GRID', 's', gr%pt%s)\n"
            "         call tap_val('SCAN', 'omega_re', omega_re)\n",
            1,
        ),
        (
            "         if (md%n_pg < md_p_m(i)%n_pg_min .OR. md%n_pg > md_p_m(i)%n_pg_max) return\n",
            "         if (md%n_pg < md_p_m(i)%n_pg_min .OR. "
            "md%n_pg > md_p_m(i)%n_pg_max) return\n\n"
            "         call tap_val('MODE_AD', 'omega', md%omega)\n"
            "         call tap_val('MODE_AD', 'l,n_p,n_g,n_pg,n', "
            "[md%l, md%n_p, md%n_g, md%n_pg, md%n])\n",
            1,
        ),
        (
            "         retcode = 0\n\n         call user_sub(md, ipar, rpar, retcode)\n",
            "         retcode = 0\n\n"
            "         call tap_val('MODE_NAD', 'omega', md%omega)\n"
            "         call tap_val('MODE_NAD', 'l,n_p,n_g,n_pg,n', "
            "[md%l, md%n_p, md%n_g, md%n_pg, md%n])\n\n"
            "         call user_sub(md, ipar, rpar, retcode)\n",
            1,
        ),
    ],
    "search/discrim_m.fypp": [
        (
            "         discrim = bp%det()\n",
            "         discrim = bp%det()\n\n"
            "         call tap_val('DISCRIM', 'omega', st%omega)\n"
            "         call tap_val('DISCRIM', 'det', discrim)\n",
            1,
        ),
    ],
    "bvp/bvp_m.fypp": [
        (
            "         call this%sm%set_B_o(B_o, scl_o)\n",
            "         call this%sm%set_B_o(B_o, scl_o)\n\n"
            "         call tap_begin_build(this%n)\n"
            "         if (tap_build_dumped()) then\n"
            "            call tap_val('BUILD', 'omega', st%omega)\n"
            "            call tap_val('BOUND', 'B_i', B_i)\n"
            "            call tap_val('BOUND', 'scl_i', scl_i)\n"
            "            call tap_val('BOUND', 'B_o', B_o)\n"
            "            call tap_val('BOUND', 'scl_o', scl_o)\n"
            "         end if\n",
            1,
        ),
        (
            "            call this%sm%set_E(j, E_l, E_r, scl)\n",
            "            call this%sm%set_E(j, E_l, E_r, scl)\n"
            "            if (tap_want_block(j, this%n)) "
            "call tap_block(j, E_l, E_r, scl)\n",
            1,
        ),
    ],
    "matrix/linalg_m.fypp": [
        (
            "         call XGEEV('V', 'V', n, A, n, lambda_re, lambda_im, &\n"
            "            V_l_r, n, V_r_r, n, work, SIZE(work), info)\n",
            "         if (tap_lapack_on('DGEEV')) "
            "call tap_val('LAPACK_IN', 'DGEEV:A', A)\n"
            "         call XGEEV('V', 'V', n, A, n, lambda_re, lambda_im, &\n"
            "            V_l_r, n, V_r_r, n, work, SIZE(work), info)\n"
            "         if (tap_lapack_last()) then\n"
            "            call tap_val('LAPACK_OUT', 'DGEEV:info', info)\n"
            "            call tap_val('LAPACK_OUT', 'DGEEV:wr', lambda_re)\n"
            "            call tap_val('LAPACK_OUT', 'DGEEV:wi', lambda_im)\n"
            "            call tap_val('LAPACK_OUT', 'DGEEV:VL', V_l_r)\n"
            "            call tap_val('LAPACK_OUT', 'DGEEV:VR', V_r_r)\n"
            "         end if\n",
            1,
        ),
        (
            "         call XGEEV('V', 'V', n, A, n, lambda, &\n"
            "            V_l, n, V_r, n, work, SIZE(work, 1), rwork, info)\n",
            "         if (tap_lapack_on('ZGEEV')) "
            "call tap_val('LAPACK_IN', 'ZGEEV:A', A)\n"
            "         call XGEEV('V', 'V', n, A, n, lambda, &\n"
            "            V_l, n, V_r, n, work, SIZE(work, 1), rwork, info)\n"
            "         if (tap_lapack_last()) then\n"
            "            call tap_val('LAPACK_OUT', 'ZGEEV:info', info)\n"
            "            call tap_val('LAPACK_OUT', 'ZGEEV:w', lambda)\n"
            "            call tap_val('LAPACK_OUT', 'ZGEEV:VL', V_l)\n"
            "            call tap_val('LAPACK_OUT', 'ZGEEV:VR', V_r)\n"
            "         end if\n",
            1,
        ),
        (
            "      call XGESVD('A', 'A', n, n, A, n, sigma, U, n, V_H, n, "
            "work, SIZE(work), info)\n",
            "      if (tap_lapack_on('DGESVD')) "
            "call tap_val('LAPACK_IN', 'DGESVD:A', A)\n"
            "      call XGESVD('A', 'A', n, n, A, n, sigma, U, n, V_H, n, "
            "work, SIZE(work), info)\n"
            "      if (tap_lapack_last()) then\n"
            "         call tap_val('LAPACK_OUT', 'DGESVD:info', info)\n"
            "         call tap_val('LAPACK_OUT', 'DGESVD:S', sigma)\n"
            "         call tap_val('LAPACK_OUT', 'DGESVD:U', U)\n"
            "         call tap_val('LAPACK_OUT', 'DGESVD:VT', V_H)\n"
            "      end if\n",
            1,
        ),
        (
            "      call XGESVD('A', 'A', n, n, A, n, sigma, U, n, V_H, n, "
            "work, SIZE(work), rwork, info)\n",
            "      if (tap_lapack_on('ZGESVD')) "
            "call tap_val('LAPACK_IN', 'ZGESVD:A', A)\n"
            "      call XGESVD('A', 'A', n, n, A, n, sigma, U, n, V_H, n, "
            "work, SIZE(work), rwork, info)\n"
            "      if (tap_lapack_last()) then\n"
            "         call tap_val('LAPACK_OUT', 'ZGESVD:info', info)\n"
            "         call tap_val('LAPACK_OUT', 'ZGESVD:S', sigma)\n"
            "         call tap_val('LAPACK_OUT', 'ZGESVD:U', U)\n"
            "         call tap_val('LAPACK_OUT', 'ZGESVD:VH', V_H)\n"
            "      end if\n",
            1,
        ),
        (
            "         call XGESVX('E', 'N', SIZE(A_, 1), SIZE(B_, 2), A_, "
            "SIZE(A_, 1), AF, SIZE(AF, 1), ipiv, &\n"
            "            equed, R, C, B_, SIZE(B_, 1), X, SIZE(X, 1), rcond, "
            "ferr, berr, work, &\n"
            "            xwork, info)\n",
            "         if (tap_lapack_on("
            "'${'DGESVX' if T == 'real(RD)' else 'ZGESVX'}$')) then\n"
            "            call tap_val('LAPACK_IN', 'XGESVX:A', A_)\n"
            "            call tap_val('LAPACK_IN', 'XGESVX:B', B_)\n"
            "         end if\n"
            "         call XGESVX('E', 'N', SIZE(A_, 1), SIZE(B_, 2), A_, "
            "SIZE(A_, 1), AF, SIZE(AF, 1), ipiv, &\n"
            "            equed, R, C, B_, SIZE(B_, 1), X, SIZE(X, 1), rcond, "
            "ferr, berr, work, &\n"
            "            xwork, info)\n"
            "         if (tap_lapack_last()) then\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:info', info)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:equed', "
            "ICHAR(equed))\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:ipiv', ipiv)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:A_equ', A_)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:AF', AF)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:R', R)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:C', C)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:B_equ', B_)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:X', X)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:rcond', rcond)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:ferr', ferr)\n"
            "            call tap_val('LAPACK_OUT', 'XGESVX:berr', berr)\n"
            "         end if\n",
            1,
        ),
    ],
    "matrix/block_sysmtx_m.fypp": [
        (
            "                  call XGETRF(2*n_e, n_e, M_U, 2*n_e, ipiv, info)\n",
            "                  if (tap_lapack_on('XGETRF_FACTOR')) "
            "call tap_val('LAPACK_IN', 'XGETRF_FACTOR:A', M_U)\n"
            "                  call XGETRF(2*n_e, n_e, M_U, 2*n_e, ipiv, info)\n"
            "                  if (tap_lapack_last()) then\n"
            "                     call tap_val('LAPACK_OUT', "
            "'XGETRF_FACTOR:info', info)\n"
            "                     call tap_val('LAPACK_OUT', "
            "'XGETRF_FACTOR:ipiv', ipiv)\n"
            "                     call tap_val('LAPACK_OUT', "
            "'XGETRF_FACTOR:LU', M_U)\n"
            "                  end if\n",
            1,
        ),
        (
            "            call XGETRF(2*n_e, 2*n_e, M, 2*n_e, ipiv, info)\n"
            "            @:ASSERT(info >= 0, 'negative return from XGETRF')\n\n"
            "            ! Add together",
            "            if (tap_lapack_on('XGETRF_DET')) "
            "call tap_val('LAPACK_IN', 'XGETRF_DET:A', M)\n"
            "            call XGETRF(2*n_e, 2*n_e, M, 2*n_e, ipiv, info)\n"
            "            if (tap_lapack_last()) then\n"
            "               call tap_val('LAPACK_OUT', 'XGETRF_DET:info', info)\n"
            "               call tap_val('LAPACK_OUT', 'XGETRF_DET:ipiv', ipiv)\n"
            "               call tap_val('LAPACK_OUT', 'XGETRF_DET:LU', M)\n"
            "            end if\n"
            "            @:ASSERT(info >= 0, 'negative return from XGETRF')\n\n"
            "            ! Add together",
            1,
        ),
    ],
}

MODE_BUILD = (
    "      call bp%build(st)\n      call bp%factor()\n",
    "      call tap_set_mode_build()\n      call bp%build(st)\n      call bp%factor()\n",
)

MODE_BUILD_FILES = ("ad/ad_bvp_m.fypp", "rad/rad_bvp_m.fypp", "nad/nad_bvp_m.fypp")


def apply_taps(root: pathlib.Path, tap_src: pathlib.Path) -> None:
    src = root / "src"
    for rel, edits in EDITS.items():
        path = src / rel
        text = path.read_text()
        text = _add_use(text, rel)
        for old, new, count in edits:
            text = _replace(text, old, new, count, rel)
        path.write_text(text)
    for rel in MODE_BUILD_FILES:
        path = src / rel
        text = _add_use(path.read_text(), rel)
        # Only the homogeneous (mode) constructor: the first occurrence.
        text = _first_only(text, MODE_BUILD[0], MODE_BUILD[1], rel)
        path.write_text(text)
    shutil.copyfile(tap_src, src / "common" / "gyre_tap_m.fypp")
    mk = root / "build" / "Makefile"
    mk.write_text(
        _replace(
            mk.read_text(),
            "libgyre_SRCS = ad_bound_m.fypp",
            "libgyre_SRCS = gyre_tap_m.fypp ad_bound_m.fypp",
            1,
            "build/Makefile",
        )
    )


def apply_coverage(root: pathlib.Path) -> None:
    inc = root / "build" / "Make.inc"
    inc.write_text(
        _replace(
            inc.read_text(),
            "   FFLAGS += -O2 -fno-backtrace\n",
            "   FFLAGS += -O0 -fno-backtrace --coverage\n",
            1,
            "build/Make.inc",
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("overlay", type=pathlib.Path, help="overlay GYRE root")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--taps", action="store_true")
    mode.add_argument("--coverage", action="store_true")
    args = ap.parse_args()
    root = args.overlay.resolve()
    if not (root / MARKER).is_file():
        print(f"refusing: {root} has no {MARKER} marker", file=sys.stderr)
        return 2
    if args.taps:
        here = pathlib.Path(__file__).resolve().parent
        apply_taps(root, here / "gyre_tap_m.fypp")
    else:
        apply_coverage(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
