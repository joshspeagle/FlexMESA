#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Instrumented MESA 26.4.1 run of the original binary RLOF fixture.

Builds a private overlay of a compiled MESA 26.4.1 tree in which only
``star/private/star_solver.f90`` and ``star/private/init_model.f90`` gain
``write`` statements (no arithmetic changes), runs the original
``(1.5, 1.0, 0.45 d)`` fixture case from the parent directory with
``max_number_retries = 40``, and writes IEEE-754 hexadecimal reference
tables into this directory:

* ``zams_model0_26_4_1.tsv``: ``lnd lnT lnR L`` per zone right after ZAMS
  interpolation (end of ``get_zams_model``, ``init_model.f90:65-135``).
* ``first_call_state_26_4_1.tsv``: ``lnd lnT lnR L`` and the mass fractions
  per zone at the first solver call of model 1 (after
  ``prepare_for_new_step``).
* ``solver_calls_26_4_1.tsv``: per solver call, the model number, ``dt``,
  initial ``residual_norm`` and ``max_residual``.
* ``line_search_model1_26_4_1.tsv``: the model-1 line-search records of
  ``do_solver_work`` and ``adjust_correction``
  (``star_solver.f90:395-460, 737-951``).

The pristine MESA tree is never modified: the overlay holds real copies of
``star/``, ``lib/`` and ``include/`` and symbolic links for everything else,
and the script checks afterwards that the pristine ``libstar.a`` carries no
instrumentation marker.

Usage::

    python run_instrumented.py --mesa-dir "$MESA_DIR" \\
        --mesasdk-root "$MESASDK_ROOT" --scratch <empty scratch directory>

Linking may need ``LIBRARY_PATH`` to point at a ``libz`` the SDK can use.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE_DIR = HERE.parent
MARKER = "LSTRACE"
HEX = "Z16.16"


def _h(name: str) -> str:
    """Returns the Fortran expression printing ``name``'s IEEE-754 bits."""
    return f"transfer({name}, 0_8)"


# (file, anchor, text inserted after the anchor). Each anchor must occur
# exactly once in the upstream file.
SOLVER_EDITS: list[tuple[str, str]] = [
    (
        "         integer :: nz, iter, max_tries, tiny_corr_cnt, i, &\n",
        "            lstrace_k, &\n",
    ),
    (
        "            s% num_solver_iterations = 0\n",
        "            if (s% model_number == 1 .and. s% dt > 3.1d10) then\n"
        "               do lstrace_k = 1, nz\n"
        f"                  write(*,'(A,1X,I0,4(1X,{HEX}))') "
        "'LSTRACE XH', lstrace_k, "
        + ", ".join(_h(f"s% xh({i},lstrace_k)") for i in range(1, 5))
        + "\n"
        f"                  write(*,'(A,1X,I0,99(1X,{HEX}))') "
        "'LSTRACE XA', lstrace_k, "
        f"({_h('s% xa(i,lstrace_k)')}, i=1,s% species)\n"
        "               end do\n"
        "            end if\n"
        f"            write(*,'(A,1X,I0,3(1X,{HEX}))') 'LSTRACE CALL', "
        f"s% model_number, {_h('s% dt')}, {_h('residual_norm')}, "
        f"{_h('max_residual')}\n",
    ),
    (
        "               f_target = max(fold/2, fold + alf*coeff*slope)\n",
        f"               write(*,'(A,1X,I0,1X,I0,1X,I0,4(1X,{HEX}))') "
        "'LSTRACE TRY', s% model_number, s% solver_iter, iter, "
        f"{_h('alam')}, {_h('coeff')}, {_h('f')}, {_h('f_target')}\n",
    ),
    (
        "            fold = f\n",
        f"            write(*,'(A,1X,I0,1X,I0,1X,{HEX})') 'LSTRACE FOLD', "
        f"s% model_number, s% solver_iter, {_h('f')}\n",
    ),
]
# The NEWTON record goes before the anchor, so it is handled separately.
NEWTON_ANCHOR = (
    "               f = 0d0\n               call adjust_correction( &\n"
)
NEWTON_TEXT = (
    f"               write(*,'(A,1X,I0,1X,I0,3(1X,{HEX}))') "
    "'LSTRACE NEWTON', s% model_number, iter, "
    f"{_h('min_corr_coeff')}, {_h('correction_factor')}, {_h('slope')}\n"
)
ZAMS_EDITS: list[tuple[str, str]] = [
    (
        "         integer :: nz\n"
        "         real(dp), dimension(:,:), pointer :: xh, xa\n",
        "         integer :: lstrace_k\n",
    ),
    (
        "         s% omega(1:nz) = omega(1:nz)\n",
        "         do lstrace_k = 1, nz\n"
        f"            write(*,'(A,1X,I0,4(1X,{HEX}))') 'LSTRACE ZAMS0', "
        "lstrace_k, "
        + ", ".join(_h(f"s% xh({i},lstrace_k)") for i in range(1, 5))
        + "\n"
        "         end do\n",
    ),
]


def _patch(path: pathlib.Path, edits: list[tuple[str, str]]) -> None:
    """Inserts each edit's text after its unique anchor in ``path``."""
    text = path.read_text(encoding="utf-8")
    for anchor, insert in edits:
        count = text.count(anchor)
        if count != 1:
            raise SystemExit(
                f"{path.name}: anchor found {count} times:\n{anchor}"
            )
        text = text.replace(anchor, anchor + insert)
    if path.name == "star_solver.f90":
        count = text.count(NEWTON_ANCHOR)
        if count != 1:
            raise SystemExit(f"{path.name}: NEWTON anchor found {count} times")
        text = text.replace(NEWTON_ANCHOR, NEWTON_TEXT + NEWTON_ANCHOR)
    path.write_text(text, encoding="utf-8")


def _sdk_env(
    mesasdk_root: pathlib.Path, mesa_dir: pathlib.Path
) -> dict[str, str]:
    """Returns the environment of a shell that sourced ``mesasdk_init.sh``."""
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{mesasdk_root}/bin/mesasdk_init.sh" && env -0',
        ],
        env={**os.environ, "MESASDK_ROOT": str(mesasdk_root)},
        capture_output=True,
        check=True,
    ).stdout
    env = dict(
        item.split("=", 1) for item in out.decode().split("\0") if "=" in item
    )
    env["MESA_DIR"] = str(mesa_dir)
    env["OMP_NUM_THREADS"] = "4"
    return env


def build_overlay(mesa_dir: pathlib.Path, overlay: pathlib.Path) -> None:
    """Creates the instrumented overlay tree of ``mesa_dir`` at ``overlay``."""
    overlay.mkdir(parents=True)
    for entry in mesa_dir.iterdir():
        target = overlay / entry.name
        if entry.name in ("star", "lib", "include"):
            shutil.copytree(entry, target, symlinks=True)
        else:
            target.symlink_to(entry)
    _patch(overlay / "star" / "private" / "star_solver.f90", SOLVER_EDITS)
    _patch(overlay / "star" / "private" / "init_model.f90", ZAMS_EDITS)


def _hdr(work: pathlib.Path) -> list[str]:
    """Returns provenance header lines read from ``binary_history.data``."""
    lines = (
        (work / "binary_history.data").read_text(encoding="utf-8").splitlines()
    )
    names = lines[1].split()
    vals = [v.strip('"') for v in lines[2].split()]
    h = dict(zip(names, vals, strict=False))
    return [
        f"# mesa_version: {h['version_number']}",
        f"# compiler: {h['compiler']} {h['build']}",
        f"# mesa_sdk_version: {h['MESA_SDK_version']}",
        "# omp_num_threads: 4",
        "# case: original (1.5 Msun donor, 1.0 Msun point mass, P = 0.45 d), "
        "max_number_retries = 40",
        "# produced by: run_instrumented.py (write statements only); values "
        "are IEEE-754 binary64 bit patterns in hexadecimal",
    ]


def write_tables(work: pathlib.Path) -> None:
    """Extracts the reference tables from ``work/run.log`` into ``HERE``."""
    header = _hdr(work)
    recs = [
        line.split()[1:]
        for line in (work / "run.log").read_text(encoding="utf-8").splitlines()
        if line.startswith(MARKER)
    ]
    zams = [r[1:] for r in recs if r[0] == "ZAMS0"]
    xh = {r[1]: r[2:] for r in recs if r[0] == "XH"}
    xa = {r[1]: r[2:] for r in recs if r[0] == "XA"}
    calls = [r[1:] for r in recs if r[0] == "CALL"]
    first_call = next(i for i, r in enumerate(recs) if r[0] == "CALL")
    model1 = [
        r
        for r in recs[first_call:]
        if r[0] in ("CALL", "NEWTON", "FOLD", "TRY") and r[1] == "1"
    ]
    n_species = len(next(iter(xa.values())))

    def emit(name: str, cols: list[str], rows: list[list[str]]) -> None:
        body = ["\t".join(cols)] + ["\t".join(r) for r in rows]
        (HERE / name).write_text(
            "\n".join(header + body) + "\n", encoding="utf-8"
        )

    emit("zams_model0_26_4_1.tsv", ["k", "lnd", "lnT", "lnR", "L"], zams)
    emit(
        "first_call_state_26_4_1.tsv",
        ["k", "lnd", "lnT", "lnR", "L"]
        + [f"xa{j}" for j in range(1, n_species + 1)],
        [[k, *xh[k], *xa[k]] for k in xh],
    )
    emit(
        "solver_calls_26_4_1.tsv",
        ["call", "model", "dt", "residual_norm", "max_residual"],
        [[str(i + 1), *c] for i, c in enumerate(calls)],
    )
    emit(
        "line_search_model1_26_4_1.tsv",
        ["record", "fields"],
        [[r[0], " ".join(r[2:])] for r in model1],
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesa-dir", type=pathlib.Path, required=True)
    parser.add_argument("--mesasdk-root", type=pathlib.Path, required=True)
    parser.add_argument("--scratch", type=pathlib.Path, required=True)
    args = parser.parse_args()
    mesa_dir = args.mesa_dir.resolve()
    args.scratch = args.scratch.resolve()
    args.mesasdk_root = args.mesasdk_root.resolve()
    if any(args.scratch.iterdir()) if args.scratch.exists() else False:
        raise SystemExit("--scratch must be empty or absent")
    overlay = args.scratch / "mesa_overlay"
    build_overlay(mesa_dir, overlay)
    env = _sdk_env(args.mesasdk_root, overlay)
    star = overlay / "star"
    subprocess.run(["bash", "./mk"], cwd=star, env=env, check=True)
    subprocess.run(["bash", "./export"], cwd=star, env=env, check=True)
    pristine = (mesa_dir / "lib" / "libstar.a").read_bytes()
    if MARKER.encode() in pristine:
        raise SystemExit(
            "pristine libstar.a carries the instrumentation marker"
        )

    work = args.scratch / "work"
    shutil.copytree(mesa_dir / "binary" / "work", work, symlinks=True)
    for name in ("inlist", "inlist_project", "inlist1", "inlist2"):
        shutil.copy2(FIXTURE_DIR / name, work / name)
    shutil.copy2(
        FIXTURE_DIR / "run_binary_extras.f90",
        work / "src" / "run_binary_extras.f90",
    )
    subprocess.run(["bash", "./mk"], cwd=work, env=env, check=True)
    with (work / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            ["bash", "./rn"],
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1800,
        )
    write_tables(work)


if __name__ == "__main__":
    main()
