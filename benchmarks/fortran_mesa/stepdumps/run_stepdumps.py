#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""MESA 26.4.1 step dumps for upstream test-suite cases (#372).

Builds a private overlay of a compiled MESA 26.4.1 tree in which only
``star/private/star_solver.f90`` gains the ``write`` statements of
``stepdump_patch.py``, then runs each requested case through
``scripts/mesa_reference/run_mesa_case_26_4_1.sh`` (unmodified inlists,
``MESA_SKIP_OPTIONAL=t``, as for the stored reference histories) with the dump
switched on. Each run stops after ``--steps`` models per star. By default
these are the first models of the run, so the dump covers the start of the
first ``./rn`` stage. With ``--first-model N`` the dump starts at model number
``N`` instead, which reaches later in the run (and into later stages, whose
earlier stages then run to completion first).

Output, per case, ``<out>/<suite>/<case>.stepdump``: a header of ``#`` lines
recording the MESA version, compiler and SDK (``data/version_number``,
``gfortran -dumpfullversion``, ``mesasdk_version``), the window, the thread
count and the SHA-256 of every inlist, followed by the records documented in
``stepdump_patch.py``.

The pristine tree is never modified: the overlay holds real copies of
``star/``, ``lib/`` and ``include/`` and symbolic links for everything else,
and the script checks afterwards that the pristine ``libstar.a`` carries no
instrumentation marker.

Usage::

    python run_stepdumps.py --mesa-dir "$MESA_DIR" \\
        --mesasdk-root "$MESASDK_ROOT" --scratch <empty directory> \\
        --out <output directory> [--steps 3] [--jobs 2] \\
        [--cases star/1.5M_with_diffusion binary/evolve_both_stars ...]

Without ``--cases`` every case of the three test suites runs. Linking may need
``LIBRARY_PATH`` to point at a ``libz`` the SDK can use.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
from concurrent import futures

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stepdump_patch  # noqa: E402

REPO = HERE.parents[2]
RUN_CASE = REPO / "scripts" / "mesa_reference" / "run_mesa_case_26_4_1.sh"
SUITES = ("star", "binary", "astero")
OMP_THREADS = 4


def sdk_env(
    mesasdk_root: pathlib.Path, mesa_dir: pathlib.Path
) -> dict[str, str]:
    """Returns the environment of a shell that sourced ``mesasdk_init.sh``.

    Args:
        mesasdk_root: MESA SDK root.
        mesa_dir: Value for ``MESA_DIR`` (the overlay).

    Returns:
        The environment, with ``MESA_DIR`` and ``OMP_NUM_THREADS`` set.
    """
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
    env["OMP_NUM_THREADS"] = str(OMP_THREADS)
    return env


def build_overlay(
    mesa_dir: pathlib.Path, overlay: pathlib.Path, env: dict[str, str]
) -> None:
    """Creates and compiles the instrumented overlay of ``mesa_dir``.

    Args:
        mesa_dir: Pristine compiled MESA 26.4.1 tree.
        overlay: Absent directory for the overlay.
        env: SDK environment whose ``MESA_DIR`` is ``overlay``.

    Raises:
        SystemExit: If the pristine ``libstar.a`` carries the marker.
    """
    overlay.mkdir(parents=True)
    for entry in mesa_dir.iterdir():
        target = overlay / entry.name
        if entry.name in ("star", "lib", "include"):
            shutil.copytree(entry, target, symlinks=True)
        else:
            target.symlink_to(entry)
    solver = overlay / "star" / "private" / "star_solver.f90"
    solver.write_text(
        stepdump_patch.patch_text(solver.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    star = overlay / "star"
    for cmd in (["bash", "./mk"], ["bash", "./export"]):
        subprocess.run(cmd, cwd=star, env=env, check=True)
    pristine = (mesa_dir / "lib" / "libstar.a").read_bytes()
    if stepdump_patch.MARKER.encode() in pristine:
        raise SystemExit("pristine libstar.a carries the step-dump marker")


def list_cases(mesa_dir: pathlib.Path) -> list[str]:
    """Returns ``suite/case`` for every case in ``do1_test_source``.

    Args:
        mesa_dir: MESA tree.

    Returns:
        The case identifiers in upstream order.
    """
    cases = []
    for suite in SUITES:
        src = mesa_dir / suite / "test_suite" / "do1_test_source"
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "do_one":
                cases.append(f"{suite}/{parts[1]}")
    return cases


def build_meta(mesa_dir: pathlib.Path, env: dict[str, str]) -> dict[str, str]:
    """Returns the MESA version, compiler version and SDK version of a build.

    Args:
        mesa_dir: The MESA tree (``data/version_number``).
        env: SDK environment (``gfortran``, ``mesasdk_version``).

    Returns:
        Keys ``mesa_version``, ``compiler`` and ``mesa_sdk_version``.
    """

    def run(cmd: list[str]) -> str:
        return subprocess.run(
            cmd, env=env, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "mesa_version": (mesa_dir / "data" / "version_number")
        .read_text(encoding="utf-8")
        .strip(),
        "compiler": "gfortran " + run(["gfortran", "-dumpfullversion"]),
        "mesa_sdk_version": run(["mesasdk_version"]),
    }


def _header(
    case_dir: pathlib.Path, ident: str, steps: int, meta: dict[str, str]
) -> list[str]:
    """Returns the provenance header lines of one dump."""
    lines = [
        f"# case: {ident}",
        f"# mesa_version: {meta['mesa_version']}",
        f"# compiler: {meta['compiler']}",
        f"# mesa_sdk_version: {meta['mesa_sdk_version']}",
        f"# omp_num_threads: {OMP_THREADS}",
        f"# steps_per_star: {steps}",
        f"# first_model: {meta.get('first_model', 'first of the run')}",
        "# produced by: benchmarks/fortran_mesa/stepdumps/run_stepdumps.py "
        "(write statements only, MESA_SKIP_OPTIONAL=t); "
        "values are IEEE-754 binary64 bit patterns in hexadecimal",
    ]
    for inlist in sorted(case_dir.rglob("inlist*")):
        if not inlist.is_file():
            continue
        digest = hashlib.sha256(inlist.read_bytes()).hexdigest()
        rel = inlist.relative_to(case_dir)
        lines.append(f"# inlist_sha256: {rel} {digest}")
    return lines


def run_case(
    ident: str,
    env: dict[str, str],
    work: pathlib.Path,
    out: pathlib.Path,
    steps: int,
    timeout_sec: int,
    meta: dict[str, str],
) -> str:
    """Runs one case with the dump on and writes its ``.stepdump`` file.

    Args:
        ident: ``suite/case``.
        env: SDK environment with the overlay as ``MESA_DIR``.
        work: Scratch directory for case copies.
        out: Output directory.
        steps: Models dumped per star.
        timeout_sec: Wall-clock limit for build and run.
        meta: Build provenance from ``build_meta``.

    Returns:
        A one-line status.
    """
    suite, case = ident.split("/", 1)
    dest = out / suite / f"{case}.stepdump"
    window = f"# first_model: {meta.get('first_model', 'first of the run')}"
    steps_line = f"# steps_per_star: {steps}"
    if dest.exists():
        old = dest.read_text(encoding="utf-8")
        if window not in old or steps_line not in old:
            raise SystemExit(f"{dest}: exists with a different window")
        if "\nDONE\n" in old:
            return f"{ident}: kept existing complete dump"
    # The window is part of the raw file name, so a run with another window
    # never reuses these records.
    tag = f"s{steps}_f{meta.get('first_model', 'start')}"
    raw = work / f"{suite}__{case}__{tag}.raw"
    case_root = _case_root(env, suite, case, work / tag)
    if (case_root / case).exists() and not raw.exists():
        return f"{ident}: case copy exists in scratch, not rerun"
    run_env = {
        **env,
        "OUT": str(case_root),
        stepdump_patch.MARKER: str(raw),
        f"{stepdump_patch.MARKER}_STEPS": str(steps),
        f"{stepdump_patch.MARKER}_STOP": "1",
    }
    if (case_root / case).exists():
        status = "reused earlier run"
    else:
        status = _run(ident, run_env, timeout_sec)
    if not raw.exists():
        return f"{ident}: {status}, no dump"
    return _write_dump(ident, raw, dest, case_root / case, steps, meta, status)


def _case_root(
    env: dict[str, str], suite: str, case: str, work: pathlib.Path
) -> pathlib.Path:
    """Returns the directory the case is copied into.

    A case whose ``rn`` sets ``MESA_DIR=../../..`` runs only from inside a
    MESA tree, so it is copied into the overlay's own ``star/`` (a real copy;
    ``binary/`` and ``astero/`` are links to the pristine tree, which is never
    written to).

    Raises:
        SystemExit: For such a case outside the star suite.
    """
    overlay = pathlib.Path(env["MESA_DIR"])
    rn = overlay / suite / "test_suite" / case / "rn"
    if "MESA_DIR=../../.." not in rn.read_text(encoding="utf-8"):
        return work / suite
    if suite != "star":
        raise SystemExit(f"{suite}/{case}: rn needs a MESA tree")
    return overlay / "star" / f"stepdump_cases_{work.name}"


def _run(ident: str, run_env: dict[str, str], timeout_sec: int) -> str:
    """Runs one case through the reference runner; returns its status."""
    suite, case = ident.split("/", 1)
    try:
        proc = subprocess.run(
            ["bash", str(RUN_CASE), suite, case],
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        status = f"rc={proc.returncode}"
    except subprocess.TimeoutExpired:
        status = "timeout"
    return status


def _write_dump(
    ident: str,
    raw: pathlib.Path,
    dest: pathlib.Path,
    case_dir: pathlib.Path,
    steps: int,
    meta: dict[str, str],
    status: str,
) -> str:
    """Writes the header and the raw records to ``dest``."""
    records = raw.read_text(encoding="utf-8").splitlines()
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = _header(case_dir, ident, steps, meta)
    dest.write_text("\n".join(header + records) + "\n", encoding="utf-8")
    done = "DONE" in records
    return f"{ident}: {status}, {len(records)} records, done={done}"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesa-dir", type=pathlib.Path, required=True)
    parser.add_argument("--mesasdk-root", type=pathlib.Path, required=True)
    parser.add_argument("--scratch", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--first-model",
        type=int,
        default=None,
        help="First model number dumped (a window later in the run).",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument(
        "--reuse-overlay",
        action="store_true",
        help="Use an overlay already built under --scratch.",
    )
    args = parser.parse_args()
    mesa_dir = args.mesa_dir.resolve()
    scratch = args.scratch.resolve()
    overlay = scratch / "mesa_overlay"
    env = sdk_env(args.mesasdk_root.resolve(), overlay)
    if not args.reuse_overlay:
        if scratch.exists() and any(scratch.iterdir()):
            raise SystemExit("--scratch must be empty or absent")
        build_overlay(mesa_dir, overlay, env)
    meta = build_meta(mesa_dir, env)
    if args.first_model is not None:
        env[f"{stepdump_patch.MARKER}_FIRST"] = str(args.first_model)
        meta["first_model"] = str(args.first_model)
    cases = args.cases or list_cases(mesa_dir)
    work = scratch / "cases"
    work.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    with futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [
            pool.submit(
                run_case, c, env, work, out, args.steps, args.timeout_sec, meta
            )
            for c in cases
        ]
        for job in futures.as_completed(jobs):
            try:
                print(job.result(), flush=True)
            except Exception as exc:
                print(f"error: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
