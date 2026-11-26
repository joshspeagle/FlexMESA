#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Captures MESA 26.4.1 routine records for record/replay (#552).

Builds a private overlay of a compiled MESA 26.4.1 tree in which
``star/private/turb_support.f90`` gains the capture calls of
``replay_patch.py`` and ``star/private/star_solver.f90`` the step-dump
records of ``../stepdumps/stepdump_patch.py``. The step dump bounds the run:
it stops MESA once ``--models`` models have been solved, and it records the
same models' solver state, so each capture comes with the step dump of the
run that produced it. Each case then runs through
``scripts/mesa_reference/run_mesa_case_26_4_1.sh`` (unmodified inlists,
``MESA_SKIP_OPTIONAL=t``) with both switched on.

Output, per case, under ``<out>/<suite>/``:

* ``<case>.replay``: a header of ``#`` lines recording the MESA version,
  compiler, SDK, thread count, window and the SHA-256 of every inlist,
  followed by the records documented in ``replay_patch.py``;
* ``<case>.stepdump``: the step dump of the same run.

The pristine tree is never modified: the overlay holds real copies of
``star/``, ``lib/`` and ``include/`` and symbolic links for everything else,
and the script checks afterwards that the pristine ``libstar.a`` carries
neither marker.

Usage::

    python run_capture.py --mesa-dir "$MESA_DIR" \\
        --mesasdk-root "$MESASDK_ROOT" --scratch <empty directory> \\
        --out <output directory> [--models 1] \\
        --cases star/conserve_angular_momentum [...]

Linking may need ``LIBRARY_PATH`` to point at a ``libz`` the SDK can use.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stepdumps"))
import replay_patch  # noqa: E402
import run_stepdumps  # noqa: E402
import stepdump_patch  # noqa: E402

# Capture window when the step dump bounds the run (see ``run_case``).
UNBOUNDED_MODELS = 1_000_000_000


def build_overlay(
    mesa_dir: pathlib.Path, overlay: pathlib.Path, env: dict[str, str]
) -> None:
    """Creates and compiles the instrumented overlay of ``mesa_dir``.

    Args:
        mesa_dir: Pristine compiled MESA 26.4.1 tree.
        overlay: Absent directory for the overlay.
        env: SDK environment whose ``MESA_DIR`` is ``overlay``.

    Raises:
        SystemExit: If the pristine ``libstar.a`` carries a marker.
    """
    overlay.mkdir(parents=True)
    for entry in mesa_dir.iterdir():
        target = overlay / entry.name
        if entry.name in ("star", "lib", "include"):
            shutil.copytree(entry, target, symlinks=True)
        else:
            target.symlink_to(entry)
    private = overlay / "star" / "private"
    for name, patch in (
        ("turb_support.f90", replay_patch.patch_text),
        ("star_solver.f90", stepdump_patch.patch_text),
    ):
        src = private / name
        src.write_text(patch(src.read_text(encoding="utf-8")), encoding="utf-8")
    star = overlay / "star"
    for cmd in (["bash", "./mk"], ["bash", "./export"]):
        subprocess.run(cmd, cwd=star, env=env, check=True)
    pristine = (mesa_dir / "lib" / "libstar.a").read_bytes()
    for marker in (replay_patch.MARKER, stepdump_patch.MARKER):
        if marker.encode() in pristine:
            raise SystemExit(f"pristine libstar.a carries {marker}")


def _header(
    case_dir: pathlib.Path,
    ident: str,
    models: int,
    first_model: str,
    meta: dict[str, str],
) -> list[str]:
    """Returns the provenance header lines of one capture."""
    lines = [
        f"# case: {ident}",
        f"# mesa_version: {meta['mesa_version']}",
        f"# compiler: {meta['compiler']}",
        f"# mesa_sdk_version: {meta['mesa_sdk_version']}",
        f"# omp_num_threads: {run_stepdumps.OMP_THREADS}",
        f"# models_per_star: {models}",
        f"# first_model: {first_model}",
        "# produced by: benchmarks/fortran_mesa/replay/run_capture.py "
        "(capture calls only, MESA_SKIP_OPTIONAL=t); "
        "values are IEEE-754 binary64 bit patterns in hexadecimal",
    ]
    stepdump_header = run_stepdumps._header(case_dir, ident, models, meta)
    lines += [ln for ln in stepdump_header if ln.startswith("# inlist_")]
    return lines


def run_case(
    ident: str,
    env: dict[str, str],
    work: pathlib.Path,
    out: pathlib.Path,
    models: int,
    first_model: int | None,
    timeout_sec: int,
    meta: dict[str, str],
) -> str:
    """Runs one case with capture on and writes its output files.

    Args:
        ident: ``suite/case``.
        env: SDK environment with the overlay as ``MESA_DIR``.
        work: Scratch directory for case copies.
        out: Output directory.
        models: Models captured per star.
        first_model: First model number captured, or ``None`` for the
            first model of the run.
        timeout_sec: Wall-clock limit for build and run.
        meta: Build provenance from ``run_stepdumps.build_meta``.

    Returns:
        A one-line status.
    """
    suite, case = ident.split("/", 1)
    tag = f"m{models}_f{first_model if first_model is not None else 'start'}"
    raw = work / f"{suite}__{case}__{tag}.replay.raw"
    dump = work / f"{suite}__{case}__{tag}.stepdump.raw"
    if raw.exists() or dump.exists():
        raise SystemExit(f"{raw}: earlier capture present; use a new scratch")
    case_root = run_stepdumps._case_root(env, suite, case, work / tag)
    run_env = {
        **env,
        "OUT": str(case_root),
        replay_patch.MARKER: str(raw),
        # The step dump's stop bounds the run, so the capture window is left
        # open: a loaded model keeps its saved model number until
        # set_initial_model_number resets it, so a count of distinct model
        # numbers would not line up with the solved models.
        f"{replay_patch.MARKER}_MODELS": str(UNBOUNDED_MODELS),
        stepdump_patch.MARKER: str(dump),
        f"{stepdump_patch.MARKER}_STEPS": str(models),
        f"{stepdump_patch.MARKER}_STOP": "1",
    }
    if first_model is not None:
        run_env[f"{replay_patch.MARKER}_FIRST"] = str(first_model)
        run_env[f"{stepdump_patch.MARKER}_FIRST"] = str(first_model)
    status = run_stepdumps._run(ident, run_env, timeout_sec)
    if not raw.exists():
        return f"{ident}: {status}, no capture"
    first = str(first_model) if first_model is not None else "first of run"
    header = _header(case_root / case, ident, models, first, meta)
    dest = out / suite / f"{case}.replay"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n")
        with raw.open(encoding="utf-8") as src:
            shutil.copyfileobj(src, fh)
    meta_dump = {**meta, "first_model": first}
    run_stepdumps._write_dump(
        ident,
        dump,
        out / suite / f"{case}.stepdump",
        case_root / case,
        models,
        meta_dump,
        status,
    )
    counts: dict[str, int] = {}
    with raw.open(encoding="utf-8") as src:
        for line in src:
            tag_ = line.split(" ", 1)[0].strip()
            counts[tag_] = counts.get(tag_, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    size_mb = dest.stat().st_size / 1e6
    return f"{ident}: {status}, {summary}, {size_mb:.1f} MB"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesa-dir", type=pathlib.Path, required=True)
    parser.add_argument("--mesasdk-root", type=pathlib.Path, required=True)
    parser.add_argument("--scratch", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--models", type=int, default=1)
    parser.add_argument("--first-model", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument(
        "--reuse-overlay",
        action="store_true",
        help="Use an overlay already built under --scratch.",
    )
    args = parser.parse_args()
    mesa_dir = args.mesa_dir.resolve()
    scratch = args.scratch.resolve()
    overlay = scratch / "mesa_overlay"
    env = run_stepdumps.sdk_env(args.mesasdk_root.resolve(), overlay)
    if not args.reuse_overlay:
        if scratch.exists() and any(scratch.iterdir()):
            raise SystemExit("--scratch must be empty or absent")
        build_overlay(mesa_dir, overlay, env)
    meta = run_stepdumps.build_meta(mesa_dir, env)
    work = scratch / "cases"
    work.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    for ident in args.cases:
        print(
            run_case(
                ident,
                env,
                work,
                out,
                args.models,
                args.first_model,
                args.timeout_sec,
                meta,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
