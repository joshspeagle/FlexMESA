#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""MESA 26.4.1 initial-model captures for the relax_154 oracle tests.

Builds a private overlay of a compiled MESA 26.4.1 tree in which only
``star/private/init.f90`` gains the ``write`` statements of
``relax154_patch.py``, runs each inlist in ``inlists/`` from a copy of
``star/work`` until the initial model exists (before any evolve step), and
writes, per setup, ``<out>/<setup>.dump`` (full capture) and
``<out>/<setup>.inlist``. With ``--sample-out`` it also writes the small
committed subset (scalars plus the zones the Rust test inspects).

The pristine tree is never modified: the overlay holds real copies of
``star/``, ``lib/`` and ``include/`` and symbolic links for everything else,
and the script checks afterwards that the pristine ``libstar.a`` carries no
instrumentation marker.

Usage::

    python run_relax154_oracles.py --mesa-dir "$MESA_DIR" \\
        --mesasdk-root "$MESASDK_ROOT" --scratch <empty directory> \\
        --out <output directory> [--sample-out <directory>]

Linking may need ``LIBRARY_PATH`` to point at a ``libz`` the SDK can use.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "stepdumps"))
import relax154_patch  # noqa: E402
import run_stepdumps  # noqa: E402

# setup name -> (inlist file, 1-based zones kept in the committed sample).
# Zones are the ones the Rust test inspects (0-based 0, 387, 773 and
# 0, 516, 1032), converted to MESA's 1-based zone index.
SETUPS: dict[str, tuple[str, tuple[int, ...]]] = {
    "create_initial_model_pp_and_cno_extras": (
        "create_initial_model_pp_and_cno_extras.inlist",
        (1, 388, 774),
    ),
    "make_brown_dwarf_basic_net": (
        "make_brown_dwarf_basic_net.inlist",
        (1, 388, 774),
    ),
    "pre_ms_15m_dynamo": ("pre_ms_15m_dynamo.inlist", (1, 517, 1033)),
}


def build_overlay(
    mesa_dir: pathlib.Path, overlay: pathlib.Path, env: dict[str, str]
) -> None:
    """Creates and compiles the instrumented overlay of ``mesa_dir``.

    Args:
        mesa_dir: Pristine compiled MESA 26.4.1 tree.
        overlay: Absent directory for the overlay.
        env: SDK environment whose ``MESA_DIR`` is ``overlay``.
    """
    overlay.mkdir(parents=True)
    for entry in mesa_dir.iterdir():
        target = overlay / entry.name
        if entry.name in ("star", "lib", "include"):
            shutil.copytree(entry, target, symlinks=True)
        else:
            target.symlink_to(entry)
    init = overlay / "star" / "private" / "init.f90"
    init.write_text(
        relax154_patch.patch_text(init.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    star = overlay / "star"
    for cmd in (["bash", "./mk"], ["bash", "./export"]):
        subprocess.run(cmd, cwd=star, env=env, check=True)


def run_setup(
    name: str,
    inlist: pathlib.Path,
    mesa_dir: pathlib.Path,
    env: dict[str, str],
    scratch: pathlib.Path,
    out: pathlib.Path,
    meta: dict[str, str],
) -> pathlib.Path:
    """Builds and runs one setup, returning the path of its ``.dump``.

    Args:
        name: Setup name.
        inlist: Inlist file to run as ``inlist``.
        mesa_dir: Pristine tree (source of ``star/work``).
        env: SDK environment (``MESA_DIR`` is the overlay).
        scratch: Scratch directory.
        out: Output directory.
        meta: Version metadata from ``run_stepdumps.build_meta``.

    Returns:
        The written dump path.
    """
    work = scratch / "work" / name
    shutil.copytree(mesa_dir / "star" / "work", work, symlinks=True)
    shutil.copyfile(inlist, work / "inlist")
    raw = scratch / f"{name}.raw"
    raw.unlink(missing_ok=True)
    subprocess.run(["bash", "./mk"], cwd=work, env=env, check=True)
    run_env = dict(env)
    run_env["FLEXMESA_RELAX154_DUMP"] = str(raw)
    run_env["FLEXMESA_RELAX154_STOP"] = "1"
    subprocess.run(["bash", "./rn"], cwd=work, env=run_env, check=True)
    text = inlist.read_text(encoding="utf-8")
    digest = hashlib.sha256(inlist.read_bytes()).hexdigest()
    header = [
        f"# setup: {name}",
        "# point: initial model built by model_builder in star/private/"
        "init.f90, before any evolve step; stages BUILD, LOADED, FINAL as in "
        "relax154_patch.py; ZONE k is MESA's 1-based zone index (k=1 is the "
        "surface)",
        f"# mesa_version: {meta['mesa_version']}",
        f"# compiler: {meta['compiler']}",
        f"# mesa_sdk_version: {meta['mesa_sdk_version']}",
        f"# omp_num_threads: {run_stepdumps.OMP_THREADS}",
        f"# inlist_sha256: {inlist.name} {digest}",
        "# produced by: benchmarks/fortran_mesa/relax_154_oracles/"
        "run_relax154_oracles.py (write statements only); each float is "
        "'decimal(ES25.16E3) hex(IEEE-754 binary64)'",
    ]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(inlist, out / f"{name}.inlist")
    dump = out / f"{name}.dump"
    dump.write_text(
        "\n".join(header) + "\n" + raw.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert text  # inlist was readable
    return dump


def write_sample(
    dump: pathlib.Path, zones: tuple[int, ...], sample: pathlib.Path
) -> None:
    """Writes the committed subset of ``dump``.

    Keeps the header, every non-zone record, and the ``ZONE`` and ``XA``
    records of ``zones``.

    Args:
        dump: Full dump.
        zones: 1-based zone indices to keep.
        sample: Output path.
    """
    keep = {str(k) for k in zones}
    lines = []
    for line in dump.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts[0] in ("ZONE", "XA") and parts[2] not in keep:
            continue
        lines.append(line)
    sample.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Runs every setup and writes the dumps."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mesa-dir", type=pathlib.Path, required=True)
    ap.add_argument("--mesasdk-root", type=pathlib.Path, required=True)
    ap.add_argument("--scratch", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--sample-out", type=pathlib.Path)
    args = ap.parse_args()
    overlay = args.scratch / "mesa_overlay"
    env = run_stepdumps.sdk_env(args.mesasdk_root, overlay)
    meta = run_stepdumps.build_meta(args.mesa_dir, env)
    if not overlay.exists():
        build_overlay(args.mesa_dir, overlay, env)
    pristine = (args.mesa_dir / "lib" / "libstar.a").read_bytes()
    if relax154_patch.MARKER.encode() in pristine:
        raise SystemExit("pristine libstar.a carries the dump marker")
    for name, (inlist, zones) in SETUPS.items():
        dump = run_setup(
            name,
            HERE / "inlists" / inlist,
            args.mesa_dir,
            env,
            args.scratch,
            args.out,
            meta,
        )
        if args.sample_out:
            args.sample_out.mkdir(parents=True, exist_ok=True)
            write_sample(dump, zones, args.sample_out / f"{name}.sample")
    if (
        relax154_patch.MARKER.encode()
        in (args.mesa_dir / "lib" / "libstar.a").read_bytes()
    ):
        raise SystemExit("pristine libstar.a carries the dump marker")


if __name__ == "__main__":
    main()
