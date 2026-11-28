#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""MESA 26.4.1 step dumps for upstream test-suite cases (#372).

Builds a private overlay of a compiled MESA 26.4.1 tree in which only
``star/private/star_solver.f90`` gains the ``write`` statements of
``stepdump_patch.py``, then runs each requested case through
``scripts/mesa_reference/run_mesa_case_26_4_1.sh`` (unmodified inlists,
``MESA_SKIP_OPTIONAL=t``, as for the stored reference histories) with the dump
switched on. Each run stops after ``--steps`` models per star. By default
these are the first models of the run, so the dump covers the start of the
first ``./rn`` stage. The window can be placed later in the run:

* ``--first-model N``: the dump starts at model number ``N``.
* ``--occurrence K``: with ``--first-model``, the window opens the K-th time
  the model number rises to ``N`` (default 1). Model numbers restart when a
  relax phase is followed by the evolve phase, so ``--first-model 70
  --occurrence 2`` selects model 70 of the second phase.
* ``--stage S``: only the S-th process of the case (1-based; every ``RUN``
  section of a multi-inlist ``rn``) is dumped. Earlier stages run to
  completion first.
* ``--keep-running``: the run continues past the window instead of stopping
  when it is dumped (a ``DONE`` record still marks the end of the window).
  A selected ``--stage`` needs no such option: earlier stages are never
  stopped, only the selected one.

Opt-in diagnostics (``xtra_patch.py``; the default dump is unchanged):

* ``--xtra star``: tagged ``XTRA`` solver records in the same dump file.
  ``--xtra-zones`` and ``--xtra-corr-zones`` (``1,262,474:481``) choose the
  zones of the per-zone values.
* ``--xtra binary-rlo``: implicit mass-transfer iteration records in
  ``<out>/<suite>/<case>.xtra``. ``--xtra-binary-models FIRST:LAST`` bounds
  the binary models written; the run stops after ``LAST``. The case runs to
  that point (the step-dump window is not used to stop it).

Output, per case, ``<out>/<suite>/<case>.stepdump``: a header of ``#`` lines
recording the MESA version, compiler and SDK (``data/version_number``,
``gfortran -dumpfullversion``, ``mesasdk_version``), the window, the thread
count and the SHA-256 of every inlist, followed by the records documented in
``stepdump_patch.py``. The window is recorded in the ``stage``,
``occurrence``, ``first_model`` and ``steps_per_star`` lines. A dump written
by a patched solver is complete, and ``stepdump_compare`` skips record types
it does not know (and reports their count), so a dump that carries extra
diagnostic records compares as it is.

The pristine tree is never modified: the overlay holds real copies of
``star/``, ``lib/`` and ``include/`` and symbolic links for everything else,
and the script checks afterwards that the pristine ``libstar.a`` carries no
instrumentation marker.

Usage::

    python run_stepdumps.py --mesa-dir "$MESA_DIR" \\
        --mesasdk-root "$MESASDK_ROOT" --scratch <empty directory> \\
        --out <output directory> [--steps 3] [--jobs 2] \\
        [--first-model 70 --occurrence 2 --stage 2 --keep-running] \\
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
import signal
import subprocess
import sys
import time
from concurrent import futures

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import snapshot_patch  # noqa: E402
import stepdump_patch  # noqa: E402
import xtra_patch  # noqa: E402

REPO = HERE.parents[2]
POLL_SEC = 5.0  # how often a watched run is checked for a finished stage
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
    mesa_dir: pathlib.Path,
    overlay: pathlib.Path,
    env: dict[str, str],
    xtra: tuple[str, ...] = (),
    zones: tuple[list[int], list[int] | None] = ([], None),
    snapshots: bool = False,
) -> None:
    """Creates and compiles the instrumented overlay of ``mesa_dir``.

    Args:
        mesa_dir: Pristine compiled MESA 26.4.1 tree.
        overlay: Absent directory for the overlay.
        env: SDK environment whose ``MESA_DIR`` is ``overlay``.
        xtra: Diagnostic patches to add (members of ``xtra_patch.KINDS``).
        zones: ``(zones, corr_zones)`` for the ``star`` diagnostics.
        snapshots: Also patch ``star/job/run_star_support.f90`` with opt-in
            window snapshot and per-step wall-clock instrumentation.

    Raises:
        SystemExit: If the pristine ``libstar.a`` carries the marker.
    """
    overlay.mkdir(parents=True)
    for entry in mesa_dir.iterdir():
        target = overlay / entry.name
        copied: tuple[str, ...] = ("star", "lib", "include")
        if "binary-rlo" in xtra:
            copied += ("binary",)
        if entry.name in copied:
            shutil.copytree(entry, target, symlinks=True)
        else:
            target.symlink_to(entry)
    solver = overlay / "star" / "private" / "star_solver.f90"
    text = solver.read_text(encoding="utf-8")
    if "star" in xtra:
        text = xtra_patch.patch_star(text, zones[0], zones[1])
    solver.write_text(stepdump_patch.patch_text(text), encoding="utf-8")
    if snapshots:
        rss = overlay / "star" / "job" / "run_star_support.f90"
        rss.write_text(
            snapshot_patch.patch_text(rss.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    modules = ["star"]
    if "binary-rlo" in xtra:
        mdot = overlay / xtra_patch.BINARY_FILE
        mdot.write_text(
            xtra_patch.patch_binary_rlo(mdot.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        modules.append("binary")
    for module in modules:
        for cmd in (["bash", "./mk"], ["bash", "./export"]):
            subprocess.run(cmd, cwd=overlay / module, env=env, check=True)
    pristine = (mesa_dir / "lib" / "libstar.a").read_bytes()
    if stepdump_patch.MARKER.encode() in pristine:
        raise SystemExit("pristine libstar.a carries the step-dump marker")
    if snapshots and snapshot_patch.MARKER.encode() in pristine:
        raise SystemExit("pristine libstar.a carries the snapshot marker")


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


def meta_xtra_lines(meta: dict[str, str]) -> list[str]:
    """Returns the header lines describing the enabled diagnostics."""
    return [f"# xtra: {meta['xtra']}"] if meta.get("xtra") else []


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
        f"# occurrence: {meta.get('occurrence', '1')}",
        f"# stage: {meta.get('stage', 'all')}",
        *meta_xtra_lines(meta),
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


def _run_snapshot_case(
    ident: str,
    env: dict[str, str],
    work: pathlib.Path,
    out: pathlib.Path,
    timeout_sec: int,
    meta: dict[str, str],
) -> str:
    """Runs one ``star/<case>`` uncapped with snapshot & clock logging."""
    suite, case = ident.split("/", 1)
    if suite != "star":
        raise SystemExit(f"{ident}: snapshot mode supports star cases only")
    clock_dest = out / suite / f"{case}.clock.tsv"
    if clock_dest.exists() and not meta.get("overwrite_snapshots"):
        old = clock_dest.read_text(encoding="utf-8", errors="replace")
        if "STAGE_END\t" in old:
            return f"{ident}: kept existing clock log"
    snap_dir = pathlib.Path(meta["snapshot_dir"])
    snap_dir.mkdir(parents=True, exist_ok=True)
    clock_dest.parent.mkdir(parents=True, exist_ok=True)
    spec_path = ""
    if meta.get("snapshot_spec_dir"):
        candidate = pathlib.Path(meta["snapshot_spec_dir"]) / f"{case}.spec"
        if not candidate.is_file():
            raise SystemExit(f"{ident}: missing snapshot spec {candidate}")
        spec_path = str(candidate)
    elif meta.get("snapshot_spec"):
        spec_path = meta["snapshot_spec"]
    case_work = work / f"snap_{case}"
    case_root = _case_root(env, suite, case, case_work)
    if (case_root / case).exists() and meta.get("overwrite_snapshots"):
        shutil.rmtree(case_root / case)
    raw_clock = work / f"{suite}__{case}.clock.tsv"
    raw_clock.unlink(missing_ok=True)
    run_env = {
        **env,
        "OUT": str(case_root),
        snapshot_patch.MARKER: str(raw_clock),
        "FLEXMESA_SNAPSHOT_DIR": str(snap_dir),
        "FLEXMESA_SNAPSHOT_CASE": case,
        "FLEXMESA_SNAPSHOT_SPEC": spec_path,
    }
    status = _run(ident, run_env, timeout_sec)
    if not raw_clock.exists():
        return f"{ident}: {status}, no clock log"
    lines = raw_clock.read_text(encoding="utf-8").splitlines()
    hdr = [
        f"# case: {ident}",
        f"# mesa_version: {meta['mesa_version']}",
        f"# compiler: {meta['compiler']}",
        f"# mesa_sdk_version: {meta['mesa_sdk_version']}",
        f"# omp_num_threads: {OMP_THREADS}",
    ]
    clock_dest.write_text("\n".join(hdr + lines) + "\n", encoding="utf-8")
    n_steps = sum(1 for ln in lines if ln.startswith("STEP\t"))
    n_snaps = sum(1 for ln in lines if ln.startswith("SNAP\t"))
    n_stages = sum(1 for ln in lines if ln.startswith("STAGE_END\t"))
    if meta.get("clean_case_work") and (case_root / case).exists():
        shutil.rmtree(case_root / case)
    return (
        f"{ident}: {status}, stages={n_stages}, steps={n_steps}, "
        f"snaps={n_snaps}"
    )


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
    if meta.get("snapshots_mode"):
        return _run_snapshot_case(ident, env, work, out, timeout_sec, meta)
    suite, case = ident.split("/", 1)
    dest = out / suite / f"{case}.stepdump"
    window = [
        f"# first_model: {meta.get('first_model', 'first of the run')}",
        f"# steps_per_star: {steps}",
        f"# occurrence: {meta.get('occurrence', '1')}",
        f"# stage: {meta.get('stage', 'all')}",
    ]
    if dest.exists():
        old = dest.read_text(encoding="utf-8")
        # A dump from before ``occurrence`` and ``stage`` existed has the
        # default of each.
        old_lines = set(old.splitlines())
        for key, default in (("occurrence", "1"), ("stage", "all")):
            if not any(x.startswith(f"# {key}:") for x in old_lines):
                old_lines.add(f"# {key}: {default}")
        if not all(w in old_lines for w in window):
            raise SystemExit(f"{dest}: exists with a different window")
        if "\nDONE\n" in old:
            return f"{ident}: kept existing complete dump"
    # The window is part of the raw file name, so a run with another window
    # never reuses these records.
    tag = (
        f"s{steps}_f{meta.get('first_model', 'start')}"
        f"_k{meta.get('occurrence', '1')}_st{meta.get('stage', 'all')}"
    )
    raw = work / f"{suite}__{case}__{tag}.raw"
    case_root = _case_root(env, suite, case, work / tag)
    if (case_root / case).exists() and not raw.exists():
        return f"{ident}: case copy exists in scratch, not rerun"
    run_env = {
        **env,
        "OUT": str(case_root),
        stepdump_patch.MARKER: str(raw),
        f"{stepdump_patch.MARKER}_STEPS": str(steps),
    }
    if not meta.get("keep_running"):
        run_env[f"{stepdump_patch.MARKER}_STOP"] = "1"
    xtra_raw = raw.with_suffix(".xtra.raw")
    if meta.get("xtra_binary_models"):
        first, _, last = meta["xtra_binary_models"].partition(":")
        run_env["FLEXMESA_XTRA_BIN"] = str(xtra_raw)
        run_env["FLEXMESA_XTRA_FIRSTMODEL"] = first
        run_env["FLEXMESA_XTRA_STOPMODEL"] = last
        xtra_raw.unlink(missing_ok=True)
    if (case_root / case).exists():
        status = "reused earlier run"
    else:
        # Records are appended, and the patch numbers its processes by the
        # RUN records already in the file, so a rerun starts from none.
        raw.unlink(missing_ok=True)
        stage = int(meta["stage"]) if meta.get("stage") else 0
        watch = stage if not meta.get("keep_running") else 0
        status = _run(ident, run_env, timeout_sec, raw, watch)
    if not raw.exists():
        return f"{ident}: {status}, no dump"
    out = _write_dump(ident, raw, dest, case_root / case, steps, meta, status)
    if meta.get("xtra_binary_models") and xtra_raw.exists():
        header = _header(case_root / case, ident, steps, meta)
        xtra_dest = dest.with_suffix(".xtra")
        xtra_dest.write_text(
            "\n".join(
                header + xtra_raw.read_text(encoding="utf-8").splitlines()
            )
            + "\n",
            encoding="utf-8",
        )
        out += f"; {xtra_dest.name} written"
    return out


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


def _run(
    ident: str,
    run_env: dict[str, str],
    timeout_sec: int,
    raw: pathlib.Path | None = None,
    stage: int = 0,
) -> str:
    """Runs one case through the reference runner; returns its status.

    With ``stage`` and ``raw`` given, the run is watched: the patch writes
    ``DONE`` only when it sees a model after the window, which a stage that
    ends on the window's last model never does. When the selected stage's
    process has ended with the window open (the next stage has written its
    ``RUN`` record, or the whole run has exited), ``DONE`` is appended to
    ``raw`` and, if a later stage started, the run is stopped.
    """
    suite, case = ident.split("/", 1)
    proc = subprocess.Popen(
        ["bash", str(RUN_CASE), suite, case],
        env=run_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout_sec
    status = ""
    while True:
        try:
            proc.wait(timeout=POLL_SEC)
            status = f"rc={proc.returncode}"
            break
        except subprocess.TimeoutExpired:
            pass
        if time.monotonic() > deadline:
            status = "timeout"
            break
        if stage and raw is not None and raw.exists():
            state = stage_window_state(raw.read_text().splitlines(), stage)
            if state == "ended_open":
                status = "stage ended with the window open"
                break
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    if stage and raw is not None and raw.exists() and status != "timeout":
        state = stage_window_state(raw.read_text().splitlines(), stage)
        if state in ("ended_open", "running_open"):
            with raw.open("a", encoding="utf-8") as fh:
                fh.write("DONE\n")
            status += "; DONE appended at the end of the stage"
    return status


def stage_window_state(records: list[str], stage: int) -> str:
    """Classifies the selected stage's section of a raw dump.

    Args:
        records: Raw dump lines; each process starts with ``RUN``.
        stage: 1-based number of the selected process.

    Returns:
        ``"not_started"`` (fewer than ``stage`` processes yet);
        ``"running_closed"`` (``DONE`` written, or the window has not
        opened and the stage is still running); ``"running_open"`` (window
        open, no later stage yet); ``"ended_open"`` (a later process has
        started while the window was open without ``DONE``);
        ``"ended_unopened"`` (a later process started and the window never
        opened).
    """
    runs = [i for i, line in enumerate(records) if line == "RUN"]
    if len(runs) < stage:
        return "not_started"
    end = runs[stage] if len(runs) > stage else len(records)
    section = records[runs[stage - 1] : end]
    opened = any(line.startswith("CALL ") for line in section)
    ended = len(runs) > stage
    if "DONE" in section:
        return "running_closed"
    if not opened:
        return "ended_unopened" if ended else "running_closed"
    return "ended_open" if ended else "running_open"


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
    if meta.get("stage"):
        # The other processes of the case wrote only their RUN line.
        records = _selected_section(records)
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = _header(case_dir, ident, steps, meta)
    dest.write_text("\n".join(header + records) + "\n", encoding="utf-8")
    done = "DONE" in records
    return f"{ident}: {status}, {len(records)} records, done={done}"


def _selected_section(records: list[str]) -> list[str]:
    """Returns ``records`` without the ``RUN`` lines that have no records.

    Args:
        records: Raw dump lines; each process starts with ``RUN``.

    Returns:
        The lines of the processes that wrote records.
    """
    out: list[str] = []
    for i, line in enumerate(records):
        if line == "RUN" and (i + 1 == len(records) or records[i + 1] == "RUN"):
            continue
        out.append(line)
    return out


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
    parser.add_argument(
        "--occurrence",
        type=int,
        default=1,
        help="With --first-model: open the window the K-th time the model "
        "number rises to it (model numbers restart between a relax phase "
        "and the evolve phase). Default 1.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Dump only the S-th process of the case (1-based RUN section "
        "of a multi-inlist rn). Default every stage.",
    )
    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Do not stop the run after the window is dumped.",
    )
    parser.add_argument(
        "--xtra",
        action="append",
        choices=xtra_patch.KINDS,
        default=[],
        help="Opt-in diagnostics (see xtra_patch.py); repeatable. The "
        "overlay under --scratch must be built with the same choices.",
    )
    parser.add_argument(
        "--xtra-zones",
        default="",
        help="Zones (1,262,474:481) of the per-zone XTRA star values at each "
        "residual evaluation, and of the correction values unless "
        "--xtra-corr-zones is given.",
    )
    parser.add_argument("--xtra-corr-zones", default=None)
    parser.add_argument(
        "--xtra-binary-models",
        default=None,
        metavar="FIRST:LAST",
        help="Binary models written by --xtra binary-rlo (the run stops "
        "after LAST).",
    )
    parser.add_argument(
        "--snapshots",
        action="store_true",
        help="Build/run opt-in W1 window snapshot and per-step clock logging.",
    )
    parser.add_argument(
        "--snapshot-spec",
        type=pathlib.Path,
        default=None,
        help="Single <stage> <model> spec file for --snapshots.",
    )
    parser.add_argument(
        "--snapshot-spec-dir",
        type=pathlib.Path,
        default=None,
        help="Directory holding <case>.spec files for --snapshots.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for <case>_<stage>_<model>.mod snapshots.",
    )
    parser.add_argument(
        "--overwrite-snapshots",
        action="store_true",
        help="Re-run even when <case>.clock.tsv already has STAGE_END.",
    )
    parser.add_argument(
        "--clean-case-work",
        action="store_true",
        help="Remove scratch case directory after extracting clock/snapshots.",
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
    zones = xtra_patch.parse_zones(args.xtra_zones)
    corr_zones = (
        None
        if args.xtra_corr_zones is None
        else xtra_patch.parse_zones(args.xtra_corr_zones)
    )
    if "binary-rlo" in args.xtra and not args.xtra_binary_models:
        raise SystemExit("--xtra binary-rlo needs --xtra-binary-models")
    snap_mode = bool(
        args.snapshots or args.snapshot_spec or args.snapshot_spec_dir
    )
    if (args.snapshot_spec or args.snapshot_spec_dir) and not args.snapshot_dir:
        raise SystemExit("--snapshot-spec* requires --snapshot-dir")
    env = sdk_env(args.mesasdk_root.resolve(), overlay)
    if not args.reuse_overlay:
        if scratch.exists() and any(scratch.iterdir()):
            raise SystemExit("--scratch must be empty or absent")
        build_overlay(
            mesa_dir,
            overlay,
            env,
            tuple(args.xtra),
            (zones, corr_zones),
            snapshots=args.snapshots,
        )
    meta = build_meta(mesa_dir, env)
    if snap_mode:
        meta["snapshots_mode"] = "1"
        if args.snapshot_dir is not None:
            meta["snapshot_dir"] = str(args.snapshot_dir.resolve())
        else:
            meta["snapshot_dir"] = str(args.out.resolve() / "snapshots")
        if args.snapshot_spec is not None:
            meta["snapshot_spec"] = str(args.snapshot_spec.resolve())
        if args.snapshot_spec_dir is not None:
            meta["snapshot_spec_dir"] = str(args.snapshot_spec_dir.resolve())
        if args.overwrite_snapshots:
            meta["overwrite_snapshots"] = "1"
        if args.clean_case_work:
            meta["clean_case_work"] = "1"
    if args.occurrence < 1 or (args.stage is not None and args.stage < 1):
        raise SystemExit("--occurrence and --stage are 1-based")
    if args.occurrence > 1 and args.first_model is None:
        raise SystemExit("--occurrence needs --first-model")
    if args.first_model is not None:
        env[f"{stepdump_patch.MARKER}_FIRST"] = str(args.first_model)
        meta["first_model"] = str(args.first_model)
    env[f"{stepdump_patch.MARKER}_OCCURRENCE"] = str(args.occurrence)
    meta["occurrence"] = str(args.occurrence)
    if args.stage is not None:
        env[f"{stepdump_patch.MARKER}_STAGE"] = str(args.stage)
        meta["stage"] = str(args.stage)
    if args.xtra:
        meta["xtra"] = ", ".join(args.xtra)
    if "binary-rlo" in args.xtra:
        meta["xtra_binary_models"] = args.xtra_binary_models
        meta["keep_running"] = "1"
    if args.keep_running:
        meta["keep_running"] = "1"
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
