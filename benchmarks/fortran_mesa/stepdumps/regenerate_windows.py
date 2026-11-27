#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regenerates the mid-run step-dump windows listed in ``windows.toml`` (#608).

Runs ``run_stepdumps.py`` once per window, one case at a time, writing
``<out>/<window id>/<suite>/<case>.stepdump``. The window id is
``<case>_s<stage>_k<occurrence>_m<first model>_n<steps>``. A window whose
output already exists and carries its ``DONE`` record is skipped, so an
interrupted run resumes where it stopped. The overlay is built under
``--scratch`` on first use and reused afterwards.

Usage::

    python regenerate_windows.py --mesa-dir "$MESA_DIR" \\
        --mesasdk-root "$MESASDK_ROOT" --scratch <dir> --out <dir> \\
        [--only star/ns_h ...] [--dry-run] [--timeout-sec 3600]

``--dry-run`` prints the ``run_stepdumps.py`` command of every window and runs
nothing. Each real run is MESA time (minutes to an hour per window), so wrap
the script in whatever compute reservation the machine uses. Linking may need
``LIBRARY_PATH`` to point at a ``libz`` the SDK can use.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tomllib
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
WINDOWS = HERE / "windows.toml"
RUNNER = HERE / "run_stepdumps.py"


def window_id(w: dict[str, Any]) -> str:
    """Returns the id of a window entry.

    Args:
        w: One ``[[window]]`` table.

    Returns:
        ``<case>_s<stage>_k<occurrence>_m<first model>_n<steps>``.
    """
    case = str(w["case"]).split("/", 1)[1]
    return (
        f"{case}_s{w['stage']}_k{w['occurrence']}"
        f"_m{w['first_model']}_n{w['steps']}"
    )


def load_windows(path: pathlib.Path = WINDOWS) -> list[dict[str, Any]]:
    """Returns the ``[[window]]`` tables of ``windows.toml``.

    Args:
        path: The manifest.

    Returns:
        The windows in file order.

    Raises:
        ValueError: If a window lacks a key or two windows share an id.
    """
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    keys = ("case", "stage", "occurrence", "first_model", "steps", "why")
    seen: set[str] = set()
    for w in doc["window"]:
        missing = [k for k in keys if k not in w]
        if missing:
            raise ValueError(f"window {w.get('case')}: missing {missing}")
        ident = window_id(w)
        if ident in seen:
            raise ValueError(f"duplicate window {ident}")
        seen.add(ident)
    return list(doc["window"])


def command(
    w: dict[str, Any],
    mesa_dir: pathlib.Path,
    mesasdk_root: pathlib.Path,
    scratch: pathlib.Path,
    out: pathlib.Path,
    timeout_sec: int,
) -> list[str]:
    """Returns the ``run_stepdumps.py`` command line of one window."""
    cmd = [
        sys.executable,
        str(RUNNER),
        "--mesa-dir",
        str(mesa_dir),
        "--mesasdk-root",
        str(mesasdk_root),
        "--scratch",
        str(scratch),
        "--out",
        str(out / window_id(w)),
        "--steps",
        str(w["steps"]),
        "--first-model",
        str(w["first_model"]),
        "--occurrence",
        str(w["occurrence"]),
        "--stage",
        str(w["stage"]),
        "--timeout-sec",
        str(timeout_sec),
        "--jobs",
        "1",
        "--cases",
        str(w["case"]),
    ]
    if (scratch / "mesa_overlay").exists():
        cmd.append("--reuse-overlay")
    return cmd


def is_done(w: dict[str, Any], out: pathlib.Path) -> bool:
    """Returns whether the window's dump exists and ends its window."""
    suite, case = str(w["case"]).split("/", 1)
    dump = out / window_id(w) / suite / f"{case}.stepdump"
    return dump.exists() and "\nDONE\n" in dump.read_text(encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesa-dir", type=pathlib.Path, required=True)
    parser.add_argument("--mesasdk-root", type=pathlib.Path, required=True)
    parser.add_argument("--scratch", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Cases (suite/case) or window ids to run; default every window.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    windows = load_windows()
    if args.only:
        wanted = set(args.only)
        windows = [
            w for w in windows if w["case"] in wanted or window_id(w) in wanted
        ]
        if not windows:
            raise SystemExit(f"no window matches {sorted(wanted)}")
    failed = 0
    for w in windows:
        ident = window_id(w)
        if is_done(w, args.out):
            print(f"{ident}: done, skipped", flush=True)
            continue
        cmd = command(
            w,
            args.mesa_dir,
            args.mesasdk_root,
            args.scratch,
            args.out,
            args.timeout_sec,
        )
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        print(f"{ident}: running", flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0 or not is_done(w, args.out):
            failed += 1
            print(f"{ident}: not complete (rc={proc.returncode})", flush=True)
    if failed:
        raise SystemExit(f"{failed} window(s) not complete")


if __name__ == "__main__":
    main()
