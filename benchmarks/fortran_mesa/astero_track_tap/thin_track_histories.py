#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thin the per-track MESA 26.4.1 histories of an astero search case (#430).

``run_mesa_case_26_4_1.sh`` run with ``HISTORY_TAP_OBJ`` set (the compiled
``history_tap_wrap.c``) leaves ``history_tap/track_NNNN.data`` in the case
work directory, one MESA ``LOGS/history.data`` per search track. This script
thins each track with the same column selection and 6-line MESA header as
the stored references (``scripts/data/build_mesa_histories.py``) and writes
``<out_dir>/track_NNNN.data``.

With ``--last-reference`` it also checks that the thinned last track is
byte-identical to the stored thinned reference of the whole run, below its
header (whose ``date`` field records the day of the run), which shows that
the tap reproduced the reference run.

Usage:
    python thin_track_histories.py <case_work_dir> <out_dir> \
        [--last-reference <thinned reference .data>]
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import re
import sys

_TRACK_RE = re.compile(r"^track_(\d{4})\.data$")
_HEADER_LINES = 3


def _load_thinner() -> object:
    """Import the bundle builder's history thinning function."""
    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "scripts"
        / "data"
        / "build_mesa_histories.py"
    )
    spec = importlib.util.spec_from_file_location("build_mesa_histories", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.thin_history_preserving_mesa_header


def main() -> int:
    """Thin every tapped track and optionally check the last one.

    Returns:
        Process exit code (``0`` on success, ``1`` on a failed check).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case_work_dir", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--last-reference", type=pathlib.Path, default=None)
    args = parser.parse_args()

    tap_dir = args.case_work_dir / "history_tap"
    tracks = {
        int(m.group(1)): p
        for p in tap_dir.iterdir()
        if (m := _TRACK_RE.match(p.name)) is not None
    }
    numbers = sorted(tracks)
    if not numbers or numbers != list(range(1, len(numbers) + 1)):
        print(f"track files not numbered 1..N: {numbers}", file=sys.stderr)
        return 1
    thin = _load_thinner()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    last_bytes = b""
    for number in numbers:
        last_bytes = thin(tracks[number], "astero")  # type: ignore[operator]
        dst = args.out_dir / f"track_{number:04d}.data"
        dst.write_bytes(last_bytes)
        n_rows = len(last_bytes.splitlines()) - 6
        print(f"{dst.name}: {n_rows} rows")
    if args.last_reference is not None:
        ref = args.last_reference.read_bytes().split(b"\n")[_HEADER_LINES:]
        got = last_bytes.split(b"\n")[_HEADER_LINES:]
        if ref != got:
            print(
                f"last track differs from {args.last_reference.name}",
                file=sys.stderr,
            )
            return 1
        print(f"last track matches {args.last_reference.name} below header")
    return 0


if __name__ == "__main__":
    sys.exit(main())
