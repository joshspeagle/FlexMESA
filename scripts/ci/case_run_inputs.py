#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate the inputs of the ``case-run`` workflow and resolve step dumps.

The workflow's free-text inputs reach this script only through environment
variables or arguments, never through shell interpolation. Two subcommands:

``plan``
    Reads ``CASES``, ``MODE``, ``WINDOW``, ``TIMEOUT_SEC`` and
    ``MAX_RSS_GB`` from the environment, validates them against the case
    manifest, and appends the job matrix and the per-job timeout to the file
    named by ``--github-output``.

``dump``
    Prints the path of the MESA step dump that ``flexmesa trace --compare``
    should read for one case, from the ``mesa_stepdumps`` bundle or, with a
    window, from the ``mesa_stepdump_windows`` bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
import tomllib
from collections.abc import Iterable, Mapping

MAX_CASES = 16
MODES = ("harness", "trace", "compare")
# Time added to the case budget for the build artifact download, the data
# fetch and the post-step uploads (CLAUDE.md: at least 45 minutes headroom).
HEADROOM_MIN = 45
MAX_TIMEOUT_SEC = 18000
MAX_RSS_LIMIT_GB = 14.0
WINDOWS_BUNDLE = "mesa_stepdump_windows"
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
_WINDOW_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
# <case>_s<stage>_k<occurrence>_m<first model>_n<models>; the case name may
# itself contain underscores, so the four trailing fields anchor the match.
_WINDOW_FIELDS_RE = re.compile(r"_s(\d+)_k(\d+)_m(\d+)_n(\d+)")


class InputError(ValueError):
    """Raised when a workflow input is invalid."""


def manifest_keys(root: pathlib.Path) -> set[tuple[str, str]]:
    """Returns the ``(suite, case)`` pairs of the case manifest.

    Args:
        root: Repository root.
    """
    path = root / "benchmarks" / "history_harness" / "case_manifest.toml"
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return {(str(c["suite"]), str(c["case"])) for c in data["cases"]}


def declared_bundles(root: pathlib.Path) -> set[str]:
    """Returns the bundle names declared in ``bundles.toml``.

    Args:
        root: Repository root.
    """
    path = root / "src" / "flexmesa" / "data" / "bundles.toml"
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return {str(b["name"]) for b in data["bundles"]}


def parse_cases(
    raw: str, known: Iterable[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Validates a space-separated ``suite/case`` list.

    Args:
        raw: The ``cases`` input.
        known: ``(suite, case)`` pairs of the case manifest.

    Returns:
        The cases in input order, without duplicates.

    Raises:
        InputError: On an empty list, more than ``MAX_CASES`` cases, a
            malformed token or a case absent from the manifest.
    """
    known_set = set(known)
    tokens = raw.split()
    if not tokens:
        raise InputError("cases is empty")
    out: list[tuple[str, str]] = []
    for tok in tokens:
        suite, sep, case = tok.partition("/")
        if (
            not sep
            or not _NAME_RE.fullmatch(suite)
            or not _NAME_RE.fullmatch(case)
        ):
            raise InputError(f"not a suite/case token: {tok!r}")
        if (suite, case) not in known_set:
            raise InputError(f"{tok!r} is not in the case manifest")
        if (suite, case) not in out:
            out.append((suite, case))
    if len(out) > MAX_CASES:
        raise InputError(f"{len(out)} cases requested, at most {MAX_CASES}")
    return out


def plan(env: Mapping[str, str], root: pathlib.Path) -> dict[str, str]:
    """Validates all inputs and builds the workflow outputs.

    Args:
        env: Mapping with ``CASES``, ``MODE``, ``WINDOW``, ``TIMEOUT_SEC``
            and ``MAX_RSS_GB``.
        root: Repository root of the checked-out ref.

    Returns:
        Output name to string value: ``matrix`` (JSON), ``job_minutes``.

    Raises:
        InputError: On an invalid input.
    """
    cases = parse_cases(env.get("CASES", ""), manifest_keys(root))
    mode = env.get("MODE", "")
    if mode not in MODES:
        raise InputError(f"mode must be one of {MODES}, got {mode!r}")
    window = env.get("WINDOW", "").strip()
    if window:
        if mode != "compare":
            raise InputError("window applies to mode=compare only")
        if not _WINDOW_RE.fullmatch(window):
            raise InputError(f"bad window id: {window!r}")
        if WINDOWS_BUNDLE not in declared_bundles(root):
            raise InputError(
                f"window given, but the {WINDOWS_BUNDLE} bundle is not "
                "declared on this ref"
            )
    try:
        timeout = int(env.get("TIMEOUT_SEC", ""))
        rss = float(env.get("MAX_RSS_GB", ""))
    except ValueError as exc:
        raise InputError(f"bad number: {exc}") from exc
    if not 1 <= timeout <= MAX_TIMEOUT_SEC:
        raise InputError(f"timeout_sec must be in 1..{MAX_TIMEOUT_SEC}")
    if not (math.isfinite(rss) and 0.5 <= rss <= MAX_RSS_LIMIT_GB):
        raise InputError(f"max_rss_gb must be in 0.5..{MAX_RSS_LIMIT_GB}")
    matrix = {
        "include": [{"suite": s, "case": c, "id": f"{s}-{c}"} for s, c in cases]
    }
    return {
        "matrix": json.dumps(matrix, separators=(",", ":")),
        "job_minutes": str(math.ceil(timeout / 60) + HEADROOM_MIN),
    }


def resolve_dump(
    mesa_dir: pathlib.Path,
    suite: str,
    case: str,
    window: str,
    known: Iterable[tuple[str, str]],
) -> pathlib.Path:
    """Returns the MESA step dump to compare one case with.

    Args:
        mesa_dir: Directory holding the unpacked test bundles.
        suite: ``star``, ``binary`` or ``astero``.
        case: Case name.
        window: Window id, or empty for the first window of the run.
        known: ``(suite, case)`` pairs of the case manifest.

    Raises:
        InputError: If the case is unknown or no dump exists for it.
    """
    if (suite, case) not in set(known):
        raise InputError(f"{suite}/{case} is not in the case manifest")
    if window:
        if not _WINDOW_RE.fullmatch(window):
            raise InputError(f"bad window id: {window!r}")
        path = mesa_dir / WINDOWS_BUNDLE / window / suite / f"{case}.stepdump"
    else:
        path = mesa_dir / "mesa_stepdumps" / suite / f"{case}.stepdump"
    if not path.is_file():
        raise InputError(f"no MESA step dump at {path}")
    return path


def window_trace_args(window: str) -> list[str]:
    """Returns the ``flexmesa trace`` options that select a window.

    Args:
        window: Window id ``<case>_s<S>_k<K>_m<M>_n<N>`` (stage, occurrence,
            first model, models), or empty for the first window of the run.

    Returns:
        ``--stage S`` when ``S > 1`` and ``--occurrence K`` when ``K > 1``;
        empty for the default window.

    Raises:
        InputError: If a non-empty id has no such fields.
    """
    if not window:
        return []
    found = _WINDOW_FIELDS_RE.search(window)
    if found is None or found.end() != len(window):
        raise InputError(
            f"window id has no _s<S>_k<K>_m<M>_n<N> fields: {window!r}"
        )
    stage, occ = int(found.group(1)), int(found.group(2))
    args: list[str] = []
    if stage > 1:
        args += ["--stage", str(stage)]
    if occ > 1:
        args += ["--occurrence", str(occ)]
    return args


def main(argv: list[str] | None = None) -> int:
    """Runs the ``plan`` or ``dump`` subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--github-output", type=pathlib.Path, required=True)
    p_dump = sub.add_parser("dump")
    p_dump.add_argument("--suite", required=True)
    p_dump.add_argument("--case", required=True)
    p_dump.add_argument("--window", default="")
    p_dump.add_argument("--mesa-dir", type=pathlib.Path, required=True)
    p_args = sub.add_parser("window-args")
    p_args.add_argument("--window", default="")
    ns = parser.parse_args(argv)
    root = pathlib.Path.cwd()
    try:
        if ns.cmd == "window-args":
            print(" ".join(window_trace_args(ns.window)))
        elif ns.cmd == "plan":
            outputs = plan(os.environ, root)
            with ns.github_output.open("a", encoding="utf-8") as fh:
                for key, val in outputs.items():
                    fh.write(f"{key}={val}\n")
        else:
            path = resolve_dump(
                ns.mesa_dir,
                ns.suite,
                ns.case,
                ns.window,
                manifest_keys(root),
            )
            print(path)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
