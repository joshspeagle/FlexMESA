#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run one shard of a crate's `cargo test` targets in CI.

A crate's tests are its library unit tests, each integration-test binary
under ``crates/<crate>/tests/``, and its doctests. This script assigns every
one of those to exactly one of ``--total`` shards and runs the shard given by
``--index``, so running all shards runs what ``cargo test -p <crate>`` runs.

Integration-test binaries named in ``SPLIT_BY_TEST`` are too long to treat as
one unit. For those the script asks the built binary for its test names
(``-- --list``) and spreads the individual tests over the shards, running
them with ``--exact``.

Units are placed largest first onto the least-loaded shard, using the wall
times in ``scripts/ci/cargo_test_durations.json`` (seconds, from a CI log).
A target missing from that table gets ``DEFAULT_SECONDS``, so a new test file
is still run; the table only affects balance, never coverage. The table may
also give a time for one test of a split binary, under the key
``<binary>::<test name>``; other tests of that binary share its time equally.

``--skip NAME`` (repeatable) is passed to every test harness invocation as
``-- --skip NAME``. The dashboard's coverage job uses it to leave out long
evolution tests under instrumentation; CI's own test shards do not pass it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DURATIONS_PATH = pathlib.Path(__file__).with_name("cargo_test_durations.json")
DEFAULT_SECONDS = 15.0
LIB = "lib"
DOC = "doc"
# (package, integration-test binary) pairs split into individual tests.
SPLIT_BY_TEST = frozenset(
    {
        ("flexmesa-astero", "astero_suite"),
        ("flexmesa-binary", "tier5_coupled_suite"),
    }
)


def discover_targets(package: str, repo_root: pathlib.Path) -> list[str]:
    """Return ``lib``, ``doc`` and every integration-test binary name."""
    tests_dir = repo_root / "crates" / package / "tests"
    names = sorted(p.stem for p in tests_dir.glob("*.rs"))
    return [LIB, DOC, *names]


def assign(weights: dict[str, float], total: int) -> list[list[str]]:
    """Partition unit names into ``total`` shards, heaviest first.

    Args:
        weights: Unit name to estimated seconds.
        total: Number of shards (at least 1).

    Returns:
        ``total`` lists of unit names; every key of ``weights`` appears in
        exactly one list. Ties are broken by name, so the result does not
        depend on dictionary order.
    """
    if total < 1:
        raise ValueError("total must be at least 1")
    shards: list[list[str]] = [[] for _ in range(total)]
    loads = [0.0] * total
    for name in sorted(weights, key=lambda n: (-weights[n], n)):
        i = min(range(total), key=lambda j: (loads[j], j))
        shards[i].append(name)
        loads[i] += weights[name]
    return shards


def plan(
    package: str,
    targets: list[str],
    split_tests: dict[str, list[str]],
    durations: dict[str, float],
    total: int,
) -> list[list[str]]:
    """Assign targets, and the tests of split binaries, to shards.

    Args:
        package: Cargo package name.
        targets: Output of `discover_targets`.
        split_tests: For each split binary, its test names.
        durations: Seconds per target for this package.
        total: Number of shards.

    Returns:
        Shards of unit names. A unit is a target name, or
        ``<binary>::<test name>`` for a test of a split binary.
    """
    weights: dict[str, float] = {}
    for target in targets:
        seconds = float(durations.get(target, DEFAULT_SECONDS))
        if (package, target) in SPLIT_BY_TEST:
            tests = split_tests[target]
            for test in tests:
                unit = f"{target}::{test}"
                weights[unit] = float(
                    durations.get(unit, seconds / max(len(tests), 1))
                )
        else:
            weights[target] = seconds
    return assign(weights, total)


def _cargo(args: list[str], skips: list[str] | None = None) -> None:
    harness = [arg for name in (skips or []) for arg in ("--skip", name)]
    if harness:
        args = [*args, *([] if "--" in args else ["--"]), *harness]
    cmd = ["cargo", "test", *args]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _list_tests(package: str, profile: list[str], target: str) -> list[str]:
    out = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            package,
            *profile,
            "--test",
            target,
            "--",
            "--list",
            "--format",
            "terse",
        ],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout
    names = [
        line.rsplit(": ", 1)[0]
        for line in out.splitlines()
        if line.endswith(": test")
    ]
    if not names:
        raise SystemExit(f"no tests listed for {package} --test {target}")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--index", type=int, required=True, help="1-based")
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--profile", default="ci")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="NAME",
        help="Test name filter passed to the harness as --skip (repeatable).",
    )
    ns = parser.parse_args(argv)
    if not 1 <= ns.index <= ns.total:
        parser.error("--index must be between 1 and --total")

    profile = ["--profile", ns.profile]
    targets = discover_targets(ns.package, REPO_ROOT)
    durations = json.loads(DURATIONS_PATH.read_text(encoding="utf-8")).get(
        ns.package, {}
    )
    # Build everything once, then list the tests of the split binaries.
    _cargo(["-p", ns.package, *profile, "--no-run"])
    split_tests = {
        t: _list_tests(ns.package, profile, t)
        for t in targets
        if (ns.package, t) in SPLIT_BY_TEST
    }
    mine = plan(ns.package, targets, split_tests, durations, ns.total)[
        ns.index - 1
    ]
    print(f"shard {ns.index}/{ns.total}: {len(mine)} unit(s)", flush=True)

    whole: list[str] = []
    for unit in mine:
        if unit == LIB:
            whole.append("--lib")
        elif unit != DOC and "::" not in unit:
            whole.extend(["--test", unit])
    if whole:
        _cargo(["-p", ns.package, *profile, *whole], ns.skip)
    for target in split_tests:
        prefix = f"{target}::"
        names = [u[len(prefix) :] for u in mine if u.startswith(prefix)]
        if names:
            _cargo(
                [
                    "-p",
                    ns.package,
                    *profile,
                    "--test",
                    target,
                    "--",
                    "--exact",
                    *names,
                ],
                ns.skip,
            )
    if DOC in mine:
        _cargo(["-p", ns.package, *profile, "--doc"], ns.skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
