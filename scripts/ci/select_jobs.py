#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decide which CI job groups a pull request needs from its changed paths.

The rule is conservative. Every job group runs unless every changed file
falls into one of three narrow classes whose effect is known:

- documentation: ``docs/``, ``notes/``, top-level ``*.md``, ``mkdocs.yml``,
  and the rule files under ``.claude/`` and ``.agents/``;
- Python tests: ``*.py`` files under ``tests/`` (not ``tests/fixtures/``,
  which Rust tests also read);
- Rust tests: files under ``crates/<crate>/tests/``.

Rust library source (``crates/<crate>/src/``, except the PyO3 bridge) runs
the Rust and engine groups but not the Python suites (DECISIONS #81). A
change to anything else (Python library code, the bridge, build files,
``benchmarks/``, ``scripts/``, workflows, fixtures) runs every group, and so
does every pull request from a ``train-*`` branch. A push to ``main`` and a
manual run also run every group, so a job skipped on a pull request still
runs on ``main`` after the merge.

The ``docs`` job (lint, type checks, generated-page checks, the known-gaps
gate and the documentation build) always runs and is not listed here.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys

GROUPS = ("rust", "python", "engine_runs")

_DOC_GLOBS = (
    "docs/*",
    "notes/*",
    "*.md",
    "mkdocs.yml",
    ".claude/*",
    ".agents/*",
    "LICENSE",
    "CITATION.cff",
)


def classify(path: str) -> str:
    """Return ``docs``, ``py_tests``, ``rust_tests``, ``rust_src`` or ``other``.

    ``rust_src`` is Rust library source outside the PyO3 bridge
    (``crates/<crate>/src/**.rs`` for every crate but ``flexmesa-py``).
    """
    parts = path.split("/")
    if (
        len(parts) >= 4
        and parts[0] == "crates"
        and parts[2] == "tests"
        and path.endswith(".rs")
    ):
        return "rust_tests"
    if (
        len(parts) >= 4
        and parts[0] == "crates"
        and parts[1] != "flexmesa-py"
        and parts[2] == "src"
        and path.endswith(".rs")
    ):
        return "rust_src"
    if (
        parts[0] == "tests"
        and path.endswith(".py")
        and not path.startswith("tests/fixtures/")
    ):
        return "py_tests"
    if "/" not in path and path.endswith(".md"):
        return "docs"
    if any(fnmatch.fnmatch(path, g) for g in _DOC_GLOBS if g != "*.md"):
        return "docs"
    return "other"


def select(
    paths: list[str], event: str, train: bool = False
) -> dict[str, bool]:
    """Return which job groups must run.

    Args:
        paths: Changed files, relative to the repository root.
        event: The GitHub event name (``pull_request``, ``push``, ...).

    Returns:
        A mapping from each name in ``GROUPS`` to whether it runs.
        ``rust`` is Clippy, rustdoc and the Rust test shards; ``python`` is
        the Python shards (which include the end-to-end files) and the
        harness unit suite;
        ``engine_runs`` is the history-harness subset and the two
        strict-determinism jobs.
    """
    everything = dict.fromkeys(GROUPS, True)
    if event != "pull_request" or train or not paths:
        return everything
    classes = {classify(p) for p in paths}
    if "other" in classes:
        return everything
    # DECISIONS #81: a pull request that changes only Rust library source
    # (no Python, bridge, build or data files) skips the Python suites. They
    # run on every train pull request (T1) and every push to main, where a
    # Rust change that breaks a Python test is caught before and after merge.
    return {
        "rust": bool(classes & {"rust_tests", "rust_src"}),
        "python": "py_tests" in classes,
        "engine_runs": "rust_src" in classes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument(
        "--head-ref",
        default="",
        help="Pull request head branch; a 'train-*' branch runs every group.",
    )
    parser.add_argument(
        "--paths-file",
        required=True,
        help="File with one changed path per line ('-' for standard input).",
    )
    ns = parser.parse_args(argv)
    if ns.paths_file == "-":
        text = sys.stdin.read()
    else:
        with open(ns.paths_file, encoding="utf-8") as fh:
            text = fh.read()
    paths = [ln.strip() for ln in text.splitlines() if ln.strip()]
    train = ns.head_ref.startswith("train-")
    for group, run in select(paths, ns.event, train).items():
        print(f"{group}={'true' if run else 'false'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
