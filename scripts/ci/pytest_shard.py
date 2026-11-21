#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Split the Python test files into balanced shards for parallel CI jobs.

Every test file under ``tests/`` that pytest would collect is assigned to
exactly one shard, so running all shards runs the same tests as one
``pytest src tests`` call with the same ``--ignore`` rules. The ``src`` tree
(doctest modules) always goes to shard 1.

Files are placed largest first onto the least-loaded shard, using the
per-file wall times in ``scripts/ci/pytest_durations.json`` (seconds, taken
from a CI log). A file missing from that table gets ``DEFAULT_SECONDS``, so a
new test file is still run; the table only affects balance, never coverage.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DURATIONS_PATH = pathlib.Path(__file__).with_name("pytest_durations.json")
DEFAULT_SECONDS = 10.0
# Files run by other CI jobs (`ci.yml`: docs, history-harness and e2e jobs).
IGNORED_GLOBS = (
    "tests/test_docs_snippets.py",
    "tests/test_mesa_paper_figures.py",
    "tests/test_history_harness.py",
    "tests/test_e2e_*.py",
)


def discover_test_files(repo_root: pathlib.Path) -> list[str]:
    """Return repo-relative test files pytest collects, minus other jobs' files.

    Args:
        repo_root: Repository root directory.

    Returns:
        Sorted POSIX paths matching pytest's default ``test_*.py`` and
        ``*_test.py`` patterns under ``tests/``.
    """
    tests_dir = repo_root / "tests"
    found: set[str] = set()
    for pattern in ("test_*.py", "*_test.py"):
        for path in tests_dir.rglob(pattern):
            found.add(path.relative_to(repo_root).as_posix())
    return sorted(
        f
        for f in found
        if not any(fnmatch.fnmatch(f, glob) for glob in IGNORED_GLOBS)
    )


def assign_shards(
    files: list[str],
    durations: dict[str, float],
    total: int,
) -> list[list[str]]:
    """Assign each file to one of ``total`` shards, balancing wall time.

    Args:
        files: Test files to distribute.
        durations: Seconds per file; missing files use ``DEFAULT_SECONDS``.
        total: Number of shards (``>= 1``).

    Returns:
        ``total`` lists of files; every input file appears in exactly one.

    Raises:
        ValueError: If ``total < 1``.
    """
    if total < 1:
        raise ValueError(f"total shards must be >= 1, got {total}")
    shards: list[list[str]] = [[] for _ in range(total)]
    loads = [0.0] * total
    ordered = sorted(
        files,
        key=lambda f: (-durations.get(f, DEFAULT_SECONDS), f),
    )
    for name in ordered:
        k = min(range(total), key=lambda i: (loads[i], i))
        shards[k].append(name)
        loads[k] += durations.get(name, DEFAULT_SECONDS)
    return [sorted(s) for s in shards]


def shard_targets(index: int, total: int) -> list[str]:
    """Return pytest path arguments for shard ``index`` of ``total``.

    Args:
        index: One-based shard index.
        total: Number of shards.

    Returns:
        Path arguments; shard 1 also carries ``src`` for doctest modules.

    Raises:
        ValueError: If ``index`` is outside ``1..=total``.
    """
    if not 1 <= index <= total:
        raise ValueError(f"shard index {index} outside 1..={total}")
    durations: dict[str, float] = {
        str(k): float(v)
        for k, v in json.loads(
            DURATIONS_PATH.read_text(encoding="utf-8")
        ).items()
    }
    shards = assign_shards(discover_test_files(REPO_ROOT), durations, total)
    targets = shards[index - 1]
    return ["src", *targets] if index == 1 else targets


def main() -> int:
    """Print the pytest path arguments for one shard, space separated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--total", type=int, required=True)
    args = parser.parse_args()
    print(" ".join(shard_targets(args.index, args.total)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
