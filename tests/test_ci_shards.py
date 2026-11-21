# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checks that the CI Python shards together run every test file once."""

from __future__ import annotations

import importlib.util
import pathlib
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_shard_module() -> types.ModuleType:
    path = REPO_ROOT / "scripts" / "ci" / "pytest_shard.py"
    spec = importlib.util.spec_from_file_location("pytest_shard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_python_shards_partition_every_collected_test_file() -> None:
    """Every test file lands in exactly one shard for any shard count."""
    shard = _load_shard_module()
    files = shard.discover_test_files(REPO_ROOT)
    assert "tests/test_ci_shards.py" in files
    assert "tests/test_history_harness.py" not in files
    assert not [f for f in files if f.startswith("tests/test_e2e_")]
    on_disk = {
        p.relative_to(REPO_ROOT).as_posix()
        for pattern in ("test_*.py", "*_test.py")
        for p in (REPO_ROOT / "tests").rglob(pattern)
    }
    ignored = {
        f
        for f in on_disk
        if f
        in {
            "tests/test_docs_snippets.py",
            "tests/test_mesa_paper_figures.py",
            "tests/test_history_harness.py",
        }
        or f.startswith("tests/test_e2e_")
    }
    assert set(files) == on_disk - ignored

    for total in (1, 2, 3, 4, 7):
        seen: list[str] = []
        for index in range(1, total + 1):
            targets = shard.shard_targets(index, total)
            assert ("src" in targets) == (index == 1)
            seen.extend(t for t in targets if t != "src")
        assert sorted(seen) == files
        assert len(seen) == len(set(seen))


def test_ci_python_shards_place_unknown_files_and_balance_known_ones() -> None:
    """A file absent from the durations table is still assigned."""
    shard = _load_shard_module()
    durations = {"tests/test_a.py": 300.0, "tests/test_b.py": 100.0}
    files = [
        "tests/test_a.py",
        "tests/test_b.py",
        "tests/test_new.py",
        "tests/test_other.py",
    ]
    parts = shard.assign_shards(files, durations, 2)
    assert sorted(parts[0] + parts[1]) == files
    assert parts[0] == ["tests/test_a.py"]
