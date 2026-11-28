# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the generated per-case wall-time budgets (#640)."""

from __future__ import annotations

import importlib.util
import pathlib
import types

import pytest

from flexmesa.testing import case_budgets, case_manifest, harness_shards

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TABLE = REPO_ROOT / "benchmarks" / "history_harness" / "case_budgets.toml"


def _load_script() -> types.ModuleType:
    path = REPO_ROOT / "scripts" / "audit" / "case_budgets.py"
    spec = importlib.util.spec_from_file_location("case_budgets_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_table_covers_every_manifest_case_within_one_job() -> None:
    """Each manifest case has a budget in [floor, one hosted job]."""
    table = case_budgets.load_case_budgets(TABLE)
    keys = {e.key for e in case_manifest.load_case_manifest().cases}
    assert set(table) == keys
    assert min(table.values()) >= 300
    assert max(table.values()) <= case_budgets.USABLE_JOB_SEC


def test_committed_table_is_what_the_committed_timings_give() -> None:
    """Rewriting from case_timings.toml reproduces case_budgets.toml."""
    script = _load_script()
    text, _ = script.derive(script.TIMINGS_PATH.read_text(encoding="utf-8"))
    assert TABLE.read_text(encoding="utf-8") == text


def _row(**over: object) -> dict[str, object]:
    row: dict[str, object] = {
        "flex_status": "ok",
        "flex_completed": True,
        "flex_wall_sec": 100.0,
        "flex_steps": 10,
        "mesa_status": "ok",
        "mesa_wall_sec": 80.0,
        "mesa_steps": 10,
    }
    row.update(over)
    return row


def test_measured_case_gets_twice_its_wall_time_above_the_floor() -> None:
    script = _load_script()
    got = script.project(_row(flex_wall_sec=1000.0), 1.3)
    assert got["basis"] == "measured"
    assert got["uncapped_sec"] == 2040  # 2 x 1000 s, rounded up to 60 s
    assert script.project(_row(flex_wall_sec=5.0), 1.3)["uncapped_sec"] == 300


def test_cut_off_case_is_projected_from_mesa_wall_time() -> None:
    script = _load_script()
    row = _row(
        flex_status="error",
        flex_completed=False,
        flex_wall_sec=0.1,
        flex_steps=0,
        mesa_wall_sec=1000.0,
    )
    got = script.project(row, 1.5)
    assert got["basis"] == "mesa_wall_ratio"
    assert got["projected_sec"] == 1500
    assert got["uncapped_sec"] == 4500


def test_timed_out_case_keeps_at_least_its_old_budget_as_projection() -> None:
    script = _load_script()
    row = _row(
        flex_status="timeout",
        flex_completed=False,
        flex_wall_sec=3000.0,
        mesa_wall_sec=100.0,
    )
    got = script.project(row, 1.3)
    assert got["basis"] == "timeout_lower_bound"
    assert got["projected_sec"] == 3000


def test_load_case_budgets_rejects_malformed_tables(
    tmp_path: pathlib.Path,
) -> None:
    bad = tmp_path / "b.toml"
    bad.write_text('[[cases]]\nsuite = "star"\ncase = "x"\nbudget_sec = 0\n')
    with pytest.raises(ValueError, match="not positive"):
        case_budgets.load_case_budgets(bad)
    bad.write_text('[[cases]]\nsuite = "star"\n')
    with pytest.raises(ValueError, match="malformed"):
        case_budgets.load_case_budgets(bad)
    bad.write_text("x = 1\n")
    with pytest.raises(ValueError, match="no 'cases'"):
        case_budgets.load_case_budgets(bad)


@pytest.mark.parametrize("count", [2, 5, 16])
def test_weighted_shards_partition_cases_and_balance_budgets(
    count: int,
) -> None:
    """Every case is in one shard and no shard is far above the mean load."""
    entries = list(case_manifest.load_case_manifest().cases)
    weights = case_budgets.load_case_budgets(TABLE)
    shards = [
        harness_shards.shard_entries(entries, i, count, weights=weights)
        for i in range(1, count + 1)
    ]
    keys = [e.key for s in shards for e in s]
    assert sorted(keys) == sorted(e.key for e in entries)
    loads = [sum(weights[e.key] for e in s) for s in shards]
    heaviest_case = max(weights.values())
    mean = sum(loads) / count
    assert max(loads) <= mean + heaviest_case
    again = harness_shards.shard_entries(entries, 1, count, weights=weights)
    assert [e.key for e in again] == [e.key for e in shards[0]]


def test_harness_case_timeout_prefers_the_table_over_timeout_sec() -> None:
    """``--case-budgets`` sets each case's wall budget; others keep the cap."""
    from flexmesa.testing import trace_cli

    ns = trace_cli._build_harness_parser().parse_args(
        ["run", "--case-budgets", str(TABLE), "--timeout-sec", "999"]
    )
    ns.case_budget_table = case_budgets.load_case_budgets(ns.case_budgets)
    ns.short_budget_keys = frozenset()
    entries = {e.key: e for e in case_manifest.load_case_manifest().cases}
    key = ("star", "ns_h")
    assert trace_cli._case_timeout(ns, entries[key]) == float(
        ns.case_budget_table[key]
    )
    ns.case_budget_table = {}
    assert trace_cli._case_timeout(ns, entries[key]) == 999.0
