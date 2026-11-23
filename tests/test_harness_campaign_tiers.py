# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the short-budget campaign tier and sharded harness campaigns."""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
from typing import Any

import pytest

from flexmesa.testing import (
    baseline_regression,
    case_manifest,
    case_report,
    harness_runner,
    harness_shards,
    history_compare,
    trace_cli,
)

Record = harness_runner.CaseHarnessRecord


def _comparison(
    first_step: int | None, first_x: float | None = None
) -> history_compare.HistoryComparison:
    return history_compare.HistoryComparison(
        independent_var="star_age",
        ref_rows=300,
        run_rows=75,
        overlap_min=1.0,
        overlap_max=2.0,
        final_diffs={},
        within_final_tolerances=False,
        failed_final_columns=(),
        track_deviations=(),
        within_track_tolerances=False,
        failed_track_columns=(),
        first_diverging_step=first_step,
        first_diverging_x=first_x,
        ref_span=10.0,
    )


@pytest.fixture(scope="module")
def baseline_records() -> list[Record]:
    """Records rebuilt from the committed baseline run."""
    _, _, records = case_report.build_manifest_inventory_records()
    return records


def _timeout_base(
    baseline_records: list[Record],
) -> tuple[Record, list[Record]]:
    """Return a synthetic baseline row cut off by its budget, and the list.

    The row replaces the first executed baseline record, so the test does
    not depend on which committed rows currently end on the budget.
    """
    idx = next(i for i, r in enumerate(baseline_records) if r.executed)
    base = dataclasses.replace(
        baseline_records[idx],
        run_status="timeout",
        observed_termination="",
        termination_matches=False,
        steps=75,
        comparison=_comparison(2616, 1.5),
    )
    records = list(baseline_records)
    records[idx] = base
    return base, records


def test_short_budget_run_that_reproduces_a_prefix_is_not_a_regression(
    baseline_records: list[Record],
) -> None:
    """A shorter run of an unchanged trajectory passes the gate."""
    base, records = _timeout_base(baseline_records)
    short = frozenset({(base.suite, base.case)})
    find = baseline_regression.find_baseline_regressions

    prefix = dataclasses.replace(base, steps=20, comparison=_comparison(None))
    assert find([prefix], records, short_budget=short) == []
    same_divergence = dataclasses.replace(base, steps=80)
    assert find([same_divergence], records, short_budget=short) == []
    # Reaching upstream's termination early is an improvement.
    finished = dataclasses.replace(
        base, run_status="ok", termination_matches=True, steps=10
    )
    assert find([finished], records, short_budget=short) == []
    # Running out of memory says nothing about the trajectory.
    oom = dataclasses.replace(base, run_status="oom_killed", steps=0)
    assert find([oom], records, short_budget=short) == []


def test_short_budget_run_that_takes_no_steps_is_a_regression(
    baseline_records: list[Record],
) -> None:
    """A short-budget run with zero steps against a row with steps fails."""
    base, records = _timeout_base(baseline_records)
    short = frozenset({(base.suite, base.case)})
    stalled = dataclasses.replace(base, steps=0, comparison=None)
    (reg,) = baseline_regression.find_baseline_regressions(
        [stalled], records, short_budget=short
    )
    assert reg.kind == "stopped taking steps"
    # Without the short-budget tier a run cut off by its budget is
    # inconclusive, as before.
    assert (
        baseline_regression.find_baseline_regressions([stalled], records) == []
    )


def test_short_budget_run_ending_before_baseline_steps_is_a_regression(
    baseline_records: list[Record],
) -> None:
    """An error before the step count the baseline passed is a regression."""
    base, records = _timeout_base(baseline_records)
    short = frozenset({(base.suite, base.case)})
    find = baseline_regression.find_baseline_regressions

    crashed = dataclasses.replace(base, run_status="error", steps=40)
    (reg,) = find([crashed], records, short_budget=short)
    assert reg.kind == "ended before the baseline's step count"
    assert "75" in reg.detail and "40" in reg.detail
    early_stop = dataclasses.replace(
        base,
        run_status="ok",
        observed_termination="min_timestep_limit",
        steps=40,
    )
    (reg,) = find([early_stop], records, short_budget=short)
    assert reg.kind == "ended before the baseline's step count"
    # Beyond the baseline's step count the baseline says nothing.
    later = dataclasses.replace(base, run_status="error", steps=90)
    assert find([later], records, short_budget=short) == []


def test_short_budget_run_diverging_from_mesa_earlier_is_a_regression(
    baseline_records: list[Record],
) -> None:
    """Leaving a track tolerance at an earlier model is a regression."""
    base, records = _timeout_base(baseline_records)
    short = frozenset({(base.suite, base.case)})
    find = baseline_regression.find_baseline_regressions

    earlier = dataclasses.replace(base, steps=30, comparison=_comparison(2600))
    (reg,) = find([earlier], records, short_budget=short)
    assert reg.kind == "diverges from MESA earlier"

    # A baseline that never left tolerance within its coordinate range.
    clean_base = dataclasses.replace(base, comparison=_comparison(None))
    clean_records = [clean_base if r is base else r for r in records]
    inside = dataclasses.replace(base, comparison=_comparison(2601, 1.9))
    (reg,) = find([inside], clean_records, short_budget=short)
    assert reg.kind == "diverges from MESA earlier"
    beyond = dataclasses.replace(base, comparison=_comparison(2700, 2.5))
    assert find([beyond], clean_records, short_budget=short) == []


def test_short_budget_case_with_terminating_baseline_uses_default_rules(
    baseline_records: list[Record],
) -> None:
    """The prefix rules apply only to rows that ended on the time budget."""
    ok = next(
        r
        for r in baseline_records
        if r.executed and r.termination_matches and r.steps > 0
    )
    short = frozenset({(ok.suite, ok.case)})
    lost = dataclasses.replace(
        ok, termination_matches=False, observed_termination="max_age"
    )
    (reg,) = baseline_regression.find_baseline_regressions(
        [lost], baseline_records, short_budget=short
    )
    assert reg.kind == "lost upstream termination"


def test_update_baseline_keeps_committed_row_of_short_budget_timeouts(
    baseline_records: list[Record],
) -> None:
    """A short-budget run cut off by its budget never replaces its row."""
    base, records = _timeout_base(baseline_records)
    short = frozenset({(base.suite, base.case)})
    merge = baseline_regression.merge_records_into_baseline
    idx = records.index(base)

    cut_off = dataclasses.replace(base, steps=20)
    assert merge([cut_off], records, short_budget=short)[idx] is base
    # Without the tier (a full-budget run) the row is replaced as before.
    assert merge([cut_off], records)[idx] is cut_off
    finished = dataclasses.replace(
        base, run_status="ok", termination_matches=True, steps=10
    )
    assert merge([finished], records, short_budget=short)[idx] is finished
    crashed = dataclasses.replace(base, run_status="error", steps=5)
    assert merge([crashed], records, short_budget=short)[idx] is crashed


def test_short_budget_cases_are_the_baseline_timeouts(
    baseline_records: list[Record],
) -> None:
    """``--short-budget-sec`` applies to exactly the baseline timeout rows."""
    ns = trace_cli._build_harness_parser().parse_args(
        ["run", "--preset", "all", "--short-budget-sec", "120"]
    )
    manifest = case_manifest.load_case_manifest()
    keys = trace_cli._short_budget_keys(ns, list(manifest.cases))
    expected = {
        (r.suite, r.case)
        for r in baseline_records
        if r.executed and r.run_status == "timeout"
    }
    assert keys == expected
    ns.short_budget_keys = keys
    for entry in manifest.cases:
        want = 120.0 if entry.key in expected else 600.0
        assert trace_cli._case_timeout(ns, entry) == want

    default_ns = trace_cli._build_harness_parser().parse_args(["run"])
    assert default_ns.short_budget_sec is None
    assert trace_cli._short_budget_keys(default_ns, list(manifest.cases)) == (
        frozenset()
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--short-budget-sec", "600"],
        ["--short-budget-sec", "120", "--save-baseline", "unused.json"],
    ],
)
def test_short_budget_rejects_long_budgets_and_baseline_candidates(
    extra: list[str],
) -> None:
    """A short budget must be shorter and cannot write a baseline candidate."""
    argv = ["run", "--timeout-sec", "600", *extra]
    assert trace_cli.harness_main(argv) == 1


@pytest.mark.parametrize("raw", ["0/4", "5/4", "1", "a/b", "1/2/3", "-1/2"])
def test_parse_shard_rejects_malformed_specifications(raw: str) -> None:
    """Shard specifications must be ``I/N`` with ``1 <= I <= N``."""
    with pytest.raises(ValueError, match="shard"):
        harness_shards.parse_shard(raw)


def test_parse_shard_accepts_valid_specifications() -> None:
    """``I/N`` parses to a 1-based index and a count."""
    assert harness_shards.parse_shard("3/8") == (3, 8)
    assert harness_shards.parse_shard("1/1") == (1, 1)


@pytest.mark.parametrize("count", [1, 2, 3, 8, 9, 106, 120])
def test_shards_partition_cases_and_spread_costly_ones(count: int) -> None:
    """Every case runs in exactly one shard; costly cases are spread."""
    entries = list(case_manifest.load_case_manifest().cases)
    costly = frozenset(e.key for e in entries[5:19])
    shards = [
        harness_shards.shard_entries(entries, i, count, costly)
        for i in range(1, count + 1)
    ]
    keys = [e.key for shard in shards for e in shard]
    assert sorted(keys) == sorted(e.key for e in entries)
    assert len(keys) == len(set(keys))
    order = {e.key: i for i, e in enumerate(entries)}
    for shard in shards:
        assert [order[e.key] for e in shard] == sorted(
            order[e.key] for e in shard
        )
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1
    n_costly = [sum(e.key in costly for e in s) for s in shards]
    assert max(n_costly) - min(n_costly) <= 1
    # The assignment depends only on its inputs.
    again = harness_shards.shard_entries(entries, 1, count, costly)
    assert again == shards[0]


def _baseline_payload() -> dict[str, Any]:
    path = case_report.default_baseline_json_path()
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _shard_payloads(
    payload: dict[str, Any], count: int, **settings: Any
) -> list[dict[str, Any]]:
    """Split a full report into ``count`` shard reports with run blocks."""
    rows = [r for r in payload["cases"] if r.get("executed")]
    run = {
        "numerics_mode": "strict",
        "max_steps": None,
        "allow_unsupported_controls": False,
        "timeout_sec": 600.0,
        "short_budget_sec": None,
        "max_rss_gb": 5.5,
    }
    run.update(settings)
    out: list[dict[str, Any]] = []
    for i in range(count):
        mine = rows[i::count]
        out.append(
            {
                "summary": {},
                "cases": copy.deepcopy(mine),
                "run": dict(
                    run,
                    shard=[i + 1, count],
                    short_budget_cases=[],
                    selected_cases=[f"{r['suite']}/{r['case']}" for r in mine],
                ),
            }
        )
    return out


def test_merge_shard_payloads_combines_complete_campaign() -> None:
    """Complete shards merge into every row with no problems."""
    payload = _baseline_payload()
    merged = harness_shards.merge_shard_payloads(_shard_payloads(payload, 3))
    executed = [r for r in payload["cases"] if r.get("executed")]
    assert merged.problems == ()
    assert len(merged.rows) == len(executed)
    assert merged.settings["timeout_sec"] == 600.0


def test_merge_shard_payloads_reports_missing_and_repeated_work() -> None:
    """Missing shards and cases run twice are completeness problems."""
    shards = _shard_payloads(_baseline_payload(), 3)
    missing = harness_shards.merge_shard_payloads(shards[:2])
    assert any("missing shards [3]" in p for p in missing.problems)

    twice = copy.deepcopy(shards)
    twice[1]["cases"].append(copy.deepcopy(twice[0]["cases"][0]))
    repeated = harness_shards.merge_shard_payloads(twice)
    assert any("in two shards" in p for p in repeated.problems)

    dropped = copy.deepcopy(shards)
    dropped[2]["cases"].pop()
    lost = harness_shards.merge_shard_payloads(dropped)
    assert any("selected but not executed" in p for p in lost.problems)


def test_merge_shard_payloads_rejects_mixed_settings() -> None:
    """Shards run with different settings are not one campaign."""
    payload = _baseline_payload()
    shards = _shard_payloads(payload, 2)
    shards[1]["run"]["timeout_sec"] = 300.0
    with pytest.raises(ValueError, match="timeout_sec"):
        harness_shards.merge_shard_payloads(shards)
    other = _shard_payloads(payload, 2)
    other[1]["run"]["shard"] = [2, 3]
    with pytest.raises(ValueError, match="shard count"):
        harness_shards.merge_shard_payloads(other)
    with pytest.raises(ValueError, match="no shard reports"):
        harness_shards.merge_shard_payloads([])


def test_harness_merge_of_baseline_shards_reproduces_the_baseline(
    tmp_path: pathlib.Path,
) -> None:
    """Merging shards of the committed baseline rewrites it byte for byte."""
    base_path = case_report.default_baseline_json_path()
    reports = []
    for i, shard in enumerate(_shard_payloads(_baseline_payload(), 4)):
        p = tmp_path / f"shard_{i}.json"
        p.write_text(json.dumps(shard), encoding="utf-8")
        reports.append(str(p))
    candidate = tmp_path / "candidate.json"
    table = tmp_path / "case_table.json"
    rc = trace_cli.harness_main(
        [
            "merge",
            *reports,
            "--baseline-json",
            str(base_path),
            "--json-out",
            str(table),
            "--markdown-out",
            str(tmp_path / "case_table.md"),
            "--save-baseline",
            str(candidate),
            "--gate",
        ]
    )
    assert rc == 0
    assert candidate.read_text(encoding="utf-8") == base_path.read_text(
        encoding="utf-8"
    )
    out = json.loads(table.read_text(encoding="utf-8"))
    assert out["summary"]["regressions_vs_baseline"] == []
    assert out["run"]["merged_shards"] == 4

    # An incomplete campaign fails the gate.
    assert (
        trace_cli.harness_main(
            ["merge", *reports[:3], "--baseline-json", str(base_path), "--gate"]
        )
        == 1
    )
