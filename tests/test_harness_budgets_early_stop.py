# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for per-case budgets, the ``passes`` preset and the early stop.

The early divergence check is judged against the final comparison on real
committed MESA ``26.4.1`` reference histories: a partial history that the
early check flags must also fail the final along-track comparison of the
complete run, and a run that passes the final comparison must never be
flagged at any prefix (``DECISIONS #69``).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from flexmesa.testing import (
    _case_verdict,
    baseline_regression,
    case_manifest,
    case_report,
    early_stop,
    harness_budgets,
    harness_runner,
    harness_shards,
    history_compare,
    stage_steps,
    trace_cli,
)

Record = harness_runner.CaseHarnessRecord
Table = history_compare.HistoryTable


@pytest.fixture(scope="module")
def manifest() -> case_manifest.HarnessManifest:
    return case_manifest.load_case_manifest()


@pytest.fixture(scope="module")
def baseline_records() -> list[Record]:
    _, _, records = case_report.build_manifest_inventory_records()
    return records


# ---------------------------------------------------------------------------
# Per-case budgets
# ---------------------------------------------------------------------------


def _executed(records: list[Record], n: int) -> list[Record]:
    return [r for r in records if r.executed][:n]


def _entries(
    manifest: case_manifest.HarnessManifest, recs: list[Record]
) -> list[case_manifest.CaseManifestEntry]:
    keys = {(r.suite, r.case) for r in recs}
    return [e for e in manifest.cases if e.key in keys]


def test_budget_is_k_times_wall_time_bounded_by_floor_and_cap(
    manifest: case_manifest.HarnessManifest, baseline_records: list[Record]
) -> None:
    """The budget is max(K x wall, floor), capped at the scheduled budget."""
    recs = _executed(baseline_records, 4)
    recs = [dataclasses.replace(r, run_status="ok") for r in recs]
    entries = _entries(manifest, recs)
    k0, k1, k2, k3 = (e.key for e in entries)
    walls = {k0: 10.0, k1: 200.0, k2: 400.0}
    out = harness_budgets.derive_budgets(
        entries,
        recs,
        walls,
        timeout_sec=600.0,
        factor=2.0,
        floor_sec=120.0,
        step_factor=None,
    )
    assert out[k0].timeout_sec == 120.0  # floor
    assert out[k1].timeout_sec == 400.0  # K x wall
    assert out[k2].timeout_sec == 600.0  # capped
    assert out[k3].timeout_sec == 600.0  # no recorded wall time
    assert [out[k].from_baseline for k in (k0, k1, k2, k3)] == [
        True,
        True,
        False,
        False,
    ]


def test_budget_of_a_baseline_timeout_keeps_the_full_budget(
    manifest: case_manifest.HarnessManifest, baseline_records: list[Record]
) -> None:
    """A case whose baseline ended on a budget has no wall time to scale."""
    (rec,) = _executed(baseline_records, 1)
    rec = dataclasses.replace(rec, run_status="timeout")
    (entry,) = _entries(manifest, [rec])
    out = harness_budgets.derive_budgets(
        [entry],
        [rec],
        {entry.key: 5.0},
        timeout_sec=600.0,
        factor=2.0,
        floor_sec=120.0,
        step_factor=None,
    )
    assert out[entry.key].timeout_sec == 600.0
    assert not out[entry.key].from_baseline


def test_step_bound_is_a_multiple_of_the_mesa_step_count(
    manifest: case_manifest.HarnessManifest, baseline_records: list[Record]
) -> None:
    """``mesa_steps`` of the baseline row feeds the step bound."""
    (rec,) = _executed(baseline_records, 1)
    rec = dataclasses.replace(rec, mesa_steps=445)
    (entry,) = _entries(manifest, [rec])
    out = harness_budgets.derive_budgets(
        [entry],
        [rec],
        {},
        timeout_sec=600.0,
        factor=2.0,
        floor_sec=120.0,
        step_factor=3.0,
    )
    assert out[entry.key].mesa_steps == 445
    assert out[entry.key].step_factor == 3.0


def test_non_positive_budget_arguments_are_rejected(
    manifest: case_manifest.HarnessManifest,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        harness_budgets.derive_budgets(
            [],
            [],
            {},
            timeout_sec=600.0,
            factor=0.0,
            floor_sec=120.0,
            step_factor=None,
        )


def test_wall_times_skip_budget_cut_and_unmeasured_rows(
    tmp_path: pathlib.Path,
) -> None:
    rows = [
        {"suite": "star", "case": "a", "executed": True, "wall_time_sec": 12.5},
        {
            "suite": "star",
            "case": "b",
            "executed": True,
            "wall_time_sec": 600.0,
            "run_status": "timeout",
        },
        {"suite": "star", "case": "c", "executed": True, "wall_time_sec": 0.0},
        {"suite": "star", "case": "d", "executed": False},
    ]
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"cases": rows}), encoding="utf-8")
    assert harness_budgets.load_wall_times(path) == {("star", "a"): 12.5}


def test_budget_from_baseline_needs_recorded_wall_times(
    manifest: case_manifest.HarnessManifest, baseline_records: list[Record]
) -> None:
    """The committed baseline stores zero wall times, so the option says so."""
    ns = trace_cli._build_harness_parser().parse_args(
        ["run", "--budget-from-baseline"]
    )
    with pytest.raises(ValueError, match="wall-times-json"):
        trace_cli._case_budgets(ns, list(manifest.cases), baseline_records)


def test_budget_from_baseline_sets_case_timeouts(
    manifest: case_manifest.HarnessManifest,
    baseline_records: list[Record],
    tmp_path: pathlib.Path,
) -> None:
    (rec,) = _executed(baseline_records, 1)
    (entry,) = _entries(manifest, [rec])
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "suite": entry.suite,
                        "case": entry.case,
                        "executed": True,
                        "wall_time_sec": 100.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ns = trace_cli._build_harness_parser().parse_args(
        [
            "run",
            "--budget-from-baseline",
            "--wall-times-json",
            str(path),
            "--budget-factor",
            "1.5",
            "--budget-floor-sec",
            "60",
        ]
    )
    ns.short_budget_keys = frozenset()
    ns.budgets = trace_cli._case_budgets(ns, [entry], baseline_records)
    assert trace_cli._case_timeout(ns, entry) == 150.0
    other = next(e for e in manifest.cases if e.key != entry.key)
    assert trace_cli._case_timeout(ns, other) == 600.0


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_passes_preset_selects_the_baseline_pass_cases(
    manifest: case_manifest.HarnessManifest, baseline_records: list[Record]
) -> None:
    passing = harness_budgets.passing_keys(baseline_records)
    assert passing
    chosen = trace_cli._select_entries(manifest, "passes", None, passing)
    assert {e.key for e in chosen} == passing
    assert all(
        r.verdict == "pass"
        for r in baseline_records
        if (r.suite, r.case) in passing
    )
    # Explicit --cases still wins over the preset.
    one = trace_cli._select_entries(
        manifest, "passes", [f"{chosen[0].suite}/{chosen[0].case}"], passing
    )
    assert [e.key for e in one] == [chosen[0].key]


def test_passes_is_a_cli_preset() -> None:
    ns = trace_cli._build_harness_parser().parse_args(
        ["run", "--preset", "passes"]
    )
    assert ns.preset == "passes"


# ---------------------------------------------------------------------------
# Early stop: refusals and records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra", [["--update-baseline", "b.json"], ["--save-baseline", "b.json"]]
)
def test_early_stop_refuses_baseline_updating_runs(extra: list[str]) -> None:
    ns = trace_cli._build_harness_parser().parse_args(
        ["run", "--early-stop-on-divergence", *extra]
    )
    assert "DECISIONS #69" in trace_cli._early_stop_refusal(ns)
    assert (
        trace_cli.harness_main(["run", "--early-stop-on-divergence", *extra])
        == 1
    )


def test_early_stop_refuses_step_capped_runs() -> None:
    ns = trace_cli._build_harness_parser().parse_args(
        ["run", "--early-stop-on-divergence", "--max-steps", "3"]
    )
    assert trace_cli._early_stop_refusal(ns)


def test_shards_with_early_stop_cannot_update_the_baseline(
    tmp_path: pathlib.Path,
) -> None:
    run = {
        "shard": [1, 1],
        "numerics_mode": "strict",
        "max_steps": None,
        "allow_unsupported_controls": False,
        "timeout_sec": 600.0,
        "short_budget_sec": None,
        "max_rss_gb": 5.5,
        "early_stop_on_divergence": True,
        "selected_cases": [],
    }
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"run": run, "cases": []}), encoding="utf-8")
    merged = harness_shards.merge_shard_payloads(
        [json.loads(path.read_text(encoding="utf-8"))]
    )
    assert merged.settings["early_stop_on_divergence"] is True
    assert (
        trace_cli.harness_main(
            ["merge", str(path), "--update-baseline", str(tmp_path / "b.json")]
        )
        == 1
    )


def _spec(
    manifest: case_manifest.HarnessManifest, key: tuple[str, str]
) -> case_manifest.CaseManifestEntry:
    return next(e for e in manifest.cases if e.key == key)


def test_early_stopped_case_without_gap_is_an_undeclared_fail(
    manifest: case_manifest.HarnessManifest,
) -> None:
    spec = _spec(manifest, ("star", "high_z"))
    outcome = _case_verdict.TraceRunOutcome(
        suite="star",
        case="high_z",
        status="diverged",
        termination_code="diverged",
        steps=40,
        retries=0,
        wall_time_sec=3.0,
        peak_rss_gb=0.1,
        history_path=None,
        error_message="early stop: log_L left tolerance 0.005 at star_age = 1",
    )
    rec = _case_verdict.evaluate_case_record(spec, {}, outcome)
    assert rec.verdict == "fail (early stop)"
    assert rec.is_undeclared_failure
    gap = case_manifest.KnownGapSpec(
        suite="star", case="high_z", issue="#161", reason="r"
    )
    gapped = _case_verdict.evaluate_case_record(spec, {spec.key: gap}, outcome)
    assert gapped.verdict == "declared gap (#161)"
    assert not gapped.is_undeclared_failure


def test_early_stop_of_a_passing_baseline_case_is_a_regression(
    baseline_records: list[Record],
) -> None:
    base = next(r for r in baseline_records if r.verdict == "pass")
    stopped = dataclasses.replace(
        base,
        run_status="diverged",
        steps=30,
        termination_matches=False,
        error_message="early stop: log_L left tolerance 0.005 at star_age = "
        "1.0e6 (model 30)",
    )
    (reg,) = baseline_regression.find_baseline_regressions(
        [stopped], baseline_records
    )
    assert reg.kind == "left tolerance"
    assert "model 30" in reg.detail


def _stop_message(model: int) -> str:
    return f"early stop: log_L left tolerance 0.005 at x (model {model})"


def test_early_stop_of_a_gap_case_that_already_diverged_is_not_a_regression(
    baseline_records: list[Record],
) -> None:
    base = next(
        r
        for r in baseline_records
        if r.executed
        and r.comparison is not None
        and r.comparison.first_diverging_step is not None
        and not r.is_within_tolerances
    )
    first = base.comparison.first_diverging_step  # type: ignore[union-attr]
    assert first is not None
    stopped = dataclasses.replace(
        base,
        run_status="diverged",
        steps=first + 10,
        error_message=_stop_message(first + 5),
    )
    assert (
        baseline_regression.find_baseline_regressions(
            [stopped], baseline_records
        )
        == []
    )
    earlier = dataclasses.replace(
        stopped,
        error_message=_stop_message(first - 1),
    )
    kinds = [
        r.kind
        for r in baseline_regression.find_baseline_regressions(
            [earlier], baseline_records
        )
    ]
    assert kinds == ["diverges from MESA earlier"]


# ---------------------------------------------------------------------------
# Early stop: soundness against the final comparison on real references
# ---------------------------------------------------------------------------

_REAL_CASES = [
    ("star", "high_z"),
    ("star", "hot_cool_wind"),
    ("star", "1.3M_ms_high_Z"),
    ("star", "hb_2M"),
    ("star", "conv_core_cpm"),
    ("star", "7M_prems_to_AGB"),
]


def _load_reference(
    spec: case_manifest.CaseManifestEntry,
) -> Table:
    ref = history_compare.load_history_file(
        case_manifest.repo_root() / spec.reference_path
    )
    return _case_verdict._augment_ref_columns(
        ref, spec, "r26.4.1", case_manifest.repo_root()
    )


def _slice(tbl: Table, n: int) -> Table:
    return Table(
        columns={k: v[:n].copy() for k, v in tbl.columns.items()},
        num_rows=n,
    )


def _perturbed(
    ref: Table,
    spec: case_manifest.CaseManifestEntry,
    rng: np.random.Generator,
    scale: float,
    frac: float,
) -> Table:
    """Return ``ref`` with tracked columns offset by ``scale`` x tolerance."""
    cols = {k: v.copy() for k, v in ref.columns.items()}
    n = ref.num_rows
    hit = rng.random(n) < frac
    for key, tol in spec.track_tolerances.items():
        col = key[4:] if key.startswith("rel_") else key
        if col not in cols or col == spec.independent_var:
            continue
        shift = scale * tol * rng.choice([-1.0, 1.0])
        if key.startswith("rel_"):
            cols[col] = np.where(hit, cols[col] * (1.0 + shift), cols[col])
        else:
            cols[col] = np.where(hit, cols[col] + shift, cols[col])
    return Table(columns=cols, num_rows=n)


@pytest.mark.parametrize("key", _REAL_CASES, ids=lambda k: f"{k[0]}/{k[1]}")
def test_early_check_is_sound_on_real_reference_histories(
    manifest: case_manifest.HarnessManifest, key: tuple[str, str]
) -> None:
    """An early hit on a prefix implies the final comparison fails.

    For perturbed copies of the committed MESA reference histories, every
    prefix flagged by the early check belongs to a run whose complete
    comparison fails that column, and a run within tolerance everywhere is
    never flagged.
    """
    spec = _spec(manifest, key)
    ref = _load_reference(spec)
    n = ref.num_rows
    assert n >= 10
    prefixes = sorted({max(4, n * i // 12) for i in range(1, 13)} | {n})
    rng = np.random.default_rng(20261126)
    flagged_any = False
    for scale, frac in ((0.0, 0.0), (0.4, 1.0), (1.6, 0.05), (3.0, 0.5)):
        run = _perturbed(ref, spec, rng, scale, frac)
        final = history_compare.compare_histories(ref, run, spec)
        for m in prefixes:
            hit = early_stop.find_early_divergence(ref, _slice(run, m), spec)
            if hit is None:
                continue
            flagged_any = True
            assert hit.column in final.failed_track_columns, (
                key,
                scale,
                m,
                hit,
                final.failed_track_columns,
            )
        if final.within_track_tolerances:
            assert all(
                early_stop.find_early_divergence(ref, _slice(run, m), spec)
                is None
                for m in prefixes
            )
    assert flagged_any


@pytest.mark.parametrize("key", _REAL_CASES, ids=lambda k: f"{k[0]}/{k[1]}")
def test_early_check_agrees_with_final_on_identical_and_drifting_runs(
    manifest: case_manifest.HarnessManifest, key: tuple[str, str]
) -> None:
    """Identical histories are never flagged; a drifting one is, early."""
    spec = _spec(manifest, key)
    ref = _load_reference(spec)
    n = ref.num_rows
    for m in (4, n // 2, n):
        assert (
            early_stop.find_early_divergence(ref, _slice(ref, m), spec) is None
        )
    drift_key, drift_tol = next(
        (k, t)
        for k, t in spec.track_tolerances.items()
        if not k.startswith("rel_") and k in ref.columns
    )
    cols = {k: v.copy() for k, v in ref.columns.items()}
    start = n // 3
    cols[drift_key][start:] += 4.0 * drift_tol
    run = Table(columns=cols, num_rows=n)
    final = history_compare.compare_histories(ref, run, spec)
    assert drift_key in final.failed_track_columns
    hit = early_stop.find_early_divergence(ref, _slice(run, start + 6), spec)
    assert hit is not None
    assert hit.column == drift_key
    assert hit.x >= float(ref.columns[spec.independent_var][start - 1])
    assert "left tolerance" in hit.message()


def test_early_check_on_a_sparse_reference_uses_reference_rows() -> None:
    """A reference that skips models is compared only at its own rows."""
    n = 40
    x = np.linspace(1.0, 40.0, n)
    ref = Table(
        columns={
            "model_number": np.arange(1.0, n + 1) * 5.0,
            "star_age": x,
            "log_L": np.sin(x / 7.0),
        },
        num_rows=n,
    )
    spec = case_manifest.CaseManifestEntry(
        suite="star",
        case="synthetic",
        reference_status="mesa_26_4_1",
        reference_path="",
        required_termination="",
        independent_var="star_age",
        final_tolerances={},
        track_tolerances={"log_L": 0.01},
    )
    good = _slice(ref, n)
    assert early_stop.find_early_divergence(ref, good, spec) is None
    cols = {k: v.copy() for k, v in ref.columns.items()}
    cols["log_L"][20:] += 0.05
    bad = Table(columns=cols, num_rows=n)
    assert history_compare.compare_histories(
        ref, bad, spec
    ).failed_track_columns
    hit = early_stop.find_early_divergence(ref, _slice(bad, 30), spec)
    assert hit is not None
    assert hit.x == pytest.approx(x[20])


def test_early_check_waits_for_enough_rows(
    manifest: case_manifest.HarnessManifest,
) -> None:
    spec = _spec(manifest, ("star", "high_z"))
    ref = _load_reference(spec)
    assert early_stop.find_early_divergence(ref, _slice(ref, 2), spec) is None


def test_partial_history_drops_the_incomplete_last_line(
    tmp_path: pathlib.Path,
) -> None:
    """A row being written is never read."""
    lines = ["# note", "model_number\tstar_age\tlog_L"]
    lines += [f"{i}\t{float(i)}\t{0.1 * i}" for i in range(1, 7)]
    text = "\n".join(lines) + "\n7\t7.0\t0."
    path = tmp_path / "history.data"
    path.write_text(text, encoding="utf-8")
    tbl = early_stop.read_partial_history(path)
    assert tbl is not None
    assert tbl.num_rows == 6
    assert early_stop.read_partial_history(tmp_path / "missing.data") is None


def test_monitor_requests_the_step_bound(
    manifest: case_manifest.HarnessManifest, tmp_path: pathlib.Path
) -> None:
    spec = _spec(manifest, ("star", "high_z"))
    lines = ["# note", "model_number\tstar_age\tlog_L"]
    lines += [f"{i}\t{float(i)}\t{0.1 * i}" for i in range(1, 11)]
    path = tmp_path / "history.data"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    below = early_stop.EarlyStopMonitor(spec, None, 11, "bound")
    assert below(path) is None
    at = early_stop.EarlyStopMonitor(spec, None, 10, "bound")
    request = at(path)
    assert request is not None
    assert (request.status, request.message) == ("timeout", "bound")


def test_early_stop_applies_only_to_single_stage_star_cases() -> None:
    """File copies between stages do not count; several runs do."""
    for case, expected in (
        ("high_z", True),
        ("hot_cool_wind", True),
        ("1M_pre_ms_to_wd", False),
        ("make_planets", False),
    ):
        case_dir = harness_runner.resolve_case_dir("star", case)
        assert early_stop.single_stage_star_case(case_dir) is expected, case


def test_early_stop_is_not_reported_as_ending_before_the_baseline(
    baseline_records: list[Record],
) -> None:
    """A deliberate stop short of the baseline's steps is not a regression."""
    base = next(
        r
        for r in baseline_records
        if r.executed
        and r.comparison is not None
        and r.comparison.first_diverging_step is not None
        and not r.is_within_tolerances
        and r.steps > 20
    )
    first = base.comparison.first_diverging_step  # type: ignore[union-attr]
    assert first is not None
    stopped = dataclasses.replace(
        base,
        run_status="diverged",
        steps=base.steps // 2,
        termination_matches=False,
        error_message=_stop_message(first + 1),
    )
    assert (
        baseline_regression.find_baseline_regressions(
            [stopped], baseline_records
        )
        == []
    )


@pytest.mark.parametrize(
    "key",
    [("star", "high_z"), ("star", "hot_cool_wind")],
    ids=lambda k: f"{k[0]}/{k[1]}",
)
def test_dense_run_against_a_sparse_reference_is_judged_at_reference_rows(
    manifest: case_manifest.HarnessManifest, key: tuple[str, str]
) -> None:
    """Between-row excursions of a dense run are not out of tolerance.

    The reference skips models, so the final comparison looks only at the
    reference rows. A run written every model that departs from the
    reference only between those rows passes the final comparison and must
    never be stopped, at any prefix, including prefixes with fewer than two
    reference rows (where the final comparison's grid is not yet known).
    """
    spec = _spec(manifest, key)
    ref = _load_reference(spec)
    assert history_compare._reference_skips_models(ref)
    x_ref = ref.columns[spec.independent_var]
    x = np.unique(
        np.concatenate(
            [
                x_ref,
                np.geomspace(x_ref[0], x_ref[-1], 2000),
                np.linspace(x_ref[0], x_ref[-1], 600),
            ]
        )
    )
    right = np.clip(np.searchsorted(x_ref, x), 1, x_ref.size - 1)
    left_node, right_node = x_ref[right - 1], x_ref[right]
    tent = np.clip(
        2.0
        * np.minimum(x - left_node, right_node - x)
        / (right_node - left_node),
        0.0,
        1.0,
    )
    tent[np.isin(x, x_ref)] = 0.0
    cols = {
        name: np.interp(x, x_ref, values)
        for name, values in ref.columns.items()
    }
    cols["model_number"] = np.arange(1.0, x.size + 1.0)
    for tol_key, tol in spec.track_tolerances.items():
        name = tol_key[4:] if tol_key.startswith("rel_") else tol_key
        if name in cols and name not in (spec.independent_var, "num_zones"):
            scale = np.abs(cols[name]) if tol_key.startswith("rel_") else 1.0
            cols[name] = cols[name] + 5.0 * tol * tent * scale
    run = Table(columns=cols, num_rows=x.size)
    final = history_compare.compare_histories(ref, run, spec)
    assert final.within_track_tolerances
    for m in (3, 30, 200, 500, x.size // 3, x.size // 2, x.size):
        assert (
            early_stop.find_early_divergence(ref, _slice(run, m), spec) is None
        )


# ---------------------------------------------------------------------------
# Per-stage step counts
# ---------------------------------------------------------------------------


def test_mesa_stage_steps_cover_every_case_and_match_known_stages() -> None:
    table = stage_steps.load_mesa_stage_steps()
    assert len(table) == 106
    assert table[("star", "make_planets")] == (76, 22, 112)
    assert table[("star", "high_z")] == (445,)


def test_stage_deltas_align_stages_from_the_end() -> None:
    mesa = (76, 22, 112)
    full = stage_steps.stage_deltas((76, 22, 113), mesa)
    assert [d.delta for d in full] == [0, 0, 1]
    skipped = stage_steps.stage_deltas((22, 112), mesa)
    assert [(d.stage, d.mesa) for d in skipped] == [(1, 22), (2, 112)]
    extra = stage_steps.stage_deltas((5, 22, 112), (22, 112))
    assert extra[0].mesa is None and extra[0].delta is None


def test_step_bound_adds_the_larger_margin() -> None:
    assert stage_steps.step_bound(100, 5, 0.02) == 105
    assert stage_steps.step_bound(1000, 5, 0.02) == 1020


def _executed_pass(baseline_records: list[Record]) -> Record:
    return next(r for r in baseline_records if r.verdict == "pass")


def test_pass_with_other_step_count_is_a_warning_not_a_verdict_change(
    baseline_records: list[Record],
) -> None:
    base = _executed_pass(baseline_records)
    mesa = stage_steps.load_mesa_stage_steps()[(base.suite, base.case)]
    same = dataclasses.replace(base, stage_steps=mesa, mesa_stage_steps=mesa)
    assert not same.step_count_differs
    off = dataclasses.replace(
        base,
        stage_steps=(*mesa[:-1], mesa[-1] + 3),
        mesa_stage_steps=mesa,
    )
    assert off.step_count_differs
    assert off.verdict == "pass"
    row = case_report.record_to_dict(off)
    assert row["stage_step_deltas"][-1] == 3
    assert row["step_count_differs"] is True
    manifest = case_manifest.load_case_manifest()
    summary = case_report.build_report_payload(
        manifest, [off], {}, numerics_mode="strict"
    )["summary"]
    assert summary["passes_with_step_count_differences"] == [
        f"{base.suite}/{base.case}"
    ]
    md = case_report.render_case_table_markdown(manifest, [off])
    assert "Step counts that differ from MESA" in md
    assert "(warning)" in md
    assert (
        "Step counts that differ"
        not in case_report.render_case_table_markdown(manifest, [same])
    )
    # The committed baseline carries the run's stage counts (it is a
    # step-count-enforced snapshot), which equal MESA's for a pass.
    assert base.stage_steps == mesa


def test_monitor_stops_a_run_past_the_mesa_step_count(
    manifest: case_manifest.HarnessManifest, tmp_path: pathlib.Path
) -> None:
    spec = _spec(manifest, ("star", "high_z"))
    lines = ["# note", "model_number\tstar_age\tlog_L"]
    lines += [f"{i}\t{float(i)}\t{0.1 * i}" for i in range(1, 12)]
    path = tmp_path / "history.data"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    bound = stage_steps.step_bound(5, 5, 0.02)
    mon = early_stop.EarlyStopMonitor(
        spec, None, excess_limit=bound, mesa_stage_steps=5
    )
    request = mon(path)
    assert request is not None and request.status == "diverged"
    assert request.message == (
        "diverged: exceeded MESA's step count at stage 1 (11 vs 5)"
    )
    ok = early_stop.EarlyStopMonitor(
        spec, None, excess_limit=11, mesa_stage_steps=5
    )
    assert ok(path) is None


def test_stop_on_excess_steps_is_refused_for_baseline_updates() -> None:
    ns = trace_cli._build_harness_parser().parse_args(
        ["run", "--stop-on-excess-steps", "--update-baseline", "b.json"]
    )
    assert "--stop-on-excess-steps" in trace_cli._early_stop_refusal(ns)


def _stored_pass_rows(
    key: tuple[str, str], stages: tuple[int, ...]
) -> dict[tuple[str, str], dict[str, object]]:
    """Return the committed baseline row of ``key`` with run stage counts."""
    root = case_manifest.repo_root()
    raw = json.loads(
        (
            root / "benchmarks" / "history_harness" / "baseline_run_26_4_1.json"
        ).read_text(encoding="utf-8")
    )
    row = next(
        r
        for r in raw["cases"]
        if (r["suite"], r["case"]) == key and r["executed"]
    )
    assert row["verdict"] == "pass"
    return {key: {**row, "stage_steps": list(stages)}}


def _verdict(
    key: tuple[str, str],
    stages: tuple[int, ...],
    *,
    enforce: bool,
    gaps: dict[tuple[str, str], case_manifest.KnownGapSpec] | None = None,
) -> Record:
    manifest = case_manifest.load_case_manifest()
    use = case_manifest.load_known_gaps() if gaps is None else gaps
    rows = _stored_pass_rows(key, stages)
    records = case_report.records_from_stored_rows(
        manifest, use, rows, enforce_step_counts=enforce
    )
    return next(r for r in records if (r.suite, r.case) == key)


def test_enforced_step_counts_relabel_a_pass_with_another_stage_count() -> None:
    key = ("star", "high_mass")
    mesa = stage_steps.load_mesa_stage_steps()[key]
    same = _verdict(key, mesa, enforce=True)
    assert same.verdict == "pass"
    off = _verdict(key, (mesa[-1] + 2,), enforce=True)
    assert off.verdict == "fail (step count)"
    assert off.is_undeclared_failure
    assert off.error_message == (
        "step count differs from MESA at stage 1 "
        f"({mesa[-1] + 2} vs {mesa[-1]})"
    )


def test_unenforced_step_count_difference_keeps_the_pass_verdict() -> None:
    key = ("star", "high_mass")
    mesa = stage_steps.load_mesa_stage_steps()[key]
    off = _verdict(key, (mesa[-1] + 2,), enforce=False)
    assert off.verdict == "pass"
    assert off.step_count_differs
    assert not off.is_undeclared_failure


def test_enforced_step_counts_check_every_stage_of_a_multi_stage_case() -> None:
    key = ("star", "make_he_wd")
    mesa = stage_steps.load_mesa_stage_steps()[key]
    assert mesa == (78, 136)
    assert _verdict(key, mesa, enforce=True).verdict == "pass"
    first = _verdict(key, (79, 136), enforce=True)
    assert first.verdict == "fail (step count)"
    assert "stage 1 (79 vs 78)" in first.error_message
    last = _verdict(key, (78, 137), enforce=True)
    assert last.verdict == "fail (step count)"
    assert "stage 2 (137 vs 136)" in last.error_message


def test_enforced_step_count_difference_of_a_declared_gap_stays_declared() -> (
    None
):
    key = ("star", "high_mass")
    mesa = stage_steps.load_mesa_stage_steps()[key]
    gap = case_manifest.KnownGapSpec(
        suite="star", case="high_mass", issue="#1", reason="test"
    )
    rec = _verdict(key, (mesa[-1] + 1,), enforce=True, gaps={key: gap})
    assert rec.verdict == "declared gap (#1)"
    assert not rec.is_undeclared_failure


def test_a_missing_mesa_step_count_is_not_compared() -> None:
    assert _case_verdict.first_step_difference((5, 6), ()) == ""
    assert _case_verdict.first_step_difference((), (5,)) == ""
    assert _case_verdict.first_step_difference((5,), (5,)) == ""


def test_enforce_flag_is_off_by_default_and_recorded_for_shards() -> None:
    parser = trace_cli._build_harness_parser()
    assert not parser.parse_args(["run"]).enforce_step_counts
    assert parser.parse_args(
        ["run", "--enforce-step-counts"]
    ).enforce_step_counts
    assert "enforce_step_counts" in harness_shards.RUN_SETTING_KEYS


def test_rust_worker_script_reports_stage_model_numbers() -> None:
    script = harness_runner._RUST_WORKER_BOOTSTRAP
    assert "'stage_steps'" in script and "'model_number'" in script
    compile(script, "<worker>", "exec")
