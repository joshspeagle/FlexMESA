# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the stage guard, wall time per step, and star 1 step counts."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import time
from typing import Literal

import pytest

from flexmesa import _rust
from flexmesa.testing import (
    _case_verdict,
    case_manifest,
    case_report,
    harness_runner,
)

_REF = "# test\nmodel_number\tstar_age\tlog_L\n1\t1.0e6\t0.0\n2\t2.0e6\t0.1\n"
_RUN = "# test\nmodel_number\tstar_age\tlog_L\n1\t1.0e6\t0.0\n"


def _fake_stages(monkeypatch: pytest.MonkeyPatch, mods: list[str]) -> None:
    stages = [("do_one", f"inlist_{i}", m) for i, m in enumerate(mods)]
    stages.insert(1, ("cp", "a.mod", "b.mod"))
    monkeypatch.setattr(
        _rust,
        "star_parse_rn_stages",
        lambda case_dir, skip_optional: stages,
    )


def _spec(
    suite: Literal["star", "binary", "astero"] = "star",
) -> case_manifest.CaseManifestEntry:
    return case_manifest.CaseManifestEntry(
        suite=suite,
        case="demo",
        required_termination="max_age",
        independent_var="star_age",
        final_tolerances={"log_L": 0.02},
        track_tolerances={"log_L": 0.10},
        reference_status="mesa_26_4_1",
        reference_path="ref.tsv",
    )


def _outcome(
    run: pathlib.Path, **kwargs: object
) -> harness_runner.TraceRunOutcome:
    base: dict[str, object] = {
        "suite": "star",
        "case": "demo",
        "status": "timeout",
        "termination_code": "timeout",
        "steps": 1,
        "retries": 0,
        "wall_time_sec": 600.0,
        "peak_rss_gb": 0.2,
        "history_path": run,
        "error_message": "timeout",
    }
    base.update(kwargs)
    return harness_runner.TraceRunOutcome(**base)  # type: ignore[arg-type]


@pytest.fixture
def files(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "ref.tsv").write_text(_REF, encoding="utf-8")
    (tmp_path / "run.tsv").write_text(_RUN, encoding="utf-8")
    return tmp_path


def test_stage_progress_of_a_returned_report_names_the_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A run that returned stage model numbers ended in the last of them."""
    _fake_stages(monkeypatch, ["a.mod", "b.mod", "c.mod"])
    progress = harness_runner.stage_progress
    assert progress("star", tmp_path, True, 0, (10, 20)) == (3, 2)
    assert progress("star", tmp_path, True, 0, (10, 20, 30)) == (3, 3)
    # An ok run without stage numbers is taken to have run every stage.
    assert progress("star", tmp_path, True, 0, ()) == (3, 3)
    assert progress("binary", tmp_path, True, 0, None) == (0, None)


def test_stage_progress_of_a_cut_off_run_reads_fresh_stage_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The run ended in the first stage whose output model is not new."""
    _fake_stages(monkeypatch, ["a.mod", "b.mod", "c.mod"])
    started = time.time_ns()
    old = tmp_path / "a.mod"
    old.write_text("old", encoding="utf-8")
    stale_ns = started - 10**10
    os.utime(old, ns=(stale_ns, stale_ns))
    progress = harness_runner.stage_progress
    assert progress("star", tmp_path, True, started, None) == (3, 1)
    old.write_text("new", encoding="utf-8")
    assert progress("star", tmp_path, True, started, None) == (3, 2)
    (tmp_path / "b.mod").write_text("new", encoding="utf-8")
    assert progress("star", tmp_path, True, started, None) == (3, 3)


def test_stage_progress_without_a_declared_model_is_not_established(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A stage with no output model decides nothing about the stage."""
    _fake_stages(monkeypatch, ["a.mod", "", "c.mod"])
    (tmp_path / "a.mod").write_text("new", encoding="utf-8")
    assert harness_runner.stage_progress("star", tmp_path, True, 0, None) == (
        3,
        None,
    )


def test_run_that_ended_before_the_final_stage_is_not_compared(
    files: pathlib.Path,
) -> None:
    """The partial history of stage 4 is not scored against stage 5."""
    run = files / "run.tsv"
    early = harness_runner.evaluate_case_record(
        _spec(),
        {},
        _outcome(run, stages_total=5, stage_reached=4),
        root_dir=files,
    )
    assert early.comparison is None
    assert early.verdict == "inconclusive (timeout)"
    assert early.is_undeclared_failure is False
    assert "ended in stage 4 of 5" in early.error_message
    assert "not compared" in early.error_message
    assert (early.stages_total, early.stage_reached) == (5, 4)

    unknown = harness_runner.evaluate_case_record(
        _spec(),
        {},
        _outcome(run, stages_total=5, stage_reached=None),
        root_dir=files,
    )
    assert unknown.comparison is None
    assert "stage not established" in unknown.error_message

    final = harness_runner.evaluate_case_record(
        _spec(),
        {},
        _outcome(run, stages_total=5, stage_reached=5),
        root_dir=files,
    )
    assert final.comparison is not None
    assert "not compared" not in final.error_message

    single = harness_runner.evaluate_case_record(
        _spec(), {}, _outcome(run), root_dir=files
    )
    assert single.comparison is not None


def test_stage_fields_round_trip_through_the_report_row(
    files: pathlib.Path,
) -> None:
    """A stored early row is not compared again when it is re-evaluated."""
    rec = harness_runner.evaluate_case_record(
        _spec(),
        {},
        _outcome(files / "run.tsv", stages_total=5, stage_reached=4),
        root_dir=files,
    )
    row = case_report.record_to_dict(rec)
    assert row["stages_total"] == 5
    assert row["stage_reached"] == 4
    legacy = case_report.record_to_dict(
        dataclasses.replace(rec, stages_total=0, stage_reached=None)
    )
    assert "stages_total" not in legacy
    assert "stage_reached" not in legacy


def test_wall_time_per_step_is_reported_but_not_in_a_normalized_baseline(
    files: pathlib.Path,
) -> None:
    """The field is null without a step and absent once wall time is zeroed."""
    rec = harness_runner.evaluate_case_record(
        _spec(), {}, _outcome(files / "run.tsv", steps=4), root_dir=files
    )
    assert case_report.record_to_dict(rec)["wall_time_per_step_sec"] == 150.0
    no_step = dataclasses.replace(rec, steps=0)
    assert case_report.record_to_dict(no_step)["wall_time_per_step_sec"] is None
    zeroed = dataclasses.replace(rec, wall_time_sec=0.0)
    assert "wall_time_per_step_sec" not in case_report.record_to_dict(zeroed)


def test_time_budget_section_lists_only_measured_timeouts(
    files: pathlib.Path,
) -> None:
    """A slow runner shows as a large wall time per step."""
    rec = harness_runner.evaluate_case_record(
        _spec(), {}, _outcome(files / "run.tsv", steps=4), root_dir=files
    )
    text = "\n".join(case_report._render_time_budget_section([rec]))
    assert "Wall per step" in text
    assert "150.00" in text
    zeroed = dataclasses.replace(rec, wall_time_sec=0.0)
    assert case_report._render_time_budget_section([zeroed]) == []


@pytest.fixture
def binary_mesa_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _case_verdict,
        "_mesa_stage_table",
        lambda: {("binary", "demo"): (888,)},
    )


def _binary_record(
    files: pathlib.Path, **kwargs: object
) -> harness_runner.CaseHarnessRecord:
    return harness_runner.evaluate_case_record(
        _spec("binary"),
        {},
        _outcome(
            files / "run.tsv",
            suite="binary",
            status="ok",
            termination_code="max_age",
            steps=1022,
            error_message="",
            **kwargs,
        ),
        enforce_step_counts=True,
        root_dir=files,
    )


@pytest.mark.usefixtures("binary_mesa_stages")
def test_binary_step_count_uses_star_1_when_the_report_has_it(
    files: pathlib.Path,
) -> None:
    """Star 1 stopped at 888 steps while the binary took 1022."""
    with_star1 = _binary_record(
        files, star1_final_model_number=888, star1_retries=7
    )
    assert with_star1.stage_steps == (888,)
    assert with_star1.retries == 7
    assert not with_star1.step_count_differs
    assert "step count differs" not in with_star1.error_message

    without = _binary_record(files)
    assert without.stage_steps == (1022,)
    assert without.retries == 0
    assert without.step_count_differs
    assert "step count differs from MESA at stage 1" in without.error_message
