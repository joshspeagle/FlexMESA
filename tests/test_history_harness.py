# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit and smoke integration tests for the upstream MESA history harness."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import subprocess
import time
from typing import Any

import numpy as np
import pytest

from flexmesa import _rust, cli
from flexmesa.testing import (
    baseline_regression,
    case_manifest,
    case_report,
    harness_runner,
    history_compare,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_case_manifest_covers_all_106_mesa_26_4_1_cases_and_known_gaps() -> (
    None
):
    """Manifest must cover 90 star, 8 binary, and 8 astero cases cleanly."""
    manifest = case_manifest.load_case_manifest()
    gaps = case_manifest.load_known_gaps()

    assert manifest.mesa_release == "26.4.1"
    assert manifest.zenodo_record == "19722306"
    assert len(manifest.cases) == 106

    by_suite = {
        s: [c for c in manifest.cases if c.suite == s]
        for s in case_manifest.VALID_SUITES
    }
    assert len(by_suite["star"]) == 90
    assert len(by_suite["binary"]) == 8
    assert len(by_suite["astero"]) == 8

    n_interim = sum(
        1 for c in manifest.cases if c.reference_status == "interim_0b2e6bb"
    )
    n_none = sum(
        1 for c in manifest.cases if c.reference_status == "no_reference_yet"
    )
    n_26 = sum(1 for c in manifest.cases if c.reference_status == "mesa_26_4_1")
    assert (n_26, n_interim, n_none) == (102, 0, 4)

    by_key = manifest.by_key()
    assert (
        by_key[("star", "1.3M_ms_high_Z")].required_termination
        == "log_L_upper_limit"
    )
    assert gaps[("star", "conserve_angular_momentum")].issue == "#158"

    root = case_manifest.repo_root()
    for entry in manifest.cases:
        if entry.reference_status != "no_reference_yet":
            ref_file = harness_runner.resolve_reference_history_path(
                entry.reference_path, root_dir=root
            )
            assert ref_file.is_file()
            ref_tbl = history_compare.load_history_file(ref_file)
            assert ref_tbl.mesa_version_number == "r26.4.1"

    for key, spec in gaps.items():
        assert key in by_key
        assert spec.issue.startswith("#")


def test_generated_baseline_case_table_matches_committed_file() -> None:
    """Committed notes/audit/case_table.md must match script output exactly."""
    assert cli.main(["harness", "table", "--check"]) == 0
    manifest, gaps, records = case_report.build_manifest_inventory_records()
    summary = case_report.summarize_records(manifest, records, gaps)
    assert summary["executed_cases"] == summary["total_cases"]
    assert summary["undeclared_failures"] == []
    assert summary["stale_known_gaps"] == []


def test_compare_histories_detects_first_diverging_step_and_late_spikes(
    tmp_path: pathlib.Path,
) -> None:
    """Comparator detects diverging steps, late spikes, and span truncation."""
    ref_tsv = tmp_path / "ref.tsv"
    run_log = tmp_path / "history.data"

    ref_tsv.write_text(
        "# test reference\n"
        "model_number\tstar_age\tlog_L\tlog_Teff\tlog_R\n"
        "1\t0.0\t0.00\t3.76\t0.00\n"
        "2\t1.0e7\t0.10\t3.77\t0.02\n"
        "3\t1.99999e7\t0.20\t3.78\t0.04\n"
        "4\t2.0e7\t0.30\t3.79\t0.06\n",
        encoding="utf-8",
    )
    run_log.write_text(
        "  1  2\n"
        "  version_number  compiler\n"
        '  "r26.4.1"  "rustc"\n'
        "\n"
        "  1  2  3  4  5\n"
        "  model_number  star_age  log_L  log_Teff  log_R\n"
        "  10  0.0        0.00   3.760  0.00\n"
        "  15  1.5e7      9.99   3.760  0.00\n"
        "  12  1.0e7      0.10   3.770  0.02\n"
        "  30  1.99999e7  1.50   3.780  1.20-105\n"
        "  31  1.99999e7  1.60   3.780  0.04\n"
        "  40  2.0e7      0.31   3.790  0.06\n",
        encoding="utf-8",
    )

    ref_tbl = history_compare.load_history_file(ref_tsv)
    run_tbl = history_compare.load_history_file(run_log)
    # Row model_number=15 before 12 must be scrubbed as a restart backup.
    assert run_tbl.num_rows == 5
    parsed_rust = _rust.mesa_read_log_file(str(run_log))
    assert int(parsed_rust["num_rows"]) == 6

    spec = case_manifest.CaseManifestEntry(
        suite="star",
        case="synthetic",
        reference_status="mesa_26_4_1",
        reference_path="ref.tsv",
        required_termination="max_age",
        independent_var="star_age",
        final_tolerances={"rel_star_age": 0.02, "log_L": 0.02},
        track_tolerances={"log_L": 0.10, "log_Teff": 0.01},
    )
    # Constant-x plateau self-comparison must have zero divergence.
    self_comp = history_compare.compare_histories(run_tbl, run_tbl, spec)
    assert self_comp.within_final_tolerances
    assert self_comp.within_track_tolerances
    assert self_comp.first_diverging_step is None

    # Late narrow spike at 1.99999e7 (between linspace points) must fail!
    comp = history_compare.compare_histories(ref_tbl, run_tbl, spec)
    assert comp.within_final_tolerances
    assert not comp.within_track_tolerances
    assert comp.failed_track_columns == ("log_L",)
    assert comp.first_diverging_step == 30

    # Truncated prefix covering only [1.99e7, 2.0e7] must fail span coverage.
    trunc_tsv = tmp_path / "trunc.tsv"
    trunc_tsv.write_text(
        "# truncated\n"
        "model_number\tstar_age\tlog_L\tlog_Teff\tlog_R\n"
        "1\t1.99e7\t0.20\t3.78\t0.04\n"
        "2\t2.0e7\t0.30\t3.79\t0.06\n",
        encoding="utf-8",
    )
    trunc_comp = history_compare.compare_histories(
        ref_tbl, history_compare.load_history_file(trunc_tsv), spec
    )
    assert not trunc_comp.within_track_tolerances

    # NaN / -1e99 sentinels must fail tolerances and serialize safely to JSON.
    sentinel_tsv = tmp_path / "sentinel.tsv"
    sentinel_tsv.write_text(
        "# sentinel\n"
        "model_number\tstar_age\tlog_L\tlog_Teff\tlog_R\n"
        "1\t0.0\t0.00\t3.76\t0.00\n"
        "2\t2.0e7\t-1.0e99\tNaN\t0.06\n",
        encoding="utf-8",
    )
    bad_comp = history_compare.compare_histories(
        ref_tbl, history_compare.load_history_file(sentinel_tsv), spec
    )
    assert not bad_comp.within_final_tolerances
    assert not bad_comp.within_track_tolerances
    json_str = json.dumps(
        history_compare.comparison_to_dict(bad_comp), allow_nan=False
    )
    assert "NaN" not in json_str


def test_governance_verdict_semantics_and_negative_gate_paths(
    tmp_path: pathlib.Path,
) -> None:
    """Positive and negative verdicts, missing refs, and --gate obey #145."""
    ref_file = tmp_path / "ref.tsv"
    run_file = tmp_path / "run.tsv"
    bad_file = tmp_path / "bad.tsv"
    ref_text = (
        "# test\n"
        "model_number\tstar_age\tlog_L\tlog_Teff\n"
        "1\t1.0e6\t0.0\t3.76\n"
        "2\t2.0e6\t0.1\t3.77\n"
    )
    ref_file.write_text(ref_text, encoding="utf-8")
    run_file.write_text(ref_text, encoding="utf-8")
    bad_file.write_text(
        "# bad\n"
        "model_number\tstar_age\tlog_L\tlog_Teff\n"
        "1\t1.0e6\t0.0\t3.76\n"
        "2\t2.0e6\t0.9\t3.77\n",
        encoding="utf-8",
    )

    ok_out = harness_runner.TraceRunOutcome(
        suite="star",
        case="demo",
        status="ok",
        termination_code="xa_central_lower_limit(h1)",
        steps=2,
        retries=0,
        wall_time_sec=0.1,
        peak_rss_gb=0.2,
        history_path=run_file,
    )
    err_out = harness_runner.TraceRunOutcome(
        suite="star",
        case="demo",
        status="error",
        termination_code="",
        steps=1,
        retries=0,
        wall_time_sec=0.1,
        peak_rss_gb=0.2,
        history_path=run_file,
        error_message="solver failed",
    )
    bad_tol_out = harness_runner.TraceRunOutcome(
        suite="star",
        case="demo",
        status="ok",
        termination_code="xa_central_lower_limit(h1)",
        steps=2,
        retries=0,
        wall_time_sec=0.1,
        peak_rss_gb=0.2,
        history_path=bad_file,
    )

    base_kw: dict[str, Any] = {
        "suite": "star",
        "case": "demo",
        "required_termination": "xa_central_lower_limit",
        "independent_var": "star_age",
        "final_tolerances": {"log_L": 0.02},
        "track_tolerances": {"log_L": 0.10},
    }
    no_ref = case_manifest.CaseManifestEntry(
        reference_status="no_reference_yet", reference_path="", **base_kw
    )
    assert (
        harness_runner.evaluate_case_record(
            no_ref, {}, ok_out, root_dir=tmp_path
        ).verdict
        == "no reference yet"
    )
    assert (
        harness_runner.evaluate_case_record(
            no_ref, {}, ok_out, max_steps=2, root_dir=tmp_path
        ).verdict
        == "smoke: 2 steps"
    )

    scaff = case_manifest.CaseManifestEntry(
        reference_status="mesa_26_4_1",
        reference_path="ref.tsv",
        uses_pulsation_scaffolding=True,
        scaffolding_issue="#170",
        **base_kw,
    )
    assert (
        harness_runner.evaluate_case_record(
            scaff, {}, ok_out, root_dir=tmp_path
        ).verdict
        == "scaffolding (#170)"
    )
    # Crashed scaffolding case without a known gap MUST fail undeclared!
    scaff_crash = harness_runner.evaluate_case_record(
        scaff, {}, err_out, root_dir=tmp_path
    )
    assert scaff_crash.verdict == "fail (undeclared)"
    assert scaff_crash.is_undeclared_failure

    mesa_spec = case_manifest.CaseManifestEntry(
        reference_status="mesa_26_4_1", reference_path="ref.tsv", **base_kw
    )
    assert (
        harness_runner.evaluate_case_record(
            mesa_spec, {}, ok_out, root_dir=tmp_path
        ).verdict
        == "pass"
    )
    rec_fail = harness_runner.evaluate_case_record(
        mesa_spec, {}, bad_tol_out, root_dir=tmp_path
    )
    assert rec_fail.verdict == "fail (undeclared)"
    assert rec_fail.is_undeclared_failure

    gaps = {
        ("star", "demo"): case_manifest.KnownGapSpec(
            suite="star", case="demo", issue="#152", reason="known divergence"
        )
    }
    rec_gap = harness_runner.evaluate_case_record(
        mesa_spec, gaps, bad_tol_out, root_dir=tmp_path
    )
    assert rec_gap.verdict == "declared gap (#152)"
    assert not rec_gap.is_undeclared_failure

    rec_stale = harness_runner.evaluate_case_record(
        mesa_spec, gaps, ok_out, root_dir=tmp_path
    )
    assert rec_stale.is_stale_known_gap

    # Missing reference file must raise FileNotFoundError loudly.
    missing_spec = case_manifest.CaseManifestEntry(
        reference_status="mesa_26_4_1",
        reference_path="does_not_exist.tsv",
        **base_kw,
    )
    with pytest.raises(FileNotFoundError):
        harness_runner.evaluate_case_record(
            missing_spec, gaps, ok_out, root_dir=tmp_path
        )


def test_diagnostic_run_never_passes_and_is_labelled(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that ignored unsupported controls can never be a pass (#147)."""
    ref_file = tmp_path / "ref.tsv"
    text = (
        "# test\n"
        "model_number\tstar_age\tlog_L\tlog_Teff\n"
        "1\t1.0e6\t0.0\t3.76\n"
        "2\t2.0e6\t0.1\t3.77\n"
    )
    ref_file.write_text(text, encoding="utf-8")
    run_file = tmp_path / "run.tsv"
    run_file.write_text(text, encoding="utf-8")
    spec = case_manifest.CaseManifestEntry(
        suite="star",
        case="demo",
        required_termination="xa_central_lower_limit",
        independent_var="star_age",
        final_tolerances={"log_L": 0.02},
        track_tolerances={"log_L": 0.10},
        reference_status="mesa_26_4_1",
        reference_path="ref.tsv",
    )
    rejection = (
        'invalid star input: control "<inlist>" at line 1: unsupported '
        "control(s) (3): &controls::a, &controls::b, &star_job::c (pass "
        "--allow-unsupported-controls / allow_unsupported_controls=true to "
        "opt out)"
    )
    assert harness_runner.is_unsupported_controls_rejection(rejection)
    assert harness_runner.count_unsupported_controls(rejection) == 3
    with_section = (
        "unsupported upstream namelist section(s): &colors, &stella; "
        "unsupported control(s) (25 total, showing first 20): ..."
    )
    assert harness_runner.count_unsupported_controls(with_section) == 27
    # Unknown or misspelled keys fail with or without the opt-out, so a
    # diagnostic rerun must not be attempted for them.
    assert not harness_runner.is_unsupported_controls_rejection(
        "unknown/misspelled control(s) (1): &controls::typo; "
        "unsupported control(s) (2): &controls::a, &controls::b"
    )
    assert not harness_runner.is_unsupported_controls_rejection("solver died")

    identical = harness_runner.TraceRunOutcome(
        suite="star",
        case="demo",
        status="ok",
        termination_code="xa_central_lower_limit(h1)",
        steps=2,
        retries=0,
        wall_time_sec=0.1,
        peak_rss_gb=0.2,
        history_path=run_file,
    )
    # The same history from a strict run passes.
    assert (
        harness_runner.evaluate_case_record(
            spec, {}, identical, root_dir=tmp_path
        ).verdict
        == "pass"
    )
    diag = harness_runner.TraceRunOutcome(
        suite="star",
        case="demo",
        status="ok",
        termination_code="xa_central_lower_limit(h1)",
        steps=2,
        retries=0,
        wall_time_sec=0.1,
        peak_rss_gb=0.2,
        history_path=run_file,
        controls_ignored=True,
        n_controls_ignored=3,
        strict_rejection=rejection,
    )
    rec_undeclared = harness_runner.evaluate_case_record(
        spec, {}, diag, root_dir=tmp_path
    )
    assert rec_undeclared.verdict == "fail (undeclared)"
    assert rec_undeclared.is_undeclared_failure
    assert rec_undeclared.run_mode == "diagnostic (3 controls ignored)"
    gaps = {
        ("star", "demo"): case_manifest.KnownGapSpec(
            suite="star", case="demo", issue="#152", reason="demo gap"
        )
    }
    rec_gap = harness_runner.evaluate_case_record(
        spec, gaps, diag, root_dir=tmp_path
    )
    assert rec_gap.verdict == "declared gap (#152)"
    assert not rec_gap.is_stale_known_gap
    assert rec_gap.comparison is not None
    assert rec_gap.comparison.within_track_tolerances
    assert (
        harness_runner.evaluate_case_record(
            spec, gaps, diag, max_steps=2, root_dir=tmp_path
        ).verdict
        == "smoke: 2 steps (controls ignored)"
    )

    # The stored record round-trips through the baseline JSON form.
    as_dict = case_report.record_to_dict(rec_gap)
    assert as_dict["controls_ignored"] is True
    assert as_dict["n_controls_ignored"] == 3
    assert as_dict["strict_rejection"] == rejection

    # run_case_with_diagnostic reruns only on a strict control rejection.
    calls: list[bool] = []

    def fake_run(
        suite: str,
        case_dir: pathlib.Path,
        out_target: pathlib.Path,
        **kwargs: Any,
    ) -> harness_runner.TraceRunOutcome:
        allow = bool(kwargs["allow_unsupported_controls"])
        calls.append(allow)
        if allow:
            return identical
        return harness_runner.TraceRunOutcome(
            suite=suite,
            case=case_dir.name,
            status="error",
            termination_code="",
            steps=0,
            retries=0,
            wall_time_sec=0.0,
            peak_rss_gb=0.0,
            history_path=None,
            error_message=rejection,
        )

    monkeypatch.setattr(harness_runner, "run_traced_case", fake_run)
    out = harness_runner.run_case_with_diagnostic(
        "star", tmp_path / "demo", tmp_path / "out"
    )
    assert calls == [False, True]
    assert out.controls_ignored
    assert out.n_controls_ignored == 3
    assert out.strict_rejection == rejection

    calls.clear()

    def fake_other_error(
        suite: str,
        case_dir: pathlib.Path,
        out_target: pathlib.Path,
        **kwargs: Any,
    ) -> harness_runner.TraceRunOutcome:
        calls.append(bool(kwargs["allow_unsupported_controls"]))
        return harness_runner.TraceRunOutcome(
            suite=suite,
            case=case_dir.name,
            status="error",
            termination_code="",
            steps=4,
            retries=0,
            wall_time_sec=0.0,
            peak_rss_gb=0.0,
            history_path=None,
            error_message="solver failed to converge",
        )

    monkeypatch.setattr(harness_runner, "run_traced_case", fake_other_error)
    out_err = harness_runner.run_case_with_diagnostic(
        "star", tmp_path / "demo", tmp_path / "out"
    )
    assert calls == [False]
    assert not out_err.controls_ignored


def test_guarded_subprocess_worker_and_timeout_partial_history_recovery(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guarded subprocess passes PYTHONPATH and recovers partial logs."""
    ok_dir = tmp_path / "guarded_ok"
    cdir = harness_runner.resolve_case_dir("star", "wd_cool_0.6M")
    out_ok = harness_runner.run_traced_case(
        "star",
        cdir,
        ok_dir,
        max_steps=2,
        timeout_sec=60.0,
        max_rss_gb=6.0,
        # Exercises process guards and log streaming only; the real case sets
        # controls that are not ported yet (declared gap), so the documented
        # exploratory opt-out is used and no parity is claimed.
        allow_unsupported_controls=True,
    )
    assert out_ok.status == "ok"
    assert out_ok.steps == 2
    assert out_ok.history_path is not None and out_ok.history_path.is_file()

    # Simulate a slow worker flushing 2 rows to LOGS/history.data then hanging,
    # verifying SIGKILL timeout cleanup and partial-track recovery.
    real_popen = subprocess.Popen

    def slow_flushing_popen(cmd: list[str], **kwargs: Any) -> Any:
        out_root = pathlib.Path(cmd[6])
        logs_d = out_root / "LOGS"
        logs_d.mkdir(parents=True, exist_ok=True)
        (logs_d / "history.data").write_text(
            '  1\n  version_number\n  "r26.4.1"\n\n  1  2\n'
            "  model_number  star_age\n  1  100.0\n  2  200.0\n",
            encoding="utf-8",
        )
        return real_popen(
            [cmd[0], "-c", "import time; time.sleep(10)"],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", slow_flushing_popen)
    t0 = time.monotonic()
    out_to = harness_runner.run_traced_case(
        "star",
        cdir,
        tmp_path / "guarded_timeout",
        timeout_sec=0.15,
    )
    assert time.monotonic() - t0 < 3.0
    assert out_to.status == "timeout"
    assert out_to.steps == 2
    assert out_to.history_path is not None


def test_flexmesa_trace_and_harness_run_cli_smoke(
    tmp_path: pathlib.Path,
) -> None:
    """CLI trace and harness run subcommands stream logs and enforce --gate."""
    star_out = tmp_path / "star_out"
    assert (
        cli.main(
            [
                "trace",
                "star",
                "wd_cool_0.6M",
                "--out",
                str(star_out),
                "--max-steps",
                "2",
                "--allow-unsupported-controls",
            ]
        )
        == 0
    )
    star_hist = star_out / "LOGS" / "history.data"
    assert int(_rust.mesa_read_log_file(str(star_hist))["num_rows"]) == 2

    bin_out = tmp_path / "bin_out"
    assert (
        cli.main(
            [
                "trace",
                "binary",
                "jdot_gr_check",
                "--out",
                str(bin_out),
                "--max-steps",
                "2",
                "--allow-unsupported-controls",
            ]
        )
        == 0
    )
    assert (bin_out / "binary_history.data").is_file()

    ast_out = tmp_path / "ast_out"
    assert (
        cli.main(
            [
                "trace",
                "astero",
                "surface_effects",
                "--out",
                str(ast_out),
                "--max-steps",
                "1",
                "--allow-unsupported-controls",
            ]
        )
        == 0
    )
    assert (ast_out / "LOGS" / "history.data").is_file()
    # `freqs.dat` is written by upstream's own
    # `surface_effects/src/run_star_extras.f90`, which is not ported (#148),
    # so the library must not produce it.
    assert not (ast_out / "freqs.dat").exists()

    # Smoke harness run writes Markdown + JSON with full 106-case counts.
    # `starspots` sets only `&star_job::mesa_dir` beyond the ported controls,
    # which is a supported key since #548 (`star/defaults/star_job.defaults:
    # 10-18`), so the unmodified inlists pass strict validation and the run
    # is labelled as a strict smoke run, not a diagnostic.
    md_out = tmp_path / "smoke_table.md"
    js_out = tmp_path / "smoke_table.json"
    rc_h = cli.main(
        [
            "harness",
            "run",
            "--cases",
            "starspots",
            "--max-steps",
            "2",
            "--timeout-sec",
            "60",
            "--markdown-out",
            str(md_out),
            "--json-out",
            str(js_out),
            "--gate",
        ]
    )
    assert rc_h == 0
    payload = json.loads(js_out.read_text(encoding="utf-8"))
    assert payload["summary"]["total_cases"] == 106
    assert payload["summary"]["selected_cases"] == 1
    smoke_case = payload["cases"][0]
    assert smoke_case["verdict"] == "smoke: 2 steps"
    assert smoke_case["controls_ignored"] is False
    assert smoke_case["n_controls_ignored"] == 0
    assert not smoke_case["strict_rejection"]
    assert payload["summary"]["diagnostic_cases"] == 0
    assert payload["summary"]["strict_validated_cases"] == 1

    # Undeclared failure under --gate must return exit code 1. `timing` is
    # the shortest uncapped star case (50 steps); the reference below is
    # deliberately wrong, so the run fails on termination and history.
    custom_manifest = tmp_path / "custom_manifest.toml"
    custom_gaps = tmp_path / "empty_gaps.toml"
    (tmp_path / "ref_fail.tsv").write_text(
        "# ref\nmodel_number\tstar_age\tlog_L\n1\t1.0e9\t5.0\n2\t2.0e9\t6.0\n",
        encoding="utf-8",
    )
    custom_manifest.write_text(
        '[metadata]\nmesa_release = "26.4.1"\nzenodo_record = "19722306"\n'
        '[defaults.star]\nindependent_var = "star_age"\n'
        "[defaults.star.final_tolerances]\nlog_L = 0.01\n"
        "[defaults.star.track_tolerances]\nlog_L = 0.01\n"
        '[[cases]]\nsuite = "star"\ncase = "timing"\n'
        'reference_status = "mesa_26_4_1"\nreference_path = "ref_fail.tsv"\n'
        'required_termination = "max_age"\n',
        encoding="utf-8",
    )
    custom_gaps.write_text("gaps = []\n", encoding="utf-8")
    rc_fail = cli.main(
        [
            "harness",
            "run",
            "--manifest",
            str(custom_manifest),
            "--known-gaps",
            str(custom_gaps),
            "--cases",
            "timing",
            "--gate",
        ]
    )
    assert rc_fail == 1

    # Fast mode must never be accepted under --gate (#165).
    rc_fast_gate = cli.main(
        [
            "harness",
            "run",
            "--cases",
            "wd_cool_0.6M",
            "--max-steps",
            "1",
            "--numerics-mode",
            "fast",
            "--gate",
        ]
    )
    assert rc_fast_gate == 1


def test_comparison_json_roundtrip_preserves_nan_and_signed_infinity() -> None:
    """JSON serialization must keep NaN, +inf, and -inf distinct."""
    import math

    comp = history_compare.HistoryComparison(
        independent_var="star_age",
        ref_rows=4,
        run_rows=4,
        overlap_min=0.0,
        overlap_max=10.0,
        final_diffs={
            "star_age": 0.0,
            "rel_star_age": float("nan"),
            "log_L": float("inf"),
            "log_Teff": float("-inf"),
        },
        within_final_tolerances=False,
        failed_final_columns=("rel_star_age", "log_L", "log_Teff"),
        track_deviations=(
            history_compare.TrackDeviation(
                column="log_L",
                max_abs_diff=float("inf"),
                median_abs_diff=float("nan"),
                x_at_max=0.0,
                tolerance=0.005,
                within_tolerance=False,
            ),
        ),
        within_track_tolerances=False,
        failed_track_columns=("log_L",),
        first_diverging_step=1,
        first_diverging_x=0.0,
        ref_span=10.0,
    )
    dumped = json.dumps(
        history_compare.comparison_to_dict(comp), allow_nan=False
    )
    restored = history_compare.comparison_from_dict(json.loads(dumped))
    assert math.isnan(restored.final_diffs["rel_star_age"])
    assert math.isinf(restored.final_diffs["log_L"])
    assert restored.final_diffs["log_L"] > 0.0
    assert math.isinf(restored.final_diffs["log_Teff"])
    assert restored.final_diffs["log_Teff"] < 0.0
    assert math.isnan(restored.track_deviations[0].median_abs_diff)


def test_precomputed_comparison_re_evaluates_updated_manifest_tolerances() -> (
    None
):
    """Tightening tolerances on precomputed comparisons flips verdict."""
    comp = history_compare.HistoryComparison(
        independent_var="star_age",
        ref_rows=10,
        run_rows=10,
        overlap_min=0.0,
        overlap_max=100.0,
        final_diffs={"star_age": 0.0, "rel_star_age": 0.0, "log_L": 0.002},
        within_final_tolerances=True,
        failed_final_columns=(),
        track_deviations=(
            history_compare.TrackDeviation(
                column="log_L",
                max_abs_diff=0.003,
                median_abs_diff=0.001,
                x_at_max=50.0,
                tolerance=0.01,
                within_tolerance=True,
            ),
        ),
        within_track_tolerances=True,
        failed_track_columns=(),
        first_diverging_step=None,
        first_diverging_x=None,
        ref_span=100.0,
    )
    tight_spec = case_manifest.CaseManifestEntry(
        suite="star",
        case="1.3M_ms_high_Z",
        reference_status="mesa_26_4_1",
        reference_path="dummy.data",
        required_termination="max_age",
        independent_var="star_age",
        final_tolerances={"rel_star_age": 0.001, "log_L": 0.001},
        track_tolerances={"log_L": 0.002},
    )
    outcome = harness_runner.TraceRunOutcome(
        suite="star",
        case="1.3M_ms_high_Z",
        status="ok",
        termination_code="max_age",
        steps=10,
        retries=0,
        wall_time_sec=0.1,
        peak_rss_gb=0.1,
        history_path=None,
    )
    rec = harness_runner.evaluate_case_record(
        tight_spec, {}, outcome, precomputed_comparison=comp
    )
    assert rec.verdict == "fail (undeclared)"
    assert rec.is_undeclared_failure is True
    assert rec.comparison is not None
    assert rec.comparison.failed_final_columns == ("log_L",)
    assert rec.comparison.failed_track_columns == ("log_L",)


def test_library_extras_flag_prevents_pass_verdict_until_148_lands(
    tmp_path: pathlib.Path,
) -> None:
    """Cases with uses_library_extras=True emit declared gap (#148)."""
    comp = history_compare.HistoryComparison(
        independent_var="star_age",
        ref_rows=10,
        run_rows=10,
        overlap_min=0.0,
        overlap_max=10.0,
        final_diffs={"star_age": 0.0, "rel_star_age": 0.0, "log_L": 0.0},
        within_final_tolerances=True,
        failed_final_columns=(),
        track_deviations=(),
        within_track_tolerances=True,
        failed_track_columns=(),
        first_diverging_step=None,
        first_diverging_x=None,
        ref_span=10.0,
    )
    spec = case_manifest.CaseManifestEntry(
        suite="star",
        case="rsp_BLAP",
        reference_status="mesa_26_4_1",
        reference_path="dummy.data",
        required_termination="extras_finish_step",
        independent_var="star_age",
        final_tolerances={"rel_star_age": 0.001, "log_L": 0.001},
        track_tolerances={"log_L": 0.005},
        uses_library_extras=True,
        library_extras_issue="#148",
    )
    outcome = harness_runner.TraceRunOutcome(
        suite="star",
        case="rsp_BLAP",
        status="ok",
        termination_code="extras_finish_step",
        steps=7318,
        retries=0,
        wall_time_sec=1.0,
        peak_rss_gb=0.2,
        history_path=None,
    )
    rec_undecl = harness_runner.evaluate_case_record(
        spec, {}, outcome, precomputed_comparison=comp
    )
    assert rec_undecl.verdict == "fail (undeclared)"
    assert rec_undecl.is_undeclared_failure is True

    gap = case_manifest.KnownGapSpec(
        suite="star",
        case="rsp_BLAP",
        issue="#152",
        reason="Step retry divergence (#152).",
    )
    rec_decl = harness_runner.evaluate_case_record(
        spec, {("star", "rsp_BLAP"): gap}, outcome, precomputed_comparison=comp
    )
    assert rec_decl.verdict == "declared gap (#152)"
    assert rec_decl.is_undeclared_failure is False
    assert rec_decl.is_stale_known_gap is False

    with pytest.raises(FileNotFoundError, match="missing baseline JSON"):
        case_report.build_manifest_inventory_records(
            baseline_json_path=tmp_path / "nonexistent_baseline.json"
        )


def test_harness_gate_rejects_fast_mode_from_flag_env_and_process_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`flexmesa harness run --gate` rejects fast mode via CLI, env, or API."""
    from flexmesa import _dispatch

    assert (
        cli.main(["harness", "run", "--gate", "--numerics-mode", "fast"]) == 1
    )

    monkeypatch.setenv("FLEXMESA_NUMERICS_MODE", "fast")
    assert cli.main(["harness", "run", "--gate"]) == 1
    monkeypatch.delenv("FLEXMESA_NUMERICS_MODE", raising=False)

    with _dispatch.numerics_mode_context("fast"):
        assert cli.main(["harness", "run", "--gate"]) == 1


def test_check_strict_determinism_rejects_same_arch_duplicates_and_bad_records(
    tmp_path: pathlib.Path,
) -> None:
    """`check_strict_determinism.py` enforces distinct x86_64+aarch64 JSONs."""
    import importlib.util
    import os
    import sys

    # Verify invalid FLEXMESA_NUMERICS_MODE raises ValueError cleanly on import.
    env = dict(os.environ, FLEXMESA_NUMERICS_MODE="bogus_mode")
    proc = subprocess.run(
        [sys.executable, "-c", "import flexmesa"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "ValueError" in proc.stderr
    assert "bogus_mode" in proc.stderr

    script_path = (
        case_manifest.repo_root() / "scripts" / "check_strict_determinism.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_strict_determinism", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    valid_sha = "a" * 64
    base_rec = {
        "os": "Linux",
        "arch": "x86_64",
        "numerics_mode": "strict",
        "math_vector_count": 474_062,
        "math_vectors_sha256": valid_sha,
        "evolution_suite": "star",
        "evolution_case": "1.3M_ms_high_Z",
        "evolution_max_steps": 5,
        "history_bytes": 4096,
        "history_1t_sha256": valid_sha,
        "history_4t_sha256": valid_sha,
        "thread_1_vs_4_byte_identical": True,
    }
    p_x86 = tmp_path / "x86_64.json"
    p_x86_dup = tmp_path / "x86_64_b.json"
    p_arm = tmp_path / "aarch64.json"
    p_bad = tmp_path / "bad.json"

    p_x86.write_text(json.dumps(base_rec), encoding="utf-8")
    p_x86_dup.write_text(json.dumps(base_rec), encoding="utf-8")
    arm_rec = dict(base_rec, os="Darwin", arch="arm64")
    p_arm.write_text(json.dumps(arm_rec), encoding="utf-8")
    bad_rec = dict(arm_rec, numerics_mode="fast")
    p_bad.write_text(json.dumps(bad_rec), encoding="utf-8")

    # Duplicate path must fail even with --allow-same-arch.
    assert (
        mod.main(["--compare", str(p_x86), str(p_x86), "--allow-same-arch"])
        == 1
    )
    # Two x86_64 files without --allow-same-arch must fail.
    assert mod.main(["--compare", str(p_x86), str(p_x86_dup)]) == 1
    # With --allow-same-arch, two distinct identical x86_64 records pass.
    assert (
        mod.main(["--compare", str(p_x86), str(p_x86_dup), "--allow-same-arch"])
        == 0
    )
    # Distinct x86_64 + arm64 records pass without --allow-same-arch.
    assert mod.main(["--compare", str(p_x86), str(p_arm)]) == 0
    # Non-strict record must fail.
    assert mod.main(["--compare", str(p_x86), str(p_bad)]) == 1


def test_baseline_regressions_flag_cases_that_get_worse() -> None:
    """A declared gap that stops evolving is reported (``DECISIONS #27``)."""
    _, _, base = case_report.build_manifest_inventory_records()
    by_key = {(r.suite, r.case): r for r in base}
    ok = by_key[("star", "1.3M_ms_high_Z")]
    assert ok.executed
    assert ok.steps > 0
    assert ok.termination_matches

    find = baseline_regression.find_baseline_regressions
    assert find([ok], base) == []

    stopped = dataclasses.replace(
        ok, steps=0, run_status="error", termination_matches=False
    )
    (reg,) = find([stopped], base)
    assert reg.kind == "stopped taking steps"
    assert reg.label.startswith("star/1.3M_ms_high_Z: stopped taking steps")

    lost = dataclasses.replace(
        ok, termination_matches=False, observed_termination="max_age"
    )
    (reg,) = find([lost], base)
    assert reg.kind == "lost upstream termination"
    assert "max_age" in reg.detail

    # A run cut off by its budget is inconclusive, not a regression.
    cut_off = dataclasses.replace(
        ok, run_status="timeout", termination_matches=False
    )
    assert find([cut_off], base) == []

    within = [
        r
        for r in base
        if r.comparison is not None
        and r.comparison.within_final_tolerances
        and r.comparison.within_track_tolerances
    ]
    if within:
        outside = dataclasses.replace(within[0], comparison=None)
        (reg,) = find([outside], base)
        assert reg.kind == "left tolerance"

    merged = baseline_regression.merge_records_into_baseline([lost], base)
    assert len(merged) == len(base)
    assert merged[base.index(ok)] is lost
    assert sum(1 for m, b in zip(merged, base, strict=True) if m is not b) == 1


def test_unported_extras_never_count_as_pass() -> None:
    """A case whose upstream extras logic is not executed cannot pass."""
    comp = history_compare.HistoryComparison(
        independent_var="star_age",
        ref_rows=10,
        run_rows=10,
        overlap_min=0.0,
        overlap_max=10.0,
        final_diffs={"star_age": 0.0, "rel_star_age": 0.0, "log_L": 0.0},
        within_final_tolerances=True,
        failed_final_columns=(),
        track_deviations=(),
        within_track_tolerances=True,
        failed_track_columns=(),
        first_diverging_step=None,
        first_diverging_x=None,
        ref_span=10.0,
    )
    outcome = harness_runner.TraceRunOutcome(
        suite="star",
        case="demo",
        status="ok",
        termination_code="max_age",
        steps=10,
        retries=0,
        wall_time_sec=1.0,
        peak_rss_gb=0.2,
        history_path=None,
    )

    def _verdict(status: str) -> str:
        spec = case_manifest.CaseManifestEntry(
            suite="star",
            case="demo",
            reference_status="mesa_26_4_1",
            reference_path="dummy.data",
            required_termination="max_age",
            independent_var="star_age",
            final_tolerances={"rel_star_age": 0.001, "log_L": 0.001},
            track_tolerances={},
            extras_status=status,
            extras_issue="#231" if status == "unported" else "",
        )
        rec = harness_runner.evaluate_case_record(
            spec, {}, outcome, precomputed_comparison=comp
        )
        return str(rec.verdict)

    assert _verdict("stock") == "pass"
    assert _verdict("ported") == "pass"
    assert _verdict("unported") == "fail (undeclared)"


def test_extras_status_ledger_covers_every_case_and_is_current() -> None:
    """Every manifest case has a classified, up-to-date extras status."""
    manifest = case_manifest.load_case_manifest()
    statuses = {e.extras_status for e in manifest.cases}
    assert statuses <= {"ported", "stock", "unported"}
    assert all(
        e.extras_issue.startswith("#")
        for e in manifest.cases
        if e.extras_status == "unported"
    )
    spec = importlib.util.spec_from_file_location(
        "render_extras_status",
        REPO_ROOT / "scripts" / "audit" / "render_extras_status.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--check"]) == 0


def test_binary_references_fall_back_to_the_bundle_for_another_root(
    tmp_path: pathlib.Path,
) -> None:
    """A root other than this checkout still finds the bundle's references.

    A harness run whose manifest root is another checkout must keep the
    binary LOGS1/LOGS2 tracks and the full MESA history (#409).
    """
    other_root = tmp_path / "other_checkout"
    other_root.mkdir()
    comp_paths = harness_runner.resolve_binary_component_reference_paths(
        "double_bh", root_dir=other_root
    )
    assert set(comp_paths) == {"LOGS1", "LOGS2"}
    full = harness_runner.resolve_full_reference_history_path(
        "binary", "double_bh", root_dir=other_root
    )
    assert full is not None and full.name == "binary_history.data"


def test_binary_harness_compares_component_logs1_and_logs2_tracks(
    tmp_path: pathlib.Path,
) -> None:
    """Binary harness enforces LOGS1/LOGS2 stellar tolerances with binary."""
    # Verify real cached MESA 26.4.1 component logs resolve for all 8 cases.
    manifest = case_manifest.load_case_manifest()
    bin_cases = [c for c in manifest.cases if c.suite == "binary"]
    assert len(bin_cases) == 8
    for entry in bin_cases:
        comp_paths = harness_runner.resolve_binary_component_reference_paths(
            entry.case
        )
        assert "LOGS1" in comp_paths
        if entry.case in ("double_bh", "evolve_both_stars"):
            assert "LOGS2" in comp_paths

    # Hermetic synthetic two-star binary run under tmp_path.
    bin_ref = tmp_path / "ref_bin.tsv"
    bin_ref.write_text(
        "# test binary reference\n"
        "model_number\tage\tstar_1_mass\tstar_2_mass\tperiod_days\n"
        "1\t0.0\t2.000\t1.500\t1.200\n"
        "2\t1.0e6\t1.950\t1.550\t1.250\n",
        encoding="utf-8",
    )
    star_ref_txt = (
        "# test component reference\n"
        "model_number\tstar_age\tlog_L\tlog_Teff\tlog_R\tlog_center_T\tlog_center_Rho\n"
        "1\t0.0\t0.800\t3.900\t0.200\t7.300\t1.800\n"
        "2\t1.0e6\t0.850\t3.890\t0.240\t7.310\t1.790\n"
    )
    for sub in ("LOGS1", "LOGS2"):
        rpath = tmp_path / "runs" / "binary" / "demo_bin" / sub / "history.data"
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(star_ref_txt, encoding="utf-8")

    spec = case_manifest.CaseManifestEntry(
        suite="binary",
        case="demo_bin",
        reference_status="mesa_26_4_1",
        reference_path="ref_bin.tsv",
        required_termination="max_age",
        independent_var="age",
        final_tolerances={
            "rel_age": 0.001,
            "star_1_mass": 0.001,
            "star_2_mass": 0.001,
            "period_days": 0.001,
        },
        track_tolerances={"star_1_mass": 0.002, "star_2_mass": 0.002},
        extras_status="stock",
    )

    out_ok = tmp_path / "run_ok"
    out_ok.mkdir()
    (out_ok / "binary_history.data").write_text(
        bin_ref.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for sub in ("LOGS1", "LOGS2"):
        p = out_ok / sub / "history.data"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(star_ref_txt, encoding="utf-8")

    rec_ok = harness_runner.evaluate_case_record(
        spec,
        {},
        harness_runner.TraceRunOutcome(
            suite="binary",
            case="demo_bin",
            status="ok",
            termination_code="max_age",
            steps=2,
            retries=0,
            wall_time_sec=0.1,
            peak_rss_gb=0.1,
            history_path=out_ok / "binary_history.data",
        ),
        root_dir=tmp_path,
    )
    assert rec_ok.verdict == "pass"
    assert rec_ok.is_within_tolerances
    assert set(rec_ok.component_comparisons) == {"LOGS1", "LOGS2"}

    # Diverging LOGS2 must block a pass even when binary_history.data matches.
    out_bad_comp = tmp_path / "run_bad_comp"
    out_bad_comp.mkdir()
    (out_bad_comp / "binary_history.data").write_text(
        bin_ref.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_bad_comp / "LOGS1").mkdir()
    (out_bad_comp / "LOGS1" / "history.data").write_text(
        star_ref_txt, encoding="utf-8"
    )
    (out_bad_comp / "LOGS2").mkdir()
    (out_bad_comp / "LOGS2" / "history.data").write_text(
        star_ref_txt.replace("0.850", "1.250"), encoding="utf-8"
    )
    rec_bad_comp = harness_runner.evaluate_case_record(
        spec,
        {},
        harness_runner.TraceRunOutcome(
            suite="binary",
            case="demo_bin",
            status="ok",
            termination_code="max_age",
            steps=2,
            retries=0,
            wall_time_sec=0.1,
            peak_rss_gb=0.1,
            history_path=out_bad_comp / "binary_history.data",
        ),
        root_dir=tmp_path,
    )
    assert rec_bad_comp.verdict == "fail (undeclared)"
    assert not rec_bad_comp.is_within_tolerances
    assert rec_bad_comp.comparison is not None
    assert rec_bad_comp.comparison.within_final_tolerances
    assert not rec_bad_comp.component_comparisons[
        "LOGS2"
    ].within_final_tolerances

    # Round-trip serialization preserves component failure on baseline reload.
    manifest_toml = tmp_path / "manifest.toml"
    manifest_toml.write_text(
        '[metadata]\nmesa_release = "26.4.1"\nzenodo_record = "19722306"\n'
        '[defaults.star]\nindependent_var = "star_age"\n'
        "[defaults.star.final_tolerances]\nlog_L = 0.001\n"
        "[defaults.star.track_tolerances]\nlog_L = 0.005\n"
        '[defaults.binary]\nindependent_var = "age"\n'
        "[defaults.binary.final_tolerances]\nstar_1_mass = 0.001\n"
        "[[cases]]\n"
        'suite = "binary"\ncase = "demo_bin"\n'
        'reference_status = "mesa_26_4_1"\nreference_path = "ref_bin.tsv"\n'
        'required_termination = "max_age"\n',
        encoding="utf-8",
    )
    gaps_toml = tmp_path / "gaps.toml"
    gaps_toml.write_text("gaps = []\n", encoding="utf-8")
    bjson = tmp_path / "baseline.json"
    mini_manifest = case_manifest.load_case_manifest(manifest_toml)
    bjson.write_text(
        json.dumps(
            case_report.build_deterministic_baseline_payload(
                mini_manifest, [rec_bad_comp], {}
            )
        ),
        encoding="utf-8",
    )
    _, _, reloaded = case_report.build_manifest_inventory_records(
        manifest_toml, gaps_toml, bjson
    )
    assert reloaded[0].verdict == "fail (undeclared)"
    assert not reloaded[0].is_within_tolerances

    # Missing LOGS2/history.data when reference has LOGS2 must fail.
    out_missing = tmp_path / "run_missing"
    out_missing.mkdir()
    (out_missing / "binary_history.data").write_text(
        bin_ref.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out_missing / "LOGS1").mkdir()
    (out_missing / "LOGS1" / "history.data").write_text(
        star_ref_txt, encoding="utf-8"
    )
    rec_missing = harness_runner.evaluate_case_record(
        spec,
        {},
        harness_runner.TraceRunOutcome(
            suite="binary",
            case="demo_bin",
            status="ok",
            termination_code="max_age",
            steps=2,
            retries=0,
            wall_time_sec=0.1,
            peak_rss_gb=0.1,
            history_path=out_missing / "binary_history.data",
        ),
        root_dir=tmp_path,
    )
    assert rec_missing.verdict == "fail (undeclared)"
    assert rec_missing.has_component_error
    assert "missing LOGS2/history.data" in rec_missing.error_message


def test_extract_history_run_stats_and_rel_num_zones_and_mass_loss_tables(
    tmp_path: pathlib.Path,
) -> None:
    """Verify #152/#153/#155/#159 stats, rel_num_zones, and table sections."""
    # 1. Step/retry/iter extraction across full, interval, and .mod-offset logs.
    full_ref = tmp_path / "full.tsv"
    full_ref.write_text(
        "# full every-step reference\n"
        "model_number\tstar_age\tnum_zones\tnum_retries\tnum_iters\t"
        "log_L\tmass_conv_core\tstar_mass\tlog_abs_mdot\n"
        "1\t0.0\t1000\t0\t5\t1.000\t0.200\t7.000\t-8.000\n"
        "2\t1.0e6\t1020\t1\t7\t1.010\t0.220\t6.990\t-7.800\n"
        "3\t2.0e6\t1040\t2\t6\t1.020\t0.240\t6.980\t-7.600\n",
        encoding="utf-8",
    )
    full_tbl = history_compare.load_history_file(full_ref)
    full_stats = history_compare.extract_history_run_stats(full_tbl, "star")
    assert full_stats.steps == 3
    assert full_stats.retries == 2
    assert full_stats.iters_sum == 18
    assert full_stats.mean_iters == 6.0
    assert full_stats.iters_rows == 3

    mod_offset = tmp_path / "mod_offset.tsv"
    mod_offset.write_text(
        "# started from .mod at model 10\n"
        "model_number\tstar_age\tnum_retries\tnum_iters\n"
        "11\t1.0e5\t0\t6\n"
        "20\t2.0e5\t1\t8\n"
        "30\t3.0e5\t3\t7\n"
        "387\t4.0e5\t3\t5\n",
        encoding="utf-8",
    )
    mod_stats = history_compare.extract_history_run_stats(
        history_compare.load_history_file(mod_offset), "star"
    )
    assert mod_stats.steps == 377
    assert mod_stats.retries == 3
    assert mod_stats.iters_sum == 26
    assert mod_stats.mean_iters == 6.5
    assert mod_stats.iters_rows == 4

    zero_stride = tmp_path / "zero_stride.tsv"
    zero_stride.write_text(
        "# started at model 0 with history_interval=5\n"
        "model_number\tstar_age\tnum_retries\tnum_iters\n"
        "5\t1.0e4\t0\t4\n"
        "10\t2.0e4\t0\t6\n"
        "15\t3.0e4\t1\t5\n"
        "61\t4.0e4\t1\t5\n",
        encoding="utf-8",
    )
    zs_stats = history_compare.extract_history_run_stats(
        history_compare.load_history_file(zero_stride), "star"
    )
    assert zs_stats.steps == 61
    assert zs_stats.retries == 1

    # 2. Along-track rel_num_zones tolerance, mass_conv_core, and log_abs_mdot.
    run_good = tmp_path / "run_good.tsv"
    run_good.write_text(
        "# run within rel_num_zones=0.02\n"
        "model_number\tstar_age\tnum_zones\tnum_retries\tnum_iters\t"
        "log_L\tmass_conv_core\tstar_mass\tlog_abs_mdot\n"
        "1\t0.0\t1005\t0\t5\t1.000\t0.200\t7.000\t-8.000\n"
        "2\t1.0e6\t1030\t1\t8\t1.010\t0.221\t6.989\t-7.790\n"
        "3\t2.0e6\t1050\t2\t5\t1.020\t0.240\t6.979\t-7.590\n",
        encoding="utf-8",
    )
    run_bad_nz = tmp_path / "run_bad_nz.tsv"
    run_bad_nz.write_text(
        "# run exceeding rel_num_zones=0.02 at step 2\n"
        "model_number\tstar_age\tnum_zones\tnum_retries\tnum_iters\t"
        "log_L\tmass_conv_core\tstar_mass\tlog_abs_mdot\n"
        "1\t0.0\t1000\t0\t5\t1.000\t0.200\t7.000\t-8.000\n"
        "2\t1.0e6\t1100\t1\t7\t1.010\t0.220\t6.990\t-7.800\n"
        "3\t2.0e6\t1040\t2\t6\t1.020\t0.240\t6.980\t-7.600\n",
        encoding="utf-8",
    )
    spec = case_manifest.CaseManifestEntry(
        suite="star",
        case="hot_cool_wind",
        reference_status="mesa_26_4_1",
        reference_path="full.tsv",
        required_termination="log_L_upper_limit",
        independent_var="star_age",
        final_tolerances={"rel_star_age": 0.001, "log_L": 0.001},
        track_tolerances={"log_L": 0.005, "rel_num_zones": 0.02},
        extras_status="stock",
    )
    comp_ok = history_compare.compare_histories(
        full_tbl, history_compare.load_history_file(run_good), spec
    )
    assert comp_ok.within_track_tolerances
    nz_dev = next(
        d for d in comp_ok.track_deviations if d.column == "num_zones"
    )
    assert nz_dev.max_rel_diff is not None
    assert nz_dev.max_rel_diff < 0.02
    assert any(d.column == "mass_conv_core" for d in comp_ok.track_deviations)
    assert any(d.column == "star_mass" for d in comp_ok.track_deviations)
    assert any(d.column == "log_abs_mdot" for d in comp_ok.track_deviations)

    # JSON round-trip preserves max_rel_diff and re-evaluation enforces it.
    rt = history_compare.comparison_from_dict(
        history_compare.comparison_to_dict(comp_ok)
    )
    rt_nz = next(d for d in rt.track_deviations if d.column == "num_zones")
    assert rt_nz.max_rel_diff == pytest.approx(nz_dev.max_rel_diff)
    tight_spec = dataclasses.replace(
        spec, track_tolerances={"log_L": 0.005, "rel_num_zones": 0.001}
    )
    re_eval = history_compare.re_evaluate_comparison(rt, tight_spec)
    assert not re_eval.within_track_tolerances
    assert "rel_num_zones" in re_eval.failed_track_columns

    comp_bad = history_compare.compare_histories(
        full_tbl, history_compare.load_history_file(run_bad_nz), spec
    )
    assert not comp_bad.within_track_tolerances
    assert "rel_num_zones" in comp_bad.failed_track_columns
    assert comp_bad.first_diverging_step == 2

    # 3. Render A2 case table with Steps/Retries/Iters/num_zones & #155/#159.
    hb_spec = dataclasses.replace(
        spec,
        case="hb_2M",
        track_tolerances={
            "log_L": 0.005,
            "rel_num_zones": 0.02,
            "mass_conv_core": 0.002,
        },
    )
    mini_manifest = case_manifest.HarnessManifest(
        mesa_release="26.4.1",
        zenodo_record="19722306",
        cases=(spec, hb_spec),
    )
    rec_wind = harness_runner.evaluate_case_record(
        spec,
        {},
        harness_runner.TraceRunOutcome(
            suite="star",
            case="hot_cool_wind",
            status="ok",
            termination_code="log_L_upper_limit",
            steps=3,
            retries=2,
            wall_time_sec=0.1,
            peak_rss_gb=0.1,
            history_path=run_good,
        ),
        root_dir=tmp_path,
    )
    rec_hb = harness_runner.evaluate_case_record(
        hb_spec,
        {},
        harness_runner.TraceRunOutcome(
            suite="star",
            case="hb_2M",
            status="ok",
            termination_code="log_L_upper_limit",
            steps=3,
            retries=2,
            wall_time_sec=0.1,
            peak_rss_gb=0.1,
            history_path=run_good,
        ),
        root_dir=tmp_path,
    )
    assert rec_wind.mesa_steps == 3
    assert rec_wind.mesa_retries == 2
    assert rec_wind.mesa_iters == 18
    assert rec_wind.iters == 18
    md = case_report.render_case_table_markdown(
        mini_manifest, [rec_wind, rec_hb], known_gaps={}
    )
    assert "| Steps (MESA) | Retries (MESA) | Iters (MESA) |" in md
    assert "| Iters (MESA) | Peak RSS | num_zones Track Diff |" in md
    assert "| num_zones Track Diff |" in md
    assert "3 (3) | 2 (2) | 6.0 [18] (6.0 [18]) | 0.10 GB |" in md
    assert (
        "## Convective-Core Track Comparisons (`mass_conv_core`, `#155`)" in md
    )
    assert (
        "## Mass-Loss Track Comparisons (`star_mass` and `mdot`, `#159`)" in md
    )
    assert "`log_abs_mdot`" in md
    assert "`hb_2M`" in md
    assert "`hot_cool_wind`" in md

    # 4. Full-history runs/ augmentation merges log_abs_mdot onto thinned refs.
    thin_ref = tmp_path / "thin_wind.tsv"
    thin_ref.write_text(
        "# thinned ref without log_abs_mdot\n"
        "model_number\tstar_age\tnum_zones\tnum_retries\tnum_iters\t"
        "log_L\tstar_mass\n"
        "1\t0.0\t1000\t0\t5\t1.000\t7.000\n"
        "2\t1.0e6\t1020\t1\t7\t1.010\t6.990\n"
        "3\t2.0e6\t1040\t2\t6\t1.020\t6.980\n",
        encoding="utf-8",
    )
    full_run_dir = tmp_path / "runs" / "star" / "hot_cool_wind" / "LOGS"
    full_run_dir.mkdir(parents=True)
    (full_run_dir / "history.data").write_text(
        full_ref.read_text(encoding="utf-8"), encoding="utf-8"
    )
    thin_spec = dataclasses.replace(spec, reference_path="thin_wind.tsv")
    rec_aug = harness_runner.evaluate_case_record(
        thin_spec,
        {},
        harness_runner.TraceRunOutcome(
            suite="star",
            case="hot_cool_wind",
            status="ok",
            termination_code="log_L_upper_limit",
            steps=3,
            retries=2,
            wall_time_sec=0.1,
            peak_rss_gb=0.1,
            history_path=run_good,
        ),
        root_dir=tmp_path,
    )
    assert rec_aug.comparison is not None
    assert "log_abs_mdot" in rec_aug.comparison.final_diffs
    assert any(
        d.column == "log_abs_mdot" for d in rec_aug.comparison.track_deviations
    )

    # 5. Discrete num_zones avoids false chord errors across history_interval>1.
    sparse_ref = tmp_path / "sparse_nz_ref.tsv"
    sparse_ref.write_text(
        "# MESA logged at history_interval=10 (models 1, 10, 20)\n"
        "model_number\tstar_age\tnum_zones\tlog_L\n"
        "1\t1.0e5\t800\t1.000\n"
        "10\t1.0e6\t1600\t1.010\n"
        "20\t2.0e6\t1600\t1.020\n",
        encoding="utf-8",
    )
    dense_run = tmp_path / "dense_nz_run.tsv"
    dense_run.write_text(
        "# FlexMESA logged every step; step-2 remesh jumps 800->1600\n"
        "model_number\tstar_age\tnum_zones\tlog_L\n"
        "1\t1.0e5\t800\t1.000\n"
        "2\t2.0e5\t1600\t1.001\n"
        "10\t1.0e6\t1600\t1.010\n"
        "20\t2.0e6\t1600\t1.020\n",
        encoding="utf-8",
    )
    nz_spec = case_manifest.CaseManifestEntry(
        suite="star",
        case="sparse_nz",
        reference_status="mesa_26_4_1",
        reference_path="sparse_nz_ref.tsv",
        required_termination="max_age",
        independent_var="star_age",
        final_tolerances={"rel_star_age": 0.001, "log_L": 0.001},
        track_tolerances={"log_L": 0.005, "rel_num_zones": 0.02},
    )
    nz_comp = history_compare.compare_histories(
        history_compare.load_history_file(sparse_ref),
        history_compare.load_history_file(dense_run),
        nz_spec,
    )
    assert nz_comp.within_track_tolerances
    nz_dev_s = next(
        d for d in nz_comp.track_deviations if d.column == "num_zones"
    )
    assert nz_dev_s.max_abs_diff == pytest.approx(0.0)
    assert nz_dev_s.max_rel_diff == pytest.approx(0.0)
    assert nz_comp.first_diverging_step is None


def _make_energy_harness_fixture(
    tmp_path: pathlib.Path,
) -> tuple[
    case_manifest.CaseManifestEntry,
    pathlib.Path,
    pathlib.Path,
    harness_runner.TraceRunOutcome,
]:
    ref_file = tmp_path / "energy_ref.tsv"
    run_file = tmp_path / "energy_run.tsv"
    run_file.write_text(
        "# FlexMESA run with log_rel_run_E_err\n"
        "model_number\tstar_age\tlog_L\tlog_rel_run_E_err\n"
        "1\t0.0\t0.000\t-10.400\n"
        "2\t1.0e6\t0.010\t-9.050\n"
        "3\t2.0e6\t0.020\t-8.150\n",
        encoding="utf-8",
    )
    spec = case_manifest.CaseManifestEntry(
        suite="star",
        case="hse_riemann",
        reference_status="mesa_26_4_1",
        reference_path="energy_ref.tsv",
        required_termination="max_age",
        independent_var="star_age",
        final_tolerances={"rel_star_age": 0.01, "log_L": 0.01},
        track_tolerances={"log_L": 0.01},
    )
    outcome = harness_runner.TraceRunOutcome(
        suite="star",
        case="hse_riemann",
        status="ok",
        termination_code="max_age",
        steps=3,
        retries=0,
        wall_time_sec=0.1,
        peak_rss_gb=0.1,
        history_path=run_file,
    )
    return spec, ref_file, run_file, outcome


def test_energy_conservation_log_rel_run_e_err_section_and_roundtrip(
    tmp_path: pathlib.Path,
) -> None:
    """Verify #151 log_rel_run_E_err comparison, JSON roundtrip, and table."""
    spec, ref_file, run_file, outcome = _make_energy_harness_fixture(tmp_path)
    ref_file.write_text(
        "# MESA 26.4.1 reference with log_rel_run_E_err\n"
        "model_number\tstar_age\tlog_L\tlog_rel_run_E_err\n"
        "1\t0.0\t0.000\t-10.500\n"
        "2\t1.0e6\t0.010\t-9.200\n"
        "3\t2.0e6\t0.020\t-8.400\n",
        encoding="utf-8",
    )
    comp = history_compare.compare_histories(
        history_compare.load_history_file(ref_file),
        history_compare.load_history_file(run_file),
        spec,
    )
    assert comp.final_run_values["log_rel_run_E_err"] == pytest.approx(-8.15)
    assert comp.final_ref_values["log_rel_run_E_err"] == pytest.approx(-8.40)
    assert comp.final_diffs["log_rel_run_E_err"] == pytest.approx(0.25)

    raw = history_compare.comparison_to_dict(comp)
    json_blob = json.dumps(raw, allow_nan=False)
    restored = history_compare.comparison_from_dict(json.loads(json_blob))
    assert restored.final_run_values == comp.final_run_values
    assert restored.final_ref_values == comp.final_ref_values
    assert restored.final_diffs["log_rel_run_E_err"] == pytest.approx(0.25)

    manifest = case_manifest.HarnessManifest(
        mesa_release="26.4.1",
        zenodo_record="19722306",
        cases=(spec,),
    )
    rec = harness_runner.evaluate_case_record(
        spec,
        {},
        outcome,
        root_dir=tmp_path,
        precomputed_comparison=restored,
    )
    md = case_report.render_case_table_markdown(manifest, [rec])
    assert (
        "## Energy Conservation Track Comparisons (`log_rel_run_E_err`, `#151`)"
        in md
    )
    assert "| `star` | `hse_riemann` |" in md
    assert "| `log_rel_run_E_err` | -8.150 | -8.400 | +2.500e-01 |" in md

    # Records lacking energy-error columns omit the #151 section cleanly.
    plain_comp = dataclasses.replace(
        restored,
        final_diffs={
            k: v
            for k, v in restored.final_diffs.items()
            if k != "log_rel_run_E_err"
        },
        track_deviations=tuple(
            d
            for d in restored.track_deviations
            if d.column != "log_rel_run_E_err"
        ),
        final_run_values={},
        final_ref_values={},
    )
    rec_plain = dataclasses.replace(rec, comparison=plain_comp)
    md_plain = case_report.render_case_table_markdown(manifest, [rec_plain])
    assert "Energy Conservation Track Comparisons" not in md_plain


def test_evaluate_case_record_augments_energy_columns_from_full_history(
    tmp_path: pathlib.Path,
) -> None:
    """Verify full-run energy columns augment thinned MESA reference tracks."""
    spec, ref_file, _, outcome = _make_energy_harness_fixture(tmp_path)
    ref_file.write_text(
        "# MESA 26.4.1 thinned reference without energy columns\n"
        "model_number\tstar_age\tlog_L\n"
        "1\t0.0\t0.000\n"
        "2\t1.0e6\t0.010\n"
        "3\t2.0e6\t0.020\n",
        encoding="utf-8",
    )
    full_hist = (
        tmp_path / "runs" / "star" / "hse_riemann" / "LOGS" / "history.data"
    )
    full_hist.parent.mkdir(parents=True, exist_ok=True)
    full_hist.write_text(
        "# MESA 26.4.1 full history with log_rel_run_E_err\n"
        "model_number\tstar_age\tlog_L\tlog_rel_run_E_err\n"
        "1\t0.0\t0.000\t-10.500\n"
        "2\t1.0e6\t0.010\t-9.200\n"
        "3\t2.0e6\t0.020\t-8.400\n",
        encoding="utf-8",
    )
    rec = harness_runner.evaluate_case_record(
        spec,
        {},
        outcome,
        root_dir=tmp_path,
    )
    assert rec.comparison is not None
    comp = rec.comparison
    assert comp.final_ref_values["log_rel_run_E_err"] == pytest.approx(-8.40)
    assert comp.final_run_values["log_rel_run_E_err"] == pytest.approx(-8.15)
    assert comp.final_diffs["log_rel_run_E_err"] == pytest.approx(0.25)


def test_render_case_table_markdown_peak_rss_formatting(
    tmp_path: pathlib.Path,
) -> None:
    """Peak RSS renders `X.XX GB` for positive runs and `-` for 0/unexecuted."""
    spec, ref_file, _, outcome = _make_energy_harness_fixture(tmp_path)
    ref_file.write_text(
        "# MESA 26.4.1 reference\n"
        "model_number\tstar_age\tlog_L\n"
        "1\t0.0\t0.000\n"
        "2\t1.0e6\t0.010\n"
        "3\t2.0e6\t0.020\n",
        encoding="utf-8",
    )
    manifest = case_manifest.HarnessManifest(
        mesa_release="26.4.1",
        zenodo_record="19722306",
        cases=(spec,),
    )
    rec_live = harness_runner.evaluate_case_record(
        spec,
        {},
        dataclasses.replace(outcome, peak_rss_gb=3.847),
        root_dir=tmp_path,
    )
    rec_baseline = dataclasses.replace(rec_live, peak_rss_gb=0.0)
    rec_unexecuted = dataclasses.replace(
        rec_live, executed=False, peak_rss_gb=2.5
    )

    md_live = case_report.render_case_table_markdown(manifest, [rec_live])
    md_base = case_report.render_case_table_markdown(manifest, [rec_baseline])
    md_skip = case_report.render_case_table_markdown(manifest, [rec_unexecuted])

    assert "| Iters (MESA) | Peak RSS | num_zones Track Diff |" in md_live
    assert "| 3 (3) | 0 (-) | - | 3.85 GB | - |" in md_live
    assert "| 3 (3) | 0 (-) | - | - | - |" in md_base
    assert "| 2.50 GB |" not in md_skip
    assert "| - (3) | - | - | - | - |" in md_skip


def test_flexmesa_import_respects_worker_rust_only_env(
    tmp_path: pathlib.Path,
) -> None:
    """`FLEXMESA_WORKER_RUST_ONLY=1` skips JAX/plotting while keeping checks."""
    import os
    import sys

    src_root = str(case_manifest.repo_root() / "src")
    base_env = dict(os.environ, PYTHONPATH=src_root)
    base_env.pop("FLEXMESA_WORKER_RUST_ONLY", None)

    probe = (
        "import json, sys, flexmesa; "
        "print(json.dumps({"
        "'jax': 'jax' in sys.modules, "
        "'mpl': 'matplotlib' in sys.modules, "
        "'scipy': 'scipy' in sys.modules, "
        "'rust': 'flexmesa._rust' in sys.modules"
        "}))"
    )
    proc_full = subprocess.run(
        [sys.executable, "-c", probe],
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    full_mods = json.loads(proc_full.stdout)
    assert full_mods["jax"] is True
    assert full_mods["rust"] is True

    rust_env = dict(base_env, FLEXMESA_WORKER_RUST_ONLY="1")
    proc_rust = subprocess.run(
        [sys.executable, "-c", probe],
        env=rust_env,
        capture_output=True,
        text=True,
        check=True,
    )
    rust_mods = json.loads(proc_rust.stdout)
    assert rust_mods["jax"] is False
    assert rust_mods["mpl"] is False
    assert rust_mods["scipy"] is False
    assert rust_mods["rust"] is True

    bad_env = dict(rust_env, FLEXMESA_NUMERICS_MODE="bogus_mode")
    proc_bad = subprocess.run(
        [sys.executable, "-c", "import flexmesa"],
        env=bad_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_bad.returncode != 0
    assert "ValueError" in proc_bad.stderr
    assert "bogus_mode" in proc_bad.stderr

    # Verify _build_worker_env only enables FLEXMESA_WORKER_RUST_ONLY for
    # rust_only=True and explicitly clears it otherwise.
    env_rust = harness_runner._build_worker_env(
        tmp_path, "strict", rust_only=True
    )
    assert env_rust["FLEXMESA_WORKER_RUST_ONLY"] == "1"
    env_orig = os.environ.get("FLEXMESA_WORKER_RUST_ONLY")
    os.environ["FLEXMESA_WORKER_RUST_ONLY"] = "1"
    try:
        env_extras = harness_runner._build_worker_env(
            tmp_path, "strict", rust_only=False
        )
    finally:
        if env_orig is None:
            os.environ.pop("FLEXMESA_WORKER_RUST_ONLY", None)
        else:
            os.environ["FLEXMESA_WORKER_RUST_ONLY"] = env_orig
    assert "FLEXMESA_WORKER_RUST_ONLY" not in env_extras

    # Verify _has_active_extras_handler classifies empty, no-op, active, and
    # malformed factory specs cleanly.
    amj_dir = harness_runner.resolve_case_dir("star", "accreted_material_j")
    cam_dir = harness_runner.resolve_case_dir(
        "star", "conserve_angular_momentum"
    )
    unserved_dir = harness_runner.resolve_case_dir("star", "c13_pocket")
    assert not harness_runner._has_active_extras_handler("star", amj_dir, None)
    assert not harness_runner._has_active_extras_handler(
        "star", unserved_dir, "tests.case_extras:resolve_case_extras"
    )
    assert harness_runner._has_active_extras_handler(
        "star", amj_dir, "tests.case_extras:resolve_case_extras"
    )
    assert harness_runner._has_active_extras_handler(
        "star", cam_dir, "tests.case_extras:resolve_case_extras"
    )
    assert harness_runner._has_active_extras_handler(
        "star", amj_dir, "bad_factory_spec"
    )


def test_compare_histories_sparse_reference_is_judged_only_at_its_rows(
    tmp_path: pathlib.Path,
) -> None:
    """A reference written every 10 models is compared at its rows (#459).

    The run follows a curved track that agrees with the reference at every
    reference row; the chord between the sparse rows departs from it by far
    more than the tolerance, which must not count as a divergence. A run
    that differs at a reference row must still fail.
    """
    ages = np.linspace(0.0, 1.0e7, 31)
    curve = np.sin(np.pi * ages / 1.0e7)
    ref_rows = range(0, 31, 10)
    ref_tsv = tmp_path / "ref.tsv"
    ref_tsv.write_text(
        "# sparse reference\nmodel_number\tstar_age\tlog_L\n"
        + "".join(
            f"{i + 1}\t{float(ages[i])!r}\t{float(curve[i])!r}\n"
            for i in ref_rows
        ),
        encoding="utf-8",
    )
    ref_tbl = history_compare.load_history_file(ref_tsv)

    def run_table(
        values: np.ndarray, name: str
    ) -> history_compare.HistoryTable:
        path = tmp_path / name
        path.write_text(
            "# run\nmodel_number\tstar_age\tlog_L\n"
            + "".join(
                f"{i + 1}\t{float(ages[i])!r}\t{float(values[i])!r}\n"
                for i in range(31)
            ),
            encoding="utf-8",
        )
        return history_compare.load_history_file(path)

    spec = case_manifest.CaseManifestEntry(
        suite="star",
        case="synthetic",
        reference_status="mesa_26_4_1",
        reference_path="ref.tsv",
        required_termination="max_age",
        independent_var="star_age",
        final_tolerances={"rel_star_age": 0.02},
        track_tolerances={"log_L": 1.0e-3},
    )
    comp = history_compare.compare_histories(
        ref_tbl, run_table(curve, "run.tsv"), spec
    )
    assert comp.within_track_tolerances
    assert comp.first_diverging_step is None

    shifted = curve.copy()
    shifted[20] += 2.0e-3
    bad = history_compare.compare_histories(
        ref_tbl, run_table(shifted, "bad.tsv"), spec
    )
    assert bad.failed_track_columns == ("log_L",)
    assert bad.first_diverging_step == 21
