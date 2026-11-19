# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and smoke integration tests for the upstream MESA history harness."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import subprocess
import time
from typing import Any

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
    assert gaps[("star", "hb_2M")].issue == "#155"

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
    md_out = tmp_path / "smoke_table.md"
    js_out = tmp_path / "smoke_table.json"
    rc_h = cli.main(
        [
            "harness",
            "run",
            "--cases",
            "wd_cool_0.6M",
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
    # Strict validation rejects the unmodified upstream inlists (#147), so
    # the harness reruns the case as a diagnostic and labels it as such.
    smoke_case = payload["cases"][0]
    assert smoke_case["verdict"] == "smoke: 2 steps (controls ignored)"
    assert smoke_case["controls_ignored"] is True
    assert smoke_case["n_controls_ignored"] > 0
    assert "unsupported control" in smoke_case["strict_rejection"]
    assert payload["summary"]["diagnostic_cases"] == 1
    assert payload["summary"]["strict_validated_cases"] == 0
    assert "diagnostic (" in md_out.read_text(encoding="utf-8")

    # Undeclared failure under --gate must return exit code 1.
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
        '[[cases]]\nsuite = "star"\ncase = "custom_colors"\n'
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
            "custom_colors",
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
