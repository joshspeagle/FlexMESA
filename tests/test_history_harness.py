# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit and smoke integration tests for the upstream MESA history harness."""

from __future__ import annotations

import json
import pathlib
import subprocess
import time
from typing import Any

import pytest

from flexmesa import _rust, cli
from flexmesa.testing import (
    case_manifest,
    harness_runner,
    history_compare,
)


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
    assert (n_26, n_interim, n_none) == (0, 9, 97)

    by_key = manifest.by_key()
    assert (
        by_key[("star", "1.3M_ms_high_Z")].required_termination
        == "log_L_upper_limit"
    )
    assert gaps[("star", "hb_2M")].issue == "#155"

    root = case_manifest.repo_root()
    for entry in manifest.cases:
        if entry.reference_status != "no_reference_yet":
            assert (root / entry.reference_path).is_file()

    for key, spec in gaps.items():
        assert key in by_key
        assert spec.issue.startswith("#")


def test_generated_baseline_case_table_matches_committed_file() -> None:
    """Committed notes/audit/case_table.md must match script output exactly."""
    assert cli.main(["harness", "table", "--check"]) == 0


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
            ]
        )
        == 0
    )
    assert (ast_out / "LOGS" / "history.data").is_file()
    assert (ast_out / "freqs.dat").is_file()

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
    assert payload["cases"][0]["verdict"] == "smoke: 2 steps"

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
