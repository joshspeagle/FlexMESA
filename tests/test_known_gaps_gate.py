# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit and repository-gate tests for `flexmesa.testing.known_gaps_gate`."""

from __future__ import annotations

import json
import pathlib

import pytest

from flexmesa.testing import _dashboard_github, case_manifest, known_gaps_gate


def _write_fake_repo(
    tmp_path: pathlib.Path,
    *,
    rust_src: str = '#[test]\n#[ignore = "gap (#161)"]\nfn test_alpha() {}\n',
    py_src: str = (
        "import pytest\n\n"
        '@pytest.mark.xfail(strict=True, reason="gap (#153)")\n'
        "def test_beta() -> None:\n"
        "    assert False\n"
    ),
    expect_fail_rust: bool = True,
    issue_states: dict[int, str] | None = None,
) -> pathlib.Path:
    """Create a minimal synthetic repository tree for gate tests."""
    (tmp_path / "crates" / "demo" / "tests").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "benchmarks" / "history_harness").mkdir(parents=True)
    (tmp_path / "benchmarks" / "dashboard").mkdir(parents=True)
    (tmp_path / "crates" / "demo" / "tests" / "demo.rs").write_text(
        rust_src, encoding="utf-8"
    )
    (tmp_path / "tests" / "test_demo.py").write_text(py_src, encoding="utf-8")

    states = (
        issue_states if issue_states is not None else {161: "open", 153: "open"}
    )
    snapshot = {
        "tracker_issue": 140,
        "backlog_issue": 169,
        "phases_140": [
            {
                "phase": "Phase C",
                "issues": [
                    {"number": n, "title": f"Issue {n}", "state": st}
                    for n, st in sorted(states.items())
                ],
            }
        ],
        "deferred_169": [],
    }
    (tmp_path / "benchmarks" / "dashboard" / "github_snapshot.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    toml_text = (
        "[[gaps]]\n"
        'suite = "star"\n'
        'case = "synthetic_case"\n'
        'issue = "#161"\n'
        'reason = "Expected history gap."\n\n'
        "[[test_gaps]]\n"
        'kind = "rust_ignore"\n'
        'crate = "demo"\n'
        'file = "crates/demo/tests/demo.rs"\n'
        'test_name = "test_alpha"\n'
        'issue = "#161"\n'
        'role = "physics_gap"\n'
        f"expect_fail_when_run = {str(expect_fail_rust).lower()}\n"
        'reason = "Expected outcome."\n\n'
        "[[test_gaps]]\n"
        'kind = "python_xfail"\n'
        'file = "tests/test_demo.py"\n'
        'test_name = "test_beta"\n'
        'issue = "#153"\n'
        'role = "physics_gap"\n'
        "expect_fail_when_run = true\n"
        'reason = "Expected xfail."\n'
    )
    (
        tmp_path / "benchmarks" / "history_harness" / "known_gaps.toml"
    ).write_text(toml_text, encoding="utf-8")
    return tmp_path


def test_live_repo_known_test_gaps_gate_passes() -> None:
    """Verify bijective correspondence across live repo markers and manifest."""
    root = case_manifest.repo_root()
    single_ledger = root / "benchmarks" / "history_harness" / "known_gaps.toml"
    assert single_ledger.is_file()
    res = known_gaps_gate.verify_known_gaps_gate(
        root, fail_on_imperative_skips=True
    )
    assert res.declared_count == res.discovered_count
    assert res.declared_count >= 1
    assert res.ok, known_gaps_gate.render_markdown_summary(res)


def test_require_zero_skips_defaults_on_and_respects_explicit_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Verify FLEXMESA_REQUIRE_ZERO_SKIPS defaults to active unless disabled."""
    monkeypatch.delenv("FLEXMESA_REQUIRE_ZERO_SKIPS", raising=False)
    monkeypatch.delenv("MESA_DIR", raising=False)
    assert known_gaps_gate.require_zero_skips_enabled() is True

    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(AssertionError, match="FLEXMESA_REQUIRE_ZERO_SKIPS=0"):
        known_gaps_gate.require_upstream_file(missing, label="golden output")

    monkeypatch.setenv("FLEXMESA_REQUIRE_ZERO_SKIPS", "0")
    assert known_gaps_gate.require_zero_skips_enabled() is False
    assert (
        known_gaps_gate.require_upstream_file(missing, label="golden output")
        is False
    )

    monkeypatch.setenv("MESA_DIR", str(tmp_path))
    with pytest.raises(AssertionError, match="golden output"):
        known_gaps_gate.require_upstream_file(missing, label="golden output")

    present = tmp_path / "present.txt"
    present.write_text("ok", encoding="utf-8")
    assert known_gaps_gate.require_upstream_file(present) is True

    monkeypatch.delenv("FLEXMESA_REQUIRE_ZERO_SKIPS", raising=False)
    monkeypatch.setenv("MESA_DIR", str(tmp_path / "empty_mesa"))
    with pytest.raises(AssertionError, match="FLEXMESA_REQUIRE_ZERO_SKIPS=0"):
        known_gaps_gate.require_mesa_test_inputs_dir()

    monkeypatch.setenv("FLEXMESA_REQUIRE_ZERO_SKIPS", "0")
    monkeypatch.delenv("MESA_DIR", raising=False)
    monkeypatch.setattr(
        "flexmesa.data.fetch.resolve_mesa_test_inputs_dir",
        lambda: tmp_path / "missing_cache",
    )
    assert known_gaps_gate.require_mesa_test_inputs_dir() is None

    (tmp_path / "star" / "test_suite").mkdir(parents=True)
    monkeypatch.setenv("MESA_DIR", str(tmp_path))
    assert known_gaps_gate.require_mesa_test_inputs_dir() == tmp_path


def test_synthetic_gate_detects_undeclared_and_stale_markers(
    tmp_path: pathlib.Path,
) -> None:
    """Reject undeclared source markers and stale manifest entries."""
    repo = _write_fake_repo(tmp_path)
    res_ok = known_gaps_gate.verify_known_gaps_gate(repo)
    assert res_ok.ok

    py_path = repo / "tests" / "test_demo.py"
    existing = py_path.read_text(encoding="utf-8")
    py_path.write_text(
        f'{existing}\n@pytest.mark.skip(reason="extra (#153)")\n'
        "def test_gamma() -> None:\n    pass\n",
        encoding="utf-8",
    )
    res_undec = known_gaps_gate.verify_known_gaps_gate(repo)
    assert not res_undec.ok
    assert any("test_gamma" in s for s in res_undec.undeclared_markers)

    (repo / "crates" / "demo" / "tests" / "demo.rs").write_text(
        "#[test]\nfn test_alpha() {}\n", encoding="utf-8"
    )
    res_stale = known_gaps_gate.verify_known_gaps_gate(repo)
    assert not res_stale.ok
    assert any("test_alpha" in s for s in res_stale.stale_declarations)


def test_synthetic_gate_rejects_closed_issue_and_unexpected_ignored_outcomes(
    tmp_path: pathlib.Path,
) -> None:
    """Reject closed issues, unexpected passes, failures, and missing runs."""
    repo = _write_fake_repo(tmp_path, issue_states={161: "closed", 153: "open"})
    res_closed = known_gaps_gate.verify_known_gaps_gate(repo)
    assert not res_closed.ok
    assert len(res_closed.closed_issue_violations) == 1

    repo_open = _write_fake_repo(
        tmp_path / "open_case",
        expect_fail_rust=True,
        issue_states={161: "open", 153: "open"},
    )
    log_fail = tmp_path / "ignored_fail.log"
    log_fail.write_text(
        "test nested::test_alpha ... FAILED\n", encoding="utf-8"
    )
    res_expected_fail = known_gaps_gate.verify_known_gaps_gate(
        repo_open,
        cargo_ignored_logs=[log_fail],
        require_all_rust_ignores_executed=True,
    )
    assert res_expected_fail.ok

    log_pass = tmp_path / "ignored_pass.log"
    log_pass.write_text("test nested::test_alpha ... ok\n", encoding="utf-8")
    res_stale_pass = known_gaps_gate.verify_known_gaps_gate(
        repo_open, cargo_ignored_logs=[log_pass]
    )
    assert not res_stale_pass.ok
    assert any("stale_known_gap" in s for s in res_stale_pass.unexpected_passes)
    # DECISIONS #81: an improvement alone is told apart from a regression.
    assert res_stale_pass.only_unexpected_passes

    # When expect_fail_when_run = false, FAILED must fail the gate.
    repo_pass_expected = _write_fake_repo(
        tmp_path / "pass_expected", expect_fail_rust=False
    )
    res_unexpected_fail = known_gaps_gate.verify_known_gaps_gate(
        repo_pass_expected, cargo_ignored_logs=[log_fail]
    )
    assert not res_unexpected_fail.ok
    assert any(
        "unexpected_failure" in s
        for s in res_unexpected_fail.unexpected_failures
    )

    # Requiring all rust_ignores executed must reject empty/missing logs.
    log_empty = tmp_path / "ignored_empty.log"
    log_empty.write_text("error: could not compile\n", encoding="utf-8")
    res_missing = known_gaps_gate.verify_known_gaps_gate(
        repo_pass_expected,
        cargo_ignored_logs=[log_empty],
        require_all_rust_ignores_executed=True,
    )
    assert not res_missing.ok
    assert any("test_alpha" in s for s in res_missing.missing_ignored_runs)


def test_fail_on_imperative_skips_and_live_github_pagination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Verify --fail-on-imperative-skips and paginated GitHub API fallback."""
    py_with_skip = (
        "import pytest\n\n"
        '@pytest.mark.xfail(strict=True, reason="gap (#153)")\n'
        "def test_beta() -> None:\n"
        "    if True:\n"
        '        pytest.skip("runtime skip")\n'
    )
    repo = _write_fake_repo(tmp_path, py_src=py_with_skip)

    res_soft = known_gaps_gate.verify_known_gaps_gate(repo)
    assert res_soft.ok
    assert len(res_soft.imperative_python_skips) == 1

    res_strict = known_gaps_gate.verify_known_gaps_gate(
        repo, fail_on_imperative_skips=True
    )
    assert not res_strict.ok
    assert "FAIL" in known_gaps_gate.render_markdown_summary(res_strict)
    assert (
        known_gaps_gate.main(
            ["--repo-root", str(repo), "--fail-on-imperative-skips"]
        )
        == 1
    )

    pages: dict[int, list[dict[str, object]]] = {
        1: [{"number": i, "state": "open"} for i in range(1, 100)]
        + [{"number": 999, "state": "open", "pull_request": {}}],
        2: [{"number": 161, "state": "closed"}],
    }

    def fake_gh_api(endpoint: str) -> list[dict[str, object]]:
        if endpoint.endswith("&page=1"):
            return pages[1]
        if endpoint.endswith("&page=2"):
            return pages[2]
        return []

    monkeypatch.setattr(_dashboard_github, "_gh_api_json", fake_gh_api)
    states = known_gaps_gate.load_issue_states(repo, fetch_github_api=True)
    assert states[161] == "closed"
    assert 999 not in states


def test_stale_gap_status_only_when_unexpected_passes_are_the_sole_finding(
    tmp_path: pathlib.Path,
) -> None:
    """An ignored test that now passes exits with --stale-gap-status (3).

    Any other finding, such as a declared failure that fails when it is
    expected to pass, still exits 1 (DECISIONS #81).
    """
    repo = _write_fake_repo(
        tmp_path / "improve",
        expect_fail_rust=True,
        issue_states={161: "open", 153: "open"},
    )
    log_pass = tmp_path / "pass.log"
    log_pass.write_text("test nested::test_alpha ... ok\n", encoding="utf-8")
    args = [
        "--repo-root",
        str(repo),
        "--cargo-ignored-log",
        str(log_pass),
        "--stale-gap-status",
        "3",
    ]
    assert known_gaps_gate.main(args) == 3

    repo_bad = _write_fake_repo(tmp_path / "regress", expect_fail_rust=False)
    log_fail = tmp_path / "fail.log"
    log_fail.write_text(
        "test nested::test_alpha ... FAILED\n", encoding="utf-8"
    )
    args_bad = [
        "--repo-root",
        str(repo_bad),
        "--cargo-ignored-log",
        str(log_fail),
        "--stale-gap-status",
        "3",
    ]
    assert known_gaps_gate.main(args_bad) == 1
