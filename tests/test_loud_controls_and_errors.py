# SPDX-License-Identifier: GPL-3.0-or-later
"""Issue #147 fail-loudly CLI, report, and Python binding tests."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import pytest

from flexmesa import astero, binary, star
from flexmesa.testing import harness_runner, known_gaps_gate, trace_cli


def test_cli_allow_unsupported_controls_defaults_false_and_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `--allow-unsupported-controls` defaults to False and forwards."""
    trace_ns = trace_cli._build_trace_parser().parse_args(
        ["star", "make_zams", "--out", "/unused_out"]
    )
    assert trace_ns.allow_unsupported_controls is False
    trace_opt = trace_cli._build_trace_parser().parse_args(
        [
            "star",
            "make_zams",
            "--out",
            "/unused_out",
            "--allow-unsupported-controls",
        ]
    )
    assert trace_opt.allow_unsupported_controls is True

    h_ns = trace_cli._build_harness_parser().parse_args(["run"])
    assert h_ns.allow_unsupported_controls is False
    h_opt = trace_cli._build_harness_parser().parse_args(
        ["run", "--allow-unsupported-controls"]
    )
    assert h_opt.allow_unsupported_controls is True

    seen_flags: list[bool] = []
    with tempfile.TemporaryDirectory() as tmp:
        fake_case = pathlib.Path(tmp) / "fake_case"
        fake_case.mkdir()
        monkeypatch.setattr(
            harness_runner,
            "resolve_case_dir",
            lambda _s, _c, mesa_dir=None: fake_case,
        )

        def _fake_run(
            suite: str,
            case_dir: pathlib.Path,
            out_target: pathlib.Path,
            **kwargs: object,
        ) -> harness_runner.TraceRunOutcome:
            seen_flags.append(bool(kwargs.get("allow_unsupported_controls")))
            return harness_runner.TraceRunOutcome(
                suite=suite,
                case=case_dir.name,
                status="ok",
                termination_code="max_model_number",
                steps=1,
                retries=0,
                wall_time_sec=0.01,
                peak_rss_gb=0.1,
                history_path=None,
            )

        monkeypatch.setattr(harness_runner, "run_traced_case", _fake_run)
        rc0 = trace_cli.trace_main(
            ["star", str(fake_case), "--out", str(pathlib.Path(tmp) / "o0")]
        )
        rc1 = trace_cli.trace_main(
            [
                "star",
                str(fake_case),
                "--out",
                str(pathlib.Path(tmp) / "o1"),
                "--allow-unsupported-controls",
            ]
        )
        assert rc0 == 0 and rc1 == 0
    assert seen_flags == [False, True]


def test_unsupported_controls_audit_report_matches_checked_in_files() -> None:
    """Verify `report_unsupported_controls.py --check` passes when enabled."""
    mesa_base = known_gaps_gate.require_mesa_test_inputs_dir()
    if mesa_base is None:
        return
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "audit" / "report_unsupported_controls.py"
    res = subprocess.run(
        [
            sys.executable,
            str(script),
            "--check",
            "--mesa-dir",
            str(mesa_base),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr + res.stdout


def test_python_inlist_and_runner_loud_failures_and_opt_out() -> None:
    """Verify strict inlist/option rejection and explicit Python opt-out."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)

        bad_opt = root / "inlist_simple_phot"
        bad_opt.write_text(
            "&controls\n  atm_option = 'simple_photosphere'\n/\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="simple_photosphere"):
            star.parse_inlist_bundle(bad_opt)

        unimpl = root / "inlist_unimpl"
        unimpl.write_text(
            "&controls\n  initial_mass = 1.0\n"
            "  write_model_with_profile = .true.\n/\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unsupported control"):
            star.parse_inlist_bundle(unimpl)
        bundle = star.parse_inlist_bundle(
            unimpl, allow_unsupported_controls=True
        )
        assert bundle["initial_mass"] == 1.0

        photo_ok = root / "inlist_photo_ok"
        photo_ok.write_text(
            "&controls\n  initial_mass = 1.0\n  photo_interval = 5\n/\n",
            encoding="utf-8",
        )
        assert star.parse_inlist_bundle(photo_ok)["initial_mass"] == 1.0

        pg_ok = root / "inlist_pgstar_ok"
        pg_ok.write_text(
            "&star_job\n  pgstar_flag = .true.\n/\n"
            "&pgstar\n  pgstar_interval = 1\n/\n"
            "&controls\n  initial_mass = 1.0\n/\n",
            encoding="utf-8",
        )
        assert star.parse_inlist_bundle(pg_ok)["initial_mass"] == 1.0

        pg_bad = root / "inlist_pgstar_bad"
        pg_bad.write_text(
            "&star_job\n  pgstar_flag = .true.\n/\n"
            "&pgstar\n  not_a_real_pgstar_key = 1\n/\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="not_a_real_pgstar_key"):
            star.parse_inlist_bundle(pg_bad)

        # MESA reads &pgstar only when pgstar_flag is true, so a stale key in
        # an inactive block does not stop the run.
        pg_inactive = root / "inlist_pgstar_inactive"
        pg_inactive.write_text(
            "&pgstar\n  not_a_real_pgstar_key = 1\n/\n"
            "&controls\n  initial_mass = 1.0\n/\n",
            encoding="utf-8",
        )
        assert star.parse_inlist_bundle(pg_inactive)["initial_mass"] == 1.0

        bin_case = root / "bin_case"
        bin_case.mkdir()
        (bin_case / "inlist").write_text(
            "&binary_controls\n"
            "  m1 = 10.0\n"
            "  m2 = 8.0\n"
            "  photo_interval = 10\n"
            "/\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="photo_interval"):
            binary.run_binary_test_suite_case(bin_case, max_steps_cap=1)
        ast_case = root / "ast_case"
        ast_case.mkdir()
        (ast_case / "rn").write_text(
            "#!/bin/sh\ndo_one inlist final.mod\n",
            encoding="utf-8",
        )
        (ast_case / "inlist").write_text(
            "&controls\n  initial_mass = 1.0\n  max_model_number = 1\n/\n",
            encoding="utf-8",
        )
        (ast_case / "inlist_astero_search_controls").write_text(
            "&astero_search_controls\n  correction_scheme = 'bg14_1'\n/\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="correction_scheme"):
            astero.run_astero_test_suite_case(ast_case, max_steps_per_stage=1)
        with pytest.raises(ValueError, match="astero"):
            harness_runner.run_trace_inprocess(
                "astero",
                ast_case,
                root / "ast_out",
                max_steps=1,
                extras=object(),
            )


def test_parse_inlist_bundle_exposes_remove_center_entropy_keys(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `star.parse_inlist_bundle` exposes the 4 entropy-cut keys."""
    inlist_default = tmp_path / "inlist_default"
    inlist_default.write_text(
        "&controls\n  initial_mass = 1.0\n/\n",
        encoding="utf-8",
    )
    default_bundle = star.parse_inlist_bundle(str(inlist_default))
    assert default_bundle["remove_initial_center_by_entropy"] == 0.0
    assert default_bundle["remove_center_by_entropy"] == 0.0
    assert default_bundle["remove_center_set_zero_v_center"] is False
    assert default_bundle["remove_center_adjust_L_center"] is True

    inlist_custom = tmp_path / "inlist_custom"
    inlist_custom.write_text(
        "&star_job\n"
        "  remove_initial_center_by_entropy = 4.0\n"
        "  remove_center_by_entropy = 3.5\n"
        "  remove_center_set_zero_v_center = .true.\n"
        "  remove_center_adjust_L_center = .false.\n"
        "/\n"
        "&controls\n  initial_mass = 1.0\n/\n",
        encoding="utf-8",
    )
    custom_bundle = star.parse_inlist_bundle(str(inlist_custom))
    assert custom_bundle["remove_initial_center_by_entropy"] == 4.0
    assert custom_bundle["remove_center_by_entropy"] == 3.5
    assert custom_bundle["remove_center_set_zero_v_center"] is True
    assert custom_bundle["remove_center_adjust_L_center"] is False


def test_extras_missing_hooks_callback_errors_and_redo_fail_loudly(
    tmp_path: pathlib.Path,
) -> None:
    """Verify missing hooks, callback errors, and redo fail loudly (`#148`)."""
    wind_case = tmp_path / "wind_case"
    wind_case.mkdir()
    (wind_case / "inlist").write_text(
        "&kap\n"
        "  zbase = 0.02\n"
        "/\n"
        "&controls\n"
        "  initial_mass = 1.0\n"
        "  use_other_wind = .true.\n"
        "  max_model_number = 1\n"
        "/\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="use_other_wind"):
        star.run_test_suite_case(
            wind_case,
            skip_optional=True,
            max_steps_per_stage=1,
        )

    class _BadStarExtras:
        def extras_check_model(self, _state: dict[str, object]) -> str:
            raise ValueError("boom from python star extras")

    ok_star = tmp_path / "ok_star"
    ok_star.mkdir()
    # ZAMS models start at `model_number = 1` (becoming `2` after 1 step);
    # keep `max_model_number > 2` so step 1 does not already terminate inside
    # `star_check_model` (`run_star_support.f90:211-212`) before
    # `extras_check_model` runs (`max_steps_per_stage=1` caps to 1 step).
    (ok_star / "inlist").write_text(
        "&kap\n  zbase = 0.02\n/\n"
        "&controls\n  initial_mass = 1.0\n  max_model_number = 10\n/\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="boom from python star extras"):
        star.run_test_suite_case(
            ok_star,
            skip_optional=True,
            max_steps_per_stage=1,
            extras=_BadStarExtras(),
        )

    class _OnceRedoStarExtras:
        def __init__(self) -> None:
            self.calls = 0

        def extras_check_model(self, _state: dict[str, object]) -> str:
            self.calls += 1
            return "redo" if self.calls == 1 else "keep_going"

    redo_star = tmp_path / "redo_star"
    redo_star.mkdir()
    (redo_star / "inlist").write_text(
        "&kap\n  zbase = 0.02\n/\n"
        "&controls\n  initial_mass = 1.0\n  redo_limit = 2\n/\n",
        encoding="utf-8",
    )
    once_redo = _OnceRedoStarExtras()
    rep_once = star.run_test_suite_case(
        redo_star,
        skip_optional=True,
        max_steps_per_stage=1,
        extras=once_redo,
    )
    assert once_redo.calls == 2
    assert rep_once["total_steps"] == 1

    class _AlwaysRedoStarExtras:
        def extras_check_model(self, _state: dict[str, object]) -> str:
            return "redo"

    rep_lim = star.run_test_suite_case(
        redo_star,
        skip_optional=True,
        max_steps_per_stage=1,
        extras=_AlwaysRedoStarExtras(),
    )
    assert rep_lim["termination_code"] == "redo_limit"

    class _BadBinaryExtras:
        def finish_step(self, _state: dict[str, object]) -> str:
            raise ValueError("boom from python binary extras")

    ok_bin = tmp_path / "ok_bin"
    ok_bin.mkdir()
    (ok_bin / "inlist").write_text(
        "&binary_job\n"
        "  evolve_both_stars = .false.\n"
        "/\n"
        "&binary_controls\n"
        "  m1 = 10.0\n"
        "  m2 = 8.0\n"
        "  initial_period_in_days = 5.0\n"
        "/\n",
        encoding="utf-8",
    )
    (ok_bin / "inlist1").write_text(
        "&kap\n  zbase = 0.02\n/\n&controls\n  initial_mass = 10.0\n/\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="boom from python binary extras"):
        binary.run_binary_test_suite_case(
            ok_bin,
            max_steps_cap=1,
            extras=_BadBinaryExtras(),
        )
