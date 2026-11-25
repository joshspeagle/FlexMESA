# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage E2E-5 (#96) verification test suite for test-side `run_star_extras`
and `run_binary_extras` dispatch and coupled binary execution (`#148`).
"""

from __future__ import annotations

import math
import pathlib
import shutil

import pytest

from flexmesa import _rust, binary, constants, star
from flexmesa.testing import case_manifest, harness_runner, known_gaps_gate
from tests.case_extras import resolve_case_extras
from tests.case_extras.binary_extras import (
    DoubleBhBinaryExtras,
    JdotGrCheckBinaryExtras,
    JdotMlCheckBinaryExtras,
)
from tests.case_extras.star_extras import CheckRedoStarExtras

_BINARY_CASES = (
    "double_bh",
    "evolve_both_stars",
    "jdot_gr_check",
    "jdot_ls_check",
    "jdot_ml_check",
    "star_plus_point_mass",
    "star_plus_point_mass_explicit_mdot",
    "wind_fed_bhhmxb",
)


def test_e2e_5_complete_106_active_test_suite_manifest_inventory() -> None:
    """Verify manifest coverage across all 106 release MESA `26.4.1` cases."""
    entries = case_manifest.load_case_manifest().cases
    by_suite: dict[str, list[str]] = {"star": [], "binary": [], "astero": []}
    for entry in entries:
        by_suite[entry.suite].append(entry.case)
    assert len(by_suite["star"]) == 90
    assert len(set(by_suite["star"])) == 90
    assert len(by_suite["astero"]) == 8
    assert len(by_suite["binary"]) == 8
    assert set(by_suite["binary"]) == set(_BINARY_CASES)


def test_e2e_5_test_side_star_extras_and_photo_restart() -> None:
    """Verify test-side `CheckRedoStarExtras` hooks and 0-ULP restart."""
    extras = CheckRedoStarExtras()
    extras.extras_startup({}, restart=False)
    assert extras.extras_check_model({"model_number": 9}) == "keep_going"
    assert extras.extras_check_model({"model_number": 10}) == "redo"
    assert extras.extras_check_model({"model_number": 10}) == "keep_going"
    assert extras.extras_finish_step({}) == "keep_going"
    assert extras.extras_check_model({"model_number": 10}) == "redo"

    m = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    m.eval_hydro_vars()
    h1, h2 = m.verify_photo_restart(n_steps=1)
    assert h1 == h2


def test_e2e_5_test_side_binary_extras_oracles() -> None:
    """Verify test-side `JdotGrCheckBinaryExtras`, `JdotMlCheckBinaryExtras`,
    and `DoubleBhBinaryExtras.other_wind_cgs` follow the upstream routines.
    """
    gr = JdotGrCheckBinaryExtras()
    # `b% xtra` starts at zero, so nothing is tested before a history write.
    assert gr.check_model({"s1_mstar_dot": 0.0}) == "keep_going"
    gr_state = {
        "initial_separation_in_rsuns": 1.0,
        "m1": 1.4,
        "m2": 1.4,
        "separation": 0.95 * constants.RSUN,
        "s1_star_age": 1.0e7,
    }
    sep_cols = gr.extra_binary_history_columns(gr_state)
    assert sep_cols[0][0] == "expected_separation"
    assert 0.0 < sep_cols[0][1] < 1.0
    # The stored error from the history write drives the next check
    # (binary/test_suite/jdot_gr_check/src/run_binary_extras.f90:119,164).
    assert abs((sep_cols[0][1] - 0.95) / sep_cols[0][1]) > 1.0e-2
    assert gr.check_model({"s1_mstar_dot": 0.0}) == (
        "terminate",
        "Terminate because of large error in orbital separation",
    )
    assert gr.check_model({"s1_mstar_dot": -1.0e18}) == (
        "terminate",
        "terminate due to beginning of RLO",
    )
    with pytest.raises(KeyError):
        gr.check_model({})
    with pytest.raises(KeyError):
        gr.extra_binary_history_columns({"m1": 1.4, "m2": 1.4})

    ml = JdotMlCheckBinaryExtras()
    assert ml.check_model({}) == "keep_going"
    ml_state = {
        "initial_separation_in_rsuns": 10.0,
        "m1": 2.0,
        "m2": 1.0,
        "m": [1.8 * constants.MSUN, 1.13 * constants.MSUN],
        "separation": 10.0 * constants.RSUN,
        "mass_transfer_alpha": 0.1,
        "mass_transfer_beta": 0.2,
        "mass_transfer_delta": 0.05,
        "mass_transfer_gamma": 1.5,
    }
    ml_cols = ml.extra_binary_history_columns(ml_state)
    assert ml_cols[0][0] == "expected_separation"
    assert ml_cols[0][1] > 0.0
    assert abs((ml_cols[0][1] - 10.0) / ml_cols[0][1]) > 1.0e-4
    assert ml.check_model({}) == (
        "terminate",
        "Terminate because of large error in orbital separation",
    )
    ml_state["separation"] = ml_cols[0][1] * constants.RSUN
    ml.extra_binary_history_columns(ml_state)
    assert ml.check_model({}) == "keep_going"

    dbh = DoubleBhBinaryExtras()
    l_surf = 5.0e5 * constants.LSUN
    m_surf = 60.0 * constants.MSUN
    r_surf = 15.0 * constants.RSUN
    wind_state = {"surface_h1": 0.70, "zbase": 0.02}
    assert (
        dbh.other_wind_cgs(wind_state, 1, True, l_surf, m_surf, r_surf, 4.5e4)
        == 0.0
    )
    # Upstream reads `s% xa(h1,1)` and `Zbase` before anything else; a
    # snapshot without them raises instead of using a default.
    with pytest.raises(KeyError):
        dbh.other_wind_cgs({}, 1, True, l_surf, m_surf, r_surf, 4.5e4)
    w_cgs = dbh.other_wind_cgs(
        wind_state, 1, False, l_surf, m_surf, r_surf, 4.5e4
    )
    # T1 is above the jump temperature and Xs == 0.7 leaves alfa = 0
    # (binary/test_suite/double_bh/src/run_star_extras.f90:113-119), so the
    # wind is one tenth of the Hamann rate (:106, :199-203).
    hamann_msun_yr = 10.0 ** (
        -11.95
        + 1.5 * math.log10(5.0e5)
        - 2.85 * 0.70
        + 0.85 * math.log10(0.02 / 0.017)
    )
    expected_cgs = -(hamann_msun_yr / 10.0) * constants.MSUN / constants.SECYER
    assert w_cgs == pytest.approx(expected_cgs, rel=1.0e-12)
    # The radius argument is used as given (Nieuwenhuijzen term, :186-190):
    # below the jump the wind is max(Vink, Nieuwenhuijzen), and a larger
    # `Rsurf` raises the latter.
    w_small_r = dbh.other_wind_cgs(
        wind_state, 1, False, l_surf, m_surf, r_surf, 1.0e4
    )
    w_large_r = dbh.other_wind_cgs(
        wind_state, 1, False, l_surf, m_surf, 1.0e3 * r_surf, 1.0e4
    )
    assert w_large_r < w_small_r < 0.0


class _BogusActionExtras:
    """Binary hook returning a directive string the adapter does not know."""

    def check_model(self, _state: dict[str, object]) -> str:
        return "large_separation_error"


class _TerminateWithMessageExtras:
    """Binary hook terminating with a diagnostic message."""

    def finish_step(self, _state: dict[str, object]) -> tuple[str, str]:
        return ("terminate", "anything")


def test_e2e_5_hook_cannot_choose_termination_code(
    tmp_path: pathlib.Path,
) -> None:
    """A hook's message never becomes the termination code, and an
    unrecognised directive string is an error.
    """
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    clean_case = tmp_path / "jdot_gr_check"
    star.copy_cleaned_case(
        base / "binary" / "test_suite" / "jdot_gr_check",
        clean_case,
        suite="binary",
    )
    with pytest.raises(
        ValueError,
        match=r"unrecognised action 'large_separation_error' returned by "
        r"extras_binary_check_model",
    ):
        binary.run_binary_test_suite_case(
            clean_case, max_steps_cap=4, extras=_BogusActionExtras()
        )
    rep = binary.run_binary_test_suite_case(
        clean_case, max_steps_cap=4, extras=_TerminateWithMessageExtras()
    )
    assert rep["total_steps"] == 1
    assert rep["termination_code"] == "extras_binary_finish_step"


def test_e2e_5_tier5_star_and_binary_suites_execution(
    tmp_path: pathlib.Path,
) -> None:
    """Verify raw `star` and `binary` cases reject unsupported controls under
    `#147` and cleaned copies execute with test-side extras (`#148`).
    """
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    clean_root = tmp_path / "mesa"
    assert (base / "star" / "test_suite").is_dir()
    star_cases = ("check_redo", "other_physics_hooks", "timing")
    strict_star_cases = frozenset({"check_redo", "timing"})
    for case_name in star_cases:
        raw_case = base / "star" / "test_suite" / case_name
        if case_name in strict_star_cases:
            raw_copy = tmp_path / "strict_star" / case_name
            shutil.copytree(raw_case, raw_copy)
            rep = star.run_test_suite_case(
                raw_copy,
                skip_optional=True,
                max_steps_per_stage=2,
                verify_restart=True,
                extras=resolve_case_extras("star", raw_case),
            )
            assert rep["tier"] == 0
            assert rep["assertions_passed"] is True
            assert rep["photo_restart_verified"] is True
            continue
        with pytest.raises(
            ValueError,
            match=(
                r"(unsupported|unknown|unrecognized).*control"
                r"|requires an? (custom )?\w+ hook"
            ),
        ):
            star.run_test_suite_case(
                raw_case,
                skip_optional=True,
                max_steps_per_stage=2,
                verify_restart=True,
                extras=resolve_case_extras("star", raw_case),
            )
        star.copy_cleaned_case(
            raw_case,
            clean_root / "star" / "test_suite" / case_name,
            suite="star",
        )
    star_root = clean_root / "star" / "test_suite"
    with pytest.raises(ValueError, match="overshoot_scheme"):
        star.run_test_suite_case(
            star_root / "other_physics_hooks",
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
        )

    assert (base / "binary" / "test_suite").is_dir()
    bad_bin = tmp_path / "bad_bin_unsupported"
    bad_bin.mkdir()
    (bad_bin / "inlist").write_text(
        "&binary_job\n  evolve_both_stars = .false.\n/\n"
        "&binary_controls\n"
        "  m1 = 10.0\n"
        "  m2 = 8.0\n"
        "  initial_period_in_days = 5.0\n"
        "/\n",
        encoding="utf-8",
    )
    (bad_bin / "inlist1").write_text(
        "&star_job\n  echo_at_start = 'hi'\n/\n"
        "&controls\n  initial_mass = 10.0\n/\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=r"(unsupported|unknown|unrecognized).*control",
    ):
        binary.run_binary_test_suite_case(bad_bin, max_steps_cap=4)

    strict_bin_root = tmp_path / "strict_binary"
    for case_name in _BINARY_CASES:
        raw_case = base / "binary" / "test_suite" / case_name
        run_case = strict_bin_root / case_name
        shutil.copytree(raw_case, run_case)
        brep = binary.run_binary_test_suite_case(
            run_case,
            max_steps_cap=4,
            extras=resolve_case_extras("binary", raw_case),
        )
        assert brep["total_steps"] == 4
        assert brep["termination_code"] == "MaxStepsCap"
        assert brep["passed"] is False

    for case_name, tol in (
        ("jdot_gr_check", 1.0e-2),
        ("jdot_ml_check", 1.0e-4),
    ):
        raw_case = base / "binary" / "test_suite" / case_name
        trace_out = tmp_path / f"trace_{case_name}"
        harness_runner.run_trace_inprocess(
            "binary",
            strict_bin_root / case_name,
            trace_out,
            max_steps=4,
            allow_unsupported_controls=False,
            extras=resolve_case_extras("binary", raw_case),
        )
        parsed = _rust.mesa_read_log_file(
            str(trace_out / "binary_history.data")
        )
        cols = parsed["columns"]
        exp_sep = cols["expected_separation"]
        act_sep = cols["binary_separation"]
        assert len(exp_sep) == 4 and len(act_sep) == 4
        for exp_val, act_val in zip(exp_sep, act_sep, strict=True):
            assert abs((exp_val - act_val) / exp_val) < tol
