# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage E2E-5 (#96) verification test suite for test-side `run_star_extras`
and `run_binary_extras` dispatch and coupled binary execution (`#148`).
"""

from __future__ import annotations

import pathlib

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
    and `DoubleBhBinaryExtras.other_wind_cgs` match analytical formulas.
    """
    gr = JdotGrCheckBinaryExtras()
    sep_cols = gr.extra_binary_history_columns(
        {
            "initial_separation_cm": 1.0 * constants.RSUN,
            "initial_mass_msun": [1.4, 1.4],
            "separation": 0.9 * constants.RSUN,
            "donor_mass_msun": 1.4,
            "accretor_mass_msun": 1.4,
            "s1_star_age": 1.0e7,
            "binary_age_yr": 1.0e7,
        }
    )
    assert sep_cols[0][0] == "expected_separation"
    assert 0.0 < sep_cols[0][1] < 1.0

    ml = JdotMlCheckBinaryExtras()
    ml_cols = ml.extra_binary_history_columns(
        {
            "initial_separation_cm": 10.0 * constants.RSUN,
            "initial_mass_msun": [2.0, 1.0],
            "separation": 10.0 * constants.RSUN,
            "donor_i": 1,
            "donor_mass_msun": 1.8,
            "accretor_mass_msun": 1.13,
            "mass_transfer_alpha": 0.1,
            "mass_transfer_beta": 0.2,
            "mass_transfer_delta": 0.05,
            "mass_transfer_gamma": 1.5,
        }
    )
    assert ml_cols[0][0] == "expected_separation"
    assert ml_cols[0][1] > 0.0

    dbh = DoubleBhBinaryExtras()
    assert dbh.other_wind_cgs({}, 1, ignore_rlof_flag=True) == 0.0
    w_cgs = dbh.other_wind_cgs(
        {
            "surface_l_lsun": 5.0e5,
            "star_mass": 60.0,
            "surface_r_rsun": 15.0,
            "teff": 45_000.0,
            "surface_h1": 0.70,
            "surface_he4": 0.28,
            "zbase": 0.02,
        },
        1,
        ignore_rlof_flag=False,
    )
    assert w_cgs is not None and w_cgs < 0.0


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
    for case_name in star_cases:
        raw_case = base / "star" / "test_suite" / case_name
        with pytest.raises(
            ValueError,
            match=r"(unsupported|unknown|unrecognized).*control",
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
    for case_name in ("check_redo", "timing"):
        raw_case = base / "star" / "test_suite" / case_name
        rep = star.run_test_suite_case(
            star_root / case_name,
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
            extras=resolve_case_extras("star", raw_case),
        )
        assert rep["tier"] == 0
        assert rep["assertions_passed"] is True
        assert rep["photo_restart_verified"] is True

    assert (base / "binary" / "test_suite").is_dir()
    for case_name in _BINARY_CASES:
        raw_case = base / "binary" / "test_suite" / case_name
        with pytest.raises(
            ValueError,
            match=r"(unsupported|unknown|unrecognized).*control",
        ):
            binary.run_binary_test_suite_case(
                raw_case,
                max_steps_cap=4,
                extras=resolve_case_extras("binary", raw_case),
            )
        clean_case = clean_root / "binary" / "test_suite" / case_name
        star.copy_cleaned_case(raw_case, clean_case, suite="binary")
        brep = binary.run_binary_test_suite_case(
            clean_case,
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
        clean_case = clean_root / "binary" / "test_suite" / case_name
        trace_out = tmp_path / f"trace_{case_name}"
        harness_runner.run_trace_inprocess(
            "binary",
            clean_case,
            trace_out,
            max_steps=4,
            allow_unsupported_controls=True,
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
