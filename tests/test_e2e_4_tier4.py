# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage E2E-4 (#95) verification test suite for pulsation, `RSP`, and `astero`
interfaces without in-library test-case overrides (`#148`).
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

from flexmesa import astero, star
from flexmesa.testing import case_manifest, known_gaps_gate
from tests.case_extras import resolve_case_extras


def test_manifest_counts_match_106_release_test_suite_cases() -> None:
    """Verify MESA 26.4.1 release manifest covers 90 star, 8 binary, and
    8 astero cases (`106` total, `#141`, `#143`, `#148`).
    """
    entries = case_manifest.load_case_manifest().cases
    star_cases = [e.case for e in entries if e.suite == "star"]
    bin_cases = [e.case for e in entries if e.suite == "binary"]
    ast_cases = [e.case for e in entries if e.suite == "astero"]
    assert len(star_cases) == 90
    assert len(bin_cases) == 8
    assert len(ast_cases) == 8
    assert len(entries) == 106


def test_linear_oscillation_modes_and_astero_scaling_invariants() -> None:
    """Verify linear oscillation mode scaffolding and astero scaling (#4)."""
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    p_modes = star.eval_linear_oscillation_modes(model, 0, [1, 2, 3, 4])
    assert len(p_modes) == 4
    assert p_modes[0]["freq_uhz"] < p_modes[1]["freq_uhz"]
    assert p_modes[1]["freq_uhz"] < p_modes[2]["freq_uhz"]
    assert p_modes[2]["freq_uhz"] < p_modes[3]["freq_uhz"]
    assert all(m["period_days"] > 0.0 for m in p_modes)
    assert all(m["inertia"] > 0.0 for m in p_modes)

    g_modes = star.eval_linear_oscillation_modes(model, 1, [-3, -2, -1])
    assert len(g_modes) == 3
    assert g_modes[0]["period_days"] > g_modes[1]["period_days"]
    assert g_modes[1]["period_days"] > g_modes[2]["period_days"]

    mode_dict, matched = star.eval_target_mode_frequency_and_growth(model)
    assert matched is True
    assert mode_dict["freq_hz"] > 0.0

    dnu, numax = star.eval_astero_scaling_dnu_numax_uhz(model)
    assert dnu > 50.0
    assert numax > 500.0


def test_rsp_lina_modes_and_create_rsp_envelope_model() -> None:
    """Verify RSP LINA modes and envelope builder (#5) without case stubs."""
    modes = star.eval_rsp_lina_modes(4.165, 1438.8, 6050.0, 0.73, 0.007, 3)
    assert len(modes) == 3
    # MESA 26.4.1 reference (`tests/fixtures/golden/rsp/lina_26_4_1.tsv`):
    assert abs(modes[0]["period_days"] - 3.978609425761593) < 1.0e-5
    assert abs(modes[1]["period_days"] - 2.830643645081431) < 1.0e-5
    assert abs(modes[2]["period_days"] - 2.163348404507672) < 1.0e-5
    assert abs(modes[0]["growth_rate"] - 1.4481857441727532e-2) < 1.0e-5
    assert abs(modes[1]["growth_rate"] - 2.828851181618169e-2) < 1.0e-5
    assert abs(modes[2]["growth_rate"] - (-8.73972025237343e-2)) < 1.0e-5
    assert modes[0]["period_days"] > modes[1]["period_days"]
    assert modes[1]["period_days"] > modes[2]["period_days"]

    rsp_model = star.build_rsp_envelope_model(
        m_msun=0.65,
        l_lsun=45.0,
        teff_k=6700.0,
        x_h=0.75,
        z_metal=0.0014,
    )
    assert rsp_model.nz >= 64
    step_rep = rsp_model.evolve_step()
    assert step_rep.converged is True


@pytest.mark.parametrize(
    "case_name",
    [
        "gyre_in_mesa_ms",
        "gyre_in_mesa_wd",
        "gyre_in_mesa_envelope",
        "rsp_RR_Lyrae",
        "rsp_Cepheid",
        "rsp_Delta_Scuti",
        "rsp_Cepheid_6M",
    ],
)
def test_synthetic_tier4_star_case_execution_and_photo_restart(
    tmp_path: pathlib.Path,
    case_name: str,
) -> None:
    """Verify synthetic offline pulsation/RSP directories reach
    `max_model_number = 2` and verify 0-ULP `.photo` restarts.
    """
    case_dir = tmp_path / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "rn").write_text(
        "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
        encoding="utf-8",
    )
    if case_name.startswith("rsp_"):
        (case_dir / "inlist").write_text(
            "&star_job\n"
            "  create_RSP_model = .true.\n"
            "  save_model_when_terminate = .true.\n"
            "  save_model_filename = 'final.mod'\n"
            "/\n&eos\n/\n&kap\n  zbase = 0.0014\n/\n&controls\n"
            "  RSP_mass = 0.65\n"
            "  RSP_Teff = 6700.0\n"
            "  RSP_L = 45.0\n"
            "  RSP_X = 0.75\n"
            "  RSP_Z = 0.0014\n"
            "  max_model_number = 2\n"
            "/\n",
            encoding="utf-8",
        )
    else:
        (case_dir / "inlist").write_text(
            "&star_job\n"
            "  save_model_when_terminate = .true.\n"
            "  save_model_filename = 'final.mod'\n"
            "/\n&eos\n/\n&kap\n  zbase = 0.02\n/\n&controls\n"
            "  initial_mass = 1.5\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "  x_integer_ctrl(3) = 1\n"
            "  x_integer_ctrl(4) = 5\n"
            "  x_ctrl(1) = 4.9e-4\n"
            "  x_ctrl(2) = 1.6e-11\n"
            "/\n",
            encoding="utf-8",
        )
    rep = star.run_test_suite_case(
        case_dir,
        skip_optional=True,
        max_steps_per_stage=3,
        verify_restart=True,
    )
    assert rep["case_name"] == case_name
    assert rep["tier"] == 0
    assert rep["passed"] is True
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True


def test_synthetic_astero_case_execution_and_photo_restart(
    tmp_path: pathlib.Path,
) -> None:
    """Verify synthetic offline `astero/test_suite` execution and restart."""
    case_dir = tmp_path / "example_astero"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "rn").write_text(
        "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
        encoding="utf-8",
    )
    (case_dir / "inlist").write_text(
        "&star_job\n"
        # A plain star run, as upstream does with this control
        # (`astero/private/astero_run_support.f90:171-174`).
        "  astero_just_call_my_extras_check_model = .true.\n"
        "  save_model_when_terminate = .true.\n"
        "  save_model_filename = 'final.mod'\n"
        "/\n&eos\n/\n&kap\n  zbase = 0.02\n/\n&controls\n"
        "  initial_mass = 1.1\n"
        "  initial_z = 0.02\n"
        "  max_model_number = 2\n"
        "/\n",
        encoding="utf-8",
    )
    (case_dir / "inlist_astero_search_controls").write_text(
        "&astero_search_controls\n  chi2_seismo_fraction = 0.666\n/\n",
        encoding="utf-8",
    )
    rep = astero.run_astero_test_suite_case(
        case_dir,
        skip_optional=True,
        max_steps_per_stage=None,
        verify_restart=True,
    )
    assert rep["case_name"] == "example_astero"
    assert rep["tier"] == 0
    assert rep["passed"] is True
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True


def test_rsp_and_astero_raw_cases_rejected_and_cleaned_copies_smoke(
    tmp_path: pathlib.Path,
) -> None:
    """Run strict-valid raw `rsp_BLAP`; reject raw `rsp_Delta_Scuti` and
    `astero` cases under `#147` and run cleaned 2-step smoke checks.
    """
    root = known_gaps_gate.require_mesa_test_inputs_dir()
    if root is None:
        return

    raw_rsp = root / "star" / "test_suite" / "rsp_BLAP"
    assert raw_rsp.is_dir(), f"Missing upstream case: {raw_rsp}"
    strict_rsp = tmp_path / "strict_rsp_BLAP"
    shutil.copytree(raw_rsp, strict_rsp)
    strict_rsp_rep = star.run_test_suite_case(
        strict_rsp,
        skip_optional=True,
        max_steps_per_stage=2,
        verify_restart=True,
        extras=resolve_case_extras("star", raw_rsp),
    )
    assert strict_rsp_rep["tier"] == 0
    assert strict_rsp_rep["assertions_passed"] is True
    assert strict_rsp_rep["photo_restart_verified"] is True

    raw_delta = root / "star" / "test_suite" / "rsp_Delta_Scuti"
    assert raw_delta.is_dir(), f"Missing upstream case: {raw_delta}"
    with pytest.raises(
        ValueError,
        match=r"(unsupported|unknown|unrecognized).*control",
    ):
        star.run_test_suite_case(
            raw_delta,
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
            extras=resolve_case_extras("star", raw_delta),
        )
    clean_rsp = tmp_path / "rsp_Delta_Scuti"
    star.copy_cleaned_case(raw_delta, clean_rsp, suite="star")
    rsp_rep = star.run_test_suite_case(
        clean_rsp,
        skip_optional=True,
        max_steps_per_stage=2,
        verify_restart=True,
        extras=resolve_case_extras("star", raw_delta),
    )
    assert rsp_rep["tier"] == 0
    assert rsp_rep["passed"] is False
    assert rsp_rep["assertions_passed"] is True
    assert rsp_rep["photo_restart_verified"] is True

    for case_name in ("example_astero", "surface_effects"):
        raw_ast = root / "astero" / "test_suite" / case_name
        assert raw_ast.is_dir(), f"Missing upstream case: {raw_ast}"
        with pytest.raises(
            ValueError,
            match=r"(unsupported|unknown|unrecognized).*control",
        ):
            astero.run_astero_test_suite_case(
                raw_ast,
                max_steps_per_stage=2,
                verify_restart=True,
            )
        clean_ast = tmp_path / case_name
        star.copy_cleaned_case(raw_ast, clean_ast, suite="astero")
        ast_rep = astero.run_astero_test_suite_case(
            clean_ast,
            max_steps_per_stage=2,
            verify_restart=True,
            extras=resolve_case_extras("astero", raw_ast),
        )
        assert ast_rep["tier"] == 0
        expected_passed = case_name == "surface_effects"
        assert ast_rep["passed"] is expected_passed
        assert ast_rep["assertions_passed"] is True
        assert ast_rep["photo_restart_verified"] is True
