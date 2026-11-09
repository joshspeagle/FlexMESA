# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage E2E-4 (#95, closing #4 and #5) verification test suite for all 16
Tier 4 `star/test_suite` pulsation cases (`15` active + `rsp_Cepheid_6M`), all
8 `astero/test_suite` cases, and all 32 `star/dev_cases_*` pulsation cases.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from flexmesa import star


def test_tier4_cases_inventory_and_cumulative_counts() -> None:
    """Verify Tier 4 inventories and cumulative 95/107 + 32/32 case counts."""
    assert len(star.TIER4_STAR_CASES) == 16
    assert len(set(star.TIER4_STAR_CASES)) == 16
    assert len(star.TIER4_ACTIVE_STAR_CASES) == 15
    assert len(set(star.TIER4_ACTIVE_STAR_CASES)) == 15
    assert set(star.TIER4_STAR_CASES) - set(star.TIER4_ACTIVE_STAR_CASES) == {
        "rsp_Cepheid_6M"
    }
    assert len(star.TIER4_ASTERO_CASES) == 8
    assert len(set(star.TIER4_ASTERO_CASES)) == 8
    assert len(star.TIER4_DEV_CASES) == 32
    assert len(set(star.TIER4_DEV_CASES)) == 32

    prior_star = (
        set(star.TIER1_CASES) | set(star.TIER2_CASES) | set(star.TIER3_CASES)
    )
    assert len(prior_star) == 72
    assert set(star.TIER4_STAR_CASES).isdisjoint(prior_star)
    cumulative_active = (
        len(prior_star)
        + len(star.TIER4_ACTIVE_STAR_CASES)
        + len(star.TIER4_ASTERO_CASES)
    )
    assert cumulative_active == 95


def test_linear_oscillation_modes_and_astero_scaling_invariants() -> None:
    """Verify GYRE/ADIPLS linear oscillation modes and scaling (#4)."""
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


def test_rsp_lina_modes_and_create_rsp_envelope_model(
    tmp_path: pathlib.Path,
) -> None:
    """Verify RSP LINA modes, envelope builder (#5), and 2nd-crossing edges."""
    modes_unforced = star.eval_rsp_lina_modes(
        4.165,
        1438.8,
        6050.0,
        0.73,
        0.007,
        3,
        expected_f_period_days=None,
    )
    assert len(modes_unforced) == 3
    assert 2.5 < modes_unforced[0]["period_days"] < 6.0
    assert modes_unforced[0]["growth_rate"] > 0.0

    modes = star.eval_rsp_lina_modes(
        4.165,
        1438.8,
        6050.0,
        0.73,
        0.007,
        3,
        expected_f_period_days=3.9927,
    )
    assert len(modes) == 3
    assert abs(modes[0]["period_days"] - 3.9927) < 1.0e-3
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

    hist_file = tmp_path / "history_7M.data"
    hist_file.write_text(
        "1 2 3\n1 2 3\n1 2 3\n1 2 3\n"
        "model_number star_age star_mass log_L log_Teff\n"
        "100 1.0e7 7.0 3.69 3.82\n"
        "110 1.1e7 7.0 3.70 3.76\n"
        "120 1.2e7 7.0 3.71 3.72\n"
        "130 1.3e7 7.0 3.72 3.68\n",
        encoding="utf-8",
    )
    tb, tr, dt = star.eval_rsp_2nd_crossing_edges(tmp_path, 100.0)
    assert abs(tb - 6109.4) < 25.0
    assert abs(tr - 6009.4) < 25.0
    assert abs(dt - (tb - tr)) < 1.0e-9


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
    """Verify synthetic offline Tier 4 star case execution and 0-ULP restart."""
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
            "/\n&eos\n/\n&kap\n/\n&controls\n"
            "  RSP_mass = 0.65\n"
            "  RSP_Teff = 6700.0\n"
            "  RSP_L = 45.0\n"
            "  RSP_X = 0.75\n"
            "  RSP_Z = 0.0014\n"
            "  max_model_number = 2\n"
            "  x_ctrl(23) = 0.50\n"
            "  x_ctrl(24) = 0.036\n"
            "/\n",
            encoding="utf-8",
        )
    else:
        (case_dir / "inlist").write_text(
            "&star_job\n"
            "  save_model_when_terminate = .true.\n"
            "  save_model_filename = 'final.mod'\n"
            "/\n&eos\n/\n&kap\n/\n&controls\n"
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
        max_steps_per_stage=2,
        verify_restart=True,
    )
    assert rep["case_name"] == case_name
    assert rep["tier"] == 4
    assert rep["passed"] is True
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True


def test_synthetic_astero_case_execution_and_photo_restart(
    tmp_path: pathlib.Path,
) -> None:
    """Verify synthetic offline astero/test_suite case execution and restart."""
    case_dir = tmp_path / "example_astero"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "rn").write_text(
        "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
        encoding="utf-8",
    )
    (case_dir / "inlist").write_text(
        "&star_job\n"
        "  save_model_when_terminate = .true.\n"
        "  save_model_filename = 'final.mod'\n"
        "/\n&eos\n/\n&kap\n/\n&controls\n"
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
    rep = star.run_astero_test_suite_case(
        case_dir,
        skip_optional=True,
        max_steps_per_stage=2,
        verify_restart=True,
    )
    assert rep["case_name"] == "example_astero"
    assert rep["tier"] == 4
    assert rep["passed"] is True
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True


def test_upstream_mesa_tier4_and_dev_cases_suites_pass() -> None:
    """Run all 24 Tier 4 cases and 32 dev_cases_* when MESA_DIR is set."""
    mesa_dir = os.environ.get("MESA_DIR", "")
    if not mesa_dir:
        pytest.skip("MESA_DIR not set; skipping upstream MESA Tier 4 suites")
    root = pathlib.Path(mesa_dir)
    if not (root / "star" / "test_suite").is_dir():
        pytest.skip(f"Upstream star/test_suite not found under {root}")

    t4_reports = star.run_tier4_suite(
        root, max_steps_per_stage=2, verify_restart=True
    )
    assert len(t4_reports) == 24
    assert all(r["passed"] and r["assertions_passed"] for r in t4_reports)

    dev_reports = star.run_dev_cases_suite(
        root, max_steps_per_stage=2, verify_restart=True
    )
    assert len(dev_reports) == 32
    assert all(r["passed"] and r["assertions_passed"] for r in dev_reports)
