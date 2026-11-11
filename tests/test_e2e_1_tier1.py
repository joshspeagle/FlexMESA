# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage E2E-1 (#92) Tier 1 hydrostatic, solar, mixing, and diffusion
`test_suite` reproduction and physics-hook verification suite.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import pytest

from flexmesa import star
from flexmesa.star import e2e


def test_tier1_cases_inventory_has_33_canonical_entries() -> None:
    """Verify `TIER1_CASES` contains all 33 canonical Tier 1 case names."""
    assert isinstance(star.TIER1_CASES, tuple)
    assert len(star.TIER1_CASES) == 33
    assert len(set(star.TIER1_CASES)) == 33
    assert star.TIER1_CASES[0] == "make_zams"
    assert star.TIER1_CASES[-1] == "simplex_solar_calibration"


def test_starspot_flux_blocking_scales_effective_luminosity() -> None:
    """Verify YREC starspot flux blocking (`starspots`) satisfies analytical
    scaling $L_{\\mathrm{eff}} = L / (f_{\\mathrm{spot}} x_{\\mathrm{spot}}^4
    + 1 - f_{\\mathrm{spot}})$.
    """
    f_spot = 0.34
    x_spot = 0.85
    expected_alpha = f_spot * (x_spot**4) + (1.0 - f_spot)
    l_eff, alpha = e2e.eval_starspot_luminosity(2.5, f_spot, x_spot)
    assert np.isclose(alpha, expected_alpha, rtol=1e-14, atol=1e-15)
    assert np.isclose(l_eff, 2.5 / expected_alpha, rtol=1e-14, atol=1e-15)
    assert l_eff > 2.5

    l_unspotted, alpha_unspotted = e2e.eval_starspot_luminosity(1.0, 0.0, 0.85)
    assert np.isclose(alpha_unspotted, 1.0, rtol=1e-15, atol=1e-15)
    assert np.isclose(l_unspotted, 1.0, rtol=1e-15, atol=1e-15)


def test_t_tau_gradr_factor_matches_eddington_krishna_swamy_and_hopf() -> None:
    """Verify $T(\\tau)$ Hopf function $q(\\tau)$ and $\\nabla_{\\mathrm{rad}}$
    correction factor $1 + \\tau\\,dq/d\\tau$ (`T_tau_gradr`).
    """
    tau = 2.0 / 3.0
    q_edd, fac_edd = e2e.eval_t_tau_gradr_factor(tau, "Eddington")
    assert np.isclose(q_edd, 2.0 / 3.0, rtol=1e-14, atol=1e-15)
    assert np.isclose(fac_edd, 1.0, rtol=1e-14, atol=1e-15)

    q_ks, fac_ks = e2e.eval_t_tau_gradr_factor(tau, "Krishna_Swamy")
    e1 = np.exp(-2.54 * tau)
    e2 = np.exp(-30.0 * tau)
    expected_q_ks = 1.39 - 0.815 * e1 - 0.025 * e2
    expected_fac_ks = 1.0 + tau * (2.54 * 0.815 * e1 + 30.0 * 0.025 * e2)
    assert np.isclose(q_ks, expected_q_ks, rtol=1e-14, atol=1e-15)
    assert np.isclose(fac_ks, expected_fac_ks, rtol=1e-14, atol=1e-15)

    q_hopf, fac_hopf = e2e.eval_t_tau_gradr_factor(tau, "solar_Hopf_grey")
    assert q_hopf > 0.5
    assert fac_hopf > 1.0


def test_custom_colors_and_pulse_atm_export_preserve_invariants() -> None:
    """Verify synthetic photometry (`custom_colors`) and pulsation atmosphere
    exports (`check_pulse_atm`) on a 1 Msun ZAMS model.
    """
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    cols = e2e.eval_model_colors(model)
    assert len(cols) == 10
    assert all(np.isfinite(c) for c in cols)
    mag_bol, flux_bol = cols[0], cols[1]
    assert 0.0 < mag_bol < 25.0
    assert flux_bol > 0.0

    gyre_txt = model.to_gyre_str(add_center_point=True)
    fgong_txt = model.to_fgong_str(add_center_point=True)
    assert len(gyre_txt.splitlines()) == model.nz + 2
    assert len(fgong_txt.splitlines()) > model.nz


def test_solar_calibration_objective_and_nelder_mead_simplex_step() -> None:
    """Verify solar calibration $\\chi^2$ and Nelder-Mead reflection step
    (`simplex_solar_calibration`).
    """
    chi2_zero = e2e.eval_solar_calibration_objective(0.0, 0.0, 0.0229, 0.0229)
    assert np.isclose(chi2_zero, 0.0, atol=1e-15)

    chi2_perturbed = e2e.eval_solar_calibration_objective(
        1.0e-4, -1.0e-4, 0.0239, 0.0229
    )
    assert np.isclose(chi2_perturbed, 3.0, rtol=1e-12, atol=1e-12)

    simplex = [
        ((0.270, 0.018, 1.80), 12.0),
        ((0.275, 0.019, 1.90), 4.0),
        ((0.280, 0.020, 2.00), 8.0),
        ((0.260, 0.015, 1.60), 25.0),
    ]
    y_r, z_r, alpha_r = e2e.nelder_mead_solar_step(simplex)
    assert 0.270 < y_r <= 0.35
    assert 0.018 < z_r <= 0.04
    assert 1.80 < alpha_r <= 2.8


def test_convective_premix_predictive_mix_and_diffusion_hooks() -> None:
    """Verify convective premixing, predictive mixing, Castellani penetration,
    and Burgers atomic diffusion preserve species mass conservation.
    """
    model = star.StarModel.from_zams(initial_mass=4.0, initial_z=0.02)
    n_cpm = model.do_conv_premix(max_iters=3, avoid_fuel_increase=False)
    assert n_cpm >= 0
    n_pred = model.add_predictive_mixing(
        ingested_mass_limit_coeff=1.0e-2,
        superad_thresh=1.0e-4,
    )
    assert n_pred >= 0
    v_gt, eps_sed = model.do_element_diffusion(dt=1.0e10)
    assert v_gt.shape == (model.nz, model.species)
    assert eps_sed.shape == (model.nz,)
    xa = model.xa
    row_sums = np.sum(xa, axis=1)
    assert np.allclose(row_sums, 1.0, rtol=1e-12, atol=1e-12)
    assert np.all((model.xh_mass >= 0.0) & (model.xh_mass <= 1.0))
    assert np.all((model.yhe_mass >= 0.0) & (model.yhe_mass <= 1.0))
    assert np.any(model.d_mix > 0.0)


@pytest.mark.parametrize(
    "case_name,inlist_body",
    [
        (
            "starspots",
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  x_ctrl(1) = 0.34\n"
            "  x_ctrl(2) = 0.85\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "T_tau_gradr",
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  atm_option = 'T_tau'\n"
            "  atm_T_tau_relation = 'Krishna_Swamy'\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "check_pulse_atm",
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "custom_colors",
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "adjust_net",
            "&star_job\n"
            "  change_net = .true.\n"
            "  new_net_name = 'co_burn.net'\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 2.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "semiconvection",
            "&controls\n"
            "  initial_mass = 4.0\n"
            "  initial_z = 0.02\n"
            "  use_Ledoux_criterion = .true.\n"
            "  alpha_semiconvection = 0.04\n"
            "  thermohaline_coeff = 2.0\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "simplex_solar_calibration",
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  mixing_length_alpha = 1.90\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
    ],
)
def test_synthetic_tier1_case_runners_pass_assertions_and_restart(
    case_name: str,
    inlist_body: str,
) -> None:
    """Verify synthetic Tier 1 case directories execute `Tier1CaseExtras`,
    pass physical assertions, and satisfy bit-identical `.photo` restart parity.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = pathlib.Path(tmp) / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "inlist").write_text(inlist_body, encoding="utf-8")
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
            encoding="utf-8",
        )
        rep = e2e.run_test_suite_case(
            case_dir,
            skip_optional=True,
            max_steps_per_stage=3,
            verify_restart=True,
        )
        assert rep["case_name"] == case_name
        assert rep["tier"] == 1
        assert rep["passed"] is True
        assert rep["assertions_passed"] is True
        assert rep["photo_restart_passed"] is True
        assert rep["photo_restart_verified"] is True
        assert rep["total_steps"] >= 1
        assert rep["total_newton_iters"] >= 1
        assert rep["wall_time_ms"] > 0.0
        assert np.isfinite(rep["final_teff"]) and rep["final_teff"] > 0.0
        assert np.isfinite(rep["final_log_l"])
        assert np.isfinite(rep["final_log_r"])
        assert np.isfinite(rep["final_log_tc"])
        assert np.isfinite(rep["final_log_rhoc"])
        assert 0.0 <= rep["final_center_h1"] <= 1.0
        assert 0.0 <= rep["final_center_he4"] <= 1.0


def test_full_stages_skip_optional_false_multi_stage_tier1_execution() -> None:
    """Verify `skip_optional=False` (`--full-stages`) executes all `do_one`
    stages sequentially from start to finish on a multi-stage Tier 1 case.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = pathlib.Path(tmp) / "make_zams"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\n"
            'if [ -n "$MESA_SKIP_OPTIONAL" ]; then\n'
            "  cp standard_zams.mod zams.mod\n"
            "else\n"
            "  do_one inlist_stage1_header zams.mod\n"
            "fi\n"
            "do_one inlist_stage2_header final.mod\n",
            encoding="utf-8",
        )
        (case_dir / "inlist_stage1_header").write_text(
            "&star_job\n"
            "  save_model_when_terminate = .true.\n"
            "  save_model_filename = 'zams.mod'\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "/\n",
            encoding="utf-8",
        )
        (case_dir / "inlist_stage2_header").write_text(
            "&star_job\n"
            "  load_saved_model = .true.\n"
            "  saved_model_name = 'zams.mod'\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 4\n"
            "/\n",
            encoding="utf-8",
        )
        rep = e2e.run_test_suite_case(
            case_dir,
            skip_optional=False,
            max_steps_per_stage=2,
            verify_restart=True,
        )
        assert rep["case_name"] == "make_zams"
        assert rep["tier"] == 1
        assert rep["passed"] is True
        assert rep["assertions_passed"] is True
        assert rep["photo_restart_passed"] is True
        assert len(rep["stages"]) == 2
        assert rep["total_steps"] >= 2


@pytest.mark.parametrize("case_name", star.TIER1_CASES)
def test_upstream_tier1_test_suite_case_reproduction_and_restart_parity(
    case_name: str,
) -> None:
    """When `MESA_DIR` is present, verify each of the 33 upstream Tier 1
    `star/test_suite/<case>` directories executes cleanly, passes physical
    target assertions, and achieves 0 ULP `.photo` restart parity.
    """
    mesa_dir = os.environ.get("MESA_DIR", "")
    if not mesa_dir:
        pytest.skip("MESA_DIR environment variable is not set")
    case_dir = pathlib.Path(mesa_dir) / "star" / "test_suite" / case_name
    assert case_dir.is_dir(), f"Upstream case directory not found: {case_dir}"

    rep = e2e.run_test_suite_case(
        case_dir,
        skip_optional=True,
        max_steps_per_stage=2,
        verify_restart=True,
    )
    assert rep["case_name"] == case_name
    assert rep["tier"] == 1
    assert rep["passed"] is True
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_passed"] is True
    assert rep["total_steps"] >= 1
    assert rep["total_newton_iters"] >= 1
    assert np.isfinite(rep["final_teff"]) and rep["final_teff"] > 0.0
    assert np.isfinite(rep["final_log_l"])
    assert np.isfinite(rep["final_log_r"])
    assert np.isfinite(rep["final_log_tc"])
    assert np.isfinite(rep["final_log_rhoc"])
