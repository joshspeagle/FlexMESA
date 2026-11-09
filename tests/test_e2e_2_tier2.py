# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage E2E-2 (#93) Tier 2 substellar, planetary, white dwarf,
phase-separation, accretion, nova, and conductive flame `test_suite`
reproduction and physics-hook verification suite.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import pytest

from flexmesa import star
from flexmesa.star import e2e


def test_tier2_cases_inventory_has_17_canonical_entries() -> None:
    """Verify `TIER2_CASES` contains all 17 canonical Tier 2 case names."""
    assert isinstance(star.TIER2_CASES, tuple)
    assert len(star.TIER2_CASES) == 17
    assert len(set(star.TIER2_CASES)) == 17
    assert star.TIER2_CASES[0] == "make_brown_dwarf"
    assert star.TIER2_CASES[-1] == "conductive_flame"
    assert set(star.TIER1_CASES).isdisjoint(set(star.TIER2_CASES))


def test_planetary_irradiation_heating_and_core_boundary() -> None:
    """Verify planetary surface irradiation heating profile and inflated radius
    (`irradiated_planet`, `make_planets`).
    """
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    f_irr = 4.9e8
    sigma_star = 300.0
    eps_irr, r_rjup = e2e.eval_planetary_irradiation_heating(
        model, f_irr=f_irr, col_depth_irr=sigma_star
    )
    assert len(eps_irr) == model.nz
    assert eps_irr[0] > 0.0
    assert eps_irr[0] <= f_irr / (4.0 * sigma_star) * (1.0 + 1.0e-12)
    assert eps_irr[0] >= eps_irr[-1]
    assert np.all(np.diff(eps_irr) <= 1.0e-12)
    assert r_rjup > 1.0

    eps_zero, _ = e2e.eval_planetary_irradiation_heating(
        model, f_irr=0.0, col_depth_irr=sigma_star
    )
    assert np.allclose(eps_zero, 0.0, atol=1e-15)


def test_wd_crystallization_and_three_phase_separation() -> None:
    """Verify white dwarf Coulomb crystallization and 2/3-component phase
    separation (`wd_cool_0.6M` and `wd_o_ne_3_phase`).
    """
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    cryst_co, l_lat_co = e2e.eval_wd_phase_separation(model, option="CO")
    assert 0.0 <= cryst_co <= 1.0
    assert l_lat_co >= 0.0
    assert np.isfinite(cryst_co) and np.isfinite(l_lat_co)

    cryst_one, l_lat_one = e2e.eval_wd_phase_separation(model, option="ONe")
    assert 0.0 <= cryst_one <= 1.0
    assert l_lat_one >= 0.0
    row_sums = np.sum(model.xa, axis=1)
    assert np.allclose(row_sums, 1.0, rtol=1e-12, atol=1e-12)

    # Cool high-density crystallized white dwarf regime (Gamma_k > 185)
    lnd_cool = np.full(model.nz, np.log(2.0e6), dtype=np.float64)
    lnt_cool = np.full(model.nz, np.log(2.0e6), dtype=np.float64)
    model.set_primary_state(
        lnd_cool, lnt_cool, model.ln_r, model.lum, model.dq, model.xa
    )
    cryst_co_cool, l_lat_co_cool = e2e.eval_wd_phase_separation(
        model, option="CO"
    )
    assert cryst_co_cool > 0.0
    assert l_lat_co_cool > 0.0

    cryst_one_cool, l_lat_one_cool = e2e.eval_wd_phase_separation(
        model, option="ONe"
    )
    assert cryst_one_cool > 0.0
    assert l_lat_one_cool > 0.0
    assert np.allclose(np.sum(model.xa, axis=1), 1.0, rtol=1e-12, atol=1e-12)


def test_accretion_composition_and_super_eddington_nova_wind() -> None:
    """Verify accretion composition and super-Eddington nova wind execution on
    synthetic white dwarf accretion configurations (`wd_acc_small_dm`,
    `wd_nova_burst`).
    """
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = pathlib.Path(tmp) / "wd_nova_burst"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "inlist").write_text(
            "&star_job\n"
            "  relax_initial_mass_to_remove_h_env = .true.\n"
            "  new_mass = 0.60\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 0.60\n"
            "  initial_z = 0.02\n"
            "  mass_change = 1.0d-9\n"
            "  accrete_same_as_surface = .false.\n"
            "  accretion_h1 = 0.70\n"
            "  accretion_he4 = 0.28\n"
            "  super_eddington_scaling_factor = 1.0\n"
            "  max_model_number = 2\n"
            "/\n",
            encoding="utf-8",
        )
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
            encoding="utf-8",
        )
        rep = e2e.run_test_suite_case(
            case_dir,
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
        )
        assert rep["case_name"] == "wd_nova_burst"
        assert rep["tier"] == 2
        assert rep["passed"] is True
        assert rep["assertions_passed"] is True
        assert rep["photo_restart_passed"] is True
        assert rep["final_log_tc"] > 6.0


def test_conductive_flame_speed_and_front_propagation() -> None:
    """Verify conductive flame speed scaling and flame front mass coordinate
    (`conductive_flame` and `cburn_inward`).
    """
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    v_flame, q_flame = e2e.eval_conductive_flame_state(model)
    assert np.isfinite(v_flame) and v_flame > 0.0
    assert 0.0 < q_flame <= 1.0


@pytest.mark.parametrize(
    "case_name,inlist_body",
    [
        (
            "make_brown_dwarf",
            "&star_job\n"
            "  create_initial_model = .true.\n"
            "  mass_in_gm_for_create_initial_model = 1.9892d31\n"
            "  radius_in_cm_for_create_initial_model = 3.5d10\n"
            "/\n"
            "&controls\n"
            "  initial_z = 0.0001\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "make_planets",
            "&star_job\n"
            "  create_initial_model = .true.\n"
            "  mass_in_gm_for_create_initial_model = 1.8389d30\n"
            "  radius_in_cm_for_create_initial_model = 1.39822d10\n"
            "  relax_core = .true.\n"
            "  new_core_mass = 3.0012d-5\n"
            "  core_avg_rho = 10.0\n"
            "/\n"
            "&controls\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "irradiated_planet",
            "&star_job\n"
            "  relax_initial_irradiation = .true.\n"
            "  relax_to_this_irrad_flux = 1.0d8\n"
            "  irrad_col_depth = 300.0\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 0.001\n"
            "  initial_z = 0.02\n"
            "  irradiation_flux = 4.9d8\n"
            "  column_depth_for_irradiation = 300.0\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "make_he_wd",
            "&star_job\n"
            "  relax_initial_mass = .true.\n"
            "  new_mass = 0.15\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.5\n"
            "  initial_z = 0.01\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "wd_cool_0.6M",
            "&star_job\n"
            "  relax_initial_mass_to_remove_h_env = .true.\n"
            "  new_mass = 0.60\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 0.60\n"
            "  do_phase_separation = .true.\n"
            "  phase_separation_option = 'CO'\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "wd_o_ne_3_phase",
            "&star_job\n"
            "  relax_initial_mass_to_remove_h_env = .true.\n"
            "  new_mass = 1.10\n"
            "/\n"
            "&controls\n"
            "  initial_mass = 1.10\n"
            "  do_phase_separation = .true.\n"
            "  phase_separation_option = 'ONe'\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
        (
            "conductive_flame",
            "&star_job\n"
            "  create_initial_model = .true.\n"
            "/\n"
            "&controls\n"
            "  use_other_build_initial_model = .true.\n"
            "  mlt_option = 'none'\n"
            "  max_model_number = 2\n"
            "/\n",
        ),
    ],
)
def test_synthetic_tier2_case_runners_pass_assertions_and_restart(
    case_name: str,
    inlist_body: str,
) -> None:
    """Verify synthetic Tier 2 case directories execute `Tier2CaseExtras`,
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
            max_steps_per_stage=2,
            verify_restart=True,
        )
        assert rep["case_name"] == case_name
        assert rep["tier"] == 2
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


@pytest.mark.parametrize("case_name", star.TIER2_CASES)
def test_upstream_tier2_test_suite_case_reproduction_and_restart_parity(
    case_name: str,
) -> None:
    """When `MESA_DIR` is present, verify each of the 17 upstream Tier 2
    `star/test_suite/<case>` directories executes cleanly, passes physical
    target assertions, and achieves 0 ULP `.photo` restart parity.
    """
    mesa_dir = os.environ.get("MESA_DIR", "")
    if not mesa_dir:
        pytest.skip("MESA_DIR environment variable is not set")
    case_dir = pathlib.Path(mesa_dir) / "star" / "test_suite" / case_name
    if not case_dir.is_dir():
        pytest.skip(f"Upstream case directory not found: {case_dir}")

    rep = e2e.run_test_suite_case(
        case_dir,
        skip_optional=True,
        max_steps_per_stage=2,
        verify_restart=True,
    )
    assert rep["case_name"] == case_name
    assert rep["tier"] == 2
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
