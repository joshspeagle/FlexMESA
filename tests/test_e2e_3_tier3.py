# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage E2E-3 (#94) verification test suite for all 22 Tier 3 massive-star,
rotation, magnetic-dynamo, magnetospheric-braking, core-collapse, supernova,
pair-instability, neutron-star-envelope, Thorne-Zytkow, and twin-star
`star/test_suite` cases.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from flexmesa import star


def test_tier3_cases_inventory_has_22_unique_cases_disjoint() -> None:
    """Verify TIER3_CASES lists 22 unique cases disjoint from Tiers 1 and 2."""
    assert len(star.TIER3_CASES) == 22
    assert len(set(star.TIER3_CASES)) == 22
    assert set(star.TIER3_CASES).isdisjoint(set(star.TIER1_CASES))
    assert set(star.TIER3_CASES).isdisjoint(set(star.TIER2_CASES))
    expected_cases = {
        "high_mass",
        "hot_cool_wind",
        "20M_z2m2_high_rotation",
        "high_rot_darkening",
        "15M_dynamo",
        "conserve_angular_momentum",
        "accreted_material_j",
        "magnetic_braking",
        "split_burn_big_net",
        "12M_pre_ms_to_core_collapse",
        "20M_pre_ms_to_core_collapse",
        "zams_to_cc_80",
        "make_pre_ccsn_13bvn",
        "ccsn_IIp",
        "pisn",
        "ppisn",
        "hse_riemann",
        "ns_h",
        "ns_he",
        "ns_c",
        "tzo",
        "twin_studies",
    }
    assert set(star.TIER3_CASES) == expected_cases


def test_hot_cool_blended_wind_and_roche_gravity_darkening_invariants() -> None:
    """Verify temperature-blended wind mass-loss and Roche gravity darkening."""
    model = star.StarModel.from_zams(initial_mass=15.0, initial_z=0.02)
    mdot = star.e2e.eval_hot_cool_blended_wind_mdot(model)
    assert mdot < 0.0

    r_rat_low, fp_low, ft_low, t_rat_low = (
        star.e2e.eval_roche_gravity_darkening(0.2)
    )
    r_rat_high, fp_high, ft_high, t_rat_high = (
        star.e2e.eval_roche_gravity_darkening(0.95)
    )
    assert r_rat_high > r_rat_low > 1.0
    assert fp_high < fp_low < 1.0
    assert ft_high < ft_low < 1.0
    assert t_rat_high < t_rat_low < 1.0


def test_spruit_tayler_dynamo_and_magnetic_braking_torque_profiles() -> None:
    """Verify 13-element Spruit-Tayler dynamo diagnostics and braking torque."""
    model = star.StarModel.from_zams(initial_mass=15.0, initial_z=0.02)
    diag = star.e2e.eval_spruit_tayler_dynamo_diagnostics(model)
    assert len(diag) == 13
    assert all(v == v for v in diag)

    jdot_low, v_rot_low = star.e2e.eval_magnetic_braking_torque(model, 500.0)
    jdot_high, v_rot_high = star.e2e.eval_magnetic_braking_torque(model, 5000.0)
    assert len(jdot_low) == model.nz
    assert len(jdot_high) == model.nz
    assert abs(jdot_high[0]) > abs(jdot_low[0]) > 0.0
    assert 0.05 <= v_rot_high <= 0.55
    assert 0.05 <= v_rot_low <= 0.55


def test_core_collapse_duffell_rti_stella_and_pair_instability_states() -> None:
    """Verify Fe-core infall, Duffell RTI shock, STELLA ASCII, and PISN."""
    model = star.StarModel.from_zams(initial_mass=15.0, initial_z=0.02)
    m_fe, v_inf = star.e2e.eval_fe_core_infall_state(model)
    assert m_fe > 1.0
    assert v_inf >= 0.0

    m_sh, d_rti, h1_m = star.e2e.eval_duffell_rti_shock_state(model)
    assert m_sh > 0.0
    assert d_rti > 0.0
    assert 0.0 < h1_m <= 1.0

    hyd, abn = star.e2e.export_stella_ascii(model)
    assert len(hyd.strip().splitlines()) == model.nz + 1
    assert len(abn.strip().splitlines()) == model.nz + 1

    dg1, etot = star.e2e.eval_pair_instability_gamma1_and_energy(model)
    assert dg1 == dg1
    assert etot == etot


def test_tov_effective_cgrav_and_tzo_core_accretion_step() -> None:
    """Verify post-Newtonian TOV effective gravity and TZO core accretion."""
    model = star.StarModel.from_zams(initial_mass=15.0, initial_z=0.02)
    geff = star.e2e.eval_tov_effective_cgrav(model)
    assert len(geff) == model.nz
    assert geff[-1] > 6.67430e-8

    m_core, l_edd = star.e2e.eval_tzo_core_accretion_step(
        model, eta_ledd=1.0, eta_medd=1.0, target_mass_msun=1.4
    )
    assert m_core > 1.351
    assert l_edd > 1.0e37


@pytest.mark.parametrize(
    "case_name",
    [
        "high_mass",
        "hot_cool_wind",
        "high_rot_darkening",
        "15M_dynamo",
        "conserve_angular_momentum",
        "magnetic_braking",
        "12M_pre_ms_to_core_collapse",
        "ccsn_IIp",
        "pisn",
        "hse_riemann",
        "ns_h",
        "tzo",
        "ppisn",
        "twin_studies",
    ],
)
def test_synthetic_tier3_case_execution_and_photo_restart(
    tmp_path: pathlib.Path,
    case_name: str,
) -> None:
    """Verify synthetic offline Tier 3 case execution and 0-ULP restart."""
    case_dir = tmp_path / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    if case_name == "twin_studies":
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\ndo_one\n",
            encoding="utf-8",
        )
        (case_dir / "inlist_multi_stars_job").write_text(
            "&multi_stars_job\n"
            "  num_stars = 2\n"
            "  inlist_names(1) = 'inlist_star1'\n"
            "  inlist_names(2) = 'inlist_star2'\n"
            "/\n",
            encoding="utf-8",
        )
        for sname in ("inlist_star1", "inlist_star2"):
            (case_dir / sname).write_text(
                "&star_job\n/\n&eos\n/\n&kap\n/\n&controls\n"
                "  initial_mass = 20.0\n  initial_z = 0.02\n"
                "  max_model_number = 2\n/\n",
                encoding="utf-8",
            )
    else:
        (case_dir / "rn").write_text(
            "#!/usr/bin/env bash\ndo_one inlist final.mod\n",
            encoding="utf-8",
        )
        extra_ctrl = "  x_integer_ctrl(1) = 4\n" if case_name == "tzo" else ""
        (case_dir / "inlist").write_text(
            "&star_job\n"
            "  save_model_when_terminate = .true.\n"
            "  save_model_filename = 'final.mod'\n"
            "/\n&eos\n/\n&kap\n/\n&controls\n"
            "  initial_mass = 15.0\n"
            "  initial_z = 0.02\n"
            "  max_model_number = 2\n"
            f"{extra_ctrl}"
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
    assert rep["tier"] == 3
    assert rep["passed"] is True
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True


@pytest.mark.parametrize("case_name", star.TIER3_CASES)
def test_upstream_mesa_tier3_test_suite_case_passes_with_restart_parity(
    case_name: str,
) -> None:
    """Run each upstream MESA Tier 3 case when MESA_DIR is available."""
    mesa_dir = os.environ.get("MESA_DIR", "")
    if not mesa_dir:
        pytest.skip("MESA_DIR not set; skipping upstream MESA Tier 3 case")
    case_dir = pathlib.Path(mesa_dir) / "star" / "test_suite" / case_name
    assert case_dir.is_dir(), f"Upstream case directory not found: {case_dir}"

    rep = star.run_test_suite_case(
        case_dir,
        skip_optional=True,
        max_steps_per_stage=2,
        verify_restart=True,
    )
    assert rep["case_name"] == case_name
    assert rep["tier"] == 3
    assert rep["passed"] is True
    assert rep["assertions_passed"] is True
    assert rep["photo_restart_verified"] is True
    assert rep["total_steps"] >= 1
    assert rep["final_teff"] > 0.0
