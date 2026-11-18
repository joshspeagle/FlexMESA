# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stage `E2E-5` (`#96`, closing Epic `#89`) end-to-end verification tests for:
1. All 4 Tier 5/6 `star/test_suite` cases (`check_redo`, `other_physics_hooks`,
   `relax_composition_j_entropy`, `timing`) and their custom `run_star_extras`
   physics hooks (`eval_overshoot_dbl_exp`,
   `eval_grafener_kudritzki_wind_msun_yr`, `verify_step_redo_bit_parity`).
2. All 8 active `binary/test_suite` cases (`double_bh`, `evolve_both_stars`,
   `jdot_gr_check`, `jdot_ls_check`, `jdot_ml_check`, `star_plus_point_mass`,
   `star_plus_point_mass_explicit_mdot`, `wind_fed_bhhmxb`) and their analytical
   orbital-evolution and wind-accretion physics oracles.
3. Complete 107-case active MESA `test_suite` inventory (`91` `star/test_suite`
   + `8` `astero/test_suite` + `8` `binary/test_suite`) plus `32`
   `star/dev_cases_*` pulsation cases.
"""

from __future__ import annotations

import pathlib

import pytest

from flexmesa import binary, star
from flexmesa.testing import known_gaps_gate


def test_e2e_5_complete_106_active_test_suite_and_dev_cases_inventory() -> None:
    """Verify exact 1-to-1 inventory coverage across all 106 active MESA
    `26.4.1` `test_suite` cases (`90` `star` + `8` `astero` + `8` `binary`) and
    `dev_cases_*` (`#141`).
    """
    assert len(star.TIER1_CASES) == 33
    assert len(star.TIER2_CASES) == 16
    assert len(star.TIER3_CASES) == 22
    assert len(star.TIER4_ACTIVE_STAR_CASES) == 15
    assert len(star.TIER5_STAR_CASES) == 4
    assert len(star.ALL_ACTIVE_STAR_CASES) == 90
    assert len(set(star.ALL_ACTIVE_STAR_CASES)) == 90
    assert len(star.TIER4_ASTERO_CASES) == 8
    assert len(binary.TIER5_BINARY_CASES) == 8
    assert (
        len(star.ALL_ACTIVE_STAR_CASES)
        + len(star.TIER4_ASTERO_CASES)
        + len(binary.TIER5_BINARY_CASES)
        == 106
    )
    assert len(star.TIER4_DEV_CASES) == 1


def test_e2e_5_tier5_star_hooks_and_step_redo_reproducibility() -> None:
    """Verify double-exponential overshooting (`other_overshooting_scheme`),
    Gräfener & Kudritzki (2008) / Vink et al. (2001) high-Eddington wind
    (`other_wind`), and bit-identical step redo reproducibility (`check_redo`).
    """
    d_ov_near = star.eval_overshoot_dbl_exp(
        1.0e14, 2.0e7, 1.0e9, 0.015, 0.005, 1.0e12
    )
    d_ov_far = star.eval_overshoot_dbl_exp(
        1.0e14, 8.0e7, 1.0e9, 0.015, 0.005, 1.0e12
    )
    assert d_ov_near > d_ov_far > 0.0

    mdot_gk = star.eval_grafener_kudritzki_wind_msun_yr(
        60.0 * 1.98840987e33,
        1.0e6 * 3.828e33,
        30.0 * 6.957e10,
        40_000.0,
        0.70,
        0.02,
        1.0,
    )
    assert mdot_gk > 0.0

    m = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    m.eval_hydro_vars()
    assert star.verify_step_redo_bit_parity(m)


def test_e2e_5_binary_analytical_physics_oracles() -> None:
    """Verify Peters (1964) gravitational-wave inspiral, Tauris & van den Heuvel
    (2003, Eq. 16.20) non-conservative mass transfer, and Brott et al. (2011)
    massive-star wind oracles (`#96`).
    """
    a_gr = binary.eval_gr_expected_separation_rsun(
        1.0, 1.4, 1.4, 1.0e7 * 3.1558149984e7
    )
    assert 0.0 < a_gr < 1.0

    a_ml = binary.eval_tauris_vdh_expected_separation_rsun(
        10.0, 2.0, 1.5, 0.4, 0.4, 0.19, 2.0
    )
    assert a_ml > 0.0

    w_brott = binary.eval_brott_wind_msun_yr(
        5.0e5, 60.0, 15.0, 45_000.0, 0.70, 0.28, 0.02
    )
    assert w_brott > 0.0


def test_e2e_5_tier5_star_and_binary_suites_execution(
    tmp_path: pathlib.Path,
) -> None:
    """Verify raw Tier 5/6 `star` and `binary` upstream suites reject
    unsupported controls under `#147` and cleaned copies pass E2E checks.
    """
    base = known_gaps_gate.require_mesa_test_inputs_dir()
    if base is None:
        return
    clean_root = tmp_path / "mesa"
    assert (base / "star" / "test_suite").is_dir()
    for case_name in star.TIER5_STAR_CASES:
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
            )
        star.copy_cleaned_case(
            raw_case,
            clean_root / "star" / "test_suite" / case_name,
            suite="star",
        )
    star_root = clean_root / "star" / "test_suite"
    # The cleaned other_physics_hooks inlist still selects
    # overshoot_scheme(1) = 'other', which the engine now rejects instead of
    # silently disabling overshoot (declared gap under #148).
    with pytest.raises(ValueError, match="overshoot_scheme"):
        star.run_test_suite_case(
            star_root / "other_physics_hooks",
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
        )
    star_reps = [
        star.run_test_suite_case(
            star_root / case_name,
            skip_optional=True,
            max_steps_per_stage=2,
            verify_restart=True,
        )
        for case_name in star.TIER5_STAR_CASES
        if case_name != "other_physics_hooks"
    ]
    assert len(star_reps) == 3
    assert all(r["passed"] and r["photo_restart_verified"] for r in star_reps)

    assert (base / "binary" / "test_suite").is_dir()
    for case_name in binary.TIER5_BINARY_CASES:
        raw_case = base / "binary" / "test_suite" / case_name
        with pytest.raises(
            ValueError,
            match=r"(unsupported|unknown|unrecognized).*control",
        ):
            binary.run_binary_test_suite_case(raw_case, max_steps_cap=4)
        star.copy_cleaned_case(
            raw_case,
            clean_root / "binary" / "test_suite" / case_name,
            suite="binary",
        )
    bin_reps = binary.run_binary_suite(clean_root, max_steps_cap=4)
    assert len(bin_reps) == 8
    assert all(r["passed"] for r in bin_reps)
    by_name = {r["case_name"]: r for r in bin_reps}
    assert by_name["jdot_gr_check"]["max_relative_oracle_error"] < 1.0e-2
    assert by_name["jdot_ml_check"]["max_relative_oracle_error"] < 1.0e-4
    assert by_name["jdot_ls_check"]["max_relative_oracle_error"] < 1.0e-12
    assert by_name["wind_fed_bhhmxb"]["max_relative_oracle_error"] < 1.0e-14
