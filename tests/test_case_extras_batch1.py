# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit, relocation, and adapter tests for `#231` Batch 1 `case_extras`."""

from __future__ import annotations

import math
import pathlib
import shutil
from typing import Any

import pytest

from flexmesa import _rust, constants
from flexmesa.testing import harness_runner
from tests import case_extras
from tests.case_extras import (
    star_burn_flame,
    star_extras,
    star_mixing_init,
    star_wd_hooks,
    star_winds_hooks,
)

_STOCK_CASES_BATCH1: tuple[str, ...] = (
    "make_zams",
    "make_zams_low_mass",
    "make_zams_ultra_high_mass",
    "high_z",
    "7M_prems_to_AGB",
    "wd_c_core_ignition",
    "16M_conv_premix",
    "16M_predictive_mix",
    "custom_colors",
)

_PORTED_CASES_BATCH1: tuple[tuple[str, type[Any]], ...] = (
    ("other_physics_hooks", star_extras.OtherPhysicsHooksStarExtras),
    ("cburn_inward", star_extras.CburnInwardStarExtras),
    (
        "extended_convective_penetration",
        star_extras.ExtendedConvectivePenetrationStarExtras,
    ),
    ("5M_cepheid_blue_loop", star_extras.CepheidBlueLoopStarExtras),
    ("wd_acc_small_dm", star_extras.WdAccSmallDmStarExtras),
    ("conductive_flame", star_extras.ConductiveFlameStarExtras),
    ("carbon_kh", star_extras.CarbonKhStarExtras),
    ("magnetic_braking", star_extras.MagneticBrakingStarExtras),
    ("wd_aic", star_extras.WdAicStarExtras),
    ("make_env", star_extras.MakeEnvStarExtras),
    ("make_o_ne_wd", star_extras.MakeONeWdStarExtras),
    ("adjust_net", star_extras.AdjustNetStarExtras),
    ("make_co_wd", star_extras.MakeCoWdStarExtras),
    ("tzo", star_extras.TzoStarExtras),
    ("pisn", star_extras.PisnStarExtras),
)


def _copy_src_only(case_name: str, dst_dir: pathlib.Path) -> pathlib.Path:
    """Copy `<case_dir>/src` into `dst_dir/src` for relocation checks."""
    orig = harness_runner.resolve_case_dir("star", case_name)
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(orig / "src", dst_dir / "src")
    return dst_dir


@pytest.mark.parametrize("case_name", _STOCK_CASES_BATCH1)
def test_batch1_stock_cases_classify_as_stock_under_relocation(
    tmp_path: pathlib.Path, case_name: str
) -> None:
    """Reviewed zero-logic templates classify as `stock` under any folder."""
    orig = harness_runner.resolve_case_dir("star", case_name)
    assert case_extras.classify_case_extras("star", orig) == "stock"
    assert case_extras.resolve_case_extras("star", orig) is None
    moved = _copy_src_only(case_name, tmp_path / "renamed_stock_case")
    assert case_extras.classify_case_extras("star", moved) == "stock"


@pytest.mark.parametrize(("case_name", "expected_cls"), _PORTED_CASES_BATCH1)
def test_batch1_ported_cases_resolve_by_fortran_content_only(
    tmp_path: pathlib.Path, case_name: str, expected_cls: type[Any]
) -> None:
    """Handlers match from Fortran text only; unserved ones are not attached.

    A handler outside `ADAPTER_SERVED_STAR_HANDLERS` reads state the adapter
    does not provide, so it is matched but not attached and its case stays
    `unported`.
    """
    served = expected_cls.__name__ in case_extras.ADAPTER_SERVED_STAR_HANDLERS
    orig = harness_runner.resolve_case_dir("star", case_name)
    assert case_extras.classify_case_extras("star", orig) == (
        "ported" if served else "unported"
    )
    assert isinstance(case_extras.match_case_extras("star", orig), expected_cls)
    moved = _copy_src_only(case_name, tmp_path / "arbitrary_case_dir")
    assert isinstance(
        case_extras.match_case_extras("star", moved), expected_cls
    )
    for case_dir in (orig, moved):
        attached = case_extras.resolve_case_extras("star", case_dir)
        if served:
            assert isinstance(attached, expected_cls)
        else:
            assert attached is None
    (moved / "src" / "run_star_extras.f90").unlink()
    assert case_extras.match_case_extras("star", moved) is None


def test_cepheid_blue_loop_crossings_and_restart_guard() -> None:
    """`CepheidBlueLoopStarExtras` terminates on the 3rd blue-edge crossing."""
    h = star_extras.CepheidBlueLoopStarExtras()
    with pytest.raises(NotImplementedError, match="#150"):
        h.extras_startup({}, restart=True)
    h.extras_startup({}, restart=False)
    with pytest.raises(KeyError):
        h.extras_check_model({"x_ctrl": [0.0] * 10})

    # Red edge at logT = 3.7, blue edge at logT = 3.85 on logL in [3.0, 4.0].
    x_ctrl = [0.0, 3.85, 4.0, 3.85, 3.0, 3.70, 4.0, 3.70, 3.0]
    snap = {"x_ctrl": x_ctrl, "surface_l_lsun": 1.0e3, "model_number": 5}
    assert h.extras_check_model({**snap, "teff": 4500.0}) == "keep_going"
    assert h.extras_check_model({**snap, "teff": 6000.0}) == "keep_going"
    assert h.extras_check_model({**snap, "teff": 7500.0}) == "keep_going"
    assert h.extras_check_model({**snap, "teff": 6800.0}) == "terminate"
    assert h.extras_finish_step(snap) == "keep_going"


def test_cburn_inward_ignition_columns_and_halfway_stop() -> None:
    """`CburnInwardStarExtras` records ignition and stops at half `ign_mass`."""
    h = star_extras.CburnInwardStarExtras()
    h.extras_startup({"x_integer_ctrl": [1]}, restart=True)
    with pytest.raises(NotImplementedError, match="#150"):
        h.extras_startup({"x_integer_ctrl": [2]}, restart=True)
    h.extras_startup({}, restart=False)

    msun = constants.MSUN
    base = {
        "x_integer_ctrl": [2],
        "nz": 3,
        "co_core_mass": 1.05,
        "m": [1.02 * msun, 0.80 * msun, 0.35 * msun],
        "rho": [1.0e5, 2.0e6, 5.0e6],
        "net_nuclear_energy": [-1.0, 10.0, 10.0],
        "xa_ne20": [0.1, 0.4, 0.1],
        "xa_na23": [0.05, 0.2, 0.2],
        "xa_mg24": [0.02, 0.1, 0.05],
        "xa_he4": [0.0, 1.0e-10, 0.0],
    }
    assert h.extras_finish_step(base) == "keep_going"
    cols = dict(h.extra_history_columns(base))
    assert math.isclose(cols["ign_mass"], 0.80, rel_tol=0.0, abs_tol=1.0e-14)
    assert math.isclose(
        cols["log(ign_den)"], _rust.safe_log10(2.0e6), abs_tol=1.0e-14
    )
    inward = {**base, "xa_ne20": [0.1, 0.1, 0.5]}
    assert h.extras_finish_step(inward) == (
        "terminate",
        "Terminate as flame reached half way",
    )
    h.extras_after_evolve(inward)
    assert math.isclose(h.testhub_extras["ign_co_core_mass"], 1.05)


def test_conductive_flame_missing_hooks_and_surface_pt() -> None:
    """`ConductiveFlameStarExtras` guards missing library callbacks."""
    h = star_extras.ConductiveFlameStarExtras()
    with pytest.raises(NotImplementedError, match="#161"):
        h.extras_controls({})
    with pytest.raises(NotImplementedError, match="#154"):
        h.other_build_initial_model(object())
    with pytest.raises(NotImplementedError, match="#150"):
        h.extras_startup({}, restart=True)
    assert h.constant_surface_pt()["dlnP_dL"] == 0.0


def test_conductive_flame_properties_speed_and_halfway_stop() -> None:
    """`ConductiveFlameStarExtras` tracks flame position, width, and speed."""
    h = star_extras.ConductiveFlameStarExtras()
    h.extras_startup({}, restart=False)
    r_arr = [100.0, 80.0, 60.0, 35.0, 10.0]
    eps1 = [0.0, 0.0, 2.0, 100.0, 5.0]
    assert star_burn_flame.flame_properties(eps1, r_arr) == (35.0, 50.0)
    s1 = {"nz": 5, "r": r_arr, "eps_nuc": eps1, "star_age": 1.0}
    assert h.extras_finish_step(s1) == "keep_going"
    assert dict(h.extra_history_columns(s1)) == {
        "r_flame": 35.0,
        "l_flame": 50.0,
    }
    x_ctrl = [0.0] * 8 + [25.0 / (2.0 * constants.SECYER), 45.0]
    s2 = {
        "nz": 5,
        "r": r_arr,
        "eps_nuc": [0.0, 2.0, 100.0, 5.0, 0.0],
        "star_age": 3.0,
        "x_ctrl": x_ctrl,
    }
    assert h.extras_finish_step(s2) == (
        "terminate",
        "flame reached halfway point",
    )
    h.extras_after_evolve(s2)
    assert h.within_tolerance is True
    assert len(h.extra_history_header_items(s2)) == 5


def _sample_carbon_kh_state() -> dict[str, Any]:
    """Build a 2-zone `carbon_kh` snapshot with cell 1 in an EOS blend."""
    zeros = [0.0, 0.0]
    return {
        "nz": 2,
        "dt": 10.0,
        "total_energy_end": -100.0,
        "ergs_error": [2.0, 3.0],
        "dm": [4.0, 5.0],
        "energy": [12.0, 20.0],
        "energy_start": [10.0, 18.0],
        "t_start": [1.0, 2.0],
        "t": [1.0, 2.0],
        "cv_start": [1.0, 1.0],
        "cv": [1.0, 1.0],
        "dxh_lnt": [0.5, 0.25],
        "rho_start": [2.0, 2.0],
        "rho": [2.0, 2.0],
        "de_drho_start": [0.5, 0.5],
        "de_drho": [0.5, 0.5],
        "dxh_lnd": [0.5, 0.5],
        "eos_frac_opal_scvh": zeros,
        "eos_frac_helm": [1.0, 0.4],
        "eos_frac_skye": zeros,
        "eos_frac_pc": zeros,
        "eos_frac_cms": zeros,
        "eos_frac_free_eos": [0.0, 0.6],
    }


def test_carbon_kh_energy_accumulators_and_columns() -> None:
    """`CarbonKhStarExtras` accumulates blend and non-blend EOS residuals."""
    h = star_extras.CarbonKhStarExtras()
    with pytest.raises(NotImplementedError, match="#150"):
        h.extras_startup({}, restart=True)
    h.extras_startup({}, restart=False)
    assert h.extras_start_step({}) == "keep_going"
    snap = _sample_carbon_kh_state()
    assert h.extras_finish_step(snap) == "keep_going"
    assert h.extras_finish_step(snap) == "keep_going"
    hcols = dict(h.extra_history_columns(snap))
    assert math.isclose(hcols["rel_e_res_err_run"], -0.10, abs_tol=1.0e-15)
    assert math.isclose(hcols["rel_e_eos_err_step"], -0.09, abs_tol=1.0e-15)
    assert math.isclose(
        hcols["rel_e_eos_err_run_blend"], -0.10, abs_tol=1.0e-15
    )
    pcols = dict(h.extra_profile_columns(snap))
    assert pcols["eps_eos"] == [10.0, 10.0] and pcols["rse_eps_grav"] == [
        -10.0,
        -10.0,
    ]


def test_carbon_kh_preserves_left_to_right_floating_point_association() -> None:
    """`_cell_eos_energy_diff` evaluates `(de - term_t) - term_rho` cleanly."""
    # In IEEE-754 float64: (1.0 - 1.0) - 1e-16 == -1e-16, whereas the
    # reassociated 1.0 + (-1.0 - 1e-16) underflows to 0.0.
    snap = {
        "nz": 1,
        "dt": 1.0,
        "energy": [2.0],
        "energy_start": [1.0],
        "t_start": [1.0],
        "t": [1.0],
        "cv_start": [1.0],
        "cv": [1.0],
        "dxh_lnt": [1.0],
        "rho_start": [1.0],
        "rho": [1.0],
        "de_drho_start": [1.0],
        "de_drho": [1.0],
        "dxh_lnd": [1.0e-16],
        "eos_frac_helm": [1.0],
        "eos_frac_free_eos": [0.0],
    }
    h = star_extras.CarbonKhStarExtras()
    pcols = dict(h.extra_profile_columns(snap))
    assert pcols["eps_eos"] == [-1.0e-16]
    assert pcols["rse_eps_grav"] == [-1.0]


def test_pisn_remove_surface_guards_and_gamma1_pulse() -> None:
    """`PisnStarExtras` guards `#153` and tracks `gamma1_cntr_pulse_start`."""
    pisn = star_extras.PisnStarExtras()
    pisn.extras_controls({"use_other_remove_surface": False})
    with pytest.raises(NotImplementedError, match="#153"):
        pisn.extras_controls({"use_other_remove_surface": True})
    pisn.other_remove_surface({"star_ejecta_mass": 0.05 * constants.MSUN})
    with pytest.raises(NotImplementedError, match="#153"):
        pisn.other_remove_surface({"star_ejecta_mass": 0.2 * constants.MSUN})
    pisn.extras_startup({"x_integer_ctrl": [1]}, restart=True)
    with pytest.raises(NotImplementedError, match="#150"):
        pisn.extras_startup({"x_integer_ctrl": [7]}, restart=True)
    pisn.extras_startup({}, restart=False)
    assert pisn.extras_start_step({}) == "keep_going"

    snap = {
        "nz": 2,
        "peos": [2.0, 6.0],
        "dm": [1.0, 1.0],
        "rho": [1.0, 3.0],
        "gamma1": [1.30, 1.26],
        "total_energy": 1.0e48,
        "x_integer_ctrl": [7],
    }
    expected_int = 0.5 * ((1.30 - 4.0 / 3.0) + (1.26 - 4.0 / 3.0))
    assert math.isclose(
        pisn.extra_history_columns(snap)[0][1], expected_int, abs_tol=1.0e-15
    )
    assert pisn.extras_finish_step(snap) == ("terminate", "Star is unbound")
    pisn.extras_after_evolve(snap)
    assert math.isclose(
        pisn.testhub_extras["gamma1_cntr_pulse_start"],
        1.26 - 4.0 / 3.0,
        abs_tol=1.0e-15,
    )


def test_adjust_net_records_num_isos_in_part2() -> None:
    """`AdjustNetStarExtras` records `num_isos` only in part 2."""
    adj = star_extras.AdjustNetStarExtras()
    adj.extras_after_evolve({"x_integer_ctrl": [1]})
    assert not adj.testhub_extras
    adj.extras_after_evolve({"x_integer_ctrl": [2], "species": 62})
    assert adj.testhub_extras == {"num_isos": 62.0}


def test_wd_acc_small_dm_accretion_buildup_and_profile_checks() -> None:
    """`WdAccSmallDmStarExtras` verifies accreted `H1` mass and layer jump."""
    msun = constants.MSUN
    acc = star_extras.WdAccSmallDmStarExtras()
    with pytest.raises(NotImplementedError, match="#150"):
        acc.extras_startup({"accretion_h1": 0.7}, restart=True)
    init = {
        "accretion_h1": 0.7,
        "nz": 5,
        "dm": [0.5 * msun] * 5,
        "xa_h1": [0.1] * 5,
        "zbar": [6.0] * 5,
        "abar": [12.0] * 5,
    }
    acc.extras_startup(init, restart=False)
    bad_step = {
        **init,
        "star_age": 1.0,
        "star_mdot": 1.0,
        "xa_h1": [0.7, 0.7, 0.4, 0.1, 0.1],
    }
    with pytest.raises(RuntimeError, match="check_initial_accretion_ok"):
        acc.extras_finish_step(bad_step)
    exact_step = {**bad_step, "xa_h1": [0.7, 0.7, 0.3, 0.1, 0.1]}
    assert acc.extras_finish_step(exact_step) == "keep_going"
    assert math.isclose(
        acc.extra_history_columns(exact_step)[0][1], 0.70, abs_tol=1.0e-14
    )
    assert acc.extra_profile_columns(exact_step)[0][1] == [0.5] * 5
    acc.extras_after_evolve(exact_step)
    assert acc.all_ok is True


def test_wd_aic_and_make_one_wd_hooks() -> None:
    """`WdAicStarExtras` and `MakeONeWdStarExtras` match upstream thresholds."""
    aic = star_extras.WdAicStarExtras()
    avg = star_wd_hooks.center_avg_x([0.4, 0.3, 0.3], [0.0, 0.2, 0.8], 0.45)
    assert math.isclose(avg, 0.6, abs_tol=1.0e-15)
    assert aic.extras_check_model({"x_integer_ctrl": [0]}) == "keep_going"
    res_aic = aic.extras_check_model(
        {
            "x_integer_ctrl": [1],
            "nz": 2,
            "dq": [0.5, 0.5],
            "center_avg_value_dq": 0.5,
            "xa_mg24": [0.0, 0.0],
            "xa_na24": [0.0, 0.0],
            "xa_ne20": [0.4, 0.4],
            "log_center_density": 9.8,
            "log_center_temperature": 8.9,
        }
    )
    assert res_aic[0] == "terminate" and "PASS: A=20" in res_aic[1]

    one = star_extras.MakeONeWdStarExtras()
    assert (
        one.extras_finish_step(
            {
                "x_integer_ctrl": [2],
                "x_ctrl": [1.0, 1.0e4, -2.0],
                "surface_l_lsun": 0.5,
                "teff": 2.0e4,
                "power_he_burn": 1.0e-3,
            }
        )
        == "terminate"
    )


def test_make_co_wd_ignition_diffusion_and_surface_cut_guards() -> None:
    """`MakeCoWdStarExtras` checks `icc`, toggles diffusion, and guards
    `#153`.
    """
    msun = constants.MSUN
    co = star_extras.MakeCoWdStarExtras()
    assert co.extras_startup(
        {"x_integer_ctrl": [3], "power_he_burn": 10.0}, restart=False
    ) == {"do_element_diffusion": True}
    assert co.extras_startup(
        {"x_integer_ctrl": [3], "power_he_burn": 2.0e3}, restart=False
    ) == {"do_element_diffusion": False}
    assert co.extras_check_model(
        {
            "l_by_category_icc": 1.0e2,
            "x_integer_ctrl": [3],
            "power_he_burn": 10.0,
        }
    ) == ("keep_going", {"do_element_diffusion": True})
    res_term = co.extras_check_model(
        {
            "l_by_category_icc": 2.0e4,
            "x_integer_ctrl": [3],
            "power_he_burn": 5.0e3,
        }
    )
    assert isinstance(res_term, dict)
    assert res_term["action"] == "terminate"
    assert res_term["updates"] == {"do_element_diffusion": False}
    co.extras_startup({"x_integer_ctrl": [2]}, restart=False)
    res_part2 = co.extras_check_model(
        {"l_by_category_icc": 2.0e4, "x_integer_ctrl": [2]}
    )
    assert isinstance(res_part2, tuple) and res_part2[0] == "terminate"
    with pytest.raises(NotImplementedError, match="#153"):
        co.extras_finish_step(
            {
                "x_integer_ctrl": [2],
                "x_ctrl": [0.01],
                "nz": 3,
                "x_h1": [0.7, 0.7, 0.0],
                "m": [0.65 * msun, 0.60 * msun, 0.55 * msun],
                "he_core_boundary_h1_fraction": 0.1,
            }
        )


def test_extended_convective_penetration_and_make_env() -> None:
    """`ExtendedConvectivePenetrationStarExtras` and `MakeEnvStarExtras`."""
    ecp = star_extras.ExtendedConvectivePenetrationStarExtras()
    ecp.extras_controls({"create_pre_main_sequence_model": True})
    with pytest.raises(NotImplementedError, match="#155"):
        ecp.extras_controls({"create_pre_main_sequence_model": False})
    pen_state = {
        "nz": 4,
        "overshoot_f": [0.1],
        "overshoot_f0": [0.02],
        "x_ctrl": [0.2],
        "overshoot_d0": [0.0],
        "overshoot_delta0": [1.0],
        "star_mass": 1.5,
        "overshoot_mass_full_on": [2.0],
        "overshoot_mass_full_off": [1.0],
        "conv_bdy_hp": [10.0],
        "over_bdy_k": [2],
        "over_bdy_r": [100.0],
        "over_bdy_d": [1.0e12],
        "over_bdy_vc": [1.0e4],
        "top_conv_bdy": [True],
        "overshoot_d_min": 1.0e2,
        "r": [105.0, 102.0, 100.2, 95.0],
    }
    k_a, k_b, d_arr, _ = ecp.eval_penetration_profile(pen_state, 0, 0)
    assert (k_a, k_b) == (2, 0)
    assert d_arr[2] == 1.0e12 and 0.0 < d_arr[0] < d_arr[1] < d_arr[2]

    dq1 = 1.0e-15
    r_fac = star_mixing_init.calc_dq_factor(1500, dq1)
    f_lo, _ = star_mixing_init.dq_f(r_fac - 1.0e-10, 1500, dq1)
    f_hi, _ = star_mixing_init.dq_f(r_fac + 1.0e-10, 1500, dq1)
    assert f_lo * f_hi <= 0.0
    menv = star_extras.MakeEnvStarExtras()
    menv.extras_controls({"x_logical_ctrl": [False]})
    with pytest.raises(NotImplementedError, match="#154"):
        menv.extras_controls({"x_logical_ctrl": [True]})


def test_extended_convective_penetration_peclet_and_after_evolve() -> None:
    """`ExtendedConvectivePenetrationStarExtras` Peclet and 11 post-checks."""
    ecp = star_extras.ExtendedConvectivePenetrationStarExtras()
    msun = constants.MSUN
    state: dict[str, Any] = {
        "create_pre_main_sequence_model": False,
        "nz": 4,
        "m": [1.6 * msun, 1.2 * msun, 0.98 * msun, 0.85 * msun],
        "mixing_type": [7, 2, 2, 1],
        "grada": [0.4] * 4,
        "gradr": [0.2, 0.2, 0.2, 0.5],
        "d_mix": [1.0, 1.0e6, 1.0e12, 1.0e13],
        "min_d_mix": 10.0,
        "num_conv_boundaries": 1,
        "conv_bdy_loc": [2],
        "adjust_mlt_gradt_fraction": [-1.0] * 4,
        "t": [1.0e7] * 4,
        "rho": [10.0] * 4,
        "opacity": [1.0] * 4,
        "cp": [1.0e8] * 4,
        "ln_peos": [35.0] * 4,
        "lnd": [math.log(10.0)] * 4,
        "cgrav": [constants.STANDARD_CGRAV] * 4,
        "m_center": 0.0,
        "xmstar": 2.0 * msun,
        "q": [0.8, 0.6, 0.49, 0.425],
        "r": [2.0e10] * 4,
        "conv_vel": [1.0, 5.0e-3, 1.0e5, 1.0e6],
        "mixing_length_alpha": 1.8,
    }
    fracs = ecp.other_adjust_mlt_gradt_fraction_peclet(state)
    assert fracs[0] == -1.0 and 0.1 <= fracs[1] <= 0.9 and fracs[3] == 1.0
    state["gradt"] = [0.2, fracs[1] * 0.4 + (1.0 - fracs[1]) * 0.2, 0.4, 0.4]
    ecp.extras_after_evolve(state)
    assert ecp.checks_okay is True


def test_tzo_part4_guards_and_tov_other_cgrav() -> None:
    """`TzoStarExtras` computes TOV `other_cgrav` and guards part-4 hydro."""
    tzo = star_extras.TzoStarExtras()
    tzo.extras_controls(
        {
            "x_logical_ctrl": [True],
            "x_integer_ctrl": [4],
            "x_ctrl": [0.5, 0.5, 1.4],
        }
    )
    part4 = {
        "nz": 1,
        "m_center": constants.MSUN,
        "opacity": [0.2],
        "dt": 10.0,
        "star_age": 200.0,
        "v_flag": False,
        "mstar": 5.0 * constants.MSUN,
    }
    with pytest.raises(NotImplementedError, match="#152"):
        tzo.extras_start_step(part4)
    assert (
        tzo.extras_check_model({"m_center": 1.5 * constants.MSUN})[0]
        == "terminate"
    )
    cgrav = tzo.other_cgrav(
        {
            "nz": 2,
            "r": [1.0e11, 5.0e10],
            "m": [2.0 * constants.MSUN, constants.MSUN],
            "rho": [1.0e-4, 1.0e-2],
            "peos": [1.0e8, 1.0e12],
        }
    )
    r_s = 4.0 * constants.STANDARD_CGRAV * constants.MSUN / constants.CLIGHT**2
    assert math.isclose(
        cgrav[0], constants.STANDARD_CGRAV / (1.0 - r_s / 1.0e11), rel_tol=1e-15
    ) and (cgrav[1] > constants.STANDARD_CGRAV)


def test_other_physics_hooks_kudritzki_grafener_and_invalid_scheme() -> None:
    """`OtherPhysicsHooksStarExtras` evaluates Kudritzki and Grafener winds."""
    oph = star_extras.OtherPhysicsHooksStarExtras()
    with pytest.raises(NotImplementedError, match="#161"):
        oph.extras_controls({})
    orig_dir = harness_runner.resolve_case_dir("star", "other_physics_hooks")
    snap = {
        "case_dir": str(orig_dir),
        "model_number": 1,
        "surface_h1": 0.70,
        "surface_he4": 0.28,
        "zbase": 0.02,
    }
    w_kud = oph.other_wind(
        snap,
        1e5 * constants.LSUN,
        30.0 * constants.MSUN,
        10.0 * constants.RSUN,
        3.8e4,
    )
    assert w_kud > 0.0
    oph.wind_scheme = "Grafener"
    w_graf = oph.other_wind(
        snap,
        1e6 * constants.LSUN,
        25.0 * constants.MSUN,
        10.0 * constants.RSUN,
        4.5e4,
    )
    assert w_graf > 0.0
    oph.wind_scheme = "Unknown"
    with pytest.raises(ValueError, match="invalid wind_scheme"):
        oph.other_wind(
            snap,
            1e5 * constants.LSUN,
            30.0 * constants.MSUN,
            10.0 * constants.RSUN,
            3.8e4,
        )


def test_other_physics_hooks_stern51_wind_branches() -> None:
    """`eval_stern51_wind` matches Kudritzki scaling and WR CNO branches."""
    oph = star_extras.OtherPhysicsHooksStarExtras()
    orig_dir = harness_runner.resolve_case_dir("star", "other_physics_hooks")
    snap = {
        "case_dir": str(orig_dir),
        "model_number": 1,
        "surface_h1": 0.70,
        "surface_he4": 0.28,
        "center_h1": 0.50,
        "has_h1_iso": True,
        "zbase": 0.02,
    }
    args = (
        1e5 * constants.LSUN,
        30.0 * constants.MSUN,
        10.0 * constants.RSUN,
        3.8e4,
    )
    w_kud = oph.other_wind(snap, *args)
    oph.wind_scheme = "Stern51"
    w_stern = oph.other_wind(snap, *args)
    znow = 0.02 * (0.5 * (0.0014 / 1.32e-3) + 0.5 * (0.0186 / 0.01868))
    assert math.isclose(
        w_stern, w_kud * _rust.pow(znow / 0.02, 0.69), rel_tol=1.0e-15
    )
    w_wr = star_winds_hooks.eval_stern51_wind(
        {**snap, "surface_h1": 0.1},
        args[0],
        15.0 * constants.MSUN,
        2.0 * constants.RSUN,
        8e4,
        0.1,
        0.04,
    )
    assert w_wr > 0.0


def test_magnetic_braking_torque_conservation_and_spindown_limit() -> None:
    """`MagneticBrakingStarExtras` redistributes capped shell torque."""
    mb = star_extras.MagneticBrakingStarExtras()
    with pytest.raises(NotImplementedError, match="#152"):
        mb.extras_controls({"use_other_timestep_limit": True})
    mb.extras_controls({"use_other_timestep_limit": False})
    mb_snap: dict[str, Any] = {
        "nz": 3,
        "x_ctrl": [1.0e3, 0.05],
        "x_logical_ctrl": [False],
        "v_rot_avg_surf": 2.0e7,
        "omega_avg_surf": 1.0e-4,
        "photosphere_r": 5.0,
        "mstar": 10.0 * constants.MSUN,
        "mstar_dot": -1.0e-7 * constants.MSUN / constants.SECYER,
        "dt": 1.0e11,
        "j_rot": [1.0e14, 1.0e19, 1.0e19],
        "dm_bar": [1.0e32, 5.0e33, 5.0e33],
    }
    ej = mb.other_torque(mb_snap)
    assert math.isclose(ej[0], -1.0e14 / 1.0e11, rel_tol=1.0e-15)
    r_cm = 5.0 * constants.RSUN
    vinf = 1.92 * math.sqrt(
        2.0 * constants.STANDARD_CGRAV * mb_snap["mstar"] / r_cm
    )
    eta = (r_cm * 1.0e3) ** 2 / (abs(mb_snap["mstar_dot"]) * vinf)
    j_dot_expected = (2.0 / 3.0) * mb_snap["mstar_dot"] * 1.0e-4 * r_cm**2 * eta
    torque_applied = sum(ej[k] * mb_snap["dm_bar"][k] for k in range(3))
    assert math.isclose(torque_applied, j_dot_expected, rel_tol=1.0e-12)
    act, ratio = mb.other_timestep_limit(mb_snap, 0.0)
    assert act == "keep_going" and ratio > 0.0
    mb.extras_after_evolve({**mb_snap, "v_rot_avg_surf": 5.0e4})
    assert mb.spindown_ok is True


def test_cepheid_blue_loop_runs_live_through_py_star_extras_adapter(
    tmp_path: pathlib.Path,
) -> None:
    """`5M_cepheid_blue_loop` runs 4 live steps via `PyStarExtrasAdapter`."""
    case_dir = harness_runner.resolve_case_dir("star", "5M_cepheid_blue_loop")
    rep = harness_runner.run_trace_inprocess(
        "star",
        case_dir,
        tmp_path / "out_cepheid_blue_loop",
        max_steps=4,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert rep["total_steps"] == 4
