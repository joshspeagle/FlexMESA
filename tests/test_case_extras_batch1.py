# SPDX-License-Identifier: GPL-3.0-or-later
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
    """`ConductiveFlameStarExtras` builds isothermal balls and guards
    unexposed surface-PT/photo hooks.
    """
    h = star_extras.ConductiveFlameStarExtras()
    with pytest.raises(NotImplementedError, match="#161"):
        h.extras_controls({})
    with pytest.raises(NotImplementedError, match="#150"):
        h.extras_startup({}, restart=True)
    assert h.constant_surface_pt()["dlnP_dL"] == 0.0

    class _FakeBuildCtx:
        def __init__(self) -> None:
            self.x_ctrl = [
                10.0,
                1.0e10,
                2.0e8,
                0.2,
                5.0e9,
                0.5,
                0.5,
                0.0,
            ]
            self.x_integer_ctrl = [10]
            self.built: dict[str, Any] = {}

        def net_info(self) -> tuple[list[int], list[str], dict[str, int]]:
            return (
                [4, 8, 12],
                ["c12", "o16", "ne20"],
                {"c12": 0, "o16": 1, "ne20": 2},
            )

        def eos_dt_get(
            self, _rho: float, _temp: float, _xa: list[float]
        ) -> dict[str, float]:
            return {"ln_pgas": 64.5}

        def build_star_model(self, *args: Any, **kwargs: Any) -> str:
            self.built = {"args": args, "kwargs": kwargs}
            return "mock_star_model"

    bad_ctx = _FakeBuildCtx()
    bad_ctx.x_integer_ctrl = [1]
    with pytest.raises(ValueError, match="nz >= 2"):
        h.other_build_initial_model(bad_ctx)

    ctx = _FakeBuildCtx()
    res = h.other_build_initial_model(ctx)
    assert res == "mock_star_model"
    assert math.isclose(h.constant_ln_p, 64.5)
    assert math.isclose(h.constant_ln_t, _rust.safe_log(2.0e8))
    _lnd, ln_t, _ln_r, lum, dq, xa_grid = ctx.built["args"]
    assert len(dq) == 10
    assert math.isclose(sum(dq), 1.0, rel_tol=1.0e-12)
    assert all(val == 0.0 for val in lum)
    assert math.isclose(_rust.crmath_exp(ln_t[9]), 5.0e9, rel_tol=1.0e-12)
    assert math.isclose(_rust.crmath_exp(ln_t[0]), 2.0e8, rel_tol=1.0e-12)
    assert all(row == [0.5, 0.5, 0.0] for row in xa_grid)


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
    assert ecp.other_adjust_mlt_gradt_fraction({}) is None
    assert ecp.other_overshooting_scheme({}, 0, 0) is None
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
    assert menv.extras_controls({"x_logical_ctrl": [False]}) is None
    # Only the recorded `make_env` source is served (`#453`).
    assert star_mixing_init.match_make_env_extras("") is None


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


def test_tzo_part4_updates_termination_and_tov_other_cgrav() -> None:
    """`TzoStarExtras` updates part-4 core/hydro, terminates, and sets TOV."""
    tzo = star_extras.TzoStarExtras()
    tzo.extras_controls(
        {
            "x_logical_ctrl": [True],
            "x_integer_ctrl": [3],
            "x_ctrl": [0.5, 0.5, 1.4],
        }
    )
    assert tzo.extras_start_step({}) == "keep_going"
    assert (
        tzo.extras_check_model({"m_center": 1.5 * constants.MSUN})
        == "keep_going"
    )

    tzo.extras_controls(
        {
            "x_logical_ctrl": [True],
            "x_integer_ctrl": [4],
            "x_ctrl": [0.5, 0.5, 1.4],
        }
    )
    part4: dict[str, Any] = {
        "nz": 1,
        "m_center": constants.MSUN,
        "opacity": [0.2],
        "dt": 10.0,
        "star_age": 200.0,
        "v_flag": False,
        "mstar": 5.0 * constants.MSUN,
    }
    res_hydro = tzo.extras_start_step(part4)
    assert isinstance(res_hydro, tuple) and res_hydro[0] == "keep_going"
    updates = res_hydro[1]
    l_edd = (
        4.0
        * constants.PI
        * constants.STANDARD_CGRAV
        * constants.MSUN
        * constants.CLIGHT
    ) / 0.2
    mdot_c = 0.5 * l_edd / (constants.CLIGHT * constants.CLIGHT)
    m_center_exp = constants.MSUN + mdot_c * 10.0
    assert math.isclose(updates["m_center"], m_center_exp, rel_tol=0.0)
    assert math.isclose(
        updates["xmstar"], 5.0 * constants.MSUN - m_center_exp, rel_tol=0.0
    )
    assert math.isclose(updates["l_center"], 0.5 * l_edd, rel_tol=0.0)
    assert updates["v_flag"] is True

    # `v_flag` is omitted when age <= 100, v_flag is True, or use_hydro=False.
    res_young = tzo.extras_start_step({**part4, "star_age": 100.0})
    assert isinstance(res_young, tuple) and "v_flag" not in res_young[1]
    res_v_on = tzo.extras_start_step({**part4, "v_flag": True})
    assert isinstance(res_v_on, tuple) and "v_flag" not in res_v_on[1]
    tzo_no_hydro = star_extras.TzoStarExtras()
    tzo_no_hydro.extras_controls(
        {
            "x_logical_ctrl": [False],
            "x_integer_ctrl": [4],
            "x_ctrl": [0.5, 0.5, 1.4],
        }
    )
    res_no_hydro = tzo_no_hydro.extras_start_step(part4)
    assert isinstance(res_no_hydro, tuple) and "v_flag" not in res_no_hydro[1]

    assert (
        tzo.extras_check_model({"m_center": 1.3 * constants.MSUN})
        == "keep_going"
    )
    assert tzo.extras_check_model({"m_center": 1.5 * constants.MSUN}) == {
        "action": "terminate",
        "termination_code_slot": "t_xtra1",
        "termination_code_str": "PASS: Have reached requested NS mass",
    }
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


def test_star_snapshot_l_center_and_apply_center_v_flag_updates(
    tmp_path: pathlib.Path,
) -> None:
    """Verify ``l_center`` snapshots and ``v_flag`` / center updates.

    Checks ``star_extras_snapshot`` and ``star_apply_extras_updates``.
    """
    model = _rust.RustStarModel.from_zams(1.0, 0.02)
    snap0 = _rust.star_extras_snapshot(
        model, str(tmp_path), include_history=False
    )
    assert "l_center" in snap0 and float(snap0["l_center"]) == 0.0
    assert snap0["v_flag"] is False

    m_c = 0.25 * constants.MSUN
    xm = float(snap0["mstar"]) - m_c
    l_c = 1.0e37
    _rust.star_apply_extras_updates(
        model,
        {
            "m_center": m_c,
            "xmstar": xm,
            "l_center": l_c,
            "v_flag": True,
        },
    )
    snap1 = _rust.star_extras_snapshot(
        model, str(tmp_path), include_history=False
    )
    assert snap1["v_flag"] is True
    assert math.isclose(float(snap1["m_center"]), m_c, rel_tol=0.0)
    assert math.isclose(float(snap1["xmstar"]), xm, rel_tol=0.0)
    assert math.isclose(float(snap1["l_center"]), l_c, rel_tol=0.0)
    # Matching upstream `tzo` (`extras_start_step`), mutating center mass
    # scalars leaves `m`/`dm`/`m_grav` unchanged until `set_vars` runs.
    assert list(snap1["m"]) == list(snap0["m"])

    _rust.star_apply_extras_updates(model, {"v_flag": False})
    snap2 = _rust.star_extras_snapshot(
        model, str(tmp_path), include_history=False
    )
    assert snap2["v_flag"] is False

    # Like upstream, non-positive scalar assignments are accepted directly
    # without synthetic bounds errors.
    _rust.star_apply_extras_updates(
        model, {"m_center": -1.0, "xmstar": 0.5, "l_center": -2.0}
    )
    snap3 = _rust.star_extras_snapshot(
        model, str(tmp_path), include_history=False
    )
    assert math.isclose(float(snap3["m_center"]), -1.0, rel_tol=0.0)
    assert math.isclose(float(snap3["xmstar"]), 0.5, rel_tol=0.0)
    assert math.isclose(float(snap3["l_center"]), -2.0, rel_tol=0.0)


def test_star_directive_custom_termination_code_and_validation(
    tmp_path: pathlib.Path,
) -> None:
    """`parse_star_directive` accepts `t_xtra1..9` slots on `terminate`
    dicts, rejects free-form/invalid ones, and preserves tuple/string defaults.
    """
    case_dir = tmp_path / "term_code_case"
    case_dir.mkdir()
    (case_dir / "inlist").write_text(
        "&star_job\n"
        "/\n"
        "&kap\n"
        "  Zbase = 0.02\n"
        "/\n"
        "&controls\n"
        "  initial_mass = 1.0\n"
        "  max_model_number = 5\n"
        "/\n",
        encoding="utf-8",
    )

    class _ReturnDirectiveProbe:
        def __init__(self, result: object) -> None:
            self._result = result

        def extras_start_step(self, _state: dict[str, Any]) -> object:
            return self._result

    rep_custom = _rust.trace_star_case(
        str(case_dir),
        str(tmp_path / "out_custom"),
        extras=_ReturnDirectiveProbe(
            {
                "action": "terminate",
                "termination_code_slot": "t_xtra1",
                "termination_code_str": "PASS: Have reached requested NS mass",
                "message": "diagnostic note",
            }
        ),
    )
    assert rep_custom["total_steps"] == 0
    assert (
        rep_custom["termination_code"] == "PASS: Have reached requested NS mass"
    )

    rep_default_slot = _rust.trace_star_case(
        str(case_dir),
        str(tmp_path / "out_slot_default"),
        extras=_ReturnDirectiveProbe(
            {"action": "terminate", "termination_code_slot": "t_xtra4"}
        ),
    )
    assert rep_default_slot["total_steps"] == 0
    assert (
        rep_default_slot["termination_code"]
        == "customize by setting termination_code_str(t_xtra4)"
    )

    rep_ws = _rust.trace_star_case(
        str(case_dir),
        str(tmp_path / "out_ws"),
        extras=_ReturnDirectiveProbe(
            {
                "action": "terminate",
                "termination_code_slot": "t_xtra2",
                "termination_code_str": "  keep_leading  ",
            }
        ),
    )
    assert rep_ws["total_steps"] == 0
    assert rep_ws["termination_code"] == "  keep_leading"

    rep_tuple = _rust.trace_star_case(
        str(case_dir),
        str(tmp_path / "out_tuple"),
        extras=_ReturnDirectiveProbe(("terminate", "diagnostic note")),
    )
    assert rep_tuple["total_steps"] == 0
    assert rep_tuple["termination_code"] == "extras_start_step"

    for bad_res, pat in (
        (
            {"action": "terminate", "termination_code": "free-form"},
            "free-form 'termination_code' is unsupported",
        ),
        (
            {
                "action": "keep_going",
                "termination_code_slot": "t_xtra1",
                "termination_code_str": "bad",
            },
            "only valid when action == 'terminate'",
        ),
        (
            {"action": "terminate", "termination_code_slot": "t_xtra0"},
            "invalid termination_code_slot",
        ),
        (
            {"action": "terminate", "termination_code_slot": " t_xtra1 "},
            "invalid termination_code_slot",
        ),
        (
            {"action": "terminate", "termination_code_str": "missing slot"},
            "requires 'termination_code_slot'",
        ),
        (
            {
                "action": "terminate",
                "termination_code_slot": "t_xtra1",
                "termination_code_str": "",
            },
            "empty or whitespace-only",
        ),
        (
            {
                "action": "terminate",
                "termination_code_slot": "t_xtra1",
                "termination_code_str": "   ",
            },
            "empty or whitespace-only",
        ),
        (
            {
                "action": "terminate",
                "termination_code_slot": "t_xtra1",
                "termination_code_str": 123,
            },
            "expected str",
        ),
    ):
        with pytest.raises(ValueError, match=pat):
            _rust.trace_star_case(
                str(case_dir),
                str(tmp_path / "out_bad"),
                extras=_ReturnDirectiveProbe(bad_res),
            )

    dt_case_dir = tmp_path / "dt_limit_term_slot"
    dt_case_dir.mkdir()
    (dt_case_dir / "inlist").write_text(
        "&star_job\n"
        "/\n"
        "&kap\n"
        "  Zbase = 0.02\n"
        "/\n"
        "&controls\n"
        "  initial_mass = 1.0\n"
        "  max_model_number = 5\n"
        "  use_other_timestep_limit = .true.\n"
        "/\n",
        encoding="utf-8",
    )

    class _DtLimitSlotProbe:
        def other_timestep_limit(
            self,
            _state: dict[str, Any],
            _skip_hard_limit: bool,
            _dt: float,
        ) -> object:
            return {
                "action": "terminate",
                "termination_code_slot": "t_xtra1",
                "termination_code_str": "disallowed from dt limit",
            }

    with pytest.raises(ValueError, match="only supported by"):
        _rust.trace_star_case(
            str(dt_case_dir),
            str(tmp_path / "out_dt_bad"),
            extras=_DtLimitSlotProbe(),
        )


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


def test_magnetic_braking_torque_conservation_and_spindown_limit(
    tmp_path: pathlib.Path,
) -> None:
    """`MagneticBrakingStarExtras` redistributes capped torque and limits dt."""
    mb = star_extras.MagneticBrakingStarExtras()
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
    lim_res = mb.other_timestep_limit(mb_snap, False, 1.0e11)
    assert lim_res["action"] == "keep_going"
    assert lim_res["dt_limit_ratio"] > 0.0
    assert mb.other_timestep_limit(
        {**mb_snap, "v_rot_avg_surf": 5.0e4}, False, 1.0e11
    ) == {"action": "keep_going", "dt_limit_ratio": 0.0}
    mb.extras_after_evolve({**mb_snap, "v_rot_avg_surf": 5.0e4})
    assert mb.spindown_ok is True

    mb_dir = harness_runner.resolve_case_dir("star", "magnetic_braking")
    attached = case_extras.resolve_case_extras("star", mb_dir)
    assert isinstance(attached, star_extras.MagneticBrakingStarExtras)
    rep = harness_runner.run_trace_inprocess(
        "star",
        mb_dir,
        tmp_path / "out_magnetic_braking",
        max_steps=2,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep["total_steps"]) >= 2


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


def test_carbon_kh_runs_live_through_py_star_extras_adapter(
    tmp_path: pathlib.Path,
) -> None:
    """`carbon_kh` runs live steps through `PyStarExtrasAdapter` (#152/#231)."""
    case_dir = harness_runner.resolve_case_dir("star", "carbon_kh")
    out_dir = tmp_path / "out_carbon_kh"
    rep = harness_runner.run_trace_inprocess(
        "star",
        case_dir,
        out_dir,
        max_steps=5,
        write_profiles=True,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert rep["total_steps"] == 5
    history_lines = (
        (out_dir / "LOGS" / "history.data")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    header = next(
        line.split() for line in history_lines if "model_number" in line.split()
    )
    values = [float(v) for v in history_lines[-1].split()]
    row = dict(zip(header, values, strict=True))
    expected_cols = (
        "rel_e_res_err_step",
        "rel_e_res_err_run",
        "rel_e_eos_err_step",
        "rel_e_eos_err_run",
        "rel_e_eos_err_step_blend",
        "rel_e_eos_err_run_blend",
    )
    for col in expected_cols:
        assert col in row
        assert math.isfinite(row[col])
    assert abs(row["rel_e_res_err_step"]) < 1.0e-6
    assert abs(row["rel_e_eos_err_step"]) < 1.0e-2
    profile_files = sorted((out_dir / "LOGS").glob("profile*.data"))
    assert profile_files, "expected profile*.data output with write_profiles"
    prof_lines = profile_files[-1].read_text(encoding="utf-8").splitlines()
    hdr_idx, prof_header = next(
        (i, ln.split())
        for i, ln in enumerate(prof_lines)
        if "zone" in ln.split()
    )
    prof_rows = [
        dict(zip(prof_header, [float(v) for v in ln.split()], strict=True))
        for ln in prof_lines[hdr_idx + 1 :]
        if ln.strip()
    ]
    for pcol in (
        "eos_frac_HELM",
        "eos_frac_FreeEOS",
        "eps_eos",
        "rse_eps_grav",
    ):
        assert pcol in prof_header
    assert any(abs(r["eps_eos"]) > 0.0 for r in prof_rows)
    assert any(abs(r["rse_eps_grav"]) > 0.0 for r in prof_rows)


def test_other_physics_hooks_dbl_exp_timestep_and_xtrans_mesh() -> None:
    """`OtherPhysicsHooksStarExtras` runs dbl-exp overshoot, timestep, and
    `xtrans` mesh hooks.
    """
    oph = star_extras.OtherPhysicsHooksStarExtras()
    orig_dir = harness_runner.resolve_case_dir("star", "other_physics_hooks")
    oph.read_inlist_overshoot_dbl_exp(orig_dir)
    oph.read_inlist_other_timestep_limit(orig_dir)
    oph.read_inlist_xtrans_mesh_factor(orig_dir)
    assert math.isclose(oph.overshoot_f2[0], 0.1)
    assert math.isclose(oph.overshoot_d2[0], 1.0e10)
    assert math.isclose(oph.delta_lg_t_limit, 1.0e-5)
    assert math.isclose(oph.delta_lg_t_hard_limit, 1.0e-4)
    assert math.isclose(oph.xtra_dist_above_xtrans, 0.2)
    assert oph.other_mesh_functions({}) == []

    # 1. Double-exponential overshoot switches to `(f2, d2)` past `dr2`.
    ov_state: dict[str, Any] = {
        "nz": 4,
        "star_mass": 1.0,
        "overshoot_f": [0.01],
        "overshoot_f0": [0.004],
        "overshoot_d0": [0.0],
        "overshoot_delta0": [1.0],
        "overshoot_mass_full_on": [1.8],
        "overshoot_mass_full_off": [1.1],
        "overshoot_d_min": 1.0e-2,
        "conv_bdy_hp": [1.0e10],
        "over_bdy_k": [1],
        "over_bdy_r": [5.0e10],
        "over_bdy_d": [1.0e12],
        "over_bdy_vc": [1.0e5],
        "top_conv_bdy": [False],
        "r": [6.0e10, 5.0e10, 4.99e10, 4.9e10],
    }
    # Below `mass_full_off`, tapering zeroes `f`, so diffusivity vanishes.
    k_a0, _k_b0, d_off, _vc_off = oph.other_overshooting_scheme(ov_state, 0, 0)
    assert k_a0 == 2
    assert d_off == [0.0, 0.0, 0.0, 0.0]

    ov_state["star_mass"] = 2.0
    k_a, k_b, d_arr, vc_arr = oph.other_overshooting_scheme(ov_state, 0, 0)
    assert (k_a, k_b) == (2, 3)
    assert d_arr[2] > 1.0e10
    assert 0.0 < d_arr[3] < d_arr[2]
    assert vc_arr[2] > vc_arr[3] > 0.0

    # 2. `other_timestep_limit` only acts on `model_number in [29, 42]`.
    assert oph.other_timestep_limit({"model_number": 28}, False, 1.0e5) == {
        "action": "keep_going",
        "dt_limit_ratio": 0.0,
    }
    ts_base: dict[str, Any] = {
        "model_number": 30,
        "num_retries": 0,
        "rotation_flag": False,
        "num_rotation_solver_steps": 0,
        "mstar_dot": 0.0,
        "mstar_dot_old": 0.0,
        "nz": 2,
        "lnt": [10.0, 10.0 + 2.0e-5 * constants.LN10],
        "lnt_start": [10.0, 10.0],
        "time_delta_coeff": 1.0,
    }
    assert oph.other_timestep_limit(ts_base, False, 1.0e5)["action"] == "retry"
    res_soft = oph.other_timestep_limit(
        {**ts_base, "num_retries": 1}, False, 1.0e5
    )
    assert res_soft["action"] == "keep_going"
    assert math.isclose(res_soft["dt_limit_ratio"], 2.0, rel_tol=1.0e-12)
    res_hard = oph.other_timestep_limit(
        {
            **ts_base,
            "num_retries": 1,
            "lnt": [10.0, 10.0 + 2.0e-4 * constants.LN10],
        },
        False,
        1.0e5,
    )
    assert res_hard["action"] == "retry"

    # 3. `other_mesh_delta_coeff_factor` refines around transitions in `20..40`.
    assert (
        oph.other_mesh_delta_coeff_factor({"model_number": 19, "nz": 4}) is None
    )
    assert oph.other_mesh_delta_coeff_factor({"model_number": 25, "nz": 3}) == [
        0.75,
        0.75,
        0.75,
    ]
    oph.xtra_coef_below_xtrans = 0.5
    oph.xtra_dist_below_xtrans = 0.5
    xm_res = oph.other_mesh_delta_coeff_factor(
        {
            "model_number": 25,
            "nz": 4,
            "xa": [[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]],
            "peos": [1.0e12] * 4,
            "rho": [1.0] * 4,
            "grav": [1.0e4] * 4,
            "r": [1.0e10, 0.99e10, 0.986e10, 0.50e10],
        }
    )
    assert xm_res == [0.75, 0.5, 0.5, 0.75]


def test_231_adapter_live_initial_model_and_physics_hooks(
    tmp_path: pathlib.Path,
) -> None:
    """Exercise `other_build_initial_model`, `other_timestep_limit`,
    `other_mesh_delta_coeff_factor`, `other_mesh_functions`, and
    `other_overshooting_scheme` end-to-end through `PyStarExtrasAdapter`.
    """
    # 1. Live `ConductiveFlameStarExtras.other_build_initial_model` via
    #    `PyRspBuildContext.{net_info, eos_dt_get, build_star_model}`.
    flame_dir = tmp_path / "flame_build_init"
    flame_dir.mkdir()
    (flame_dir / "inlist").write_text(
        "&star_job\n"
        "  read_extra_star_job_inlist(1) = .true.\n"
        "  extra_star_job_inlist_name(1) = 'inlist_case'\n"
        "/\n"
        "&eos\n"
        "  read_extra_eos_inlist(1) = .true.\n"
        "  extra_eos_inlist_name(1) = 'inlist_case'\n"
        "/\n"
        "&kap\n"
        "  read_extra_kap_inlist(1) = .true.\n"
        "  extra_kap_inlist_name(1) = 'inlist_case'\n"
        "/\n"
        "&controls\n"
        "  read_extra_controls_inlist(1) = .true.\n"
        "  extra_controls_inlist_name(1) = 'inlist_case'\n"
        "/\n",
        encoding="utf-8",
    )
    (flame_dir / "inlist_case").write_text(
        "&star_job\n"
        "  create_initial_model = .true.\n"
        "  change_net = .true.\n"
        "  new_net_name = 'approx21.net'\n"
        "  set_initial_dt = .true.\n"
        "  seconds_for_initial_dt = 1d-15\n"
        "/\n"
        "&eos\n"
        "  use_Skye = .false.\n"
        "  use_PC = .false.\n"
        "/\n"
        "&kap\n"
        "  Zbase = 0.02\n"
        "/\n"
        "&controls\n"
        "  use_other_build_initial_model = .true.\n"
        "  use_zero_pgas_outer_bc = .true.\n"
        "  use_rsp_l_eqn_outer_bc = .true.\n"
        "  x_integer_ctrl(1) = 30\n"
        "  x_ctrl(1) = 10.0\n"
        "  x_ctrl(2) = 1d10\n"
        "  x_ctrl(3) = 2d8\n"
        "  x_ctrl(4) = 0.05\n"
        "  x_ctrl(5) = 5d9\n"
        "  x_ctrl(6) = 0.5\n"
        "  x_ctrl(7) = 0.5\n"
        "  x_ctrl(8) = 0.0\n"
        "/\n",
        encoding="utf-8",
    )

    flame = star_extras.ConductiveFlameStarExtras()
    observed_init: dict[str, Any] = {}

    class _FlameBuildProbe:
        def other_build_initial_model(self, ctx: Any) -> Any:
            return flame.other_build_initial_model(ctx)

        def extras_start_step(self, state: dict[str, Any]) -> tuple[str, str]:
            observed_init["nz"] = int(state["nz"])
            observed_init["species"] = int(state["species"])
            observed_init["t_center"] = float(state["t"][-1])
            observed_init["t_surf"] = float(state["t"][0])
            observed_init["xa_c12_0"] = float(state["xa_c12"][0])
            return ("terminate", "initial model verified")

    rep_flame = _rust.trace_star_case(
        str(flame_dir),
        str(tmp_path / "flame_out"),
        extras=_FlameBuildProbe(),
    )
    assert rep_flame["termination_code"] == "extras_start_step"
    assert observed_init["nz"] == 30
    assert observed_init["species"] == 21
    assert observed_init["t_center"] > 4.0e9
    assert math.isclose(observed_init["t_surf"], 2.0e8, rel_tol=1.0e-5)
    assert math.isclose(observed_init["xa_c12_0"], 0.5, rel_tol=1.0e-5)
    assert math.isfinite(flame.constant_ln_p)
    assert flame.constant_ln_p > 50.0

    # 2. Live `OtherPhysicsHooksStarExtras` hooks (`other_timestep_limit`,
    #    `other_mesh_delta_coeff_factor`, `other_mesh_functions`,
    #    `other_overshooting_scheme`, `other_wind`) + snapshot updates.
    orig_oph = harness_runner.resolve_case_dir("star", "other_physics_hooks")
    oph_dir = tmp_path / "oph_live"
    oph_dir.mkdir()
    for fname in (
        "inlist_overshoot_dbl_exp",
        "inlist_other_timestep_limit",
        "inlist_other_winds",
        "inlist_xtrans_mesh_factor",
    ):
        shutil.copy2(orig_oph / fname, oph_dir / fname)
    (oph_dir / "inlist").write_text(
        "&star_job\n"
        "  read_extra_star_job_inlist(1) = .true.\n"
        "  extra_star_job_inlist_name(1) = 'inlist_case'\n"
        "/\n"
        "&kap\n"
        "  read_extra_kap_inlist(1) = .true.\n"
        "  extra_kap_inlist_name(1) = 'inlist_case'\n"
        "/\n"
        "&controls\n"
        "  read_extra_controls_inlist(1) = .true.\n"
        "  extra_controls_inlist_name(1) = 'inlist_case'\n"
        "/\n",
        encoding="utf-8",
    )
    (oph_dir / "inlist_case").write_text(
        "&star_job\n"
        "  set_initial_model_number = .true.\n"
        "  initial_model_number = 28\n"
        "/\n"
        "&kap\n"
        "  Zbase = 0.02\n"
        "/\n"
        "&controls\n"
        "  initial_mass = 1.0\n"
        "  initial_z = 0.02\n"
        "  max_model_number = 31\n"
        "  max_years_for_timestep = 1d4\n"
        "  never_skip_hard_limits = .false.\n"
        "  use_other_wind = .true.\n"
        "  use_other_timestep_limit = .true.\n"
        "  overshoot_scheme(1) = 'other'\n"
        "  overshoot_zone_type(1) = 'any'\n"
        "  overshoot_zone_loc(1) = 'any'\n"
        "  overshoot_bdy_loc(1) = 'bottom'\n"
        "  overshoot_f(1) = 0.010\n"
        "  overshoot_f0(1) = 0.004\n"
        "/\n",
        encoding="utf-8",
    )

    oph = star_extras.OtherPhysicsHooksStarExtras()
    hook_calls: dict[str, int] = {
        "mesh_factor": 0,
        "mesh_fcns": 0,
        "timestep": 0,
        "overshoot": 0,
    }
    saw_snapshot_shape: dict[str, Any] = {}

    class _OphAdapterProbe:
        def extras_controls(self, state: dict[str, Any]) -> dict[str, Any]:
            saw_snapshot_shape["atm_t_tau_opacity"] = state["atm_t_tau_opacity"]
            # The model is not loaded when `extras_controls` runs, so `xa` is
            # read in `extras_check_model` below (`#453`).
            return {
                "use_other_mesh_delta_coeff_factor": True,
                "use_other_mesh_functions": True,
                "atm_t_tau_opacity": "varying",
            }

        def other_overshooting_scheme(
            self, state: dict[str, Any], i_bdy: int, j_param: int
        ) -> tuple[int, int, list[float], list[float]]:
            hook_calls["overshoot"] += 1
            return oph.other_overshooting_scheme(state, i_bdy, j_param)

        def other_timestep_limit(
            self, state: dict[str, Any], skip_hard_limit: bool, dt: float
        ) -> dict[str, Any]:
            hook_calls["timestep"] += 1
            return oph.other_timestep_limit(state, skip_hard_limit, dt)

        def other_mesh_delta_coeff_factor(
            self, state: dict[str, Any], factor: list[float]
        ) -> list[float] | None:
            hook_calls["mesh_factor"] += 1
            saw_snapshot_shape["factor_len"] = len(factor)
            return oph.other_mesh_delta_coeff_factor(state, factor)

        def other_mesh_functions(
            self, state: dict[str, Any]
        ) -> list[tuple[str, bool, list[float]]]:
            hook_calls["mesh_fcns"] += 1
            return oph.other_mesh_functions(state)

        def other_wind(
            self,
            state: dict[str, Any],
            l_surf: float,
            m_surf: float,
            r_surf: float,
            t_surf: float,
        ) -> float:
            return oph.other_wind(state, l_surf, m_surf, r_surf, t_surf)

        def extras_check_model(self, state: dict[str, Any]) -> str:
            saw_snapshot_shape["final_opacity_opt"] = state["atm_t_tau_opacity"]
            saw_snapshot_shape["xa_shape"] = tuple(state["xa"].shape)
            return "keep_going"

    rep_oph = _rust.trace_star_case(
        str(oph_dir),
        str(tmp_path / "oph_out"),
        extras=_OphAdapterProbe(),
    )
    assert rep_oph["termination_code"] == "max_model_number"
    assert int(rep_oph["total_steps"]) == 3
    assert int(rep_oph["total_retries"]) >= 1
    assert hook_calls["mesh_factor"] >= 1
    assert hook_calls["mesh_fcns"] >= 1
    assert hook_calls["timestep"] >= 2
    assert hook_calls["overshoot"] >= 1
    assert saw_snapshot_shape["atm_t_tau_opacity"] == "fixed"
    assert saw_snapshot_shape["final_opacity_opt"] == "varying"
    assert saw_snapshot_shape["xa_shape"][1] == 8


def test_alpha_mlt_and_fe_infall_snapshot_hooks(
    tmp_path: pathlib.Path,
) -> None:
    """`extras_py.rs` exposes `alpha_mlt`, `fe_core_infall*`, and vector
    `other_alpha_mlt` (`#231`).
    """
    m = _rust.RustStarModel.from_zams(1.0, 0.02)
    snap0 = _rust.star_extras_snapshot(m, str(tmp_path), include_history=False)
    nz = int(snap0["nz"])
    assert len(snap0["alpha_mlt"]) == nz
    assert float(snap0["fe_core_infall"]) == 0.0
    assert float(snap0["fe_core_infall_limit"]) > 0.0

    vec = [1.8 + 0.1 * (k % 3) for k in range(nz)]
    _rust.star_apply_extras_updates(m, {"alpha_mlt": vec})
    snap1 = _rust.star_extras_snapshot(m, str(tmp_path), include_history=False)
    assert list(snap1["alpha_mlt"]) == pytest.approx(vec)

    with pytest.raises(ValueError, match="got bool"):
        _rust.star_apply_extras_updates(m, {"alpha_mlt": True})
    with pytest.raises(ValueError, match="nz ="):
        _rust.star_apply_extras_updates(m, {"alpha_mlt": [1.5, 1.5]})

    class _VecAlphaProbe:
        def __init__(self) -> None:
            self.seen_alpha_len: int | None = None

        def extras_controls(self, _state: dict[str, Any]) -> dict[str, Any]:
            return {"use_other_alpha_mlt": True}

        def other_alpha_mlt(self, state: dict[str, Any]) -> list[float]:
            n = int(state["nz"])
            self.seen_alpha_len = n
            return [2.1] * n

    adj_dir = harness_runner.resolve_case_dir("star", "adjust_net")
    probe = _VecAlphaProbe()
    rep = harness_runner.run_trace_inprocess(
        "star",
        adj_dir,
        tmp_path / "alpha_probe",
        max_steps=1,
        allow_unsupported_controls=True,
        extras=probe,
    )
    assert int(rep["total_steps"]) == 1
    assert probe.seen_alpha_len is not None and probe.seen_alpha_len > 0
