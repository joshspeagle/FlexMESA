# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit, relocation, and live adapter tests for `#231` Batch 2 `case_extras`."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import shutil
import struct
from typing import Any

import pytest

from flexmesa import _rust, constants
from flexmesa.testing import harness_runner
from tests import case_extras
from tests.case_extras import star_atm_pulse, star_extras

_PORTED_CASES_BATCH2: tuple[tuple[str, type[Any]], ...] = (
    ("gyre_in_mesa_bcep", star_extras.GyreInMesaStarExtras),
    ("gyre_in_mesa_envelope", star_extras.GyreInMesaStarExtras),
    ("gyre_in_mesa_ms", star_extras.GyreInMesaStarExtras),
    ("gyre_in_mesa_rsg", star_extras.GyreInMesaStarExtras),
    ("gyre_in_mesa_spb", star_extras.GyreInMesaStarExtras),
    ("gyre_in_mesa_wd", star_extras.GyreInMesaStarExtras),
    (
        "relax_composition_j_entropy",
        star_extras.RelaxCompositionJEntropyStarExtras,
    ),
    ("T_tau_gradr", star_extras.TTauGradrStarExtras),
    ("hot_cool_wind", star_extras.HotCoolWindStarExtras),
    ("accreted_material_j", star_extras.AccretedMaterialJStarExtras),
    ("conv_core_cpm", star_extras.ConvCoreCpmStarExtras),
    ("wd_diffusion", star_extras.WdDiffusionStarExtras),
    ("c13_pocket", star_extras.C13PocketStarExtras),
    ("check_pulse_atm", star_extras.CheckPulseAtmStarExtras),
    ("high_rot_darkening", star_extras.HighRotDarkeningStarExtras),
    ("semiconvection", star_extras.SemiconvectionStarExtras),
    ("hb_2M", star_extras.Hb2MStarExtras),
    ("low_z", star_extras.LowZStarExtras),
    ("make_sdb", star_extras.MakeSdbStarExtras),
    ("1.4M_ms_op_mono", star_extras.MsOpMonoStarExtras),
    ("R_CrB_star", star_extras.RCrBStarExtras),
    (
        "conserve_angular_momentum",
        star_extras.ConserveAngularMomentumStarExtras,
    ),
    ("hse_riemann", star_extras.HseRiemannStarExtras),
)


def _copy_src_only(case_name: str, dst_dir: pathlib.Path) -> pathlib.Path:
    """Copy `<case_dir>/src` into `dst_dir/src` for relocation checks."""
    orig = harness_runner.resolve_case_dir("star", case_name)
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(orig / "src", dst_dir / "src")
    return dst_dir


@pytest.mark.parametrize(("case_name", "expected_cls"), _PORTED_CASES_BATCH2)
def test_batch2_ported_cases_resolve_by_fortran_content_only(
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
    moved = _copy_src_only(case_name, tmp_path / "renamed_batch2_case")
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


def test_extras_status_ledger_matches_generated_output() -> None:
    """Committed `extras_status.toml` stays in sync with `case_extras`."""
    script_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit"
        / "render_extras_status.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_extras_status", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--check"]) == 0


def test_gyre_in_mesa_pre_ms_bypass_cadence_and_issue_170_guards() -> None:
    """`GyreInMesaStarExtras` skips pre-MS and raises `#170` when active."""
    h = star_extras.GyreInMesaStarExtras()
    h.extras_startup({"create_pre_main_sequence_model": True}, False)
    assert (
        h.extras_finish_step({"create_pre_main_sequence_model": True})
        == "keep_going"
    )
    h.extras_after_evolve({"create_pre_main_sequence_model": True})
    with pytest.raises(NotImplementedError, match="#170"):
        h.extras_startup({"create_pre_main_sequence_model": False}, False)
    with pytest.raises(NotImplementedError, match="#170"):
        h.extras_after_evolve({"create_pre_main_sequence_model": False})

    step_base = {
        "create_pre_main_sequence_model": False,
        "x_integer_ctrl": [4, 2, 1],
        "x_ctrl": [1.0e-4, 1.0e3],
        "x_logical_ctrl": [False, False],
        "need_to_save_profiles_now": False,
        "model_number": 3,
    }
    assert h.extras_finish_step(step_base) == "keep_going"
    assert (
        h.extras_finish_step({**step_base, "x_integer_ctrl": [0, 2, 1]})
        == "keep_going"
    )
    with pytest.raises(NotImplementedError, match="#170"):
        h.extras_finish_step({**step_base, "model_number": 4})
    with pytest.raises(NotImplementedError, match="#170"):
        h.extras_finish_step(
            {
                **step_base,
                "need_to_save_profiles_now": True,
                "x_logical_ctrl": [True, False],
            }
        )


def test_eval_t_tau_q_and_relation_cycles() -> None:
    """`eval_t_tau_q` and `cycle_atm_t_tau_*` match upstream branches."""
    assert star_atm_pulse.eval_t_tau_q("Eddington", 0.5) == constants.TWO_THIRDS
    assert math.isclose(
        star_atm_pulse.eval_t_tau_q("solar_Hopf", 0.0),
        (1.0361 - 0.3134) - 0.29589999,
        abs_tol=1.0e-15,
    )
    assert math.isclose(
        star_atm_pulse.eval_t_tau_q("Krishna_Swamy", 0.0),
        (1.39 - 0.815) - 0.025,
        abs_tol=1.0e-15,
    )
    assert star_atm_pulse.eval_t_tau_q("Trampedach_solar", 1.0) > 0.5
    with pytest.raises(ValueError, match="Invalid name in q"):
        star_atm_pulse.eval_t_tau_q("unknown", 0.1)

    rel = "Eddington"
    for _ in range(4):
        rel = star_atm_pulse.cycle_atm_t_tau_relation(rel)
    assert rel == "Eddington"
    with pytest.raises(ValueError, match="Invalid atm_T_tau_relation"):
        star_atm_pulse.cycle_atm_t_tau_relation("bad")


def test_t_tau_gradr_rms_profile_column_and_cycling_guard() -> None:
    """`TTauGradrStarExtras` evaluates `T_rms` and guards 10-step cycling."""
    teff = 5777.0
    tau1 = 0.2
    t_exact = teff * _rust.pow(0.75 * (tau1 + constants.TWO_THIRDS), 0.25)
    lnt_val = math.log(t_exact)
    state = {
        "nz": 3,
        "dq": [0.5, 0.5, 0.5],
        "lnt": [lnt_val, lnt_val, 10.0],
        "tau": [0.01, tau1, 5.0],
        "teff": teff,
        "atm_t_tau_relation": "Eddington",
        "x_ctrl": [1.0e-6],
        "model_number": 7,
    }
    h = star_extras.TTauGradrStarExtras()
    h.extras_startup(state, False)
    assert h.extras_start_step(state) == "keep_going"
    assert h.extras_check_model(state) == "keep_going"
    assert h.failed is False and (h.last_t_rms or 0.0) < 1.0e-11
    pcols = dict(h.extra_profile_columns(state))
    assert math.isclose(pcols["T_check"][1], t_exact, rel_tol=1.0e-15)
    assert h.extras_finish_step(state) == "keep_going"
    h.extras_after_evolve(state)

    bad_state = {**state, "lnt": [lnt_val + 0.1, lnt_val + 0.1, 10.0]}
    assert h.extras_check_model(bad_state) == "keep_going"
    assert h.failed is True
    assert h.extras_check_model({**state, "model_number": 10}) == (
        "keep_going",
        {"atm_t_tau_relation": "solar_Hopf"},
    )


def test_check_pulse_atm_cycling_guards() -> None:
    """`CheckPulseAtmStarExtras` guards 5-step and 20-step pulse/atm hooks."""
    h = star_extras.CheckPulseAtmStarExtras()
    h.extras_startup({}, False)
    assert h.extras_start_step({}) == "keep_going"
    assert h.extras_check_model({"model_number": 4}) == "keep_going"
    with pytest.raises(NotImplementedError, match="#162"):
        h.extras_check_model(
            {"model_number": 5, "atm_t_tau_relation": "Eddington"}
        )
    with pytest.raises(NotImplementedError, match="#162"):
        h.extras_check_model(
            {
                "model_number": 20,
                "atm_t_tau_relation": "solar_Hopf",
                "atm_t_tau_opacity": "varying",
            }
        )
    with pytest.raises(ValueError, match="Invalid atm_T_tau_opacity"):
        h.extras_check_model(
            {
                "model_number": 20,
                "atm_t_tau_relation": "Eddington",
                "atm_t_tau_opacity": "invalid",
            }
        )


def test_check_pulse_atm_fgong_v1300_parser() -> None:
    """`CheckPulseAtmStarExtras.compute_fgong_t_rms` parses v1300 payloads."""
    h = star_extras.CheckPulseAtmStarExtras()
    teff = 6000.0
    tau0 = 0.05
    t0 = teff * _rust.pow(0.75 * (tau0 + constants.TWO_THIRDS), 0.25)
    t0_fmt = float(f"{t0:16.9E}")
    globals_15 = [0.0] * 15
    globals_15[13] = teff
    pt0 = [0.0] * 40
    pt1 = [0.0] * 40
    pt0[0], pt0[2], pt0[4], pt0[7] = 100.0, t0_fmt, 1.0e-6, 1.0
    pt1[0], pt1[2], pt1[4], pt1[7] = 0.0, t0_fmt, 1.0e-6, 1.0
    nums = [*globals_15, *pt0, *pt1]
    rows = [
        "".join(f"{val:16.9E}" for val in nums[i : i + 5])
        for i in range(0, len(nums), 5)
    ]
    fgong = "\n".join(["h1", "h2", "h3", "h4", "   2  15  40 1300", *rows])
    rms = h.compute_fgong_t_rms(fgong, tau0, "Eddington", 1.0e-3)
    assert h.failed is False
    assert math.isclose(rms, abs(t0_fmt - t0), rel_tol=0.0, abs_tol=1.0e-12)
    h.compute_fgong_t_rms(fgong, tau0, "Eddington", 0.0)
    assert h.failed is True
    with pytest.raises(RuntimeError, match="not as expected"):
        h.compute_fgong_t_rms(
            "a\nb\nc\nd\n 2 15 39 1300\n", tau0, "Eddington", 1.0
        )


def test_ms_op_mono_path_validation_and_profile_columns(
    tmp_path: pathlib.Path,
) -> None:
    """`MsOpMonoStarExtras` validates `a06.140` and computes `dlnkap` ratios."""
    h = star_extras.MsOpMonoStarExtras()
    with pytest.raises(RuntimeError, match="empty"):
        h.extras_controls({"op_mono_data_path": "  "})
    with pytest.raises(FileNotFoundError, match=r"a06\.140"):
        h.extras_controls({"op_mono_data_path": str(tmp_path)})
    (tmp_path / "a06.140").write_text("stub\n", encoding="utf-8")
    h.extras_controls({"op_mono_data_path": str(tmp_path)})
    assert h.extras_start_step({}) == "keep_going"
    assert h.extras_check_model({}) == "keep_going"
    assert h.extras_finish_step({}) == "keep_going"
    h.extras_after_evolve({})
    cols = dict(
        h.extra_profile_columns(
            {
                "nz": 2,
                "opacity": [2.0, 4.0],
                "d_opacity_dlnd": [1.0, 2.0],
                "d_opacity_dlnt": [-6.0, -8.0],
            }
        )
    )
    assert cols["dlnkap_dlnd"] == [0.5, 0.5]
    assert cols["dlnkap_dlnT"] == [-3.0, -2.0]


def test_relax_composition_j_entropy_file_generation(
    tmp_path: pathlib.Path,
) -> None:
    """`RelaxCompositionJEntropyStarExtras` writes part-1 `.dat` files."""
    h = star_extras.RelaxCompositionJEntropyStarExtras()
    h.extras_controls({"x_integer_ctrl": [1]})
    p1 = {
        "x_integer_ctrl": [1],
        "case_dir": str(tmp_path),
        "teff": 8500.0,
        "nz": 2,
        "species": 2,
        "q": [1.0, 0.5],
        "dq": [0.5, 0.5],
        "entropy": [10.0, 8.0],
        "j_rot": [1.0e15, 5.0e14],
        "xa": [[0.7, 0.3], [0.6, 0.4]],
        "star_mass": 1.5,
        "l": [2.0 * constants.LSUN, constants.LSUN],
    }
    assert h.extras_finish_step(p1) == "keep_going"
    assert h.extras_finish_step({**p1, "teff": 7900.0}) == (
        "terminate",
        "produced files for relax",
    )
    with pytest.raises(NotImplementedError, match="#154"):
        h.extras_controls({"x_integer_ctrl": [2], "case_dir": str(tmp_path)})
    assert math.isclose(h.loaded_initial_mass or 0.0, 1.5, rel_tol=1.0e-15)


def test_relax_composition_j_entropy_part2_tolerance_checks() -> None:
    """`RelaxCompositionJEntropyStarExtras` checks 0.5% `(Teff, L)` bounds."""
    h = star_extras.RelaxCompositionJEntropyStarExtras()
    h.target_teff = 7900.0
    h.target_l = 2.0 * constants.LSUN
    ok_res = h.extras_check_model(
        {
            "x_integer_ctrl": [2],
            "teff": 7900.0 * 1.004,
            "l": [2.0 * constants.LSUN * 0.996],
        }
    )
    assert ok_res == ("terminate", "Values for Teff and L are within tolerance")
    assert h.within_tolerance is True
    bad_res = h.extras_check_model(
        {
            "x_integer_ctrl": [2],
            "teff": 7900.0 * 1.01,
            "l": [2.0 * constants.LSUN],
        }
    )
    assert bad_res == ("terminate", "Relaxed model does not match input")
    assert h.within_tolerance is False


def test_accreted_material_j_real4_threshold_and_wind_switch() -> None:
    """`AccretedMaterialJStarExtras` checks `J` error and `lxtra` wind flip."""
    h = star_extras.AccretedMaterialJStarExtras()
    h.extras_controls({})
    adj_snap = {
        "x_ctrl": [0.1, -3.0],
        "photosphere_r": 2.0,
        "cgrav": [constants.STANDARD_CGRAV],
        "mstar": constants.MSUN,
    }
    j_acc, mdot_new = h.compute_accretor_adjust_mdot(adj_snap, 0.0)
    assert j_acc > 0.0 and math.isclose(
        mdot_new, 1.0e-3 * constants.MSUN / constants.SECYER, rel_tol=1.0e-15
    )
    assert h.other_adjust_mdot(adj_snap, 0.0) == {
        "mstar_dot": mdot_new,
        "accreted_material_j": j_acc,
    }

    # Test real(4) `1e-5` (`0x1.4f8b58p-17` < `1.0e-5`) boundary (:125).
    real4_1e5 = struct.unpack("f", struct.pack("f", 1.0e-5))[0]
    j_tot = 1.0e50
    step_err = {
        "model_number": 2,
        "generations": 2,
        "mstar_dot": 1.0,
        "mstar_dot_old": 1.0,
        "accreted_material_j": 0.5 * (real4_1e5 + 1.0e-5) * j_tot,
        "dt": 1.0,
        "total_angular_momentum": j_tot,
        "total_angular_momentum_old": j_tot,
        "center_h1": 0.5,
    }
    assert h.extras_check_model(step_err)[0] == "terminate"
    h.extras_controls({})
    wind_flip = {**step_err, "mstar_dot": -1.0, "center_h1": 1.0e-7}
    assert h.extras_check_model(wind_flip) == (
        "terminate",
        "Test has finished with no problems.",
    )
    assert h.wind_switch_ok is True


def test_make_sdb_retry_and_wind_shutoff_guards() -> None:
    """`MakeSdbStarExtras` retries below `h_lower_limit` and zeroes winds."""
    sdb = star_extras.MakeSdbStarExtras()
    assert (
        sdb.extras_startup(
            {"x_ctrl": [0.01, 0.1], "star_mass": 0.50, "he_core_mass": 0.48},
            False,
        )
        is None
    )
    assert sdb.extras_startup(
        {"x_ctrl": [0.05, 0.1], "star_mass": 0.50, "he_core_mass": 0.48},
        False,
    ) == {"reimers_scaling_factor": 0.0, "blocker_scaling_factor": 0.0}
    assert (
        sdb.extras_check_model(
            {"x_ctrl": [0.01, 0.1], "star_mass": 0.50, "he_core_mass": 0.48}
        )
        == "keep_going"
    )
    assert (
        sdb.extras_check_model(
            {
                "x_ctrl": [0.05, 0.1],
                "star_mass": 0.50,
                "he_core_mass": 0.48,
                "reimers_scaling_factor": 0.5,
            }
        )
        == "retry"
    )
    assert sdb.extras_check_model(
        {
            "x_ctrl": [0.02, 0.5],
            "star_mass": 0.495,
            "he_core_mass": 0.48,
            "reimers_scaling_factor": 0.5,
        }
    ) == (
        "keep_going",
        {"reimers_scaling_factor": 0.0, "blocker_scaling_factor": 0.0},
    )


def test_conserve_angular_momentum_drift_guard() -> None:
    """`ConserveAngularMomentumStarExtras` enforces `1e-9` `J` tolerance."""
    cj = star_extras.ConserveAngularMomentumStarExtras()
    cj.extras_controls({})
    assert (
        cj.extras_check_model({"create_pre_main_sequence_model": True})
        == "keep_going"
    )
    s_j0 = {
        "create_pre_main_sequence_model": False,
        "total_angular_momentum": 1.0e50,
        "model_number": 10,
    }
    assert cj.extras_check_model(s_j0) == "keep_going"
    assert (
        cj.extras_check_model(
            {**s_j0, "total_angular_momentum": 1.0e50 * (1.0 + 5.0e-10)}
        )
        == "keep_going"
    )
    assert (
        cj.extras_check_model(
            {**s_j0, "total_angular_momentum": 1.0e50 * (1.0 + 2.0e-9)}
        )[0]
        == "terminate"
    )
    cj.extras_after_evolve(s_j0)
    assert cj.rel_err_total_j == 0.0


def test_hse_riemann_shock_stop_and_mach_check() -> None:
    """`HseRiemannStarExtras` stops on shock target radius and checks `u`."""
    hse = star_extras.HseRiemannStarExtras()
    assert not hse.compute_pmax_history_columns({})
    hse_snap = {
        "nz": 2,
        "x_ctrl": [0.0] * 8 + [1.0],
        "peos": [1.0e10, 5.0e12],
        "r": [2.0 * constants.RSUN, constants.RSUN],
        "v": [2.0e7, 1.0e6],
        "u": [1.0e4, 1.0e4],
        "csound": [1.0e7, 1.0e7],
        "rho": [1.0e-3, 1.0],
        "t": [1.0e6, 1.0e7],
        "m": [constants.MSUN, 0.5 * constants.MSUN],
        "time": 1.0e5,
        "u_flag": True,
    }
    assert dict(hse.compute_pmax_history_columns(hse_snap, n=12))["Pmax_P"] == (
        1.0e10
    )
    assert hse.extras_finish_step(hse_snap) == (
        "terminate",
        "shock has reached target location",
    )
    hse.extras_after_evolve(hse_snap)
    assert hse.within_tolerance is True


def test_c13_pocket_part1_tracking() -> None:
    """`C13PocketStarExtras` tracks MS/CHeB and 1DUP in part 1."""
    h = star_extras.C13PocketStarExtras()
    h.extras_startup({"x_integer_ctrl": [3]}, True)
    with pytest.raises(NotImplementedError, match="#231"):
        h.extras_startup({"x_integer_ctrl": [2]}, True)

    h.extras_startup({"x_integer_ctrl": [1]}, False)
    s_p1 = {
        "x_integer_ctrl": [1],
        "star_age": 1.0e7,
        "surface_l_lsun": 10.0,
        "power_h_burn": 10.0,
        "power_he_burn": 0.0,
        "center_h1": 0.7,
        "center_he4": 0.28,
        "he_core_mass": 0.0,
        "conv_mx1_top": 1.0,
        "conv_mx1_bot": 0.3,
        "star_mass": 2.0,
    }
    h.extras_finish_step(s_p1)
    h.extras_finish_step({**s_p1, "star_age": 1.01e9, "center_h1": 1.0e-5})
    h.extras_finish_step({**s_p1, "star_age": 1.05e9, "power_he_burn": 1.0})
    h.extras_finish_step(
        {
            **s_p1,
            "star_age": 1.2e9,
            "center_he4": 1.0e-5,
            "he_core_mass": 0.52,
        }
    )
    h.extras_after_evolve(s_p1)
    assert math.isclose(h.testhub_extras["MS_lifetime"], 1.0)
    assert math.isclose(h.testhub_extras["m_1DUP"], 0.6)


def test_c13_pocket_part2_3dup_termination() -> None:
    """`C13PocketStarExtras` stops after the 3rd thermal pulse with 3DUP."""
    h = star_extras.C13PocketStarExtras()
    h.extras_startup({"x_integer_ctrl": [2]}, False)
    end_res: str | tuple[str, str] = "keep_going"
    for pulse_idx in range(3):
        age_base = (pulse_idx + 1) * 1.0e4
        m_peak = 0.58 + 0.002 * pulse_idx
        h.extras_finish_step(
            {
                "x_integer_ctrl": [2],
                "power_he_burn": 2.0e4,
                "power_h_burn": 1.0e3,
                "he_core_mass": m_peak,
                "star_age": age_base,
            }
        )
        end_res = h.extras_finish_step(
            {
                "x_integer_ctrl": [2],
                "power_he_burn": 1.0e2,
                "power_h_burn": 5.0e3,
                "he_core_mass": m_peak - 5.0e-4,
                "star_age": age_base + 500.0,
            }
        )
    assert end_res == ("terminate", "third pulse with 3DUP has occurred")
    h.extras_after_evolve({"x_integer_ctrl": [2]})
    assert len(h.testhub_extras) == 6


def test_c13_pocket_part4_characterization() -> None:
    """`C13PocketStarExtras` stops after 1 TP and characterizes `c13` pocket."""
    h = star_extras.C13PocketStarExtras()
    h.extras_startup({"x_integer_ctrl": [4], "surface_c12": 0.003}, False)
    p4 = {
        "x_integer_ctrl": [4],
        "nz": 2,
        "xa_c13": [0.02, 0.001],
        "m": [0.58 * constants.MSUN, 0.50 * constants.MSUN],
        "q": [0.29, 0.25],
        "dq": [1.0e-5, 0.01],
        "mstar": 2.0 * constants.MSUN,
        "star_mass": 2.0,
        "he_core_mass": 0.58,
        "surface_c12": 0.0033,
        "power_he_burn": 2.0e4,
        "power_h_burn": 1.0e2,
        "x_ctrl": [0.02, 2.0 * (0.29 + 0.5e-5), 2.0e-5, 3.0e-4],
    }
    assert h.extras_finish_step(p4) == "keep_going"
    assert h.extras_finish_step(
        {**p4, "power_he_burn": 100.0, "power_h_burn": 2000.0}
    ) == ("terminate", "one thermal pulse cycle complete")
    h.extras_after_evolve(p4)
    assert h.within_tolerance is True


def test_hb_2m_and_low_z_hooks() -> None:
    """`Hb2MStarExtras` and `LowZStarExtras` match upstream hooks."""
    hb = star_extras.Hb2MStarExtras()
    hb.extras_startup({"x_integer_ctrl": [1]}, True)
    with pytest.raises(NotImplementedError, match="#231"):
        hb.extras_startup({"x_integer_ctrl": [3]}, True)
    hb.extras_startup({"x_integer_ctrl": [3]}, False)
    hb.extras_finish_step(
        {"x_integer_ctrl": [3], "center_he4": 0.48, "mass_conv_core": 0.19}
    )
    hb.extras_finish_step(
        {"x_integer_ctrl": [3], "center_he4": 0.20, "mass_conv_core": 0.24}
    )
    assert hb.extra_history_columns({}) == [("mass_conv_core_y050", 0.19)]
    hb.extras_after_evolve({"x_integer_ctrl": [3], "x_ctrl": [0.14, 0.25]})
    assert hb.within_tolerance is True

    lz = star_extras.LowZStarExtras()
    lz.extras_controls({})
    assert lz.extra_history_header_items({"mixing_length_alpha": 1.8}) == [
        ("mixing_length_alpha", 1.8)
    ]


def test_r_crb_star_writes_inlist_nzhi(tmp_path: pathlib.Path) -> None:
    """`RCrBStarExtras` writes 1-based `set_nzhi` or `nz + 1` fallthrough."""
    rcrb = star_extras.RCrBStarExtras()
    rcrb.extras_after_evolve({"x_logical_ctrl": [False]})
    assert not (tmp_path / "inlist_nzhi").exists()
    rcrb.extras_after_evolve(
        {
            "x_logical_ctrl": [True],
            "nz": 3,
            "t": [1.0e7, 2.0e8, 3.0e8],
            "case_dir": str(tmp_path),
        }
    )
    assert rcrb.saved_nzhi == 2
    rcrb.extras_after_evolve(
        {
            "x_logical_ctrl": [True],
            "nz": 3,
            "t": [1.0e7, 2.0e7, 3.0e7],
            "case_dir": str(tmp_path),
        }
    )
    assert rcrb.saved_nzhi == 4


def test_post_evolve_wind_cpm_and_wd_diffusion() -> None:
    """`HotCoolWindStarExtras`, `ConvCoreCpmStarExtras`, `WdDiffusion`."""
    hcw = star_extras.HotCoolWindStarExtras()
    hcw.extras_after_evolve(
        {
            "mstar": 6.75 * constants.MSUN,
            "log_surface_luminosity": 4.15,
            "log_l_upper_limit": 4.10,
        }
    )
    assert hcw.within_tolerance is True

    cpm = star_extras.ConvCoreCpmStarExtras()
    cpm.extras_startup({}, True)
    cpm.extras_after_evolve({"x_ctrl": [0.125, 0.135], "mass_conv_core": 0.13})
    assert cpm.within_tolerance is True

    wdd = star_extras.WdDiffusionStarExtras()
    with pytest.raises(RuntimeError, match="failed to reach target"):
        wdd.extras_after_evolve({"teff": 9500.0, "teff_lower_limit": 9000.0})
    g_val = 1.0e8
    e_val = 2.0 * constants.AMU * g_val / constants.QE
    wdd.extras_after_evolve(
        {
            "teff": 8900.0,
            "teff_lower_limit": 9000.0,
            "nz": 2,
            "m": [0.6 * constants.MSUN, 0.4 * constants.MSUN],
            "e_field": [0.0, e_val],
            "g_field_element_diffusion": [1.0, g_val],
        }
    )
    assert wdd.within_tolerance is True


def test_high_rot_darkening_and_semiconvection_checks() -> None:
    """`HighRotDarkeningStarExtras` and `SemiconvectionStarExtras`."""
    hrd = star_extras.HighRotDarkeningStarExtras()
    assert (
        hrd.extras_finish_step({"create_pre_main_sequence_model": True})
        == "keep_going"
    )
    assert (
        hrd.extras_finish_step(
            {
                "create_pre_main_sequence_model": False,
                "w_div_w_crit_avg_surf": 0.85,
                "w_div_w_crit_roche": [0.82],
            }
        )
        == "keep_going"
    )
    assert hrd.last_w_div_w_crit_roche_surf == 0.82

    sc = star_extras.SemiconvectionStarExtras()
    msun = constants.MSUN
    sc.extras_after_evolve(
        {
            "nz": 4,
            "m": [0.15 * msun, 0.14 * msun, 0.13 * msun, 0.10 * msun],
            "mixing_type": [0, 3, 1, 1],
            "dq": [0.25, 0.25, 0.25, 0.25],
            "lnt": [7.20 * constants.LN10] * 4,
            "lnd": [1.78 * constants.LN10] * 4,
        }
    )
    assert sc.checks_okay is True


def test_batch2_live_py_star_extras_adapter_execution_and_loud_key_error(
    tmp_path: pathlib.Path,
) -> None:
    """`R_CrB_star`, `hb_2M`, and `low_z` attach and step live."""
    rcrb_dir = harness_runner.resolve_case_dir("star", "R_CrB_star")
    rep = harness_runner.run_trace_inprocess(
        "star",
        rcrb_dir,
        tmp_path / "out_r_crb_star",
        max_steps=4,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert rep["total_steps"] == 4

    hb_dir = harness_runner.resolve_case_dir("star", "hb_2M")
    assert isinstance(
        case_extras.resolve_case_extras("star", hb_dir),
        star_extras.Hb2MStarExtras,
    )

    low_z_dir = harness_runner.resolve_case_dir("star", "low_z")
    assert isinstance(
        case_extras.resolve_case_extras("star", low_z_dir),
        star_extras.LowZStarExtras,
    )


_AMJ_STEP_OBS: list[dict[str, float | int | bool]] = []


class _RecordingAccretedMaterialJExtras(
    star_extras.AccretedMaterialJStarExtras
):
    """Records snapshot scalars seen during `other_adjust_mdot` / checks."""

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, str]:
        _AMJ_STEP_OBS.append(
            {
                "model_number": int(state["model_number"]),
                "generations": int(state["generations"]),
                "use_accreted_material_j": bool(
                    state["use_accreted_material_j"]
                ),
                "accreted_material_j": float(state["accreted_material_j"]),
                "mstar_dot": float(state["mstar_dot"]),
                "mstar_dot_old": float(state["mstar_dot_old"]),
                "total_angular_momentum": float(
                    state["total_angular_momentum"]
                ),
                "total_angular_momentum_old": float(
                    state["total_angular_momentum_old"]
                ),
            }
        )
        return super().extras_check_model(state)


class _SnapMutateNoneAdjustMdotExtras:
    """Mutates snapshot in place and returns `None`."""

    def other_adjust_mdot(
        self, state: dict[str, Any], current_mdot: float
    ) -> None:
        mdot_new = float(current_mdot) + 2.0e17
        state["accreted_material_j"] = 1.25e18
        state["mstar_dot"] = mdot_new
        _AMJ_STEP_OBS.append({"expected_mdot": mdot_new})

    def extras_check_model(self, state: dict[str, Any]) -> str:
        _AMJ_STEP_OBS.append(
            {
                "accreted_material_j": float(state["accreted_material_j"]),
                "mstar_dot": float(state["mstar_dot"]),
            }
        )
        return "keep_going"


class _AccJOnlyDictExtras:
    """Returns `{"accreted_material_j": ...}` while keeping `current_mdot`."""

    def other_adjust_mdot(
        self, state: dict[str, Any], current_mdot: float
    ) -> dict[str, float]:
        _AMJ_STEP_OBS.append({"current_mdot": float(current_mdot)})
        return {"accreted_material_j": 2.5e18}

    def extras_check_model(self, state: dict[str, Any]) -> str:
        _AMJ_STEP_OBS.append(
            {
                "accreted_material_j": float(state["accreted_material_j"]),
                "mstar_dot": float(state["mstar_dot"]),
            }
        )
        return "keep_going"


class _ScalarFloatAdjustMdotExtras:
    """Legacy scalar `float` return compatibility."""

    def other_adjust_mdot(
        self, state: dict[str, Any], current_mdot: float
    ) -> float:
        mdot_new = float(current_mdot) + 3.0e17
        _AMJ_STEP_OBS.append({"expected_mdot": mdot_new})
        return mdot_new

    def extras_check_model(self, state: dict[str, Any]) -> str:
        _AMJ_STEP_OBS.append({"mstar_dot": float(state["mstar_dot"])})
        return "keep_going"


class _UnknownKeyAdjustMdotExtras:
    """Returns an unsupported key from `other_adjust_mdot`."""

    def other_adjust_mdot(
        self, state: dict[str, Any], current_mdot: float
    ) -> dict[str, float]:
        return {"bogus_key": float(current_mdot)}


def _recording_amj_factory(suite: str, case_dir: pathlib.Path) -> object:
    _AMJ_STEP_OBS.clear()
    return _RecordingAccretedMaterialJExtras()


def _snap_mutate_none_factory(suite: str, case_dir: pathlib.Path) -> object:
    _AMJ_STEP_OBS.clear()
    return _SnapMutateNoneAdjustMdotExtras()


def _acc_j_only_dict_factory(suite: str, case_dir: pathlib.Path) -> object:
    _AMJ_STEP_OBS.clear()
    return _AccJOnlyDictExtras()


def _scalar_float_adjust_factory(suite: str, case_dir: pathlib.Path) -> object:
    _AMJ_STEP_OBS.clear()
    return _ScalarFloatAdjustMdotExtras()


def _unknown_key_adjust_factory(suite: str, case_dir: pathlib.Path) -> object:
    return _UnknownKeyAdjustMdotExtras()


def test_accreted_material_j_live_and_adjust_mdot_adapter_modes(
    tmp_path: pathlib.Path,
) -> None:
    """`accreted_material_j` runs via `resolve_case_extras` and adapter."""
    amj_dir = harness_runner.resolve_case_dir("star", "accreted_material_j")
    assert isinstance(
        case_extras.resolve_case_extras("star", amj_dir),
        star_extras.AccretedMaterialJStarExtras,
    )

    rep_live = harness_runner.run_trace_inprocess(
        "star",
        amj_dir,
        tmp_path / "out_amj_live",
        max_steps=2,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep_live["total_steps"]) == 2

    harness_runner.run_trace_inprocess(
        "star",
        amj_dir,
        tmp_path / "out_amj_obs",
        max_steps=2,
        allow_unsupported_controls=True,
        extras_factory="tests.test_case_extras_batch2:_recording_amj_factory",
    )
    assert len(_AMJ_STEP_OBS) == 2
    step1, step2 = _AMJ_STEP_OBS
    assert step1["use_accreted_material_j"] is True
    assert float(step1["accreted_material_j"]) > 0.0
    assert float(step1["mstar_dot"]) > 0.0
    assert int(step2["generations"]) == 2
    assert float(step2["mstar_dot_old"]) > 0.0
    assert (
        float(step2["total_angular_momentum"])
        > float(step2["total_angular_momentum_old"])
        > 0.0
    )

    harness_runner.run_trace_inprocess(
        "star",
        amj_dir,
        tmp_path / "out_snap_mut",
        max_steps=1,
        allow_unsupported_controls=True,
        extras_factory=(
            "tests.test_case_extras_batch2:_snap_mutate_none_factory"
        ),
    )
    assert len(_AMJ_STEP_OBS) == 2
    assert _AMJ_STEP_OBS[1]["accreted_material_j"] == 1.25e18
    assert _AMJ_STEP_OBS[1]["mstar_dot"] == _AMJ_STEP_OBS[0]["expected_mdot"]

    harness_runner.run_trace_inprocess(
        "star",
        amj_dir,
        tmp_path / "out_dict_accj_only",
        max_steps=1,
        allow_unsupported_controls=True,
        extras_factory="tests.test_case_extras_batch2:_acc_j_only_dict_factory",
    )
    assert len(_AMJ_STEP_OBS) == 2
    assert _AMJ_STEP_OBS[1]["accreted_material_j"] == 2.5e18
    assert _AMJ_STEP_OBS[1]["mstar_dot"] == _AMJ_STEP_OBS[0]["current_mdot"]

    harness_runner.run_trace_inprocess(
        "star",
        amj_dir,
        tmp_path / "out_scalar_float",
        max_steps=1,
        allow_unsupported_controls=True,
        extras_factory=(
            "tests.test_case_extras_batch2:_scalar_float_adjust_factory"
        ),
    )
    assert len(_AMJ_STEP_OBS) == 2
    assert _AMJ_STEP_OBS[1]["mstar_dot"] == _AMJ_STEP_OBS[0]["expected_mdot"]

    with pytest.raises(
        Exception, match="unsupported star adjust_mdot dict key"
    ):
        harness_runner.run_trace_inprocess(
            "star",
            amj_dir,
            tmp_path / "out_unknown_key",
            max_steps=1,
            allow_unsupported_controls=True,
            extras_factory=(
                "tests.test_case_extras_batch2:_unknown_key_adjust_factory"
            ),
        )


def test_batch2_handlers_raise_key_error_on_missing_snapshot_fields() -> None:
    """Batch 2 hooks raise `KeyError` without default fallbacks when missing."""
    calls = (
        lambda: star_extras.GyreInMesaStarExtras().extras_startup({}, False),
        lambda: star_extras.TTauGradrStarExtras().extras_check_model({}),
        lambda: star_extras.CheckPulseAtmStarExtras().extras_check_model({}),
        lambda: star_extras.MsOpMonoStarExtras().extra_profile_columns({}),
        lambda: (
            star_extras.RelaxCompositionJEntropyStarExtras().extras_finish_step(
                {}
            )
        ),
        lambda: star_extras.AccretedMaterialJStarExtras().extras_check_model(
            {}
        ),
        lambda: star_extras.MakeSdbStarExtras().extras_check_model({}),
        lambda: (
            star_extras.ConserveAngularMomentumStarExtras().extras_check_model(
                {}
            )
        ),
        lambda: star_extras.HseRiemannStarExtras().extras_finish_step({}),
        lambda: star_extras.C13PocketStarExtras().extras_finish_step({}),
        lambda: star_extras.Hb2MStarExtras().extras_finish_step({}),
        lambda: star_extras.LowZStarExtras().extra_history_header_items({}),
        lambda: star_extras.RCrBStarExtras().extras_after_evolve({}),
        lambda: star_extras.HotCoolWindStarExtras().extras_after_evolve({}),
        lambda: star_extras.ConvCoreCpmStarExtras().extras_after_evolve({}),
        lambda: star_extras.WdDiffusionStarExtras().extras_after_evolve({}),
        lambda: star_extras.HighRotDarkeningStarExtras().extras_finish_step({}),
        lambda: star_extras.SemiconvectionStarExtras().extras_after_evolve({}),
    )
    for call in calls:
        with pytest.raises(KeyError):
            call()
