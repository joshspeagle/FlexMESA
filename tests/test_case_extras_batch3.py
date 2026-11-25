# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit, relocation, adapter-detachment, and error-propagation tests for `#231`
Batch 3 `StarExtras` and `BinaryExtras` ports (`19` `star` + `2` `binary`
cases).
"""

from __future__ import annotations

import math
import pathlib
import shutil
from collections.abc import Sequence
from typing import Any

import pytest

from flexmesa import _rust, constants
from flexmesa import math as fm_math
from flexmesa.testing import case_manifest, harness_runner
from tests import case_extras
from tests.case_extras import (
    binary_extras,
    star_ccsn_iip,
    star_core_collapse,
    star_ns_wd_simplex,
    star_ppisn,
    star_rot_mix_checks,
)

BATCH3_STAR_CASES: tuple[tuple[str, type[Any]], ...] = (
    (
        "12M_pre_ms_to_core_collapse",
        star_core_collapse.PreMs12MCoreCollapseStarExtras,
    ),
    (
        "20M_pre_ms_to_core_collapse",
        star_core_collapse.PreMs20MCoreCollapseStarExtras,
    ),
    ("zams_to_cc_80", star_core_collapse.PreMs20MCoreCollapseStarExtras),
    ("split_burn_big_net", star_core_collapse.SplitBurnBigNetStarExtras),
    ("make_pre_ccsn_13bvn", star_core_collapse.MakePreCcsn13bvnStarExtras),
    ("15M_dynamo", star_rot_mix_checks.Dynamo15MStarExtras),
    ("1M_thermohaline", star_rot_mix_checks.Thermohaline1MStarExtras),
    (
        "20M_z2m2_high_rotation",
        star_rot_mix_checks.HighRotation20MZ2m2StarExtras,
    ),
    (
        "diffusion_smoothness",
        star_rot_mix_checks.DiffusionSmoothnessStarExtras,
    ),
    (
        "radiative_levitation",
        star_rot_mix_checks.RadiativeLevitationStarExtras,
    ),
    ("ns_c", star_ns_wd_simplex.NeutronStarEnvStarExtras),
    ("ns_h", star_ns_wd_simplex.NeutronStarEnvStarExtras),
    ("ns_he", star_ns_wd_simplex.NeutronStarEnvStarExtras),
    ("wd_cool_0.6M", star_ns_wd_simplex.WdCool06MStarExtras),
    ("wd_nova_burst", star_ns_wd_simplex.WdNovaBurstStarExtras),
    ("1M_pre_ms_to_wd", star_ns_wd_simplex.PreMs1MToWdStarExtras),
    (
        "simplex_solar_calibration",
        star_ns_wd_simplex.SimplexSolarCalibrationStarExtras,
    ),
    ("ppisn", star_ppisn.PpisnStarExtras),
    ("ccsn_IIp", star_ccsn_iip.CcsnIIpStarExtras),
)

BATCH3_BINARY_CASES: tuple[tuple[str, type[Any]], ...] = (
    ("jdot_ls_check", binary_extras.JdotLsCheckBinaryExtras),
    ("wind_fed_bhhmxb", binary_extras.WindFedBhHmxbBinaryExtras),
)


def _copy_src_only(
    suite: str, case_name: str, dst_dir: pathlib.Path
) -> pathlib.Path:
    """Copy `<case>/src` into `dst_dir/src` without inlists or folder names."""
    orig = harness_runner.resolve_case_dir(suite, case_name)
    shutil.copytree(orig / "src", dst_dir / "src")
    return dst_dir


@pytest.mark.parametrize(("case_name", "expected_cls"), BATCH3_STAR_CASES)
def test_batch3_star_cases_resolve_by_fortran_content_only(
    tmp_path: pathlib.Path, case_name: str, expected_cls: type[Any]
) -> None:
    """Handlers match from Fortran text; unserved ones stay detached."""
    served = expected_cls.__name__ in case_extras.ADAPTER_SERVED_STAR_HANDLERS
    orig = harness_runner.resolve_case_dir("star", case_name)
    assert case_extras.classify_case_extras("star", orig) == (
        "ported" if served else "unported"
    )
    assert isinstance(case_extras.match_case_extras("star", orig), expected_cls)
    moved = _copy_src_only("star", case_name, tmp_path / "renamed_star_case")
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


@pytest.mark.parametrize(("case_name", "expected_cls"), BATCH3_BINARY_CASES)
def test_batch3_binary_cases_resolve_by_fortran_content_only(
    tmp_path: pathlib.Path, case_name: str, expected_cls: type[Any]
) -> None:
    """Binary handlers match from Fortran text; unserved ones stay detached."""
    served = expected_cls.__name__ in case_extras.ADAPTER_SERVED_BINARY_HANDLERS
    orig = harness_runner.resolve_case_dir("binary", case_name)
    assert case_extras.classify_case_extras("binary", orig) == (
        "ported" if served else "unported"
    )
    assert isinstance(
        case_extras.match_case_extras("binary", orig), expected_cls
    )
    moved = _copy_src_only("binary", case_name, tmp_path / "renamed_bin_case")
    assert isinstance(
        case_extras.match_case_extras("binary", moved), expected_cls
    )
    for case_dir in (orig, moved):
        attached = case_extras.resolve_case_extras("binary", case_dir)
        if served:
            assert isinstance(attached, expected_cls)
        else:
            assert attached is None
    (moved / "src" / "run_binary_extras.f90").unlink()
    assert case_extras.match_case_extras("binary", moved) is None


def test_every_manifest_star_and_binary_case_is_stock_or_matched() -> None:
    """All 98 `star` and `binary` manifest cases are stock or matched."""
    manifest = case_manifest.load_case_manifest()
    checked = 0
    for entry in manifest.cases:
        if entry.suite not in ("star", "binary"):
            continue
        checked += 1
        case_dir = harness_runner.resolve_case_dir(entry.suite, entry.case)
        is_stock = (
            case_extras.extras_source_digest(case_dir)
            in case_extras.STOCK_EXTRAS_DIGESTS
        )
        matched = case_extras.match_case_extras(entry.suite, case_dir)
        assert is_stock != (matched is not None), (
            f"{entry.suite}/{entry.case}: is_stock={is_stock}, "
            f"matched={matched!r}"
        )
    assert checked == 98


def test_alpha_mlt_and_core_collapse_split_burn_handlers() -> None:
    """Verify `alpha_mlt_routine`, core-collapse, and split-burn hooks."""
    x_ctrl = [0.0] * 30
    x_ctrl[20] = 3.0
    x_ctrl[21] = 1.5
    x_ctrl[22] = 0.5
    vec = star_core_collapse.compute_alpha_mlt_vector(
        {
            "x_ctrl": x_ctrl,
            "net_iso_h1": 1,
            "nz": 3,
            "xa_h1": [0.7, 0.5, 0.1],
        }
    )
    assert vec == [3.0, 3.0, 1.5]
    assert (
        star_core_collapse.compute_alpha_mlt_vector(
            {"x_ctrl": x_ctrl, "net_iso_h1": 0, "nz": 2, "xa_h1": [0.7, 0.1]}
        )
        is None
    )

    cc12 = star_core_collapse.PreMs12MCoreCollapseStarExtras()
    with pytest.raises(NotImplementedError, match="#155"):
        cc12.other_alpha_mlt({"x_ctrl": x_ctrl, "net_iso_h1": 2})
    assert cc12.other_alpha_mlt({"x_ctrl": [0.0] * 30, "net_iso_h1": 2}) is None
    assert cc12.other_alpha_mlt({"x_ctrl": x_ctrl, "net_iso_h1": 0}) is None
    x_log = [False] * 40
    cc12.extras_startup({"x_logical_ctrl": x_log}, False)
    assert cc12.extras_finish_step({"x_logical_ctrl": x_log}) == "keep_going"
    cc12.extras_after_evolve(
        {
            "nz": 4,
            "x_integer_ctrl": [0, 0, 0, 0, 6],
            "fe_core_mass": 1.45,
            "fe_core_infall": 1.2e8,
            "fe_core_infall_limit": 1.0e8,
            "x_logical_ctrl": x_log,
        }
    )
    assert cc12.testhub_extras == {"fe_core_mass": 1.45}
    assert cc12.within_tolerance is True

    cc20 = star_core_collapse.PreMs20MCoreCollapseStarExtras()
    cc20.extras_after_evolve(
        {
            "nz": 4,
            "x_integer_ctrl": [0, 0, 0, 0, 7],
            "fe_core_mass": 0.85,
            "fe_core_infall": 1.2e8,
            "fe_core_infall_limit": 1.0e8,
            "x_logical_ctrl": x_log,
        }
    )
    assert cc20.within_tolerance is False

    sb = star_core_collapse.SplitBurnBigNetStarExtras()
    sb.extras_after_evolve(
        {"x_integer_ctrl": [1], "nz": 3, "burn_num_iters": [4, 7, 9]}
    )
    assert sb.testhub_extras == {"total_burn_solver_steps": 20.0}


def test_make_pre_ccsn_13bvn_drag_and_mass_change_hooks() -> None:
    """Verify `MakePreCcsn13bvnStarExtras` drag columns and `mass_change`."""
    bvn = star_core_collapse.MakePreCcsn13bvnStarExtras()
    bvn.extras_startup({}, False)
    bvn.extras_startup({}, True)
    hcols = bvn.extra_history_columns(
        {
            "v_flag": True,
            "drag_coefficient": 1.0,
            "nz": 3,
            "q": [0.9, 0.5, 0.1],
            "min_q_for_drag": 0.4,
            "fdotv_drag_energy": [2.0e10, 3.0e10, 9.0e10],
            "dm": [1.0e30, 2.0e30, 5.0e30],
        }
    )
    assert hcols == [("log_total_drag_energy", _rust.log10(8.0e40))]
    assert bvn.extra_history_columns(
        {"v_flag": False, "drag_coefficient": 1.0}
    ) == [("log_total_drag_energy", 0.0)]
    pcols = bvn.extra_profile_columns(
        {
            "nz": 2,
            "zbar": [2.0, 6.0],
            "abar": [4.0, 12.0],
            "v_flag": False,
            "drag_coefficient": 0.0,
        }
    )
    assert pcols == [
        ("zbar_div_abar", [0.5, 0.5]),
        ("dvdt_drag", [0.0, 0.0]),
        ("FdotV_drag_energy", [0.0, 0.0]),
    ]
    assert (
        bvn.extras_finish_step(
            {
                "star_mass": 3.0001,
                "he_core_mass": 3.0,
                "x_logical_ctrl": [True],
                "star_species_mass_min_limit": 1.0e-3,
            }
        )
        == "keep_going"
    )
    with pytest.raises(NotImplementedError, match="#159"):
        bvn.extras_finish_step(
            {
                "star_mass": 3.5,
                "he_core_mass": 3.0,
                "x_logical_ctrl": [True],
                "star_species_mass_min_limit": 1.0e-5,
            }
        )


def test_dynamo_and_thermohaline_checks() -> None:
    """Verify `Dynamo15MStarExtras` and `Thermohaline1MStarExtras`."""
    msun = constants.MSUN
    ln10 = constants.LN10
    dyn = star_rot_mix_checks.Dynamo15MStarExtras()
    dyn.extras_after_evolve({"x_logical_ctrl": [False]})
    assert dyn.checks_okay is None

    dyn.extras_after_evolve(
        {
            "x_logical_ctrl": [True],
            "nz": 4,
            "star_mass": 13.5,
            "total_angular_momentum": 1.0e51,
            "center_omega": 1.0e-4,
            "he_core_omega": 1.0e-4,
            "he_core_mass": 3.5,
            "j_rot": [1.0e18, 1.0e15, 1.0e15, 1.0e14],
            "omega": [1.0e-6, 1.0e-5, 1.0e-5, 1.0e-4],
            "r": [2.0e10, 1.0e10, 5.0e9, 1.0e9],
            "m": [13.5 * msun, 3.5 * msun, 3.2 * msun, 2.5 * msun],
            "dq": [0.25, 0.25, 0.25, 0.25],
            "lnt": [7.5 * ln10] * 4,
            "lnd": [2.0 * ln10] * 4,
            "d_es": [1.0e3] * 4,
            "d_st": [1.0e5] * 4,
            "nu_st": [1.0e10] * 4,
        }
    )
    assert dyn.checks_okay is True
    assert not dyn.failed_checks

    th = star_rot_mix_checks.Thermohaline1MStarExtras()
    th.extras_after_evolve(
        {
            "nz": 4,
            "total_angular_momentum": 1.0e50,
            "center_omega": 1.0e-3,
            "he_core_omega": 1.0e-4,
            "j_rot": [1.0e15] * 4,
            "omega": [1.0e-6] * 4,
            "r": [1.0e10] * 4,
            "m": [0.9 * msun, 0.26 * msun, 0.245 * msun, 0.20 * msun],
            "dq": [0.25] * 4,
            "lnt": [7.2 * ln10] * 4,
            "lnd": [1.2 * ln10] * 4,
            "d_st": [1.0e4] * 4,
            "nu_st": [1.0e6] * 4,
        }
    )
    assert th.checks_okay is True


def test_dynamo_and_thermohaline_live_snapshots_and_adapter(
    tmp_path: pathlib.Path,
) -> None:
    """`extras_py.rs` exposes `d_es`/`d_st`/`nu_st` and steps 15M/1M cases."""
    dyn_dir = harness_runner.resolve_case_dir("star", "15M_dynamo")
    assert isinstance(
        case_extras.resolve_case_extras("star", dyn_dir),
        star_rot_mix_checks.Dynamo15MStarExtras,
    )
    he_mod = _rust.RustStarModel.from_mod_str(
        (dyn_dir / "standard_he_burn.mod").read_text(encoding="utf-8")
    )
    rot_cols = _rust.star_set_rotation_mixing_info(he_mod)
    he_snap = _rust.star_extras_snapshot(
        he_mod, str(tmp_path), include_history=False
    )
    assert list(he_snap["d_es"]) == pytest.approx(rot_cols[:, 3])
    assert list(he_snap["d_st"]) == pytest.approx(rot_cols[:, 5])
    assert list(he_snap["nu_st"]) == pytest.approx(rot_cols[:, 6])

    rep_dyn = harness_runner.run_trace_inprocess(
        "star",
        dyn_dir,
        tmp_path / "out_15m_dynamo",
        max_steps=1,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep_dyn["total_steps"]) == 3

    th_dir = harness_runner.resolve_case_dir("star", "1M_thermohaline")
    assert isinstance(
        case_extras.resolve_case_extras("star", th_dir),
        star_rot_mix_checks.Thermohaline1MStarExtras,
    )
    rep_th = harness_runner.run_trace_inprocess(
        "star",
        th_dir,
        tmp_path / "out_1m_thermohaline",
        max_steps=1,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep_th["total_steps"]) == 1


def test_diffusion_smoothness_and_radiative_levitation() -> None:
    """Verify diffusion smoothness and radiative levitation checks."""
    diff = star_rot_mix_checks.DiffusionSmoothnessStarExtras()
    diff.extras_after_evolve(
        {
            "x_integer_ctrl": [1],
            "nz": 160,
            "brunt_n2_composition_term": [2.0] * 160,
        }
    )
    assert diff.avg_d2n2 == 0.0
    assert diff.smooth_enough is True

    diff.extras_after_evolve(
        {
            "x_integer_ctrl": [1],
            "nz": 160,
            "brunt_n2_composition_term": [float(i + 1) for i in range(160)],
        }
    )
    expected_avg = (
        1.0 / 160.0 + sum(0.5 / float(k_1 + 1) for k_1 in range(157, 9, -1))
    ) / 150.0
    assert diff.avg_d2n2 == pytest.approx(expected_avg)
    assert diff.smooth_enough is True

    assert star_rot_mix_checks.compute_op_mono_factors(
        {"species": 3, "net_iso_fe56": 2, "x_ctrl": [1.75]}
    ) == [1.0, 1.75, 1.0]
    lev = star_rot_mix_checks.RadiativeLevitationStarExtras()
    lev.extras_controls({"x_ctrl": [0.0]})
    with pytest.raises(NotImplementedError, match="#161"):
        lev.extras_controls({"x_ctrl": [1.0], "op_mono_data_path": ""})
    lev.extras_after_evolve(
        {
            "x_ctrl": [1.0],
            "star_age": 1.0e7,
            "max_age": 1.0e7,
            "nz": 3,
            "species": 4,
            "dq": [2.0e-10, 0.5, 0.5],
            "mixing_type": [
                0,
                int(constants.MixingType.CONVECTIVE_MIXING),
                0,
            ],
            "net_iso_fe56": 3,
            "net_iso_ni58": 4,
            "xa_fe56": [0.01, 0.03, 0.01],
            "xa_ni58": [0.005, 0.01, 0.005],
        }
    )
    assert lev.levitation_ok is True


def test_ns_tov_and_wd_cooling_nova_burst_hooks() -> None:
    """Verify `NeutronStarEnvStarExtras`, `WdCool06MStarExtras`, and
    `WdNovaBurstStarExtras`.
    """
    ns = star_ns_wd_simplex.NeutronStarEnvStarExtras()
    cgrav = ns.other_cgrav(
        {
            "nz": 2,
            "r": [1.2e6, 1.1e6],
            "m": [1.4 * constants.MSUN, 1.39 * constants.MSUN],
            "rho": [1.0e6, 1.0e8],
            "peos": [1.0e22, 1.0e25],
        }
    )
    assert len(cgrav) == 2
    assert cgrav[0] > constants.STANDARD_CGRAV
    assert cgrav[1] > cgrav[0]

    wdc = star_ns_wd_simplex.WdCool06MStarExtras()
    cols = wdc.extra_profile_columns(
        {
            "nz": 3,
            "zbar": [6.0, 6.0, 8.0],
            "abar": [12.0, 12.0, 16.0],
            "eos_frac_pc": [0.0, 0.75, 0.5],
        }
    )
    assert cols == [
        ("zbar_div_abar", [0.5, 0.5, 0.5]),
        ("eosPC_fraction", [0.0, 0.75, 0.5]),
    ]
    assert wdc.max_frac == 0.75
    assert wdc.kmax_1based == 2

    nova = star_ns_wd_simplex.WdNovaBurstStarExtras()
    with pytest.raises(NotImplementedError, match="#150"):
        nova.extras_startup({}, True)
    nova.extras_startup({}, False)
    assert nova.extras_check_model({"l_phot": 500.0}) == "keep_going"
    assert nova.extras_check_model({"l_phot": 2.0e4}) == "keep_going"
    assert nova.num_bursts == 1
    assert nova.extras_check_model({"l_phot": 5.0e3}) == "keep_going"
    assert nova.extras_check_model({"l_phot": 800.0}) == (
        "terminate",
        "have finished burst",
    )


def test_xtra_coeff_os_and_simplex_solar_helpers(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `&xtra_coeff_os`, `PyOtherMeshDeltaCoeffFactorHook`, and simplex
    helpers.
    """
    pm_wd = star_ns_wd_simplex.PreMs1MToWdStarExtras()
    case_dir = harness_runner.resolve_case_dir("star", "1M_pre_ms_to_wd")
    assert isinstance(
        case_extras.resolve_case_extras("star", case_dir),
        star_ns_wd_simplex.PreMs1MToWdStarExtras,
    )
    pm_wd.extras_controls({"use_other_mesh_delta_coeff_factor": False})
    pm_wd.extras_controls(
        {
            "use_other_mesh_delta_coeff_factor": True,
            "case_dir": str(case_dir),
        }
    )
    assert pm_wd.params["xtra_coef_os_below_burn_h"] == 0.15
    conv = int(constants.MixingType.CONVECTIVE_MIXING)
    over = int(constants.MixingType.OVERSHOOT_MIXING)
    nomix = int(constants.MixingType.NO_MIXING)
    os_snap: dict[str, Any] = {
        "nz": 5,
        "mesh_delta_coeff_factor": [1.0] * 5,
        "xa_he4": [0.28, 0.28, 0.28, 0.28, 0.0],
        "mixing_type": [conv, conv, over, nomix, nomix],
        "eps_nuc": [0.0, 10.0, 0.0, 0.0, 0.0],
        "eps_nuc_ipp": [0.0, 8.0, 0.0, 0.0, 0.0],
        "eps_nuc_icno": [0.0, 1.0, 0.0, 0.0, 0.0],
        "eps_nuc_i3alf": [0.0] * 5,
        "peos": [1.0e15] * 5,
        "rho": [1.0] * 5,
        "grav": [1.0e5] * 5,
        "r": [5.0e10, 4.0e10, 3.0e10, 2.99e10, 1.0e10],
    }
    factors = pm_wd.eval_other_mesh_delta_coeff_factor(os_snap)
    assert factors[2] == pytest.approx(0.15)
    assert factors[3] == pytest.approx(0.15)
    assert factors[4] == 1.0

    no_snap_key = {
        k: v for k, v in os_snap.items() if k != "mesh_delta_coeff_factor"
    }
    factors_arg = pm_wd.other_mesh_delta_coeff_factor(no_snap_key, [1.0] * 5)
    assert factors_arg == factors

    zams = _rust.RustStarModel.from_zams(1.0, 0.02)
    zams_snap = _rust.star_extras_snapshot(
        zams, str(tmp_path), include_history=False
    )
    nz_zams = int(zams_snap["nz"])
    assert len(zams_snap["eps_nuc_ipp"]) == nz_zams
    assert len(zams_snap["eps_nuc_icno"]) == nz_zams
    assert len(zams_snap["eps_nuc_i3alf"]) == nz_zams

    rep = harness_runner.run_trace_inprocess(
        "star",
        case_dir,
        tmp_path / "out_1m_pre_ms_to_wd",
        max_steps=1,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep["total_steps"]) == 5

    hook_seen: dict[str, Any] = {"calls": 0}

    class _PreMsMeshProbe:
        def __init__(self) -> None:
            self._inner = star_ns_wd_simplex.PreMs1MToWdStarExtras()

        def extras_controls(self, _state: dict[str, Any]) -> dict[str, Any]:
            self._inner.read_inlist_xtra_coeff_os(case_dir)
            return {"use_other_mesh_delta_coeff_factor": True}

        def other_mesh_delta_coeff_factor(
            self,
            state: dict[str, Any],
            current: Sequence[float] | None = None,
        ) -> list[float]:
            hook_seen["calls"] += 1
            hook_seen["has_cats"] = all(
                k in state
                for k in ("eps_nuc_ipp", "eps_nuc_icno", "eps_nuc_i3alf")
            )
            return self._inner.other_mesh_delta_coeff_factor(state, current)

    adj_dir = harness_runner.resolve_case_dir("star", "adjust_net")
    probe_rep = harness_runner.run_trace_inprocess(
        "star",
        adj_dir,
        tmp_path / "out_prems_mesh_probe",
        max_steps=2,
        allow_unsupported_controls=True,
        extras=_PreMsMeshProbe(),
    )
    assert int(probe_rep["total_steps"]) >= 1
    assert hook_seen["calls"] >= 1
    assert hook_seen["has_cats"] is True

    sim = star_ns_wd_simplex.SimplexSolarCalibrationStarExtras()
    assert sim.extras_controls({}) == {}
    sim.extras_startup({}, False)
    assert sim.extras_check_model({}) == "keep_going"
    assert sim.extras_finish_step({}) == "keep_going"
    sim.extras_after_evolve({})
    simplex_dir = harness_runner.resolve_case_dir(
        "star", "simplex_solar_calibration"
    )
    assert isinstance(
        case_extras.resolve_case_extras("star", simplex_dir),
        star_ns_wd_simplex.SimplexSolarCalibrationStarExtras,
    )
    assert sim.other_mesh_functions(
        {
            "x_ctrl": [500.0, 0.02, 0.0],
            "nz": 2,
            "gradr": [0.42, 0.40],
            "grada": [0.40, 0.40],
        }
    ) == [
        (
            "gradr_grada_function",
            False,
            [pytest.approx(500.0 * float(fm_math.tanh(1.0)) * 0.02), 0.0],
        )
    ]
    assert star_ns_wd_simplex.set_simplex_my_vars({"teff": 5777.0}, True) == (
        5777.0
    )
    assert star_ns_wd_simplex.will_set_simplex_my_param(1, 1.92, True) == 1.92
    name, is_xa, gvals = star_ns_wd_simplex.eval_gradr_grada_mesh_fcn(
        {
            "x_ctrl": [500.0, 0.02, 0.0],
            "nz": 2,
            "gradr": [0.42, 0.40],
            "grada": [0.40, 0.40],
        }
    )
    assert name == "gradr_grada_function"
    assert is_xa is False
    assert gvals[0] == pytest.approx(500.0 * float(fm_math.tanh(1.0)) * 0.02)
    assert gvals[1] == 0.0
    p_val = star_ns_wd_simplex.simplex_param(0.25, 1.0, 0.8, 1.4)
    assert star_ns_wd_simplex.simplex_inverse(
        p_val, 1.0, 0.8, 1.4
    ) == pytest.approx(0.25)
    chi2, n_terms = star_ns_wd_simplex.eval_simplex_chi2(
        [(True, 5780.0, 5777.0, 3.0), (False, 10.0, 0.0, 1.0)]
    )
    assert (chi2, n_terms) == (1.0, 1)


def test_ppisn_wind_roche_kap_and_columns() -> None:
    """Verify `PpisnStarExtras` wind, Roche, opacity, and 22 history columns."""
    pp = star_ppisn.PpisnStarExtras()
    base_ctrl = {
        "x_ctrl": [0.01] * 17,
        "x_integer_ctrl": [100],
        "x_logical_ctrl": [True, False],
        "delta_lgrho_cntr_hard_limit": 0.05,
        "dt_div_min_dr_div_cs_limit": 2.0,
        "use_other_kap": False,
        "use_other_eval_fp_ft": False,
    }
    pp.extras_controls(base_ctrl)
    with pytest.raises(NotImplementedError, match="#161"):
        pp.extras_controls({**base_ctrl, "use_other_kap": True})

    w_msun = pp.other_wind(
        {"surface_h1": 0.7, "surface_he4": 0.28, "zbase": 0.0142},
        1.0e5 * constants.LSUN,
        40.0 * constants.MSUN,
        15.0 * constants.RSUN,
        35000.0,
    )
    assert w_msun > 0.0
    assert pp.other_adjust_mdot(
        {"generations": 3, "mstar_dot_old": -1.0e20}, -1.2e20
    ) == pytest.approx(-1.05e20)

    eps = 1.0e-7
    r0 = star_ppisn.eval_roche_fp_ft(
        0.5, 1.0e11, 0.5, u_flag=False, w_div_wc_flag=True
    )
    rp = star_ppisn.eval_roche_fp_ft(
        0.5 + eps, 1.0e11, 0.5, u_flag=False, w_div_wc_flag=True
    )
    rm = star_ppisn.eval_roche_fp_ft(
        0.5 - eps, 1.0e11, 0.5, u_flag=False, w_div_wc_flag=True
    )
    assert r0["dfp_daw"] == pytest.approx(
        (rp["fp"] - rm["fp"]) / (2.0 * eps), rel=1.0e-6
    )
    assert r0["dft_daw"] == pytest.approx(
        (rp["ft"] - rm["ft"]) / (2.0 * eps), rel=1.0e-6
    )

    kap_ov = star_ppisn.eval_surface_escape_kap_override(
        {
            "u_flag": True,
            "solver_iter": 0,
            "u": [1.0e9],
            "r": [1.0e11],
            "cgrav": [constants.STANDARD_CGRAV],
            "m": [10.0 * constants.MSUN],
            "x": [0.2],
        },
        1,
    )
    assert kap_ov is not None
    assert kap_ov[0] == pytest.approx(0.24)
    assert kap_ov[1:] == (0.0, 0.0)

    msun = constants.MSUN
    hstate = {
        "nz": 3,
        "u_flag": True,
        "rotation_flag": True,
        "r": [1.0e12, 5.0e11, 1.0e11],
        "q": [1.0, 0.94, 0.5],
        "m": [40.0 * msun, 37.6 * msun, 20.0 * msun],
        "x": [0.0, 0.0, 0.0],
        "y": [0.8, 0.5, 0.0],
        "cgrav": [constants.STANDARD_CGRAV] * 3,
        "dm": [1.0e33] * 3,
        "dm_bar": [1.0e33] * 3,
        "energy": [1.0e15] * 3,
        "u": [1.0e9, 1.0e6, 1.0e5],
        "j_rot": [1.0e16] * 3,
        "t": [1.0e8] * 3,
        "mu": [1.5] * 3,
        "rho": [1.0e2] * 3,
        "mlt_vc": [1.0e4] * 3,
    }
    pp.extras_startup({}, False)
    assert len(pp.extra_history_columns(hstate)) == 22
    assert len(pp.extra_profile_columns(hstate)) == 8
    assert len(pp.extra_profile_columns({**hstate, "u_flag": False})) == 1


def test_ccsn_iip_wind_nico_magnetar_and_rti() -> None:
    """Verify `CcsnIIpStarExtras`, `#245` magnetar, and `#246` RTI check."""
    ccsn = star_ccsn_iip.CcsnIIpStarExtras()
    ccsn_dir = harness_runner.resolve_case_dir("star", "ccsn_IIp")
    ccsn.extras_controls({"case_dir": str(ccsn_dir)})
    assert ccsn.stella_params["stella_num_points"] == 400

    w = ccsn.other_wind(
        {
            "i_lnd": 1,
            "nz": 3,
            "lnd": [-35.0, -34.0, -20.0],
            "dm": [1.0e28, 2.0e28, 5.0e30],
            "dt": constants.SECYER,
        },
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert w == pytest.approx(3.0e28 / constants.MSUN)

    msun = constants.MSUN
    dm = [0.1 * msun, 0.1 * msun, 0.1 * msun]
    m_arr = [0.3 * msun, 0.2 * msun, 0.1 * msun]
    xa = [[0.9, 0.9, 0.9], [0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]
    rev_ni = star_ccsn_iip.set_nico_mass_arrays(
        xa,
        dm,
        m_arr,
        i_o16_1=1,
        i_ni56_1=2,
        i_co56_1=3,
        new_ni=0.06,
        m_center=0.0,
        he_core_mass=0.0,
        x_ni_mass_start=-1.0e99,
        x_ni_mass_end=-1.0e99,
    )
    assert rev_ni == pytest.approx(0.06)

    x_ctrl = [0.0] * 100
    star_ccsn_iip.enable_magnetar({"x_ctrl": x_ctrl})
    x_ctrl_mag = list(x_ctrl)
    x_ctrl_mag[54] = 1.0e42
    with pytest.raises(RuntimeError, match="#245"):
        star_ccsn_iip.enable_magnetar({"x_ctrl": x_ctrl_mag})

    ccsn.max_mass_h = 5.0 * msun
    # approx21-like layout: neut (chem_id=1) -> slot 1, h1 (ih1=2) -> slot 2.
    h1_val = ccsn.check_rti(
        {
            "nz": 3,
            "species": 2,
            "net_iso": [1, 2, 0],
            "m": [6.0 * msun, 4.0 * msun, 2.0 * msun],
            "xa": [[0.65, 0.95, 1.0], [0.35, 0.05, 0.0]],
        }
    )
    assert h1_val == pytest.approx(0.35)

    # basic-like layout without neut: net_iso(ih1=2) = 1 -> net_iso(1) = 0 OOB.
    with pytest.raises(IndexError, match="#246"):
        ccsn.check_rti(
            {
                "nz": 3,
                "species": 2,
                "net_iso": [0, 1, 2],
                "m": [6.0 * msun, 4.0 * msun, 2.0 * msun],
                "xa": [[0.35, 0.05, 0.0], [0.65, 0.95, 1.0]],
            }
        )

    with pytest.raises(IndexError, match="#246"):
        ccsn.check_rti(
            {
                "nz": 3,
                "species": 2,
                "net_iso": [1, 0, 2],
                "m": [6.0 * msun, 4.0 * msun, 2.0 * msun],
                "xa": [[0.65, 0.95, 1.0], [0.35, 0.05, 0.0]],
            }
        )


def test_batch3_binary_jdot_ls_and_wind_fed_bhhmxb(
    tmp_path: pathlib.Path,
) -> None:
    """Verify `JdotLsCheckBinaryExtras`, `WindFedBhHmxbBinaryExtras`, and
    live adapter attachment on `binary/jdot_ls_check`.
    """
    jls = binary_extras.JdotLsCheckBinaryExtras()
    omega = 2.0 * constants.PI / constants.SECDAY
    cols = jls.extra_star_history_columns(
        {"s1_omega_avg_surf": omega, "period": 2.0 * constants.SECDAY}, 1
    )
    assert [name for name, _ in cols] == [
        "spin_period_days",
        "spin_period_hr",
        "spin_period_minutes",
        "spin_orital_period_ratio",
    ]
    assert cols[0][1] == pytest.approx(1.0)
    assert cols[1][1] == pytest.approx(24.0)
    assert cols[2][1] == pytest.approx(1440.0)
    assert cols[3][1] == pytest.approx(0.5)

    wf = binary_extras.WindFedBhHmxbBinaryExtras()
    assert wf.extras_binary_startup({}, False) == "keep_going"
    c2 = constants.CLIGHT * constants.CLIGHT
    acc_lum = 1.0e38
    mdot_edd = 1.0e18
    donor_mdot = -1.0e19
    wind_frac = 0.05
    mdot_wind_xfer = donor_mdot * wind_frac
    sub_state = {
        "model_number": 5,
        "donor_i": 1,
        "component_mdot": [0.0, 0.5 * mdot_edd],
        "mdot_edd": mdot_edd,
        "mdot_edd_eta": 0.1,
        "mdot_wind_transfer": [mdot_wind_xfer, 0.0],
        "step_mtransfer_rate": 0.0,
        "accretion_luminosity": acc_lum,
        "mdot_system_transfer": [0.0, -acc_lum / c2],
        "s1_mstar_dot": donor_mdot,
        "mtransfer_rate": 0.0,
        "wind_xfer_fraction": [wind_frac, 0.0],
        "mdot_system_wind": [donor_mdot * (1.0 - wind_frac), 0.0],
    }
    assert wf.finish_step(sub_state) == "keep_going"
    assert wf.lx_tested_sub_edd is True

    sup_wind_frac = 0.2
    sup_wind_xfer = donor_mdot * sup_wind_frac
    sup_expected_sys = sup_wind_xfer + mdot_edd - acc_lum / c2
    sup_state = {
        **sub_state,
        "mdot_wind_transfer": [sup_wind_xfer, 0.0],
        "mdot_system_transfer": [0.0, sup_expected_sys],
        "wind_xfer_fraction": [sup_wind_frac, 0.0],
        "mdot_system_wind": [donor_mdot * (1.0 - sup_wind_frac), 0.0],
    }
    assert wf.finish_step(sup_state) == "keep_going"
    wf.extras_binary_after_evolve({})
    assert wf.both_phases_tested is True

    with pytest.raises(RuntimeError, match="above Eddington"):
        wf.finish_step(
            {
                **sub_state,
                "component_mdot": [0.0, mdot_edd * 0.9 * (1.0 + 1.0e-8)],
            }
        )

    jls_dir = harness_runner.resolve_case_dir("binary", "jdot_ls_check")
    assert isinstance(
        case_extras.resolve_case_extras("binary", jls_dir),
        binary_extras.JdotLsCheckBinaryExtras,
    )
    rep = harness_runner.run_trace_inprocess(
        "binary",
        jls_dir,
        tmp_path / "out_jdot_ls",
        max_steps=2,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert rep["total_steps"] == 2


def test_batch3_handlers_raise_key_error_on_missing_snapshot_fields() -> None:
    """Every Batch 3 hook indexes snapshots directly and raises `KeyError`."""
    calls: list[Any] = [
        lambda: star_core_collapse.compute_alpha_mlt_vector({}),
        lambda: star_core_collapse.require_per_cell_alpha_mlt(
            {"x_ctrl": [2.0] * 30}, "12M_pre_ms_to_core_collapse", ":63-93"
        ),
        lambda: (
            star_core_collapse.PreMs12MCoreCollapseStarExtras().extras_after_evolve(
                {}
            )
        ),
        lambda: (
            star_core_collapse.SplitBurnBigNetStarExtras().extras_after_evolve(
                {}
            )
        ),
        lambda: (
            star_core_collapse.MakePreCcsn13bvnStarExtras().extra_history_columns(
                {}
            )
        ),
        lambda: star_rot_mix_checks.Dynamo15MStarExtras().extras_after_evolve(
            {}
        ),
        lambda: (
            star_rot_mix_checks.Thermohaline1MStarExtras().extras_after_evolve(
                {}
            )
        ),
        lambda: (
            star_rot_mix_checks.HighRotation20MZ2m2StarExtras().extra_profile_columns(
                {}
            )
        ),
        lambda: (
            star_rot_mix_checks.DiffusionSmoothnessStarExtras().extras_after_evolve(
                {}
            )
        ),
        lambda: (
            star_rot_mix_checks.RadiativeLevitationStarExtras().extras_after_evolve(
                {}
            )
        ),
        lambda: star_ns_wd_simplex.NeutronStarEnvStarExtras().other_cgrav({}),
        lambda: star_ns_wd_simplex.WdCool06MStarExtras().extra_profile_columns(
            {}
        ),
        lambda: star_ns_wd_simplex.WdNovaBurstStarExtras().extras_check_model(
            {}
        ),
        lambda: star_ns_wd_simplex.PreMs1MToWdStarExtras().extras_controls({}),
        lambda: star_ppisn.PpisnStarExtras().extras_controls({}),
        lambda: star_ccsn_iip.CcsnIIpStarExtras().other_wind(
            {}, 0.0, 0.0, 0.0, 0.0
        ),
        lambda: (
            binary_extras.JdotLsCheckBinaryExtras().extra_star_history_columns(
                {}, 1
            )
        ),
        lambda: binary_extras.WindFedBhHmxbBinaryExtras().finish_step({}),
    ]
    for fn in calls:
        with pytest.raises(KeyError):
            fn()
    assert not math.isnan(0.0)
