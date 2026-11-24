# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end PyO3 snapshot, directive, and live adapter tests for Issue #231
Batch 4 (`ADAPTER_SERVED_STAR_HANDLERS` and `ADAPTER_SERVED_BINARY_HANDLERS`
graduation).
"""

from __future__ import annotations

import pathlib
import shutil
import tomllib
from typing import Any

import numpy as np
import pytest

from flexmesa import _rust, constants
from flexmesa.testing import harness_runner
from tests import case_extras
from tests.case_extras import astero_extras, binary_extras, star_extras

_MINIMAL_HISTORY_COLUMNS = "model_number\nstar_age\nlog_L\nlog_Teff\n"

_GRADUATED_LIVE_CASES: tuple[tuple[str, str, int], ...] = (
    ("binary", "jdot_ls_check", 1),
    ("binary", "wind_fed_bhhmxb", 2),
    ("star", "diffusion_smoothness", 1),
    ("star", "conv_core_cpm", 1),
    ("star", "hot_cool_wind", 1),
    ("star", "semiconvection", 1),
    ("star", "conserve_angular_momentum", 2),
    ("star", "hse_riemann", 1),
    ("star", "ns_h", 1),
    ("star", "ns_he", 1),
)

_STAGED_HISTORY_COLUMN_CASES: tuple[tuple[str, int], ...] = (
    ("adjust_net", 1),
    ("make_o_ne_wd", 1),
    ("ns_c", 1),
    ("cburn_inward", 1),
    ("high_rot_darkening", 1),
)

_BATCH4_GRADUATED_STAR_CASES: tuple[str, ...] = (
    "adjust_net",
    "cburn_inward",
    "conserve_angular_momentum",
    "conv_core_cpm",
    "diffusion_smoothness",
    "high_rot_darkening",
    "hot_cool_wind",
    "hse_riemann",
    "make_o_ne_wd",
    "ns_c",
    "ns_h",
    "ns_he",
    "semiconvection",
)


@pytest.fixture(scope="module")
def zams_star_model() -> _rust.RustStarModel:
    """Build one shared solar ZAMS `RustStarModel` for snapshot unit tests."""
    return _rust.RustStarModel.from_zams(1.0, 0.02)


def test_star_extras_snapshot_supplies_general_fields_and_omits_absent_isos(
    zams_star_model: _rust.RustStarModel, tmp_path: pathlib.Path
) -> None:
    """`star_extras_snapshot` exposes MESA scalars, controls, report summaries,
    radial arrays, and active isotopes while omitting absent isotopes.
    """
    snap = _rust.star_extras_snapshot(
        zams_star_model, str(tmp_path), include_history=False
    )
    nz = int(snap["nz"])
    assert nz > 100
    assert int(snap["species"]) == len(snap["net_iso"])

    for scalar_key in (
        "model_number",
        "star_age",
        "time",
        "dt",
        "dt_next",
        "mstar",
        "star_mass",
        "mstar_dot",
        "star_mdot",
        "teff",
        "surface_l_lsun",
        "log_surface_luminosity",
        "total_energy",
        "total_energy_end",
        "atm_t_tau_relation",
        "log_l_upper_limit",
        "accretion_h1",
        "reimers_scaling_factor",
        "blocker_scaling_factor",
        "do_element_diffusion",
        "center_avg_value_dq",
        "he_core_boundary_h1_fraction",
        "create_pre_main_sequence_model",
        "need_to_save_profiles_now",
        "use_other_remove_surface",
        "he_core_mass",
        "co_core_mass",
        "power_h_burn",
        "power_he_burn",
        "l_by_category_icc",
        "l_phot",
        "mass_conv_core",
        "conv_mx1_top",
        "conv_mx1_bot",
        "w_div_w_crit_avg_surf",
        "omega_avg_surf",
        "v_rot_avg_surf",
        "log_center_temperature",
        "log_center_density",
        "center_omega",
        "he_core_omega",
        "center_h1",
        "center_he4",
        "total_angular_momentum",
        "surface_c12",
    ):
        assert scalar_key in snap, f"missing scalar key {scalar_key!r}"

    for radial_key in (
        "r",
        "m",
        "q",
        "dq",
        "dm",
        "dm_bar",
        "t",
        "rho",
        "lnd",
        "lnt",
        "peos",
        "energy",
        "entropy",
        "csound",
        "gamma1",
        "abar",
        "zbar",
        "tau",
        "v",
        "u",
        "omega",
        "j_rot",
        "net_nuclear_energy",
        "mixing_type",
        "brunt_n2_composition_term",
        "w_div_w_crit_roche",
        "eos_frac_opal_scvh",
        "eos_frac_helm",
        "eos_frac_skye",
        "eos_frac_pc",
        "eos_frac_free_eos",
        "eos_frac_cms",
        "eos_frac_ideal",
        "xa_h1",
        "xa_he4",
        "xa_c12",
    ):
        arr = snap[radial_key]
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (nz,)

    # Absent isotopes in `basic.net` must stay omitted so hooks fail loudly.
    for absent_key in ("xa_c13", "xa_na24", "xa_fe56", "net_iso_fe56"):
        assert absent_key not in snap


def test_star_apply_extras_updates_controls_and_invalid_keys() -> None:
    """`star_apply_extras_updates` updates supported controls and raises
    `ValueError` on unrecognized keys or invalid `atm_t_tau_relation` values.
    """
    model = _rust.RustStarModel.from_zams(1.0, 0.02)
    for rel in (
        "Eddington",
        "solar_Hopf",
        "Krishna_Swamy",
        "Trampedach_solar",
    ):
        _rust.star_apply_extras_updates(model, {"atm_t_tau_relation": rel})
        assert _rust.star_extras_snapshot(model)["atm_t_tau_relation"] == rel

    _rust.star_apply_extras_updates(
        model,
        {
            "reimers_scaling_factor": 0.25,
            "blocker_scaling_factor": 0.5,
            "do_element_diffusion": True,
            "mass_change": -1.0e-8,
            "mixing_length_alpha": 1.9,
            "max_timestep": 1.0e12,
            "initial_mass": 1.2,
            "log_l_upper_limit": 4.5,
            "teff_lower_limit": 3000.0,
            "dt_next": 1.0e9,
        },
    )
    snap = _rust.star_extras_snapshot(model)
    assert snap["reimers_scaling_factor"] == pytest.approx(0.25)
    assert snap["blocker_scaling_factor"] == pytest.approx(0.5)
    assert snap["do_element_diffusion"] is True
    assert snap["mass_change"] == pytest.approx(-1.0e-8)
    assert snap["mixing_length_alpha"] == pytest.approx(1.9)
    assert snap["max_timestep"] == pytest.approx(1.0e12)
    assert snap["initial_mass"] == pytest.approx(1.2)
    assert snap["log_l_upper_limit"] == pytest.approx(4.5)
    assert snap["teff_lower_limit"] == pytest.approx(3000.0)
    assert snap["dt_next"] == pytest.approx(1.0e9)

    with pytest.raises(ValueError, match="atm_t_tau_relation"):
        _rust.star_apply_extras_updates(model, {"atm_t_tau_relation": "bogus"})
    with pytest.raises(ValueError, match="unsupported star extras update key"):
        _rust.star_apply_extras_updates(model, {"not_a_real_control": 1.0})


def test_all_graduated_star_handlers_execute_on_real_star_model_snapshots(
    zams_star_model: _rust.RustStarModel, tmp_path: pathlib.Path
) -> None:
    """Every newly graduated star handler runs cleanly on real `RustStarModel`
    snapshots built by `extras_py.rs` with zero `KeyError` or `TypeError`.
    """
    snap = _rust.star_extras_snapshot(
        zams_star_model, str(tmp_path), include_history=True
    )
    nz = int(snap["nz"])

    # RCrB optional stage writes `inlist_nzhi` using radial `t` array.
    rcrb = star_extras.RCrBStarExtras()
    rcrb.extras_after_evolve(
        {**snap, "x_logical_ctrl": [True, *snap["x_logical_ctrl"][1:]]}
    )
    assert (tmp_path / "inlist_nzhi").is_file()
    assert rcrb.saved_nzhi is not None and 1 <= rcrb.saved_nzhi <= nz + 1

    an = star_extras.AdjustNetStarExtras()
    an.extras_after_evolve({**snap, "x_integer_ctrl": [2]})
    assert an.testhub_extras["num_isos"] == float(snap["species"])

    assert star_extras.MakeONeWdStarExtras().extras_finish_step(
        {**snap, "x_integer_ctrl": [2], "x_ctrl": [10.0, 1.0e5, 1.0]}
    ) in ("keep_going", "terminate")

    hcw = star_extras.HotCoolWindStarExtras()
    hcw.extras_after_evolve(snap)
    assert isinstance(hcw.within_tolerance, bool)

    ccc = star_extras.ConvCoreCpmStarExtras()
    ccc.extras_after_evolve({**snap, "x_ctrl": [0.0, 1.0]})
    assert ccc.mass_conv_core is not None

    sc = star_extras.SemiconvectionStarExtras()
    sc.extras_after_evolve(snap)
    assert isinstance(sc.checks_okay, bool)

    ds = star_extras.DiffusionSmoothnessStarExtras()
    ds.extras_after_evolve({**snap, "x_integer_ctrl": [1]})
    assert ds.avg_d2n2 is not None

    hr = star_extras.HseRiemannStarExtras()
    hr.extras_finish_step({**snap, "x_ctrl": [0.0] * 8 + [0.5]})
    hr.extras_after_evolve(snap)
    assert len(hr.compute_pmax_history_columns(snap, n=12)) == 12

    ns = star_extras.NeutronStarEnvStarExtras()
    assert len(ns.other_cgrav(snap)) == nz

    nb = star_extras.WdNovaBurstStarExtras()
    nb.extras_startup(snap, False)
    assert nb.extras_check_model(snap) == "keep_going"
    assert len(nb.extra_profile_columns(snap)[0][1]) == nz

    wa = star_extras.WdAccSmallDmStarExtras()
    wa.extras_startup(snap, False)
    assert len(wa.extra_history_columns(snap)) == 1

    hb = star_extras.Hb2MStarExtras()
    hb.extras_startup(snap, False)
    assert hb.extras_finish_step({**snap, "x_integer_ctrl": [3]}) == (
        "keep_going"
    )
    assert len(hb.extra_history_columns(snap)) == 1
    hb.extras_after_evolve(
        {**snap, "x_integer_ctrl": [3], "x_ctrl": [0.1, 0.3]}
    )
    assert "mass_conv_core_y050" in hb.testhub_extras

    pi = star_extras.PisnStarExtras()
    pi.extras_controls(snap)
    pi.extras_startup(snap, False)
    assert len(pi.extra_history_columns(snap)) == 1
    pi.extras_finish_step({**snap, "x_integer_ctrl": [7]})
    pi.extras_after_evolve({**snap, "x_integer_ctrl": [7]})
    assert "gamma1_cntr_pulse_start" in pi.testhub_extras

    tt = star_extras.TTauGradrStarExtras()
    tt.extras_startup(snap, False)
    assert tt.extras_check_model(
        {**snap, "model_number": 10, "x_ctrl": [1.0]}
    ) == ("keep_going", {"atm_t_tau_relation": "solar_Hopf"})

    cj = star_extras.ConserveAngularMomentumStarExtras()
    cj.extras_controls(snap)
    assert cj.extras_check_model(snap) == "keep_going"
    cj.extras_after_evolve(snap)

    hrd = star_extras.HighRotDarkeningStarExtras()
    assert hrd.extras_finish_step(snap) == "keep_going"
    assert hrd.last_w_div_w_crit_avg_surf is not None
    assert hrd.last_w_div_w_crit_roche_surf is not None

    sdb = star_extras.MakeSdbStarExtras()
    sdb.extras_startup(snap, False)
    assert sdb.extras_check_model(snap) == "keep_going"

    # Real upstream `.mod` files carrying heavy isotopes (`c13`, `ne20`,
    # `na23`, `na24`, `mg24`).
    cb_dir = harness_runner.resolve_case_dir("star", "cburn_inward")
    cb_mod = _rust.RustStarModel.from_mod_str(
        (cb_dir / "standard_end_core_he_burn.mod").read_text(encoding="utf-8")
    )
    cb_snap = _rust.star_extras_snapshot(cb_mod, str(tmp_path), True)
    cb = star_extras.CburnInwardStarExtras()
    cb.extras_startup(cb_snap, False)
    cb.extras_finish_step({**cb_snap, "x_integer_ctrl": [2]})
    cb.extras_after_evolve({**cb_snap, "x_integer_ctrl": [2]})
    assert len(cb.extra_history_columns(cb_snap)) == 4

    c13_dir = harness_runner.resolve_case_dir("star", "c13_pocket")
    c13_mod = _rust.RustStarModel.from_mod_str(
        (c13_dir / "standard_2M_3DUP.mod").read_text(encoding="utf-8")
    )
    c13_snap = _rust.star_extras_snapshot(c13_mod, str(tmp_path), True)
    c13 = star_extras.C13PocketStarExtras()
    for part in (1, 2, 4):
        part_snap = {
            **c13_snap,
            "x_integer_ctrl": [part],
            "x_ctrl": [0.075, 0.58, 4.0e-5, 2.7e-4],
        }
        c13.extras_startup(part_snap, False)
        c13.extras_finish_step(part_snap)
        c13.extras_after_evolve(part_snap)
    assert "max_c13" in c13.testhub_extras

    aic_dir = harness_runner.resolve_case_dir("star", "wd_aic")
    aic_txt = (aic_dir / "wd_aic.mod").read_text(encoding="utf-8")
    # The case's own `wd_aic.net` (which has `o20`) is resolved from the case
    # directory, as upstream `read_model` does.
    aic_mod = _rust.RustStarModel.from_mod_str(aic_txt, str(aic_dir))
    aic_snap = _rust.star_extras_snapshot(aic_mod, str(tmp_path), True)
    aic = star_extras.WdAicStarExtras()
    assert aic.extras_check_model({**aic_snap, "x_integer_ctrl": [1]}) == (
        "keep_going"
    )


def test_all_graduated_binary_handlers_execute_on_real_binary_system_snapshot(
    zams_star_model: _rust.RustStarModel, tmp_path: pathlib.Path
) -> None:
    """`JdotLsCheckBinaryExtras` and `WindFedBhHmxbBinaryExtras` read real
    `binary_extras_snapshot` fields cleanly.
    """
    bin_sys = _rust.RustBinarySystem.new_point_mass(zams_star_model, 1.4, 2.0)
    bsnap = _rust.binary_extras_snapshot(bin_sys, str(tmp_path))

    jl = binary_extras.JdotLsCheckBinaryExtras()
    cols = jl.extra_star_history_columns(bsnap, 1)
    assert [name for name, _ in cols] == [
        "spin_period_days",
        "spin_period_hr",
        "spin_period_minutes",
        "spin_orital_period_ratio",
    ]

    wf = binary_extras.WindFedBhHmxbBinaryExtras()
    assert wf.extras_binary_startup(bsnap, False) == "keep_going"
    for required_key in (
        "component_mdot",
        "mdot_edd",
        "mdot_edd_eta",
        "donor_i",
        "mdot_wind_transfer",
        "step_mtransfer_rate",
        "accretion_luminosity",
        "mdot_system_transfer",
        "model_number",
        "s1_mstar_dot",
        "mtransfer_rate",
        "wind_xfer_fraction",
        "mdot_system_wind",
    ):
        assert required_key in bsnap


@pytest.mark.parametrize(("suite", "case", "cap"), _GRADUATED_LIVE_CASES)
def test_graduated_cases_run_live_through_pyo3_adapters(
    suite: str, case: str, cap: int, tmp_path: pathlib.Path
) -> None:
    """Graduated star and binary cases attach their handlers via
    `resolve_case_extras` and step cleanly through the real PyO3 adapters.
    """
    case_dir = harness_runner.resolve_case_dir(suite, case)
    assert case_extras.resolve_case_extras(suite, case_dir) is not None
    rep = harness_runner.run_trace_inprocess(
        suite,
        case_dir,
        tmp_path / f"out_{suite}_{case}",
        max_steps=cap,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep["total_steps"]) >= cap


@pytest.mark.parametrize(("case", "cap"), _STAGED_HISTORY_COLUMN_CASES)
def test_history_column_blocked_graduated_cases_step_live_when_staged(
    case: str, cap: int, tmp_path: pathlib.Path
) -> None:
    """Graduated cases whose raw `history_columns.list` hits `#163` still step
    cleanly through `PyStarExtrasAdapter` once a valid history list is staged.
    """
    src_dir = harness_runner.resolve_case_dir("star", case)
    assert case_extras.resolve_case_extras("star", src_dir) is not None
    staged = tmp_path / f"staged_{case}"
    shutil.copytree(src_dir, staged)
    (staged / "history_columns.list").write_text(
        _MINIMAL_HISTORY_COLUMNS, encoding="utf-8"
    )
    rep = harness_runner.run_trace_inprocess(
        "star",
        staged,
        tmp_path / f"out_{case}",
        max_steps=cap,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep["total_steps"]) >= cap


def test_unserved_star_handlers_detached_and_binaries_covered() -> None:
    """Every star case still marked `unported` in `extras_status.toml` stays
    detached, all Batch-4 graduated star cases are `ported`, and zero binary
    manifest cases remain `unported`.
    """
    ledger_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "benchmarks/history_harness/extras_status.toml"
    )
    ledger = tomllib.loads(ledger_path.read_text(encoding="utf-8"))

    star_unported = [
        c["case"]
        for c in ledger["cases"]
        if c["suite"] == "star" and c["status"] == "unported"
    ]
    for case in star_unported:
        case_dir = harness_runner.resolve_case_dir("star", case)
        assert case_extras.match_case_extras("star", case_dir) is not None
        assert case_extras.resolve_case_extras("star", case_dir) is None
        assert case_extras.classify_case_extras("star", case_dir) == "unported"

    binary_unported = [
        c["case"]
        for c in ledger["cases"]
        if c["suite"] == "binary" and c["status"] == "unported"
    ]
    assert binary_unported == []

    star_ported = [
        c["case"]
        for c in ledger["cases"]
        if c["suite"] == "star" and c["status"] == "ported"
    ]
    assert len(star_ported) >= 26
    assert set(_BATCH4_GRADUATED_STAR_CASES) <= set(star_ported)


class _NonDictStartupExtras:
    """Hook object whose `extras_startup` returns a non-dict, non-None value."""

    def extras_startup(self, snapshot: dict[str, object], restart: bool) -> str:
        return "not a dict"


class _NonDictUpdatesExtras:
    """Hook object whose `extras_check_model` returns non-dict `updates`."""

    def extras_check_model(
        self, snapshot: dict[str, object]
    ) -> dict[str, object]:
        return {"action": "keep_going", "updates": ["mixing_length_alpha"]}


def _non_dict_startup_factory(suite: str, case_dir: pathlib.Path) -> object:
    return _NonDictStartupExtras()


def _non_dict_updates_factory(suite: str, case_dir: pathlib.Path) -> object:
    return _NonDictUpdatesExtras()


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        ("_non_dict_startup_factory", "must return None or a dict"),
        ("_non_dict_updates_factory", "expected a dict"),
    ],
)
def test_adapter_rejects_non_dict_hook_returns(
    factory: str, match: str, tmp_path: pathlib.Path
) -> None:
    """A hook that returns updates in any form other than a dict stops the run
    with an error instead of having the value ignored.
    """
    case_dir = harness_runner.resolve_case_dir("star", "semiconvection")
    with pytest.raises(Exception, match=match):
        harness_runner.run_trace_inprocess(
            "star",
            case_dir,
            tmp_path / f"out_{factory}",
            max_steps=1,
            allow_unsupported_controls=True,
            extras_factory=f"tests.test_case_extras_batch4:{factory}",
        )


def test_astero_adipls_resolves_and_graduated_in_extras_status(
    tmp_path: pathlib.Path,
) -> None:
    """`astero_adipls` matches by Fortran source and is `ported` in ledger."""
    orig = harness_runner.resolve_case_dir("astero", "astero_adipls")
    assert isinstance(
        case_extras.match_case_extras("astero", orig),
        astero_extras.AsteroAdiplsExtras,
    )
    assert isinstance(
        case_extras.resolve_case_extras("astero", orig),
        astero_extras.AsteroAdiplsExtras,
    )
    assert case_extras.classify_case_extras("astero", orig) == "ported"

    moved = tmp_path / "renamed_astero_adipls"
    shutil.copytree(orig / "src", moved / "src")
    assert case_extras.classify_case_extras("astero", moved) == "ported"
    (moved / "src" / "run_star_extras.f90").unlink()
    assert case_extras.match_case_extras("astero", moved) is None

    ledger_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "benchmarks/history_harness/extras_status.toml"
    )
    ledger = tomllib.loads(ledger_path.read_text(encoding="utf-8"))
    by_case = {(c["suite"], c["case"]): c for c in ledger["cases"]}
    assert by_case[("astero", "astero_adipls")]["status"] == "ported"


def test_astero_adipls_pyo3_binding_and_extras_unit(
    tmp_path: pathlib.Path,
) -> None:
    """Exercise `AsteroAdipls` error checks, caching, and extras hook."""
    orig = harness_runner.resolve_case_dir("astero", "astero_adipls")

    empty_dir = tmp_path / "missing_adipls"
    empty_dir.mkdir()
    (empty_dir / "inlist_astero_search_controls").write_text(
        "&astero_search_controls\n  oscillation_code = 'adipls'\n/\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        _rust.AsteroAdipls(str(empty_dir))

    # Default `do_redistribute_mesh = .true.` requires `redistrb.c.pruned.in`.
    no_redistrb = tmp_path / "no_redistrb"
    no_redistrb.mkdir()
    shutil.copy2(
        orig / "adipls.c.pruned.in", no_redistrb / "adipls.c.pruned.in"
    )
    (no_redistrb / "inlist_astero_search_controls").write_text(
        "&astero_search_controls\n  oscillation_code = 'adipls'\n/\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        _rust.AsteroAdipls(str(no_redistrb))

    gyre_code = tmp_path / "wrong_osc_code"
    gyre_code.mkdir()
    shutil.copy2(orig / "adipls.c.pruned.in", gyre_code / "adipls.c.pruned.in")
    (gyre_code / "inlist_astero_search_controls").write_text(
        "&astero_search_controls\n"
        "  oscillation_code = 'gyre'\n"
        "  do_redistribute_mesh = .false.\n"
        "/\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        _rust.AsteroAdipls(str(gyre_code))

    # Capture a real `StarModelHandle` from a 1-step `astero_adipls` trace.
    captured: dict[str, Any] = {}

    class _HandleCaptureProbe:
        wants_star_model = True

        def extras_after_evolve(self, state: dict[str, Any]) -> None:
            captured["handle"] = state["star_model"]
            captured["photosphere_r"] = float(state["photosphere_r"])
            captured["cgrav0"] = float(state["cgrav"][0])
            captured["star_mass"] = float(state["star_mass"])

    _rust.trace_astero_case(
        str(orig),
        str(tmp_path / "cap_out"),
        max_steps_per_stage=1,
        allow_unsupported_controls=False,
        extras=_HandleCaptureProbe(),
    )
    handle = captured["handle"]
    r_cm = constants.RSUN * captured["photosphere_r"]
    g_cgs = captured["cgrav0"]
    m_g = constants.MSUN * captured["star_mass"]

    adipls = _rust.AsteroAdipls(str(orig))
    with pytest.raises(RuntimeError, match="before store_model"):
        adipls.get_one_el_info(
            handle,
            0,
            50.0,
            1000.0,
            200,
            r_cm,
            g_cgs,
            m_g,
            True,
            False,
            True,
            False,
            False,
        )
    with pytest.raises(RuntimeError, match="not ported"):
        adipls.get_one_el_info(
            handle,
            0,
            50.0,
            1000.0,
            200,
            r_cm,
            g_cgs,
            m_g,
            True,
            False,
            True,
            True,
            True,
        )
    with pytest.raises(ValueError, match="overflows i32"):
        adipls.get_one_el_info(
            handle,
            1 << 40,
            50.0,
            1000.0,
            200,
            r_cm,
            g_cgs,
            m_g,
            True,
            False,
            True,
            False,
            True,
        )

    l0_modes = adipls.get_one_el_info(
        handle,
        0,
        50.0,
        1000.0,
        200,
        r_cm,
        g_cgs,
        m_g,
        True,
        False,
        True,
        False,
        True,
    )
    assert len(l0_modes) >= 5
    assert all(m[0] == 0 and m[4] > 0.0 and m[3] > 0.0 for m in l0_modes)
    assert [m[4] for m in l0_modes] == sorted(m[4] for m in l0_modes)

    # Reuse the cached AMDL store (`store_for_adipls=False`) for `l=1`.
    l1_modes = adipls.get_one_el_info(
        handle,
        1,
        50.0,
        1000.0,
        200,
        r_cm,
        g_cgs,
        m_g,
        True,
        False,
        True,
        False,
        False,
    )
    assert len(l1_modes) >= 5
    assert all(m[0] == 1 for m in l1_modes)

    ex = astero_extras.AsteroAdiplsExtras()
    assert ex.wants_star_model is True
    ex.extras_startup({"case_dir": str(orig)}, False)
    base_state: dict[str, Any] = {
        "case_dir": str(orig),
        "star_model": handle,
        "photosphere_r": captured["photosphere_r"],
        "cgrav": [captured["cgrav0"]],
        "star_mass": captured["star_mass"],
        "x_ctrl": [375.0],
        "model_number": 49,
    }
    assert ex.extras_check_model(base_state) == "keep_going"
    assert len(ex.calls) == 0
    assert ex.extras_check_model({**base_state, "model_number": 50}) == (
        "keep_going"
    )
    assert len(ex.calls) == 1 and ex.calls[0][0] == 50

    n4_freq = next(m[4] for m in l0_modes if m[1] == 4)
    ex.extras_after_evolve({**base_state, "x_ctrl": [n4_freq]})
    assert ex.okay is True
    assert ex.actual_freq == pytest.approx(n4_freq)
    ex.extras_after_evolve({**base_state, "x_ctrl": [n4_freq * 1.1]})
    assert ex.okay is False


def test_astero_adipls_runs_strict_live_through_pyo3_adapter(
    tmp_path: pathlib.Path,
) -> None:
    """`astero/astero_adipls` runs strictly (`controls_ignored=False`) and
    executes `AsteroAdiplsExtras` at model 50 and `extras_after_evolve`.
    """
    case_dir = harness_runner.resolve_case_dir("astero", "astero_adipls")
    rep = harness_runner.run_trace_inprocess(
        "astero",
        case_dir,
        tmp_path / "out_astero_adipls",
        max_steps=50,
        allow_unsupported_controls=False,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep["total_steps"]) == 50
