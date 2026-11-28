# SPDX-License-Identifier: GPL-3.0-or-later
"""Wave 3 `#231` audit tests: `make_sdb` graduation, `5M_cepheid_blue_loop`
and `accreted_material_j` audits, and unported case accounting.
"""

from __future__ import annotations

import math
import pathlib
import shutil
import tomllib
from typing import Any

import pytest

from flexmesa import _rust, constants
from flexmesa.testing import harness_runner
from tests import case_extras
from tests.case_extras import astero_extras, star_extras

_EXPECTED_UNPORTED_STAR: dict[str, type[Any]] = {
    "1.4M_ms_op_mono": star_extras.MsOpMonoStarExtras,
    "c13_pocket": star_extras.C13PocketStarExtras,
    "ccsn_IIp": star_extras.CcsnIIpStarExtras,
    "conductive_flame": star_extras.ConductiveFlameStarExtras,
    "make_co_wd": star_extras.MakeCoWdStarExtras,
    "make_pre_ccsn_13bvn": star_extras.MakePreCcsn13bvnStarExtras,
    "other_physics_hooks": star_extras.OtherPhysicsHooksStarExtras,
    "pisn": star_extras.PisnStarExtras,
    "ppisn": star_extras.PpisnStarExtras,
    "radiative_levitation": star_extras.RadiativeLevitationStarExtras,
    "wd_aic": star_extras.WdAicStarExtras,
}

_EXPECTED_UNPORTED_ASTERO: dict[str, type[Any]] = {}

# Search cases whose shared `extras_after_evolve` moves the `best_model_*`
# files (`astero/test_suite/fast_{from_file,scan_grid,simplex}`).
_EXPECTED_SAMPLE_FILES_ASTERO: dict[str, type[Any]] = {
    "fast_from_file": astero_extras.FastFromFileAsteroExtras,
    "fast_scan_grid": astero_extras.SampleFilesAsteroExtras,
    "fast_simplex": astero_extras.SampleFilesAsteroExtras,
}

_SDB_OBS: list[dict[str, Any]] = []


class _RecordingMakeSdbExtras(star_extras.MakeSdbStarExtras):
    """Test subclass recording `MakeSdbStarExtras` snapshots and directives."""

    def __init__(self, mode: str) -> None:
        super().__init__()
        self._mode = mode
        self._did_retry = False

    def extras_check_model(
        self, state: dict[str, Any]
    ) -> str | tuple[str, dict[str, float]]:
        x_ctrl = list(state["x_ctrl"])
        if self._mode == "retry_then_shutoff" and not self._did_retry:
            # Force h_lower_limit > star_mass - he_core_mass on attempt 1 so
            # MakeSdbStarExtras returns "retry", then step into shutoff window.
            x_ctrl = [2.0, 0.05, *x_ctrl[2:]]
            self._did_retry = True
        elif self._mode in ("shutoff_check", "retry_then_shutoff"):
            x_ctrl = [1.05, 0.1, *x_ctrl[2:]]
        probe_state = {**state, "x_ctrl": x_ctrl}
        res = super().extras_check_model(probe_state)
        _SDB_OBS.append(
            {
                "model_number": int(state["model_number"]),
                "reimers_scaling_factor": float(
                    state["reimers_scaling_factor"]
                ),
                "blocker_scaling_factor": float(
                    state["blocker_scaling_factor"]
                ),
                "result": res,
            }
        )
        return res


def _startup_shutoff_sdb_factory(
    _suite: str, _case_dir: pathlib.Path
) -> object:
    _SDB_OBS.clear()
    return _RecordingMakeSdbExtras(mode="startup_shutoff")


def _shutoff_sdb_factory(_suite: str, _case_dir: pathlib.Path) -> object:
    _SDB_OBS.clear()
    return _RecordingMakeSdbExtras(mode="shutoff_check")


def _retry_sdb_factory(_suite: str, _case_dir: pathlib.Path) -> object:
    _SDB_OBS.clear()
    return _RecordingMakeSdbExtras(mode="retry_then_shutoff")


def _write_synthetic_sdb_case(
    case_dir: pathlib.Path, *, x_ctrl_1: float
) -> None:
    """Create a 1 Msun ZAMS case with active Reimers/Blocker winds."""
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "rn").write_text(
        "#!/usr/bin/env bash\ndo_one inlist\n",
        encoding="utf-8",
    )
    (case_dir / "inlist").write_text(
        "&star_job\n"
        "  show_log_description_at_start = .false.\n"
        "/\n"
        "&kap\n"
        "  Zbase = 0.02d0\n"
        "/\n"
        "&controls\n"
        "  initial_mass = 1.0d0\n"
        "  max_model_number = 10\n"
        "  reimers_scaling_factor = 0.5d0\n"
        "  blocker_scaling_factor = 0.2d0\n"
        f"  x_ctrl(1) = {x_ctrl_1!r}d0\n"
        "  x_ctrl(2) = 0.1d0\n"
        "/\n",
        encoding="utf-8",
    )


def test_make_sdb_graduated_ported_and_runs_live(
    tmp_path: pathlib.Path,
) -> None:
    """`make_sdb` graduates to `ported` and runs via `PyStarExtrasAdapter`."""
    assert "MakeSdbStarExtras" in case_extras.ADAPTER_SERVED_STAR_HANDLERS
    sdb_dir = harness_runner.resolve_case_dir("star", "make_sdb")
    assert case_extras.classify_case_extras("star", sdb_dir) == "ported"
    assert isinstance(
        case_extras.resolve_case_extras("star", sdb_dir),
        star_extras.MakeSdbStarExtras,
    )

    moved = tmp_path / "relocated_sdb"
    shutil.copytree(sdb_dir / "src", moved / "src")
    assert case_extras.classify_case_extras("star", moved) == "ported"
    assert isinstance(
        case_extras.resolve_case_extras("star", moved),
        star_extras.MakeSdbStarExtras,
    )

    rep = harness_runner.run_trace_inprocess(
        "star",
        sdb_dir,
        tmp_path / "out_make_sdb",
        max_steps=1,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep["total_steps"]) == 1


def test_make_sdb_live_adapter_retry_and_wind_shutoff(
    tmp_path: pathlib.Path,
) -> None:
    """`MakeSdbStarExtras` drives live retry and wind shutoff via adapter."""
    startup_dir = tmp_path / "sdb_startup_shutoff"
    _write_synthetic_sdb_case(startup_dir, x_ctrl_1=1.05)
    rep_startup = harness_runner.run_trace_inprocess(
        "star",
        startup_dir,
        tmp_path / "out_startup_shutoff",
        max_steps=1,
        extras_factory=(
            "tests.test_case_extras_wave3:_startup_shutoff_sdb_factory"
        ),
    )
    assert int(rep_startup["total_steps"]) == 1
    assert len(_SDB_OBS) == 1
    assert _SDB_OBS[0]["reimers_scaling_factor"] == 0.0
    assert _SDB_OBS[0]["blocker_scaling_factor"] == 0.0

    shutoff_dir = tmp_path / "sdb_shutoff"
    _write_synthetic_sdb_case(shutoff_dir, x_ctrl_1=0.6)
    rep_shutoff = harness_runner.run_trace_inprocess(
        "star",
        shutoff_dir,
        tmp_path / "out_shutoff",
        max_steps=2,
        extras_factory="tests.test_case_extras_wave3:_shutoff_sdb_factory",
    )
    assert int(rep_shutoff["total_steps"]) == 2
    assert len(_SDB_OBS) == 2
    assert _SDB_OBS[0]["reimers_scaling_factor"] == 0.5
    assert _SDB_OBS[0]["blocker_scaling_factor"] == 0.2
    assert _SDB_OBS[0]["result"] == (
        "keep_going",
        {"reimers_scaling_factor": 0.0, "blocker_scaling_factor": 0.0},
    )
    assert _SDB_OBS[1]["reimers_scaling_factor"] == 0.0
    assert _SDB_OBS[1]["blocker_scaling_factor"] == 0.0

    retry_dir = tmp_path / "sdb_retry"
    _write_synthetic_sdb_case(retry_dir, x_ctrl_1=0.6)
    rep_retry = harness_runner.run_trace_inprocess(
        "star",
        retry_dir,
        tmp_path / "out_retry",
        max_steps=2,
        extras_factory="tests.test_case_extras_wave3:_retry_sdb_factory",
    )
    assert int(rep_retry["total_retries"]) >= 1
    assert len(_SDB_OBS) == 3
    assert _SDB_OBS[0]["reimers_scaling_factor"] == 0.5
    assert _SDB_OBS[0]["result"] == "retry"
    assert _SDB_OBS[1]["result"] == (
        "keep_going",
        {"reimers_scaling_factor": 0.0, "blocker_scaling_factor": 0.0},
    )
    assert _SDB_OBS[2]["reimers_scaling_factor"] == 0.0
    assert _SDB_OBS[2]["blocker_scaling_factor"] == 0.0


def test_cepheid_blue_loop_edge_fallbacks_and_full_run(
    tmp_path: pathlib.Path,
) -> None:
    """`CepheidBlueLoopStarExtras` matches upstream edges and runs full."""
    h = star_extras.CepheidBlueLoopStarExtras()
    h.extras_startup({}, restart=False)
    # Degenerate equal-luminosity edges hit exact `-1d99` / `1d-99` fallbacks.
    # For log_t = 0.0 (teff = 1.0 K), log_t > red_log_t (-1e99) and
    # log_t < blue_log_t (1e-99); teff = 2.0 gives log_t > 1e-99 to cross blue.
    degen_ctrl = [0.0, 3.8, 3.0, 3.7, 3.0, 3.6, 3.0, 3.5, 3.0]
    snap_base = {
        "x_ctrl": degen_ctrl,
        "surface_l_lsun": 1.0e3,
        "model_number": 1,
    }
    assert h.extras_check_model({**snap_base, "teff": 2.0}) == "keep_going"
    assert h.extras_check_model({**snap_base, "teff": 1.0}) == "terminate"

    cep_dir = harness_runner.resolve_case_dir("star", "5M_cepheid_blue_loop")
    rep = harness_runner.run_trace_inprocess(
        "star",
        cep_dir,
        tmp_path / "out_cepheid_full",
        max_steps=None,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep["total_steps"]) > 200
    assert rep["termination_code"] == "extras_check_model"


def test_accreted_material_j_multi_step_live_and_combined_branch(
    tmp_path: pathlib.Path,
) -> None:
    """`accreted_material_j` runs 10 live steps and handles combined stop."""
    amj_dir = harness_runner.resolve_case_dir("star", "accreted_material_j")
    rep = harness_runner.run_trace_inprocess(
        "star",
        amj_dir,
        tmp_path / "out_amj_10",
        max_steps=10,
        allow_unsupported_controls=True,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert int(rep["total_steps"]) == 10
    assert int(rep["total_retries"]) == 0
    assert rep["termination_code"] == "max_steps_cap"

    # When both am_error exceeds 1e-5 and center_h1 < 1e-6 on the same step,
    # upstream evaluates center_h1 < 1e-6 second and reports the wind status.
    h = star_extras.AccretedMaterialJStarExtras()
    h.extras_controls({})
    res = h.extras_check_model(
        {
            "model_number": 3,
            "generations": 2,
            "mstar_dot": 1.0,
            "mstar_dot_old": 1.0,
            "accreted_material_j": 1.0e18,
            "dt": 1.0e10,
            "total_angular_momentum": 1.0e20,
            "total_angular_momentum_old": 1.0e20,
            "center_h1": 1.0e-7,
        }
    )
    assert res == ("terminate", "Apparent problem with implicit wind.")
    assert h.wind_switch_ok is False


def test_relax_composition_j_entropy_surface_l_lsun_fallback_and_controls(
    tmp_path: pathlib.Path,
) -> None:
    """`RelaxCompositionJEntropyStarExtras` reads `surface_l_lsun`, updates
    `initial_mass` in part 2, and attaches under `resolve_case_extras`.
    """
    assert (
        "RelaxCompositionJEntropyStarExtras"
        in case_extras.ADAPTER_SERVED_STAR_HANDLERS
    )
    case_dir = harness_runner.resolve_case_dir(
        "star", "relax_composition_j_entropy"
    )
    assert case_extras.classify_case_extras("star", case_dir) == "ported"
    assert isinstance(
        case_extras.resolve_case_extras("star", case_dir),
        star_extras.RelaxCompositionJEntropyStarExtras,
    )

    h = star_extras.RelaxCompositionJEntropyStarExtras()
    with pytest.raises(KeyError):
        h.extras_controls({})
    assert h.extras_controls({"x_integer_ctrl": [0]}) is None
    assert h.extras_controls({"x_integer_ctrl": [1]}) is None
    p1: dict[str, Any] = {
        "x_integer_ctrl": [1],
        "case_dir": str(tmp_path),
        "teff": 7800.0,
        "nz": 1,
        "species": 2,
        "q": [1.0],
        "dq": [1.0],
        "entropy": [10.0],
        "j_rot": [1.0e15],
        "xa": [[0.7, 0.3]],
        "star_mass": 9.8,
        "surface_l_lsun": 5.0e3,
    }
    assert h.extras_finish_step(p1) == {
        "action": "terminate",
        "termination_code_slot": "t_xtra1",
        "termination_code_str": "produced files for relax",
    }
    updates = h.extras_controls(
        {"x_integer_ctrl": [2], "case_dir": str(tmp_path)}
    )
    assert updates is not None
    assert math.isclose(updates["initial_mass"], 9.8, rel_tol=1.0e-12)
    assert math.isclose(h.loaded_initial_mass or 0.0, 9.8, rel_tol=1.0e-12)
    ok_res = h.extras_check_model(
        {
            "x_integer_ctrl": [2],
            "teff": 7800.0,
            "surface_l_lsun": 5.0e3,
        }
    )
    assert ok_res == {
        "action": "terminate",
        "termination_code_slot": "t_xtra1",
        "termination_code_str": "Values for Teff and L are within tolerance",
    }
    assert math.isclose(
        _rust.log10(float(p1["surface_l_lsun"]) * constants.LSUN),
        math.log10(5.0e3 * constants.LSUN),
        rel_tol=1.0e-12,
    )
    smoke_rep = harness_runner.run_trace_inprocess(
        "star",
        case_dir,
        tmp_path / "relax_smoke_out",
        max_steps=1,
        extras_factory="tests.case_extras:resolve_case_extras",
    )
    assert smoke_rep["total_steps"] == 1


def test_wave3_ledger_counts_and_unported_audit() -> None:
    """All 11 unported `star` cases and no `astero` case match ledger/guards."""
    for case_name, expected_cls in _EXPECTED_UNPORTED_STAR.items():
        case_dir = harness_runner.resolve_case_dir("star", case_name)
        assert (
            expected_cls.__name__
            not in case_extras.ADAPTER_SERVED_STAR_HANDLERS
        )
        assert isinstance(
            case_extras.match_case_extras("star", case_dir), expected_cls
        )
        assert case_extras.resolve_case_extras("star", case_dir) is None
        assert case_extras.classify_case_extras("star", case_dir) == "unported"

    for case_name, expected_cls in _EXPECTED_UNPORTED_ASTERO.items():
        case_dir = harness_runner.resolve_case_dir("astero", case_name)
        matched = case_extras.match_case_extras("astero", case_dir)
        assert isinstance(matched, expected_cls)
        assert matched.unported_routines
        assert (
            case_extras.classify_case_extras("astero", case_dir) == "unported"
        )

    for case_name, expected_cls in _EXPECTED_SAMPLE_FILES_ASTERO.items():
        case_dir = harness_runner.resolve_case_dir("astero", case_name)
        matched = case_extras.match_case_extras("astero", case_dir)
        assert isinstance(matched, expected_cls)
        assert matched.unported_routines == ()
        assert callable(matched.extras_after_evolve)
        assert case_extras.classify_case_extras("astero", case_dir) == "ported"

    ex_dir = harness_runner.resolve_case_dir("astero", "example_astero")
    ex_matched = case_extras.match_case_extras("astero", ex_dir)
    assert isinstance(ex_matched, astero_extras.ExampleAsteroExtras)
    assert ex_matched.unported_routines == ()
    assert case_extras.classify_case_extras("astero", ex_dir) == "ported"

    ledger_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "history_harness"
        / "extras_status.toml"
    )
    rows = tomllib.loads(ledger_path.read_text(encoding="utf-8"))["cases"]
    assert len(rows) == 106
    by_case = {(r["suite"], r["case"]): r for r in rows}
    assert by_case[("star", "make_sdb")]["status"] == "ported"
    assert by_case[("star", "tzo")]["status"] == "ported"
    assert by_case[("astero", "example_astero")]["status"] == "ported"
    assert (
        by_case[("star", "relax_composition_j_entropy")]["status"] == "ported"
    )
    unported_star = {
        r["case"]
        for r in rows
        if r["suite"] == "star" and r["status"] == "unported"
    }
    unported_astero = {
        r["case"]
        for r in rows
        if r["suite"] == "astero" and r["status"] == "unported"
    }
    assert unported_star == set(_EXPECTED_UNPORTED_STAR)
    assert unported_astero == set(_EXPECTED_UNPORTED_ASTERO) == set()
    for case_name in _EXPECTED_SAMPLE_FILES_ASTERO:
        assert by_case[("astero", case_name)]["status"] == "ported"
