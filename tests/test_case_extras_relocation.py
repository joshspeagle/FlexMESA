# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify test-side extras dispatch works identically across renamed folders."""

from __future__ import annotations

import pathlib
import shutil

import pytest

from flexmesa import _rust
from flexmesa.testing import harness_runner
from tests.case_extras import resolve_case_extras
from tests.case_extras.astero_extras import SurfaceEffectsExtras
from tests.case_extras.binary_extras import JdotGrCheckBinaryExtras
from tests.case_extras.rsp_extras import RspPeriodLimitExtras
from tests.case_extras.star_extras import CheckRedoStarExtras

_EXTRAS_FACTORY = "tests.case_extras:resolve_case_extras"


def _copy_case(
    suite: str, case_name: str, dst_dir: pathlib.Path
) -> pathlib.Path:
    src_dir = harness_runner.resolve_case_dir(suite, case_name)
    return pathlib.Path(shutil.copytree(src_dir, dst_dir))


def test_resolve_case_extras_ignores_directory_name(
    tmp_path: pathlib.Path,
) -> None:
    """Extras dispatch inspects Fortran source content, never path basename."""
    rsp_a = _copy_case("star", "rsp_BLAP", tmp_path / "alpha_dir")
    rsp_b = _copy_case("star", "rsp_BLAP", tmp_path / "zeta_dir")
    assert isinstance(resolve_case_extras("star", rsp_a), RspPeriodLimitExtras)
    assert isinstance(resolve_case_extras("star", rsp_b), RspPeriodLimitExtras)

    bin_a = _copy_case("binary", "jdot_gr_check", tmp_path / "beta_dir")
    bin_b = _copy_case("binary", "jdot_gr_check", tmp_path / "omega_dir")
    assert isinstance(
        resolve_case_extras("binary", bin_a), JdotGrCheckBinaryExtras
    )
    assert isinstance(
        resolve_case_extras("binary", bin_b), JdotGrCheckBinaryExtras
    )

    redo_a = _copy_case("star", "check_redo", tmp_path / "gamma_dir")
    redo_b = _copy_case("star", "check_redo", tmp_path / "theta_dir")
    assert isinstance(resolve_case_extras("star", redo_a), CheckRedoStarExtras)
    assert isinstance(resolve_case_extras("star", redo_b), CheckRedoStarExtras)

    ast_a = _copy_case("astero", "surface_effects", tmp_path / "delta_dir")
    ast_b = _copy_case("astero", "surface_effects", tmp_path / "kappa_dir")
    assert isinstance(
        resolve_case_extras("astero", ast_a), SurfaceEffectsExtras
    )
    assert isinstance(
        resolve_case_extras("astero", ast_b), SurfaceEffectsExtras
    )


@pytest.mark.parametrize(
    ("suite", "case_name", "dir_a", "dir_b", "max_steps", "rel_log", "cols"),
    [
        (
            "star",
            "rsp_BLAP",
            "alpha_rsp",
            "zeta_rsp",
            8,
            pathlib.Path("LOGS") / "history.data",
            (
                "num_periods",
                "period",
                "growth",
                "max_v_div_cs",
                "delta_R",
                "delta_Teff",
                "delta_logL",
                "delta_Mag",
            ),
        ),
        (
            "binary",
            "jdot_gr_check",
            "beta_bin",
            "omega_bin",
            4,
            pathlib.Path("binary_history.data"),
            ("expected_separation",),
        ),
        (
            "astero",
            "surface_effects",
            "delta_ast",
            "kappa_ast",
            1,
            pathlib.Path("LOGS") / "history.data",
            (),
        ),
    ],
)
def test_relocated_case_produces_byte_identical_logs(
    tmp_path: pathlib.Path,
    suite: str,
    case_name: str,
    dir_a: str,
    dir_b: str,
    max_steps: int,
    rel_log: pathlib.Path,
    cols: tuple[str, ...],
) -> None:
    """Renamed copies produce identical output and upstream extras columns."""
    case_a = _copy_case(suite, case_name, tmp_path / dir_a)
    case_b = _copy_case(suite, case_name, tmp_path / dir_b)
    out_a = tmp_path / f"out_{dir_a}"
    out_b = tmp_path / f"out_{dir_b}"

    cap = max(max_steps, 4)
    res_a = harness_runner.run_trace_inprocess(
        suite,
        case_a,
        out_a,
        max_steps=cap,
        allow_unsupported_controls=True,
        extras_factory=_EXTRAS_FACTORY,
    )
    res_b = harness_runner.run_trace_inprocess(
        suite,
        case_b,
        out_b,
        max_steps=cap,
        allow_unsupported_controls=True,
        extras_factory=_EXTRAS_FACTORY,
    )
    assert res_a["total_steps"] == max_steps
    assert res_b["total_steps"] == max_steps
    assert res_a["termination_code"] == res_b["termination_code"]

    log_a = out_a / rel_log
    log_b = out_b / rel_log
    assert log_a.is_file()
    assert log_b.is_file()
    assert log_a.read_bytes() == log_b.read_bytes()

    if cols:
        parsed = _rust.mesa_read_log_file(str(log_a))
        headers = set(parsed["column_names"])
        assert set(cols).issubset(headers)


def test_stripped_fortran_extras_disables_hooks_and_extra_columns(
    tmp_path: pathlib.Path,
) -> None:
    """Deleting upstream extras Fortran sources removes hook dispatch."""
    rsp_dir = _copy_case("star", "rsp_BLAP", tmp_path / "rsp_no_extras")
    (rsp_dir / "src" / "run_star_extras.f90").unlink()
    assert resolve_case_extras("star", rsp_dir) is None
    rsp_out = tmp_path / "out_rsp_no_extras"
    harness_runner.run_trace_inprocess(
        "star",
        rsp_dir,
        rsp_out,
        max_steps=4,
        allow_unsupported_controls=True,
        extras_factory=_EXTRAS_FACTORY,
    )
    rsp_parsed = _rust.mesa_read_log_file(
        str(rsp_out / "LOGS" / "history.data")
    )
    assert "num_periods" not in set(rsp_parsed["column_names"])

    bin_dir = _copy_case("binary", "jdot_gr_check", tmp_path / "bin_no_extras")
    (bin_dir / "src" / "run_binary_extras.f90").unlink()
    assert resolve_case_extras("binary", bin_dir) is None
    bin_out = tmp_path / "out_bin_no_extras"
    harness_runner.run_trace_inprocess(
        "binary",
        bin_dir,
        bin_out,
        max_steps=4,
        allow_unsupported_controls=True,
        extras_factory=_EXTRAS_FACTORY,
    )
    bin_parsed = _rust.mesa_read_log_file(str(bin_out / "binary_history.data"))
    assert "expected_separation" not in set(bin_parsed["column_names"])


def test_rsp_check_2nd_crossing_terminates_cleanly_in_build_model(
    tmp_path: pathlib.Path,
) -> None:
    """Relocated `rsp_check_2nd_crossing` finishes cleanly at 0 steps."""
    case_dir = _copy_case(
        "star", "rsp_check_2nd_crossing", tmp_path / "relocated_2nd_crossing"
    )
    out_dir = tmp_path / "out_2nd_crossing"
    rep = harness_runner.run_trace_inprocess(
        "star",
        case_dir,
        out_dir,
        allow_unsupported_controls=True,
        extras_factory=_EXTRAS_FACTORY,
    )
    assert rep["total_steps"] == 0
    assert rep["termination_code"] == "other_rsp_build_model"
