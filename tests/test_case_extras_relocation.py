# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify test-side extras dispatch works identically across renamed folders."""

from __future__ import annotations

import math
import pathlib
import shutil
import types
from typing import Any

import pytest

from flexmesa import _rust
from flexmesa.testing import artifact_compare, case_manifest, harness_runner
from tests.case_extras import resolve_case_extras
from tests.case_extras.astero_extras import SurfaceEffectsExtras
from tests.case_extras.binary_extras import JdotGrCheckBinaryExtras
from tests.case_extras.rsp_extras import (
    RspCheck2ndCrossingExtras,
    RspPeriodLimitExtras,
)
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
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Relocated `rsp_check_2nd_crossing` writes to `out_dir`, emits stdout,
    and exits via `other_rsp_build_model` at 0 steps.
    """
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

    out_artifact = out_dir / "7B_2nd_crossing.data"
    assert out_artifact.is_file()
    assert not (case_dir / "7B_2nd_crossing.data").exists()

    captured = capfd.readouterr().out
    stripped_lines = [ln.strip() for ln in captured.splitlines()]
    assert "rsp_check_2nd_crossing" in stripped_lines
    assert "done rsp_check_2nd_crossing" in stripped_lines
    assert stripped_lines.index("rsp_check_2nd_crossing") < (
        stripped_lines.index("done rsp_check_2nd_crossing")
    )
    assert "7B_2nd_crossing.data" in stripped_lines
    assert "failed in other_rsp_build_model" in stripped_lines

    repo_root = case_manifest.repo_root()
    entry = case_manifest.load_case_manifest().by_key()[
        ("star", "rsp_check_2nd_crossing")
    ]
    assert entry.artifact is not None
    assert entry.artifact.stdout_reference
    ref_scan = artifact_compare.parse_instability_strip(
        (repo_root / entry.artifact.reference).read_text(encoding="utf-8")
    )
    ref_stdout = artifact_compare.parse_stdout_models(
        (repo_root / entry.artifact.stdout_reference).read_text(
            encoding="utf-8"
        )
    )
    scan = artifact_compare.parse_instability_strip(
        out_artifact.read_text(encoding="utf-8")
    )
    assert len(scan.header) == 7
    assert list(scan.header) == list(ref_scan.header)
    for key, ref_val in ref_scan.header.items():
        assert math.isclose(scan.header[key], ref_val, rel_tol=1.0e-12)

    parsed_models = artifact_compare.parse_stdout_models(captured)
    assert len(scan.models) == 16
    assert (
        [m[0] for m in scan.models]
        == [m[0] for m in ref_scan.models]
        == [m.model_number for m in parsed_models]
        == [m.model_number for m in ref_stdout]
    )
    for run_row, ref_row in zip(scan.models, ref_scan.models, strict=True):
        assert math.isclose(run_row[1], ref_row[1], rel_tol=1.0e-6)
        assert math.isclose(run_row[2], ref_row[2], abs_tol=1.0e-4)
        assert math.isclose(run_row[3], ref_row[3], rel_tol=1.0e-10)
        assert math.isclose(run_row[4], ref_row[4], rel_tol=1.0e-10)
        assert math.isclose(run_row[5], ref_row[5], rel_tol=1.0e-10)

    assert set(scan.edges) == {"Teff_red_edge", "Teff_blue_edge"}
    assert (
        abs(scan.edges["Teff_red_edge"] - ref_scan.edges["Teff_red_edge"])
        < 1.0e-2
    )
    assert (
        abs(scan.edges["Teff_blue_edge"] - ref_scan.edges["Teff_blue_edge"])
        < 1.0e-2
    )
    assert len(scan.table) == 11
    assert [row[0] for row in scan.table] == [row[0] for row in ref_scan.table]
    for run_t, ref_t in zip(scan.table, ref_scan.table, strict=True):
        assert math.isclose(run_t[1], ref_t[1], abs_tol=1.0e-2)
        assert math.isclose(run_t[2], ref_t[2], abs_tol=1.0e-2)

    stdout_edges: dict[str, float] = {}
    for ln in stripped_lines:
        toks = ln.split()
        if len(toks) == 2 and toks[0] in ("Teff_red_edge", "Teff_blue_edge"):
            stdout_edges[toks[0]] = float(toks[1])
    assert set(stdout_edges) == {"Teff_red_edge", "Teff_blue_edge"}
    assert math.isclose(
        stdout_edges["Teff_red_edge"],
        scan.edges["Teff_red_edge"],
        abs_tol=1.0e-9,
    )
    assert math.isclose(
        stdout_edges["Teff_blue_edge"],
        scan.edges["Teff_blue_edge"],
        abs_tol=1.0e-9,
    )
    offset_idx = next(
        i
        for i, ln in enumerate(stripped_lines)
        if ln.split() == ["offset", "Teff", "L"]
    )
    stdout_offsets: list[int] = []
    for ln in stripped_lines[offset_idx + 1 :]:
        if not ln:
            break
        toks = ln.split()
        if len(toks) == 3:
            stdout_offsets.append(int(toks[0]))
    assert stdout_offsets == [row[0] for row in ref_scan.table]

    comp = artifact_compare.compare_files(
        entry.artifact,
        repo_root,
        [out_dir],
        run_stdout=captured,
    )
    assert comp.stdout_checked is True
    assert comp.n_fields == 218
    assert comp.status in ("ok", "differs")
    assert comp.largest is not None and math.isfinite(comp.largest.excess)
    assert comp.largest.excess < 2.0e7


def test_rsp_check_2nd_crossing_falls_back_to_case_dir_when_no_work_dir(
    tmp_path: pathlib.Path,
) -> None:
    """Fallback writes to `case_dir` without `work_dir` or `output_dir`."""
    case_dir = tmp_path / "case_only"
    case_dir.mkdir()
    # Seven rows: one below red edge (finished_1st_crossing), then six
    # rising Teff points spanning negative -> positive -> negative growth.
    log_ts = [3.60, 3.66, 3.68, 3.70, 3.72, 3.74, 3.76]
    growth_by_idx = [-0.05, -0.02, 0.02, 0.04, 0.02, -0.02]
    hist_lines = [
        f"{2100 + idx} 1.0e7 {log_t:.6f} 3.70"
        for idx, log_t in enumerate(log_ts)
    ]
    (case_dir / "hist.data").write_text(
        "\n".join(hist_lines) + "\n", encoding="utf-8"
    )
    calls = 0

    def _fake_eval_lina(
        _teff: float, _lum: float, _nmodes: int
    ) -> tuple[list[float], list[float]]:
        nonlocal calls
        g = growth_by_idx[calls]
        calls += 1
        return ([15.0], [g])

    ctx = types.SimpleNamespace(
        case_dir=str(case_dir),
        x_ctrl=[100.0, 3.76, 4.5, 3.83, 2.4, 3.65, 4.5, 3.77, 2.4],
        x_integer_ctrl=[0, 1, 2, 3, 4],
        x_character_ctrl=["hist.data", "out.data"],
        rsp_mass=7.0,
        rsp_x=0.736,
        rsp_z=0.008,
        rsp_alfam=0.5,
        rsp_alfap=0.0,
        rsp_alfat=0.0,
        rsp_gammar=1.0,
        eval_lina=_fake_eval_lina,
    )
    extras = RspCheck2ndCrossingExtras(
        emit_eight_history_columns=False, write_lina_file=False
    )
    assert extras.other_rsp_build_model(ctx) == "terminate"
    assert calls == 6
    out_file = case_dir / "out.data"
    assert out_file.is_file()
    scan = artifact_compare.parse_instability_strip(
        out_file.read_text(encoding="utf-8")
    )
    assert len(scan.header) == 7
    assert [m[0] for m in scan.models] == [2101, 2102, 2103, 2104, 2105, 2106]
    assert set(scan.edges) == {"Teff_red_edge", "Teff_blue_edge"}
    assert 10.0**3.68 < scan.edges["Teff_red_edge"] < 10.0**3.70
    assert 10.0**3.74 < scan.edges["Teff_blue_edge"] < 10.0**3.76
    assert len(scan.table) >= 2
    assert scan.table[0][0] == 0
    assert math.isclose(
        scan.table[-1][1], scan.edges["Teff_red_edge"], abs_tol=1.0e-9
    )


def test_rsp_build_context_output_dir_fallback_and_explicit_history_parent(
    tmp_path: pathlib.Path,
) -> None:
    """`PyRspBuildContext` falls back to `case_dir` and uses history parent."""

    class _RspBuildDirProbe:
        def __init__(self) -> None:
            self.seen: list[tuple[str, str, str]] = []

        def other_rsp_build_model(self, ctx: Any) -> str:
            self.seen.append((ctx.case_dir, ctx.output_dir, ctx.work_dir))
            return "terminate"

    case_dir = _copy_case(
        "star", "rsp_check_2nd_crossing", tmp_path / "probe_2nd_crossing"
    )
    probe_default = _RspBuildDirProbe()
    rep_default = _rust.star_run_test_suite_case(
        str(case_dir),
        allow_unsupported_controls=True,
        extras=probe_default,
    )
    assert rep_default["total_steps"] == 0
    assert len(probe_default.seen) == 1
    def_case, def_out, def_work = probe_default.seen[0]
    assert def_out == def_case
    assert def_work == def_case

    hist_dir = tmp_path / "explicit_hist_parent"
    hist_dir.mkdir()
    hist_file = hist_dir / "history.data"
    probe_explicit = _RspBuildDirProbe()
    rep_explicit = _rust.trace_star_case(
        str(case_dir),
        str(hist_file),
        is_logs_dir=False,
        allow_unsupported_controls=True,
        extras=probe_explicit,
    )
    assert rep_explicit["total_steps"] == 0
    assert len(probe_explicit.seen) == 1
    exp_case, exp_out, exp_work = probe_explicit.seen[0]
    assert pathlib.Path(exp_out) == hist_dir
    assert exp_work == exp_out
    assert exp_out != exp_case
