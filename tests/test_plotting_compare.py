# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MESA 26.4.1 comparison figures, profile panels, and PGSTAR map."""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import pathlib
import sys
from typing import Any, Protocol, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from flexmesa import cli, plotting
from flexmesa.data import fetch
from flexmesa.plotting import cli as plot_cli
from flexmesa.plotting import compare, pgstar_coverage, stellar_plots
from flexmesa.testing import (
    case_manifest,
    case_report,
    harness_runner,
)

matplotlib.use("Agg")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "benchmarks" / "history_harness" / "reference_histories"


class _PgstarRenderer(Protocol):
    def render_pgstar_coverage_markdown(
        self, repo_root: pathlib.Path
    ) -> str: ...

    def main(self, argv: list[str] | None = None) -> int: ...


def _write_mesa_profile(
    path: pathlib.Path,
    *,
    model_number: int,
    star_age: float,
    x_shift: float = 0.0,
    include_lum: bool = True,
    mix_types: np.ndarray | None = None,
    pp_override: np.ndarray | None = None,
    include_mass: bool = True,
) -> pathlib.Path:
    hdr_cols = [
        "version_number",
        "model_number",
        "num_zones",
        "star_age",
        "math_backend",
    ]
    hdr_vals = [
        '"FlexMESA"',
        f"{model_number}",
        "6",
        f"{star_age:.16e}",
        '"strict"',
    ]
    lines = [
        " ".join(f"{i:24d}" for i in range(1, len(hdr_cols) + 1)),
        " ".join(f"{k:>24s}" for k in hdr_cols),
        " ".join(f"{v:>24s}" for v in hdr_vals),
        "",
    ]
    cols = ["zone"]
    if include_mass:
        cols.append("mass")
    cols.extend(["logT", "logRho"])
    if include_lum:
        cols.append("luminosity")
    else:
        cols.extend(["pp", "cno", "tri_alpha"])
    cols.extend(
        [
            "x_mass_fraction_H",
            "y_mass_fraction_He",
            "z_mass_fraction_metals",
            "log_D_mix",
            "mixing_type",
            "eps_nuc",
        ]
    )
    lines.append(" ".join(f"{i:24d}" for i in range(1, len(cols) + 1)))
    lines.append(" ".join(f"{c:>24s}" for c in cols))

    masses = np.linspace(1.3, 0.05, 6, dtype=np.float64)
    log_t = np.linspace(4.2, 7.2, 6, dtype=np.float64)
    log_rho = np.linspace(-6.0, 1.9, 6, dtype=np.float64)
    lum = np.linspace(2.2, 0.0, 6, dtype=np.float64)
    pp = (
        pp_override
        if pp_override is not None
        else np.array([0.0, 1.0, 10.0, 100.0, 500.0, 1500.0], dtype=np.float64)
    )
    cno = np.array([0.0, 0.0, 2.0, 20.0, 200.0, 800.0], dtype=np.float64)
    tri_a = np.zeros(6, dtype=np.float64)
    xh = np.linspace(0.70, 0.35 + x_shift, 6, dtype=np.float64)
    yhe = 0.98 - xh
    zm = np.full(6, 0.02, dtype=np.float64)
    log_dmix = np.array([12.0, 11.5, 4.0, 8.0, 13.0, 13.2], dtype=np.float64)
    mix_t = (
        mix_types
        if mix_types is not None
        else np.array([1, 2, 0, 3, 1, 1], dtype=np.int64)
    )
    eps = np.array([0.0, 1.0, 5.0, 20.0, 800.0, 2500.0], dtype=np.float64)

    for z_idx in range(6):
        row = [f"{z_idx + 1:24d}"]
        if include_mass:
            row.append(f"{masses[z_idx]:24.16e}")
        row.extend(
            [
                f"{log_t[z_idx]:24.16e}",
                f"{log_rho[z_idx]:24.16e}",
            ]
        )
        if include_lum:
            row.append(f"{lum[z_idx]:24.16e}")
        else:
            row.extend(
                [
                    f"{pp[z_idx]:24.16e}",
                    f"{cno[z_idx]:24.16e}",
                    f"{tri_a[z_idx]:24.16e}",
                ]
            )
        row.extend(
            [
                f"{xh[z_idx]:24.16e}",
                f"{yhe[z_idx]:24.16e}",
                f"{zm[z_idx]:24.16e}",
                f"{log_dmix[z_idx]:24.16e}",
                f"{int(mix_t[z_idx]):24d}",
                f"{eps[z_idx]:24.16e}",
            ]
        )
        lines.append(" ".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_history_tsv(
    path: pathlib.Path,
    rows: list[dict[str, float]],
    *,
    columns: tuple[str, ...] = (
        "model_number",
        "star_age",
        "log_L",
        "log_Teff",
        "log_center_T",
        "log_center_Rho",
        "log_dt",
        "num_zones",
    ),
) -> pathlib.Path:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(f"{row[col]:.16e}" for col in columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_resolve_case_entry_supports_qualified_and_unqualified() -> None:
    """Resolve case entries by name and reject unknown or ambiguous keys."""
    e1 = compare.resolve_case_entry("1.3M_ms_high_Z")
    assert e1.suite == "star"
    assert e1.case == "1.3M_ms_high_Z"

    e2 = compare.resolve_case_entry("binary/evolve_both_stars")
    assert e2.suite == "binary"
    assert e2.case == "evolve_both_stars"

    with pytest.raises(ValueError, match="not found"):
        compare.resolve_case_entry("nonexistent_case_name")


@pytest.mark.parametrize(
    ("suite", "case", "rel_ref", "check_inverted_x"),
    [
        ("star", "1.3M_ms_high_Z", "star/1.3M_ms_high_Z.data", True),
        ("binary", "evolve_both_stars", "binary/evolve_both_stars.data", False),
        ("astero", "example_astero", "astero/example_astero.data", True),
    ],
)
def test_plot_case_comparison_suite_layouts(
    tmp_path: pathlib.Path,
    suite: str,
    case: str,
    rel_ref: str,
    check_inverted_x: bool,
) -> None:
    """Comparison figures overplot MESA 26.4.1 histories and residuals."""
    ref_path = REF_DIR / rel_ref
    out_png = tmp_path / f"{suite}_cmp.png"
    fig, axes = plotting.plot_case_comparison(
        case,
        suite=suite,
        flexmesa_history=ref_path,
        mesa_history=ref_path,
        save_path=out_png,
        numerics_mode="strict",
    )
    try:
        assert axes.shape == (3, 2)
        assert out_png.is_file() and out_png.stat().st_size > 10_000
        if check_inverted_x:
            assert axes[0, 0].xaxis_inverted()
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        assert case in suptitle
        assert "within tol" in suptitle
    finally:
        plt.close(fig)


def test_plot_case_comparison_diverging_residuals_and_sentinel_masking(
    tmp_path: pathlib.Path,
) -> None:
    """Diverging residuals plot symlog ratios, step marker, and NaN tails."""
    base_entry = compare.resolve_case_entry("1.3M_ms_high_Z")
    custom_entry = dataclasses.replace(
        base_entry,
        final_tolerances={
            "rel_star_age": 1.0e-3,
            "log_L": 1.0e-3,
            "log_center_Rho": 2.0e-3,
        },
        track_tolerances={
            "log_L": 5.0e-3,
            "log_Teff": 2.0e-3,
        },
    )
    ages_ref = [0.0, 1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6]
    ages_run = [0.0, 1.0e6, 2.0e6, 3.0e6, 4.0e6, 4.9e6]
    run_log_l = [0.0, 0.0, 0.06, 0.10, 0.12, 0.15]
    run_log_teff = [3.77, 3.77, 3.77, 3.77, 3.77, -1.0e99]

    ref_rows = [
        {
            "model_number": float(i + 1),
            "star_age": ages_ref[i],
            "log_L": 0.0,
            "log_Teff": 3.77,
            "log_center_T": 7.1,
            "log_center_Rho": 2.0,
            "log_dt": 5.0,
            "num_zones": 100.0,
        }
        for i in range(6)
    ]
    run_rows = [
        {
            "model_number": float(i + 1),
            "star_age": ages_run[i],
            "log_L": run_log_l[i],
            "log_Teff": run_log_teff[i],
            "log_center_T": 7.1,
            "log_center_Rho": 2.005,
            "log_dt": 5.0,
            "num_zones": 100.0,
        }
        for i in range(6)
    ]
    ref_p = _write_history_tsv(tmp_path / "ref.tsv", ref_rows)
    run_p = _write_history_tsv(tmp_path / "run.tsv", run_rows)

    fig, axes = plotting.plot_case_comparison(
        "1.3M_ms_high_Z",
        suite="star",
        flexmesa_history=run_p,
        mesa_history=ref_p,
        case_entry=custom_entry,
        numerics_mode="strict",
    )
    try:
        res_ax = axes[2, 1]
        assert res_ax.get_yscale() == "symlog"
        by_label = {line.get_label(): line for line in res_ax.get_lines()}
        l_line = by_label["log_L (5e-03)"]
        l_y = np.asarray(l_line.get_ydata(), dtype=np.float64)
        assert np.nanmax(l_y) == pytest.approx(0.15 / 5.0e-3, rel=1e-2)

        teff_line = by_label["log_Teff (2e-03)"]
        teff_x = np.asarray(teff_line.get_xdata(), dtype=np.float64)
        teff_y = np.asarray(teff_line.get_ydata(), dtype=np.float64)
        tail_mask = teff_x > 4.01
        assert np.any(tail_mask)
        assert np.all(np.isnan(teff_y[tail_mask]))

        div_line = by_label["1st div: step 3"]
        div_x = np.asarray(div_line.get_xdata(), dtype=np.float64)
        assert div_x[0] == pytest.approx(2.0)

        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        assert "outside tol" in suptitle
        assert "1st div step=3" in suptitle
        title_txt = res_ax.get_title()
        assert "rel_star_age=-2.00e-02 (tol 1e-03)" in title_txt
        assert "log_center_Rho=+5.00e-03 (tol 2e-03)" in title_txt
    finally:
        plt.close(fig)


def test_plot_case_comparison_handles_single_row_nonmonotonic_and_missing(
    tmp_path: pathlib.Path,
) -> None:
    """Handle single-row/non-monotonic runs gracefully and fail on missing."""
    star_ref = REF_DIR / "star" / "1.3M_ms_high_Z.data"
    raw_lines = star_ref.read_text(encoding="utf-8").splitlines()
    one_row_hist = tmp_path / "one_row.data"
    one_row_hist.write_text("\n".join(raw_lines[:7]) + "\n", encoding="utf-8")

    fig, axes = compare.plot_case_comparison(
        "1.3M_ms_high_Z",
        suite="star",
        flexmesa_history=one_row_hist,
        mesa_history=star_ref,
    )
    try:
        assert axes.shape == (3, 2)
    finally:
        plt.close(fig)

    bad_rows = [
        {
            "model_number": 1.0,
            "star_age": 100.0,
            "log_L": 0.0,
            "log_Teff": 3.77,
            "log_center_T": 7.1,
            "log_center_Rho": 2.0,
            "log_dt": 2.0,
            "num_zones": 50.0,
        },
        {
            "model_number": 2.0,
            "star_age": 50.0,
            "log_L": 0.1,
            "log_Teff": 3.76,
            "log_center_T": 7.1,
            "log_center_Rho": 2.0,
            "log_dt": 2.0,
            "num_zones": 50.0,
        },
    ]
    nonmono = _write_history_tsv(tmp_path / "nonmono.tsv", bad_rows)
    fig_nm, axes_nm = compare.plot_case_comparison(
        "1.3M_ms_high_Z",
        suite="star",
        flexmesa_history=nonmono,
        mesa_history=star_ref,
    )
    try:
        assert axes_nm[2, 1].texts[0].get_text() == (
            "No track tolerances configured"
        )
    finally:
        plt.close(fig_nm)

    with pytest.raises(FileNotFoundError):
        compare.plot_case_comparison(
            "1.3M_ms_high_Z",
            suite="star",
            flexmesa_dir=tmp_path / "empty_dir",
        )


def test_profile_loader_panels_and_kippenhahn(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render structure panels, sentinel-safe pp/cno/3a, and Kippenhahn."""
    p1 = _write_mesa_profile(
        tmp_path / "profile1.data", model_number=1, star_age=1.0e6
    )
    p2 = _write_mesa_profile(
        tmp_path / "profile2.data",
        model_number=10,
        star_age=5.0e8,
        x_shift=-0.12,
    )
    tbl = plotting.load_profile_file(p1)
    assert tbl.num_zones == 6
    assert "x_mass_fraction_H" in tbl.columns
    assert int(tbl.header["model_number"]) == 1

    prof_png = tmp_path / "profile_panels.png"
    fig_p, axes_p = plotting.plot_profile_panels(
        p2, ref_profile_path=p1, save_path=prof_png
    )
    try:
        assert axes_p.shape == (2, 2)
        assert prof_png.is_file() and prof_png.stat().st_size > 10_000
    finally:
        plt.close(fig_p)

    # Default pp/cno/tri_alpha profile (no luminosity/logL) with -1e99 sentinel
    pp_ref = _write_mesa_profile(
        tmp_path / "pp_ref.data",
        model_number=1,
        star_age=1.0e6,
        include_lum=False,
    )
    pp_run = _write_mesa_profile(
        tmp_path / "pp_run.data",
        model_number=2,
        star_age=2.0e6,
        include_lum=False,
        pp_override=np.array(
            [0.0, 5.0, -1.0e99, 100.0, 500.0, 1500.0], dtype=np.float64
        ),
    )
    fig_burn, axes_burn = plotting.plot_profile_panels(
        pp_run, ref_profile_path=pp_ref
    )
    try:
        burn_ax = axes_burn[1, 0]
        dashed = [
            ln for ln in burn_ax.get_lines() if ln.get_linestyle() == "--"
        ]
        assert len(dashed) >= 3
        for ln in burn_ax.get_lines():
            yd = np.asarray(ln.get_ydata(), dtype=np.float64)
            assert float(np.nanmax(np.abs(yd))) < 50.0
    finally:
        plt.close(fig_burn)

    # Categorical nearest-neighbor Kippenhahn across [0, 3, 0] (no 1.0 cells)
    mix_jump = np.array([0, 0, 3, 3, 0, 0], dtype=np.int64)
    k1 = _write_mesa_profile(
        tmp_path / "k1.data", model_number=1, star_age=1.0e6, mix_types=mix_jump
    )
    k2 = _write_mesa_profile(
        tmp_path / "k2.data", model_number=2, star_age=5.0e6, mix_types=mix_jump
    )
    captured_mix: list[np.ndarray] = []
    orig_kipp = stellar_plots.plot_kippenhahn

    def _spy_kippenhahn(
        age_myr: Any,
        mass_grid_msun: Any,
        mix_type_grid: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        captured_mix.append(np.asarray(mix_type_grid, dtype=np.float64).copy())
        return orig_kipp(
            age_myr, mass_grid_msun, mix_type_grid, *args, **kwargs
        )

    monkeypatch.setattr(stellar_plots, "plot_kippenhahn", _spy_kippenhahn)
    kipp_png = tmp_path / "kipp.png"
    fig_k, ax_k = plotting.plot_profile_kippenhahn(
        [k1, k2], mass_grid_points=32, save_path=kipp_png
    )
    try:
        assert ax_k is not None
        assert kipp_png.is_file() and kipp_png.stat().st_size > 10_000
        assert len(captured_mix) == 1
        assert set(np.unique(captured_mix[0])).issubset({0.0, 3.0})
    finally:
        plt.close(fig_k)

    bad_k = _write_mesa_profile(
        tmp_path / "no_mass.data",
        model_number=1,
        star_age=1.0e6,
        include_mass=False,
    )
    with pytest.raises(ValueError, match="mass"):
        plotting.plot_profile_kippenhahn([bad_k, k2])

    with pytest.raises(ValueError, match="at least 2"):
        plotting.plot_profile_kippenhahn([p1])


def test_flexmesa_plot_cli_explicit_dir_auto_trace_and_fast_fail(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI handles stored logs, profile, auto-run, and fast reference checks."""
    logs_dir = tmp_path / "run_out" / "LOGS"
    logs_dir.mkdir(parents=True)
    star_ref = REF_DIR / "star" / "1.3M_ms_high_Z.data"
    (logs_dir / "history.data").write_text(
        star_ref.read_text(encoding="utf-8"), encoding="utf-8"
    )
    prof = _write_mesa_profile(
        logs_dir / "profile1.data", model_number=1, star_age=1.0e6
    )

    out_cmp = tmp_path / "cli_cmp.png"
    out_prof = tmp_path / "cli_prof.png"
    rc = cli.main(
        [
            "plot",
            "compare",
            "1.3M_ms_high_Z",
            "--flexmesa-dir",
            str(tmp_path / "run_out"),
            "--profile",
            str(prof),
            "--out",
            str(out_cmp),
            "--profile-out",
            str(out_prof),
        ]
    )
    assert rc == 0
    assert out_cmp.is_file() and out_prof.is_file()

    out_prof2 = tmp_path / "cli_prof_only.png"
    rc2 = plot_cli.main(["profile", str(prof), "--out", str(out_prof2)])
    assert rc2 == 0 and out_prof2.is_file()

    # No-reference case rejects before launching tracer
    def _forbid_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("must not invoke tracer without reference")

    monkeypatch.setattr(harness_runner, "run_case_with_diagnostic", _forbid_run)
    rc_noref = cli.main(["plot", "compare", "1.4M_ms_op_mono"])
    assert rc_noref == 2
    assert "no stored reference_path" in capsys.readouterr().err
    monkeypatch.undo()

    # No-manual-step auto-trace execution against bundled MESA test inputs
    auto_cmp = tmp_path / "auto_cmp.png"
    rc_auto = cli.main(
        [
            "plot",
            "compare",
            "1.3M_ms_high_Z",
            "--max-steps",
            "2",
            "--allow-unsupported-controls",
            "--out",
            str(auto_cmp),
        ]
    )
    assert rc_auto == 0
    assert auto_cmp.is_file() and auto_cmp.stat().st_size > 10_000


def test_case_report_optional_figure_links() -> None:
    """Case table renders comparison links when given and is unchanged else."""
    manifest = case_manifest.load_case_manifest()
    gaps = case_manifest.load_known_gaps()
    entry = compare.resolve_case_entry("1.3M_ms_high_Z")
    outcome = harness_runner.TraceRunOutcome(
        suite="star",
        case="1.3M_ms_high_Z",
        status="ok",
        termination_code="log_L_upper_limit",
        steps=10,
        retries=0,
        wall_time_sec=0.1,
        peak_rss_gb=0.1,
        history_path=REF_DIR / "star" / "1.3M_ms_high_Z.data",
    )
    rec = harness_runner.evaluate_case_record(
        entry, gaps, outcome, root_dir=REPO_ROOT
    )
    md_plain = case_report.render_case_table_markdown(
        manifest, [rec], known_gaps=gaps
    )
    assert "| Figure |" not in md_plain

    md_linked = case_report.render_case_table_markdown(
        manifest,
        [rec],
        known_gaps=gaps,
        figure_links={("star", "1.3M_ms_high_Z"): "figures/star/1.3M.png"},
    )
    assert "| Figure |" in md_linked
    assert "[compare](figures/star/1.3M.png)" in md_linked


def test_harness_run_figures_dir_renders_and_survives_zero_row_history(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--figures-dir links valid PNGs and continues past 0-row histories."""
    figs_dir = tmp_path / "figs"
    md_out = tmp_path / "report.md"
    rc = cli.main(
        [
            "harness",
            "run",
            "--cases",
            "1.3M_ms_high_Z",
            "--max-steps",
            "2",
            "--allow-unsupported-controls",
            "--figures-dir",
            str(figs_dir),
            "--markdown-out",
            str(md_out),
        ]
    )
    assert rc == 0
    expected_png = figs_dir / "star" / "1.3M_ms_high_Z.png"
    assert expected_png.is_file() and expected_png.stat().st_size > 10_000
    md_text = md_out.read_text(encoding="utf-8")
    assert "[compare](figs/star/1.3M_ms_high_Z.png)" in md_text

    # Zero-row header-only history must skip figure with warning, not abort
    def _zero_row_run(
        suite: str,
        _case_dir: pathlib.Path,
        out_dir: pathlib.Path,
        **_kwargs: Any,
    ) -> harness_runner.TraceRunOutcome:
        logs = out_dir / "LOGS"
        logs.mkdir(parents=True, exist_ok=True)
        hist = logs / "history.data"
        star_ref = REF_DIR / "star" / "1.3M_ms_high_Z.data"
        header_only = star_ref.read_text(encoding="utf-8").splitlines()[:6]
        hist.write_text("\n".join(header_only) + "\n", encoding="utf-8")
        return harness_runner.TraceRunOutcome(
            suite=suite,
            case="1.3M_ms_high_Z",
            status="error",
            termination_code="",
            steps=0,
            retries=0,
            wall_time_sec=0.05,
            peak_rss_gb=0.05,
            history_path=hist,
            error_message="zero steps completed",
        )

    monkeypatch.setattr(
        harness_runner, "run_case_with_diagnostic", _zero_row_run
    )
    md_zero = tmp_path / "zero_report.md"
    json_zero = tmp_path / "zero_report.json"
    rc_z = cli.main(
        [
            "harness",
            "run",
            "--cases",
            "1.3M_ms_high_Z",
            "--figures-dir",
            str(tmp_path / "zero_figs"),
            "--markdown-out",
            str(md_zero),
            "--json-out",
            str(json_zero),
        ]
    )
    assert rc_z == 0
    assert md_zero.is_file() and json_zero.is_file()
    assert "skipping figure" in capsys.readouterr().err


def test_pgstar_coverage_and_rendered_doc_match(
    tmp_path: pathlib.Path,
) -> None:
    """All MESA 26.4.1 PGSTAR/PGBINARY windows catalog and check cleanly."""
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    rep = pgstar_coverage.build_pgstar_coverage_report(mesa_dir)
    assert rep.total_windows == 123
    assert len(rep.panels) == 42
    assert rep.source_counts["pgstar.defaults"] == 2900
    assert rep.source_counts["pgbinary.defaults"] == 1329

    scripts_docs = str(REPO_ROOT / "scripts" / "docs")
    if scripts_docs not in sys.path:
        sys.path.insert(0, scripts_docs)
    spec = importlib.util.spec_from_file_location(
        "render_pgstar_coverage",
        REPO_ROOT / "scripts" / "docs" / "render_pgstar_coverage.py",
    )
    assert spec is not None and spec.loader is not None
    mod_raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod_raw)
    mod = cast(_PgstarRenderer, mod_raw)

    rendered = mod.render_pgstar_coverage_markdown(REPO_ROOT)
    committed = (
        REPO_ROOT / "docs" / "benchmarks" / "pgstar-coverage.md"
    ).read_text(encoding="utf-8")
    assert rendered == committed
    assert (
        mod.main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--output",
                str(tmp_path / "out.md"),
                "--check",
            ]
        )
        == 1
    )


def test_plotting_modules_contain_no_large_hardcoded_data_arrays() -> None:
    """Ensure flexmesa.plotting modules hold zero hardcoded trajectory lists."""
    plot_dir = REPO_ROOT / "src" / "flexmesa" / "plotting"
    for py_file in sorted(plot_dir.glob("*.py")):
        tree = ast.parse(
            py_file.read_text(encoding="utf-8"), filename=str(py_file)
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 6:
                num_elts = sum(
                    1
                    for el in node.elts
                    if isinstance(el, ast.Constant)
                    and isinstance(el.value, (int, float))
                )
                assert num_elts < 6, (
                    f"{py_file.name}: unexpected numeric array literal of "
                    f"length {num_elts}"
                )
