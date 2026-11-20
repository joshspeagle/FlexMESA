# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for `#173` Increment 3 `pgstar`, `pgbinary`, and live plotting."""

from __future__ import annotations

import math
import pathlib
import threading
import time
import warnings

import matplotlib
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import numpy as np
import pytest

from flexmesa import chem, cli, constants, plotting, star
from flexmesa.data import fetch
from flexmesa.plotting import _roche_orbit
from flexmesa.testing import history_compare

matplotlib.use("Agg")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REF_DIR = REPO_ROOT / "benchmarks" / "history_harness" / "reference_histories"


def _write_mesa_log(
    path: pathlib.Path,
    header: dict[str, str],
    columns: dict[str, np.ndarray],
) -> pathlib.Path:
    """Write a 6-line MESA ASCII history or profile table for testing."""
    h_keys = list(header.keys())
    h_vals = [header[k] for k in h_keys]
    c_keys = list(columns.keys())
    n_rows = int(next(iter(columns.values())).size)

    lines = [
        " ".join(f"{i:24d}" for i in range(1, len(h_keys) + 1)),
        " ".join(f"{k:>24s}" for k in h_keys),
        " ".join(f"{v:>24s}" for v in h_vals),
        "",
        " ".join(f"{i:24d}" for i in range(1, len(c_keys) + 1)),
        " ".join(f"{c:>24s}" for c in c_keys),
    ]
    for r_idx in range(n_rows):
        row_tokens = []
        for col in c_keys:
            val = float(columns[col][r_idx])
            if col in ("model_number", "zone", "mixing_type") or col.startswith(
                (
                    "mix_type_",
                    "rti_type_",
                    "donor_index",
                    "point_mass_index",
                )
            ):
                row_tokens.append(f"{round(val):24d}")
            else:
                row_tokens.append(f"{val:24.16e}")
        lines.append(" ".join(row_tokens))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_full_profile(
    path: pathlib.Path,
    *,
    model_number: int = 42,
    n14_scale: float = 1.0,
    include_dynamo: bool = True,
    include_rotation_details: bool = True,
    linear_rot_names: bool = False,
) -> pathlib.Path:
    """Build a synthetic profileN.data with rotation, dynamo, and isotopes."""
    nz = 12
    mass = np.linspace(1.60, 0.05, nz, dtype=np.float64)
    q = mass / 1.60
    log_t = np.linspace(3.85, 7.38, nz, dtype=np.float64)
    cols: dict[str, np.ndarray] = {
        "zone": np.arange(1, nz + 1, dtype=np.float64),
        "mass": mass,
        "q": q,
        "radius": np.linspace(1.75, 0.06, nz, dtype=np.float64),
        "logR": np.log10(np.linspace(1.75, 0.06, nz, dtype=np.float64)),
        "logP": np.linspace(5.2, 17.3, nz, dtype=np.float64),
        "logT": log_t,
        "temperature": np.power(10.0, log_t),
        "logRho": np.linspace(-6.2, 1.95, nz, dtype=np.float64),
        "luminosity": np.linspace(4.5, 0.08, nz, dtype=np.float64),
        "entropy": np.linspace(21.5, 11.2, nz, dtype=np.float64),
        "opacity": np.linspace(0.45, 1.65, nz, dtype=np.float64),
        "eps_nuc": np.linspace(0.0, 950.0, nz, dtype=np.float64),
        "non_nuc_neu": np.linspace(0.0, 2.5, nz, dtype=np.float64),
        "pp": np.linspace(0.0, 250.0, nz, dtype=np.float64),
        "cno": np.linspace(0.0, 700.0, nz, dtype=np.float64),
        "h1": np.linspace(0.70, 0.24, nz, dtype=np.float64)
        - 1.1e-3 * (n14_scale - 1.0),
        "he3": np.linspace(4.0e-5, 1.5e-4, nz, dtype=np.float64),
        "he4": np.linspace(0.28, 0.74, nz, dtype=np.float64),
        "c12": np.linspace(3.0e-3, 5.0e-4, nz, dtype=np.float64),
        "n14": np.full(nz, 1.1e-3 * n14_scale, dtype=np.float64),
        "o16": np.linspace(9.0e-3, 4.5e-3, nz, dtype=np.float64),
        "ne20": np.full(nz, 1.8e-3, dtype=np.float64),
        "mg24": np.full(nz, 6.5e-4, dtype=np.float64),
        "fe56": np.full(nz, 1.2e-3, dtype=np.float64),
        "x_mass_fraction_H": np.linspace(0.70, 0.24, nz, dtype=np.float64),
        "y_mass_fraction_He": np.linspace(0.28, 0.74, nz, dtype=np.float64),
        "z_mass_fraction_metals": np.full(nz, 0.02, dtype=np.float64),
        "log_D_mix": np.linspace(12.8, 14.1, nz, dtype=np.float64),
        "log_D_conv": np.where(mass < 0.40, 14.1, -99.0),
        "log_D_ovr": np.where((mass >= 0.40) & (mass < 0.65), 9.2, -99.0),
        "log_D_semi": np.where((mass >= 0.65) & (mass < 0.90), 6.8, -99.0),
        "mixing_type": np.where(
            mass < 0.40, 1.0, np.where(mass < 0.65, 2.0, 0.0)
        ),
        "alpha_RTI": np.where((mass > 0.70) & (mass < 1.10), 0.15, 0.0),
    }
    if include_rotation_details:
        rot_vals = {
            "ST": np.linspace(5.0, 9.0, nz, dtype=np.float64),
            "visc": np.linspace(4.2, 7.6, nz, dtype=np.float64),
            "DSI": np.linspace(3.4, 6.8, nz, dtype=np.float64),
            "SH": np.linspace(2.2, 5.4, nz, dtype=np.float64),
            "SSI": np.linspace(1.6, 4.7, nz, dtype=np.float64),
            "ES": np.linspace(4.8, 8.2, nz, dtype=np.float64),
            "GSF": np.linspace(3.0, 6.3, nz, dtype=np.float64),
        }
        for key, arr in rot_vals.items():
            if linear_rot_names:
                cols[f"D_{key}"] = np.power(10.0, arr)
            else:
                cols[f"am_log_D_{key}"] = arr
    if include_dynamo:
        cols["dynamo_log_B_phi"] = np.linspace(3.1, 6.4, nz, dtype=np.float64)
        cols["dynamo_log_B_r"] = np.linspace(0.6, 3.9, nz, dtype=np.float64)
        cols["log_omega"] = np.linspace(-5.4, -4.1, nz, dtype=np.float64)
        cols["log_j_rot"] = np.linspace(17.4, 14.6, nz, dtype=np.float64)

    hdr = {
        "version_number": '"FlexMESA"',
        "compiler": '"rustc"',
        "model_number": str(model_number),
        "num_zones": str(nz),
        "star_age": "8.5000000000000000e+08",
        "star_mass": "1.6000000000000000e+00",
        "initial_z": "2.0000000000000000e-02",
        "initial_y": "2.8000000000000000e-01",
        "rsun": "6.9570000000000000e+10",
    }
    return _write_mesa_log(path, hdr, cols)


def _make_full_history(
    path: pathlib.Path, *, n_rows: int = 10, include_rti: bool = True
) -> pathlib.Path:
    """Build a synthetic history.data with tracks, summary, RTI, and CMD."""
    steps = np.arange(1, n_rows + 1, dtype=np.float64)
    cols: dict[str, np.ndarray] = {
        "model_number": steps,
        "star_age": np.linspace(1.0e6, 1.5e9, n_rows, dtype=np.float64),
        "log_dt": np.linspace(6.0, 7.6, n_rows, dtype=np.float64),
        "star_mass": np.linspace(1.60, 1.57, n_rows, dtype=np.float64),
        "log_abs_mdot": np.full(n_rows, -12.8, dtype=np.float64),
        "log_L": np.linspace(0.55, 1.25, n_rows, dtype=np.float64),
        "log_Teff": np.linspace(3.84, 3.70, n_rows, dtype=np.float64),
        "log_R": np.linspace(0.14, 0.70, n_rows, dtype=np.float64),
        "log_g": np.linspace(4.28, 3.15, n_rows, dtype=np.float64),
        "log_center_T": np.linspace(7.28, 7.62, n_rows, dtype=np.float64),
        "log_center_Rho": np.linspace(1.80, 3.55, n_rows, dtype=np.float64),
        "log_center_P": np.linspace(17.1, 18.2, n_rows, dtype=np.float64),
        "log_LH": np.linspace(0.54, 1.24, n_rows, dtype=np.float64),
        "log_LHe": np.full(n_rows, -9.5, dtype=np.float64),
        "log_Lneu": np.linspace(-0.9, 0.0, n_rows, dtype=np.float64),
        "he_core_mass": np.linspace(0.0, 0.28, n_rows, dtype=np.float64),
        "co_core_mass": np.zeros(n_rows, dtype=np.float64),
        "center_h1": np.linspace(0.70, 0.01, n_rows, dtype=np.float64),
        "center_he4": np.linspace(0.28, 0.97, n_rows, dtype=np.float64),
        "center_c12": np.linspace(3.0e-3, 3.0e-4, n_rows, dtype=np.float64),
        "center_n14": np.linspace(1.0e-3, 1.0e-2, n_rows, dtype=np.float64),
        "center_o16": np.linspace(9.0e-3, 5.0e-3, n_rows, dtype=np.float64),
        "num_zones": np.full(n_rows, 512.0, dtype=np.float64),
        "num_retries": np.zeros(n_rows, dtype=np.float64),
        "delta_nu": np.linspace(78.0, 14.0, n_rows, dtype=np.float64),
        "delta_Pg": np.linspace(115.0, 64.0, n_rows, dtype=np.float64),
        "mix_type_1": np.ones(n_rows, dtype=np.float64),
        "mix_qtop_1": np.linspace(0.18, 0.06, n_rows, dtype=np.float64),
        "burn_type_1": np.full(n_rows, 4.0, dtype=np.float64),
        "burn_qtop_1": np.linspace(0.20, 0.25, n_rows, dtype=np.float64),
        "abs_mag_B": np.linspace(4.1, 2.7, n_rows, dtype=np.float64),
        "abs_mag_V": np.linspace(3.6, 1.6, n_rows, dtype=np.float64),
        "abs_mag_I": np.linspace(3.0, 0.5, n_rows, dtype=np.float64),
    }
    if include_rti:
        cols["rti_type_1"] = np.where(steps >= 4, 1.0, 0.0)
        cols["rti_qtop_1"] = np.linspace(0.12, 0.44, n_rows, dtype=np.float64)
    hdr = {
        "version_number": '"FlexMESA"',
        "compiler": '"rustc"',
        "msun": f"{constants.MSUN:.16e}",
    }
    return _write_mesa_log(path, hdr, cols)


def test_mixing_profile_draws_all_seven_rotational_curves_and_aggregate(
    tmp_path: pathlib.Path,
) -> None:
    """Mixing draws all 7 rotational curves and supports aggregate sum mode."""
    plt.close("all")
    prof_am = _make_full_profile(tmp_path / "prof_am.data")
    prof_lin = _make_full_profile(
        tmp_path / "prof_lin.data", linear_rot_names=True
    )

    fig, ax = plotting.plot_mixing_profile(prof_am, ref_profile_path=prof_am)
    try:
        labels = {line.get_label() for line in ax.get_lines()}
        for sym in (
            r"$D_{\mathrm{ST}}$",
            r"$D_{\mathrm{visc}}$",
            r"$D_{\mathrm{DSI}}$",
            r"$D_{\mathrm{SH}}$",
            r"$D_{\mathrm{SSI}}$",
            r"$D_{\mathrm{ES}}$",
            r"$D_{\mathrm{GSF}}$",
        ):
            assert sym in labels, f"missing rotational curve {sym}"
    finally:
        plt.close(fig)

    fig_lin, ax_lin = plotting.plot_mixing_profile(prof_lin)
    try:
        labels_lin = {line.get_label() for line in ax_lin.get_lines()}
        assert r"$D_{\mathrm{ST}}$" in labels_lin
        assert r"$D_{\mathrm{GSF}}$" in labels_lin
    finally:
        plt.close(fig_lin)

    fig_agg, ax_agg = plotting.plot_mixing_profile(
        prof_am, show_rotation_details=False
    )
    try:
        labels_agg = {line.get_label() for line in ax_agg.get_lines()}
        assert r"$D_{\mathrm{rot}}$" in labels_agg
        assert r"$D_{\mathrm{ST}}$" not in labels_agg
    finally:
        plt.close(fig_agg)

    assert not plt.get_fignums()


def test_pgbinary_orbit_recovers_keplerian_eccentricity_and_fails_loudly(
    tmp_path: pathlib.Path,
) -> None:
    """Orbit recovers Keplerian e from J_orb and rejects missing inputs."""
    plt.close("all")
    # 1. All stored MESA 26.4.1 binary histories render cleanly via J_orb.
    for ref_bin in sorted((REF_DIR / "binary").glob("*.data")):
        fig, _ = plotting.plot_pgbinary_orbit(ref_bin)
        plt.close(fig)

    # 2. Exact analytical Keplerian inversion for eccentric binary (e = 0.45)
    # without an eccentricity column.
    m1_msun, m2_msun, sep_rsun, e_true = 2.4, 1.3, 18.0, 0.45
    m1_g = m1_msun * constants.MSUN
    m2_g = m2_msun * constants.MSUN
    a_cm = sep_rsun * constants.RSUN
    j_orb = (
        m1_g
        * m2_g
        * math.sqrt(
            constants.STANDARD_CGRAV
            * a_cm
            * (1.0 - e_true * e_true)
            / (m1_g + m2_g)
        )
    )
    ecc_hist = _write_mesa_log(
        tmp_path / "ecc_jorb_binary.data",
        {"version_number": '"FlexMESA"'},
        {
            "model_number": np.array([1.0, 2.0]),
            "age": np.array([1.0e5, 2.0e5]),
            "period_days": np.array([3.5, 3.5]),
            "binary_separation": np.array([sep_rsun, sep_rsun]),
            "star_1_mass": np.array([m1_msun, m1_msun]),
            "star_2_mass": np.array([m2_msun, m2_msun]),
            "J_orb": np.array([j_orb, j_orb]),
            "rl_1": np.array([7.5, 7.5]),
            "rl_2": np.array([5.8, 5.8]),
            "rl_relative_overflow_1": np.array([-0.15, -0.15]),
            "rl_relative_overflow_2": np.array([-0.30, -0.30]),
            "lg_mtransfer_rate": np.array([-20.0, -20.0]),
        },
    )
    # Direct unit verification of recovered eccentricity to < 1e-12.
    ecc_tbl = history_compare.load_history_file(ecc_hist)
    e_rec = _roche_orbit._resolve_eccentricity(
        ecc_tbl,
        1,
        m1_msun=m1_msun,
        m2_msun=m2_msun,
        sep_rsun=sep_rsun,
    )
    assert e_rec == pytest.approx(e_true, abs=1.0e-12)
    fig_e, ax_e = plotting.plot_pgbinary_orbit(ecc_hist)
    try:
        assert len(ax_e.get_lines()) >= 3
    finally:
        plt.close(fig_e)

    # 3. Non-physical J_orb exceeding circular bound raises ValueError loudly.
    bad_jorb_hist = _write_mesa_log(
        tmp_path / "bad_jorb_binary.data",
        {"version_number": '"FlexMESA"'},
        {
            "model_number": np.array([1.0, 2.0]),
            "age": np.array([1.0e5, 2.0e5]),
            "period_days": np.array([3.5, 3.5]),
            "binary_separation": np.array([sep_rsun, sep_rsun]),
            "star_1_mass": np.array([m1_msun, m1_msun]),
            "star_2_mass": np.array([m2_msun, m2_msun]),
            "J_orb": np.array([j_orb * 5.0, j_orb * 5.0]),
            "rl_1": np.array([7.5, 7.5]),
            "rl_2": np.array([5.8, 5.8]),
        },
    )
    with pytest.raises(ValueError, match="non-physical"):
        plotting.plot_pgbinary_orbit(bad_jorb_hist)
    assert not plt.get_fignums()

    # 4. Missing both eccentricity and J_orb raises MissingOrbitColumnsError
    # without leaking any Matplotlib figure.
    no_ecc_no_jorb = _write_mesa_log(
        tmp_path / "no_ecc_no_jorb.data",
        {"version_number": '"FlexMESA"'},
        {
            "model_number": np.array([1.0, 2.0]),
            "age": np.array([1.0e5, 2.0e5]),
            "period_days": np.array([3.5, 3.5]),
            "binary_separation": np.array([sep_rsun, sep_rsun]),
            "star_1_mass": np.array([m1_msun, m1_msun]),
            "star_2_mass": np.array([m2_msun, m2_msun]),
            "rl_1": np.array([7.5, 7.5]),
            "rl_2": np.array([5.8, 5.8]),
            "rl_relative_overflow_1": np.array([-0.15, -0.15]),
            "rl_relative_overflow_2": np.array([-0.30, -0.30]),
            "lg_mtransfer_rate": np.array([-20.0, -20.0]),
        },
    )
    with pytest.raises(_roche_orbit.MissingOrbitColumnsError):
        plotting.plot_pgbinary_orbit(no_ecc_no_jorb)
    assert not plt.get_fignums()

    fig_p, axes_p = plotting.plot_pgbinary_panels(binary_history=no_ecc_no_jorb)
    try:
        orbit_texts = " ".join(t.get_text() for t in axes_p[0, 0].texts)
        assert "Orbit not drawn" in orbit_texts
    finally:
        plt.close(fig_p)
    assert not plt.get_fignums()


def test_text_summary_renders_defaults_and_inlist_2d_layouts(
    tmp_path: pathlib.Path,
) -> None:
    """Text_Summary renders defaults, 2D inlists, and raises on empty match."""
    plt.close("all")
    ref_star = REF_DIR / "star" / "1.3M_ms_high_Z.data"
    ref_bin = REF_DIR / "binary" / "evolve_both_stars.data"

    fig_s, ax_s = plotting.plot_text_summary(ref_star)
    try:
        rendered = " ".join(t.get_text() for t in ax_s.texts)
        assert "model_number" in rendered
        assert "star_mass" in rendered
    finally:
        plt.close(fig_s)

    fig_b, ax_b = plotting.plot_pgbinary_text_summary(ref_bin)
    try:
        rendered_b = " ".join(t.get_text() for t in ax_b.texts)
        assert "period_days" in rendered_b
        assert "rl_relative_overflow_1" in rendered_b
    finally:
        plt.close(fig_b)

    inlist_ts = tmp_path / "inlist_ts"
    inlist_ts.write_text(
        "&pgstar\n"
        "  Text_Summary2_title = 'Custom Telemetry'\n"
        "  Text_Summary2_num_rows = 2\n"
        "  Text_Summary2_num_cols = 2\n"
        "  Text_Summary2_name(:,:) = ''\n"
        "  Text_Summary2_name(1,1) = 'model_number'\n"
        "  Text_Summary2_name(1,2) = 'star_mass'\n"
        "  Text_Summary2_name(2,1) = 'log_cntr_T'\n"
        "  Text_Summary2_name(2,2) = 'center h1'\n"
        "/\n",
        encoding="utf-8",
    )
    spec = plotting.parse_inlist_text_summary_spec(inlist_ts, summary_index=2)
    assert (spec.num_rows, spec.num_cols) == (2, 2)
    assert spec.title == "Custom Telemetry"
    fig_c, ax_c = plotting.plot_text_summary(
        ref_star, inlist_path=inlist_ts, summary_index=2
    )
    try:
        assert ax_c.get_title() == "Custom Telemetry"
        texts = " ".join(t.get_text() for t in ax_c.texts)
        assert "log_cntr_T" in texts
    finally:
        plt.close(fig_c)

    with pytest.raises(KeyError, match="none of requested columns"):
        plotting.plot_text_summary(ref_star, layout=(("nonexistent_col",),))
    assert not plt.get_fignums()


def test_upstream_default_grids_and_full_pgbinary_star_embedding(
    tmp_path: pathlib.Path,
) -> None:
    """Default Grid1..4 (star & binary) and arbitrary Star1/Star2 cells work."""
    plt.close("all")
    hist = _make_full_history(tmp_path / "history.data")
    prof = _make_full_profile(tmp_path / "profile1.data")
    ref_bin = REF_DIR / "binary" / "evolve_both_stars.data"

    # 1. Built-in default layout fallback when only Grid<N>_win_flag is set.
    inl_star = tmp_path / "inlist_pgstar_defaults"
    inl_star.write_text(
        "&pgstar\n"
        "  Grid1_win_flag = .true.\n"
        "  Grid2_win_flag = .true.\n"
        "  Grid3_win_flag = .true.\n"
        "  Grid4_win_flag = .true.\n"
        "/\n",
        encoding="utf-8",
    )
    for g_idx, expected_min_axes in ((1, 4), (2, 5), (3, 3), (4, 6)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig_g, axes_g = plotting.plot_pgstar_grid(
                inlist_path=inl_star,
                grid_index=g_idx,
                history_path=hist,
                profile_path=prof,
            )
            fig_g.canvas.draw()
            plt.close(fig_g)
        assert len(axes_g) >= expected_min_axes
        cl_warns = [w for w in caught if "constrained_layout" in str(w.message)]
        assert not cl_warns, f"Grid{g_idx} layout warning: {cl_warns}"

    # 2. Direct parsing of unmodified upstream MESA 26.4.1 *.defaults files.
    mesa_dir = fetch.resolve_mesa_test_inputs_dir()
    pgstar_def = mesa_dir / "star" / "defaults" / "pgstar.defaults"
    pgbin_def = mesa_dir / "binary" / "defaults" / "pgbinary.defaults"
    assert pgstar_def.is_file() and pgbin_def.is_file()
    fig_u1, axes_u1 = plotting.plot_pgstar_grid(
        inlist_path=pgstar_def,
        grid_index=1,
        history_path=hist,
        profile_path=prof,
    )
    try:
        assert len(axes_u1) == 4
    finally:
        plt.close(fig_u1)
    fig_ub1, axes_ub1 = plotting.plot_pgstar_grid(
        inlist_path=pgbin_def,
        grid_index=1,
        binary_history_path=ref_bin,
    )
    try:
        assert len(axes_ub1) == 1
    finally:
        plt.close(fig_ub1)

    # 3. Star1/Star2 embedding arbitrary pgstar panels & nested Grid1 +
    # point_mass_index == 2 banner.
    fig_s, axes_s = plotting.plot_pgstar_grid(
        panels=[
            "Orbit",
            "Star1:Dynamo",
            "Star2:Network",
            "Star1:Grid1",
            "Star2:Profile_Panels1",
            "Text_Summary1",
        ],
        num_rows=2,
        num_cols=3,
        binary_history_path=ref_bin,
        star1_history_path=hist,
        star1_profile_path=prof,
        star2_history_path=hist,
        star2_profile_path=prof,
    )
    try:
        # Star1:Grid1 expands into 4 axes and Star2:Profile_Panels1 into 2.
        assert len(axes_s) == 1 + 1 + 1 + 4 + 2 + 1
    finally:
        plt.close(fig_s)

    pm_bin = _write_mesa_log(
        tmp_path / "pm_binary.data",
        {"version_number": '"FlexMESA"'},
        {
            "model_number": np.array([1.0, 2.0]),
            "age": np.array([1.0e5, 2.0e5]),
            "star_1_mass": np.array([1.6, 1.5]),
            "star_2_mass": np.array([1.4, 1.4]),
            "point_mass_index": np.array([2.0, 2.0]),
            "period_days": np.array([1.2, 1.1]),
            "rl_relative_overflow_1": np.array([-0.05, 0.01]),
        },
    )
    fig_pm, ax_pm = plotting.plot_pgbinary_star(
        2,
        binary_history=pm_bin,
        star_history=hist,
    )
    try:
        assert "Star 2 not simulated" in " ".join(
            t.get_text() for t in ax_pm[0].texts
        )
    finally:
        plt.close(fig_pm)
    assert not plt.get_fignums()


def test_dynamo_profile_and_dpg_dnu_render_and_fail_loudly(
    tmp_path: pathlib.Path,
) -> None:
    """Dynamo and dPg_dnu render cleanly and raise KeyError when absent."""
    plt.close("all")
    prof = _make_full_profile(tmp_path / "prof_dyn.data")
    prof_no_b = _make_full_profile(
        tmp_path / "prof_no_dyn.data", include_dynamo=False
    )
    hist = _make_full_history(tmp_path / "hist.data")

    fig_d, ax_d = plotting.plot_dynamo_profile(prof, ref_profile_path=prof)
    try:
        assert r"$\log B_\phi$" in {ln.get_label() for ln in ax_d.get_lines()}
        # Twin axis created for log_omega / log_j_rot.
        assert len(fig_d.axes) == 2
    finally:
        plt.close(fig_d)

    with pytest.raises(KeyError, match="Dynamo panel requires magnetic"):
        plotting.plot_dynamo_profile(prof_no_b)
    assert not plt.get_fignums()

    fig_p, ax_p = plotting.plot_dpg_dnu(
        hist, ref_history_path=hist, xmin=5.0, xmax=90.0, ymin=50.0, ymax=130.0
    )
    try:
        assert ax_p.get_xlim() == pytest.approx((5.0, 90.0))
        assert ax_p.get_ylim() == pytest.approx((50.0, 130.0))
    finally:
        plt.close(fig_p)

    # Stored standard 1.3M_ms_high_Z.data lacks delta_Pg -> must raise KeyError.
    with pytest.raises(KeyError, match="delta_"):
        plotting.plot_dpg_dnu(REF_DIR / "star" / "1.3M_ms_high_Z.data")
    assert not plt.get_fignums()


def test_rti_history_supports_rti_mix_type_6_and_profiles(
    tmp_path: pathlib.Path,
) -> None:
    """rti renders from rti_type_*, mix_type_*==6, and profile alpha_RTI."""
    plt.close("all")
    hist_rti = _make_full_history(tmp_path / "hist_rti.data", include_rti=True)
    fig1, ax1 = plotting.plot_rti_history(
        hist_rti,
        ref_history_path=hist_rti,
        xaxis="star_age",
        age_units="Myr",
        log_xaxis=True,
        time_from_present=True,
    )
    try:
        assert ax1.xaxis_inverted()
        assert len(ax1.collections) >= 1
    finally:
        plt.close(fig1)

    # History without rti_type_* falls back to mix_type_* == 6.
    hist_mix = _write_mesa_log(
        tmp_path / "hist_mix6.data",
        {"version_number": '"FlexMESA"'},
        {
            "model_number": np.array([1.0, 2.0, 3.0]),
            "star_age": np.array([1.0e2, 2.0e2, 3.0e2]),
            "star_mass": np.array([15.0, 15.0, 15.0]),
            "mix_type_1": np.array([0.0, 6.0, 6.0]),
            "mix_qtop_1": np.array([0.2, 0.5, 0.6]),
        },
    )
    fig2, ax2 = plotting.plot_rti_history(hist_mix)
    try:
        assert len(ax2.collections) == 1
    finally:
        plt.close(fig2)

    # Profile-only RTI via alpha_RTI.
    prof = _make_full_profile(tmp_path / "prof_rti.data", model_number=10)
    fig3, ax3 = plotting.plot_rti_history(profile_paths=[prof])
    try:
        assert len(ax3.collections) == 1
    finally:
        plt.close(fig3)

    # History without any mix/rti columns raises KeyError loudly.
    bare_hist = _write_mesa_log(
        tmp_path / "bare.data",
        {"version_number": '"FlexMESA"'},
        {
            "model_number": np.array([1.0, 2.0]),
            "star_mass": np.array([1.0, 1.0]),
        },
    )
    with pytest.raises(KeyError, match="rti panel requires"):
        plotting.plot_rti_history(bare_hist)
    assert not plt.get_fignums()


def test_network_chart_and_production_profile_match_chem_oracles(
    tmp_path: pathlib.Path,
) -> None:
    """Network cells match (N, Z) and Production matches log10(X/X_init)."""
    plt.close("all")
    prof_init = _make_full_profile(tmp_path / "prof_init.data", n14_scale=1.0)
    prof_2x = _make_full_profile(tmp_path / "prof_2x.data", n14_scale=2.0)

    fig_n, ax_n = plotting.plot_network_chart(prof_init)
    try:
        rects = [
            p for p in ax_n.patches if isinstance(p, mpl_patches.Rectangle)
        ]
        # 9 isotopes written in _make_full_profile:
        # h1, he3, he4, c12, n14, o16, ne20, mg24, fe56.
        assert len(rects) == 9
        fe_info = chem.get_nuclide_info(chem.get_nuclide_index("fe56"))
        coords = {
            (round(r.get_x() + 0.5), round(r.get_y() + 0.5)) for r in rects
        }
        assert (fe_info.n, fe_info.z) in coords
    finally:
        plt.close(fig_n)

    # Comparing a profile against itself yields exact 0.0 production everywhere;
    # doubling n14 yields exact log10(2.0) at A=14!
    fig_p, ax_p = plotting.plot_production_profile(
        prof_2x, init_profile_path=prof_init, ref_profile_path=prof_init
    )
    try:
        a14_vals: list[float] = []
        for ln in ax_p.get_lines():
            if ln.get_linestyle() == ":":
                continue
            if ln.get_linestyle() == "--":
                continue
            xs = np.asarray(ln.get_xdata(), dtype=np.float64)
            ys = np.asarray(ln.get_ydata(), dtype=np.float64)
            for x_v, y_v in zip(xs, ys, strict=True):
                if round(float(x_v)) == 14:
                    a14_vals.append(float(y_v))
        assert a14_vals == pytest.approx([math.log10(2.0)], abs=1.0e-12)
    finally:
        plt.close(fig_p)

    # Also verify AG89 default scaling works cleanly.
    fig_ag, _ = plotting.plot_production_profile(prof_init)
    plt.close(fig_ag)

    no_iso = _write_mesa_log(
        tmp_path / "no_iso.data",
        {"version_number": '"FlexMESA"'},
        {"mass": np.array([1.0, 0.5]), "logT": np.array([4.0, 7.0])},
    )
    with pytest.raises(ValueError, match="no recognized isotope"):
        plotting.plot_network_chart(no_iso)
    assert not plt.get_fignums()


def test_color_magnitude_explicit_inlist_autodetect_and_grid(
    tmp_path: pathlib.Path,
) -> None:
    """Color_magnitude<N> renders explicit, inlist, auto-detected, and Grid."""
    plt.close("all")
    hist = _make_full_history(tmp_path / "hist_cmd.data")

    # 1. Auto-detection of abs_mag_B / abs_mag_V.
    fig_a, axes_a = plotting.plot_color_magnitude(hist)
    try:
        assert axes_a.shape == (1,)
        assert axes_a[0].yaxis_inverted()
    finally:
        plt.close(fig_a)

    # 2. Inlist-configured stacked 2-panel CMD with secondary right axis.
    inl_cmd = tmp_path / "inlist_cmd"
    inl_cmd.write_text(
        "&pgstar\n"
        "  Color_magnitude2_title = 'UBV-I Tracks'\n"
        "  Color_magnitude2_xaxis1_name = 'abs_mag_B'\n"
        "  Color_magnitude2_xaxis2_name = 'abs_mag_V'\n"
        "  Color_magnitude2_num_panels = 2\n"
        "  Color_magnitude2_yaxis1_name(1) = 'abs_mag_V'\n"
        "  Color_magnitude2_other_yaxis1_name(1) = 'abs_mag_I'\n"
        "  Color_magnitude2_yaxis1_name(2) = 'abs_mag_V'\n"
        "  Color_magnitude2_yaxis2_name(2) = 'abs_mag_I'\n"
        "/\n",
        encoding="utf-8",
    )
    spec = plotting.parse_inlist_cmd_spec(inl_cmd, panel_index=2)
    assert len(spec.panels) == 2
    fig_i, axes_i = plotting.plot_color_magnitude(
        hist, inlist_path=inl_cmd, panel_index=2, ref_history_path=hist
    )
    try:
        assert axes_i.shape == (2,)
    finally:
        plt.close(fig_i)

    # 3. Grid embedding of Color_magnitude2 and inline Color_magnitude syntax.
    fig_g, axes_g = plotting.plot_pgstar_grid(
        panels=[
            "Color_magnitude2",
            "Color_magnitude:abs_mag_B-abs_mag_V:abs_mag_V,abs_mag_V-abs_mag_I",
        ],
        inlist_path=None,
        history_path=hist,
    )
    try:
        assert len(axes_g) == 3
    finally:
        plt.close(fig_g)

    # 4. Standard history lacking abs_mag_*/bc_* raises KeyError loudly.
    with pytest.raises(KeyError, match="Color_magnitude panel requires"):
        plotting.plot_color_magnitude(REF_DIR / "star" / "1.3M_ms_high_Z.data")
    assert not plt.get_fignums()


def test_live_evolution_plotter_and_watch_logs_emit_frames_without_leaks(
    tmp_path: pathlib.Path,
) -> None:
    """LiveEvolutionPlotter drives StarModel steps and watch_logs cleanly."""
    plt.close("all")
    logs_dir = tmp_path / "LOGS"
    out_dir = tmp_path / "frames"
    logs_dir.mkdir(parents=True)

    # Evolve a real 1.0 Msun StarModel for 4 steps, writing genuine MESA ASCII
    # history rows after each step and driving LiveEvolutionPlotter hooks.
    model = star.StarModel.from_zams(initial_mass=1.0, initial_z=0.02)
    seen: list[plotting.LiveFrameRecord] = []
    plotter = plotting.LiveEvolutionPlotter(
        out_dir,
        logs_dir=logs_dir,
        panels=("HR", "TRho", "Text_Summary1"),
        file_cnt=2,
        dpi=100,
        on_frame=seen.append,
    )
    plotter.extras_startup({"case_dir": str(tmp_path)}, restart=False)

    rows: dict[str, list[float]] = {
        "model_number": [],
        "star_age": [],
        "star_mass": [],
        "log_L": [],
        "log_Teff": [],
        "log_R": [],
        "log_g": [],
        "log_center_T": [],
        "log_center_Rho": [],
        "center_h1": [],
        "center_he4": [],
    }
    for step_idx in range(1, 5):
        if step_idx > 1:
            model.evolve_step(do_remesh=False, use_bcyclic=True)
        hrow = model.history_row()
        rows["model_number"].append(float(step_idx))
        rows["star_age"].append(float(hrow.star_age))
        rows["star_mass"].append(float(hrow.star_mass))
        rows["log_L"].append(float(hrow.log_l))
        rows["log_Teff"].append(float(hrow.log_teff))
        rows["log_R"].append(float(hrow.log_r))
        rows["log_g"].append(float(hrow.log_g))
        rows["log_center_T"].append(float(hrow.log_center_t))
        rows["log_center_Rho"].append(float(hrow.log_center_rho))
        rows["center_h1"].append(float(hrow.center_h1))
        rows["center_he4"].append(float(hrow.center_he4))
        _write_mesa_log(
            logs_dir / "history.data",
            {"version_number": '"FlexMESA"', "compiler": '"rustc"'},
            {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()},
        )
        directive = plotter.extras_finish_step({"model_number": step_idx})
        assert directive == "keep_going"

    plotter.extras_after_evolve({"model_number": 4})
    # file_cnt=2 over steps 1..4 emits models 2 and 4; after_evolve sees 4
    # already rendered and does not duplicate.
    assert [rec.model_number for rec in seen] == [2, 4]
    assert [rec.num_rows for rec in seen] == [2, 4]
    for rec in seen:
        assert rec.path.is_file() and rec.path.stat().st_size > 2_000
        assert rec.latest_path is not None and rec.latest_path.is_file()
    assert not plt.get_fignums()

    # Verify watch_logs polling incremental updates from a worker thread.
    w_logs = tmp_path / "watch_LOGS"
    w_out = tmp_path / "watch_frames"
    stop = threading.Event()
    _make_full_history(w_logs / "history.data", n_rows=3)

    def _writer() -> None:
        time.sleep(0.15)
        _make_full_history(w_logs / "history.data", n_rows=6)

    t_w = threading.Thread(target=_writer, daemon=True)
    t_w.start()
    watched = plotting.watch_logs(
        w_logs,
        w_out,
        panels=("HR", "Text_Summary1"),
        poll_interval_sec=0.04,
        max_frames=2,
        max_wait_sec=5.0,
        stop_event=stop,
        dpi=90,
    )
    t_w.join(timeout=2.0)
    assert len(watched) == 2
    assert watched[0].num_rows == 3
    assert watched[1].num_rows == 6
    assert (w_out / "frame_latest.png").is_file()
    assert not plt.get_fignums()


def test_plot_cli_increment3_subcommands_render_png_files(
    tmp_path: pathlib.Path,
) -> None:
    """Increment 3 CLI subcommands exit 0, write PNGs, and leak 0 figures."""
    plt.close("all")
    hist = _make_full_history(tmp_path / "LOGS" / "history.data")
    prof = _make_full_profile(tmp_path / "LOGS" / "profile1.data")
    ref_bin = REF_DIR / "binary" / "evolve_both_stars.data"

    cmds = [
        [
            "plot",
            "profile",
            str(prof),
            "--panel",
            "dynamo",
            "--out",
            str(tmp_path / "dynamo.png"),
        ],
        [
            "plot",
            "profile",
            str(prof),
            "--panel",
            "network",
            "--out",
            str(tmp_path / "network.png"),
        ],
        [
            "plot",
            "profile",
            str(prof),
            "--panel",
            "production",
            "--out",
            str(tmp_path / "production.png"),
        ],
        [
            "plot",
            "track",
            str(hist),
            "--preset",
            "dPg_dnu",
            "--out",
            str(tmp_path / "dpg_dnu.png"),
        ],
        [
            "plot",
            "rti",
            "--history",
            str(hist),
            "--out",
            str(tmp_path / "rti.png"),
        ],
        [
            "plot",
            "cmd",
            str(hist),
            "--x1",
            "abs_mag_B",
            "--x2",
            "abs_mag_V",
            "--y",
            "abs_mag_V,abs_mag_V-abs_mag_I",
            "--out",
            str(tmp_path / "cmd.png"),
        ],
        [
            "plot",
            "text-summary",
            str(hist),
            "--out",
            str(tmp_path / "ts_star.png"),
        ],
        [
            "plot",
            "text-summary",
            str(ref_bin),
            "--binary",
            "--out",
            str(tmp_path / "ts_bin.png"),
        ],
        [
            "plot",
            "grid",
            "--panels",
            "Orbit,Star1:Dynamo,Text_Summary1",
            "--binary-history",
            str(ref_bin),
            "--star1-profile",
            str(prof),
            "--out",
            str(tmp_path / "grid_bin.png"),
        ],
        [
            "plot",
            "watch",
            str(tmp_path / "LOGS"),
            "--out-dir",
            str(tmp_path / "cli_live"),
            "--panels",
            "HR,TRho",
            "--max-frames",
            "1",
            "--poll-interval",
            "0.05",
            "--dpi",
            "90",
        ],
    ]
    for argv in cmds:
        assert cli.main(argv) == 0

    for fname in (
        "dynamo.png",
        "network.png",
        "production.png",
        "dpg_dnu.png",
        "rti.png",
        "cmd.png",
        "ts_star.png",
        "ts_bin.png",
        "grid_bin.png",
    ):
        p = tmp_path / fname
        assert p.is_file() and p.stat().st_size > 2_000
    assert (tmp_path / "cli_live" / "frame_latest.png").is_file()
    assert not plt.get_fignums()
