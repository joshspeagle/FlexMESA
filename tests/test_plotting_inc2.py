# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for `#173` Increment 2 Kippenhahn, `pgbinary`, `pgstar`, and `Grid`."""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from flexmesa import cli, plotting
from flexmesa.plotting import _plot_support, _roche_orbit, pgstar_grid

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
                ("mix_type_", "mix_relr_type_", "donor_index", "point_mass")
            ):
                row_tokens.append(f"{round(val):24d}")
            else:
                row_tokens.append(f"{val:24.16e}")
        lines.append(" ".join(row_tokens))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_rich_profile(
    path: pathlib.Path,
    *,
    model_number: int = 5,
    star_age: float = 2.5e7,
) -> pathlib.Path:
    """Create a 6-zone MESA profileN.data covering thermo, burn, and isos."""
    hdr = {
        "version_number": '"r26.4.1"',
        "compiler": '"gfortran"',
        "MESA_SDK_version": '"x86_64-linux-26.6.1"',
        "math_backend": '"CRMATH"',
        "model_number": f"{model_number}",
        "num_zones": "6",
        "star_age": f"{star_age:.16e}",
        "star_mass": "2.0000000000000000e+00",
        "rsun": "6.9570000000000000e+10",
    }
    masses = np.array([1.99, 1.60, 1.10, 0.70, 0.35, 0.08], dtype=np.float64)
    radii = np.array([1.80, 1.35, 0.95, 0.55, 0.25, 0.06], dtype=np.float64)
    cols: dict[str, np.ndarray] = {
        "zone": np.arange(1, 7, dtype=np.float64),
        "mass": masses,
        "q": masses / 2.0,
        "radius": radii,
        "radius_cm": radii * 6.957e10,
        "logR": np.log10(radii),
        "logxm": np.log10(2.0 - masses),
        "log_column_depth": np.linspace(1.0, 10.5, 6, dtype=np.float64),
        "logP": np.linspace(5.0, 17.2, 6, dtype=np.float64),
        "logT": np.linspace(4.1, 7.45, 6, dtype=np.float64),
        "temperature": np.power(10.0, np.linspace(4.1, 7.45, 6)),
        "logRho": np.linspace(-6.5, 1.85, 6, dtype=np.float64),
        "luminosity": np.array([15.0, 15.0, 14.8, 12.0, 6.0, 1.2]),
        "entropy": np.linspace(22.0, 11.5, 6, dtype=np.float64),
        "opacity": np.array([0.4, 1.2, 2.5, 1.8, 0.9, 0.6], dtype=np.float64),
        "eps_nuc": np.array([0.0, 0.0, 2.0, 80.0, 1200.0, 8500.0]),
        "non_nuc_neu": np.array([0.0, 0.0, 0.01, 0.2, 1.5, 6.0]),
        "eps_nuc_neu_total": np.array([0.0, 0.0, 0.1, 4.0, 60.0, 420.0]),
        "pp": np.array([0.0, 0.0, 1.8, 50.0, 300.0, 900.0]),
        "cno": np.array([0.0, 0.0, 0.2, 30.0, 900.0, 7600.0]),
        "tri_alpha": np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.5]),
        "burn_c": np.zeros(6, dtype=np.float64),
        "h1": np.array([0.70, 0.70, 0.69, 0.62, 0.45, 0.28]),
        "he4": np.array([0.28, 0.28, 0.29, 0.36, 0.53, 0.70]),
        "c12": np.array([3.0e-3, 3.0e-3, 2.8e-3, 1.5e-3, 4.0e-4, 2.0e-4]),
        "n14": np.array([1.0e-3, 1.0e-3, 1.2e-3, 3.0e-3, 9.0e-3, 1.1e-2]),
        "o16": np.array([9.0e-3, 9.0e-3, 9.0e-3, 8.5e-3, 5.0e-3, 3.5e-3]),
        "fe56": np.full(6, 1.2e-3, dtype=np.float64),
        "x_mass_fraction_H": np.array([0.70, 0.70, 0.69, 0.62, 0.45, 0.28]),
        "y_mass_fraction_He": np.array([0.28, 0.28, 0.29, 0.36, 0.53, 0.70]),
        "z_mass_fraction_metals": np.full(6, 0.02, dtype=np.float64),
        "log_D_mix": np.array([13.5, 8.2, -99.0, 6.5, 10.2, 14.8]),
        "log_D_conv": np.array([13.5, -99.0, -99.0, -99.0, -99.0, 14.8]),
        "log_D_ovr": np.array([-99.0, 8.2, -99.0, -99.0, 10.2, -99.0]),
        "log_D_semi": np.array([-99.0, -99.0, -99.0, 6.5, -99.0, -99.0]),
        "mixing_type": np.array([1, 2, 0, 3, 4, 1], dtype=np.float64),
        "brunt_N2": np.array(
            [-1.0e-6, 2.0e-6, 8.0e-6, 3.0e-5, 1.0e-4, -2.0e-5]
        ),
        "csound": np.linspace(1.2e6, 5.5e7, 6, dtype=np.float64),
    }
    return _write_mesa_log(path, hdr, cols)


def _with_eccentricity(
    src: pathlib.Path, dst: pathlib.Path, ecc: float
) -> pathlib.Path:
    """Copy ``binary_history`` ``src`` to ``dst`` with a synthetic column.

    The stored MESA ``26.4.1`` binary histories lack ``eccentricity`` because
    upstream's default ``binary_history_columns.list`` leaves it commented
    out. The appended value is a test fixture, not a MESA output.
    """
    lines = src.read_text(encoding="utf-8").splitlines()
    n_cols = len(lines[4].split())
    out = lines[:4]
    out.append(f"{lines[4]} {n_cols + 1:>24d}")
    out.append(f"{lines[5]} {'eccentricity':>24s}")
    out.extend(f"{row} {ecc:24.16E}" for row in lines[6:] if row.strip())
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dst


def _make_kipp_history(path: pathlib.Path) -> pathlib.Path:
    """Create a 5-step MESA history.data with mix/burn regions and tracks."""
    hdr = {
        "version_number": '"FlexMESA"',
        "compiler": '"rustc"',
        "math_backend": '"strict"',
        "msun": "1.9884098706980504e+33",
    }
    n_t = 5
    steps = np.arange(1, n_t + 1, dtype=np.float64)
    ages = np.linspace(1.0e6, 8.0e7, n_t, dtype=np.float64)
    log_teff = np.linspace(4.02, 3.92, n_t, dtype=np.float64)
    log_l = np.linspace(1.20, 1.55, n_t, dtype=np.float64)
    log_r = np.linspace(0.22, 0.52, n_t, dtype=np.float64)
    cols: dict[str, np.ndarray] = {
        "model_number": steps,
        "star_age": ages,
        "star_mass": np.linspace(2.50, 2.46, n_t, dtype=np.float64),
        "log_xmstar": np.log10(np.linspace(2.50, 2.46, n_t) * 1.98840987e33),
        "he_core_mass": np.array([0.0, 0.18, 0.26, 0.32, 0.36]),
        "co_core_mass": np.array([0.0, 0.0, 0.05, 0.12, 0.18]),
        "fe_core_mass": np.zeros(n_t, dtype=np.float64),
        "log_L": log_l,
        "luminosity": np.power(10.0, log_l),
        "log_LH": np.linspace(1.18, 1.40, n_t, dtype=np.float64),
        "log_LHe": np.array([-8.0, -4.0, 0.2, 0.8, 1.1]),
        "log_Lneu": np.linspace(-1.2, 0.4, n_t, dtype=np.float64),
        "log_Teff": log_teff,
        "effective_T": np.power(10.0, log_teff),
        "log_R": log_r,
        "radius": np.power(10.0, log_r),
        "log_g": np.linspace(4.25, 3.65, n_t, dtype=np.float64),
        "v_surf_km_s": np.linspace(0.0, 4.5, n_t, dtype=np.float64),
        "log_center_Rho": np.linspace(1.6, 3.4, n_t, dtype=np.float64),
        "log_center_T": np.linspace(7.32, 8.08, n_t, dtype=np.float64),
        "log_max_T": np.linspace(7.32, 8.15, n_t, dtype=np.float64),
        "center_h1": np.linspace(0.70, 0.02, n_t, dtype=np.float64),
        "center_he4": np.linspace(0.28, 0.96, n_t, dtype=np.float64),
        "mix_type_1": np.array([1, 1, 1, 3, 4], dtype=np.float64),
        "mix_qtop_1": np.array([0.22, 0.20, 0.16, 0.14, 0.12]),
        "mix_type_2": np.array([2, 2, 2, 2, 2], dtype=np.float64),
        "mix_qtop_2": np.array([0.26, 0.24, 0.20, 0.18, 0.16]),
        "mix_type_3": np.array([0, 0, 1, 1, 6], dtype=np.float64),
        "mix_qtop_3": np.array([0.95, 0.95, 1.00, 1.00, 1.00]),
        "mix_type_4": np.full(n_t, -1.0, dtype=np.float64),
        "mix_qtop_4": np.zeros(n_t, dtype=np.float64),
        "mix_relr_type_1": np.array([1, 1, 3, 4, 1], dtype=np.float64),
        "mix_relr_top_1": np.array([0.18, 0.16, 0.12, 0.10, 0.08]),
        "mix_relr_type_2": np.array([2, 2, 1, 1, 1], dtype=np.float64),
        "mix_relr_top_2": np.array([0.22, 0.20, 1.00, 1.00, 1.00]),
        "burn_type_1": np.array([3.8, 4.1, 4.5, 4.9, 5.2]),
        "burn_qtop_1": np.array([0.18, 0.16, 0.14, 0.12, 0.10]),
        "burn_type_2": np.array([0.0, -1.8, -2.4, -3.1, -3.5]),
        "burn_qtop_2": np.array([0.50, 0.55, 0.60, 0.65, 0.70]),
        "burn_type_3": np.full(n_t, -9999.0, dtype=np.float64),
        "burn_qtop_3": np.zeros(n_t, dtype=np.float64),
        "burn_relr_type_1": np.array([3.8, 4.1, 4.5, 4.9, 5.2]),
        "burn_relr_top_1": np.array([0.15, 0.14, 0.12, 0.10, 0.09]),
    }
    return _write_mesa_log(path, hdr, cols)


def test_plot_history_kippenhahn_renders_mixing_burning_boundaries_and_lums(
    tmp_path: pathlib.Path,
) -> None:
    """History Kippenhahn draws mixing, cooling/burning, and core curves."""
    plt.close("all")
    hist = _make_kipp_history(tmp_path / "LOGS" / "history.data")

    out_mass = tmp_path / "kipp_mass.png"
    fig, ax = plotting.plot_history_kippenhahn(
        hist,
        xaxis="model_number",
        yaxis="mass",
        show_luminosities=True,
        save_path=out_mass,
    )
    try:
        assert out_mass.is_file() and out_mass.stat().st_size > 10_000
        assert len(fig.axes) >= 3  # main ax + colorbar + twin luminosity ax
        leg = ax.get_legend()
        assert leg is not None
        labels = {t.get_text() for t in leg.get_texts()}
        assert {
            "Convective",
            "Overshoot",
            "Semiconvective",
            "Thermohaline",
        } <= labels
        assert r"$M_{\mathrm{He}}$" in labels
        assert r"$M_{\mathrm{CO}}$" in labels
        assert r"$M_{\mathrm{Fe}}$" not in labels  # zero-mass core suppressed
        assert r"$\log L$" in labels
    finally:
        plt.close(fig)

    # Radius coordinate, Myr age from present, and log10 x-scaling
    fig_r, ax_r = plotting.plot_history_kippenhahn(
        hist,
        xaxis="star_age",
        yaxis="radius",
        age_units="Myr",
        time_from_present=True,
    )
    try:
        assert "Final Model" in ax_r.get_xlabel()
        assert "R_\\odot" in ax_r.get_ylabel()
    finally:
        plt.close(fig_r)

    # Fractional mass coordinate q on stored real-MESA 26.4.1 reference history
    ref_star = REF_DIR / "star" / "1.3M_ms_high_Z.data"
    fig_q, ax_q = plotting.plot_history_kippenhahn(
        ref_star, xaxis="star_age", yaxis="q", log_xaxis=True
    )
    try:
        assert ax_q.get_ylim()[1] == pytest.approx(1.02)
    finally:
        plt.close(fig_q)
    assert not plt.get_fignums()


def test_plot_profile_kippenhahn_multiregime_and_validation_errors(
    tmp_path: pathlib.Path,
) -> None:
    """Profile Kippenhahn preserves all mixing regimes and validates inputs."""
    plt.close("all")
    p1 = _make_rich_profile(
        tmp_path / "p1.data", model_number=1, star_age=1.0e6
    )
    p2 = _make_rich_profile(
        tmp_path / "p2.data", model_number=8, star_age=4.0e7
    )
    hist = _make_kipp_history(tmp_path / "history.data")

    fig, ax = plotting.plot_profile_kippenhahn(
        [p2, p1],
        xaxis="model_number",
        yaxis="radius",
        history_path=hist,
        mass_grid_points=40,
    )
    try:
        leg = ax.get_legend()
        assert leg is not None
        labels = {t.get_text() for t in leg.get_texts()}
        assert {
            "Convective",
            "Overshoot",
            "Semiconvective",
            "Thermohaline",
        } <= labels
    finally:
        plt.close(fig)

    with pytest.raises(ValueError, match="provide history_path"):
        plotting.plot_history_kippenhahn()
    assert not plt.get_fignums()


def test_roche_l1_symmetry_and_pgbinary_orbit_geometry(
    tmp_path: pathlib.Path,
) -> None:
    """xl1_fit satisfies q <-> 1/q reflection and Orbit needs its columns."""
    plt.close("all")
    for q_val in (0.15, 0.4, 1.0, 2.5, 6.0):
        assert _roche_orbit.xl1_fit(q_val) + _roche_orbit.xl1_fit(
            1.0 / q_val
        ) == pytest.approx(1.0, abs=1.0e-12)

    raw_both = REF_DIR / "binary" / "evolve_both_stars.data"
    # A history without eccentricity is an error, never a circular orbit.
    with pytest.raises(
        _roche_orbit.MissingOrbitColumnsError, match="eccentricity"
    ):
        plotting.plot_pgbinary_orbit(raw_both)
    plt.close("all")

    ref_both = _with_eccentricity(raw_both, tmp_path / "both_ecc.data", 0.3)
    tbl, _hdr = _plot_support.load_history_with_header(ref_both)
    state = _roche_orbit.extract_orbit_state(tbl, 0)
    assert state["ecc"] == 0.3
    assert state["sep"] == float(tbl.columns["binary_separation"][0])
    assert state["rl1"] == float(tbl.columns["rl_1"][0])
    assert state["rl2"] == float(tbl.columns["rl_2"][0])

    orb_png = tmp_path / "orbit.png"
    fig, ax = plotting.plot_pgbinary_orbit(
        ref_both,
        model_number=5,
        mesa_binary_history=ref_both,
        save_path=orb_png,
    )
    try:
        assert orb_png.is_file() and orb_png.stat().st_size > 10_000
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        assert abs(x0 + x1) < 1.0e-12
        assert (x1 - x0) == pytest.approx(y1 - y0)
    finally:
        plt.close(fig)

    # Point-mass X-ray binary reference
    ref_hmxb = _with_eccentricity(
        REF_DIR / "binary" / "wind_fed_bhhmxb.data",
        tmp_path / "hmxb_ecc.data",
        0.0,
    )
    fig_bh, _ = plotting.plot_pgbinary_orbit(ref_hmxb)
    plt.close(fig_bh)
    assert not plt.get_fignums()


def test_plot_pgbinary_panels_and_summary_history_with_component_hrs(
    tmp_path: pathlib.Path,
) -> None:
    """pgbinary 6-panel grid renders RLOF, mdot, component HR, & provenance."""
    plt.close("all")
    ref_both = REF_DIR / "binary" / "evolve_both_stars.data"
    run_dir = tmp_path / "bin_run"
    (run_dir / "LOGS1").mkdir(parents=True)
    (run_dir / "LOGS2").mkdir(parents=True)
    (run_dir / "binary_history.data").write_text(
        ref_both.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _make_kipp_history(run_dir / "LOGS1" / "history.data")
    _make_kipp_history(run_dir / "LOGS2" / "history.data")

    out_png = tmp_path / "pgbinary_cmp.png"
    fig, axes = plotting.plot_pgbinary_panels(
        flexmesa_dir=run_dir,
        case="evolve_both_stars",
        numerics_mode="strict",
        save_path=out_png,
    )
    try:
        assert axes.shape == (3, 2)
        assert out_png.is_file() and out_png.stat().st_size > 15_000
        assert axes[0, 1].xaxis_inverted()
        hr_leg = axes[0, 1].get_legend()
        assert hr_leg is not None
        hr_labels = [t.get_text() for t in hr_leg.get_texts()]
        assert any("Star 1" in s for s in hr_labels)
        assert any("Star 2" in s for s in hr_labels)
        # The run history here is a copy of the MESA reference, so the
        # figure's agreement statement carries no information and is not
        # asserted.
        assert "MESA r26.4.1" in fig.get_suptitle()
        orbit_text = " ".join(t.get_text() for t in axes[0, 0].texts)
        assert "Orbit not drawn" in orbit_text
        assert "eccentricity" in orbit_text
    finally:
        plt.close(fig)

    fig_sum, ax_sum = plotting.plot_pgbinary_summary_history(
        ref_both, mesa_binary_history=ref_both, xaxis="age"
    )
    try:
        for ln in ax_sum.get_lines():
            yv = np.asarray(ln.get_ydata(), dtype=np.float64)
            finite = yv[np.isfinite(yv)]
            if finite.size > 0:
                assert float(np.min(finite)) >= -1.0e-12
                assert float(np.max(finite)) <= 1.0 + 1.0e-12
    finally:
        plt.close(fig_sum)
    assert not plt.get_fignums()


def test_pgstar_profile_panels_match_bundled_plot_info_and_rank_series(
    tmp_path: pathlib.Path,
) -> None:
    """TRho_Profile, Power, Abundance, Mixing, & Summary_Profile match MESA."""
    plt.close("all")
    prof = _make_rich_profile(tmp_path / "profile1.data")
    psi_rho, psi_t = _plot_support.load_plot_info_curve("psi4.data")

    fig_tr, ax_tr = plotting.plot_trho_profile(
        prof,
        ref_profile_path=prof,
        show_gamma1_4_3rd=True,
        show_kap_lines=True,
    )
    try:
        by_lbl = {ln.get_label(): ln for ln in ax_tr.get_lines()}
        psi_ln = by_lbl[r"$\epsilon_{\mathrm{F}}/k_{\mathrm{B}}T \approx 4$"]
        assert np.allclose(psi_ln.get_xdata(), psi_rho)
        assert np.allclose(psi_ln.get_ydata(), psi_t)
        assert r"$\Gamma_1 < 4/3$" in by_lbl
        assert r"$P_{\mathrm{rad}} \approx P_{\mathrm{gas}}$" in by_lbl
    finally:
        plt.close(fig_tr)

    # Column-depth and logxm coordinate modes suppress logRho-only EOS guides
    for alt_x in ("column_depth", "logxm"):
        fig_alt, ax_alt = plotting.plot_trho_profile(prof, xaxis=alt_x)
        try:
            labels = {ln.get_label() for ln in ax_alt.get_lines()}
            assert r"$\Gamma_1 < 4/3$" not in labels
        finally:
            plt.close(fig_alt)

    fig_pw, ax_pw = plotting.plot_power_profile(prof, ref_profile_path=prof)
    try:
        pw_labels = [
            str(ln.get_label())
            for ln in ax_pw.get_lines()
            if not str(ln.get_label()).endswith("MESA")
        ]
        # cno peak (7600) > pp peak (900) -> cno appears first
        assert pw_labels[:2] == ["cno", "pp"]
    finally:
        plt.close(fig_pw)

    fig_ab, ax_ab = plotting.plot_abundance_profile(prof, ref_profile_path=prof)
    try:
        ab_labels = [
            str(ln.get_label())
            for ln in ax_ab.get_lines()
            if not str(ln.get_label()).endswith("MESA")
        ]
        assert ab_labels[:2] == ["h1", "he4"]
        assert "n14" in ab_labels and "o16" in ab_labels
    finally:
        plt.close(fig_ab)

    fig_mx, _ = plotting.plot_mixing_profile(prof, ref_profile_path=prof)
    plt.close(fig_mx)

    fig_sp, ax_sp = plotting.plot_summary_profile(prof, ref_profile_path=prof)
    try:
        assert ax_sp.get_ylim() == pytest.approx((-0.04, 1.05))
    finally:
        plt.close(fig_sp)

    fig_sb, ax_sb = plotting.plot_summary_burn_profile(
        prof, ref_profile_path=prof
    )
    try:
        ann_texts = {txt.get_text() for txt in ax_sb.texts}
        assert "cno" in ann_texts and "pp" in ann_texts
    finally:
        plt.close(fig_sb)

    fig_cp, axes_cp = plotting.plot_profile_custom_panels(
        prof,
        [("logT", "logRho"), "entropy"],
        ref_profile_path=prof,
        xaxis="q",
    )
    try:
        assert axes_cp.shape == (2,)
    finally:
        plt.close(fig_cp)
    assert not plt.get_fignums()


def test_pgstar_history_tracks_panels_and_inlist_grid_composer(
    tmp_path: pathlib.Path,
) -> None:
    """All 13 track presets, History_Panels, and Grid<N> inlists render."""
    plt.close("all")
    hist = _make_kipp_history(tmp_path / "history.data")
    prof = _make_rich_profile(tmp_path / "profile1.data")
    ref_bin = REF_DIR / "binary" / "evolve_both_stars.data"

    for preset_name, preset_spec in pgstar_grid.TRACK_PRESETS.items():
        fig_t, ax_t = plotting.plot_history_track(
            hist, preset=preset_name, ref_history_path=hist
        )
        try:
            assert bool(ax_t.xaxis_inverted()) is preset_spec.reverse_x
            assert bool(ax_t.yaxis_inverted()) is preset_spec.reverse_y
        finally:
            plt.close(fig_t)

    fig_hp, axes_hp = plotting.plot_history_panels(
        hist, [("log_L", "log_Teff"), "he_core_mass"], ref_history_path=hist
    )
    try:
        assert axes_hp.shape == (2,)
    finally:
        plt.close(fig_hp)

    fig_sh, _ = plotting.plot_summary_history(hist, ref_history_path=hist)
    plt.close(fig_sh)

    inlist_pg = tmp_path / "inlist_pgstar"
    inlist_pg.write_text(
        "&pgstar\n"
        "  Grid1_win_flag = .true.\n"
        "  Grid1_title = 'Custom Star Dashboard'\n"
        "  Grid1_num_rows = 2\n"
        "  Grid1_num_cols = 3\n"
        "  Grid1_num_plots = 5\n"
        "  Grid1_plot_name(1) = 'HR'\n"
        "  Grid1_plot_row(1) = 1\n"
        "  Grid1_plot_col(1) = 1\n"
        "  Grid1_plot_name(2) = 'TRho_Profile'\n"
        "  Grid1_plot_row(2) = 1\n"
        "  Grid1_plot_col(2) = 2\n"
        "  Grid1_plot_name(3) = 'Power'\n"
        "  Grid1_plot_row(3) = 1\n"
        "  Grid1_plot_col(3) = 3\n"
        "  Grid1_plot_name(4) = 'Kipp'\n"
        "  Grid1_plot_row(4) = 2\n"
        "  Grid1_plot_col(4) = 1\n"
        "  Grid1_plot_colspan(4) = 2\n"
        "  Grid1_plot_name(5) = 'Mode_Prop'\n"
        "  Grid1_plot_row(5) = 2\n"
        "  Grid1_plot_col(5) = 3\n"
        "/\n",
        encoding="utf-8",
    )
    r_n, c_n, ttl, specs = plotting.parse_inlist_grid_spec(inlist_pg)
    assert (r_n, c_n, ttl, len(specs)) == (2, 3, "Custom Star Dashboard", 5)

    fig_g, g_axes = plotting.plot_pgstar_grid(
        inlist_path=inlist_pg,
        history_path=hist,
        profile_path=prof,
    )
    try:
        assert len(g_axes) == 5
        assert fig_g.get_suptitle() == "Custom Star Dashboard"
    finally:
        plt.close(fig_g)

    # Binary Grid with Orbit + Star1 + Star2 cells
    inlist_bin = tmp_path / "inlist_pgbinary"
    inlist_bin.write_text(
        "&pgbinary\n"
        "  Star1_plot_name = 'HR'\n"
        "  Star2_plot_name = 'Abundance'\n"
        "  Grid1_num_rows = 1\n"
        "  Grid1_num_cols = 3\n"
        "  Grid1_num_plots = 3\n"
        "  Grid1_plot_name(1) = 'Orbit'\n"
        "  Grid1_plot_row(1) = 1\n"
        "  Grid1_plot_col(1) = 1\n"
        "  Grid1_plot_name(2) = 'Star1'\n"
        "  Grid1_plot_row(2) = 1\n"
        "  Grid1_plot_col(2) = 2\n"
        "  Grid1_plot_name(3) = 'Star2'\n"
        "  Grid1_plot_row(3) = 1\n"
        "  Grid1_plot_col(3) = 3\n"
        "/\n",
        encoding="utf-8",
    )
    fig_bg, bg_axes = plotting.plot_pgstar_grid(
        inlist_path=inlist_bin,
        binary_history_path=_with_eccentricity(
            ref_bin, tmp_path / "grid_ecc.data", 0.2
        ),
        star1_history_path=hist,
        star2_profile_path=prof,
    )
    try:
        assert len(bg_axes) == 3
        assert "Star 1:" in bg_axes[1].get_title()
        assert "Star 2:" in bg_axes[2].get_title()
    finally:
        plt.close(fig_bg)

    # Invalid panel name closes figure cleanly and raises ValueError
    with pytest.raises(ValueError, match="unsupported"):
        plotting.plot_pgstar_grid(["Nonexistent_Panel"], history_path=hist)
    assert not plt.get_fignums()


def test_flexmesa_plot_cli_increment2_subcommands_write_figures_cleanly(
    tmp_path: pathlib.Path,
) -> None:
    """CLI subcommands kipp, binary, profile, track, and grid succeed."""
    plt.close("all")
    hist = _make_kipp_history(tmp_path / "history.data")
    prof = _make_rich_profile(tmp_path / "profile1.data")
    ref_bin = REF_DIR / "binary" / "evolve_both_stars.data"

    for p_mode in (
        "trho",
        "power",
        "abundance",
        "mixing",
        "summary",
        "summary_burn",
    ):
        out_p = tmp_path / f"prof_{p_mode}.png"
        rc = cli.main(
            [
                "plot",
                "profile",
                str(prof),
                "--panel",
                p_mode,
                "--out",
                str(out_p),
            ]
        )
        assert rc == 0 and out_p.is_file()

    out_kipp = tmp_path / "cli_kipp.png"
    assert (
        cli.main(
            [
                "plot",
                "kipp",
                "--history",
                str(hist),
                "--show-luminosities",
                "--out",
                str(out_kipp),
            ]
        )
        == 0
    )
    assert out_kipp.is_file()

    out_bin = tmp_path / "cli_bin.png"
    assert (
        cli.main(
            [
                "plot",
                "binary",
                "--binary-history",
                str(ref_bin),
                "--case",
                "evolve_both_stars",
                "--out",
                str(out_bin),
            ]
        )
        == 0
    )
    assert out_bin.is_file()

    out_orb = tmp_path / "cli_orb.png"
    assert (
        cli.main(
            [
                "plot",
                "binary",
                "--binary-history",
                str(
                    _with_eccentricity(ref_bin, tmp_path / "cli_ecc.data", 0.2)
                ),
                "--orbit-only",
                "--out",
                str(out_orb),
            ]
        )
        == 0
    )
    assert out_orb.is_file()

    out_trk = tmp_path / "cli_trk.png"
    assert (
        cli.main(
            [
                "plot",
                "track",
                str(hist),
                "--preset",
                "TmaxRho",
                "--out",
                str(out_trk),
            ]
        )
        == 0
    )
    assert out_trk.is_file()

    out_grd = tmp_path / "cli_grid.png"
    assert (
        cli.main(
            [
                "plot",
                "grid",
                "--panels",
                "HR,Abundance,Summary_Profile,History_Track:star_age:log_L",
                "--history",
                str(hist),
                "--profile",
                str(prof),
                "--out",
                str(out_grd),
            ]
        )
        == 0
    )
    assert out_grd.is_file()
    assert not plt.get_fignums()


def test_mode_prop_grid_rsun_fallback_and_bare_str_rejection(
    tmp_path: pathlib.Path,
) -> None:
    """Mode_Prop uses constants.RSUN fallback and rejects bare str."""
    plt.close("all")
    r_rsun = np.linspace(1.0, 0.05, 32, dtype=np.float64)
    minimal_prof = _write_mesa_log(
        tmp_path / "minimal_mode_prop.data",
        {
            "model_number": "10",
            "num_zones": "32",
            "version_number": '"r26.4.1"',
        },
        {
            "radius": r_rsun,
            "csound": np.linspace(8.0e5, 5.0e7, 32, dtype=np.float64),
            "brunt_N2": np.linspace(1.0e-6, 4.0e-6, 32, dtype=np.float64),
        },
    )
    fig_mp, axes_mp = plotting.plot_pgstar_grid(
        ["Mode_Prop"], profile_path=minimal_prof
    )
    try:
        lamb_lines = [
            ln
            for ln in axes_mp[0].get_lines()
            if "S_{1}" in str(ln.get_label())
        ]
        assert len(lamb_lines) == 1
        y_s1 = np.asarray(lamb_lines[0].get_ydata(), dtype=np.float64)
        peak_log_uhz = float(np.nanmax(y_s1))
        # Physical solar-like log10(S_1 / uHz) lies in ~2..4.5 (never ~14.3).
        assert 2.0 < peak_log_uhz < 4.5
    finally:
        plt.close(fig_mp)

    bad_str: Any = "Mode_Prop"
    with pytest.raises(TypeError, match="not a bare str"):
        plotting.plot_pgstar_grid(bad_str, profile_path=minimal_prof)
    with pytest.raises(TypeError, match="not a bare str"):
        plotting.plot_profile_custom_panels(minimal_prof, bad_str)
    hist = _make_kipp_history(tmp_path / "h.data")
    with pytest.raises(TypeError, match="not a bare str"):
        plotting.plot_history_panels(hist, bad_str)
    assert not plt.get_fignums()


def test_kippenhahn_log_time_from_present_and_lineless_framing(
    tmp_path: pathlib.Path,
) -> None:
    """Kippenhahn frames log(t_final - t) and line-less runs cleanly."""
    plt.close("all")
    hist = _make_kipp_history(tmp_path / "kipp_h.data")

    fig_lt, ax_lt = plotting.plot_history_kippenhahn(
        hist,
        xaxis="star_age",
        age_units="Myr",
        time_from_present=True,
        log_xaxis=True,
    )
    try:
        x0, x1 = ax_lt.get_xlim()
        # Reversed so evolution advances left-to-right with compact span.
        assert x0 > x1
        assert abs(x0 - x1) < 6.0
        assert min(x0, x1) > -5.0
    finally:
        plt.close(fig_lt)

    fig_unrev, ax_unrev = plotting.plot_history_kippenhahn(
        hist,
        xaxis="star_age",
        age_units="Myr",
        time_from_present=True,
        log_xaxis=True,
        xaxis_reversed=False,
    )
    try:
        assert ax_unrev.get_xlim()[0] < ax_unrev.get_xlim()[1]
    finally:
        plt.close(fig_unrev)

    # Line-less history call (show_mass_boundaries=False -> no Line2D).
    fig_nl, ax_nl = plotting.plot_history_kippenhahn(
        hist, show_mass_boundaries=False, yaxis="mass"
    )
    try:
        assert not ax_nl.get_lines()
        assert ax_nl.get_xlim() == pytest.approx((0.5, 5.5))
        assert ax_nl.get_ylim()[1] == pytest.approx(2.50 * 1.04)
    finally:
        plt.close(fig_nl)

    # Line-less profile_paths call.
    p1 = _make_rich_profile(
        tmp_path / "kp1.data", model_number=1, star_age=1.0e6
    )
    p2 = _make_rich_profile(
        tmp_path / "kp2.data", model_number=2, star_age=5.0e6
    )
    fig_np, ax_np = plotting.plot_history_kippenhahn(
        profile_paths=[p1, p2], xaxis="model_number", show_mass_boundaries=False
    )
    try:
        assert not ax_np.get_lines()
        assert ax_np.get_xlim() == pytest.approx((1.0, 2.0))
        assert ax_np.get_ylim()[1] > 1.5
    finally:
        plt.close(fig_np)
    assert not plt.get_fignums()


def test_reference_provenance_subtitles_and_summary_burn_neutrino_overlay(
    tmp_path: pathlib.Path,
) -> None:
    """All comparison builders display Ref provenance and reference eps_neu."""
    plt.close("all")
    prof = _make_rich_profile(tmp_path / "cand_prof.data")
    ref_prof = _make_rich_profile(tmp_path / "ref_prof.data")
    ref_star = REF_DIR / "star" / "1.3M_ms_high_Z.data"
    ref_bin = REF_DIR / "binary" / "evolve_both_stars.data"

    for fn in (
        plotting.plot_trho_profile,
        plotting.plot_power_profile,
        plotting.plot_abundance_profile,
        plotting.plot_mixing_profile,
        plotting.plot_summary_profile,
        plotting.plot_summary_burn_profile,
    ):
        fig_p, ax_p = fn(prof, ref_profile_path=ref_prof)
        try:
            assert "Ref: MESA r26.4.1" in ax_p.get_title()
            if fn is plotting.plot_summary_burn_profile:
                labels = {str(ln.get_label()) for ln in ax_p.get_lines()}
                assert r"$\log\epsilon_\nu$ MESA" in labels
        finally:
            plt.close(fig_p)

    ref_bin_ecc = _with_eccentricity(ref_bin, tmp_path / "prov_ecc.data", 0.2)
    fig_orb, ax_orb = plotting.plot_pgbinary_orbit(
        ref_bin_ecc, mesa_binary_history=ref_bin_ecc
    )
    try:
        assert "Ref: MESA r26.4.1" in ax_orb.get_title()
    finally:
        plt.close(fig_orb)

    fig_bs, ax_bs = plotting.plot_pgbinary_summary_history(
        ref_bin, mesa_binary_history=ref_bin
    )
    try:
        assert "Ref: MESA r26.4.1" in ax_bs.get_title()
    finally:
        plt.close(fig_bs)

    fig_ht, ax_ht = plotting.plot_history_track(
        ref_star, preset="HR", ref_history_path=ref_star
    )
    try:
        assert "Ref: MESA r26.4.1" in ax_ht.get_title()
        line_labels = {str(ln.get_label()) for ln in ax_ht.get_lines()}
        assert "MESA r26.4.1" in line_labels
    finally:
        plt.close(fig_ht)

    fig_sh, ax_sh = plotting.plot_summary_history(
        ref_star, ref_history_path=ref_star
    )
    try:
        assert "Ref: MESA r26.4.1" in ax_sh.get_title()
    finally:
        plt.close(fig_sh)

    fig_hp, _ = plotting.plot_history_panels(
        ref_star, ["log_L", "log_Teff"], ref_history_path=ref_star
    )
    try:
        assert "Ref: MESA r26.4.1" in fig_hp.get_suptitle()
    finally:
        plt.close(fig_hp)

    # Binary fallback in Grid TRACK_PRESETS forwards ref_binary_history_path.
    bin_with_hr = _write_mesa_log(
        tmp_path / "bin_hr.data",
        {"version_number": '"r26.4.1"'},
        {
            "model_number": np.array([1.0, 2.0, 3.0]),
            "log_Teff": np.array([3.8, 3.79, 3.78]),
            "log_L": np.array([0.4, 0.45, 0.5]),
        },
    )
    fig_bg, axes_bg = plotting.plot_pgstar_grid(
        ["HR"],
        binary_history_path=bin_with_hr,
        ref_binary_history_path=bin_with_hr,
    )
    try:
        assert "Ref: MESA r26.4.1" in axes_bg[0].get_title()
    finally:
        plt.close(fig_bg)
    assert not plt.get_fignums()
