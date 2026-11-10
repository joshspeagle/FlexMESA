# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification suite for the MESA I-VI Instrument Paper Figure Reproduction
Atlas (Track 4, Epic `#110`, Sub-Issues `#116`-#121`).

Verifies that all 18 publication-styled 300 dpi PNG figures and all 6
deterministic CSV numerical ledgers (`ledger.csv`) across MESA I-VI exist, are
non-empty, and satisfy quantitative physical landmark bounds.
"""

from __future__ import annotations

import csv
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_BASE = ROOT / "docs" / "assets" / "figures" / "mesa_papers"

EXPECTED_FIGURES: dict[str, tuple[str, ...]] = {
    "mesa_i": (
        "mesa_i_eos_kap_regimes.png",
        "mesa_i_hr_and_central_tracks.png",
        "mesa_i_kippenhahn_and_abundances.png",
    ),
    "mesa_ii": (
        "mesa_ii_giant_planets.png",
        "mesa_ii_asteroseismology_echelle.png",
        "mesa_ii_rotation_and_ccsn_progenitor.png",
    ),
    "mesa_iii": (
        "mesa_iii_binary_rlof_and_jdot.png",
        "mesa_iii_weak_rates_urca.png",
        "mesa_iii_hllc_shock_and_ccsn.png",
    ),
    "mesa_iv": (
        "mesa_iv_conv_premix_and_predictive.png",
        "mesa_iv_diffusion_and_levitation.png",
        "mesa_iv_rsp_periods_and_lightcurves.png",
    ),
    "mesa_v": (
        "mesa_v_rsp_work_and_multiband.png",
        "mesa_v_energy_conservation_budget.png",
        "mesa_v_wd_crystallization_and_22ne.png",
    ),
    "mesa_vi": (
        "mesa_vi_tdc_vs_cox_mlt.png",
        "mesa_vi_autodiff_vcurve_and_scaling.png",
        "mesa_vi_skye_wd_phase_separation.png",
    ),
}


def _read_ledger(paper_dir: str) -> dict[str, float]:
    """Reads `ledger.csv` for `paper_dir` as a float dictionary."""
    csv_path = FIG_BASE / paper_dir / "ledger.csv"
    assert csv_path.is_file(), f"Missing ledger: {csv_path}"
    out: dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = float(row["value"])
            assert math.isfinite(val), f"Non-finite {row['key']} in {paper_dir}"
            out[str(row["key"])] = val
    return out


def test_all_18_mesa_paper_figures_and_ledgers_exist() -> None:
    """All 18 MESA I-VI PNG figures and 6 CSV ledgers exist and are valid."""
    for paper_key, png_names in EXPECTED_FIGURES.items():
        p_dir = FIG_BASE / paper_key
        assert p_dir.is_dir(), f"Missing paper directory: {p_dir}"
        for png in png_names:
            fig_path = p_dir / png
            assert fig_path.is_file(), f"Missing figure: {fig_path}"
            assert fig_path.stat().st_size > 15_000, f"Empty PNG: {fig_path}"
        ledger = _read_ledger(paper_key)
        assert len(ledger) >= 5, f"Incomplete ledger in {paper_key}: {ledger}"


def test_mesa_i_and_ii_quantitative_landmarks_satisfy_physical_bounds() -> None:
    """MESA I and MESA II ledger values satisfy astrophysical bounds."""
    l1 = _read_ledger("mesa_i")
    assert 0.90 <= l1["pisn_gamma1_min"] < 1.30
    assert l1["kap_fe_bump_log_kap"] > 0.5
    assert 3.74 <= l1["solar_zams_log_teff"] <= 3.79
    assert l1["rgb_tip_log_l"] >= 3.0

    l2 = _read_ledger("mesa_ii")
    assert 0.85 < l2["jup_cold_4p5gyr_rjup"] < l2["jup_hot_1500k_4p5gyr_rjup"]
    assert l2["jup_hot_1500k_4p5gyr_rjup"] < 2.2
    assert 120.0 <= l2["solar_dnu_uhz"] <= 170.0
    assert 2500.0 <= l2["solar_numax_uhz"] <= 4200.0
    assert 60.0 <= l2["rg_dpi1_asymptotic_s"] <= 95.0
    assert 1.20 <= l2["ccsn_fe_core_mass_msun"] <= 1.85
    assert l2["ccsn_infall_speed_km_s"] >= 500.0


def test_mesa_iii_and_iv_quantitative_landmarks_satisfy_physical_bounds() -> (
    None
):
    """MESA III and IV ledger values satisfy binary, hydro, and RSP bounds."""
    l3 = _read_ledger("mesa_iii")
    assert -6.0 <= l3["rlof_peak_log_mdot_msun_yr"] <= -1.0
    assert 0.50 <= l3["gr_final_sep_ratio"] < 1.0
    assert l3["ml_final_sep_ratio"] > 1.0
    assert 9.15 <= l3["urca_a23_log_rhoye_thresh"] <= 9.30
    assert l3["hllc_rankine_hugoniot_rel_err"] < 1.0e-2
    assert l3["ccsn_max_log_d_rti"] >= 10.0

    l4 = _read_ledger("mesa_iv")
    assert abs(l4["cpm_bdy_gradr_over_grada"] - 1.0) < 0.02
    assert l4["cpm_core_mass_gain_frac"] > 0.10
    assert l4["diffusion_mass_rel_err"] < 1.0e-10
    assert l4["levitation_fe_peak_over_g"] > 1.0
    assert 0.68 <= l4["cepheid_p1_over_p0"] <= 0.75


def test_mesa_v_and_vi_quantitative_landmarks_satisfy_physical_bounds() -> None:
    """MESA V and VI ledger values satisfy energy, WD, TDC, and AD bounds."""
    l5 = _read_ledger("mesa_v")
    assert 4.50 <= l5["rsp_heii_peak_log_t"] <= 4.75
    assert l5["rsp_cum_work_positive"] > 0.0
    assert l5["dedt_rel_energy_err"] < 1.0e-8
    assert l5["eps_grav_rel_energy_err"] > 1.0e-5
    assert l5["wd_cryst_delay_gyr"] > 0.5
    assert l5["wd_22ne_delay_gyr"] > 0.8

    l6 = _read_ledger("mesa_vi")
    assert l6["tdc_long_dt_vc_rel_err"] < 1.0e-3
    assert l6["tdc_long_dt_gradt_rel_err"] < 1.0e-3
    assert l6["autodiff_exact_rel_err"] <= 1.0e-15
    assert l6["fd_centered_min_rel_err"] > 1.0e-12
    assert l6["skye_phase_span"] > 0.80
    assert l6["wd_cryst_core_o16_shift"] >= 0.05
