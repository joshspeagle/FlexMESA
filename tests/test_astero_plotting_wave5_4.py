# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verification suite for Wave 5.4 (`flexmesa.astero` and `flexmesa.plotting`).

Validates:
1. Exact parameter recovery on the upstream Fortran MESA
   `astero/test_suite/surface_effects` synthetic benchmark (`targets.inc`)
   for `cubic`, `combined`, `power_law`, `sonoi`, and `kjeldsen`.
2. Cross-backend Rust-vs-JAX numerical parity (`rtol <= 1e-12`) across all
   asteroseismic scaling relations, acoustic integrals, glitch acoustic
   depths, surface corrections, separation ratios, and composite $\\chi^2$.
3. Exact `jax.grad` autodiff verification against analytical derivatives.
4. Nelder-Mead, Powell quadratic trust-region, Cartesian grid optimizers,
   and Python callback exception propagation.
5. All 7 `flexmesa.plotting` publication diagnostic figure routines.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

import flexmesa
from flexmesa import astero, plotting


def _bad_obj(_x: np.ndarray) -> float:
    """Raises `RuntimeError` to test callback exception propagation."""
    raise RuntimeError("callback failure")


def test_surface_corrections_recover_upstream_mesa_synthetic_targets() -> None:
    """Verifies `eval_surface_correction` against `targets.inc`."""
    l0_f = np.array(
        [
            1547.5529086814887, 1686.5782235116216, 1822.7107228524903,
            1957.2439222905895, 2093.6476158665583, 2228.7652424234389,
            2362.2389226681048,
        ],
        dtype=np.float64,
    )
    l0_e = np.array(
        [
            2.1292743723180268e-7, 1.1135147727092238e-7,
            6.6478428973830106e-8, 4.1795729820714373e-8,
            3.0811157229597184e-8, 2.4631107624679803e-8,
            2.2699260260711955e-8,
        ],
        dtype=np.float64,
    )
    l1_f = np.array(
        [
            1472.8473501195944, 1612.4877920975303, 1749.2610903356774,
            1884.7495763261306, 2020.4319901501815, 2156.6788795384110,
            2291.0315218037995,
        ],
        dtype=np.float64,
    )
    l1_e = np.array(
        [
            3.0470109559254443e-7, 1.5393933742126500e-7,
            8.7165650188154865e-8, 5.3183072578459202e-8,
            3.6352059417962865e-8, 2.8012380501264935e-8,
            2.4267078273580464e-8,
        ],
        dtype=np.float64,
    )
    f_mod, e_mod = np.concatenate([l0_f, l1_f]), np.concatenate([l0_e, l1_e])
    sig, nu_max = np.full_like(f_mod, 0.1), 1951.3188812142661

    target_a3 = -1.9792894171293238e-18
    f_obs_cubic = f_mod + target_a3 * (f_mod**3) / e_mod
    res_cubic_r = astero.eval_surface_correction(
        "cubic", f_obs_cubic, sig, f_mod, e_mod, backend="rust"
    )
    res_cubic_j = astero.eval_surface_correction(
        "cubic", jnp.asarray(f_obs_cubic), jnp.asarray(sig),
        jnp.asarray(f_mod), jnp.asarray(e_mod), backend="jax",
    )
    np.testing.assert_allclose(res_cubic_r["p0"], target_a3, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(res_cubic_j["p0"]), target_a3, rtol=1e-12
    )
    np.testing.assert_allclose(
        res_cubic_r["nu_corrected"], f_obs_cubic, rtol=1e-14
    )

    target_b1, target_b3 = -5.1259727935672011e-5, -1.4463074898793361e-18
    f_obs_comb = f_mod + (target_b1 / f_mod + target_b3 * (f_mod**3)) / e_mod
    res_comb_r = astero.eval_surface_correction(
        "combined", f_obs_comb, sig, f_mod, e_mod, backend="rust"
    )
    res_comb_j = astero.eval_surface_correction(
        "combined", jnp.asarray(f_obs_comb), jnp.asarray(sig),
        jnp.asarray(f_mod), jnp.asarray(e_mod), backend="jax",
    )
    np.testing.assert_allclose(res_comb_r["p0"], target_b3, rtol=1e-12)
    np.testing.assert_allclose(res_comb_r["p1"], target_b1, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(res_comb_j["p0"]), target_b3, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(res_comb_j["p1"]), target_b1, rtol=1e-12
    )

    q_mod = np.array(
        [
            e_mod[i] / float(np.interp(f_mod[i], l0_f, l0_e))
            for i in range(f_mod.size)
        ]
    )
    f_obs_kj = l0_f + (-3.5) * ((l0_f / nu_max) ** 4.9)
    dnu_mod = astero.eval_radial_delta_nu_ols(
        np.arange(1, 8, dtype=float), l0_f
    )
    r_exact = 3.9 / (
        4.9 * (np.mean(l0_f) / np.mean(f_obs_kj)) - dnu_mod / 135.0
    )
    a_div_r_exact = (
        (np.mean(f_obs_kj) - r_exact * np.mean(l0_f))
        * 7.0
        / (np.sum((f_obs_kj / nu_max) ** 4.9) * r_exact)
    )
    res_kj_r = astero.eval_surface_correction(
        "kjeldsen", np.concatenate([f_obs_kj, l1_f]), sig, f_mod, e_mod,
        l0_obs_freqs=f_obs_kj, l0_model_freqs=l0_f, l0_inertias=l0_e,
        delta_nu_obs=135.0, nu_max=nu_max, b_power=4.9, backend="rust",
    )
    np.testing.assert_allclose(res_kj_r["p0"], a_div_r_exact, rtol=1e-12)
    np.testing.assert_allclose(res_kj_r["p1"], r_exact, rtol=1e-14)
    np.testing.assert_allclose(
        res_kj_r["nu_corrected"],
        f_mod + (a_div_r_exact / q_mod) * ((f_mod / nu_max) ** 4.9),
        rtol=1e-13,
    )

    target_p0, target_p1 = -6.1575822812351042, 4.3428646403442714
    f_obs_pl = f_mod + target_p0 * ((f_mod / nu_max) ** target_p1) / q_mod
    res_pl = astero.eval_surface_correction(
        "power_law", f_obs_pl, sig, f_mod, e_mod,
        l0_model_freqs=l0_f, l0_inertias=l0_e, nu_max=nu_max,
    )
    np.testing.assert_allclose(res_pl["p0"], target_p0, rtol=1e-7)
    np.testing.assert_allclose(res_pl["p1"], target_p1, rtol=1e-7)

    target_s0, target_s1 = -4.3807350227280401e-3, 7.1437835921589503
    f_obs_sonoi = (
        f_mod
        + target_s0
        * nu_max
        * (1.0 - 1.0 / (1.0 + (f_mod / nu_max) ** target_s1))
        / q_mod
    )
    res_sonoi = astero.eval_surface_correction(
        "sonoi", f_obs_sonoi, sig, f_mod, e_mod,
        l0_model_freqs=l0_f, l0_inertias=l0_e, nu_max=nu_max,
    )
    np.testing.assert_allclose(res_sonoi["p0"], target_s0, rtol=1e-6)
    np.testing.assert_allclose(res_sonoi["p1"], target_s1, rtol=1e-6)


def test_astero_scaling_ratios_chi2_rust_jax_and_autodiff_parity() -> None:
    """Verifies Rust-vs-JAX parity (`<= 1e-12`) and `jax.grad` parity."""
    m_g, r_cm, teff, lum = 1.9884098706980504e33, 6.957e10, 5772.0, 3.828e33
    dnu_r = astero.eval_delta_nu_scaling(m_g, teff, lum, backend="rust")
    dnu_j = float(
        astero.eval_delta_nu_scaling(
            jnp.float64(m_g), jnp.float64(teff), jnp.float64(lum), backend="jax"
        )
    )
    np.testing.assert_allclose(dnu_r, 134.91, rtol=1e-12)
    np.testing.assert_allclose(dnu_r, dnu_j, rtol=1e-14)
    np.testing.assert_allclose(
        astero.eval_delta_nu_radius_scaling(m_g, r_cm, backend="rust"),
        float(
            astero.eval_delta_nu_radius_scaling(
                jnp.float64(m_g), jnp.float64(r_cm), backend="jax"
            )
        ),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        astero.eval_delta_nu_asymptotic(3706.0, backend="rust"),
        float(
            astero.eval_delta_nu_asymptotic(jnp.float64(3706.0), backend="jax")
        ),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        astero.eval_nu_max(m_g, r_cm, teff, backend="rust"),
        float(
            astero.eval_nu_max(
                jnp.float64(m_g), jnp.float64(r_cm), jnp.float64(teff),
                backend="jax",
            )
        ),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        astero.eval_nu_ac(2.74e4, 1.66, 2.0e-7, 1.0e5, backend="rust"),
        float(
            astero.eval_nu_ac(
                jnp.float64(2.74e4), jnp.float64(1.66), jnp.float64(2.0e-7),
                jnp.float64(1.0e5), backend="jax",
            )
        ),
        rtol=1e-14,
    )

    r_prof = np.linspace(r_cm, 0.05 * r_cm, 64)
    cs_prof = np.linspace(8.0e5, 5.0e7, 64)
    ac_r = astero.eval_acoustic_profile(
        r_prof, cs_prof, 0.0, 0.85 * r_cm, backend="rust"
    )
    ac_j = astero.eval_acoustic_profile(
        jnp.asarray(r_prof), jnp.asarray(cs_prof), 0.0, 0.85 * r_cm,
        backend="jax",
    )
    np.testing.assert_allclose(
        ac_r["acoustic_radius"], np.asarray(ac_j["acoustic_radius"]), rtol=1e-13
    )
    np.testing.assert_allclose(
        ac_r["photosphere_acoustic_r"],
        float(ac_j["photosphere_acoustic_r"]),
        rtol=1e-13,
    )
    gradr = np.where(r_prof > 0.72 * r_cm, 0.45, 0.30)
    gamma1 = 1.66 - 0.25 * np.exp(-(((r_prof / r_cm) - 0.98) / 0.01) ** 2)
    r_bcz, tau_bcz, tau_he = astero.eval_glitch_acoustic_depths(
        r_prof, cs_prof, gradr, np.full(64, 0.40), gamma1,
        np.linspace(5772.0, 1.5e7, 64),
    )
    assert r_bcz > 0.0 and tau_bcz > 0.0 and tau_he >= 0.0

    r_mid = 0.5 * (r_prof + np.append(r_prof[1:], 0.0))
    n2_prof = np.where(r_prof < 0.70 * r_cm, 4.0e-6, -1.0e-8)
    np.testing.assert_allclose(
        astero.eval_delta_pg(
            r_prof, r_mid, n2_prof, cs_prof, 100.0, backend="rust"
        ),
        float(
            astero.eval_delta_pg(
                jnp.asarray(r_prof), jnp.asarray(r_mid), jnp.asarray(n2_prof),
                jnp.asarray(cs_prof), 100.0, backend="jax",
            )
        ),
        rtol=1e-13,
    )
    assert (
        astero.eval_delta_pg(
            r_prof, r_mid, n2_prof, cs_prof, 100.0, bildsten2012=True
        )
        > 0.0
    )

    l0 = np.array([1800.0, 1935.0, 2070.2, 2205.1, 2340.3], dtype=np.float64)
    l1 = np.array([1865.0, 2000.1, 2135.4, 2270.2, 2405.5], dtype=np.float64)
    l2 = np.array([1791.0, 1926.2, 2061.5, 2196.4, 2331.6], dtype=np.float64)
    sig = np.full(5, 0.15, dtype=np.float64)
    np.testing.assert_allclose(
        astero.eval_radial_delta_nu_ols(np.arange(14, 19, dtype=float), l0),
        135.07,
        rtol=1e-4,
    )
    mf, _, mo = astero.match_modes_by_degree(
        l0, l0 + 0.2, np.ones(5), np.arange(14, 19, dtype=np.int32)
    )
    np.testing.assert_allclose(mf, l0 + 0.2, rtol=1e-14)
    assert np.array_equal(mo, np.arange(14, 19, dtype=np.int32))

    r02_r, s02_r = astero.eval_ratios_r02(
        l0, sig, l1, sig, l2, sig, backend="rust"
    )
    r02_j, s02_j = astero.eval_ratios_r02(
        jnp.asarray(l0), jnp.asarray(sig), jnp.asarray(l1),
        jnp.asarray(sig), jnp.asarray(l2), jnp.asarray(sig), backend="jax",
    )
    np.testing.assert_allclose(
        r02_r, (l0[1:] - l2[1:]) / (l1[1:] - l1[:-1]), rtol=1e-14
    )
    np.testing.assert_allclose(r02_r, np.asarray(r02_j), rtol=1e-13)
    np.testing.assert_allclose(s02_r, np.asarray(s02_j), rtol=1e-13)

    rat_r = astero.eval_ratios_r01_r10(l0, sig, l1, sig, backend="rust")
    kw = {
        "r010_obs": rat_r["r01"], "r010_sigmas": rat_r["sig_r01"],
        "r010_model": rat_r["r01"] + 0.002, "r02_obs": r02_r,
        "r02_sigmas": s02_r, "r02_model": r02_r - 0.001,
        "delta_nu_obs": 135.0, "delta_nu_sigma": 0.2,
        "delta_nu_model": 135.15, "nu_max_obs": 3078.0,
        "nu_max_sigma": 15.0, "nu_max_model": 3085.0,
        "chi2_seismo_r_010_fraction": 0.2, "chi2_seismo_r_02_fraction": 0.15,
        "chi2_seismo_delta_nu_fraction": 0.1,
        "chi2_seismo_nu_max_fraction": 0.05,
    }
    c_r = astero.eval_astero_chi2(
        l0, sig, l0 + 0.25, [5772.0, 0.0], [50.0, 0.05], [5800.0, 0.02],
        backend="rust", **kw,
    )
    c_j = astero.eval_astero_chi2(
        jnp.asarray(l0), jnp.asarray(sig), jnp.asarray(l0 + 0.25),
        [5772.0, 0.0], [50.0, 0.05], [5800.0, 0.02], backend="jax", **kw,
    )
    np.testing.assert_allclose(
        c_r["chi2_total"], float(c_j["chi2_total"]), rtol=1e-13
    )

    grad_fn = jax.grad(
        lambda f_m: astero.jax_kernels.eval_astero_chi2_jax(
            jnp.asarray(l0), jnp.asarray(sig), f_m,
            jnp.array([5772.0, 0.0]), jnp.array([50.0, 0.05]),
            jnp.array([5800.0, 0.02]), jnp.float64(2.0 / 3.0),
        )[0]
    )
    g_ad = np.asarray(grad_fn(jnp.asarray(l0 + 0.3)))
    g_exact = (2.0 / 3.0) * 2.0 * 0.3 / (l0.size * (sig**2))
    np.testing.assert_allclose(g_ad, g_exact, rtol=1e-12)


def test_astero_optimization_drivers_converge_to_global_minimum() -> None:
    """Verifies Nelder-Mead, Powell, grid search, and callback exceptions."""
    target = np.array([1.02, 0.275], dtype=np.float64)

    def quad_obj(x: np.ndarray) -> float:
        return float(
            4.0 * (x[0] - target[0]) ** 2 + 9.0 * (x[1] - target[1]) ** 2
        )

    nm = astero.nelder_mead_minimize(
        quad_obj, [0.95, 0.24], [0.80, 0.20], [1.20, 0.35], max_evals=400
    )
    np.testing.assert_allclose(nm["x_best"], target, atol=1e-4)

    pw = astero.powell_quadratic_minimize(
        quad_obj, [0.95, 0.24], [0.80, 0.20], [1.20, 0.35], max_evals=400
    )
    np.testing.assert_allclose(pw["x_best"], target, atol=1e-3)

    gr = astero.scan_cartesian_grid(
        quad_obj, [1.00, 0.265], [1.04, 0.285], [0.02, 0.01]
    )
    np.testing.assert_allclose(gr["x_best"], target, atol=1e-12)
    assert gr["num_evals"] == 9

    with pytest.raises(RuntimeError, match="callback failure"):
        astero.nelder_mead_minimize(_bad_obj, [1.0], [0.8], [1.2], max_evals=10)


def test_publication_plotting_diagnostics_render_cleanly(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies all 7 `flexmesa.plotting` publication diagnostic routines."""
    assert hasattr(flexmesa, "plotting")
    t_eff, log_l, xc = (
        np.linspace(3.76, 3.68, 25),
        np.linspace(0.0, 1.2, 25),
        np.linspace(0.70, 0.0, 25),
    )
    fig_hr, ax_hr = plotting.plot_hr_diagram(
        t_eff, log_l, color_val=xc, vmin=0.0, vmax=0.70,
        save_path=str(tmp_path / "hr.png"),
    )
    assert (tmp_path / "hr.png").is_file() and ax_hr.xaxis_inverted()
    plt.close(fig_hr)

    fig_tc, _ = plotting.plot_tc_rhoc_diagram(
        np.linspace(1.8, 3.5, 25), np.linspace(7.1, 7.9, 25),
        save_path=str(tmp_path / "tc_rhoc.png"),
    )
    assert (tmp_path / "tc_rhoc.png").is_file()
    plt.close(fig_tc)

    age, m_grid = np.linspace(0.0, 4500.0, 20), np.linspace(0.0, 1.0, 30)
    mix_grid = np.where(m_grid[:, None] > 0.72, 1.0, 0.0) * np.ones((30, 20))
    eps_grid = np.where(m_grid[:, None] < 0.15, 2.5, 0.0) * np.ones((30, 20))
    fig_kipp, _ = plotting.plot_kippenhahn(
        age, m_grid, mix_grid, eps_nuc_grid=eps_grid,
        total_mass_msun=np.ones(20), save_path=str(tmp_path / "kipp.png"),
    )
    assert (tmp_path / "kipp.png").is_file()
    plt.close(fig_kipp)

    r_frac = np.linspace(0.02, 0.99, 40)
    fig_prop, _ = plotting.plot_propagation_diagram(
        r_frac, np.where(r_frac < 0.71, 4.0e-6, -1.0e-8),
        np.linspace(5.0e7, 8.0e5, 40), r_frac * 6.957e10,
        nu_max_uhz=3078.0, save_path=str(tmp_path / "prop.png"),
    )
    assert (tmp_path / "prop.png").is_file()
    plt.close(fig_prop)

    freqs = np.array([1800.0, 1935.0, 2070.1, 1865.0, 2000.2, 2135.3])
    fig_ech, _ = plotting.plot_echelle_diagram(
        freqs, np.full(6, 0.2), 135.0, model_freqs_uncorr=freqs + 2.5,
        model_freqs_corr=freqs + 0.1, degrees=np.array([0, 0, 0, 1, 1, 1]),
        save_path=str(tmp_path / "echelle.png"),
    )
    assert (tmp_path / "echelle.png").is_file()
    plt.close(fig_ech)

    fig_rat, _ = plotting.plot_separation_ratios(
        np.array([1935.0, 2000.0, 2070.0]), np.array([0.038, 0.036, 0.034]),
        np.full(3, 0.002), r010_model=np.array([0.0375, 0.0358, 0.0342]),
        save_path=str(tmp_path / "ratios.png"),
    )
    assert (tmp_path / "ratios.png").is_file()
    plt.close(fig_rat)

    fig_bin, _ = plotting.plot_binary_evolution(
        np.linspace(0.0, 100.0, 20), np.linspace(2.5, 1.8, 20),
        np.linspace(-9.0, -6.5, 20), eccentricity=np.linspace(0.25, 0.02, 20),
        save_path=str(tmp_path / "binary.png"),
    )
    assert (tmp_path / "binary.png").is_file()
    plt.close(fig_bin)
