# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generates publication-styled Wave 5 diagnostic figures for FlexMESA.

Produces three 300 dpi PNG figures in `docs/assets/figures/` following the
AASTeX publication plot style (Computer Modern typography, inward ticks on all
four sides, frameless legends, and journal-derived widths):
1. `wave5_benchmark_speedup.png`: Four-way Wave 5.5 runtime comparison at
   $N_{\\mathrm{zones}} = 1024$ across Upstream Fortran MESA, Safe Rust,
   Python/PyO3 (`flexmesa._rust`), and Python/JAX (left), paired with scaling
   across $N_{\\mathrm{zones}} \\in \\{128, 512, 1024\\}$ for `w5_diff_rot` and
   `w5_riemann_rsp2` (right).
2. `wave5_astero_echelle_surface_corr.png`: Asteroseismic echelle diagram
   ($\\nu \\bmod \\Delta\\nu$ vs. $\\nu$) illustrating Ball & Gizon (2014)
   1-term cubic, 2-term combined, and Sonoi (2015) surface-effect frequency
   corrections (left), paired with Roxburgh & Vorontsov (2003) 5-point
   separation ratios ($r_{01}, r_{10}$) and quadrupole ratios ($r_{02}$)
   comparing Safe Rust and JAX (right).
3. `wave5_binary_rlof_evolution.png`: Ritter (1988) optically thin
   mass-transfer rate $\\log_{10}(|\\dot{M}_{\\mathrm{RLOF}}|)$ across relative
   Roche-lobe overflow gap $(R_1 - R_{\\mathrm{RL},1}) / R_{\\mathrm{RL},1}$
   comparing Safe Rust and JAX (left), paired with Kerr black-hole spin $a_*$
   and radiative efficiency $\\eta_{\\mathrm{ISCO}}$ evolution under Bardeen
   (1972) accretion (right).
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure

from flexmesa import _rust, astero, binary, constants

PT_PER_INCH = 72.27
TEXT_WIDTH = 513.11743 / PT_PER_INCH


def _configure_plot_style() -> None:
    """Apply AASTeX publication matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth": 0.8,
        }
    )


def _two_panel_fig(wspace: float = 0.38) -> tuple[figure.Figure, np.ndarray]:
    """Create a two-panel text-width publication figure."""
    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.38 * TEXT_WIDTH))
    fig.subplots_adjust(wspace=wspace)
    return fig, ax


def _save_fig(fig: figure.Figure, path: pathlib.Path) -> None:
    """Save `fig` as a tight 300 dpi PNG and close it."""
    fig.savefig(str(path), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_wave5_benchmark_speedup(out_dir: pathlib.Path) -> None:
    """Plot 4-way Wave 5.5 runtime comparison and zone-count scaling."""
    _configure_plot_style()
    workloads = ["diff_rot", "riemann_rsp2", "binary_rlof", "astero_seism"]
    x_pos = np.arange(len(workloads), dtype=np.float64)
    w_bar = 0.18
    fortran_best = np.array([91.83, 46.46, 28.29, 5.83])
    rust_best = np.array([70.41, 7.94, 57.79, 7.25])
    pyo3_best = np.array([49.62, 33.29, 60.69, 7.46])
    jax_best = np.array([120.91, 89.90, 101.77, 26.75])

    fig, ax = _two_panel_fig(wspace=0.40)
    bars = (
        (-1.5, fortran_best, "C2", 0.85, "Fortran MESA"),
        (-0.5, rust_best, "C0", 0.95, "Safe Rust"),
        (0.5, pyo3_best, "C1", 0.90, "PyO3 (flexmesa._rust)"),
        (1.5, jax_best, "C3", 0.85, "JAX (jit / vmap)"),
    )
    for mult, arr, col, alp, lbl in bars:
        ax[0].bar(
            x_pos + mult * w_bar,
            np.log10(arr),
            width=w_bar,
            color=col,
            alpha=alp,
            label=lbl,
        )
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(workloads, fontsize=7.5, rotation=12)
    ax[0].set_ylim(0.0, 3.20)
    ax[0].set_xlabel(r"Wave 5 Workload ($N_{\mathrm{zones}} = 1024$)")
    ax[0].set_ylabel(r"Latency [$\log_{10}(t / \mu\mathrm{s})$]")
    ax[0].legend(frameon=False, loc="upper right", fontsize=7.5)

    nz_arr = np.array([128.0, 512.0, 1024.0])
    scaling_series = (
        (
            [4.68, 22.54, 46.46],
            "C2",
            1.5,
            "--",
            "s",
            r"Fortran $\mathtt{riemann\_rsp2}$ (4t)",
        ),
        (
            [3.53, 14.05, 28.15],
            "C0",
            1.5,
            ":",
            "^",
            r"Rust $\mathtt{riemann\_rsp2}$ (1t)",
        ),
        (
            [1.20, 4.04, 7.94],
            "C0",
            2.0,
            "-",
            "o",
            r"Rust $\mathtt{riemann\_rsp2}$ (4t)",
        ),
        (
            [6.07, 34.91, 91.83],
            "C1",
            1.5,
            "--",
            "d",
            r"Fortran $\mathtt{diff\_rot}$ (4t)",
        ),
        (
            [6.12, 24.09, 70.41],
            "C3",
            1.8,
            "-",
            "x",
            r"Rust $\mathtt{diff\_rot}$ (4t)",
        ),
    )
    for vals, col, lw, ls, mk, lbl in scaling_series:
        ax[1].plot(
            nz_arr,
            np.log10(np.asarray(vals)),
            color=col,
            lw=lw,
            ls=ls,
            marker=mk,
            label=lbl,
        )
    ax[1].set_xticks([128, 512, 1024])
    ax[1].set_ylim(-0.05, 2.55)
    ax[1].set_xlabel(r"Radial Zone Count [$N_{\mathrm{zones}}$]")
    ax[1].set_ylabel(r"Kernel Latency [$\log_{10}(t / \mu\mathrm{s})$]")
    ax[1].legend(frameon=False, loc="lower right", fontsize=7.0)
    _save_fig(fig, out_dir / "wave5_benchmark_speedup.png")


def plot_wave5_astero_echelle_surface_corr(out_dir: pathlib.Path) -> None:
    """Plot asteroseismic echelle diagram and separation ratios."""
    _configure_plot_style()
    fig, ax = _two_panel_fig(wspace=0.40)

    dnu = 135.0
    n_ord = np.arange(14, 26, dtype=np.float64)
    l0_mod = dnu * (n_ord + 1.45) + 0.35 * np.sin(0.45 * n_ord)
    l0_ine = 1.0e-7 * np.exp((3100.0 - l0_mod) / 1000.0)
    l0_sig = np.full_like(l0_mod, 0.15)
    b1 = -5.125_972_793_567_201_1e-5
    b3 = -1.446_307_489_879_336_1e-18
    l0_obs = l0_mod + (b1 / l0_mod + b3 * (l0_mod**3)) / l0_ine

    res_cub = astero.eval_surface_correction(
        "cubic", l0_obs, l0_sig, l0_mod, l0_ine, nu_max=3090.0
    )
    res_comb = astero.eval_surface_correction(
        "combined", l0_obs, l0_sig, l0_mod, l0_ine, nu_max=3090.0
    )
    res_sonoi = astero.eval_surface_correction(
        "sonoi", l0_obs, l0_sig, l0_mod, l0_ine, nu_max=3090.0
    )

    echelle_lines = (
        (l0_mod, "0.55", 1.3, "--", "s", "Uncorrected Model"),
        (res_cub["nu_corrected"], "C1", 1.4, ":", None, "BG14 Cubic"),
        (res_sonoi["nu_corrected"], "C2", 1.4, "-.", None, "Sonoi (2015)"),
        (res_comb["nu_corrected"], "C0", 1.9, "-", None, "BG14 Combined"),
    )
    for nu_arr, col, lw, ls, mk, lbl in echelle_lines:
        ax[0].plot(
            np.mod(nu_arr, dnu),
            l0_mod,
            color=col,
            lw=lw,
            ls=ls,
            marker=mk,
            ms=3.8,
            label=lbl,
        )
    ax[0].scatter(
        np.mod(l0_obs, dnu),
        l0_mod,
        color="C3",
        s=20,
        zorder=5,
        label="Observed ($l=0$)",
    )
    ax[0].set_xlim(56.5, 62.2)
    ax[0].set_ylim(2000.0, 3850.0)
    ax[0].set_xlabel(
        r"Reduced Frequency [$\nu\;\mathrm{mod}\;\Delta\nu\ (\mu\mathrm{Hz})$]"
    )
    ax[0].set_ylabel(r"Mode Frequency [$\nu\ (\mu\mathrm{Hz})$]")
    ax[0].legend(frameon=False, loc="lower left", fontsize=7.2)

    l1_mod = dnu * (n_ord + 1.91) - 2.4 * np.sin(0.32 * n_ord)
    l1_sig = np.full_like(l0_mod, 0.18)
    l2_mod = l0_mod - 9.2 + 0.12 * (n_ord - 14.0)
    l2_sig = np.full_like(l0_mod, 0.22)
    rat_r = astero.eval_ratios_r01_r10(l0_mod, l0_sig, l1_mod, l1_sig)
    r02_r, _ = astero.eval_ratios_r02(
        l0_mod, l0_sig, l1_mod, l1_sig, l2_mod, l2_sig
    )
    r01_j, _, r10_j, _ = astero.jax_kernels.eval_ratios_r01_r10_jax(
        jnp.asarray(l0_mod),
        jnp.asarray(l0_sig),
        jnp.asarray(l1_mod),
        jnp.asarray(l1_sig),
    )
    r02_j, _ = astero.jax_kernels.eval_ratios_r02_jax(
        jnp.asarray(l0_mod),
        jnp.asarray(l0_sig),
        jnp.asarray(l1_mod),
        jnp.asarray(l1_sig),
        jnp.asarray(l2_mod),
        jnp.asarray(l2_sig),
    )
    nu_mid, nu_r02 = l0_mod[1:-1], l0_mod[1:]
    ratio_series = (
        (nu_mid, rat_r["r01"], r01_j, "C0", "-", "o", r"$r_{01}$ (Rust & JAX)"),
        (
            nu_mid,
            rat_r["r10"],
            r10_j,
            "C1",
            "--",
            "^",
            r"$r_{10}$ (Rust & JAX)",
        ),
        (nu_r02, r02_r, r02_j, "C2", ":", "s", r"$r_{02}$ (Rust & JAX)"),
    )
    for x_v, y_r, y_j, col, ls, mk, lbl in ratio_series:
        ax[1].plot(x_v, y_r, color=col, lw=1.7, ls=ls, label=lbl)
        ax[1].scatter(
            x_v, np.asarray(y_j), color=col, marker=mk, s=18, zorder=5
        )
    ax[1].set_xlim(2150.0, 3750.0)
    ax[1].set_ylim(0.015, 0.082)
    ax[1].set_xlabel(r"Mode Frequency [$\nu\ (\mu\mathrm{Hz})$]")
    ax[1].set_ylabel(r"Frequency Separation Ratio [$r_{01}, r_{10}, r_{02}$]")
    ax[1].legend(frameon=False, loc="upper right", fontsize=7.5)
    _save_fig(fig, out_dir / "wave5_astero_echelle_surface_corr.png")


def plot_wave5_binary_rlof_evolution(out_dir: pathlib.Path) -> None:
    """Plot RLOF mass-transfer curves and Kerr BH ISCO spin evolution."""
    _configure_plot_style()
    fig, ax = _two_panel_fig(wspace=0.40)

    m1, m2, rl1 = (
        2.0 * constants.MSUN,
        1.4 * constants.MSUN,
        2.5 * constants.RSUN,
    )
    rel_gaps = np.linspace(-0.012, 0.012, 32)
    r1_arr = rl1 * (1.0 + rel_gaps)
    mdot_r = np.array(
        [
            abs(
                float(
                    binary.eval_ritter_mdot_thin(
                        m1,
                        m2,
                        float(r1_v),
                        rl1,
                        2.0e-7,
                        2.5e5,
                        6500.0,
                        0.61,
                        is_kolb_scheme=False,
                    )
                )
            )
            / constants.MSUN
            * constants.SECYER
            for r1_v in r1_arr
        ]
    )
    ones_j = jnp.ones_like(r1_arr)
    mdot_j_arr = binary.eval_ritter_mdot_thin_jax(
        m1 * ones_j,
        m2 * ones_j,
        jnp.asarray(r1_arr),
        rl1 * ones_j,
        2.0e-7 * ones_j,
        2.5e5 * ones_j,
        6500.0 * ones_j,
        0.61 * ones_j,
        False,
    )
    mdot_j = np.abs(np.asarray(mdot_j_arr)) / constants.MSUN * constants.SECYER

    ax[0].axvline(0.0, color="0.65", lw=1.0, ls=":")
    ax[0].plot(
        rel_gaps * 1.0e2,
        np.log10(np.maximum(mdot_r, 1.0e-16)),
        color="C0",
        lw=1.9,
        label="Ritter (1988) Safe Rust",
    )
    ax[0].scatter(
        rel_gaps[::2] * 1.0e2,
        np.log10(np.maximum(mdot_j[::2], 1.0e-16)),
        color="C3",
        s=18,
        zorder=5,
        label=r"Ritter (1988) JAX $\mathtt{jit}$",
    )
    ax[0].set_xlim(-1.2, 1.2)
    ax[0].set_ylim(-13.5, -1.5)
    ax[0].set_xlabel(
        r"Relative RLOF Gap "
        r"[$100 \times (R_1 - R_{\mathrm{RL},1}) / R_{\mathrm{RL},1}$]"
    )
    ax[0].set_ylabel(
        r"Mass-Transfer Rate "
        r"[$\log_{10}(|\dot{M}_{\mathrm{RLOF}}| / M_\odot\,\mathrm{yr}^{-1})$]"
    )
    ax[0].legend(frameon=False, loc="lower right", fontsize=7.5)

    m_ratio = np.linspace(1.0, 2.45, 36)
    m_bh_init, a_init = 6.0 * constants.MSUN, 0.15
    bh_r = [
        _rust.binary_eval_bh_spin_and_edd(
            m_bh_init, a_init, float(mr) * m_bh_init, 0.34
        )
        for mr in m_ratio
    ]
    a_r = np.array([row[1] for row in bh_r])
    eta_r = np.array([row[3] for row in bh_r])
    ones_bh = jnp.ones_like(m_ratio)
    _, a_j, _, eta_j = binary.eval_bh_spin_and_edd_jax(
        m_bh_init * ones_bh,
        a_init * ones_bh,
        jnp.asarray(m_ratio * m_bh_init),
        0.34 * ones_bh,
    )
    bh_series = (
        (a_r, a_j, "C0", "C3", "-", "o", r"Kerr Spin $a_*$ (Rust & JAX)"),
        (
            eta_r,
            eta_j,
            "C1",
            "C1",
            "--",
            "^",
            r"Efficiency $\eta_{\mathrm{ISCO}}$ (Rust & JAX)",
        ),
    )
    for y_r, y_j, col_l, col_s, ls, mk, lbl in bh_series:
        ax[1].plot(m_ratio, y_r, color=col_l, lw=1.8, ls=ls, label=lbl)
        ax[1].scatter(
            m_ratio[::3],
            np.asarray(y_j)[::3],
            color=col_s,
            marker=mk,
            s=18,
            zorder=5,
        )
    ax[1].set_xlim(1.0, 2.45)
    ax[1].set_ylim(0.0, 1.08)
    ax[1].set_xlabel(
        r"Accreted BH Mass Ratio [$M_{\mathrm{BH}} / M_{\mathrm{BH},0}$]"
    )
    ax[1].set_ylabel(
        r"Dimensionless Spin $a_*$ & Efficiency $\eta_{\mathrm{ISCO}}$"
    )
    ax[1].legend(frameon=False, loc="center right", fontsize=7.2)
    _save_fig(fig, out_dir / "wave5_binary_rlof_evolution.png")


def main() -> None:
    """Build all Wave 5 documentation figures."""
    root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_wave5_benchmark_speedup(out_dir)
    plot_wave5_astero_echelle_surface_corr(out_dir)
    plot_wave5_binary_rlof_evolution(out_dir)


if __name__ == "__main__":
    main()
