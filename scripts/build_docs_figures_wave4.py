# SPDX-License-Identifier: GPL-3.0-or-later
"""Generates publication-styled Wave 4 stellar structure figures for FlexMESA.

Produces two 300 dpi PNG figures in `docs/assets/figures/` following the
AASTeX publication plot style (Computer Modern typography, inward ticks on all
four sides, frameless legends, and journal-derived widths):
1. `wave4_star_structure_evolution.png`: Radial radiative, adiabatic, and
   actual temperature gradients ($\\nabla_r, \\nabla_{\\mathrm{ad}}, \\nabla_T$)
   across enclosed mass fraction $q = m/M_\\star$ in the
   $4.0\\,M_\\odot, Z = 0.02$ ZAMS model comparing Safe Rust and JAX (left),
   paired with the 8-mass $Z = 0.02$ ZAMS Hertzsprung-Russell diagram ($0.51$
   to $15.0\\,M_\\odot$) computed by `FlexMESA` (right).
2. `wave4_benchmark_staircase.png`: MESA `26.4.1` Wave 4 runtime comparison at
   $N_{\\mathrm{zones}} = 1024$ loaded from `benchmarks/raw/26.4.1/` across
   MESA `26.4.1` Fortran (`need_to_setvars = .true.` post-`51e1adc`;
   `star_evolve_step` uses `dt = 100 s`), Safe Rust, Python/PyO3 (`Stage 0`),
   and Python/JAX (left), paired with `star_evolve_step` scaling across
   $N_{\\mathrm{zones}} \\in \\{128, 512, 1024\\}$ (right).
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from docs import raw_benchmarks as _raw_bench
from matplotlib import figure

from flexmesa import star

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


def plot_wave4_star_structure_evolution(out_dir: pathlib.Path) -> None:
    """Plot Wave 4 internal convective structure and ZAMS HR diagram."""
    _configure_plot_style()
    fig, ax = _two_panel_fig(wspace=0.40)

    model = star.StarModel.from_zams(4.0, 0.02)
    q = model.q
    gradr = np.minimum(model.gradr, 1.15)
    grada = model.grada
    gradt_r = model.gradt
    gradt_j = star.eval_face_mlt_and_brunt_jax(
        jnp.asarray(model.dq),
        jnp.asarray(model.ln_r),
        jnp.asarray(model.lum),
        jnp.asarray(model.m),
        jnp.asarray(model.rho),
        jnp.asarray(model.t),
        jnp.asarray(model.peos),
        jnp.asarray(model.ln_peos),
        jnp.asarray(model.grada),
        jnp.asarray(model.cp),
        jnp.asarray(model.chi_rho),
        jnp.asarray(model.chi_t),
        jnp.asarray(model.mu),
        jnp.asarray(model.opacity),
    )["gradt"]
    cz_core_mask = (model.gradr > model.grada) & (q < 0.5)
    q_conv_edge = (
        float(np.max(q[cz_core_mask])) if np.any(cz_core_mask) else 0.22
    )
    q_targets = np.linspace(0.02, 0.96, 22)
    idx_sub = np.array(
        [int(np.argmin(np.abs(q - qt))) for qt in q_targets], dtype=np.int64
    )

    ax[0].axvspan(
        0.0,
        q_conv_edge,
        color="C1",
        alpha=0.18,
        label="Convective Core",
    )
    ax[0].plot(
        q,
        gradr,
        color="C2",
        lw=1.3,
        ls="--",
        label=r"$\nabla_{\mathrm{rad}}$",
    )
    ax[0].plot(
        q,
        grada,
        color="C1",
        lw=1.4,
        ls=":",
        label=r"$\nabla_{\mathrm{ad}}$",
    )
    ax[0].plot(
        q,
        gradt_r,
        color="C0",
        lw=1.9,
        label=r"$\nabla_T$ (Safe Rust)",
    )
    ax[0].scatter(
        q[idx_sub],
        np.asarray(gradt_j)[idx_sub],
        color="C3",
        s=16,
        zorder=5,
        label=r"$\nabla_T$ (JAX $\mathtt{jit}$)",
    )
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(0.12, 0.92)
    ax[0].set_xlabel(r"Enclosed Mass Fraction [$q = m / M_\star$]")
    ax[0].set_ylabel(r"Temperature Gradient [$\nabla = d\ln T / d\ln P$]")
    ax[0].legend(frameon=False, loc="upper right")

    masses = (0.51, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0)
    lg_teff = np.zeros(len(masses), dtype=np.float64)
    lg_lum = np.zeros(len(masses), dtype=np.float64)
    for i, m_val in enumerate(masses):
        m_star = star.StarModel.from_zams(m_val, 0.02)
        h0 = m_star.history_row()
        lg_teff[i] = h0.log_teff
        lg_lum[i] = h0.log_l

    ax[1].plot(
        lg_teff,
        lg_lum,
        color="C0",
        lw=1.8,
        marker="o",
        ms=4.5,
        label="FlexMESA ZAMS (Z = 0.02)",
    )
    for m_lbl, xt, yt, dx, dy in (
        ("0.51", lg_teff[0], lg_lum[0], 0.14, 0.55),
        ("1.0", lg_teff[1], lg_lum[1], -0.04, 0.35),
        ("4.0", lg_teff[4], lg_lum[4], -0.05, 0.28),
        ("15", lg_teff[-1], lg_lum[-1], -0.10, -0.08),
    ):
        ax[1].annotate(
            rf"${m_lbl}\,M_\odot$",
            xy=(xt, yt),
            xytext=(xt + dx, yt + dy),
            fontsize=8.0,
        )
    ax[1].set_xlim(4.60, 3.48)
    ax[1].set_ylim(-2.1, 4.9)
    ax[1].set_xlabel(
        r"Effective Temperature [$\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$]"
    )
    ax[1].set_ylabel(r"Luminosity [$\log_{10}(L / L_\odot)$]")
    ax[1].legend(frameon=False, loc="lower left")
    _save_fig(fig, out_dir / "wave4_star_structure_evolution.png")


def plot_wave4_benchmark_staircase(out_dir: pathlib.Path) -> None:
    """Plot MESA 26.4.1 Wave 4 runtime comparison and step scaling (`#146`)."""
    _configure_plot_style()
    raw = _raw_bench.load_raw_benchmarks()
    workloads = [
        "micro_vars",
        "hydro_eqns",
        "newton_step",
        "remesh",
        "evolve_step",
    ]
    keys = [
        "star_micro_vars",
        "star_hydro_eqns",
        "star_newton_step",
        "star_remesh",
        "star_evolve_step",
    ]
    x_pos = np.arange(len(workloads), dtype=np.float64)
    w_bar = 0.18
    fortran_4t = np.array(
        [raw.get("fortran", w, 1024).stage3.median_us for w in keys]
    )
    rust_4t = np.array(
        [raw.get("rust", w, 1024).stage3.median_us for w in keys]
    )
    pyo3_best = np.array(
        [raw.get("python", w, 1024).stage0.median_us for w in keys]
    )
    jax_best = np.array(
        [raw.get("python", w, 1024).stage3.median_us for w in keys]
    )

    fig, ax = _two_panel_fig(wspace=0.40)
    bars = (
        (-1.5, fortran_4t, "C2", 0.85, "MESA 26.4.1 Fortran (4t)"),
        (-0.5, rust_4t, "C0", 0.95, "Safe Rust (4t rayon)"),
        (0.5, pyo3_best, "C1", 0.90, "PyO3 / Eager Python (Stage 0)"),
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
    ax[0].set_ylim(0.0, 6.6)
    ax[0].set_xlabel(r"Wave 4 Workload ($N_{\mathrm{zones}} = 1024$)")
    ax[0].set_ylabel(r"Latency [$\log_{10}(t / \mu\mathrm{s})$]")
    ax[0].legend(frameon=False, loc="upper left")

    nz_vals = (128, 512, 1024)
    nz_arr = np.asarray(nz_vals, dtype=np.float64)
    f90_1t_ms = np.array(
        [
            raw.get("fortran", "star_evolve_step", nz).stage2.median_us / 1.0e3
            for nz in nz_vals
        ]
    )
    f90_4t_ms = np.array(
        [
            raw.get("fortran", "star_evolve_step", nz).stage3.median_us / 1.0e3
            for nz in nz_vals
        ]
    )
    rust_1t_ms = np.array(
        [
            raw.get("rust", "star_evolve_step", nz).stage2.median_us / 1.0e3
            for nz in nz_vals
        ]
    )
    rust_4t_ms = np.array(
        [
            raw.get("rust", "star_evolve_step", nz).stage3.median_us / 1.0e3
            for nz in nz_vals
        ]
    )

    ax[1].plot(
        nz_arr,
        np.log10(f90_1t_ms),
        color="C2",
        lw=1.5,
        ls="--",
        marker="s",
        label="MESA 26.4.1 Fortran (1t)",
    )
    ax[1].plot(
        nz_arr,
        np.log10(f90_4t_ms),
        color="C2",
        lw=1.9,
        marker="o",
        label="MESA 26.4.1 Fortran (4t OpenMP)",
    )
    ax[1].plot(
        nz_arr,
        np.log10(rust_1t_ms),
        color="C0",
        lw=1.5,
        ls="--",
        marker="^",
        label="Safe Rust (1t)",
    )
    ax[1].plot(
        nz_arr,
        np.log10(rust_4t_ms),
        color="C0",
        lw=2.0,
        marker="o",
        label="Safe Rust (4t rayon)",
    )
    ax[1].set_xticks([128, 512, 1024])
    ax[1].set_ylim(0.8, 3.65)
    ax[1].set_xlabel(r"Radial Zone Count [$N_{\mathrm{zones}}$]")
    ax[1].set_ylabel(r"Coupled Step Latency [$\log_{10}(t / \mathrm{ms})$]")
    ax[1].legend(frameon=False, loc="upper left")
    _save_fig(fig, out_dir / "wave4_benchmark_staircase.png")


def main() -> None:
    """Build all Wave 4 documentation figures."""
    root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_wave4_star_structure_evolution(out_dir)
    plot_wave4_benchmark_staircase(out_dir)


if __name__ == "__main__":
    main()
