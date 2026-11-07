# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generates publication-styled Wave 3a EOS diagnostic figures for FlexMESA.

Produces two 300 dpi PNG figures in `docs/assets/figures/` following the
AASTeX publication plot style (Computer Modern typography, inward ticks on all
four sides, frameless legends, and journal-derived widths):
1. `eos_helm_skye_phase_diagram.png`: $(\\log_{10} \\rho, \\log_{10} T)$ EOS
   blending regime map and Coulomb crystallization transition $\\Gamma_i \\in
   [170, 180]$ for a $\\mathrm{C}^{12}/\\mathrm{O}^{16}$ white dwarf mixture
   (left), paired with interior $\\Gamma_1$ and crystallization phase profiles
   comparing `HELM` and `Skye` across Safe Rust and JAX (right).
2. `wave3a_benchmark_staircase.png`: Four-way Wave 3a.5 runtime comparison at
   $N_{\\mathrm{zones}} = 1024$ across Upstream Fortran MESA, Safe Rust,
   Python/PyO3 (`flexmesa._rust`), and Python/JAX (`@jax.jit(vmap)`) (left),
   paired with `Skye` vs. `PC` vs. `HELM` non-ideal Coulomb pressure profiles
   across density (right).
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure

from flexmesa import eos as _eos

PT_PER_INCH = 72.27
TEXT_WIDTH = 513.11743 / PT_PER_INCH


def _two_panel_fig(wspace: float = 0.36) -> tuple[figure.Figure, np.ndarray]:
    """Creates a two-panel text-width publication figure."""
    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.38 * TEXT_WIDTH))
    fig.subplots_adjust(wspace=wspace)
    return fig, ax


def _save_fig(fig: figure.Figure, path: pathlib.Path) -> None:
    """Saves `fig` as a tight 300 dpi PNG and closes it."""
    fig.savefig(str(path), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_eos_helm_skye_phase_diagram(out_dir: pathlib.Path) -> None:
    """Plots the $(\\log_{10}\\rho, \\log_{10}T)$ EOS blend map and WD profile.

    Args:
        out_dir: Target directory for `eos_helm_skye_phase_diagram.png`.
    """
    log10_rho_g = np.linspace(-6.0, 9.0, 72, dtype=np.float64)
    log10_t_g = np.linspace(5.0, 9.0, 72, dtype=np.float64)
    rr, tt = np.meshgrid(log10_rho_g, log10_t_g)
    alfa_pc = np.zeros_like(rr)
    for j in range(rr.shape[0]):
        for i in range(rr.shape[1]):
            alfa, _, _ = _eos.get_pc_alfa(
                float(rr[j, i]),
                float(tt[j, i]),
                13.714285714285714,
                6.857142857142857,
            )
            alfa_pc[j, i] = float(alfa)

    # Crystallization boundary Gamma_i = 175 for 50/50 C12 + O16:
    # Gamma_i = 2.2747e5 * <Z^(5/3)> * (rho * Y_e)^(1/3) / T
    z53_bar = 0.5 * (6.0 ** (5.0 / 3.0) + 8.0 ** (5.0 / 3.0))
    ye_co = 0.5
    rho_cryst = np.power(10.0, log10_rho_g)
    t_gamma_170 = 2.2747e5 * z53_bar * np.cbrt(rho_cryst * ye_co) / 170.0
    t_gamma_180 = 2.2747e5 * z53_bar * np.cbrt(rho_cryst * ye_co) / 180.0

    fig, ax = _two_panel_fig(wspace=0.42)
    cf = ax[0].contourf(
        rr,
        tt,
        1.0 - alfa_pc,
        levels=np.linspace(0.0, 1.0, 11),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(cf, ax=ax[0], pad=0.02)
    cbar.set_label(r"Coulomb Blend Weight [$1 - \alpha_{\mathrm{HELM}}$]")
    ax[0].plot(
        log10_rho_g,
        np.log10(t_gamma_170),
        color="C3",
        lw=1.6,
        ls="--",
        label=r"Melting ($\Gamma_i = 170$)",
    )
    ax[0].plot(
        log10_rho_g,
        np.log10(t_gamma_180),
        color="C1",
        lw=1.6,
        ls="-.",
        label=r"Freezing ($\Gamma_i = 180$)",
    )
    ax[0].set_xlim(-6.0, 9.0)
    ax[0].set_ylim(5.0, 9.0)
    ax[0].set_xlabel(r"Density [$\log_{10}(\rho/\mathrm{g\,cm^{-3}})$]")
    ax[0].set_ylabel(r"Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[0].legend(frameon=False, loc="upper left", fontsize=8)

    # Right panel: C12/O16 white dwarf core-to-envelope profile at T = 10^7.12 K
    log10_rho = np.linspace(5.8, 8.8, 48, dtype=np.float64)
    rho_arr = np.power(10.0, log10_rho)
    t_arr = np.full_like(rho_arr, 10.0**7.12)
    abar_arr = np.full_like(rho_arr, 13.714285714285714)
    zbar_arr = np.full_like(rho_arr, 6.857142857142857)
    z_co = [6, 8]
    azion_co = np.array([6.0, 8.0], dtype=np.float64)
    acmi_co = np.array([12.0, 15.99491461957], dtype=np.float64)
    a_co = np.array([12.0, 16.0], dtype=np.float64)
    xa_2d = np.tile(np.array([0.50, 0.50], dtype=np.float64), (48, 1))

    pc_r = np.asarray(
        _eos.eval_pc_eos_zones(
            rho_arr,
            t_arr,
            abar_arr,
            zbar_arr,
            xa_2d,
            azion_co,
            acmi_co,
            backend="rust",
        ).res,
        dtype=np.float64,
    )
    skye_r = np.asarray(
        _eos.eval_skye_eos_zones(
            rho_arr,
            t_arr,
            abar_arr,
            zbar_arr,
            xa_2d,
            z_co,
            acmi_co,
            a_co,
            backend="rust",
        ).res,
        dtype=np.float64,
    )
    skye_j = np.asarray(
        _eos.eval_skye_eos_zones(
            rho_arr[::6],
            t_arr[::6],
            abar_arr[::6],
            zbar_arr[::6],
            xa_2d[::6],
            z_co,
            acmi_co,
            a_co,
            backend="jax",
        ).res,
        dtype=np.float64,
    )

    ax[1].plot(
        log10_rho,
        pc_r[:, _eos.I_GAMMA1],
        color="C2",
        lw=1.6,
        ls="--",
        label=r"PC $\Gamma_1$ (Safe Rust)",
    )
    ax[1].plot(
        log10_rho,
        skye_r[:, _eos.I_GAMMA1],
        color="C0",
        lw=1.8,
        label=r"Skye $\Gamma_1$ (Safe Rust)",
    )
    ax[1].scatter(
        log10_rho[::6],
        skye_j[:, _eos.I_GAMMA1],
        color="C3",
        s=18,
        zorder=5,
        label=r"Skye $\Gamma_1$ (JAX $\mathtt{@jit}$)",
    )
    ax[1].plot(
        log10_rho,
        1.34 + 0.14 * skye_r[:, _eos.I_PHASE],
        color="C1",
        lw=1.4,
        ls=":",
        label=r"$1.34 + 0.14\,\phi_{\mathrm{cryst}}$",
    )
    ax[1].set_xlim(5.8, 8.8)
    ax[1].set_ylim(1.32, 1.64)
    ax[1].set_xlabel(r"Density [$\log_{10}(\rho/\mathrm{g\,cm^{-3}})$]")
    ax[1].set_ylabel(r"Adiabatic Exponent [$\Gamma_1$]")
    ax[1].legend(frameon=False, loc="upper right", fontsize=8)
    _save_fig(fig, out_dir / "eos_helm_skye_phase_diagram.png")


def plot_wave3a_benchmark_staircase(out_dir: pathlib.Path) -> None:
    """Plots the 4-way Wave 3a.5 runtime comparison and Coulomb corrections.

    Args:
        out_dir: Target directory for `wave3a_benchmark_staircase.png`.
    """
    workloads = ["eos_ideal", "eos_helm", "eos_pc", "eos_skye"]
    x_pos = np.arange(len(workloads), dtype=np.float64)
    w_bar = 0.19
    fortran_4t = np.array([349.01, 204.05, 1170.70, 2931.65])
    rust_4t = np.array([198.83, 85.72, 633.67, 2329.83])
    pyo3_4t = np.array([239.28, 160.92, 634.03, 1586.10])
    jax_vmap = np.array([327.32, 328.61, 1163.75, 605.55])

    fig, ax = _two_panel_fig(wspace=0.42)
    bars = (
        (-1.5, fortran_4t, "C2", 0.85, "Fortran MESA (4t)"),
        (-0.5, rust_4t, "C0", 0.95, "Safe Rust (4t)"),
        (0.5, pyo3_4t, "C1", 0.90, "PyO3 (4t)"),
        (1.5, jax_vmap, "C3", 0.85, "JAX (vmap)"),
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
    ax[0].set_xticklabels(workloads, fontsize=8.5)
    ax[0].set_ylim(1.5, 3.85)
    ax[0].set_xlabel(r"Wave 3a Workload ($N_{\mathrm{zones}} = 1024$)")
    ax[0].set_ylabel(r"Latency [$\log_{10}(t / \mu\mathrm{s})$]")
    ax[0].legend(frameon=False, loc="upper left", fontsize=8)

    log10_rho = np.linspace(2.0, 7.5, 48, dtype=np.float64)
    rho_arr = np.power(10.0, log10_rho)
    t_arr = np.full_like(rho_arr, 10.0**8.0)
    abar_arr = np.full_like(rho_arr, 12.0)
    zbar_arr = np.full_like(rho_arr, 6.0)
    xa_2d = np.ones((48, 1), dtype=np.float64)
    azion_c = np.array([6.0], dtype=np.float64)
    acmi_c = np.array([12.0], dtype=np.float64)
    a_c = np.array([12.0], dtype=np.float64)

    helm_p = np.asarray(
        _eos.eval_helm_eos_zones(
            t_arr, rho_arr, abar_arr, zbar_arr, backend="rust"
        ).res,
        dtype=np.float64,
    )[:, _eos.I_LN_PGAS]
    pc_p = np.asarray(
        _eos.eval_pc_eos_zones(
            rho_arr,
            t_arr,
            abar_arr,
            zbar_arr,
            xa_2d,
            azion_c,
            acmi_c,
            backend="rust",
        ).res,
        dtype=np.float64,
    )[:, _eos.I_LN_PGAS]
    skye_p = np.asarray(
        _eos.eval_skye_eos_zones(
            rho_arr,
            t_arr,
            abar_arr,
            zbar_arr,
            xa_2d,
            [6],
            acmi_c,
            a_c,
            backend="rust",
        ).res,
        dtype=np.float64,
    )[:, _eos.I_LN_PGAS]

    ax[1].axhline(0.0, color="k", ls=":", lw=0.9, alpha=0.6)
    ax[1].plot(
        log10_rho,
        pc_p - helm_p,
        color="C1",
        lw=1.7,
        ls="--",
        label=r"PC $-$ HELM ($\mathrm{C}^{12}$)",
    )
    ax[1].plot(
        log10_rho,
        skye_p - helm_p,
        color="C0",
        lw=1.8,
        label=r"Skye $-$ HELM ($\mathrm{C}^{12}$)",
    )
    ax[1].set_xlim(2.0, 7.5)
    ax[1].set_xlabel(r"Density [$\log_{10}(\rho/\mathrm{g\,cm^{-3}})$]")
    ax[1].set_ylabel(r"Coulomb Shift [$\Delta \ln P_{\mathrm{gas}}$]")
    ax[1].legend(frameon=False, loc="lower left", fontsize=8.5)
    _save_fig(fig, out_dir / "wave3a_benchmark_staircase.png")
