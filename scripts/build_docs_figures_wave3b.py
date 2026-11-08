# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generates publication-styled Wave 3b microphysics figures for FlexMESA.

Produces two 300 dpi PNG figures in `docs/assets/figures/` following the
AASTeX publication plot style (Computer Modern typography, inward ticks on all
four sides, frameless legends, and journal-derived widths):
1. `wave3b_microphysics_diagnostics.png`: Rosseland opacity $\\log_{10}\\kappa$
   across low-$T$ molecular (`lowT_fa05_gs98`), OPAL Type 1 (`gs98`), and
   Compton regimes at $\\log_{10} R = -3.0$ (left), paired with `basic.net`
   nuclear energy generation $\\log_{10}\\epsilon_{\\mathrm{nuc}}$ and
   ${}^{12}\\mathrm{C}(\\alpha,\\gamma){}^{16}\\mathrm{O}$ screened rate across
   temperature comparing Safe Rust and JAX (right).
2. `wave3b_benchmark_staircase.png`: Four-way Wave 3b.5 runtime comparison at
   $N_{\\mathrm{zones}} = 4096$ across Upstream Fortran MESA, Safe Rust,
   Python/PyO3 (`flexmesa._rust`), and Python/JAX (`@jax.jit(vmap)`) (left),
   paired with `ATM_TABLE_PHOTOSPHERE` vs. Eddington grey $T(\\tau)$ surface
   pressure boundary conditions across $T_{\\mathrm{eff}}$ (right).
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure

from flexmesa import _rust, atm, chem, constants, kap, net, rates

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


def plot_wave3b_microphysics_diagnostics(out_dir: pathlib.Path) -> None:
    """Plot Wave 3b opacity regimes and `basic.net` nuclear burning curves."""
    _configure_plot_style()
    fig, ax = _two_panel_fig(wspace=0.40)

    log_t_kap = np.linspace(3.55, 8.2, 48, dtype=np.float64)
    log_r_fixed = -3.0
    log_rho_kap = log_r_fixed + 3.0 * log_t_kap - 18.0
    kap_r = np.zeros_like(log_t_kap)
    for k in range(log_t_kap.size):
        res = kap.kap_get_results(
            z=0.02,
            x=0.70,
            zbase=0.02,
            log_rho=float(log_rho_kap[k]),
            log_t=float(log_t_kap[k]),
            lnfree_e=-0.16,
            d_lnfree_e_dln_rho=0.0,
            d_lnfree_e_dln_t=0.0,
            eta=-10.0,
            d_eta_dln_rho=1.0,
            d_eta_dln_t=-1.5,
            zbar=1.28,
        )
        kap_r[k] = res.log10_kap

    kap_fn_jax = kap.make_differentiable_kap(
        z=0.02,
        x=0.70,
        zbase=0.02,
        lnfree_e=-0.16,
        eta=-10.0,
        d_eta_dln_rho=1.0,
        d_eta_dln_t=-1.5,
        zbar=1.28,
    )
    idx_sub = np.arange(0, log_t_kap.size, 6)
    kap_j = np.array(
        [
            float(
                kap_fn_jax(
                    jnp.float64(log_rho_kap[i]), jnp.float64(log_t_kap[i])
                )
            )
            for i in idx_sub
        ]
    )

    ax[0].plot(
        log_t_kap,
        kap_r,
        color="C0",
        lw=1.8,
        label=r"$\log_{10}\kappa$ (Safe Rust)",
    )
    ax[0].scatter(
        log_t_kap[idx_sub],
        kap_j,
        color="C3",
        s=18,
        zorder=5,
        label=r"$\log_{10}\kappa$ (JAX $\mathtt{custom\_jvp}$)",
    )
    ax[0].axvspan(
        3.80, 3.88, color="C1", alpha=0.22, label="Low/High-$T$ Blend"
    )
    ax[0].set_xlim(3.55, 8.2)
    ax[0].set_xlabel(r"Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[0].set_ylabel(r"Opacity [$\log_{10}(\kappa/\mathrm{cm^2\,g^{-1}})$]")
    ax[0].legend(frameon=False, loc="lower right")

    info = net.get_net_info("basic.net")
    xa = np.array(
        [
            0.70,
            2.98e-5,
            0.28,
            0.0034,
            0.001,
            0.0096,
            0.002,
            1.0 - (0.70 + 2.98e-5 + 0.28 + 0.0034 + 0.001 + 0.0096 + 0.002),
        ],
        dtype=np.float64,
    )
    comp = chem.get_composition_info(info.chem_id, xa, backend="rust")
    log_t_net = np.linspace(7.1, 8.1, 40, dtype=np.float64)
    eps_r = np.zeros_like(log_t_net)
    for k in range(log_t_net.size):
        t_v = float(10.0 ** log_t_net[k])
        out = net.eval_net(
            "basic.net",
            xa,
            t_v,
            100.0,
            float(comp.abar),
            float(comp.zbar),
            float(comp.z2bar),
            float(comp.ye),
        )
        eps_r[k] = np.log10(max(out.eps_nuc, 1.0e-30))

    idx_n = np.arange(0, log_t_net.size, 5)
    eps_j = np.zeros(idx_n.size, dtype=np.float64)
    for m, idx in enumerate(idx_n):
        e_v, _ = net.eval_net_jax(
            "basic.net",
            jnp.asarray(xa),
            jnp.float64(10.0 ** log_t_net[idx]),
            jnp.float64(100.0),
            float(comp.abar),
            float(comp.zbar),
            float(comp.z2bar),
            float(comp.ye),
        )
        eps_j[m] = np.log10(max(float(e_v), 1.0e-30))

    _, terms = _rust.rates_reaclib_terms("r_c12_ag_o16")
    coeffs_j = jnp.asarray([row[3] for row in terms], dtype=jnp.float64)
    fw_j, _ = rates.eval_reaclib_jax(
        coeffs_j, jnp.asarray(10.0 ** (log_t_net - 9.0), dtype=jnp.float64)
    )
    log_fw = np.log10(np.maximum(np.asarray(fw_j), 1.0e-99))

    ax[1].plot(
        log_t_net,
        eps_r,
        color="C0",
        lw=1.8,
        label=r"$\log_{10}\epsilon_{\mathrm{nuc}}$ (Safe Rust)",
    )
    ax[1].scatter(
        log_t_net[idx_n],
        eps_j,
        color="C3",
        s=18,
        zorder=5,
        label=r"$\log_{10}\epsilon_{\mathrm{nuc}}$ (JAX)",
    )
    ax[1].plot(
        log_t_net,
        log_fw + 22.0,
        color="C2",
        lw=1.5,
        ls="--",
        label=r"$\log_{10}\lambda_{{}^{12}\mathrm{C}(\alpha,\gamma)} + 22$",
    )
    ax[1].set_xlim(7.1, 8.1)
    ax[1].set_xlabel(r"Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[1].set_ylabel(r"Log Rate / Energy [$\mathrm{dex}$]")
    ax[1].legend(frameon=False, loc="lower right")
    _save_fig(fig, out_dir / "wave3b_microphysics_diagnostics.png")


def plot_wave3b_benchmark_staircase(out_dir: pathlib.Path) -> None:
    """Plot the 4-way Wave 3b.5 runtime comparison and atmosphere profiles."""
    _configure_plot_style()
    workloads = ["eos_4d", "kap", "rates", "net", "atm"]
    x_pos = np.arange(len(workloads), dtype=np.float64)
    w_bar = 0.18
    fortran_4t = np.array([1451.55, 452.32, 271.28, 10185.83, 650.79])
    rust_4t = np.array([994.56, 266.31, 95.25, 3738.18, 62.66])
    pyo3_4t = np.array([1070.91, 300.27, 167.71, 3792.84, 81.50])
    jax_vmap = np.array([143.95, 401.20, 226.50, 3709.81, 167.47])

    fig, ax = _two_panel_fig(wspace=0.40)
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
    ax[0].set_xticklabels(workloads, fontsize=8.0)
    ax[0].set_ylim(1.4, 4.4)
    ax[0].set_xlabel(r"Wave 3b Workload ($N_{\mathrm{zones}} = 4096$)")
    ax[0].set_ylabel(r"Latency [$\log_{10}(t / \mu\mathrm{s})$]")
    ax[0].legend(frameon=False, loc="upper left")

    teffs = np.linspace(4800.0, 6600.0, 36, dtype=np.float64)
    lnp_tbl = np.zeros_like(teffs)
    lnp_edd = np.zeros_like(teffs)
    r_val = constants.RSUN
    m_val = constants.MSUN
    for k, teff in enumerate(teffs):
        l_val = (
            constants.PI
            * constants.CRAD
            * constants.CLIGHT
            * (r_val**2)
            * (float(teff) ** 4)
        )
        res_tbl = atm.eval_table(
            l_val,
            r_val,
            m_val,
            constants.STANDARD_CGRAV,
            atm.ATM_TABLE_PHOTOSPHERE,
            0.0179,
            False,
            float(teff),
        )
        res_edd = atm.eval_t_tau_uniform(
            2.0 / 3.0,
            l_val,
            r_val,
            m_val,
            constants.STANDARD_CGRAV,
            0.3,
            1.0,
            atm.ATM_T_TAU_EDDINGTON,
            1.0e-6,
            0,
            True,
            float(teff),
        )
        lnp_tbl[k] = res_tbl.ln_p
        lnp_edd[k] = res_edd.ln_p

    ax[1].plot(
        teffs,
        lnp_tbl,
        color="C0",
        lw=1.8,
        label=r"Photosphere Table ($\tau = 2/3$)",
    )
    ax[1].plot(
        teffs,
        lnp_edd,
        color="C1",
        lw=1.6,
        ls="--",
        label=r"Eddington Grey ($\kappa = 0.3$)",
    )
    ax[1].set_xlim(4800.0, 6600.0)
    ax[1].set_xlabel(r"Effective Temperature [$T_{\mathrm{eff}} / \mathrm{K}$]")
    ax[1].set_ylabel(r"Surface Pressure [$\ln(P / \mathrm{dyn\,cm^{-2}})$]")
    ax[1].legend(frameon=False, loc="upper right")
    _save_fig(fig, out_dir / "wave3b_benchmark_staircase.png")


def main() -> None:
    """Build all Wave 3b documentation figures."""
    root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_wave3b_microphysics_diagnostics(out_dir)
    plot_wave3b_benchmark_staircase(out_dir)


if __name__ == "__main__":
    main()
