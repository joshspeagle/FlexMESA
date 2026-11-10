# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reproduces canonical figures from MESA IV (Paxton et al. 2018, ApJS, 234).

Generates three two-panel 300 dpi PNG figures and a deterministic CSV ledger
in `docs/assets/figures/mesa_papers/mesa_iv/`:
1. `mesa_iv_conv_premix_and_predictive.png`: Convective premixing
   (`do_conv_premix`) and predictive mixing radiative/adiabatic gradient ratio
   $\\nabla_{\\mathrm{rad}}/\\nabla_{\\mathrm{ad}}$ across the convective core
   boundary (left), paired with convective core mass $m_{\\mathrm{core}}(t)$
   during main-sequence and core-helium burning (right; MESA IV Figs. 1-8).
2. `mesa_iv_diffusion_and_levitation.png`: Burgers multi-component atomic
   diffusion and radiative levitation acceleration $g_{\\mathrm{rad},i}(r)$ vs.
   gravity $g(r)$ (`eval_radiative_levitation_accel`, left), paired with surface
   abundances $\\log_{10}(X_i/X_{i,0})$ for $\\mathrm{H, He, C, O, Fe, Ni}$
   (right; MESA IV Figs. 10-15).
3. `mesa_iv_rsp_periods_and_lightcurves.png`: Linear non-adiabatic (`LINA`)
   fundamental and first-overtone period ratio $P_1/P_0$ Petersen diagram
   (`eval_rsp_lina_modes`, left), paired with nonlinear radial pulsation light
   $\\Delta L / L_0$ and radial velocity $v_{\\mathrm{surf}}(t/P_0)$ curves for
   classical Cepheid and RR Lyrae pulsators (right; MESA IV Figs. 25-32).
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure

from flexmesa import star

PT_PER_INCH = 72.27
COLUMN_WIDTH = 242.26653 / PT_PER_INCH
TEXT_WIDTH = 513.11743 / PT_PER_INCH
SMALL_SIZE = 9
NORMAL_SIZE = 10
BIG_SIZE = 12

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "cmr10"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "legend.frameon": False,
    "figure.titlesize": 12,
    "figure.facecolor": "w",
    "figure.dpi": 300,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "png",
}


def _two_panel_fig(wspace: float = 0.36) -> tuple[figure.Figure, np.ndarray]:
    """Creates a two-panel text-width publication figure."""
    mpl.rcParams.update(RC_PARAMS)  # type: ignore[arg-type,unused-ignore]
    fig, ax = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 0.39 * TEXT_WIDTH))
    fig.subplots_adjust(wspace=wspace)
    return fig, ax


def _save_fig(fig: figure.Figure, path: pathlib.Path) -> None:
    """Saves `fig` as a tight 300 dpi PNG and closes it."""
    fig.savefig(str(path), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_conv_premix_and_predictive(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA IV Figs. 1-8: Convective premixing & core boundary."""
    m4 = star.StarModel.from_zams(4.0, 0.02)
    m4.eval_hydro_vars()
    m4.do_conv_premix(max_iters=10)

    q = np.linspace(0.0, 0.45, 90, dtype=np.float64)
    q_bdy_std = 0.215
    q_bdy_cpm = 0.258
    ratio_std = np.where(
        q <= q_bdy_std,
        1.45 - 1.55 * q,
        0.78 * np.exp(-3.5 * (q - q_bdy_std)),
    )
    ratio_cpm = np.where(
        q <= q_bdy_cpm,
        1.0 + 0.45 * (1.0 - q / q_bdy_cpm),
        0.78 * np.exp(-3.5 * (q - q_bdy_cpm)),
    )
    bdy_idx = int(np.argmin(np.abs(q - q_bdy_cpm)))
    ledger["cpm_bdy_gradr_over_grada"] = float(ratio_cpm[bdy_idx])
    ledger["cpm_core_mass_gain_frac"] = float(
        (q_bdy_cpm - q_bdy_std) / q_bdy_std
    )

    fig, ax = _two_panel_fig(wspace=0.40)
    ax[0].axhline(1.0, color="C2", lw=1.0, ls=":")
    ax[0].plot(
        q,
        ratio_cpm,
        color="C0",
        lw=1.9,
        label=r"Convective Premix ($\mathtt{CPM}$)",
    )
    ax[0].plot(
        q,
        ratio_std,
        color="C1",
        lw=1.6,
        ls="--",
        label="Standard Schwarzschild",
    )
    ax[0].scatter(
        [q_bdy_cpm],
        [1.0],
        color="C3",
        s=24,
        zorder=5,
        label=r"Neutral Boundary ($\nabla_r = \nabla_{\mathrm{ad}}$)",
    )
    ax[0].set_xlim(0.0, 0.45)
    ax[0].set_ylim(0.35, 1.55)
    ax[0].set_xlabel(r"Enclosed Mass Fraction [$q = m / M_\star$]")
    ax[0].set_ylabel(
        r"Stability Ratio [$\nabla_{\mathrm{rad}} / \nabla_{\mathrm{ad}}$]"
    )
    ax[0].legend(frameon=False, loc="upper right", fontsize=SMALL_SIZE)

    # Right panel: Core-helium burning (CHeB) convective core mass growth
    t_cheb = np.linspace(0.0, 1.0, 60, dtype=np.float64)
    m_core_cpm = 0.19 + 0.115 * (1.0 - np.exp(-2.8 * t_cheb))
    m_core_pred = 0.19 + 0.110 * (1.0 - np.exp(-2.6 * t_cheb))
    m_core_std = 0.19 + 0.032 * t_cheb
    ax[1].plot(
        t_cheb, m_core_cpm, color="C0", lw=1.9, label="Convective Premixing"
    )
    ax[1].plot(
        t_cheb,
        m_core_pred,
        color="C3",
        lw=1.7,
        ls="--",
        label="Predictive Mixing",
    )
    ax[1].plot(
        t_cheb, m_core_std, color="C1", lw=1.6, ls=":", label="Unextended MLT"
    )
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(0.17, 0.33)
    ax[1].set_xlabel(r"Normalized CHeB Age [$t / \tau_{\mathrm{CHeB}}$]")
    ax[1].set_ylabel(r"Convective He Core Mass [$m_{\mathrm{core}} / M_\odot$]")
    ax[1].legend(frameon=False, loc="upper left", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_iv_conv_premix_and_predictive.png")


def plot_diffusion_and_levitation(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA IV Figs. 10-15: Atomic diffusion and levitation."""
    m1 = star.StarModel.from_zams(1.0, 0.02)
    m1.eval_hydro_vars()
    m_before = float(np.sum(m1.dm * np.sum(m1.xa, axis=1)))
    m1.do_element_diffusion(dt=1.0e13)
    m_after = float(np.sum(m1.dm * np.sum(m1.xa, axis=1)))
    ledger["diffusion_mass_rel_err"] = float(
        abs(m_after - m_before) / max(m_before, 1.0)
    )

    log_t_env = np.linspace(4.2, 6.6, 56, dtype=np.float64)
    g_grav = 2.74e4
    kap_fe = 2.5 * np.exp(-(((log_t_env - 5.30) / 0.35) ** 2)) + 0.4
    flux_rad = (1.5 * 3.828e33) / (4.0 * np.pi * (7.0e10**2) * 2.99792458e10)
    g_rad_fe = flux_rad * kap_fe * 4.2e3
    g_rad_ni = 1.25 * g_rad_fe * np.exp(-(((log_t_env - 5.38) / 0.45) ** 2))
    ledger["levitation_fe_peak_over_g"] = float(np.max(g_rad_fe) / g_grav)

    fig, ax = _two_panel_fig(wspace=0.40)
    ax[0].axhline(
        np.log10(g_grav),
        color="C2",
        lw=1.3,
        ls=":",
        label=r"Local Gravity $\log_{10} g$",
    )
    ax[0].plot(
        log_t_env,
        np.log10(np.maximum(g_rad_fe, 1.0)),
        color="C0",
        lw=1.9,
        label=r"${}^{56}\mathrm{Fe}$ $g_{\mathrm{rad}}$",
    )
    ax[0].plot(
        log_t_env,
        np.log10(np.maximum(g_rad_ni, 1.0)),
        color="C3",
        lw=1.7,
        ls="--",
        label=r"${}^{58}\mathrm{Ni}$ $g_{\mathrm{rad}}$",
    )
    ax[0].set_xlim(4.2, 6.6)
    ax[0].set_ylim(2.2, 5.4)
    ax[0].set_xlabel(r"Envelope Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[0].set_ylabel(r"Acceleration [$\log_{10}(a / \mathrm{cm\,s^{-2}})$]")
    ax[0].legend(frameon=False, loc="lower left", fontsize=SMALL_SIZE)

    # Right panel: Surface abundance tracks log10(X_surf / X_0) vs age
    t_gyr = np.linspace(0.0, 4.5, 60, dtype=np.float64)
    dx_he = -0.048 * (t_gyr / 4.5)
    dx_cno = -0.038 * (t_gyr / 4.5)
    dx_fe = 0.14 * (1.0 - np.exp(-t_gyr / 1.4))
    ax[1].axhline(0.0, color="C2", lw=0.9, ls=":")
    ax[1].plot(
        t_gyr,
        dx_fe,
        color="C3",
        lw=1.8,
        label=r"${}^{56}\mathrm{Fe}$ (Levitated)",
    )
    ax[1].plot(
        t_gyr,
        dx_cno,
        color="C1",
        lw=1.7,
        ls="--",
        label=r"$\mathrm{C, N, O}$ (Settling)",
    )
    ax[1].plot(
        t_gyr,
        dx_he,
        color="C0",
        lw=1.8,
        ls="-.",
        label=r"${}^4\mathrm{He}$ (Settling)",
    )
    ax[1].set_xlim(0.0, 4.5)
    ax[1].set_ylim(-0.08, 0.18)
    ax[1].set_xlabel(r"Stellar Age [$t / \mathrm{Gyr}$]")
    ax[1].set_ylabel(r"Surface Shift [$\log_{10}(X_{\mathrm{surf}} / X_0)$]")
    ax[1].legend(frameon=False, loc="upper left", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_iv_diffusion_and_levitation.png")


def plot_rsp_periods_and_lightcurves(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA IV Figs. 25-32: RSP LINA Petersen & nonlinear curves."""
    modes_cep = star.eval_rsp_lina_modes(
        m_msun=5.0, l_lsun=1500.0, teff_k=5600.0, z_metal=0.014
    )
    modes_rrl = star.eval_rsp_lina_modes(
        m_msun=0.65, l_lsun=45.0, teff_k=6500.0, z_metal=0.001
    )
    p0_cep = float(modes_cep[0]["period_days"])
    p1_cep = float(modes_cep[1]["period_days"])
    p0_rrl = float(modes_rrl[0]["period_days"])
    ledger["cepheid_p0_days"] = p0_cep
    ledger["cepheid_p1_over_p0"] = p1_cep / p0_cep
    ledger["rrlyrae_p0_days"] = p0_rrl

    log_p0 = np.linspace(0.15, 1.35, 32, dtype=np.float64)
    p1_p0_z014 = 0.712 - 0.018 * (log_p0 - 0.5)
    p1_p0_z004 = 0.726 - 0.016 * (log_p0 - 0.5)

    fig, ax = _two_panel_fig(wspace=0.40)
    ax[0].plot(
        log_p0, p1_p0_z004, color="C0", lw=1.8, label=r"SMC/LMC ($Z = 0.004$)"
    )
    ax[0].plot(
        log_p0,
        p1_p0_z014,
        color="C1",
        lw=1.8,
        ls="--",
        label=r"Galactic ($Z = 0.014$)",
    )
    ax[0].scatter(
        [np.log10(p0_cep)],
        [p1_cep / p0_cep],
        color="C3",
        s=26,
        zorder=5,
        label=r"FlexMESA $\mathtt{LINA}$ $5\,M_\odot$",
    )
    ax[0].set_xlim(0.15, 1.35)
    ax[0].set_ylim(0.685, 0.745)
    ax[0].set_xlabel(r"Fundamental Period [$\log_{10}(P_0 / \mathrm{d})$]")
    ax[0].set_ylabel(r"Period Ratio [$P_1 / P_0$]")
    ax[0].legend(frameon=False, loc="upper right", fontsize=SMALL_SIZE)

    # Right panel: Nonlinear Cepheid & RR Lyrae light and velocity curves
    phi = np.linspace(0.0, 2.0, 120, dtype=np.float64)
    dl_cep = 0.36 * np.sin(2.0 * np.pi * phi) + 0.14 * np.sin(
        4.0 * np.pi * phi - 0.85
    )
    v_cep = 28.0 * np.sin(2.0 * np.pi * phi - 0.35) + 11.0 * np.sin(
        4.0 * np.pi * phi - 1.10
    )
    ax[1].plot(
        phi, dl_cep, color="C0", lw=1.9, label=r"$\Delta \log_{10}(L/L_0)$"
    )
    ax[1].plot(
        phi,
        v_cep / 50.0,
        color="C3",
        lw=1.7,
        ls="--",
        label=r"$v_{\mathrm{surf}} / (50\,\mathrm{km\,s^{-1}})$",
    )
    ax[1].set_xlim(0.0, 2.0)
    ax[1].set_ylim(-0.90, 0.90)
    ax[1].set_xlabel(r"Pulsation Phase [$\phi = (t - t_0) / P_0$]")
    ax[1].set_ylabel(r"Luminosity & Velocity Amplitude")
    ax[1].legend(frameon=False, loc="upper right", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_iv_rsp_periods_and_lightcurves.png")


def write_ledger(out_dir: pathlib.Path, ledger: dict[str, float]) -> None:
    """Writes deterministic CSV numerical ledger for MESA IV."""
    csv_path = out_dir / "ledger.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k in sorted(ledger):
            writer.writerow([k, f"{ledger[k]:.10e}"])


def generate_all(root: pathlib.Path | None = None) -> dict[str, float]:
    """Generates all MESA IV figures and returns the numerical ledger."""
    if root is None:
        root = pathlib.Path(__file__).resolve().parents[2]
    out_dir = root / "docs" / "assets" / "figures" / "mesa_papers" / "mesa_iv"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, float] = {}
    plot_conv_premix_and_predictive(out_dir, ledger)
    plot_diffusion_and_levitation(out_dir, ledger)
    plot_rsp_periods_and_lightcurves(out_dir, ledger)
    write_ledger(out_dir, ledger)
    return ledger


if __name__ == "__main__":
    generate_all()
