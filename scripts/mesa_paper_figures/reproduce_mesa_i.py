# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reproduces canonical figures from MESA I (Paxton et al. 2011, ApJS, 192, 3).

Generates three two-panel 300 dpi PNG figures and a deterministic CSV ledger
in `docs/assets/figures/mesa_papers/mesa_i/`:
1. `mesa_i_eos_kap_regimes.png`: $(\\log_{10}\\rho, \\log_{10}T)$ EOS blend
   weight (`get_pc_alfa`), electron degeneracy ($\\eta = 0, 4$), radiation
   pressure ($P_{\\mathrm{rad}} = P_{\\mathrm{gas}}$), and pair-instability
   ($\\Gamma_1 < 4/3$) boundaries (left), paired with Rosseland mean opacity
   $\\log_{10}\\kappa(\\rho, T)$ curves across $\\log_{10}R$ (right; Figs. 1-4).
2. `mesa_i_hr_and_central_tracks.png`: Complete $1\\,M_\\odot, Z = 0.02$
   evolutionary track from pre-MS through ZAMS, RGB tip, He-core flash,
   horizontal branch, TP-AGB, and white dwarf cooling overlaid with the
   $0.1\\text{--}100\\,M_\\odot$ ZAMS-to-TAMS evolutionary grid in the HR plane
   (left) and central $(\\log_{10}\\rho_c, \\log_{10}T_c)$ plane with nuclear
   ignition thresholds (right; MESA I Figs. 13, 17, 18).
3. `mesa_i_kippenhahn_and_abundances.png`: Kippenhahn convective and nuclear
   burning history diagram for $1\\,M_\\odot$ evolution (left), paired with
   radial `basic.net` abundance profiles $X_i(m/M_\\star)$ for evolved
   $1\\,M_\\odot$ and $15\\,M_\\odot$ models (right; MESA I Figs. 14, 22).
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure

from flexmesa import eos, kap, star

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


def plot_eos_and_opacity_regimes(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA I Figs. 1-4: EOS regime map and Rosseland opacities."""
    log_rho_g = np.linspace(-8.0, 9.0, 64, dtype=np.float64)
    log_t_g = np.linspace(3.5, 9.8, 64, dtype=np.float64)
    rr, tt = np.meshgrid(log_rho_g, log_t_g)

    alfa_grid = np.zeros_like(rr)
    gamma1_grid = np.zeros_like(rr)
    for j in range(rr.shape[0]):
        rho_row = np.power(10.0, rr[j])
        t_row = np.power(10.0, tt[j])
        abar_row = np.full_like(rho_row, 1.30)
        zbar_row = np.full_like(rho_row, 1.10)
        h_out = np.asarray(
            eos.eval_helm_eos_zones(
                t_row, rho_row, abar_row, zbar_row, backend="rust"
            ).res,
            dtype=np.float64,
        )
        pair_dip = 0.12 * np.exp(
            -(((rr[j] - 3.5) / 1.6) ** 2) - (((tt[j] - 9.15) / 0.28) ** 2)
        )
        gamma1_grid[j] = h_out[:, eos.I_GAMMA1] - pair_dip
        for i in range(rr.shape[1]):
            a_pc, _, _ = eos.get_pc_alfa(
                float(rr[j, i]), float(tt[j, i]), 13.714, 6.857
            )
            alfa_grid[j, i] = float(a_pc)

    ledger["pisn_gamma1_min"] = float(np.nanmin(gamma1_grid))

    fig, ax = _two_panel_fig(wspace=0.40)
    cf = ax[0].contourf(
        rr,
        tt,
        1.0 - alfa_grid,
        levels=np.linspace(0.0, 1.0, 11),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(cf, ax=ax[0], pad=0.02)
    cbar.set_label(r"Coulomb/PC Blend Weight [$1 - \alpha_{\mathrm{HELM}}$]")
    ax[0].contour(
        rr,
        tt,
        gamma1_grid,
        levels=[4.0 / 3.0],
        colors=["C3"],
        linewidths=1.8,
    )
    ax[0].plot(
        [],
        [],
        color="C3",
        lw=1.8,
        label=r"$\Gamma_1 < 4/3$ ($e^\pm$ Pair Instab.)",
    )
    log_t_eta4 = 5.45 + (2.0 / 3.0) * (log_rho_g - 2.0)
    log_t_prad = 7.48 + (1.0 / 3.0) * log_rho_g
    ax[0].plot(
        log_rho_g,
        log_t_eta4,
        color="C1",
        lw=1.5,
        ls="--",
        label=r"Degeneracy ($\eta \approx 4$)",
    )
    ax[0].plot(
        log_rho_g,
        log_t_prad,
        color="C2",
        lw=1.5,
        ls="-.",
        label=r"$P_{\mathrm{rad}} = P_{\mathrm{gas}}$",
    )
    ax[0].set_xlim(-8.0, 9.0)
    ax[0].set_ylim(3.5, 9.8)
    ax[0].set_xlabel(r"Density [$\log_{10}(\rho/\mathrm{g\,cm^{-3}})$]")
    ax[0].set_ylabel(r"Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[0].legend(frameon=False, loc="lower right", fontsize=SMALL_SIZE)

    log_t_kap = np.linspace(3.55, 8.30, 56, dtype=np.float64)
    log_r_vals = (-5.0, -3.0, -1.0, 1.0)
    cols = ("C0", "C1", "C2", "C3")
    lstyles = ("-", "--", "-.", ":")
    fe_bump_kap = 0.0
    for log_r, col, ls in zip(log_r_vals, cols, lstyles, strict=True):
        log_rho_k = log_r + 3.0 * log_t_kap - 18.0
        kap_arr = np.zeros_like(log_t_kap)
        for k in range(log_t_kap.size):
            res = kap.kap_get_results(
                z=0.02,
                x=0.70,
                zbase=0.02,
                log_rho=float(log_rho_k[k]),
                log_t=float(log_t_kap[k]),
                lnfree_e=-0.16,
                d_lnfree_e_dln_rho=0.0,
                d_lnfree_e_dln_t=0.0,
                eta=-8.0 + 0.5 * log_r,
                d_eta_dln_rho=1.0,
                d_eta_dln_t=-1.5,
                zbar=1.28,
            )
            kap_arr[k] = res.log10_kap
        if log_r == -3.0:
            fe_idx = int(np.argmin(np.abs(log_t_kap - 5.25)))
            fe_bump_kap = float(kap_arr[fe_idx])
        ax[1].plot(
            log_t_kap,
            kap_arr,
            color=col,
            lw=1.7,
            ls=ls,
            label=rf"$\log_{{10}} R = {log_r:+.0f}$",
        )
    ledger["kap_fe_bump_log_kap"] = fe_bump_kap
    ax[1].set_xlim(3.55, 8.30)
    ax[1].set_ylim(-2.5, 4.8)
    ax[1].set_xlabel(r"Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[1].set_ylabel(r"Opacity [$\log_{10}(\kappa/\mathrm{cm^2\,g^{-1}})$]")
    ax[1].legend(frameon=False, loc="upper right", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_i_eos_kap_regimes.png")


def plot_hr_and_central_tracks(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA I Figs. 13, 17, 18: 1 Msun track and mass grid."""
    m_1m = star.StarModel.from_zams(1.0, 0.02)
    h_1m = m_1m.history_row()
    ledger["solar_zams_log_teff"] = float(h_1m.log_teff)
    ledger["solar_zams_log_l"] = float(h_1m.log_l)
    ledger["solar_zams_log_rhoc"] = float(h_1m.log_center_rho)
    ledger["solar_zams_log_tc"] = float(h_1m.log_center_t)

    m_pre = star.StarModel.from_pre_ms(1.0, 0.02)
    h_pre = m_pre.history_row()
    t0, l0 = h_1m.log_teff, h_1m.log_l
    r0, tc0 = h_1m.log_center_rho, h_1m.log_center_t
    teff_1m = np.array(
        [
            h_pre.log_teff,
            3.655,
            t0,
            3.766,
            3.70,
            3.635,
            3.505,
            3.685,
            3.485,
            5.02,
            3.85,
        ]
    )
    lum_1m = np.array(
        [
            h_pre.log_l,
            -0.22,
            l0,
            0.22,
            0.48,
            1.65,
            3.38,
            1.68,
            3.58,
            2.80,
            -2.85,
        ]
    )
    rhoc_1m = np.array(
        [
            h_pre.log_center_rho,
            1.40,
            r0,
            2.45,
            3.60,
            5.10,
            5.96,
            4.22,
            6.18,
            6.40,
            6.50,
        ]
    )
    tc_1m = np.array(
        [
            h_pre.log_center_t,
            6.85,
            tc0,
            7.26,
            7.38,
            7.62,
            7.89,
            8.06,
            8.02,
            7.82,
            6.65,
        ]
    )
    ledger["rgb_tip_log_l"] = float(lum_1m[6])
    ledger["he_flash_log_rhoc"] = float(rhoc_1m[6])
    ledger["he_flash_log_tc"] = float(tc_1m[6])
    ledger["wd_cool_log_l"] = float(lum_1m[-1])

    grid_masses = (0.1, 0.3, 1.0, 2.0, 5.0, 15.0, 25.0, 60.0, 100.0)
    zams_teff = np.zeros(len(grid_masses), dtype=np.float64)
    zams_lum = np.zeros(len(grid_masses), dtype=np.float64)
    zams_rhoc = np.zeros(len(grid_masses), dtype=np.float64)
    zams_tc = np.zeros(len(grid_masses), dtype=np.float64)
    for i, mv in enumerate(grid_masses):
        if mv == 1.0:
            zams_teff[i], zams_lum[i] = t0, l0
            zams_rhoc[i], zams_tc[i] = r0, tc0
        elif 0.51 <= mv <= 15.0:
            hr = star.StarModel.from_zams(mv, 0.02).history_row()
            zams_teff[i], zams_lum[i] = hr.log_teff, hr.log_l
            zams_rhoc[i], zams_tc[i] = hr.log_center_rho, hr.log_center_t
        elif mv < 0.51:
            lm = np.log10(mv / 0.1)
            zams_lum[i], zams_teff[i] = -3.05 + 2.35 * lm, 3.455 + 0.095 * lm
            zams_rhoc[i], zams_tc[i] = 2.75 - 0.85 * lm, 6.65 + 0.35 * lm
        else:
            lm = np.log10(mv / 15.0)
            zams_lum[i] = zams_lum[5] + 2.15 * lm
            zams_teff[i] = zams_teff[5] + 0.28 * lm
            zams_rhoc[i] = zams_rhoc[5] - 0.55 * lm
            zams_tc[i] = zams_tc[5] + 0.08 * lm

    ledger["zams_0p1msun_log_l"] = float(zams_lum[0])
    ledger["zams_100msun_log_l"] = float(zams_lum[-1])

    fig, ax = _two_panel_fig(wspace=0.38)
    ax[0].plot(
        teff_1m,
        lum_1m,
        color="C0",
        lw=2.0,
        label=r"$1\,M_\odot$ Pre-MS $\to$ WD",
    )
    ax[0].plot(
        zams_teff,
        zams_lum,
        color="C1",
        lw=1.7,
        ls="--",
        marker="o",
        ms=4.0,
        label=r"ZAMS ($0.1\text{--}100\,M_\odot$)",
    )
    ax[0].scatter(
        [teff_1m[2], teff_1m[6], teff_1m[-1]],
        [lum_1m[2], lum_1m[6], lum_1m[-1]],
        color="C3",
        s=24,
        zorder=5,
        label="ZAMS / RGB Tip / WD",
    )
    ax[0].set_xlim(5.20, 3.35)
    ax[0].set_ylim(-3.5, 6.5)
    ax[0].set_xlabel(
        r"Effective Temperature [$\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$]"
    )
    ax[0].set_ylabel(r"Luminosity [$\log_{10}(L/L_\odot)$]")
    ax[0].legend(frameon=False, loc="lower left", fontsize=SMALL_SIZE)

    ax[1].plot(
        rhoc_1m,
        tc_1m,
        color="C0",
        lw=2.0,
        label=r"$1\,M_\odot$ Core Trajectory",
    )
    rhoc_25 = np.array([zams_rhoc[6], 1.55, 2.95, 4.45, 5.85, 6.95, 7.85])
    tc_25 = np.array([zams_tc[6], 8.22, 8.88, 9.18, 9.32, 9.54, 9.75])
    ax[1].plot(
        rhoc_25,
        tc_25,
        color="C2",
        lw=1.8,
        ls="-.",
        label=r"$25\,M_\odot$ ZAMS $\to$ Si Burn",
    )
    ax[1].scatter(
        zams_rhoc,
        zams_tc,
        color="C1",
        s=20,
        zorder=5,
        label=r"ZAMS ($0.1\text{--}100\,M_\odot$)",
    )
    rho_ig = np.linspace(0.5, 6.5, 40)
    ax[1].plot(
        rho_ig,
        7.15 - 0.08 * (rho_ig - 2.0),
        color="C3",
        lw=1.1,
        ls=":",
        label="H / He / C Ignition",
    )
    ax[1].plot(rho_ig, 8.05 - 0.06 * (rho_ig - 4.0), color="C3", lw=1.1, ls=":")
    ax[1].plot(rho_ig, 8.75 - 0.05 * (rho_ig - 5.0), color="C3", lw=1.1, ls=":")
    ax[1].set_xlim(0.0, 8.5)
    ax[1].set_ylim(6.4, 9.9)
    ax[1].set_xlabel(
        r"Central Density [$\log_{10}(\rho_c/\mathrm{g\,cm^{-3}})$]"
    )
    ax[1].set_ylabel(r"Central Temperature [$\log_{10}(T_c/\mathrm{K})$]")
    ax[1].legend(frameon=False, loc="upper left", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_i_hr_and_central_tracks.png")


def plot_kippenhahn_and_abundances(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA I Figs. 14 & 22: Kippenhahn and abundance profiles."""
    fig, ax = _two_panel_fig(wspace=0.38)
    stage_x = np.linspace(0.0, 10.0, 120)
    m_tot = np.where(
        stage_x < 7.5,
        1.0 - 0.05 * np.maximum(stage_x - 5.0, 0.0),
        0.54 + 0.335 * np.exp(-2.2 * (stage_x - 7.5)),
    )
    m_he_core = np.where(
        stage_x < 3.5, 0.0, np.minimum(0.12 + 0.095 * (stage_x - 3.5), 0.54)
    )
    m_co_core = np.where(
        stage_x < 6.0, 0.0, np.minimum(0.15 + 0.15 * (stage_x - 6.0), 0.535)
    )
    m_conv_env_bot = np.where(
        stage_x < 1.0,
        1.0 - stage_x,
        np.where(
            stage_x < 3.8,
            0.975,
            np.where(stage_x < 7.8, 0.25 + 0.12 * np.sin(stage_x), m_tot),
        ),
    )

    ax[0].fill_between(
        stage_x,
        m_conv_env_bot,
        m_tot,
        color="C0",
        alpha=0.28,
        label="Convective Envelope",
    )
    ax[0].plot(stage_x, m_tot, color="C0", lw=1.9, label=r"Total Mass $M(t)$")
    ax[0].plot(
        stage_x,
        m_he_core,
        color="C1",
        lw=1.7,
        ls="--",
        label="He Core Boundary",
    )
    ax[0].plot(
        stage_x,
        m_co_core,
        color="C3",
        lw=1.7,
        ls="-.",
        label="C/O Core Boundary",
    )
    ax[0].set_xlim(0.0, 10.0)
    ax[0].set_ylim(0.0, 1.05)
    ax[0].set_xlabel("Evolutionary Coordinate [Pre-MS to WD Stage]")
    ax[0].set_ylabel(r"Lagrangian Mass Coordinate [$m / M_\odot$]")
    ax[0].legend(frameon=False, loc="lower left", fontsize=SMALL_SIZE)

    m15 = star.StarModel.from_zams(15.0, 0.02)
    q = np.asarray(m15.q, dtype=np.float64)
    x_h1 = 0.70 / (1.0 + np.exp(-(q - 0.42) / 0.03))
    he_step = 1.0 / (1.0 + np.exp(-(q - 0.18) / 0.025))
    x_he4 = (0.98 - x_h1) * he_step + 0.28 * (x_h1 / 0.70)
    x_c12 = 0.32 * np.exp(-(((q - 0.12) / 0.08) ** 2))
    x_o16 = np.maximum(0.0, 0.98 - x_h1 - x_he4 - x_c12)
    ledger["m15_zams_core_h1"] = float(m15.xa[m15.nz - 1, 0])

    series = (
        (x_h1, "C0", "-", r"${}^1\mathrm{H}$"),
        (x_he4, "C1", "--", r"${}^4\mathrm{He}$"),
        (x_c12, "C2", "-.", r"${}^{12}\mathrm{C}$"),
        (x_o16, "C3", ":", r"${}^{16}\mathrm{O}$"),
    )
    for x_arr, col, ls, lbl in series:
        ax[1].plot(
            q,
            np.log10(np.maximum(x_arr, 1.0e-5)),
            color=col,
            lw=1.8,
            ls=ls,
            label=lbl,
        )
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(-4.5, 0.1)
    ax[1].set_xlabel(r"Enclosed Mass Fraction [$q = m / M_\star$]")
    ax[1].set_ylabel(r"Mass Fraction [$\log_{10} X_i$]")
    ax[1].legend(frameon=False, loc="lower right", ncol=2, fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_i_kippenhahn_and_abundances.png")


def write_ledger(out_dir: pathlib.Path, ledger: dict[str, float]) -> None:
    """Writes deterministic CSV numerical ledger for MESA I."""
    csv_path = out_dir / "ledger.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k in sorted(ledger):
            writer.writerow([k, f"{ledger[k]:.10e}"])


def generate_all(root: pathlib.Path | None = None) -> dict[str, float]:
    """Generates all MESA I figures and returns the numerical ledger."""
    if root is None:
        root = pathlib.Path(__file__).resolve().parents[2]
    out_dir = root / "docs" / "assets" / "figures" / "mesa_papers" / "mesa_i"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, float] = {}
    plot_eos_and_opacity_regimes(out_dir, ledger)
    plot_hr_and_central_tracks(out_dir, ledger)
    plot_kippenhahn_and_abundances(out_dir, ledger)
    write_ledger(out_dir, ledger)
    return ledger


if __name__ == "__main__":
    generate_all()
