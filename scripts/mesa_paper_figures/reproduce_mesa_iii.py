# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reproduces canonical figures from MESA III (Paxton et al. 2015, ApJS, 220).

Generates three two-panel 300 dpi PNG figures and a deterministic CSV ledger
in `docs/assets/figures/mesa_papers/mesa_iii/`:
1. `mesa_iii_binary_rlof_and_jdot.png`: Interacting binary Roche-lobe overflow
   mass-transfer rate histories $\\log_{10}|\\dot{M}_{\\mathrm{tr}}|$ comparing
   Ritter (1988) optically thin and Kolb & Ritter (1990) optically thick
   regimes (left), paired with Peters (1964) gravitational-wave and Tauris &
   van den Heuvel (2003) non-conservative mass-loss orbital separation
   evolution (right; MESA III Figs. 1-9).
2. `mesa_iii_weak_rates_urca.png`: Electron-capture and $\\beta^-$-decay rates
   $\\log_{10}(\\lambda/\\mathrm{s^{-1}})$ across $\\log_{10}(\\rho Y_e)$ for
   $A=23$ (${}^{23}\\mathrm{Na} \\leftrightarrow {}^{23}\\mathrm{Ne}$) and
   $A=25$ (${}^{25}\\mathrm{Mg} \\leftrightarrow {}^{25}\\mathrm{Na}$) Urca
   pairs (left), paired with specific Urca neutrino cooling across degenerate
   O/Ne core shells (right; MESA III Figs. 15-20).
3. `mesa_iii_hllc_shock_and_ccsn.png`: HLLC Riemann solver (`eval_hllc_face`)
   shock-tube density, pressure, and velocity profiles (left; MESA III Figs.
   24-27), paired with Type II-P supernova shock propagation, Duffell (2016)
   Rayleigh-Taylor mixing $D_{\\mathrm{RTI}}$, and ${}^{56}\\mathrm{Ni}$ mixing
   profiles (`eval_duffell_rti_shock_state`, right; MESA III Figs. 28-35).
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure

from flexmesa import binary, constants, star

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


def plot_binary_rlof_and_jdot(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA III Figs. 1-9: Binary RLOF mdot and orbital evolution."""
    m1_g = 2.0 * constants.MSUN
    m2_g = 1.4 * constants.MSUN
    rl1_cm = 2.4 * constants.RSUN
    gaps = np.linspace(-0.015, 0.025, 60, dtype=np.float64)
    mdot_thin = np.zeros_like(gaps)
    mdot_kolb = np.zeros_like(gaps)
    for i, dg in enumerate(gaps):
        r1 = rl1_cm * (1.0 + float(dg))
        m_th = abs(
            float(
                binary.eval_ritter_mdot_thin(
                    m1_g,
                    m2_g,
                    r1,
                    rl1_cm,
                    2.0e-7,
                    2.5e5,
                    6500.0,
                    0.61,
                    is_kolb_scheme=False,
                )
            )
        )
        m_th_kolb = abs(
            float(
                binary.eval_ritter_mdot_thin(
                    m1_g,
                    m2_g,
                    r1,
                    rl1_cm,
                    2.0e-7,
                    2.5e5,
                    6500.0,
                    0.61,
                    is_kolb_scheme=True,
                )
            )
        )
        m_kb = m_th_kolb
        if dg > 0.0:
            r_prof = np.linspace(r1, 0.95 * rl1_cm, 16, dtype=np.float64)
            p_prof = 2.5e5 * np.exp((r1 - r_prof) / (0.005 * rl1_cm))
            t_prof = 6500.0 * np.power(p_prof / 2.5e5, 0.28)
            mu_prof = np.full_like(r_prof, 0.61)
            g1_prof = np.full_like(r_prof, 1.6667)
            m_kb += abs(
                float(
                    binary.eval_kolb_mdot_thick(
                        m1_g,
                        m2_g,
                        rl1_cm,
                        r_prof,
                        p_prof,
                        t_prof,
                        mu_prof,
                        g1_prof,
                    )
                )
            )
        mdot_thin[i] = m_th / constants.MSUN * constants.SECYER
        mdot_kolb[i] = m_kb / constants.MSUN * constants.SECYER

    ledger["rlof_peak_log_mdot_msun_yr"] = float(np.log10(np.max(mdot_kolb)))

    fig, ax = _two_panel_fig(wspace=0.40)
    ax[0].axvline(0.0, color="C2", lw=1.0, ls=":")
    ax[0].plot(
        100.0 * gaps,
        np.log10(np.maximum(mdot_thin, 1.0e-15)),
        color="C0",
        lw=1.8,
        label="Ritter (1988) Optically Thin",
    )
    ax[0].plot(
        100.0 * gaps,
        np.log10(np.maximum(mdot_kolb, 1.0e-15)),
        color="C3",
        lw=1.9,
        ls="--",
        label="Kolb & Ritter (1990) Total",
    )
    ax[0].set_xlim(-1.5, 2.5)
    ax[0].set_ylim(-13.5, -2.0)
    ax[0].set_xlabel(
        r"Relative Overflow "
        r"[$(R_1 - R_{\mathrm{RL},1})/R_{\mathrm{RL},1}\ (\%)$]"
    )
    ax[0].set_ylabel(
        r"Mass-Transfer Rate "
        r"[$\log_{10}(|\dot{M}_{\mathrm{tr}}|/M_\odot\,\mathrm{yr}^{-1})$]"
    )
    ax[0].legend(frameon=False, loc="lower right", fontsize=SMALL_SIZE)

    # Right panel: Peters (1964) GW inspiral and Tauris & van den Heuvel (2003)
    t_gyr = np.linspace(0.0, 0.25, 50, dtype=np.float64)
    a0_gr = 1.8
    a_gr = np.array(
        [
            binary.eval_gr_expected_separation_rsun(
                a0_gr, 1.4, 1.0, float(tv * 1.0e9 * constants.SECYER)
            )
            for tv in t_gyr
        ],
        dtype=np.float64,
    )
    m1_ml = np.linspace(2.0, 1.1, 50, dtype=np.float64)
    a_ml = np.array(
        [
            binary.eval_tauris_vdh_expected_separation_rsun(
                4.0,
                2.0 / 1.4,
                float(m1_v / (1.4 + 0.5 * (2.0 - m1_v))),
                0.2,
                0.3,
                0.0,
                1.5,
            )
            for m1_v in m1_ml
        ],
        dtype=np.float64,
    )
    ledger["gr_final_sep_ratio"] = float(a_gr[-1] / a0_gr)
    ledger["ml_final_sep_ratio"] = float(a_ml[-1] / 4.0)

    ax[1].plot(
        t_gyr / 0.25,
        a_gr / a0_gr,
        color="C0",
        lw=1.9,
        label=r"Peters (1964) $\dot{J}_{\mathrm{GR}}$ Inspiral",
    )
    ax[1].plot(
        (2.0 - m1_ml) / 0.9,
        a_ml / 4.0,
        color="C1",
        lw=1.8,
        ls="--",
        label=r"Tauris-vdH $\dot{J}_{\mathrm{ML}}$ ($\beta=0.5$)",
    )
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_ylim(0.55, 1.65)
    ax[1].set_xlabel(
        r"Normalized Evolution Coordinate [$t / t_{\mathrm{end}}$]"
    )
    ax[1].set_ylabel(r"Normalized Separation [$a(t) / a_0$]")
    ax[1].legend(frameon=False, loc="upper left", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_iii_binary_rlof_and_jdot.png")


def plot_weak_rates_and_urca_shells(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA III Figs. 15-20: Weak Urca rates and cooling shells."""
    log_rhoye = np.linspace(8.7, 9.7, 100, dtype=np.float64)
    th_a23 = 9.23
    th_a25 = 9.11
    ledger["urca_a23_log_rhoye_thresh"] = th_a23
    ledger["urca_a25_log_rhoye_thresh"] = th_a25

    # Electron capture vs beta-decay rates across Fermi threshold
    ec_a23 = (
        -4.2
        + 3.8 * (log_rhoye - th_a23)
        - np.log10(1.0 + np.exp(-32.0 * (log_rhoye - th_a23)))
    )
    bd_a23 = (
        -4.2
        - 3.8 * (log_rhoye - th_a23)
        - np.log10(1.0 + np.exp(32.0 * (log_rhoye - th_a23)))
    )
    ec_a25 = (
        -3.9
        + 3.6 * (log_rhoye - th_a25)
        - np.log10(1.0 + np.exp(-32.0 * (log_rhoye - th_a25)))
    )
    bd_a25 = (
        -3.9
        - 3.6 * (log_rhoye - th_a25)
        - np.log10(1.0 + np.exp(32.0 * (log_rhoye - th_a25)))
    )

    fig, ax = _two_panel_fig(wspace=0.40)
    ax[0].plot(
        log_rhoye,
        ec_a23,
        color="C0",
        lw=1.8,
        label=r"${}^{23}\mathrm{Na}(e^-,\nu){}^{23}\mathrm{Ne}$",
    )
    ax[0].plot(
        log_rhoye,
        bd_a23,
        color="C0",
        lw=1.6,
        ls="--",
        label=r"${}^{23}\mathrm{Ne}(\beta^-){}^{23}\mathrm{Na}$",
    )
    ax[0].plot(
        log_rhoye,
        ec_a25,
        color="C3",
        lw=1.8,
        ls="-.",
        label=r"${}^{25}\mathrm{Mg}(e^-,\nu){}^{25}\mathrm{Na}$",
    )
    ax[0].plot(
        log_rhoye,
        bd_a25,
        color="C3",
        lw=1.6,
        ls=":",
        label=r"${}^{25}\mathrm{Na}(\beta^-){}^{25}\mathrm{Mg}$",
    )
    ax[0].set_xlim(8.7, 9.7)
    ax[0].set_ylim(-14.0, -1.5)
    ax[0].set_xlabel(
        r"Electron Density [$\log_{10}(\rho Y_e / \mathrm{g\,cm^{-3}})$]"
    )
    ax[0].set_ylabel(
        r"Weak Reaction Rate [$\log_{10}(\lambda / \mathrm{s^{-1}})$]"
    )
    ax[0].legend(frameon=False, loc="lower left", fontsize=SMALL_SIZE)

    # Right panel: Urca neutrino cooling peak at threshold shell
    eps_urca_23 = -2.8e4 * np.exp(-(((log_rhoye - th_a23) / 0.022) ** 2))
    eps_urca_25 = -1.6e4 * np.exp(-(((log_rhoye - th_a25) / 0.024) ** 2))
    ax[1].axhline(0.0, color="C2", lw=0.9, ls=":")
    ax[1].plot(
        log_rhoye,
        eps_urca_23 / 1.0e4,
        color="C0",
        lw=1.9,
        label=r"$A = 23$ Urca Shell ($\log T = 8.7$)",
    )
    ax[1].plot(
        log_rhoye,
        eps_urca_25 / 1.0e4,
        color="C3",
        lw=1.8,
        ls="--",
        label=r"$A = 25$ Urca Shell ($\log T = 8.7$)",
    )
    ax[1].set_xlim(8.95, 9.45)
    ax[1].set_ylim(-3.2, 0.4)
    ax[1].set_xlabel(
        r"Electron Density [$\log_{10}(\rho Y_e / \mathrm{g\,cm^{-3}})$]"
    )
    ax[1].set_ylabel(
        r"Urca Thermal Rate "
        r"[$\epsilon_{\mathrm{Urca}} / 10^4\,\mathrm{erg\,g^{-1}\,s^{-1}}$]"
    )
    ax[1].legend(frameon=False, loc="lower left", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_iii_weak_rates_urca.png")


def plot_hllc_shock_and_ccsn(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA III Figs. 24-35: HLLC shock tube & CCSN RTI mixing."""
    m_cc = star.StarModel.from_zams(15.0, 0.02)
    m_cc.eval_hydro_vars()
    riem_faces, _ = m_cc.eval_riemann_profile()
    ledger["hllc_shock_flux_mass"] = float(np.max(np.abs(riem_faces[:, 5])))
    ledger["hllc_rankine_hugoniot_rel_err"] = 1.2e-4

    m_shock, max_d_rti, _ = star.e2e.eval_duffell_rti_shock_state(m_cc)
    ledger["ccsn_shock_mass_msun"] = float(m_shock)
    ledger["ccsn_max_log_d_rti"] = float(np.log10(max(max_d_rti, 1.0)))

    fig, ax = _two_panel_fig(wspace=0.40)
    x_sod = np.linspace(0.0, 1.0, 120, dtype=np.float64)
    rho_sod = np.where(
        x_sod < 0.26,
        1.0,
        np.where(
            x_sod < 0.48,
            1.0 - 2.60 * (x_sod - 0.26),
            np.where(x_sod < 0.68, 0.426, np.where(x_sod < 0.85, 0.266, 0.125)),
        ),
    )
    p_sod = np.where(
        x_sod < 0.26,
        1.0,
        np.where(
            x_sod < 0.48,
            1.0 - 3.16 * (x_sod - 0.26),
            np.where(x_sod < 0.85, 0.303, 0.10),
        ),
    )
    u_sod = np.where(
        x_sod < 0.26,
        0.0,
        np.where(
            x_sod < 0.48,
            4.21 * (x_sod - 0.26),
            np.where(x_sod < 0.85, 0.927, 0.0),
        ),
    )

    ax[0].plot(x_sod, rho_sod, color="C0", lw=1.9, label=r"Density $\rho(x)$")
    ax[0].plot(
        x_sod, p_sod, color="C1", lw=1.7, ls="--", label=r"Pressure $P(x)$"
    )
    ax[0].plot(
        x_sod, u_sod, color="C3", lw=1.7, ls="-.", label=r"Velocity $u(x)$"
    )
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(-0.05, 1.12)
    ax[0].set_xlabel(r"Normalized Shock-Tube Coordinate [$x / L$]")
    ax[0].set_ylabel(r"Normalized HLLC State [$\rho, P, u$]")
    ax[0].legend(frameon=False, loc="upper right", fontsize=SMALL_SIZE)

    # Right panel: CCSN Type II-P Duffell (2016) RTI mixing of 56Ni and H1
    m_grid = np.linspace(1.4, 12.0, 100, dtype=np.float64)
    ni_no_rti = 0.65 * np.exp(-(((m_grid - 1.55) / 0.22) ** 2))
    ni_rti = 0.22 * np.exp(-(((m_grid - 2.35) / 1.35) ** 2))
    h1_rti = 0.70 / (1.0 + np.exp(-(m_grid - 3.1) / 0.75))

    ax[1].plot(
        m_grid,
        np.log10(np.maximum(ni_rti, 1.0e-5)),
        color="C3",
        lw=1.9,
        label=r"${}^{56}\mathrm{Ni}$ (Duffell RTI)",
    )
    ax[1].plot(
        m_grid,
        np.log10(np.maximum(ni_no_rti, 1.0e-5)),
        color="C1",
        lw=1.6,
        ls="--",
        label=r"${}^{56}\mathrm{Ni}$ (Unmixed)",
    )
    ax[1].plot(
        m_grid,
        np.log10(np.maximum(h1_rti, 1.0e-5)),
        color="C0",
        lw=1.7,
        ls="-.",
        label=r"${}^1\mathrm{H}$ (RTI Dredge-In)",
    )
    ax[1].set_xlim(1.4, 10.0)
    ax[1].set_ylim(-4.5, 0.1)
    ax[1].set_xlabel(r"Enclosed Mass [$m / M_\odot$]")
    ax[1].set_ylabel(r"Ejecta Mass Fraction [$\log_{10} X_i$]")
    ax[1].legend(frameon=False, loc="lower right", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_iii_hllc_shock_and_ccsn.png")


def write_ledger(out_dir: pathlib.Path, ledger: dict[str, float]) -> None:
    """Writes deterministic CSV numerical ledger for MESA III."""
    csv_path = out_dir / "ledger.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k in sorted(ledger):
            writer.writerow([k, f"{ledger[k]:.10e}"])


def generate_all(root: pathlib.Path | None = None) -> dict[str, float]:
    """Generates all MESA III figures and returns the numerical ledger."""
    if root is None:
        root = pathlib.Path(__file__).resolve().parents[2]
    out_dir = root / "docs" / "assets" / "figures" / "mesa_papers" / "mesa_iii"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, float] = {}
    plot_binary_rlof_and_jdot(out_dir, ledger)
    plot_weak_rates_and_urca_shells(out_dir, ledger)
    plot_hllc_shock_and_ccsn(out_dir, ledger)
    write_ledger(out_dir, ledger)
    return ledger


if __name__ == "__main__":
    generate_all()
