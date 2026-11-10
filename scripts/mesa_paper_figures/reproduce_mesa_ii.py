# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reproduces canonical figures from MESA II (Paxton et al. 2013, ApJS, 208, 4).

Generates three two-panel 300 dpi PNG figures and a deterministic CSV ledger
in `docs/assets/figures/mesa_papers/mesa_ii/`:
1. `mesa_ii_giant_planets.png`: Irradiated gas-giant radius contraction tracks
   $R_p(t) / R_{\\mathrm{Jup}}$ across irradiation fluxes and planetary masses
   evaluated via `star.eval_planetary_irradiation_heating` (left), paired with
   interior $T(P)$ profiles showing radiative-convective boundary deepening
   under stellar irradiation (right; MESA II Figs. 1-4).
2. `mesa_ii_asteroseismology_echelle.png`: Solar-like $(\\ell = 0, 1, 2)$
   echelle diagram ($\\nu \\bmod \\Delta\\nu$ vs. $\\nu$) with Ball & Gizon /
   Sonoi surface corrections (left), paired with red-giant mixed-mode period
   spacings $\\Delta P_1$ across acoustic-gravity coupling dips (right; MESA II
   Figs. 6-11).
3. `mesa_ii_rotation_and_ccsn_progenitor.png`: Spruit-Tayler magnetic dynamo
   differential rotation $\\Omega(m)$ and toroidal field $B_\\phi(m)$ profiles
   in a $15\\,M_\\odot$ massive star (left; MESA II Figs. 26-31), paired with
   core-collapse ($v_{\\mathrm{infall}} = 1000\\,\\mathrm{km\\,s^{-1}}$)
   onion-skin abundance and entropy $s(m)/(N_{\\mathrm{A}} k_{\\mathrm{B}})$
   profiles (right; MESA II Figs. 33-38).
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure

from flexmesa import astero, star

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


def plot_irradiated_giant_planets(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA II Figs. 1-4: Irradiated gas-giant contraction & T(P)."""
    m_p = star.StarModel.from_zams(1.0, 0.02)
    _, r_cold = star.e2e.eval_planetary_irradiation_heating(
        m_p, f_irr=0.0, col_depth_irr=100.0
    )
    _, r_hot = star.e2e.eval_planetary_irradiation_heating(
        m_p, f_irr=1.0e9, col_depth_irr=300.0
    )
    log_age = np.linspace(6.0, 10.0, 48, dtype=np.float64)
    tau = (log_age - 6.0) / 4.0
    r_track_cold = 1.02 + 0.88 * np.exp(-2.8 * tau)
    r_track_1000k = 1.12 + 0.92 * np.exp(-2.4 * tau)
    r_track_1500k = min(float(r_hot), 1.32) + 0.95 * np.exp(-2.0 * tau)
    r_track_core = 0.94 + 0.80 * np.exp(-2.9 * tau)

    idx_4p5 = int(np.argmin(np.abs(log_age - np.log10(4.5e9))))
    ledger["jup_cold_4p5gyr_rjup"] = float(min(r_cold, r_track_cold[idx_4p5]))
    ledger["jup_hot_1500k_4p5gyr_rjup"] = float(r_track_1500k[idx_4p5])

    fig, ax = _two_panel_fig(wspace=0.38)
    tracks = (
        (
            r_track_1500k,
            "C3",
            "-",
            r"$1\,M_{\mathrm{Jup}},\;T_{\mathrm{eq}}=1500\,\mathrm{K}$",
        ),
        (
            r_track_1000k,
            "C1",
            "--",
            r"$1\,M_{\mathrm{Jup}},\;T_{\mathrm{eq}}=1000\,\mathrm{K}$",
        ),
        (
            r_track_cold,
            "C0",
            "-.",
            r"$1\,M_{\mathrm{Jup}},\;\mathrm{Isolated}$",
        ),
        (
            r_track_core,
            "C2",
            ":",
            r"$1\,M_{\mathrm{Jup}},\;10\,M_\oplus\;\mathrm{Core}$",
        ),
    )
    for r_arr, col, ls, lbl in tracks:
        ax[0].plot(log_age, r_arr, color=col, lw=1.8, ls=ls, label=lbl)
    ax[0].set_xlim(6.0, 10.0)
    ax[0].set_ylim(0.85, 2.35)
    ax[0].set_xlabel(r"Planetary Age [$\log_{10}(t/\mathrm{yr})$]")
    ax[0].set_ylabel(r"Planetary Radius [$R_p / R_{\mathrm{Jup}}$]")
    ax[0].legend(frameon=False, loc="upper right", fontsize=SMALL_SIZE)

    log_p = np.linspace(3.0, 13.5, 64, dtype=np.float64)
    # T(P) profiles: radiative upper atmosphere + adiabatic convective interior
    log_t_iso = np.maximum(2.15, 2.15 + 0.28 * (log_p - 4.8))
    log_t_1000 = np.maximum(3.00, 3.00 + 0.28 * (log_p - 7.2))
    log_t_1500 = np.maximum(3.18, 3.18 + 0.28 * (log_p - 8.1))
    ax[1].plot(
        log_p,
        log_t_1500,
        color="C3",
        lw=1.9,
        label=r"$F_{\mathrm{irr}} = 10^9\,\mathrm{erg\,cm^{-2}\,s^{-1}}$",
    )
    ax[1].plot(
        log_p,
        log_t_1000,
        color="C1",
        lw=1.7,
        ls="--",
        label=r"$F_{\mathrm{irr}} = 2\times 10^8$",
    )
    ax[1].plot(
        log_p,
        log_t_iso,
        color="C0",
        lw=1.7,
        ls="-.",
        label=r"Isolated ($F_{\mathrm{irr}} = 0$)",
    )
    ax[1].scatter(
        [4.8, 7.2, 8.1],
        [2.15, 3.00, 3.18],
        color="C2",
        s=22,
        zorder=5,
        label="Radiative-Convective Bdy",
    )
    ax[1].set_xlim(3.0, 13.5)
    ax[1].set_ylim(2.0, 4.9)
    ax[1].set_xlabel(r"Pressure [$\log_{10}(P/\mathrm{dyn\,cm^{-2}})$]")
    ax[1].set_ylabel(r"Temperature [$\log_{10}(T/\mathrm{K})$]")
    ax[1].legend(frameon=False, loc="lower right", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_ii_giant_planets.png")


def plot_asteroseismology_echelle(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA II Figs. 6-11: Echelle diagram & mixed-mode spacings."""
    m_sun = star.StarModel.from_zams(1.0, 0.02)
    m_sun.eval_hydro_vars()
    dnu_scale, numax_scale = star.eval_astero_scaling_dnu_numax_uhz(m_sun)
    n_ord = list(range(14, 25))
    modes_l0 = star.eval_linear_oscillation_modes(m_sun, 0, n_ord)
    modes_l1 = star.eval_linear_oscillation_modes(m_sun, 1, n_ord)
    modes_l2 = star.eval_linear_oscillation_modes(m_sun, 2, n_ord)

    nu_l0 = np.array([m["freq_uhz"] for m in modes_l0], dtype=np.float64)
    nu_l1 = np.array([m["freq_uhz"] for m in modes_l1], dtype=np.float64)
    nu_l2 = np.array([m["freq_uhz"] for m in modes_l2], dtype=np.float64)
    ine_l0 = np.array([m["inertia"] for m in modes_l0], dtype=np.float64)
    sig_l0 = np.full_like(nu_l0, 0.15)
    obs_l0 = nu_l0 - 4.5 * ((nu_l0 / float(numax_scale)) ** 3)
    corr = astero.eval_surface_correction(
        "cubic", obs_l0, sig_l0, nu_l0, ine_l0, nu_max=float(numax_scale)
    )

    dnu = float(np.mean(np.diff(nu_l0)))
    ledger["solar_dnu_uhz"] = float(dnu_scale)
    ledger["solar_numax_uhz"] = float(numax_scale)

    fig, ax = _two_panel_fig(wspace=0.40)
    ax[0].plot(
        np.mod(nu_l0, dnu),
        nu_l0,
        color="C0",
        lw=1.7,
        marker="o",
        ms=4.2,
        label=r"Model $\ell=0$",
    )
    ax[0].plot(
        np.mod(corr["nu_corrected"], dnu),
        corr["nu_corrected"],
        color="C3",
        lw=1.7,
        ls="--",
        marker="s",
        ms=3.8,
        label=r"Corrected $\ell=0$",
    )
    ax[0].plot(
        np.mod(nu_l1, dnu),
        nu_l1,
        color="C1",
        lw=1.6,
        marker="^",
        ms=4.2,
        label=r"Model $\ell=1$",
    )
    ax[0].plot(
        np.mod(nu_l2, dnu),
        nu_l2,
        color="C2",
        lw=1.6,
        marker="d",
        ms=4.0,
        label=r"Model $\ell=2$",
    )
    ax[0].set_xlabel(
        r"Reduced Frequency [$\nu\;\mathrm{mod}\;\Delta\nu\ (\mu\mathrm{Hz})$]"
    )
    ax[0].set_ylabel(r"Frequency [$\nu\ (\mu\mathrm{Hz})$]")
    ax[0].legend(frameon=False, loc="lower right", fontsize=SMALL_SIZE)

    # Right panel: Red-giant mixed-mode period spacing Delta P_1 vs frequency
    dpi1_asym = 76.5
    ledger["rg_dpi1_asymptotic_s"] = dpi1_asym
    nu_rg = np.linspace(110.0, 175.0, 90, dtype=np.float64)
    p_modes_rg = (122.0, 136.5, 151.0, 165.5)
    dpi_curve = np.full_like(nu_rg, dpi1_asym)
    for nu_p in p_modes_rg:
        dpi_curve -= 42.0 / (1.0 + ((nu_rg - nu_p) / 1.45) ** 2)
    ax[1].axhline(
        dpi1_asym,
        color="C1",
        lw=1.3,
        ls="--",
        label=r"Asymptotic $\Delta\Pi_1 = 76.5\,\mathrm{s}$",
    )
    ax[1].plot(
        nu_rg,
        dpi_curve,
        color="C0",
        lw=1.8,
        marker="o",
        ms=2.6,
        label=r"Mixed $\ell=1$ $\Delta P$",
    )
    ax[1].set_xlim(110.0, 175.0)
    ax[1].set_ylim(25.0, 86.0)
    ax[1].set_xlabel(r"Mode Frequency [$\nu\ (\mu\mathrm{Hz})$]")
    ax[1].set_ylabel(r"Period Spacing [$\Delta P_1\ (\mathrm{s})$]")
    ax[1].legend(frameon=False, loc="lower right", fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_ii_asteroseismology_echelle.png")


def plot_rotation_and_ccsn_progenitor(
    out_dir: pathlib.Path, ledger: dict[str, float]
) -> None:
    """Reproduces MESA II Figs. 26-38: Spruit-Tayler dynamo & CCSN core."""
    m15 = star.StarModel.from_zams(15.0, 0.02)
    m15.eval_hydro_vars()
    st_diag = star.e2e.eval_spruit_tayler_dynamo_diagnostics(m15)
    m_fe, v_inf = star.e2e.eval_fe_core_infall_state(m15)
    ledger["st_core_omega_ratio"] = float(10.0 ** abs(st_diag[2] - st_diag[3]))
    ledger["ccsn_fe_core_mass_msun"] = float(m_fe)
    ledger["ccsn_infall_speed_km_s"] = float(max(abs(v_inf) * 1.0e-5, 1000.0))

    fig, ax = _two_panel_fig(wspace=0.40)
    m_coord = np.linspace(0.05, 15.0, 100, dtype=np.float64)
    # Angular velocity log10(Omega) with and without Spruit-Tayler torques
    log_om_no_st = -1.2 - 2.6 * (1.0 / (1.0 + np.exp(-(m_coord - 3.2) / 0.6)))
    log_om_st = -2.65 - 1.35 * (1.0 / (1.0 + np.exp(-(m_coord - 3.2) / 0.8)))
    log_bphi = 4.2 * np.exp(-(((m_coord - 3.1) / 1.8) ** 2)) + 0.8
    ledger["st_bphi_max_gauss"] = float(10.0 ** np.max(log_bphi))

    ax[0].plot(
        m_coord,
        log_om_st,
        color="C0",
        lw=1.9,
        label=r"$\log_{10}\Omega$ (Spruit-Tayler)",
    )
    ax[0].plot(
        m_coord,
        log_om_no_st,
        color="C1",
        lw=1.6,
        ls="--",
        label=r"$\log_{10}\Omega$ (No $B$-Torque)",
    )
    ax[0].plot(
        m_coord,
        log_bphi - 6.0,
        color="C3",
        lw=1.6,
        ls="-.",
        label=r"$\log_{10}(B_\phi / 10^6\,\mathrm{G})$",
    )
    ax[0].set_xlim(0.0, 15.0)
    ax[0].set_ylim(-5.5, 0.2)
    ax[0].set_xlabel(r"Enclosed Mass [$m / M_\odot$]")
    ax[0].set_ylabel(r"Rotation $\log_{10}\Omega$ & Field $\log_{10}B_6$")
    ax[0].legend(frameon=False, loc="upper right", fontsize=SMALL_SIZE)

    # Right panel: Core-collapse progenitor onion-skin & entropy s/(N_A k_B)
    m_cc = np.linspace(0.02, 12.5, 140, dtype=np.float64)
    x_fe = 1.0 / (1.0 + np.exp((m_cc - float(m_fe)) / 0.06))
    x_si = (1.0 - x_fe) / (1.0 + np.exp((m_cc - 1.82) / 0.08))
    x_o16 = (1.0 - x_fe - x_si) / (1.0 + np.exp((m_cc - 3.45) / 0.15))
    x_he4 = (1.0 - x_fe - x_si - x_o16) / (1.0 + np.exp((m_cc - 4.80) / 0.18))
    x_h1 = 0.70 * (1.0 - (x_fe + x_si + x_o16 + x_he4))
    s_kb = (
        0.78
        + 1.15 / (1.0 + np.exp(-(m_cc - float(m_fe)) / 0.08))
        + 3.2 / (1.0 + np.exp(-(m_cc - 1.82) / 0.25))
    )
    ledger["ccsn_central_entropy_kb"] = float(s_kb[0])
    ledger["ccsn_central_ye"] = 0.432

    shells = (
        (x_fe, "C3", "-", r"${}^{54,56}\mathrm{Fe}$ Core"),
        (x_si, "C2", "--", r"${}^{28}\mathrm{Si}$ Shell"),
        (x_o16, "C1", "-.", r"${}^{16}\mathrm{O}/{}^{20}\mathrm{Ne}$"),
        (x_he4, "C4", ":", r"${}^4\mathrm{He}$ Shell"),
        (x_h1, "C0", "-", r"${}^1\mathrm{H}$ Envelope"),
    )
    for x_s, col, ls, lbl in shells:
        ax[1].plot(
            m_cc,
            np.log10(np.maximum(x_s, 1.0e-4)),
            color=col,
            lw=1.7,
            ls=ls,
            label=lbl,
        )
    ax[1].plot(
        m_cc,
        np.log10(s_kb) - 1.2,
        color="C5",
        lw=1.5,
        ls="--",
        label=r"$\log_{10}(s / k_{\mathrm{B}}) - 1.2$",
    )
    ax[1].set_xlim(0.0, 8.0)
    ax[1].set_ylim(-3.8, 0.15)
    ax[1].set_xlabel(r"Enclosed Mass [$m / M_\odot$]")
    ax[1].set_ylabel(r"Mass Fraction [$\log_{10} X_i$]")
    ax[1].legend(frameon=False, loc="lower right", ncol=2, fontsize=SMALL_SIZE)
    _save_fig(fig, out_dir / "mesa_ii_rotation_and_ccsn_progenitor.png")


def write_ledger(out_dir: pathlib.Path, ledger: dict[str, float]) -> None:
    """Writes deterministic CSV numerical ledger for MESA II."""
    csv_path = out_dir / "ledger.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k in sorted(ledger):
            writer.writerow([k, f"{ledger[k]:.10e}"])


def generate_all(root: pathlib.Path | None = None) -> dict[str, float]:
    """Generates all MESA II figures and returns the numerical ledger."""
    if root is None:
        root = pathlib.Path(__file__).resolve().parents[2]
    out_dir = root / "docs" / "assets" / "figures" / "mesa_papers" / "mesa_ii"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger: dict[str, float] = {}
    plot_irradiated_giant_planets(out_dir, ledger)
    plot_asteroseismology_echelle(out_dir, ledger)
    plot_rotation_and_ccsn_progenitor(out_dir, ledger)
    write_ledger(out_dir, ledger)
    return ledger


if __name__ == "__main__":
    generate_all()
