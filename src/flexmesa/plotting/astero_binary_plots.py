# SPDX-License-Identifier: LGPL-3.0-or-later
"""Asteroseismic and binary evolution diagnostic plots.

Provides echelle diagrams (`plot_echelle_diagram`), frequency separation
ratio plots (`plot_separation_ratios`), and binary orbital/RLOF evolution
multi-panel plots (`plot_binary_evolution`).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from flexmesa.plotting import style


def plot_echelle_diagram(
    obs_freqs: Any,
    obs_sigmas: Any,
    delta_nu: float,
    *,
    model_freqs_uncorr: Any | None = None,
    model_freqs_corr: Any | None = None,
    degrees: Any | None = None,
    ax: Any | None = None,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """Plots an asteroseismic echelle diagram.

    Args:
        obs_freqs: Observed frequencies (`(M,)`) [`uHz`].
        obs_sigmas: Observed frequency uncertainties (`(M,)`) [`uHz`].
        delta_nu: Large frequency separation $\\Delta\\nu$ [`uHz`].
        model_freqs_uncorr: Optional uncorrected model frequencies (`(M,)`)
            [`uHz`].
        model_freqs_corr: Optional surface-corrected model frequencies (`(M,)`)
            [`uHz`].
        degrees: Optional spherical degrees $l_i \\in \\{0, 1, 2, 3\\}$
            (`(M,)`).
        ax: Optional pre-existing `matplotlib.axes.Axes`.
        save_path: Optional output path (`300 dpi`).

    Returns:
        Tuple `(fig, ax)` of `matplotlib` `Figure` and `Axes`.
    """
    fig, ax = style.init_ax(ax, "single", 0.90)
    f_obs = np.asarray(obs_freqs, dtype=np.float64)
    s_obs = np.asarray(obs_sigmas, dtype=np.float64)
    dnu = max(float(delta_nu), 1.0e-12)
    ell = (
        np.zeros(f_obs.size, dtype=np.int64)
        if degrees is None
        else np.asarray(degrees, dtype=np.int64)
    )
    f_unc_all = (
        None
        if model_freqs_uncorr is None
        else np.asarray(model_freqs_uncorr, dtype=np.float64)
    )
    f_cor_all = (
        None
        if model_freqs_corr is None
        else np.asarray(model_freqs_corr, dtype=np.float64)
    )

    deg_colors = {
        0: style.OKABE_ITO["blue"],
        1: style.OKABE_ITO["vermilion"],
        2: style.OKABE_ITO["bluish_green"],
        3: style.OKABE_ITO["orange"],
    }
    for l_val in [int(v) for v in np.unique(ell)]:
        mask = ell == l_val
        col = deg_colors.get(l_val, style.OKABE_ITO["black"])
        ax.errorbar(
            np.mod(f_obs[mask], dnu),
            f_obs[mask],
            xerr=s_obs[mask],
            fmt="o",
            ms=4.5,
            color=col,
            label=f"Observed [$l={l_val}$]",
            zorder=4,
        )
        if f_unc_all is not None:
            f_unc = f_unc_all[mask]
            ax.scatter(
                np.mod(f_unc, dnu),
                f_unc,
                facecolors="none",
                edgecolors=col,
                marker="s",
                s=28,
                lw=0.9,
                label=f"Uncorrected [$l={l_val}$]" if l_val == 0 else None,
                zorder=2,
            )
        if f_cor_all is not None:
            f_cor = f_cor_all[mask]
            ax.scatter(
                np.mod(f_cor, dnu),
                f_cor,
                color=col,
                marker="^",
                s=28,
                alpha=0.85,
                label=f"Corrected [$l={l_val}$]" if l_val == 0 else None,
                zorder=3,
            )

    ax.set_xlim(0.0, dnu)
    ax.set_xlabel(
        "Frequency Modulo $\\Delta\\nu$ "
        "[$(\\nu\\,\\mathrm{mod}\\,\\Delta\\nu) / \\mu\\mathrm{Hz}$]"
    )
    ax.set_ylabel("Oscillation Frequency [$\\nu / \\mu\\mathrm{Hz}$]")
    ax.legend(loc="best", frameon=False)
    return style.finalize_fig(fig, ax, save_path)


def plot_separation_ratios(
    freqs_r010: Any,
    r010_obs: Any,
    r010_sigmas: Any,
    r010_model: Any | None = None,
    *,
    freqs_r02: Any | None = None,
    r02_obs: Any | None = None,
    r02_sigmas: Any | None = None,
    r02_model: Any | None = None,
    ax: Any | None = None,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """Plots Roxburgh & Vorontsov (2003) separation ratios ($r_{010}, r_{02}$).

    Args:
        freqs_r010: Reference frequencies for $r_{010}$ (`(K,)`) [`uHz`].
        r010_obs: Observed $r_{010}$ ratios (`(K,)`).
        r010_sigmas: Observed $r_{010}$ uncertainties (`(K,)`).
        r010_model: Optional model $r_{010}$ ratios (`(K,)`).
        freqs_r02: Optional reference frequencies for $r_{02}$ (`(K_2,)`)
            [`uHz`].
        r02_obs: Optional observed $r_{02}$ ratios (`(K_2,)`).
        r02_sigmas: Optional observed $r_{02}$ uncertainties (`(K_2,)`).
        r02_model: Optional model $r_{02}$ ratios (`(K_2,)`).
        ax: Optional pre-existing `matplotlib.axes.Axes`.
        save_path: Optional output path (`300 dpi`).

    Returns:
        Tuple `(fig, ax)` of `matplotlib` `Figure` and `Axes`.
    """
    fig, ax = style.init_ax(ax, "single", 0.78)
    f010 = np.asarray(freqs_r010, dtype=np.float64)
    y010 = np.asarray(r010_obs, dtype=np.float64)
    s010 = np.asarray(r010_sigmas, dtype=np.float64)

    ax.errorbar(
        f010,
        y010,
        yerr=s010,
        fmt="o",
        ms=4.5,
        color=style.OKABE_ITO["blue"],
        label="Observed [$r_{010}$]",
        zorder=3,
    )
    if r010_model is not None:
        ax.plot(
            f010,
            np.asarray(r010_model, dtype=np.float64),
            color=style.OKABE_ITO["blue"],
            lw=1.5,
            label="Model [$r_{010}$]",
            zorder=4,
        )
    if freqs_r02 is not None and r02_obs is not None and r02_sigmas is not None:
        f02 = np.asarray(freqs_r02, dtype=np.float64)
        y02 = np.asarray(r02_obs, dtype=np.float64)
        s02 = np.asarray(r02_sigmas, dtype=np.float64)
        ax.errorbar(
            f02,
            y02,
            yerr=s02,
            fmt="s",
            ms=4.5,
            color=style.OKABE_ITO["vermilion"],
            label="Observed [$r_{02}$]",
            zorder=3,
        )
        if r02_model is not None:
            ax.plot(
                f02,
                np.asarray(r02_model, dtype=np.float64),
                color=style.OKABE_ITO["vermilion"],
                ls="--",
                lw=1.5,
                label="Model [$r_{02}$]",
                zorder=4,
            )

    ax.set_xlabel("Oscillation Frequency [$\\nu / \\mu\\mathrm{Hz}$]")
    ax.set_ylabel("Frequency Separation Ratio [$r_{010},\\,r_{02}$]")
    ax.legend(loc="best", frameon=False)
    return style.finalize_fig(fig, ax, save_path)


def plot_binary_evolution(
    age_myr: Any,
    period_days: Any,
    lg_mtransfer_rate: Any,
    eccentricity: Any | None = None,
    *,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """Plots a multi-panel binary orbital and RLOF evolution summary.

    Args:
        age_myr: Binary system ages (`(N_t,)`) [`Myr`].
        period_days: Orbital periods (`(N_t,)`) [`days`].
        lg_mtransfer_rate: Log mass-transfer rate
            $\\log_{10}(|\\dot{M}| / M_\\odot\\,\\mathrm{yr}^{-1})$ (`(N_t,)`).
        eccentricity: Optional orbital eccentricity (`(N_t,)`).
        save_path: Optional output path (`300 dpi`).

    Returns:
        Tuple `(fig, axes)` of `matplotlib` `Figure` and array of `Axes`.
    """
    style.use_mesa_plot_style()
    t_arr = np.asarray(age_myr, dtype=np.float64)
    p_arr = np.asarray(period_days, dtype=np.float64)
    mdot_arr = np.asarray(lg_mtransfer_rate, dtype=np.float64)
    n_rows = 3 if eccentricity is not None else 2
    fig, axes = plt.subplots(
        n_rows,
        1,
        sharex=True,
        figsize=style.figsize("single", 0.55 * n_rows),
    )
    axes[0].plot(
        t_arr,
        p_arr,
        color=style.OKABE_ITO["blue"],
        lw=1.5,
        label="Orbital Period",
    )
    axes[0].set_ylabel("Period [$P_{\\mathrm{orb}} / \\mathrm{d}$]")
    axes[0].legend(loc="best", frameon=False)

    axes[1].plot(
        t_arr,
        mdot_arr,
        color=style.OKABE_ITO["vermilion"],
        lw=1.5,
        label="RLOF Rate",
    )
    axes[1].set_ylabel(
        "Mass Transfer "
        "[$\\log_{10}(|\\dot{M}| / M_\\odot\\,\\mathrm{yr}^{-1})$]"
    )
    axes[1].legend(loc="best", frameon=False)

    if eccentricity is not None:
        axes[2].plot(
            t_arr,
            np.asarray(eccentricity, dtype=np.float64),
            color=style.OKABE_ITO["bluish_green"],
            lw=1.5,
            label="Eccentricity",
        )
        axes[2].set_ylabel("Eccentricity [$e$]")
        axes[2].legend(loc="best", frameon=False)

    axes[-1].set_xlabel("Binary Age [$\\mathrm{Myr}$]")
    fig.align_ylabels(axes)
    return style.finalize_fig(fig, axes, save_path)
