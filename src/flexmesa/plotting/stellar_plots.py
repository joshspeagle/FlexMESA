# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stellar structure and evolution diagnostic plots (`pgstar` replacements).

Provides Hertzsprung-Russell (`plot_hr_diagram`), central temperature-density
(`plot_tc_rhoc_diagram`), Kippenhahn (`plot_kippenhahn`), and asteroseismic
propagation (`plot_propagation_diagram`) figures styled to journal standards.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from flexmesa.plotting import style


def plot_hr_diagram(
    log_teff: Any,
    log_l: Any,
    *,
    color_val: Any | None = None,
    color_label: str = "Center Hydrogen Fraction [$X_c$]",
    vmin: float | None = None,
    vmax: float | None = None,
    ax: Any | None = None,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """Plots a Hertzsprung-Russell diagram.

    Args:
        log_teff: Sequence of $\\log_{10}(T_{\\mathrm{eff}} / \\mathrm{K})$
            (`(N_t,)`).
        log_l: Sequence of $\\log_{10}(L / L_\\odot)$ (`(N_t,)`).
        color_val: Optional scalar sequence (`(N_t,)`) coloring the track.
        color_label: Colorbar label when `color_val` is provided.
        vmin: Explicit minimum colorbar limit.
        vmax: Explicit maximum colorbar limit.
        ax: Optional pre-existing `matplotlib.axes.Axes`.
        save_path: Optional output path (`300 dpi`).

    Returns:
        Tuple `(fig, ax)` of `matplotlib` `Figure` and `Axes`.
    """
    fig, ax = style.init_ax(ax, "single", 0.85)
    t_arr = np.asarray(log_teff, dtype=np.float64)
    l_arr = np.asarray(log_l, dtype=np.float64)

    ax.plot(
        t_arr,
        l_arr,
        color=style.OKABE_ITO["grey"],
        lw=1.1,
        alpha=0.7,
        zorder=2,
    )
    if color_val is not None:
        c_arr = np.asarray(color_val, dtype=np.float64)
        c_min = float(np.min(c_arr)) if vmin is None else float(vmin)
        c_max = float(np.max(c_arr)) if vmax is None else float(vmax)
        sc = ax.scatter(
            t_arr,
            l_arr,
            c=c_arr,
            cmap="viridis",
            vmin=c_min,
            vmax=c_max,
            s=14,
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_label)
    else:
        ax.plot(
            t_arr,
            l_arr,
            color=style.OKABE_ITO["blue"],
            lw=1.6,
            zorder=3,
            label="Evolutionary Track",
        )
        ax.scatter(
            [t_arr[0]],
            [l_arr[0]],
            color=style.OKABE_ITO["bluish_green"],
            s=28,
            zorder=4,
            label="ZAMS",
        )
        ax.scatter(
            [t_arr[-1]],
            [l_arr[-1]],
            color=style.OKABE_ITO["vermilion"],
            s=28,
            zorder=4,
            label="Terminal",
        )
        ax.legend(loc="best", frameon=False)

    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel(
        "Effective Temperature [$\\log_{10}(T_{\\mathrm{eff}} / \\mathrm{K})$]"
    )
    ax.set_ylabel("Luminosity [$\\log_{10}(L / L_\\odot)$]")
    return style.finalize_fig(fig, ax, save_path)


def plot_tc_rhoc_diagram(
    log_rhoc: Any,
    log_tc: Any,
    *,
    label: str = "Core Trajectory",
    show_burn_lines: bool = True,
    ax: Any | None = None,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """Plots the central temperature-density trajectory.

    Args:
        log_rhoc: Sequence of $\\log_{10}(\\rho_c / \\mathrm{g\\,cm^{-3}})$
            (`(N_t,)`).
        log_tc: Sequence of $\\log_{10}(T_c / \\mathrm{K})$ (`(N_t,)`).
        label: Legend label for the core trajectory.
        show_burn_lines: Whether to draw approximate H/He/C ignition thresholds.
        ax: Optional pre-existing `matplotlib.axes.Axes`.
        save_path: Optional output path (`300 dpi`).

    Returns:
        Tuple `(fig, ax)` of `matplotlib` `Figure` and `Axes`.
    """
    fig, ax = style.init_ax(ax, "single", 0.85)
    rho_arr = np.asarray(log_rhoc, dtype=np.float64)
    t_arr = np.asarray(log_tc, dtype=np.float64)

    ax.plot(
        rho_arr,
        t_arr,
        color=style.OKABE_ITO["blue"],
        lw=1.6,
        label=label,
        zorder=3,
    )
    if show_burn_lines:
        for t_ign, name in [
            (7.15, "H burn"),
            (8.05, "He burn"),
            (8.70, "C burn"),
        ]:
            ax.axhline(
                t_ign,
                color=style.OKABE_ITO["grey"],
                ls="--",
                lw=0.8,
                alpha=0.6,
                zorder=1,
            )
            ax.text(
                float(np.min(rho_arr)) + 0.05,
                t_ign + 0.03,
                name,
                fontsize=8,
                color=style.OKABE_ITO["grey"],
            )
    ax.set_xlabel(
        "Central Density [$\\log_{10}(\\rho_c / \\mathrm{g\\,cm^{-3}})$]"
    )
    ax.set_ylabel("Central Temperature [$\\log_{10}(T_c / \\mathrm{K})$]")
    ax.legend(loc="best", frameon=False)
    return style.finalize_fig(fig, ax, save_path)


def plot_kippenhahn(
    age_myr: Any,
    mass_grid_msun: Any,
    mix_type_grid: Any,
    eps_nuc_grid: Any | None = None,
    total_mass_msun: Any | None = None,
    *,
    vmin: float = 0.0,
    vmax: float = 6.0,
    ax: Any | None = None,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """Plots a Kippenhahn convective/burning structure diagram.

    Args:
        age_myr: Stellar ages (`(N_t,)`) [`Myr`].
        mass_grid_msun: Enclosed mass coordinates (`(N_m,)`) [`M_sun`].
        mix_type_grid: Mixing type integer grid (`(N_m, N_t)`, `1` =
            convective).
        eps_nuc_grid: Optional nuclear energy generation grid (`(N_m, N_t)`).
        total_mass_msun: Optional total stellar mass curve (`(N_t,)`) [`M_sun`].
        vmin: Minimum colorbar limit for $\\log_{10} \\epsilon_{\\mathrm{nuc}}$.
        vmax: Maximum colorbar limit for $\\log_{10} \\epsilon_{\\mathrm{nuc}}$.
        ax: Optional pre-existing `matplotlib.axes.Axes`.
        save_path: Optional output path (`300 dpi`).

    Returns:
        Tuple `(fig, ax)` of `matplotlib` `Figure` and `Axes`.
    """
    fig, ax = style.init_ax(ax, "double", 0.55)
    t_arr = np.asarray(age_myr, dtype=np.float64)
    m_arr = np.asarray(mass_grid_msun, dtype=np.float64)
    mix_arr = np.asarray(mix_type_grid, dtype=np.float64)

    if eps_nuc_grid is not None:
        eps_arr = np.asarray(eps_nuc_grid, dtype=np.float64)
        mesh = ax.pcolormesh(
            t_arr,
            m_arr,
            np.clip(eps_arr, vmin, vmax),
            cmap="Blues",
            vmin=float(vmin),
            vmax=float(vmax),
            shading="auto",
            zorder=1,
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(
            "Nuclear Energy Generation "
            "[$\\log_{10}(\\epsilon_{\\mathrm{nuc}} / "
            "\\mathrm{erg\\,g^{-1}\\,s^{-1}})$]"
        )

    ax.contourf(
        t_arr,
        m_arr,
        (mix_arr == 1.0).astype(np.float64),
        levels=[0.5, 1.5],
        colors=["none"],
        hatches=["///"],
        zorder=2,
    )
    if total_mass_msun is not None:
        ax.plot(
            t_arr,
            np.asarray(total_mass_msun, dtype=np.float64),
            color=style.OKABE_ITO["black"],
            lw=1.4,
            label="Surface Mass [$M(t)$]",
            zorder=3,
        )
        ax.legend(loc="upper right", frameon=False)

    ax.set_xlabel("Stellar Age [$\\mathrm{Myr}$]")
    ax.set_ylabel("Enclosed Mass [$M_r / M_\\odot$]")
    return style.finalize_fig(fig, ax, save_path)


def plot_propagation_diagram(
    r_over_rstar: Any,
    brunt_n2: Any,
    csound: Any,
    r_cm: Any,
    *,
    ell_values: tuple[int, ...] = (1, 2),
    nu_max_uhz: float | None = None,
    ax: Any | None = None,
    save_path: str | None = None,
) -> tuple[Any, Any]:
    """Plots an asteroseismic propagation diagram ($N/2\\pi$ and $S_l/2\\pi$).

    Args:
        r_over_rstar: Normalized radial coordinate $r / R_*$ (`(N,)`).
        brunt_n2: Squared Brunt-Vaisala frequency $N^2$ (`(N,)`) [`s^-2`].
        csound: Adiabatic sound speed $c_s$ (`(N,)`) [`cm s^-1`].
        r_cm: Radial coordinate $r$ (`(N,)`) [`cm`].
        ell_values: Spherical degrees $l$ for Lamb frequencies $S_l$.
        nu_max_uhz: Optional $\\nu_{\\max}$ reference line [`uHz`].
        ax: Optional pre-existing `matplotlib.axes.Axes`.
        save_path: Optional output path (`300 dpi`).

    Returns:
        Tuple `(fig, ax)` of `matplotlib` `Figure` and `Axes`.
    """
    fig, ax = style.init_ax(ax, "single", 0.82)
    x_arr = np.asarray(r_over_rstar, dtype=np.float64)
    n2_arr = np.asarray(brunt_n2, dtype=np.float64)
    cs_arr = np.asarray(csound, dtype=np.float64)
    r_arr = np.maximum(np.asarray(r_cm, dtype=np.float64), 1.0)

    rad_to_uhz = 1.0e6 / (2.0 * np.pi)
    n_uhz = np.sqrt(np.maximum(n2_arr, 0.0)) * rad_to_uhz
    ax.plot(
        x_arr,
        np.log10(np.maximum(n_uhz, 1.0e-2)),
        color=style.OKABE_ITO["vermilion"],
        lw=1.6,
        label="Brunt-Vaisala [$N / 2\\pi$]",
        zorder=3,
    )
    palette = [style.OKABE_ITO["blue"], style.OKABE_ITO["bluish_green"]]
    for idx, ell in enumerate(ell_values):
        s_ell_uhz = np.sqrt(ell * (ell + 1.0)) * (cs_arr / r_arr) * rad_to_uhz
        ax.plot(
            x_arr,
            np.log10(np.maximum(s_ell_uhz, 1.0e-2)),
            color=palette[idx % len(palette)],
            ls="--" if idx > 0 else "-",
            lw=1.4,
            label=f"Lamb [$S_{{{ell}}} / 2\\pi$]",
            zorder=2,
        )
    if nu_max_uhz is not None and nu_max_uhz > 0.0:
        ax.axhline(
            np.log10(float(nu_max_uhz)),
            color=style.OKABE_ITO["grey"],
            ls=":",
            lw=1.1,
            label="$\\nu_{\\max}$",
        )
    ax.set_xlabel("Fractional Radius [$r / R_*$]")
    ax.set_ylabel(
        "Characteristic Frequency [$\\log_{10}(\\nu / \\mu\\mathrm{Hz})$]"
    )
    ax.legend(loc="best", frameon=False)
    return style.finalize_fig(fig, ax, save_path)
