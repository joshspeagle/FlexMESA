# SPDX-License-Identifier: GPL-3.0-or-later
"""Internal radial profile and Kippenhahn builders for ``compare``."""

from __future__ import annotations

import pathlib
import typing

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np

from flexmesa.plotting import _compare_panels, stellar_plots, style

if typing.TYPE_CHECKING:
    from flexmesa.plotting import compare as _compare_mod

_MIXING_STYLE: dict[int, tuple[str, str]] = {
    1: ("Convective", style.OKABE_ITO["sky_blue"]),
    2: ("Overshoot", style.OKABE_ITO["orange"]),
    3: ("Semiconvective", style.OKABE_ITO["bluish_green"]),
    4: ("Thermohaline", style.OKABE_ITO["reddish_purple"]),
    5: ("Rotation", style.OKABE_ITO["yellow"]),
    6: ("Rayleigh-Taylor", style.OKABE_ITO["vermilion"]),
}


def profile_coord(prof: _compare_mod.ProfileTable) -> tuple[np.ndarray, str]:
    """Select the primary radial coordinate vector and axis label."""
    if "mass" in prof.columns:
        return prof.columns["mass"], "Enclosed Mass [$M_r / M_\\odot$]"
    if "q" in prof.columns:
        return prof.columns["q"], "Mass Coordinate [$q = m / M$]"
    if "radius" in prof.columns:
        return prof.columns["radius"], "Radius [$R_r / R_\\odot$]"
    if "zone" in prof.columns:
        return prof.columns["zone"], "Zone Index"
    return np.arange(1, prof.num_zones + 1, dtype=np.float64), "Zone Index"


def _signed_log_eps(arr: np.ndarray) -> np.ndarray:
    """Compute ``sgn(x)*log10(1+|x|)`` after masking MESA sentinel values."""
    valid = _compare_panels.finite_mask(arr)
    safe = np.where(valid, arr, 0.0)
    return np.where(valid, np.sign(safe) * np.log10(1.0 + np.abs(safe)), np.nan)


def _plot_profile_column_pair(
    ax: mpl_axes.Axes,
    prof: _compare_mod.ProfileTable,
    x_run: np.ndarray,
    ref_prof: _compare_mod.ProfileTable | None,
    x_ref: np.ndarray | None,
    col: str,
    *,
    run_label: str,
    ref_label: str,
    color: str,
    run_lw: float = 1.6,
    ref_lw: float = 1.1,
    ref_alpha: float = 0.7,
    run_zorder: int = 3,
    transform: typing.Callable[[np.ndarray], np.ndarray] | None = None,
) -> bool:
    """Overplot optional reference profile and candidate profile column."""
    drawn = False
    if ref_prof is not None and x_ref is not None and col in ref_prof.columns:
        y_ref = ref_prof.columns[col]
        if transform is not None:
            y_ref = transform(y_ref)
        drawn |= _compare_panels._plot_series(
            ax,
            x_ref,
            y_ref,
            label=ref_label,
            color=color,
            ls="--",
            lw=ref_lw,
            alpha=ref_alpha,
            zorder=2,
        )
    if col in prof.columns:
        y_run = prof.columns[col]
        if transform is not None:
            y_run = transform(y_run)
        drawn |= _compare_panels._plot_series(
            ax,
            x_run,
            y_run,
            label=run_label,
            color=color,
            lw=run_lw,
            zorder=run_zorder,
        )
    return drawn


def render_profile_panels(
    prof: _compare_mod.ProfileTable,
    ref_prof: _compare_mod.ProfileTable | None,
    *,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, np.ndarray]:
    """Build the four-panel interior structure comparison figure."""
    style.use_mesa_plot_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=style.figsize("double", 0.78),
        constrained_layout=True,
    )
    x_run, x_label = profile_coord(prof)
    x_ref, _ = profile_coord(ref_prof) if ref_prof is not None else (None, "")

    # (0, 0) Mass Fractions
    ax = axes[0, 0]
    ab_specs = (
        ("x_mass_fraction_H", "$X$ (H1)", style.OKABE_ITO["blue"]),
        ("y_mass_fraction_He", "$Y$ (He4)", style.OKABE_ITO["vermilion"]),
        ("z_mass_fraction_metals", "$Z$", style.OKABE_ITO["bluish_green"]),
        ("c12", "$^{12}\\mathrm{C}$", style.OKABE_ITO["orange"]),
        ("o16", "$^{16}\\mathrm{O}$", style.OKABE_ITO["reddish_purple"]),
    )
    for col, tag, c_hex in ab_specs:
        _plot_profile_column_pair(
            ax,
            prof,
            x_run,
            ref_prof,
            x_ref,
            col,
            run_label=tag,
            ref_label=f"{tag} MESA",
            color=c_hex,
            ref_lw=1.0,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mass Fraction")
    ax.set_title("Abundances")
    ax.legend(loc="best", frameon=False, fontsize=7.0, ncol=2)

    # (0, 1) logT and logRho
    ax = axes[0, 1]
    _plot_profile_column_pair(
        ax,
        prof,
        x_run,
        ref_prof,
        x_ref,
        "logT",
        run_label="$\\log T$",
        ref_label="$\\log T$ MESA",
        color=style.OKABE_ITO["vermilion"],
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("$\\log_{10}(T / \\mathrm{K})$")
    ax.set_title("Thermodynamic Profile")
    ax_r = ax.twinx()
    _plot_profile_column_pair(
        ax_r,
        prof,
        x_run,
        ref_prof,
        x_ref,
        "logRho",
        run_label="$\\log\\rho$",
        ref_label="$\\log\\rho$ MESA",
        color=style.OKABE_ITO["blue"],
    )
    ax_r.set_ylabel("$\\log_{10}(\\rho / \\mathrm{g\\,cm^{-3}})$")
    _compare_panels.merge_twinx_legend(
        ax, ax_r, loc="best", fontsize=7.0, ncol=2
    )

    # (1, 0) Luminosity or Nuclear Burning Categories
    ax = axes[1, 0]
    lum_drawn = False
    for col, tag, c_hex in (
        ("luminosity", "$L_r / L_\\odot$", style.OKABE_ITO["blue"]),
        ("logL", "$\\log_{10}(L_r / L_\\odot)$", style.OKABE_ITO["blue"]),
    ):
        if col in prof.columns:
            lum_drawn |= _plot_profile_column_pair(
                ax,
                prof,
                x_run,
                ref_prof,
                x_ref,
                col,
                run_label=tag,
                ref_label=f"{tag} MESA",
                color=c_hex,
                ref_alpha=0.75,
            )
            ax.set_ylabel(tag)
            ax.set_title("Interior Luminosity")
            break
    if not lum_drawn:
        burn_specs = (
            ("pp", "$\\epsilon_{\\mathrm{pp}}$", style.OKABE_ITO["blue"]),
            (
                "cno",
                "$\\epsilon_{\\mathrm{CNO}}$",
                style.OKABE_ITO["vermilion"],
            ),
            (
                "tri_alpha",
                "$\\epsilon_{3\\alpha}$",
                style.OKABE_ITO["bluish_green"],
            ),
        )
        for col, tag, c_hex in burn_specs:
            lum_drawn |= _plot_profile_column_pair(
                ax,
                prof,
                x_run,
                ref_prof,
                x_ref,
                col,
                run_label=tag,
                ref_label=f"{tag} MESA",
                color=c_hex,
                run_lw=1.5,
                ref_alpha=0.75,
                transform=_signed_log_eps,
            )
        ax.set_ylabel("$\\mathrm{sgn}(\\epsilon)\\log_{10}(1+|\\epsilon|)$")
        ax.set_title("Nuclear Energy Generation")
    ax.set_xlabel(x_label)
    if lum_drawn:
        ax.legend(loc="best", frameon=False, fontsize=7.5)

    # (1, 1) Mixing Diffusivity & Mixing Types (Convective Boundaries)
    ax = axes[1, 1]
    if "log_D_mix" in prof.columns:
        _plot_profile_column_pair(
            ax,
            prof,
            x_run,
            ref_prof,
            x_ref,
            "log_D_mix",
            run_label="Flex $\\log D$",
            ref_label="MESA $\\log D$",
            color=style.OKABE_ITO["grey"],
            run_lw=1.2,
            ref_alpha=0.75,
            run_zorder=2,
        )
        y_mix = prof.columns["log_D_mix"]
        ylabel = "$\\log_{10}(D_{\\mathrm{mix}} / \\mathrm{cm^2\\,s^{-1}})$"
    elif "mixing_type" in prof.columns:
        y_mix = prof.columns["mixing_type"]
        ylabel = "Mixing Type ID"
    else:
        y_mix = None
        ylabel = "Mixing Diffusivity"
        ax.text(
            0.5,
            0.5,
            "log_D_mix / mixing_type not present\nin profileN.data columns",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8.0,
        )

    if y_mix is not None and "mixing_type" in prof.columns:
        mtypes = np.rint(prof.columns["mixing_type"]).astype(np.int64)
        finite_xy = _compare_panels.finite_mask(
            x_run
        ) & _compare_panels.finite_mask(y_mix)
        for mcode, (mname, mcolor) in _MIXING_STYLE.items():
            mask = (mtypes == mcode) & finite_xy
            if np.any(mask):
                ax.scatter(
                    x_run[mask],
                    y_mix[mask],
                    c=mcolor,
                    s=12,
                    label=mname,
                    zorder=4,
                )
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title("Mixing & Convective Boundaries")
    h, _l_txt = ax.get_legend_handles_labels()
    if h:
        ax.legend(loc="best", frameon=False, fontsize=6.8, ncol=2)

    mod_no = prof.header.get("model_number", "?")
    age_yr = prof.header.get("star_age", None)
    age_str = (
        f", age={float(age_yr):.4e} yr"
        if isinstance(age_yr, (int, float))
        else ""
    )
    compiler = str(prof.header.get("compiler", "")).strip()
    src_lbl = "FlexMESA" if compiler == "rustc" else "MESA"
    ver = prof.header.get("version_number", "")
    mode = str(prof.header.get("math_backend", "")).strip()
    meta_str = ", ".join(p for p in (src_lbl, ver, mode) if p)
    fig.suptitle(
        f"Radial Profile: model={mod_no}{age_str}, zones={prof.num_zones} "
        f"({meta_str})",
        fontsize=9.5,
    )
    _compare_panels.save_figure(fig, save_path)
    return fig, axes


def render_profile_kippenhahn(
    loaded: list[_compare_mod.ProfileTable],
    *,
    mass_grid_points: int = 120,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    """Assemble categorical mixing-type and burning grids across profiles."""
    ages_myr = np.array(
        [
            float(pt.header.get("star_age", idx)) * 1.0e-6
            for idx, pt in enumerate(loaded)
        ],
        dtype=np.float64,
    )
    tot_masses = np.array(
        [float(np.nanmax(pt.columns["mass"])) for pt in loaded],
        dtype=np.float64,
    )
    max_m = float(np.nanmax(tot_masses))
    mass_grid = np.linspace(0.0, max_m, mass_grid_points, dtype=np.float64)
    mix_grid = np.zeros((mass_grid_points, len(loaded)), dtype=np.float64)
    eps_grid: np.ndarray | None = (
        np.zeros((mass_grid_points, len(loaded)), dtype=np.float64)
        if all("eps_nuc" in pt.columns for pt in loaded)
        else None
    )

    for col_i, pt in enumerate(loaded):
        m_arr = pt.columns["mass"]
        order = np.argsort(m_arr, kind="mergesort")
        m_sorted = m_arr[order]
        mix_sorted = np.rint(pt.columns["mixing_type"][order])
        idx_right = np.searchsorted(m_sorted, mass_grid, side="left")
        i0 = np.clip(idx_right - 1, 0, m_sorted.size - 1)
        i1 = np.clip(idx_right, 0, m_sorted.size - 1)
        pick = np.where(
            np.abs(mass_grid - m_sorted[i1]) < np.abs(mass_grid - m_sorted[i0]),
            i1,
            i0,
        )
        mix_grid[:, col_i] = np.where(
            mass_grid <= m_sorted[-1], mix_sorted[pick], 0.0
        )
        if eps_grid is not None:
            e_raw = pt.columns["eps_nuc"][order]
            e_clean = np.where(_compare_panels.finite_mask(e_raw), e_raw, 1.0)
            e_sorted = np.log10(np.maximum(e_clean, 1.0))
            eps_grid[:, col_i] = np.interp(
                mass_grid, m_sorted, e_sorted, left=0.0, right=0.0
            )

    return stellar_plots.plot_kippenhahn(
        ages_myr,
        mass_grid,
        mix_grid,
        eps_nuc_grid=eps_grid,
        total_mass_msun=tot_masses,
        save_path=str(save_path) if save_path is not None else None,
    )
