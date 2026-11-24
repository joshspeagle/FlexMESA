# SPDX-License-Identifier: GPL-3.0-or-later
r"""Upstream ``pgstar`` ``Network`` and ``Production`` nucleosynthesis panels.

Ports ``do_network_panel`` (`star/private/pgstar_network.f90:74-284`) and
``do_production_panel`` (`star/private/pgstar_production.f90:71-350`) strictly
from saved ``profileN.data`` isotope columns and MESA ``26.4.1`` nuclide tables
via ``flexmesa.chem`` (`#173`).
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import matplotlib as mpl
import matplotlib.axes as mpl_axes
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.figure as mpl_figure
import matplotlib.patches as mpl_patches
import numpy as np

from flexmesa import chem
from flexmesa.plotting import (
    _compare_panels,
    _plot_support,
    compare,
    pgstar_panels,
    style,
)


def _zone_mass_weights(
    prof: compare.ProfileTable,
    *,
    min_mass: float | None = None,
    max_mass: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mask, weights)`` for mass-weighted bulk isotope averaging.

    Uses saved ``dm``, ``dq``, ``mass``, or ``q`` columns strictly without
    fabricating stellar structure data.
    """
    n_z = prof.num_zones
    mask = np.ones(n_z, dtype=bool)
    if min_mass is not None or max_mass is not None:
        if "mass" not in prof.columns:
            raise KeyError(
                "mass slicing (min_mass/max_mass) requires 'mass' in profile"
            )
        m_arr = prof.columns["mass"]
        if min_mass is not None:
            mask = mask & (m_arr >= float(min_mass))
        if max_mass is not None:
            mask = mask & (m_arr <= float(max_mass))
        if not np.any(mask):
            raise ValueError(
                f"no profile zones fall within mass range "
                f"[{min_mass!r}, {max_mass!r}]"
            )

    if "dm" in prof.columns:
        weights = np.maximum(prof.columns["dm"], 0.0)
    elif "dq" in prof.columns:
        weights = np.maximum(prof.columns["dq"], 0.0)
    elif "mass" in prof.columns or "q" in prof.columns:
        coord = (
            prof.columns["mass"]
            if "mass" in prof.columns
            else prof.columns["q"]
        )
        if n_z == 1:
            weights = np.ones(1, dtype=np.float64)
        else:
            # Interior-to-surface or surface-to-center shell widths
            order = np.argsort(coord, kind="mergesort")
            c_sorted = coord[order]
            edges = np.empty(n_z + 1, dtype=np.float64)
            edges[0] = max(
                0.0, float(c_sorted[0]) - 0.5 * float(c_sorted[1] - c_sorted[0])
            )
            edges[1:-1] = 0.5 * (c_sorted[:-1] + c_sorted[1:])
            edges[-1] = float(c_sorted[-1])
            w_sorted = np.maximum(np.diff(edges), 0.0)
            weights = np.empty(n_z, dtype=np.float64)
            weights[order] = w_sorted
    else:
        raise KeyError(
            "Network/Production panel requires 'dm', 'dq', 'mass', or 'q' "
            "column in profileN.data"
        )

    sel_w = np.where(mask & _compare_panels.finite_mask(weights), weights, 0.0)
    if float(np.sum(sel_w)) <= 0.0:
        raise ValueError("selected profile zones have zero total mass weight")
    return mask, sel_w


def _extract_bulk_isotopes(
    prof: compare.ProfileTable,
    *,
    min_mass: float | None = None,
    max_mass: float | None = None,
) -> list[tuple[chem.NuclideInfo, float]]:
    """Compute mass-averaged abundances ``(NuclideInfo, X_bar)`` for profile."""
    _mask, weights = _zone_mass_weights(
        prof, min_mass=min_mass, max_mass=max_mass
    )
    w_tot = float(np.sum(weights))

    records: list[tuple[chem.NuclideInfo, float]] = []
    for col in pgstar_panels._discover_isotope_columns(prof):
        cid = chem.get_nuclide_index(col)
        if cid < 1:
            continue
        info = chem.get_nuclide_info(cid)
        xa = prof.columns[col]
        valid = _compare_panels.finite_mask(xa) & (weights > 0.0)
        if not np.any(valid):
            continue
        x_bar = float(np.dot(xa[valid], weights[valid])) / w_tot
        records.append((info, max(x_bar, 0.0)))

    if not records:
        raise ValueError(
            "no recognized isotope mass-fraction columns (h1, he4, c12, ...) "
            "found in profileN.data"
        )
    return records


def plot_network_chart(
    profile_path: pathlib.Path | str,
    *,
    nmin: int | None = None,
    nmax: int | None = None,
    zmin: int | None = None,
    zmax: int | None = None,
    log_mass_frac_min: float = -5.0,
    log_mass_frac_max: float = 0.0,
    show_mass_fraction: bool = True,
    show_element_names: bool = True,
    show_colorbar: bool = True,
    min_mass: float | None = None,
    max_mass: float | None = None,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot ``Network`` chart of nuclides ($N$ vs. $Z$, `pgstar_network.f90`).

    Implements ``do_network_panel`` from
    ``star/private/pgstar_network.f90:74-284``
    (`star/defaults/pgstar.defaults:2626-2670`), drawing one unit cell per
    nuclide present in ``profileN.data``, shaded by mass-averaged
    $\log_{10} \bar{X}_i$.

    Args:
        profile_path: Path to ``profileN.data``.
        nmin: Optional lower neutron-number bound $N_{\min}$.
        nmax: Optional upper neutron-number bound $N_{\max}$.
        zmin: Optional lower proton-number bound $Z_{\min}$.
        zmax: Optional upper proton-number bound $Z_{\max}$.
        log_mass_frac_min: Lower colorbar bound for $\log_{10} \bar{X}_i$
            (default ``-5.0``, `pgstar.defaults:2661`).
        log_mass_frac_max: Upper colorbar bound for $\log_{10} \bar{X}_i$
            (default ``0.0``, `pgstar.defaults:2662`).
        show_mass_fraction: Fill cells colored by $\log_{10} \bar{X}_i$.
        show_element_names: Annotate element symbols along the left side.
        show_colorbar: Attach a colorbar with explicit ``vmin``/``vmax``.
        min_mass: Optional lower enclosed mass cut ($M_\odot$).
        max_mass: Optional upper enclosed mass cut ($M_\odot$).
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.
    """
    if log_mass_frac_max <= log_mass_frac_min:
        raise ValueError("log_mass_frac_max must exceed log_mass_frac_min")
    prof = compare.load_profile_file(profile_path)
    species = _extract_bulk_isotopes(prof, min_mass=min_mass, max_mass=max_mass)

    all_n = [info.n for info, _ in species]
    all_z = [info.z for info, _ in species]
    eff_nmin = min(all_n) if nmin is None else int(nmin)
    eff_nmax = max(all_n) if nmax is None else int(nmax)
    eff_zmin = min(all_z) if zmin is None else int(zmin)
    eff_zmax = max(all_z) if zmax is None else int(zmax)

    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.85)
    cmap = mpl.colormaps["viridis"]
    norm = mpl_colors.Normalize(
        vmin=float(log_mass_frac_min), vmax=float(log_mass_frac_max)
    )

    visible_species = [
        (info, x_bar)
        for info, x_bar in species
        if eff_nmin <= info.n <= eff_nmax and eff_zmin <= info.z <= eff_zmax
    ]
    min_n_by_z: dict[int, int] = {}
    for info, _x_bar in visible_species:
        prev = min_n_by_z.get(info.z)
        if prev is None or info.n < prev:
            min_n_by_z[info.z] = info.n

    for info, x_bar in visible_species:
        nv, zv = info.n, info.z
        fill_color: Any = "none"
        if show_mass_fraction and x_bar > 0.0:
            lg_x = math.log10(x_bar)
            if lg_x >= log_mass_frac_min:
                fill_color = cmap(norm(min(lg_x, log_mass_frac_max)))
        rect = mpl_patches.Rectangle(
            (float(nv) - 0.5, float(zv) - 0.5),
            1.0,
            1.0,
            facecolor=fill_color,
            edgecolor=style.OKABE_ITO["black"],
            linewidth=0.65,
            zorder=2,
        )
        ax.add_patch(rect)

    if show_element_names:
        for zv, min_n in sorted(min_n_by_z.items()):
            ax.text(
                float(min_n) - 0.75,
                float(zv),
                chem.element_name(zv).capitalize(),
                ha="right",
                va="center",
                fontsize=6.2,
                color=style.OKABE_ITO["black"],
                clip_on=True,
            )

    n_span = max(eff_nmax - eff_nmin + 1, 1)
    pad_left = max(2.4, 0.10 * n_span) if show_element_names else 1.5
    ax.set_xlim(float(eff_nmin) - pad_left, float(eff_nmax) + 1.5)
    ax.set_ylim(float(eff_zmin) - 1.5, float(eff_zmax) + 1.5)
    ax.set_xlabel(r"Neutron Number [$N$]")
    ax.set_ylabel(r"Proton Number [$Z$]")
    _plot_support.set_profile_panel_title(
        ax, "Network Nuclide Chart", prof, None, standalone=standalone
    )

    if show_mass_fraction and show_colorbar:
        sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.055)
        cbar.set_label(r"$\log_{10} \bar{X}_i$")

    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def _reference_stable_abundances(
    prof: compare.ProfileTable,
    init_profile_path: pathlib.Path | str | None,
    *,
    min_mass: float | None,
    max_mass: float | None,
) -> tuple[np.ndarray, str]:
    """Build stable-isotope reference array of shape ``(286,)`` and label."""
    if init_profile_path is not None:
        init_prof = compare.load_profile_file(init_profile_path)
        init_sp = _extract_bulk_isotopes(
            init_prof, min_mass=min_mass, max_mass=max_mass
        )
        init_stable = chem.get_stable_mass_frac(
            [info.z for info, _ in init_sp],
            [info.a for info, _ in init_sp],
            [x_val for _, x_val in init_sp],
        )
        return init_stable, r"$\log_{10}(X / X_{\mathrm{init}})$"

    # Build AG89 solar stable vector, scaled by header initial_z/initial_y when
    # present (matching star/private/pgstar_production.f90:180-211).
    namsol = chem.ag89_namsol()
    z_sol_list: list[int] = []
    a_sol_list: list[int] = []
    x_sol_arr = np.zeros(len(namsol), dtype=np.float64)
    for idx, iso_name in enumerate(namsol):
        rec = chem.chem_get_solar(iso_name)
        if rec is None:
            z_sol_list.append(0)
            a_sol_list.append(0)
            continue
        z_sol_list.append(round(rec[0]))
        a_sol_list.append(round(rec[1]))
        x_sol_arr[idx] = float(rec[2])

    hdr_z = prof.header.get("initial_z")
    hdr_y = prof.header.get("initial_y")
    if isinstance(hdr_z, (int, float)) and isinstance(hdr_y, (int, float)):
        init_z = float(hdr_z)
        init_y = float(hdr_y)
        init_h1 = max(0.0, min(1.0, 1.0 - (init_z + init_y)))
        xsol_he3 = chem.chem_xsol("he3")
        xsol_he4 = chem.chem_xsol("he4")
        he_sum = max(xsol_he3 + xsol_he4, 1.0e-30)
        z_scale = init_z / max(chem.ag89_zsol(), 1.0e-30)
        scaled = np.empty_like(x_sol_arr)
        for idx, (zv, av) in enumerate(
            zip(z_sol_list, a_sol_list, strict=True)
        ):
            if zv == 1 and av == 1:
                scaled[idx] = init_h1
            elif zv == 1 and av == 2:
                scaled[idx] = chem.chem_xsol("h2")
            elif zv == 2 and av == 3:
                scaled[idx] = init_y * xsol_he3 / he_sum
            elif zv == 2 and av == 4:
                scaled[idx] = init_y * xsol_he4 / he_sum
            else:
                scaled[idx] = x_sol_arr[idx] * z_scale
        return scaled, r"$\log_{10}(X / X_{\mathrm{init,AG89}})$"

    return x_sol_arr, r"$\log_{10}(X / X_{\mathrm{AG89}})$"


def _compute_log_production_by_element(
    stable_abun: np.ndarray,
    stable_ref: np.ndarray,
    *,
    z_max_net: int,
    min_mass_frac: float,
) -> dict[int, list[tuple[int, float]]]:
    """Group surviving stable isotopes ``Z -> [(A, log10(X/X_ref)), ...]``."""
    namsol = chem.ag89_namsol()
    by_z: dict[int, list[tuple[int, float]]] = {}
    for idx, iso_name in enumerate(namsol):
        rec = chem.chem_get_solar(iso_name)
        if rec is None:
            continue
        zv, av = round(rec[0]), round(rec[1])
        if zv > z_max_net or zv < 1:
            continue
        xa = float(stable_abun[idx])
        xr = float(stable_ref[idx])
        if xa <= 0.0 or xr <= 0.0:
            continue
        la, lac = math.log10(xa), math.log10(xr)
        if la < min_mass_frac or lac < min_mass_frac:
            continue
        by_z.setdefault(zv, []).append((av, la - lac))
    return by_z


def plot_production_profile(
    profile_path: pathlib.Path | str,
    *,
    init_profile_path: pathlib.Path | str | None = None,
    ref_profile_path: pathlib.Path | str | None = None,
    amin: int | None = None,
    amax: int | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    min_mass: float | None = None,
    max_mass: float | None = None,
    min_mass_frac: float = -5.0,
    show_element_names: bool = True,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot ``Production`` factors vs. mass number $A$.

    Implements ``do_production_panel`` from
    ``star/private/pgstar_production.f90:71-350``
    (`star/defaults/pgstar.defaults:2697-2745`), mapping network species onto
    the 286 stable isotopes via :func:`flexmesa.chem.get_stable_mass_frac`.

    Args:
        profile_path: Path to ``profileN.data``.
        init_profile_path: Optional initial ``profileN.data`` providing the
            reference composition (defaults to AG89 solar scaled by header
            ``initial_z``/``initial_y`` when present).
        ref_profile_path: Optional MESA ``26.4.1`` ``profileN.data`` overlay.
        amin: Optional lower mass-number bound $A_{\min}$.
        amax: Optional upper mass-number bound $A_{\max}$.
        ymin: Optional lower log production-factor limit.
        ymax: Optional upper log production-factor limit.
        min_mass: Optional lower enclosed mass cut ($M_\odot$).
        max_mass: Optional upper enclosed mass cut ($M_\odot$).
        min_mass_frac: Minimum $\log_{10} X$ filter for both star and reference
            (default ``-5.0``, `pgstar.defaults:2735`).
        show_element_names: Annotate staggered element symbols above curves.
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.
    """
    prof = compare.load_profile_file(profile_path)
    species = _extract_bulk_isotopes(prof, min_mass=min_mass, max_mass=max_mass)
    z_max_net = max(info.z for info, _ in species)
    stable_abun = chem.get_stable_mass_frac(
        [info.z for info, _ in species],
        [info.a for info, _ in species],
        [x_val for _, x_val in species],
    )
    stable_ref, y_lbl = _reference_stable_abundances(
        prof, init_profile_path, min_mass=min_mass, max_mass=max_mass
    )
    by_z = _compute_log_production_by_element(
        stable_abun,
        stable_ref,
        z_max_net=z_max_net,
        min_mass_frac=min_mass_frac,
    )
    if not by_z:
        raise ValueError(
            f"Production panel: zero stable isotopes exceed "
            f"min_mass_frac={min_mass_frac!r}"
        )

    ref_prof: compare.ProfileTable | None = None
    ref_by_z: dict[int, list[tuple[int, float]]] = {}
    if ref_profile_path is not None:
        ref_prof = compare.load_profile_file(ref_profile_path)
        ref_sp = _extract_bulk_isotopes(
            ref_prof, min_mass=min_mass, max_mass=max_mass
        )
        ref_stable = chem.get_stable_mass_frac(
            [info.z for info, _ in ref_sp],
            [info.a for info, _ in ref_sp],
            [x_val for _, x_val in ref_sp],
        )
        ref_by_z = _compute_log_production_by_element(
            ref_stable,
            stable_ref,
            z_max_net=max(z_max_net, max(info.z for info, _ in ref_sp)),
            min_mass_frac=min_mass_frac,
        )

    all_a = [av for pts in by_z.values() for av, _ in pts]
    all_y = [yv for pts in by_z.values() for _, yv in pts]
    eff_amin = min(all_a) if amin is None else int(amin)
    eff_amax = max(all_a) if amax is None else int(amax)
    eff_ymin = min(all_y) if ymin is None else float(ymin)
    eff_ymax = max(all_y) if ymax is None else float(ymax)
    y_span = max(eff_ymax - eff_ymin, 0.5)

    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.80)
    ax.axhline(0.0, color=style.OKABE_ITO["grey"], ls=":", lw=0.9, zorder=1)

    palette = pgstar_panels.SERIES_COLORS
    alt = -1
    for z_idx, zv in enumerate(sorted(by_z)):
        pts = [
            (av, yv)
            for av, yv in sorted(by_z[zv])
            if eff_amin <= av <= eff_amax
        ]
        if not pts:
            continue
        c_hex = palette[z_idx % len(palette)]
        ls_cur = ":" if z_idx >= len(palette) else "-"
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            color=c_hex,
            ls=ls_cur,
            marker="o",
            ms=3.6,
            lw=1.35,
            zorder=3,
        )
        if zv in ref_by_z:
            rpts = [
                (av, yv)
                for av, yv in sorted(ref_by_z[zv])
                if eff_amin <= av <= eff_amax
            ]
            if rpts:
                ax.plot(
                    [p[0] for p in rpts],
                    [p[1] for p in rpts],
                    color=c_hex,
                    ls="--",
                    marker="s",
                    mfc="none",
                    ms=3.2,
                    lw=0.95,
                    alpha=0.72,
                    zorder=2,
                )
        if show_element_names:
            yloc = eff_ymax + (0.14 if alt > 0 else 0.05) * y_span
            ax.text(
                float(xs[0]),
                yloc,
                chem.element_name(zv).capitalize(),
                ha="center",
                va="bottom",
                fontsize=6.4,
                color=c_hex,
                clip_on=True,
            )
            alt = -alt

    top_pad = 0.28 * y_span if show_element_names else 0.12 * y_span
    ax.set_xlim(float(eff_amin) - 1.5, float(eff_amax) + 1.5)
    ax.set_ylim(eff_ymin - 0.12 * y_span, eff_ymax + top_pad)
    ax.set_xlabel(r"Mass Number [$A$]")
    ax.set_ylabel(f"Log Production Factor [{y_lbl}]")
    _plot_support.set_profile_panel_title(
        ax, "Isotope Production Factors", prof, ref_prof, standalone=standalone
    )
    _compare_panels.save_figure(fig, save_path)
    return fig, ax
