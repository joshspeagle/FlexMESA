# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Kippenhahn convective, mixing, and burning diagrams (`#173`).

Ports the static diagnostic content of ``star/private/pgstar_kipp.f90:59-814``
(`star/defaults/pgstar.defaults:1200-1327`) strictly from saved ``history.data``
region columns (``mix_type_*``, ``mix_qtop_*``, ``mix_relr_type_*``,
``mix_relr_top_*``, ``burn_type_*``, ``burn_qtop_*``, ``burn_relr_type_*``,
``burn_relr_top_*``, core mass boundaries, and luminosities) and/or sequences
of ``profileN.data`` files.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.axes as mpl_axes
import matplotlib.colors as mpl_colors
import matplotlib.figure as mpl_figure
import numpy as np

from flexmesa.plotting import (
    _compare_panels,
    _kipp_regions,
    _plot_support,
    compare,
    style,
)
from flexmesa.testing import history_compare

_CORE_BOUNDARY_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("he_core_mass", r"$M_{\mathrm{He}}$", style.OKABE_ITO["vermilion"], "-"),
    (
        "co_core_mass",
        r"$M_{\mathrm{CO}}$",
        style.OKABE_ITO["bluish_green"],
        "--",
    ),
    ("c_core_mass", r"$M_{\mathrm{C}}$", style.OKABE_ITO["orange"], "--"),
    (
        "o_core_mass",
        r"$M_{\mathrm{O}}$",
        style.OKABE_ITO["reddish_purple"],
        "-.",
    ),
    ("si_core_mass", r"$M_{\mathrm{Si}}$", style.OKABE_ITO["blue"], ":"),
    ("fe_core_mass", r"$M_{\mathrm{Fe}}$", style.OKABE_ITO["black"], ":"),
)

_LUMINOSITY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("log_L", r"$\log L$", style.OKABE_ITO["black"]),
    ("log_LH", r"$\log L_{\mathrm{H}}$", style.OKABE_ITO["blue"]),
    ("log_LHe", r"$\log L_{\mathrm{He}}$", style.OKABE_ITO["vermilion"]),
    ("log_Lneu", r"$\log L_\nu$", style.OKABE_ITO["reddish_purple"]),
)


def plot_history_kippenhahn(
    history_path: pathlib.Path | str | None = None,
    *,
    profile_paths: Sequence[pathlib.Path | str] | None = None,
    xaxis: Literal["model_number", "star_age"] = "model_number",
    yaxis: Literal["mass", "q", "radius"] = "mass",
    age_units: Literal["auto", "yr", "Myr"] = "auto",
    log_xaxis: bool = False,
    time_from_present: bool = False,
    xaxis_reversed: bool | None = None,
    show_mixing: bool = True,
    show_burn: bool = True,
    show_mass_boundaries: bool = True,
    show_luminosities: bool = False,
    burn_type_cutoff: float = 0.0,
    mass_grid_points: int = 140,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot a MESA ``26.4.1`` Kippenhahn diagram from history or profiles.

    Implements ``do_Kipp_Plot`` from ``star/private/pgstar_kipp.f90:59-814``
    strictly from saved run outputs without computing physics inside plotting
    code.

    Args:
        history_path: Optional path to ``history.data`` providing region columns
            (``mix_type_*``, ``burn_type_*``), core mass boundaries, and
            luminosities.
        profile_paths: Optional sequence of at least two ``profileN.data``
            files used to assemble radial mixing and burning grids when
            history region columns are absent or radial detail is preferred.
        xaxis: Horizontal coordinate (``"model_number"`` default per
            ``star/defaults/pgstar.defaults:1226``, or ``"star_age"``).
        yaxis: Vertical coordinate (``"mass"`` [$M_r / M_\odot$], ``"q"``
            [$m / M$], or ``"radius"`` [$R_r / R_\odot$]).
        age_units: Age unit when ``xaxis="star_age"`` (``"auto"``, ``"yr"``,
            or ``"Myr"``).
        log_xaxis: Apply $\log_{10}|x|$ scaling (`pgstar_kipp.f90:193-197`).
        time_from_present: Shift ``star_age`` to $t - t_{\mathrm{final}}$
            (`pgstar_kipp.f90:186-191`).
        xaxis_reversed: Optional override reversing horizontal axis direction
            (`pgstar.defaults:1232`). Defaults to ``True`` when
            ``xaxis == "star_age"``, ``time_from_present=True``, and
            ``log_xaxis=True`` so evolution advances left-to-right toward the
            final model, and ``False`` otherwise.
        show_mixing: Render convective, overshoot, semiconvective,
            thermohaline, and RTI regions (`pgstar_kipp.f90:724-796`).
        show_burn: Shade nuclear burning and cooling zones
            (`pgstar_kipp.f90:399-512`).
        show_mass_boundaries: Draw surface mass/radius and core boundaries
            (``he_core_mass``, ``co_core_mass``, ``c_core_mass``,
            ``o_core_mass``, ``si_core_mass``, ``fe_core_mass``,
            `pgstar_kipp.f90:589-645`).
        show_luminosities: Overlay ``log_L``, ``log_LH``, ``log_LHe``, and
            ``log_Lneu`` on a twin right axis (`pgstar_kipp.f90:515-586`).
        burn_type_cutoff: Minimum $|\log_{10}|\epsilon||$ threshold shown when
            rendering ``burn_type_*`` regions (`pgstar.defaults:1287`).
        mass_grid_points: Vertical interpolation grid length when rendering
            from ``profile_paths`` (default ``140``).
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)`` of ``Figure`` and primary left ``Axes``.

    Raises:
        ValueError: If neither ``history_path`` nor ``profile_paths`` is given,
            or required coordinate columns are missing.
    """
    if history_path is None and not profile_paths:
        raise ValueError(
            "provide history_path or at least 2 profile_paths for Kippenhahn"
        )

    fig, ax = style.init_ax(ax, "double", 0.58)
    hist_tbl: history_compare.HistoryTable | None = None
    hist_hdr: dict[str, Any] = {}
    if history_path is not None:
        hist_tbl, hist_hdr = _plot_support.load_history_with_header(
            history_path
        )

    handles: list[Any] = []
    x_bounds: list[np.ndarray] = []
    y_bounds: list[np.ndarray] = []
    x_label = ""
    y_label = ""

    if profile_paths:
        if len(profile_paths) < 2:
            raise ValueError(
                "profile_paths Kippenhahn requires at least 2 profile files"
            )
        loaded = [compare.load_profile_file(p) for p in profile_paths]
        sort_key = "model_number" if xaxis == "model_number" else "star_age"
        loaded.sort(key=lambda pt: float(pt.header.get(sort_key, 0.0)))
        raw_px = np.array(
            [
                float(pt.header.get(sort_key, idx + 1))
                for idx, pt in enumerate(loaded)
            ],
            dtype=np.float64,
        )
        px_vals, x_label = _kipp_regions.resolve_kipp_xaxis(
            raw_px,
            xaxis,
            age_units=age_units,
            log_xaxis=log_xaxis,
            time_from_present=time_from_present,
        )

        y_vecs: list[np.ndarray] = []
        for pt in loaded:
            if "mixing_type" not in pt.columns:
                raise ValueError("each profile requires a 'mixing_type' column")
            yv, y_label, _ = _plot_support.profile_xaxis(pt, yaxis)
            y_vecs.append(yv)

        max_y = max(float(np.nanmax(yv)) for yv in y_vecs)
        y_grid = np.linspace(0.0, max_y, mass_grid_points, dtype=np.float64)
        x_bounds.append(px_vals)
        y_bounds.append(y_grid)
        mix_grid = np.zeros((mass_grid_points, len(loaded)), dtype=np.float64)
        has_eps = show_burn and all("eps_nuc" in pt.columns for pt in loaded)
        eps_grid = (
            np.zeros((mass_grid_points, len(loaded)), dtype=np.float64)
            if has_eps
            else None
        )

        for col_i, (pt, y_col) in enumerate(zip(loaded, y_vecs, strict=True)):
            order = np.argsort(y_col, kind="mergesort")
            y_sorted = y_col[order]
            mix_sorted = np.rint(pt.columns["mixing_type"][order])
            idx_r = np.searchsorted(y_sorted, y_grid, side="left")
            i0 = np.clip(idx_r - 1, 0, y_sorted.size - 1)
            i1 = np.clip(idx_r, 0, y_sorted.size - 1)
            pick = np.where(
                np.abs(y_grid - y_sorted[i1]) < np.abs(y_grid - y_sorted[i0]),
                i1,
                i0,
            )
            mix_grid[:, col_i] = np.where(
                y_grid <= y_sorted[-1], mix_sorted[pick], 0.0
            )
            if eps_grid is not None:
                e_raw = pt.columns["eps_nuc"][order]
                neu = pt.columns.get("non_nuc_neu")
                net = e_raw - neu[order] if neu is not None else e_raw
                valid = _compare_panels.finite_mask(net)
                safe_net = np.where(valid, net, 0.0)
                signed_e = np.sign(safe_net) * np.log10(1.0 + np.abs(safe_net))
                eps_grid[:, col_i] = np.interp(
                    y_grid, y_sorted, signed_e, left=0.0, right=0.0
                )

        if eps_grid is not None:
            has_neg = bool(np.any(eps_grid < -0.05))
            e_lim = max(float(np.nanmax(np.abs(eps_grid))), 1.0)
            norm = (
                mpl_colors.TwoSlopeNorm(vmin=-e_lim, vcenter=0.0, vmax=e_lim)
                if has_neg
                else mpl_colors.Normalize(vmin=0.0, vmax=e_lim)
            )
            mesh = ax.pcolormesh(
                px_vals,
                y_grid,
                eps_grid,
                cmap="coolwarm" if has_neg else "Blues",
                norm=norm,
                shading="auto",
                zorder=1,
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label(
                r"Nuclear Energy [$\mathrm{sgn}(\epsilon)"
                r"\log_{10}(1+|\epsilon|)$]"
            )
        if show_mixing:
            handles.extend(
                _plot_support.render_mixing_grid_layers(
                    ax, px_vals, y_grid, mix_grid
                )
            )
        if show_mass_boundaries and hist_tbl is None:
            surf = np.array(
                [float(np.nanmax(yv)) for yv in y_vecs], dtype=np.float64
            )
            (ln_s,) = ax.plot(
                px_vals,
                surf,
                color=style.OKABE_ITO["black"],
                lw=1.4,
                label="Surface",
                zorder=3,
            )
            handles.append(ln_s)

    if hist_tbl is not None:
        if xaxis not in hist_tbl.columns:
            alt = (
                "age"
                if xaxis == "star_age" and "age" in hist_tbl.columns
                else None
            )
            if alt is None:
                raise ValueError(
                    f"xaxis column {xaxis!r} not found in history columns"
                )
            raw_hx = hist_tbl.columns[alt]
        else:
            raw_hx = hist_tbl.columns[xaxis]

        hx_vals, x_label = _kipp_regions.resolve_kipp_xaxis(
            raw_hx,
            xaxis,
            age_units=age_units,
            log_xaxis=log_xaxis,
            time_from_present=time_from_present,
        )
        y_center, y_span, hist_ylabel = _kipp_regions.resolve_history_scale(
            hist_tbl, hist_hdr, yaxis
        )
        if not y_label:
            y_label = hist_ylabel

        y_top_curve = y_center + y_span
        y_bounds.append(y_top_curve)

        if not profile_paths:
            x_edges = _kipp_regions.step_edges(hx_vals)
            x_bounds.append(x_edges)
            handles.extend(
                _kipp_regions.draw_history_regions(
                    fig,
                    ax,
                    x_edges,
                    y_center,
                    y_span,
                    hist_tbl,
                    yaxis=yaxis,
                    show_mixing=show_mixing,
                    show_burn=show_burn,
                    burn_type_cutoff=burn_type_cutoff,
                )
            )
        else:
            x_bounds.append(hx_vals)

        if show_mass_boundaries:
            if yaxis != "q":
                (ln_top,) = ax.plot(
                    hx_vals,
                    y_top_curve,
                    color=style.OKABE_ITO["black"],
                    lw=1.45,
                    label=(
                        r"$M_{\mathrm{total}}$" if yaxis == "mass" else r"$R_*$"
                    ),
                    zorder=4,
                )
                handles.append(ln_top)
            if yaxis in ("mass", "q"):
                s_mass = hist_tbl.columns.get(
                    "star_mass", np.ones(hist_tbl.num_rows, dtype=np.float64)
                )
                ymax_ref = max(float(np.nanmax(y_top_curve)), 1.0e-12)
                for col, lbl, c_hex, ls_style in _CORE_BOUNDARY_SPECS:
                    if col not in hist_tbl.columns:
                        continue
                    c_arr = hist_tbl.columns[col]
                    y_core = (
                        c_arr / np.maximum(s_mass, 1.0e-30)
                        if yaxis == "q"
                        else c_arr
                    )
                    valid_c = _compare_panels.finite_mask(y_core)
                    if (
                        np.any(valid_c)
                        and float(np.max(y_core[valid_c])) > 1.0e-2 * ymax_ref
                    ):
                        (ln_c,) = ax.plot(
                            hx_vals[valid_c],
                            y_core[valid_c],
                            color=c_hex,
                            ls=ls_style,
                            lw=1.35,
                            label=lbl,
                            zorder=4,
                        )
                        handles.append(ln_c)

        if show_luminosities:
            ax_lum = ax.twinx()
            lum_drawn = False
            for l_col, l_lbl, l_hex in _LUMINOSITY_SPECS:
                if l_col not in hist_tbl.columns:
                    continue
                y_l = hist_tbl.columns[l_col]
                valid_l = _compare_panels.finite_mask(hx_vals) & (
                    _compare_panels.finite_mask(y_l)
                )
                if np.any(valid_l) and float(np.max(y_l[valid_l])) > -2.0:
                    (ln_l,) = ax_lum.plot(
                        hx_vals[valid_l],
                        y_l[valid_l],
                        color=l_hex,
                        ls="-.",
                        lw=1.15,
                        alpha=0.85,
                        label=l_lbl,
                        zorder=5,
                    )
                    handles.append(ln_l)
                    lum_drawn = True
            if lum_drawn:
                ax_lum.set_ylabel(r"Luminosity [$\log_{10}(L / L_\odot)$]")

    for ln in ax.get_lines():
        x_bounds.append(np.asarray(ln.get_xdata(), dtype=np.float64))
        y_bounds.append(np.asarray(ln.get_ydata(), dtype=np.float64))

    rev_x = (
        xaxis_reversed
        if xaxis_reversed is not None
        else (xaxis in ("star_age", "age") and time_from_present and log_xaxis)
    )
    if x_bounds:
        x_cat = np.concatenate(x_bounds)
        x_fin = x_cat[np.isfinite(x_cat)]
        if x_fin.size > 0:
            x_lo, x_hi = float(np.min(x_fin)), float(np.max(x_fin))
            if x_hi <= x_lo:
                x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
            ax.set_xlim((x_hi, x_lo) if rev_x else (x_lo, x_hi))
    elif rev_x and not ax.xaxis_inverted():
        ax.invert_xaxis()

    if yaxis == "q":
        ax.set_ylim(0.0, 1.02)
    elif y_bounds:
        y_cat = np.concatenate(y_bounds)
        y_fin = y_cat[np.isfinite(y_cat)]
        if y_fin.size > 0:
            ax.set_ylim(0.0, max(float(np.max(y_fin)) * 1.04, 1.0e-3))
    else:
        ax.autoscale_view()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Kippenhahn Diagram")
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=False,
            fontsize=7.2,
            ncol=min(len(handles), 3),
        )
    _compare_panels.save_figure(fig, save_path)
    return fig, ax
