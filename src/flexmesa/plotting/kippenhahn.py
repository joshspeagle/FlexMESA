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

import math
import pathlib
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.axes as mpl_axes
import matplotlib.collections as mpl_collections
import matplotlib.colors as mpl_colors
import matplotlib.figure as mpl_figure
import matplotlib.patches as mpl_patches
import numpy as np

from flexmesa.plotting import _compare_panels, _plot_support, compare, style
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


def _step_edges(x_vals: np.ndarray) -> np.ndarray:
    """Compute N+1 cell boundaries around monotonic 1D step coordinates."""
    n_pts = x_vals.size
    edges = np.empty(n_pts + 1, dtype=np.float64)
    if n_pts == 1:
        edges[0] = x_vals[0] - 0.5
        edges[1] = x_vals[0] + 0.5
        return edges
    edges[1:-1] = 0.5 * (x_vals[:-1] + x_vals[1:])
    edges[0] = x_vals[0] - 0.5 * (x_vals[1] - x_vals[0])
    edges[-1] = x_vals[-1] + 0.5 * (x_vals[-1] - x_vals[-2])
    return edges


def _resolve_kipp_xaxis(
    raw_x: np.ndarray,
    xaxis: str,
    *,
    age_units: str,
    log_xaxis: bool,
    time_from_present: bool,
) -> tuple[np.ndarray, str]:
    """Scale and label Kippenhahn x-axis (`pgstar_kipp.f90:176-198`)."""
    vals = np.asarray(raw_x, dtype=np.float64).copy()
    if xaxis == "model_number":
        label = "Model Number"
    elif xaxis in ("star_age", "age"):
        if age_units == "Myr" or (
            age_units == "auto"
            and _compare_panels.max_finite_abs(vals) >= 1.0e5
        ):
            vals = vals * 1.0e-6
            unit_lbl = r"\mathrm{Myr}"
        else:
            unit_lbl = r"\mathrm{yr}"
        if time_from_present and vals.size > 0:
            vals = vals - vals[-1]
            label = rf"Time from Final Model [${unit_lbl}$]"
        else:
            label = rf"Stellar Age [${unit_lbl}$]"
    else:
        label = xaxis

    if log_xaxis:
        vals = np.log10(np.maximum(np.abs(vals), 1.0e-99))
        label = rf"$\log_{{10}}$({label})"
    return vals, label


def _contiguous_region_indices(
    columns: dict[str, np.ndarray],
    type_prefix: str,
    top_prefix: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect contiguous ``(type_k, top_k)`` arrays for ``k = 1..K``."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    idx = 1
    while True:
        t_col = f"{type_prefix}_{idx}"
        q_col = f"{top_prefix}_{idx}"
        if t_col not in columns or q_col not in columns:
            break
        pairs.append((columns[t_col], columns[q_col]))
        idx += 1
    return pairs


def _resolve_history_scale(
    tbl: history_compare.HistoryTable,
    header: dict[str, Any],
    yaxis: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(y_center, y_span, ylabel)`` for each history row."""
    n_rows = tbl.num_rows
    zeros = np.zeros(n_rows, dtype=np.float64)
    if yaxis == "q":
        return (
            zeros,
            np.ones(n_rows, dtype=np.float64),
            r"Mass Coordinate [$q = m / M$]",
        )
    if yaxis == "radius":
        if "radius" in tbl.columns:
            r_tot = tbl.columns["radius"]
            return zeros, r_tot, r"Radius [$R_r / R_\odot$]"
        if "log_R" in tbl.columns:
            r_tot = np.power(10.0, tbl.columns["log_R"])
            return zeros, r_tot, r"Radius [$R_r / R_\odot$]"
        return (
            zeros,
            np.ones(n_rows, dtype=np.float64),
            r"Fractional Radius [$r / R_*$]",
        )

    if "star_mass" not in tbl.columns:
        raise ValueError(
            "yaxis='mass' Kippenhahn from history requires 'star_mass' column"
        )
    s_mass = tbl.columns["star_mass"]
    m_center = zeros.copy()
    if "m_center" in tbl.columns:
        m_center = np.maximum(tbl.columns["m_center"], 0.0)
    elif "log_xmstar" in tbl.columns and isinstance(
        header.get("msun"), (int, float)
    ):
        # star/private/pgstar_kipp.f90:217-222
        msun_g = float(header["msun"])
        if msun_g > 0.0:
            xm = np.power(10.0, tbl.columns["log_xmstar"]) / msun_g
            m_center = np.maximum(s_mass - xm, 0.0)
    return (
        m_center,
        np.maximum(s_mass - m_center, 0.0),
        r"Enclosed Mass [$M_r / M_\odot$]",
    )


def _draw_history_regions(
    fig: mpl_figure.Figure,
    ax: mpl_axes.Axes,
    x_edges: np.ndarray,
    y_center: np.ndarray,
    y_span: np.ndarray,
    tbl: history_compare.HistoryTable,
    *,
    yaxis: str,
    show_mixing: bool,
    show_burn: bool,
    burn_type_cutoff: float,
) -> tuple[list[mpl_patches.Patch], bool]:
    """Render burning and mixing polygons from history region columns."""
    legend_handles: list[mpl_patches.Patch] = []
    any_region = False

    burn_prefix = (
        ("burn_relr_type", "burn_relr_top")
        if yaxis == "radius"
        else ("burn_type", "burn_qtop")
    )
    burn_pairs = _contiguous_region_indices(tbl.columns, *burn_prefix)
    if show_burn and burn_pairs:
        burn_quads: list[np.ndarray] = []
        burn_vals: list[float] = []
        for row_i in range(tbl.num_rows):
            x0, x1 = float(x_edges[row_i]), float(x_edges[row_i + 1])
            q_bot = 0.0
            for b_types, b_tops in burn_pairs:
                b_val = float(b_types[row_i])
                q_top = float(b_tops[row_i])
                if not math.isfinite(b_val) or b_val < -9990.0:
                    break
                q_clamped = min(max(q_top, q_bot), 1.0)
                if abs(b_val) > burn_type_cutoff and q_clamped > q_bot:
                    y0 = float(y_center[row_i] + y_span[row_i] * q_bot)
                    y1 = float(y_center[row_i] + y_span[row_i] * q_clamped)
                    if y1 > y0:
                        burn_quads.append(
                            np.array(
                                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                dtype=np.float64,
                            )
                        )
                        burn_vals.append(b_val)
                q_bot = q_top
                if q_bot >= 1.0:
                    break
        if burn_quads:
            any_region = True
            b_arr = np.asarray(burn_vals, dtype=np.float64)
            has_cooling = bool(np.any(b_arr < 0.0))
            b_lim = max(float(np.max(np.abs(b_arr))), 1.0)
            norm: mpl_colors.Normalize
            if has_cooling:
                norm = mpl_colors.TwoSlopeNorm(
                    vmin=-b_lim, vcenter=0.0, vmax=b_lim
                )
                cmap_name = "coolwarm"
            else:
                norm = mpl_colors.Normalize(vmin=0.0, vmax=b_lim)
                cmap_name = "Blues"
            b_coll = mpl_collections.PolyCollection(
                burn_quads,
                array=b_arr,
                cmap=cmap_name,
                norm=norm,
                edgecolors="none",
                alpha=0.55,
                zorder=1,
            )
            ax.add_collection(b_coll)
            cbar = fig.colorbar(b_coll, ax=ax, pad=0.02)
            cbar.set_label(
                r"Net Power [$\mathrm{sgn}(\epsilon)"
                r"\log_{10}\max(1,|\epsilon|)$]"
            )

    mix_prefix = (
        ("mix_relr_type", "mix_relr_top")
        if yaxis == "radius"
        else ("mix_type", "mix_qtop")
    )
    mix_pairs = _contiguous_region_indices(tbl.columns, *mix_prefix)
    by_mix: dict[int, list[np.ndarray]] = {}
    if show_mixing and mix_pairs:
        for row_i in range(tbl.num_rows):
            x0, x1 = float(x_edges[row_i]), float(x_edges[row_i + 1])
            q_bot = 0.0
            for m_types, m_tops in mix_pairs:
                raw_t = float(m_types[row_i])
                if not math.isfinite(raw_t) or raw_t < 0.0:
                    break
                m_code = round(raw_t)
                q_top = float(m_tops[row_i])
                q_clamped = min(max(q_top, q_bot), 1.0)
                if m_code in _plot_support.MIXING_STYLE and q_clamped > q_bot:
                    y0 = float(y_center[row_i] + y_span[row_i] * q_bot)
                    y1 = float(y_center[row_i] + y_span[row_i] * q_clamped)
                    if y1 > y0:
                        by_mix.setdefault(m_code, []).append(
                            np.array(
                                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                                dtype=np.float64,
                            )
                        )
                q_bot = q_top
                if q_bot >= 1.0:
                    break
    elif (
        show_mixing
        and yaxis in ("mass", "q")
        and "mass_conv_core" in tbl.columns
    ):
        mcc = tbl.columns["mass_conv_core"]
        s_mass = tbl.columns.get(
            "star_mass", np.ones(tbl.num_rows, dtype=np.float64)
        )
        for row_i in range(tbl.num_rows):
            mc_val = float(mcc[row_i])
            if not math.isfinite(mc_val) or mc_val <= 0.0:
                continue
            y1 = (
                mc_val / max(float(s_mass[row_i]), 1.0e-30)
                if yaxis == "q"
                else mc_val
            )
            x0, x1 = float(x_edges[row_i]), float(x_edges[row_i + 1])
            by_mix.setdefault(1, []).append(
                np.array(
                    [[x0, 0.0], [x1, 0.0], [x1, y1], [x0, y1]],
                    dtype=np.float64,
                )
            )

    for m_code in sorted(by_mix):
        m_name, m_hex, m_hatch = _plot_support.MIXING_STYLE[m_code]
        any_region = True
        m_coll = mpl_collections.PolyCollection(
            by_mix[m_code],
            facecolors=mpl_colors.to_rgba(m_hex, 0.24),
            edgecolors=mpl_colors.to_rgba(m_hex, 0.80),
            linewidths=0.35,
            hatch=m_hatch or None,
            zorder=2,
        )
        ax.add_collection(m_coll)
        legend_handles.append(
            mpl_patches.Patch(
                facecolor=mpl_colors.to_rgba(m_hex, 0.28),
                edgecolor=m_hex,
                hatch=m_hatch or None,
                label=m_name,
            )
        )
    return legend_handles, any_region


def plot_history_kippenhahn(
    history_path: pathlib.Path | str | None = None,
    *,
    profile_paths: Sequence[pathlib.Path | str] | None = None,
    xaxis: Literal["model_number", "star_age"] = "model_number",
    yaxis: Literal["mass", "q", "radius"] = "mass",
    age_units: Literal["auto", "yr", "Myr"] = "auto",
    log_xaxis: bool = False,
    time_from_present: bool = False,
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
        px_vals, x_label = _resolve_kipp_xaxis(
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

        hx_vals, x_label = _resolve_kipp_xaxis(
            raw_hx,
            xaxis,
            age_units=age_units,
            log_xaxis=log_xaxis,
            time_from_present=time_from_present,
        )
        y_center, y_span, hist_ylabel = _resolve_history_scale(
            hist_tbl, hist_hdr, yaxis
        )
        if not y_label:
            y_label = hist_ylabel

        if not profile_paths:
            x_edges = _step_edges(hx_vals)
            reg_handles, _ = _draw_history_regions(
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
            handles.extend(reg_handles)

        if show_mass_boundaries:
            y_top_curve = y_center + y_span
            if yaxis != "q":
                (ln_top,) = ax.plot(
                    hx_vals,
                    y_top_curve,
                    color=style.OKABE_ITO["black"],
                    lw=1.45,
                    label=r"$M_{\mathrm{total}}$"
                    if yaxis == "mass"
                    else r"$R_*$",
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
                        # star/private/pgstar_kipp.f90:704
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

    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for ln in ax.get_lines():
        x_all.append(np.asarray(ln.get_xdata(), dtype=np.float64))
        y_all.append(np.asarray(ln.get_ydata(), dtype=np.float64))
    if x_all:
        x_cat = np.concatenate(x_all)
        ax.set_xlim(float(np.nanmin(x_cat)), float(np.nanmax(x_cat)))
    elif hist_tbl is not None:
        ax.autoscale_view()

    if yaxis == "q":
        ax.set_ylim(0.0, 1.02)
    elif y_all:
        y_cat = np.concatenate(y_all)
        ax.set_ylim(0.0, max(float(np.nanmax(y_cat)) * 1.04, 1.0e-3))
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
