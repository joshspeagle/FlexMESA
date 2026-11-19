# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Internal history-region polygon helpers for ``kippenhahn`` (`#173`).

Parses ``mix_type_*`` / ``mix_qtop_*`` / ``mix_relr_*`` and ``burn_type_*`` /
``burn_qtop_*`` / ``burn_relr_*`` columns
(`star/private/pgstar_kipp.f90:399-512, 724-796`) into ``PolyCollection``s.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.axes as mpl_axes
import matplotlib.collections as mpl_collections
import matplotlib.colors as mpl_colors
import matplotlib.figure as mpl_figure
import matplotlib.patches as mpl_patches
import numpy as np

from flexmesa.plotting import _compare_panels, _plot_support
from flexmesa.testing import history_compare


def step_edges(x_vals: np.ndarray) -> np.ndarray:
    """Compute ``N + 1`` cell boundaries around 1D step coordinates."""
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


def resolve_kipp_xaxis(
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
        abs_vals = np.abs(vals)
        pos = abs_vals[np.isfinite(abs_vals) & (abs_vals > 0.0)]
        floor = 0.5 * float(np.min(pos)) if pos.size > 0 else 1.0e-99
        vals = np.log10(np.maximum(abs_vals, floor))
        label = rf"$\log_{{10}}$({label})"
    return vals, label


def contiguous_region_pairs(
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


def resolve_history_scale(
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
            return zeros, tbl.columns["radius"], r"Radius [$R_r / R_\odot$]"
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


def draw_history_regions(
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
) -> list[mpl_patches.Patch]:
    """Render burning and mixing polygons from history region columns."""
    legend_handles: list[mpl_patches.Patch] = []
    burn_prefix = (
        ("burn_relr_type", "burn_relr_top")
        if yaxis == "radius"
        else ("burn_type", "burn_qtop")
    )
    burn_pairs = contiguous_region_pairs(tbl.columns, *burn_prefix)
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
            b_arr = np.asarray(burn_vals, dtype=np.float64)
            has_cooling = bool(np.any(b_arr < 0.0))
            b_lim = max(float(np.max(np.abs(b_arr))), 1.0)
            norm: mpl_colors.Normalize = (
                mpl_colors.TwoSlopeNorm(vmin=-b_lim, vcenter=0.0, vmax=b_lim)
                if has_cooling
                else mpl_colors.Normalize(vmin=0.0, vmax=b_lim)
            )
            b_coll = mpl_collections.PolyCollection(
                burn_quads,
                array=b_arr,
                cmap="coolwarm" if has_cooling else "Blues",
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
    mix_pairs = contiguous_region_pairs(tbl.columns, *mix_prefix)
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
    return legend_handles
