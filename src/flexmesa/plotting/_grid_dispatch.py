# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal cell dispatcher and namelist parser for ``pgstar_grid`` (`#173`)."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.gridspec as mpl_gridspec
import numpy as np

from flexmesa import constants
from flexmesa.plotting import (
    _plot_support,
    compare,
    kippenhahn,
    pgbinary_panels,
    pgstar_extra,
    pgstar_nucleosyn,
    pgstar_panels,
    pgstar_summary,
    pgstar_text,
    stellar_plots,
)

_NUMBERED_PANEL_RE = re.compile(r"^([a-z_]+)([1-9])$")

# Upstream MESA 26.4.1 default Grid<N> layouts:
# star/defaults/pgstar.defaults:6797-6852 (Grid1), 6907-6974 (Grid2),
# 7031-7073 (Grid3), 7123-7204 (Grid4);
# binary/defaults/pgbinary.defaults:2267-2283 (Grid1), 2338-2354 (Grid2),
# 2463-2479 (Grid4).
_DEFAULT_GRID_LAYOUTS: dict[
    tuple[str, int], tuple[int, int, tuple[tuple[str, int, int, int, int], ...]]
] = {
    ("pgstar", 1): (
        8,
        2,
        (
            ("TRho_Profile", 0, 0, 4, 2),
            ("HR", 4, 0, 2, 1),
            ("TRho", 4, 1, 2, 1),
            ("Text_Summary1", 6, 0, 2, 2),
        ),
    ),
    ("pgstar", 2): (
        8,
        4,
        (
            ("TRho_Profile", 0, 0, 4, 2),
            ("HR", 4, 0, 2, 1),
            ("TRho", 4, 1, 2, 1),
            ("Text_Summary1", 6, 0, 2, 4),
            ("Abundance", 0, 2, 6, 2),
        ),
    ),
    ("pgstar", 3): (
        7,
        1,
        (
            ("TRho_Profile", 0, 0, 3, 1),
            ("Summary_Profile", 3, 0, 2, 1),
            ("Kipp", 5, 0, 2, 1),
        ),
    ),
    ("pgstar", 4): (
        9,
        10,
        (
            ("TRho_Profile", 0, 0, 3, 5),
            ("Summary_Profile", 0, 5, 3, 5),
            ("HR", 3, 0, 2, 2),
            ("TRho", 5, 0, 2, 2),
            ("Kipp", 3, 2, 4, 7),
            ("Text_Summary1", 7, 0, 2, 10),
        ),
    ),
    ("pgbinary", 1): (
        8,
        2,
        (("Text_Summary1", 6, 0, 2, 2),),
    ),
    ("pgbinary", 2): (
        8,
        4,
        (("Text_Summary1", 6, 0, 2, 4),),
    ),
    ("pgbinary", 4): (
        9,
        10,
        (("Text_Summary1", 7, 0, 2, 10),),
    ),
}

# Upstream MESA 26.4.1 Profile_Panels1 & History_Panels1 defaults:
# star/defaults/pgstar.defaults:2848-2852, 4476-4525;
# binary/defaults/pgbinary.defaults:1198-1202.
_DEFAULT_PROFILE_PANELS: tuple[tuple[str, str | None], ...] = (
    ("logT", "entropy"),
    ("logRho", "logP"),
)
_DEFAULT_STAR_HISTORY_PANELS: tuple[tuple[str, str | None], ...] = (
    ("log_center_T", "log_center_Rho"),
    ("log_L", "log_Teff"),
    ("star_mass", "log_abs_mdot"),
)
_DEFAULT_BINARY_HISTORY_PANELS: tuple[tuple[str, str | None], ...] = (
    ("lg_mtransfer_rate", None),
)


def parse_grid_namelist(
    inlist_path: pathlib.Path | str,
    *,
    grid_index: int,
    namelist_name: str | None,
) -> tuple[int, int, str, list[tuple[str, int, int, int, int]]]:
    """Extract ``(num_rows, num_cols, title, cells)`` from ``inlist_path``."""
    blocks = _plot_support.load_plotting_namelist_blocks(
        inlist_path, namelist_name=namelist_name
    )
    prefix = f"grid{grid_index}_"
    num_rows = num_cols = num_plots = 0
    title = ""
    names: dict[int, str] = {}
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    rspan: dict[int, int] = {}
    cspan: dict[int, int] = {}
    star_sub: dict[str, str] = {}
    matched_group = ""
    saw_grid_key = False

    for blk in blocks:
        bname = blk.name.lower()
        if namelist_name is not None and bname != namelist_name.lower():
            continue
        if bname not in ("pgstar", "pgbinary"):
            continue
        matched_group = bname
        for asg in blk.assignments:
            k_low = asg.name.lower()
            if k_low in ("star1_plot_name", "star2_plot_name"):
                star_sub[k_low[:5]] = str(asg.value).strip()
            if not k_low.startswith(prefix):
                continue
            saw_grid_key = True
            tail = k_low[len(prefix) :]
            if asg.subscript is not None and ":" in asg.subscript:
                if tail == "plot_name":
                    names.clear()
                elif tail == "plot_row":
                    rows.clear()
                elif tail == "plot_col":
                    cols.clear()
                elif tail == "plot_rowspan":
                    rspan.clear()
                elif tail == "plot_colspan":
                    cspan.clear()
                continue
            sub_i = (
                int(asg.subscript)
                if asg.subscript is not None and asg.subscript.isdigit()
                else None
            )
            if tail == "num_rows":
                num_rows = int(asg.value)
            elif tail == "num_cols":
                num_cols = int(asg.value)
            elif tail == "num_plots":
                num_plots = int(asg.value)
            elif tail == "title":
                title = str(asg.value)
            elif tail == "plot_name" and sub_i is not None:
                names[sub_i] = str(asg.value).strip()
            elif tail == "plot_row" and sub_i is not None:
                rows[sub_i] = int(asg.value)
            elif tail == "plot_col" and sub_i is not None:
                cols[sub_i] = int(asg.value)
            elif tail == "plot_rowspan" and sub_i is not None:
                rspan[sub_i] = int(asg.value)
            elif tail == "plot_colspan" and sub_i is not None:
                cspan[sub_i] = int(asg.value)

    max_slot = max(num_plots, max(names, default=0))
    cells: list[tuple[str, int, int, int, int]] = []
    for slot in range(1, max_slot + 1):
        pname = names.get(slot, "").strip()
        if not pname:
            continue
        p_low = pname.lower()
        if star_sub.get(p_low):
            pname = f"{pname}:{star_sub[p_low]}"
        cells.append(
            (
                pname,
                max(rows.get(slot, 1) - 1, 0),
                max(cols.get(slot, 1) - 1, 0),
                max(rspan.get(slot, 1), 1),
                max(cspan.get(slot, 1), 1),
            )
        )

    if not cells and saw_grid_key:
        eff_grp = (namelist_name or matched_group or "pgstar").lower()
        default_spec = _DEFAULT_GRID_LAYOUTS.get((eff_grp, grid_index))
        if default_spec is not None:
            d_rows, d_cols, d_cells = default_spec
            num_rows = num_rows if num_rows > 0 else d_rows
            num_cols = num_cols if num_cols > 0 else d_cols
            cells = list(d_cells)

    if not cells or num_rows <= 0 or num_cols <= 0:
        raise ValueError(
            f"Grid{grid_index} specification incomplete in {inlist_path}"
        )
    return num_rows, num_cols, title, cells


def _parse_stacked_panel_spec(
    inlist_path: pathlib.Path | str | None,
    prefix: str,
    *,
    is_binary: bool,
) -> tuple[str | None, list[tuple[str, str | None]]]:
    """Parse ``Profile_Panels<N>`` or ``History_Panels<N>`` rows from inlist."""
    if inlist_path is None:
        return None, []
    blocks = _plot_support.load_plotting_namelist_blocks(
        inlist_path, namelist_name="pgbinary" if is_binary else None
    )
    xaxis_name: str | None = None
    num_panels = 0
    y_left: dict[int, str] = {}
    y_right: dict[int, str] = {}
    for blk in blocks:
        if blk.name.lower() not in ("pgstar", "pgbinary"):
            continue
        for asg in blk.assignments:
            k_low = asg.name.lower()
            if not k_low.startswith(prefix):
                continue
            tail = k_low[len(prefix) :]
            if asg.subscript is not None and ":" in asg.subscript:
                if tail == "yaxis_name":
                    y_left.clear()
                elif tail == "other_yaxis_name":
                    y_right.clear()
                continue
            sub_i = (
                int(asg.subscript)
                if asg.subscript is not None and asg.subscript.isdigit()
                else None
            )
            if tail == "xaxis_name":
                val = str(asg.value).strip()
                if val:
                    xaxis_name = val
            elif tail == "num_panels":
                num_panels = int(asg.value)
            elif tail == "yaxis_name" and sub_i is not None:
                y_left[sub_i] = str(asg.value).strip()
            elif tail == "other_yaxis_name" and sub_i is not None:
                y_right[sub_i] = str(asg.value).strip()

    n_slots = num_panels if num_panels > 0 else max(y_left, default=0)
    rows: list[tuple[str, str | None]] = []
    for idx in range(1, n_slots + 1):
        left = y_left.get(idx, "").strip()
        if not left:
            continue
        right = y_right.get(idx, "").strip() or None
        rows.append((left, right))
    return xaxis_name, rows


def _parse_history_track_spec(
    inlist_path: pathlib.Path | str | None,
    track_index: int,
    *,
    is_binary: bool,
) -> tuple[str, str, bool, bool]:
    """Parse ``History_Track<N>_*`` x/y names and reversals from inlist."""
    x_col, y_col = "log_center_T", "log_L"
    rev_x, rev_y = True, False
    if inlist_path is None:
        return x_col, y_col, rev_x, rev_y
    blocks = _plot_support.load_plotting_namelist_blocks(
        inlist_path, namelist_name="pgbinary" if is_binary else None
    )
    prefix = f"history_track{track_index}_"
    for blk in blocks:
        if blk.name.lower() not in ("pgstar", "pgbinary"):
            continue
        for asg in blk.assignments:
            k_low = asg.name.lower()
            if not k_low.startswith(prefix):
                continue
            tail = k_low[len(prefix) :]
            if tail == "xname" and str(asg.value).strip():
                x_col = str(asg.value).strip()
            elif tail == "yname" and str(asg.value).strip():
                y_col = str(asg.value).strip()
            elif tail == "reverse_xaxis":
                rev_x = bool(asg.value)
            elif tail == "reverse_yaxis":
                rev_y = bool(asg.value)
    return x_col, y_col, rev_x, rev_y


def _split_cell_subgrid(
    ax: mpl_axes.Axes, n_rows: int, n_cols: int = 1
) -> tuple[mpl_figure.Figure, mpl_gridspec.GridSpecFromSubplotSpec]:
    """Replace placeholder ``ax`` with a nested ``GridSpecFromSubplotSpec``."""
    fig = ax.get_figure()
    if not isinstance(fig, mpl_figure.Figure):
        raise TypeError("grid cell axes must belong to a Figure")
    ss = ax.get_subplotspec()
    if ss is None:
        raise ValueError("grid cell axes has no SubplotSpec to subdivide")
    sub_gs = mpl_gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=ss
    )
    fig.delaxes(ax)
    return fig, sub_gs


def _filter_available_pairs(
    pairs: Sequence[tuple[str, str | None]], available: set[str]
) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for left, right in pairs:
        if left not in available:
            continue
        eff_right = (
            right if (right is not None and right in available) else None
        )
        out.append((left, eff_right))
    return out


def _render_point_mass_star_cell(
    ax: mpl_axes.Axes, star_no: int, mass_msun: float
) -> list[mpl_axes.Axes]:
    """Render upstream point-mass banner (`pgbinary_star.f90:199-220`)."""
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        (
            f"Star {star_no} not simulated\n"
            f"point mass of {mass_msun:.2f} $M_\\odot$"
        ),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8.2,
    )
    ax.set_title(f"Star {star_no}: Point Mass", fontsize=8.8)
    return [ax]


def dispatch_grid_cell(
    ax: mpl_axes.Axes,
    panel_name: str,
    *,
    history_path: pathlib.Path | str | None,
    ref_history_path: pathlib.Path | str | None,
    profile_path: pathlib.Path | str | None,
    ref_profile_path: pathlib.Path | str | None,
    profile_paths: Sequence[pathlib.Path | str] | None,
    binary_history_path: pathlib.Path | str | None,
    ref_binary_history_path: pathlib.Path | str | None,
    star1_history_path: pathlib.Path | str | None,
    star2_history_path: pathlib.Path | str | None,
    star1_profile_path: pathlib.Path | str | None,
    star2_profile_path: pathlib.Path | str | None,
    star1_profile_paths: Sequence[pathlib.Path | str] | None = None,
    star2_profile_paths: Sequence[pathlib.Path | str] | None = None,
    inlist_path: pathlib.Path | str | None = None,
    star1_inlist_path: pathlib.Path | str | None = None,
    star2_inlist_path: pathlib.Path | str | None = None,
    mesa_dir: pathlib.Path | str | None = None,
    _depth: int = 0,
) -> list[mpl_axes.Axes]:
    """Render one named ``pgstar``/``pgbinary`` panel into ``ax``."""
    from flexmesa.plotting import pgstar_grid as _pg_grid

    raw = panel_name.strip()
    low = raw.lower()

    if low.startswith(("star1", "star2")):
        star_no = 1 if low.startswith("star1") else 2
        if binary_history_path is not None:
            b_tbl, _ = _plot_support.load_history_with_header(
                binary_history_path
            )
            pm_idx = (
                round(float(b_tbl.columns["point_mass_index"][-1]))
                if "point_mass_index" in b_tbl.columns
                else 0
            )
            if pm_idx == star_no:
                m_col = f"star_{star_no}_mass"
                m_val = (
                    float(b_tbl.columns[m_col][-1])
                    if m_col in b_tbl.columns
                    else float("nan")
                )
                return _render_point_mass_star_cell(ax, star_no, m_val)
        sub_name = raw.split(":", 1)[1].strip() if ":" in raw else "HR"
        s_hist = star1_history_path if star_no == 1 else star2_history_path
        s_prof = star1_profile_path if star_no == 1 else star2_profile_path
        s_profs = star1_profile_paths if star_no == 1 else star2_profile_paths
        s_inlist = (
            star1_inlist_path if star_no == 1 else star2_inlist_path
        ) or inlist_path
        child_axes = dispatch_grid_cell(
            ax,
            sub_name,
            history_path=s_hist,
            ref_history_path=None,
            profile_path=s_prof,
            ref_profile_path=None,
            profile_paths=s_profs,
            binary_history_path=None,
            ref_binary_history_path=None,
            star1_history_path=None,
            star2_history_path=None,
            star1_profile_path=None,
            star2_profile_path=None,
            inlist_path=s_inlist,
            mesa_dir=mesa_dir,
            _depth=_depth + 1,
        )
        if child_axes:
            old_t = child_axes[0].get_title()
            child_axes[0].set_title(
                f"Star {star_no}: {old_t}" if old_t else f"Star {star_no}",
                fontsize=8.8,
            )
        return child_axes

    if low in _pg_grid.TRACK_PRESETS:
        h_src = history_path or binary_history_path
        if h_src is None:
            raise ValueError(f"panel {raw!r} requires history_path")
        r_src = (
            ref_history_path
            if history_path is not None
            else (ref_history_path or ref_binary_history_path)
        )
        _pg_grid.plot_history_track(
            h_src,
            preset=raw,
            ref_history_path=r_src,
            mesa_dir=mesa_dir,
            ax=ax,
        )
        return [ax]

    if low.startswith("history_track:"):
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"expected 'History_Track:<x_col>:<y_col>', got {raw!r}"
            )
        h_src = history_path or binary_history_path
        if h_src is None:
            raise ValueError(f"panel {raw!r} requires history_path")
        r_src = (
            ref_history_path
            if history_path is not None
            else (ref_history_path or ref_binary_history_path)
        )
        _pg_grid.plot_history_track(
            h_src,
            parts[1],
            parts[2],
            ref_history_path=r_src,
            mesa_dir=mesa_dir,
            ax=ax,
        )
        return [ax]

    if low == "kipp":
        kippenhahn.plot_history_kippenhahn(
            history_path=history_path,
            profile_paths=profile_paths,
            ax=ax,
        )
        return [ax]
    if low == "trho_profile":
        if profile_path is None:
            raise ValueError("TRho_Profile requires profile_path")
        pgstar_panels.plot_trho_profile(
            profile_path,
            ref_profile_path=ref_profile_path,
            mesa_dir=mesa_dir,
            ax=ax,
        )
        return [ax]

    prof_map: dict[str, Callable[..., Any]] = {
        "power": pgstar_panels.plot_power_profile,
        "abundance": pgstar_panels.plot_abundance_profile,
        "mixing": pgstar_panels.plot_mixing_profile,
        "dynamo": pgstar_extra.plot_dynamo_profile,
        "production": pgstar_nucleosyn.plot_production_profile,
        "summary_profile": pgstar_summary.plot_summary_profile,
        "summary_burn": pgstar_summary.plot_summary_burn_profile,
    }
    if low in prof_map:
        if profile_path is None:
            raise ValueError(f"{raw} requires profile_path")
        prof_map[low](profile_path, ref_profile_path=ref_profile_path, ax=ax)
        return [ax]

    if low == "network":
        if profile_path is None:
            raise ValueError("Network requires profile_path")
        pgstar_nucleosyn.plot_network_chart(profile_path, ax=ax)
        return [ax]

    if low == "rti":
        if history_path is None and not profile_paths:
            raise ValueError("rti requires history_path or profile_paths")
        pgstar_extra.plot_rti_history(
            history_path=history_path,
            profile_paths=profile_paths,
            ref_history_path=ref_history_path,
            ax=ax,
        )
        return [ax]

    if low == "mode_prop":
        if profile_path is None:
            raise ValueError("Mode_Prop requires profile_path")
        pt = compare.load_profile_file(profile_path)
        r_rsun = pt.columns["radius"]
        r_max = max(float(np.nanmax(r_rsun)), 1.0e-30)
        r_cm = pt.columns.get("radius_cm")
        if r_cm is None:
            r_cm = r_rsun * float(pt.header.get("rsun", constants.RSUN))
        stellar_plots.plot_propagation_diagram(
            r_rsun / r_max,
            pt.columns["brunt_N2"],
            pt.columns["csound"],
            r_cm,
            ax=ax,
        )
        return [ax]

    if low == "summary_history":
        if binary_history_path is not None and history_path is None:
            pgbinary_panels.plot_pgbinary_summary_history(
                binary_history_path,
                mesa_binary_history=ref_binary_history_path,
                ax=ax,
            )
            return [ax]
        if history_path is None:
            raise ValueError("Summary_History requires history_path")
        _pg_grid.plot_summary_history(
            history_path, ref_history_path=ref_history_path, ax=ax
        )
        return [ax]

    if low == "orbit":
        b_src = binary_history_path or history_path
        if b_src is None:
            raise ValueError("Orbit requires binary_history_path")
        pgbinary_panels.plot_pgbinary_orbit(
            b_src, mesa_binary_history=ref_binary_history_path, ax=ax
        )
        return [ax]

    # Numbered panel families: Text_Summary<N>, Profile_Panels<N>,
    # History_Panels<N>, History_Track<N>, Grid<N>.
    num_match = _NUMBERED_PANEL_RE.match(low)
    fam_stem = num_match.group(1) if num_match else low
    fam_idx = int(num_match.group(2)) if num_match else 1
    is_bin_ctx = binary_history_path is not None and history_path is None

    if fam_stem == "text_summary":
        h_src = (
            binary_history_path
            if is_bin_ctx
            else (history_path or binary_history_path)
        )
        if h_src is None:
            raise ValueError(f"{raw} requires history_path or binary_history")
        r_src = ref_binary_history_path if is_bin_ctx else ref_history_path
        ts_layout: Sequence[Sequence[str]] | None = None
        ts_title = ""
        if inlist_path is not None:
            try:
                ts_spec = pgstar_text.parse_inlist_text_summary_spec(
                    inlist_path,
                    summary_index=fam_idx,
                    namelist_name="pgbinary" if is_bin_ctx else None,
                )
                ts_layout = ts_spec.layout
                ts_title = ts_spec.title
            except ValueError:
                ts_layout = None
        pgstar_text.plot_text_summary(
            h_src,
            layout=ts_layout,
            is_binary=is_bin_ctx,
            ref_history_path=r_src,
            title=ts_title,
            ax=ax,
        )
        return [ax]

    if fam_stem == "history_track":
        h_src = (
            binary_history_path
            if is_bin_ctx
            else (history_path or binary_history_path)
        )
        if h_src is None:
            raise ValueError(f"{raw} requires history_path")
        r_src = ref_binary_history_path if is_bin_ctx else ref_history_path
        x_col, y_col, rev_x, rev_y = _parse_history_track_spec(
            inlist_path, fam_idx, is_binary=is_bin_ctx
        )
        _pg_grid.plot_history_track(
            h_src,
            x_col,
            y_col,
            ref_history_path=r_src,
            reverse_xaxis=rev_x,
            reverse_yaxis=rev_y,
            mesa_dir=mesa_dir,
            ax=ax,
        )
        return [ax]

    if fam_stem == "profile_panels" or low.startswith("profile_panels:"):
        if profile_path is None:
            raise ValueError(f"{raw} requires profile_path")
        pt = compare.load_profile_file(profile_path)
        p_xaxis = "mass"
        p_rows: list[tuple[str, str | None]] = []
        if ":" in raw:
            p_rows = [
                (tok.strip(), None)
                for tok in raw.split(":", 1)[1].split(",")
                if tok.strip()
            ]
        else:
            parsed_x, parsed_rows = _parse_stacked_panel_spec(
                inlist_path, f"profile_panels{fam_idx}_", is_binary=False
            )
            if parsed_x is not None:
                p_xaxis = parsed_x
            p_rows = parsed_rows or _filter_available_pairs(
                _DEFAULT_PROFILE_PANELS, set(pt.columns)
            )
        if not p_rows:
            raise ValueError(
                f"{raw}: no configured or default profile columns found"
            )
        if len(p_rows) == 1:
            pgstar_summary.plot_profile_custom_panels(
                profile_path,
                p_rows,
                ref_profile_path=ref_profile_path,
                xaxis=p_xaxis,
                axes=[ax],
            )
            return [ax]
        fig, sub_gs = _split_cell_subgrid(ax, len(p_rows), 1)
        sub_axes: list[mpl_axes.Axes] = []
        for r_i in range(len(p_rows)):
            share_ax = sub_axes[0] if sub_axes else None
            sub_axes.append(fig.add_subplot(sub_gs[r_i, 0], sharex=share_ax))
        pgstar_summary.plot_profile_custom_panels(
            profile_path,
            p_rows,
            ref_profile_path=ref_profile_path,
            xaxis=p_xaxis,
            axes=sub_axes,
        )
        return sub_axes

    if fam_stem == "history_panels" or low.startswith("history_panels:"):
        h_src = (
            binary_history_path
            if is_bin_ctx
            else (history_path or binary_history_path)
        )
        if h_src is None:
            raise ValueError(f"{raw} requires history_path")
        r_src = ref_binary_history_path if is_bin_ctx else ref_history_path
        h_tbl, _ = _plot_support.load_history_with_header(h_src)
        h_xaxis = "model_number"
        h_rows: list[tuple[str, str | None]] = []
        if ":" in raw:
            h_rows = [
                (tok.strip(), None)
                for tok in raw.split(":", 1)[1].split(",")
                if tok.strip()
            ]
        else:
            parsed_x, parsed_rows = _parse_stacked_panel_spec(
                inlist_path,
                f"history_panels{fam_idx}_",
                is_binary=is_bin_ctx,
            )
            if parsed_x is not None:
                h_xaxis = parsed_x
            defaults_hp = (
                _DEFAULT_BINARY_HISTORY_PANELS
                if is_bin_ctx
                else _DEFAULT_STAR_HISTORY_PANELS
            )
            h_rows = parsed_rows or _filter_available_pairs(
                defaults_hp, set(h_tbl.columns)
            )
        if not h_rows:
            raise ValueError(
                f"{raw}: no configured or default history columns found"
            )
        if len(h_rows) == 1:
            _pg_grid.plot_history_panels(
                h_src,
                h_rows,
                ref_history_path=r_src,
                xaxis=h_xaxis,
                axes=[ax],
            )
            return [ax]
        fig, sub_gs = _split_cell_subgrid(ax, len(h_rows), 1)
        sub_axes_h: list[mpl_axes.Axes] = []
        for r_i in range(len(h_rows)):
            share_ax = sub_axes_h[0] if sub_axes_h else None
            sub_axes_h.append(fig.add_subplot(sub_gs[r_i, 0], sharex=share_ax))
        _pg_grid.plot_history_panels(
            h_src,
            h_rows,
            ref_history_path=r_src,
            xaxis=h_xaxis,
            axes=sub_axes_h,
        )
        return sub_axes_h

    if fam_stem == "color_magnitude" or low.startswith("color_magnitude:"):
        h_src = history_path or binary_history_path
        if h_src is None:
            raise ValueError(f"{raw} requires history_path")
        r_src = (
            ref_history_path
            if history_path is not None
            else (ref_history_path or ref_binary_history_path)
        )
        if ":" in raw:
            parts = raw.split(":")
            if len(parts) != 3:
                raise ValueError(
                    "expected 'Color_magnitude:<x1[-x2]>:<y1[-y2],...>', "
                    f"got {raw!r}"
                )
            x_toks = [t.strip() for t in parts[1].split("-", 1)]
            cx1 = x_toks[0]
            cx2 = x_toks[1] if len(x_toks) == 2 and x_toks[1] else None
            c_rows: list[tuple[str, str | None]] = []
            for r_tok in parts[2].split(","):
                r_clean = r_tok.strip()
                if not r_clean:
                    continue
                y_toks = [t.strip() for t in r_clean.split("-", 1)]
                c_rows.append(
                    (
                        y_toks[0],
                        y_toks[1] if len(y_toks) == 2 and y_toks[1] else None,
                    )
                )
            if not c_rows:
                raise ValueError(f"{raw}: no y columns specified")
            if len(c_rows) == 1:
                pgstar_extra.plot_color_magnitude(
                    h_src,
                    cx1,
                    cx2,
                    c_rows,
                    ref_history_path=r_src,
                    axes=[ax],
                )
                return [ax]
            fig, sub_gs = _split_cell_subgrid(ax, len(c_rows), 1)
            sub_axes_c: list[mpl_axes.Axes] = []
            for r_i in range(len(c_rows)):
                share_ax = sub_axes_c[0] if sub_axes_c else None
                sub_axes_c.append(
                    fig.add_subplot(sub_gs[r_i, 0], sharex=share_ax)
                )
            pgstar_extra.plot_color_magnitude(
                h_src,
                cx1,
                cx2,
                c_rows,
                ref_history_path=r_src,
                axes=sub_axes_c,
            )
            return sub_axes_c

        n_cmd_rows = 1
        use_inlist: pathlib.Path | str | None = None
        if inlist_path is not None:
            try:
                cmd_spec = pgstar_extra.parse_inlist_cmd_spec(
                    inlist_path, panel_index=fam_idx
                )
                n_cmd_rows = len(cmd_spec.panels)
                use_inlist = inlist_path
            except ValueError:
                use_inlist = None
        if n_cmd_rows == 1:
            pgstar_extra.plot_color_magnitude(
                h_src,
                ref_history_path=r_src,
                inlist_path=use_inlist,
                panel_index=fam_idx,
                axes=[ax],
            )
            return [ax]
        fig, sub_gs = _split_cell_subgrid(ax, n_cmd_rows, 1)
        sub_axes_c2: list[mpl_axes.Axes] = []
        for r_i in range(n_cmd_rows):
            share_ax = sub_axes_c2[0] if sub_axes_c2 else None
            sub_axes_c2.append(fig.add_subplot(sub_gs[r_i, 0], sharex=share_ax))
        pgstar_extra.plot_color_magnitude(
            h_src,
            ref_history_path=r_src,
            inlist_path=use_inlist,
            panel_index=fam_idx,
            axes=sub_axes_c2,
        )
        return sub_axes_c2

    if fam_stem == "grid":
        if _depth >= 2:
            raise ValueError(
                f"nested Grid recursion depth exceeded while rendering {raw!r}"
            )
        if inlist_path is not None:
            try:
                sr, sc, _st, s_cells = parse_grid_namelist(
                    inlist_path,
                    grid_index=fam_idx,
                    namelist_name="pgbinary" if is_bin_ctx else None,
                )
            except ValueError:
                d_key = ("pgbinary" if is_bin_ctx else "pgstar", fam_idx)
                if d_key not in _DEFAULT_GRID_LAYOUTS:
                    raise
                sr, sc, d_cells = _DEFAULT_GRID_LAYOUTS[d_key]
                s_cells = list(d_cells)
        else:
            d_key = ("pgbinary" if is_bin_ctx else "pgstar", fam_idx)
            if d_key not in _DEFAULT_GRID_LAYOUTS:
                raise ValueError(
                    f"{raw} has no default layout; supply inlist_path"
                )
            sr, sc, d_cells = _DEFAULT_GRID_LAYOUTS[d_key]
            s_cells = list(d_cells)
        fig, sub_gs = _split_cell_subgrid(ax, sr, sc)
        nested_axes: list[mpl_axes.Axes] = []
        for s_name, s_row, s_col, s_rspan, s_cspan in s_cells:
            c_ax = fig.add_subplot(
                sub_gs[s_row : s_row + s_rspan, s_col : s_col + s_cspan]
            )
            nested_axes.extend(
                dispatch_grid_cell(
                    c_ax,
                    s_name,
                    history_path=history_path,
                    ref_history_path=ref_history_path,
                    profile_path=profile_path,
                    ref_profile_path=ref_profile_path,
                    profile_paths=profile_paths,
                    binary_history_path=binary_history_path,
                    ref_binary_history_path=ref_binary_history_path,
                    star1_history_path=star1_history_path,
                    star2_history_path=star2_history_path,
                    star1_profile_path=star1_profile_path,
                    star2_profile_path=star2_profile_path,
                    star1_profile_paths=star1_profile_paths,
                    star2_profile_paths=star2_profile_paths,
                    inlist_path=inlist_path,
                    star1_inlist_path=star1_inlist_path,
                    star2_inlist_path=star2_inlist_path,
                    mesa_dir=mesa_dir,
                    _depth=_depth + 1,
                )
            )
        return nested_axes

    raise ValueError(f"unsupported pgstar/pgbinary grid panel {raw!r}")
