# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Upstream ``pgstar``/``pgbinary`` track, history, and ``Grid`` builders.

Ports ``History_Track<N>`` and all named two-column evolutionary track windows
(`star/private/pgstar_hist_track.f90:53-280`, `pgstar_tmaxrho.f90:57-105`,
`pgstar_logg_teff.f90:55-88`, `pgstar_logg_logt.f90:58-88`,
`pgstar_logl_r.f90:55-96`, `pgstar_logl_v.f90:55-93`,
`pgstar_l_teff.f90:57-87`, `pgstar_l_r.f90:57-87`, `pgstar_l_v.f90:57-87`,
`pgstar_r_teff.f90:57-87`, `pgstar_r_l.f90:57-87`), ``Summary_History``
(`star/private/pgstar_summary_history.f90:54-265`), ``History_Panels<N>``
(`star/private/pgstar_history_panels.f90:31-240`), and the ``Grid<N>``
multi-window layout composer (`star/private/pgstar_grid.f90:50-260`,
`binary/private/pgbinary_grid.f90:32-260`).
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
from collections.abc import Sequence
from typing import Any

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.gridspec as mpl_gridspec
import matplotlib.pyplot as plt
import numpy as np

from flexmesa.plotting import (
    _compare_panels,
    _grid_dispatch,
    _plot_support,
    style,
)
from flexmesa.testing import history_compare


@dataclasses.dataclass(frozen=True)
class TrackPreset:
    """Column and axis specification for an upstream ``pgstar`` track window."""

    x_candidates: tuple[str, ...]
    y_candidates: tuple[str, ...]
    xlabel: str
    ylabel: str
    reverse_x: bool = False
    reverse_y: bool = False
    degeneracy_guide: bool = False


TRACK_PRESETS: dict[str, TrackPreset] = {
    "hr": TrackPreset(
        ("log_Teff",),
        ("log_L",),
        r"$\log_{10}(T_{\mathrm{eff}} / \mathrm{K})$",
        r"$\log_{10}(L / L_\odot)$",
        reverse_x=True,
    ),
    "logl_teff": TrackPreset(
        ("log_Teff",),
        ("log_L",),
        r"$\log_{10}(T_{\mathrm{eff}} / \mathrm{K})$",
        r"$\log_{10}(L / L_\odot)$",
        reverse_x=True,
    ),
    "trho": TrackPreset(
        ("log_center_Rho",),
        ("log_center_T",),
        r"$\log_{10}(\rho_c / \mathrm{g\,cm^{-3}})$",
        r"$\log_{10}(T_c / \mathrm{K})$",
        degeneracy_guide=True,
    ),
    "tmaxrho": TrackPreset(
        ("log_center_Rho",),
        ("log_max_T",),
        r"$\log_{10}(\rho_c / \mathrm{g\,cm^{-3}})$",
        r"$\log_{10}(T_{\max} / \mathrm{K})$",
        degeneracy_guide=True,
    ),
    "logg_teff": TrackPreset(
        ("effective_T", "Teff"),
        ("log_g",),
        r"$T_{\mathrm{eff}}$ [$\mathrm{K}$]",
        r"$\log_{10}(g / \mathrm{cm\,s^{-2}})$",
        reverse_x=True,
        reverse_y=True,
    ),
    "logg_logt": TrackPreset(
        ("log_Teff",),
        ("log_g",),
        r"$\log_{10}(T_{\mathrm{eff}} / \mathrm{K})$",
        r"$\log_{10}(g / \mathrm{cm\,s^{-2}})$",
        reverse_x=True,
        reverse_y=True,
    ),
    "logl_r": TrackPreset(
        ("radius", "photosphere_r", "log_R"),
        ("log_L",),
        r"$R / R_\odot$",
        r"$\log_{10}(L / L_\odot)$",
    ),
    "logl_v": TrackPreset(
        ("v_surf_km_s", "photosphere_v_km_s"),
        ("log_L",),
        r"$v_{\mathrm{surf}}$ [$\mathrm{km\,s^{-1}}$]",
        r"$\log_{10}(L / L_\odot)$",
    ),
    "l_teff": TrackPreset(
        ("effective_T", "Teff"),
        ("luminosity",),
        r"$T_{\mathrm{eff}}$ [$\mathrm{K}$]",
        r"$L / L_\odot$",
        reverse_x=True,
    ),
    "l_r": TrackPreset(
        ("radius", "photosphere_r"),
        ("luminosity",),
        r"$R / R_\odot$",
        r"$L / L_\odot$",
    ),
    "l_v": TrackPreset(
        ("v_surf_km_s",),
        ("luminosity",),
        r"$v_{\mathrm{surf}}$ [$\mathrm{km\,s^{-1}}$]",
        r"$L / L_\odot$",
    ),
    "r_teff": TrackPreset(
        ("effective_T", "Teff"),
        ("radius", "photosphere_r"),
        r"$T_{\mathrm{eff}}$ [$\mathrm{K}$]",
        r"$R / R_\odot$",
        reverse_x=True,
    ),
    "r_l": TrackPreset(
        ("luminosity",),
        ("radius", "photosphere_r"),
        r"$L / L_\odot$",
        r"$R / R_\odot$",
    ),
}

_STAR_SUMMARY_HISTORY_COLS: tuple[tuple[str, str, str], ...] = (
    ("log_center_T", r"$\log T_c$ rel", "rel"),
    ("log_center_Rho", r"$\log\rho_c$ rel", "rel"),
    ("log_L", r"$\log L$ rel", "rel"),
    ("log_Teff", r"$\log T_{\mathrm{eff}}$ rel", "rel"),
    ("log_R", r"$\log R$ rel", "rel"),
    ("center_h1", r"$X_c$", "raw"),
    ("center_he4", r"$Y_c$", "raw"),
)


@dataclasses.dataclass(frozen=True)
class GridPanelSpec:
    """Zero-based ``GridSpec`` cell placement for ``plot_pgstar_grid``.

    Attributes:
        name: Panel identifier (such as ``"HR"``, ``"Kipp"``,
            ``"TRho_Profile"``, ``"Power"``, ``"Abundance"``, ``"Orbit"``,
            ``"Star1:HR"``, or ``"History_Track:star_age:log_L"``).
        row: Zero-based top row index.
        col: Zero-based left column index.
        rowspan: Vertical span in grid rows (default ``1``).
        colspan: Horizontal span in grid columns (default ``1``).
    """

    name: str
    row: int = 0
    col: int = 0
    rowspan: int = 1
    colspan: int = 1


def _pick_column(
    tbl: history_compare.HistoryTable, cands: Sequence[str], role: str
) -> str:
    for col in cands:
        if col in tbl.columns:
            return col
    raise KeyError(
        f"none of {list(cands)} ({role}) found in history columns "
        f"{sorted(tbl.columns)}"
    )


def plot_history_track(
    history_path: pathlib.Path | str,
    x_col: str | None = None,
    y_col: str | None = None,
    *,
    preset: str | None = None,
    ref_history_path: pathlib.Path | str | None = None,
    reverse_xaxis: bool | None = None,
    reverse_yaxis: bool | None = None,
    show_degeneracy_line: bool | None = None,
    mesa_dir: pathlib.Path | str | None = None,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot a 2D evolutionary ``History_Track`` or named ``pgstar`` track.

    Implements ``do_Hist_Track`` (`star/private/pgstar_hist_track.f90:53-280`)
    and named ``pgstar`` track presets (`HR`, `TRho`, `TmaxRho`, `logg_Teff`,
    `logg_logT`, `logL_R`, `logL_v`, `L_Teff`, `L_R`, `L_v`, `R_Teff`, `R_L`).

    Args:
        history_path: Path to ``history.data`` or ``binary_history.data``.
        x_col: Horizontal column name when ``preset`` is ``None``.
        y_col: Vertical column name when ``preset`` is ``None``.
        preset: Case-insensitive ``pgstar`` preset name overriding ``x_col``
            and ``y_col``.
        ref_history_path: Optional MESA ``26.4.1`` reference history overlay.
        reverse_xaxis: Optional axis-inversion override.
        reverse_yaxis: Optional axis-inversion override.
        show_degeneracy_line: Draw ``psi4.data`` degeneracy guide line
            (`star/private/pgstar_tmaxrho.f90:90-105`).
        mesa_dir: Optional MESA ``26.4.1`` root for ``psi4.data``.
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.
    """
    tbl = history_compare.load_history_file(history_path)
    ref_tbl: history_compare.HistoryTable | None = None
    ref_hdr: dict[str, Any] | None = None
    if ref_history_path is not None:
        ref_tbl, ref_hdr = _plot_support.load_history_with_header(
            ref_history_path
        )

    if preset is not None:
        key = preset.strip().lower()
        if key not in TRACK_PRESETS:
            raise ValueError(
                f"unknown track preset {preset!r}; expected one of "
                f"{sorted(TRACK_PRESETS)}"
            )
        spec = TRACK_PRESETS[key]
        eff_x = _pick_column(tbl, spec.x_candidates, "x")
        eff_y = _pick_column(tbl, spec.y_candidates, "y")
        xlabel = (
            r"$\log_{10}(R / R_\odot)$"
            if key == "logl_r" and eff_x == "log_R"
            else spec.xlabel
        )
        ylabel = spec.ylabel
        rev_x = spec.reverse_x if reverse_xaxis is None else reverse_xaxis
        rev_y = spec.reverse_y if reverse_yaxis is None else reverse_yaxis
        do_psi = (
            spec.degeneracy_guide
            if show_degeneracy_line is None
            else show_degeneracy_line
        )
        title = preset
    else:
        if x_col is None or y_col is None:
            raise ValueError(
                "pass preset=... or both x_col and y_col to plot_history_track"
            )
        eff_x = _pick_column(tbl, (x_col,), "x")
        eff_y = _pick_column(tbl, (y_col,), "y")
        xlabel, ylabel = eff_x, eff_y
        rev_x, rev_y = bool(reverse_xaxis), bool(reverse_yaxis)
        do_psi = bool(show_degeneracy_line)
        title = f"{eff_y} vs. {eff_x}"

    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.84)
    xv, yv = tbl.columns[eff_x], tbl.columns[eff_y]
    valid = _compare_panels.finite_mask(xv) & _compare_panels.finite_mask(yv)

    if do_psi:
        psi_rho, psi_t = _plot_support.load_plot_info_curve(
            "psi4.data", mesa_dir=mesa_dir
        )
        ax.plot(
            psi_rho,
            psi_t,
            color=style.OKABE_ITO["grey"],
            ls="--",
            lw=0.9,
            alpha=0.72,
            label=r"$\epsilon_{\mathrm{F}}/k_{\mathrm{B}}T \approx 4$",
            zorder=1,
        )

    if (
        ref_tbl is not None
        and eff_x in ref_tbl.columns
        and eff_y in ref_tbl.columns
    ):
        ref_lbl = _plot_support.format_ref_label(
            ref_tbl.mesa_version_number
            or (str(ref_hdr.get("version_number", "")) if ref_hdr else "")
        )
        _compare_panels._plot_series(
            ax,
            ref_tbl.columns[eff_x],
            ref_tbl.columns[eff_y],
            label=ref_lbl,
            color=style.OKABE_ITO["black"],
            ls="--",
            lw=1.2,
            zorder=2,
        )
    _compare_panels._plot_series(
        ax,
        xv,
        yv,
        label="FlexMESA" if ref_tbl is not None else "Track",
        color=style.OKABE_ITO["blue"],
        lw=1.65,
        zorder=3,
    )
    if np.any(valid):
        ax.scatter(
            [xv[valid][-1]],
            [yv[valid][-1]],
            color=style.OKABE_ITO["vermilion"],
            s=24,
            zorder=4,
            label="Current",
        )
        if do_psi:
            x_lo, x_hi = (
                float(np.min(xv[valid])) - 0.6,
                float(np.max(xv[valid])) + 0.6,
            )
            y_lo, y_hi = (
                float(np.min(yv[valid])) - 0.35,
                float(np.max(yv[valid])) + 0.35,
            )
            ax.set_xlim(min(x_lo, x_hi - 0.5), max(x_hi, x_lo + 0.5))
            ax.set_ylim(min(y_lo, y_hi - 0.3), max(y_hi, y_lo + 0.3))

    if rev_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    if rev_y and not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _plot_support.set_history_panel_title(
        ax,
        title,
        tbl,
        ref_tbl,
        ref_path=ref_history_path,
        ref_hdr=ref_hdr,
        standalone=standalone,
    )
    ax.legend(loc="best", frameon=False, fontsize=7.0)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def plot_summary_history(
    history_path: pathlib.Path | str,
    *,
    ref_history_path: pathlib.Path | str | None = None,
    xaxis: str = "model_number",
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot normalized ``[0, 1]`` ``Summary_History`` (`pgstar`).

    Implements ``do_summary_history_plot`` from
    ``star/private/pgstar_summary_history.f90:54-265``.
    """
    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.80)
    tbl = history_compare.load_history_file(history_path)
    ref_tbl: history_compare.HistoryTable | None = None
    ref_hdr: dict[str, Any] | None = None
    if ref_history_path is not None:
        ref_tbl, ref_hdr = _plot_support.load_history_with_header(
            ref_history_path
        )
    x_run = tbl.columns.get(
        xaxis, np.arange(1, tbl.num_rows + 1, dtype=np.float64)
    )
    scale, x_lbl = _compare_panels.age_scale_and_label(
        xaxis, _compare_panels.max_finite_abs(x_run)
    )
    x_ref = (
        ref_tbl.columns.get(
            xaxis, np.arange(1, ref_tbl.num_rows + 1, dtype=np.float64)
        )
        * scale
        if ref_tbl is not None
        else None
    )

    colors = tuple(style.OKABE_ITO.values())
    for idx, (col, lbl, mode) in enumerate(_STAR_SUMMARY_HISTORY_COLS):
        if col not in tbl.columns:
            continue
        yr = tbl.columns[col]
        valid = _compare_panels.finite_mask(yr)
        if int(np.sum(valid)) < 2:
            continue
        vmin, vmax = float(np.min(yr[valid])), float(np.max(yr[valid]))
        if mode == "rel" and vmax <= vmin:
            continue
        yp = (
            np.where(valid, (yr - vmin) / (vmax - vmin), np.nan)
            if mode == "rel"
            else np.where(valid, yr, np.nan)
        )
        c_hex = colors[idx % len(colors)]
        _compare_panels._plot_series(
            ax, x_run * scale, yp, label=lbl, color=c_hex, lw=1.5
        )
        if ref_tbl is not None and x_ref is not None and col in ref_tbl.columns:
            rr = ref_tbl.columns[col]
            vr = _compare_panels.finite_mask(rr)
            rp = (
                np.where(vr, (rr - vmin) / (vmax - vmin), np.nan)
                if mode == "rel"
                else np.where(vr, rr, np.nan)
            )
            _compare_panels._plot_series(
                ax,
                x_ref,
                rp,
                label=f"{lbl} MESA",
                color=c_hex,
                ls="--",
                lw=1.0,
                alpha=0.72,
            )

    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Model Number" if xaxis == "model_number" else x_lbl)
    ax.set_ylabel("Mass Fraction / Scaled Value")
    _plot_support.set_history_panel_title(
        ax,
        "Summary History",
        tbl,
        ref_tbl,
        ref_path=ref_history_path,
        ref_hdr=ref_hdr,
        standalone=standalone,
    )
    ax.legend(loc="best", frameon=False, fontsize=6.6, ncol=2)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def plot_history_panels(
    history_path: pathlib.Path | str,
    panels: Sequence[str | tuple[str, str | None]],
    *,
    ref_history_path: pathlib.Path | str | None = None,
    xaxis: str = "model_number",
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, np.ndarray]:
    r"""Plot stacked ``History_Panels<N>`` from arbitrary history columns.

    Implements ``History_Panels`` from
    ``star/private/pgstar_history_panels.f90:31-240`` and
    ``binary/private/pgbinary_history_panels.f90:31-240``.
    """
    if isinstance(panels, (str, bytes)):
        raise TypeError(
            "panels must be a sequence of column names or pairs, not a bare str"
        )
    if not panels:
        raise ValueError("panels must contain at least one history column")
    tbl = history_compare.load_history_file(history_path)
    ref_tbl: history_compare.HistoryTable | None = None
    ref_hdr: dict[str, Any] | None = None
    if ref_history_path is not None:
        ref_tbl, ref_hdr = _plot_support.load_history_with_header(
            ref_history_path
        )
    if xaxis not in tbl.columns:
        raise KeyError(f"xaxis column {xaxis!r} not found in history")
    x_raw = tbl.columns[xaxis]
    scale, x_lbl = _compare_panels.age_scale_and_label(
        xaxis, _compare_panels.max_finite_abs(x_raw)
    )
    x_run = x_raw * scale
    x_ref = (
        ref_tbl.columns[xaxis] * scale
        if ref_tbl is not None and xaxis in ref_tbl.columns
        else None
    )

    style.use_mesa_plot_style()
    n_rows = len(panels)
    fig, raw_axes = plt.subplots(
        n_rows,
        1,
        sharex=True,
        figsize=style.figsize("single", 0.48 * max(n_rows, 2)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(raw_axes)
    palette = tuple(style.OKABE_ITO.values())

    for idx, item in enumerate(panels):
        y_col, y2_col = (item, None) if isinstance(item, str) else item
        if y_col not in tbl.columns:
            raise KeyError(f"history column {y_col!r} not found in history")
        ax_i = axes[idx]
        c_left = palette[idx % len(palette)]
        if (
            ref_tbl is not None
            and x_ref is not None
            and y_col in ref_tbl.columns
        ):
            _compare_panels._plot_series(
                ax_i,
                x_ref,
                ref_tbl.columns[y_col],
                label=f"{y_col} MESA",
                color=c_left,
                ls="--",
                lw=1.05,
                alpha=0.72,
            )
        _compare_panels._plot_series(
            ax_i, x_run, tbl.columns[y_col], label=y_col, color=c_left, lw=1.55
        )
        ax_i.set_ylabel(y_col)
        if y2_col is not None:
            if y2_col not in tbl.columns:
                raise KeyError(
                    f"secondary history column {y2_col!r} not found in history"
                )
            ax_r = ax_i.twinx()
            c_right = palette[(idx + 1) % len(palette)]
            if (
                ref_tbl is not None
                and x_ref is not None
                and y2_col in ref_tbl.columns
            ):
                _compare_panels._plot_series(
                    ax_r,
                    x_ref,
                    ref_tbl.columns[y2_col],
                    label=f"{y2_col} MESA",
                    color=c_right,
                    ls="--",
                    lw=1.05,
                    alpha=0.72,
                )
            _compare_panels._plot_series(
                ax_r,
                x_run,
                tbl.columns[y2_col],
                label=y2_col,
                color=c_right,
                lw=1.45,
            )
            ax_r.set_ylabel(y2_col)
            _compare_panels.merge_twinx_legend(
                ax_i, ax_r, loc="best", fontsize=6.6, ncol=2
            )
        else:
            ax_i.legend(loc="best", frameon=False, fontsize=6.8)

    axes[-1].set_xlabel("Model Number" if xaxis == "model_number" else x_lbl)
    if ref_tbl is not None:
        prov = _plot_support.format_history_provenance(
            tbl, ref_tbl, ref_path=ref_history_path, ref_hdr=ref_hdr
        )
        fig.suptitle(f"History Panels\n{prov}", fontsize=8.6)
    else:
        fig.suptitle("History Panels", fontsize=9.0)
    _compare_panels.save_figure(fig, save_path)
    return fig, axes


def parse_inlist_grid_spec(
    inlist_path: pathlib.Path | str,
    *,
    grid_index: int = 1,
    namelist_name: str | None = None,
) -> tuple[int, int, str, tuple[GridPanelSpec, ...]]:
    """Parse ``Grid<N>_*`` layout controls from an ``inlist`` file.

    Mirrors ``Grid_plot`` arguments in ``star/private/pgstar_grid.f90:50-79``
    and ``binary/private/pgbinary_grid.f90:32-56``.
    """
    r_cnt, c_cnt, title, raw_cells = _grid_dispatch.parse_grid_namelist(
        inlist_path, grid_index=grid_index, namelist_name=namelist_name
    )
    specs = tuple(GridPanelSpec(*cell) for cell in raw_cells)
    return r_cnt, c_cnt, title, specs


def plot_pgstar_grid(
    panels: Sequence[GridPanelSpec | str] | None = None,
    *,
    num_rows: int | None = None,
    num_cols: int | None = None,
    inlist_path: pathlib.Path | str | None = None,
    grid_index: int = 1,
    history_path: pathlib.Path | str | None = None,
    ref_history_path: pathlib.Path | str | None = None,
    profile_path: pathlib.Path | str | None = None,
    ref_profile_path: pathlib.Path | str | None = None,
    profile_paths: Sequence[pathlib.Path | str] | None = None,
    binary_history_path: pathlib.Path | str | None = None,
    ref_binary_history_path: pathlib.Path | str | None = None,
    star1_history_path: pathlib.Path | str | None = None,
    star2_history_path: pathlib.Path | str | None = None,
    star1_profile_path: pathlib.Path | str | None = None,
    star2_profile_path: pathlib.Path | str | None = None,
    mesa_dir: pathlib.Path | str | None = None,
    title: str = "",
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, list[mpl_axes.Axes]]:
    r"""Compose a multi-panel ``Grid<N>`` dashboard (`pgstar` / `pgbinary`).

    Implements ``Grid_plot`` from ``star/private/pgstar_grid.f90:50-260`` and
    ``binary/private/pgbinary_grid.f90:32-260`` via ``matplotlib.gridspec``.
    """
    inlist_title = ""
    resolved_specs: list[GridPanelSpec] = []
    if inlist_path is not None:
        r_cnt, c_cnt, inlist_title, parsed_specs = parse_inlist_grid_spec(
            inlist_path, grid_index=grid_index
        )
        eff_rows = r_cnt if num_rows is None else num_rows
        eff_cols = c_cnt if num_cols is None else num_cols
        resolved_specs = list(parsed_specs)
    else:
        if isinstance(panels, (str, bytes)):
            raise TypeError(
                "panels must be a sequence of panel names or GridPanelSpec "
                "objects, not a bare str or bytes"
            )
        if not panels:
            raise ValueError("provide either panels=... or inlist_path=...")
        if all(isinstance(p, str) for p in panels):
            n_tot = len(panels)
            eff_cols = (
                num_cols if num_cols is not None else (1 if n_tot == 1 else 2)
            )
            eff_rows = (
                num_rows
                if num_rows is not None
                else math.ceil(n_tot / eff_cols)
            )
            for idx, p_item in enumerate(panels):
                resolved_specs.append(
                    GridPanelSpec(
                        name=str(p_item),
                        row=idx // eff_cols,
                        col=idx % eff_cols,
                    )
                )
        else:
            for p_item in panels:
                if isinstance(p_item, GridPanelSpec):
                    resolved_specs.append(p_item)
                else:
                    raise TypeError(
                        "mixed str and GridPanelSpec sequences unsupported"
                    )
            eff_rows = (
                num_rows
                if num_rows is not None
                else max(s.row + s.rowspan for s in resolved_specs)
            )
            eff_cols = (
                num_cols
                if num_cols is not None
                else max(s.col + s.colspan for s in resolved_specs)
            )

    style.use_mesa_plot_style()
    aspect = max(0.50, 0.56 * eff_rows / max(eff_cols, 1) * 1.4)
    fig = plt.figure(
        figsize=style.figsize("double", aspect), constrained_layout=True
    )
    gs = mpl_gridspec.GridSpec(eff_rows, eff_cols, figure=fig)

    created_axes: list[mpl_axes.Axes] = []
    for spec in resolved_specs:
        r_end, c_end = spec.row + spec.rowspan, spec.col + spec.colspan
        if spec.row < 0 or spec.col < 0 or r_end > eff_rows or c_end > eff_cols:
            plt.close(fig)
            raise ValueError(
                f"GridPanelSpec {spec!r} exceeds grid ({eff_rows}, {eff_cols})"
            )
        cell_ax = fig.add_subplot(gs[spec.row : r_end, spec.col : c_end])
        try:
            _grid_dispatch.dispatch_grid_cell(
                cell_ax,
                spec.name,
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
                mesa_dir=mesa_dir,
            )
        except Exception:
            plt.close(fig)
            raise
        created_axes.append(cell_ax)

    eff_title = title or inlist_title
    if eff_title:
        fig.suptitle(eff_title, fontsize=9.8)
    _compare_panels.save_figure(fig, save_path)
    return fig, created_axes
