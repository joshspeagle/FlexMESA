# SPDX-License-Identifier: GPL-3.0-or-later
r"""Upstream ``pgstar`` summary and custom radial profile panels (`#173`).

Ports ``Summary_Profile`` (`star/private/pgstar_summary_profile.f90:72-307`),
``Summary_Burn`` (`star/private/pgstar_summary_burn.f90:54-299`), and
customizable ``Profile_Panels<N>``
(`star/private/pgstar_profile_panels.f90:31-220`) strictly from saved
``profileN.data`` files (`#173`).
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np

from flexmesa.plotting import (
    _compare_panels,
    _plot_support,
    compare,
    pgstar_panels,
    style,
)

# Default Summary_Profile lines from star/defaults/pgstar.defaults:985-1042.
_SUMMARY_PROFILE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("x_mass_fraction_H", "X", "raw"),
    ("y_mass_fraction_He", "Y", "raw"),
    ("log_j_rot", "log j rel", "rel"),
    ("log_omega", "log omega rel", "rel"),
    ("mass", "m rel", "rel"),
    ("radius", "r rel", "rel"),
    ("eps_nuc", "eps nuc rel", "rel"),
    ("entropy", "entropy rel", "rel"),
    ("opacity", "opacity rel", "rel"),
    ("luminosity", "L rel", "rel"),
    ("temperature", "T rel", "rel"),
    ("logT", "log T rel", "rel"),
)


def plot_summary_profile(
    profile_path: pathlib.Path | str,
    *,
    ref_profile_path: pathlib.Path | str | None = None,
    xaxis: str = "mass",
    columns: Sequence[tuple[str, str, bool]] | None = None,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot normalized ``[0, 1]`` ``Summary_Profile`` (`pgstar`).

    Implements ``do_summary_profile_panel`` from
    ``star/private/pgstar_summary_profile.f90:72-307``
    (`star/defaults/pgstar.defaults:985-1042`).

    Args:
        profile_path: Path to ``profileN.data``.
        ref_profile_path: Optional MESA ``26.4.1`` ``profileN.data`` overlay.
        xaxis: Radial coordinate name (default ``"mass"``).
        columns: Optional ``(col_name, legend_label, scale_minmax)`` triples
            overriding upstream ``Summary_Profile`` defaults.
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.
    """
    prof = compare.load_profile_file(profile_path)
    ref_prof = (
        compare.load_profile_file(ref_profile_path)
        if ref_profile_path is not None
        else None
    )
    x_run, x_label, rev_x = _plot_support.profile_xaxis(prof, xaxis)
    x_ref = (
        _plot_support.profile_xaxis(ref_prof, xaxis)[0]
        if ref_prof is not None
        else None
    )

    if columns is not None:
        specs = list(columns)
    else:
        has_temp = "temperature" in prof.columns
        specs = [
            (c, lbl, mode == "rel")
            for c, lbl, mode in _SUMMARY_PROFILE_SPECS
            if not (c == "logT" and has_temp)
        ]

    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.82)
    colors = pgstar_panels.SERIES_COLORS
    cnt = 0
    for col, lbl, do_scale in specs:
        if col == xaxis or col not in prof.columns:
            continue
        y_raw = prof.columns[col]
        valid = _compare_panels.finite_mask(y_raw)
        if int(np.sum(valid)) < 2:
            continue
        vmin, vmax = float(np.min(y_raw[valid])), float(np.max(y_raw[valid]))
        if do_scale and vmax <= vmin:
            continue

        def _scale_fn(
            arr: np.ndarray,
            lo: float = vmin,
            hi: float = vmax,
            scaled: bool = do_scale,
        ) -> np.ndarray:
            v_m = _compare_panels.finite_mask(arr)
            return (
                np.where(v_m, (arr - lo) / (hi - lo), np.nan)
                if scaled
                else np.where(v_m, arr, np.nan)
            )

        c_hex = colors[cnt % len(colors)]
        ls_run = "-." if cnt >= len(colors) else "-"
        _plot_support.plot_profile_pair(
            ax,
            prof,
            x_run,
            ref_prof,
            x_ref,
            col,
            label=lbl,
            color=c_hex,
            ls=ls_run,
            lw=1.45,
            transform=_scale_fn,
        )
        cnt += 1

    ax.set_ylim(-0.04, 1.05)
    if rev_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    _plot_support.draw_xaxis_mixing_bar(ax, x_run, prof)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mass Fraction / Normalized Profile")
    _plot_support.set_profile_panel_title(
        ax, "Summary Profile", prof, ref_prof, standalone=standalone
    )
    ax.legend(loc="best", frameon=False, fontsize=6.4, ncol=2)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def plot_summary_burn_profile(
    profile_path: pathlib.Path | str,
    *,
    ref_profile_path: pathlib.Path | str | None = None,
    xaxis: str = "mass",
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot radial ``Summary_Burn`` panel (`pgstar_summary_burn.f90`).

    Implements ``do_summary_burn_plot`` from
    ``star/private/pgstar_summary_burn.f90:54-299`` (`logRho`,
    $\log|\epsilon_{\mathrm{nuc}}|$, neutrino losses, and `logT`).

    Args:
        profile_path: Path to ``profileN.data``.
        ref_profile_path: Optional MESA ``26.4.1`` ``profileN.data`` overlay.
        xaxis: Radial coordinate name (default ``"mass"``).
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)`` (primary left ``Axes``).
    """
    prof = compare.load_profile_file(profile_path)
    ref_prof = (
        compare.load_profile_file(ref_profile_path)
        if ref_profile_path is not None
        else None
    )
    x_run, x_label, rev_x = _plot_support.profile_xaxis(prof, xaxis)
    x_ref = (
        _plot_support.profile_xaxis(ref_prof, xaxis)[0]
        if ref_prof is not None
        else None
    )

    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.84)
    _plot_support.plot_profile_pair(
        ax,
        prof,
        x_run,
        ref_prof,
        x_ref,
        "logRho",
        label=r"$\log\rho$",
        color=style.OKABE_ITO["reddish_purple"],
    )
    _plot_support.plot_profile_pair(
        ax,
        prof,
        x_run,
        ref_prof,
        x_ref,
        "eps_nuc",
        label=r"$\log|\epsilon_{\mathrm{nuc}}|$",
        color=style.OKABE_ITO["blue"],
        lw=1.6,
        transform=lambda a: pgstar_panels.safe_log10_pos(np.abs(a)),
    )
    if (
        ref_prof is not None
        and x_ref is not None
        and (
            "non_nuc_neu" in ref_prof.columns
            or "eps_nuc_neu_total" in ref_prof.columns
        )
    ):
        ref_neu = ref_prof.columns.get(
            "non_nuc_neu", np.zeros(ref_prof.num_zones, dtype=np.float64)
        ) + ref_prof.columns.get(
            "eps_nuc_neu_total", np.zeros(ref_prof.num_zones, dtype=np.float64)
        )
        _compare_panels._plot_series(
            ax,
            x_ref,
            pgstar_panels.safe_log10_pos(ref_neu),
            label=r"$\log\epsilon_\nu$ MESA",
            color=style.OKABE_ITO["bluish_green"],
            ls="--",
            lw=1.05,
            alpha=0.72,
            zorder=2,
        )
    if "non_nuc_neu" in prof.columns or "eps_nuc_neu_total" in prof.columns:
        neu_tot = prof.columns.get(
            "non_nuc_neu", np.zeros(prof.num_zones, dtype=np.float64)
        ) + prof.columns.get(
            "eps_nuc_neu_total", np.zeros(prof.num_zones, dtype=np.float64)
        )
        _compare_panels._plot_series(
            ax,
            x_run,
            pgstar_panels.safe_log10_pos(neu_tot),
            label=r"$\log\epsilon_\nu$",
            color=style.OKABE_ITO["bluish_green"],
            lw=1.45,
            zorder=3,
        )

    ax_t = ax.twinx()
    _plot_support.plot_profile_pair(
        ax_t,
        prof,
        x_run,
        ref_prof,
        x_ref,
        "logT",
        label=r"$\log T$",
        color=style.OKABE_ITO["orange"],
        lw=1.55,
    )
    ax_t.set_ylabel(r"$\log_{10}(T / \mathrm{K})$")

    # Annotate top 3 category peaks (pgstar_summary_burn.f90:218-285).
    cat_peaks: list[tuple[float, int, str]] = []
    for cat in pgstar_panels.REACTION_CATEGORIES:
        if cat == "photo" or cat not in prof.columns:
            continue
        arr = prof.columns[cat]
        valid = _compare_panels.finite_mask(arr) & (arr > 0.1)
        if np.any(valid):
            k_max = int(np.argmax(np.where(valid, arr, -1.0e99)))
            cat_peaks.append((math.log10(float(arr[k_max])), k_max, cat))
    cat_peaks.sort(key=lambda item: item[0], reverse=True)
    for rank_i, (lg_pk, k_idx, cat_name) in enumerate(cat_peaks[:3]):
        short = cat_name.removeprefix("burn_").replace("tri_alpha", "3a")
        ax.annotate(
            short,
            xy=(float(x_run[k_idx]), lg_pk),
            xytext=(0.0, 4.0 + 6.0 * rank_i),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            color=style.OKABE_ITO["black"],
        )

    if rev_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    _plot_support.draw_xaxis_mixing_bar(ax, x_run, prof)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\log_{10}\rho$ / $\log_{10}|\epsilon|$")
    _plot_support.set_profile_panel_title(
        ax, "Summary Burn", prof, ref_prof, standalone=standalone
    )
    _compare_panels.merge_twinx_legend(
        ax, ax_t, loc="best", fontsize=6.5, ncol=2
    )
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def plot_profile_custom_panels(
    profile_path: pathlib.Path | str,
    panels: Sequence[str | tuple[str, str | None]],
    *,
    ref_profile_path: pathlib.Path | str | None = None,
    xaxis: str = "mass",
    axes: Sequence[mpl_axes.Axes] | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, np.ndarray]:
    r"""Plot stacked ``Profile_Panels<N>`` from arbitrary profile columns.

    Implements ``Profile_Panels`` from
    ``star/private/pgstar_profile_panels.f90:31-220``.

    Args:
        profile_path: Path to ``profileN.data``.
        panels: Non-empty sequence of column names ``y_col`` or pairs
            ``(y_col, other_y_col)`` plotted on shared ``xaxis``.
        ref_profile_path: Optional MESA ``26.4.1`` ``profileN.data`` overlay.
        xaxis: Shared horizontal profile coordinate (default ``"mass"``).
        axes: Optional pre-created stack of ``len(panels)`` ``Axes`` (for
            embedding inside ``Grid<N>`` or ``Star<N>`` cells).
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, axes)`` where ``axes`` is a 1D array of length
        ``len(panels)``.
    """
    if isinstance(panels, (str, bytes)):
        raise TypeError(
            "panels must be a sequence of column names or pairs, not a bare str"
        )
    if not panels:
        raise ValueError("panels must contain at least one column spec")
    prof = compare.load_profile_file(profile_path)
    ref_prof = (
        compare.load_profile_file(ref_profile_path)
        if ref_profile_path is not None
        else None
    )
    x_run, x_label, rev_x = _plot_support.profile_xaxis(prof, xaxis)
    x_ref = (
        _plot_support.profile_xaxis(ref_prof, xaxis)[0]
        if ref_prof is not None
        else None
    )

    n_rows = len(panels)
    for item in panels:
        y_col, y2_col = (item, None) if isinstance(item, str) else item
        if y_col not in prof.columns:
            raise KeyError(
                f"profile column {y_col!r} not found in profileN.data"
            )
        if y2_col is not None and y2_col not in prof.columns:
            raise KeyError(
                f"secondary profile column {y2_col!r} missing in profile"
            )

    standalone = axes is None
    if axes is not None:
        if len(axes) != n_rows:
            raise ValueError(f"expected {n_rows} target axes, got {len(axes)}")
        axes_arr = np.asarray(list(axes), dtype=object)
        fig_cand = axes[0].get_figure()
        if not isinstance(fig_cand, mpl_figure.Figure):
            raise TypeError("target axes must belong to a Figure")
        fig = fig_cand
    else:
        style.use_mesa_plot_style()
        fig, raw_axes = plt.subplots(
            n_rows,
            1,
            sharex=True,
            figsize=style.figsize("single", 0.48 * max(n_rows, 2)),
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(raw_axes)
    colors = pgstar_panels.SERIES_COLORS

    for idx, item in enumerate(panels):
        y_col, y2_col = (item, None) if isinstance(item, str) else item
        ax_i = axes_arr[idx]
        _plot_support.plot_profile_pair(
            ax_i,
            prof,
            x_run,
            ref_prof,
            x_ref,
            y_col,
            label=y_col,
            color=colors[idx % len(colors)],
            lw=1.55,
        )
        ax_i.set_ylabel(y_col)
        if y2_col is not None:
            ax_r = ax_i.twinx()
            _plot_support.plot_profile_pair(
                ax_r,
                prof,
                x_run,
                ref_prof,
                x_ref,
                y2_col,
                label=y2_col,
                color=colors[(idx + 1) % len(colors)],
                lw=1.45,
            )
            ax_r.set_ylabel(y2_col)
            _compare_panels.merge_twinx_legend(
                ax_i, ax_r, loc="best", fontsize=6.6, ncol=2
            )
        else:
            ax_i.legend(loc="best", frameon=False, fontsize=6.8)
        _plot_support.draw_xaxis_mixing_bar(ax_i, x_run, prof)
        if idx < n_rows - 1:
            ax_i.tick_params(labelbottom=False)
        if rev_x and not ax_i.xaxis_inverted():
            ax_i.invert_xaxis()

    axes_arr[-1].set_xlabel(x_label)
    if standalone:
        p_sub = _plot_support.format_profile_subtitle(prof, ref_prof)
        fig.suptitle(f"Profile Panels ({p_sub})", fontsize=8.8)
    else:
        _plot_support.set_profile_panel_title(
            axes_arr[0], "Profile Panels", prof, ref_prof, standalone=False
        )
    _compare_panels.save_figure(fig, save_path)
    return fig, axes_arr
