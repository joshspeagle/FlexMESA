# SPDX-License-Identifier: LGPL-3.0-or-later
"""Internal subplot builders for MESA 26.4.1 history and profile comparisons."""

from __future__ import annotations

import math
import pathlib
import typing

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import numpy as np

from flexmesa.plotting import style
from flexmesa.testing import case_manifest, history_compare

_RESID_COLORS: tuple[str, ...] = (
    style.OKABE_ITO["blue"],
    style.OKABE_ITO["vermilion"],
    style.OKABE_ITO["bluish_green"],
    style.OKABE_ITO["orange"],
    style.OKABE_ITO["reddish_purple"],
)


def finite_mask(arr: np.ndarray) -> np.ndarray:
    """Return boolean mask filtering non-finite and MESA sentinel values."""
    thresh = history_compare.SENTINEL_ABS_THRESHOLD
    return np.isfinite(arr) & (np.abs(arr) < thresh)


def age_scale_and_label(x_col: str, max_abs_x: float) -> tuple[float, str]:
    """Choose yr or Myr scaling and axis label for an independent coordinate."""
    if x_col in {"star_age", "age"}:
        if max_abs_x >= 1.0e5:
            return 1.0e-6, "Age [$\\mathrm{Myr}$]"
        return 1.0, "Age [$\\mathrm{yr}$]"
    return 1.0, x_col


def max_finite_abs(arr: np.ndarray) -> float:
    """Return maximum absolute finite value in ``arr`` or ``1.0`` if none."""
    mask = finite_mask(arr)
    return float(np.max(np.abs(arr[mask]))) if np.any(mask) else 1.0


def scaled_coord_pair(
    ref_tbl: history_compare.HistoryTable,
    run_tbl: history_compare.HistoryTable,
    spec: case_manifest.CaseManifestEntry,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return scaled independent coordinate arrays and shared axis label."""
    x_col = (
        spec.independent_var
        if spec.independent_var in ref_tbl.columns
        and spec.independent_var in run_tbl.columns
        else "model_number"
    )
    ref_raw = ref_tbl.columns.get(
        x_col, np.arange(1, ref_tbl.num_rows + 1, dtype=np.float64)
    )
    run_raw = run_tbl.columns.get(
        x_col, np.arange(1, run_tbl.num_rows + 1, dtype=np.float64)
    )
    scale, x_label = age_scale_and_label(
        x_col, max(max_finite_abs(ref_raw), max_finite_abs(run_raw))
    )
    return ref_raw * scale, run_raw * scale, x_label


def merge_twinx_legend(
    ax_left: mpl_axes.Axes,
    ax_right: mpl_axes.Axes,
    *,
    loc: typing.Literal["best", "upper left"] = "best",
    fontsize: float = 7.0,
    ncol: int = 2,
) -> None:
    """Merge handles and labels from paired twin axes into one legend."""
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    if h1 or h2:
        ax_left.legend(
            h1 + h2,
            l1 + l2,
            loc=loc,
            frameon=False,
            fontsize=fontsize,
            ncol=ncol,
        )


def save_figure(
    fig: mpl_figure.Figure, save_path: pathlib.Path | str | None
) -> None:
    """Save ``fig`` to ``save_path`` at ``300 dpi`` when requested."""
    if save_path is None:
        return
    out_p = pathlib.Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=300, bbox_inches="tight")


def _plot_series(
    ax: mpl_axes.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    label: str | None,
    color: str,
    ls: str = "-",
    lw: float = 1.5,
    alpha: float = 1.0,
    zorder: int = 3,
    x_mask: np.ndarray | None = None,
) -> bool:
    """Plot a finite trajectory line or single-point scatter marker."""
    valid_x = finite_mask(x_vals) if x_mask is None else x_mask
    mask = valid_x & finite_mask(y_vals)
    if not np.any(mask):
        return False
    xv = x_vals[mask]
    yv = y_vals[mask]
    if xv.size == 1:
        ax.scatter(
            xv,
            yv,
            color=color,
            s=26,
            marker="o" if ls == "-" else "s",
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
    else:
        ax.plot(
            xv,
            yv,
            color=color,
            ls=ls,
            lw=lw,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
    return True


def _plot_paired_series(
    ax: mpl_axes.Axes,
    ref_x: np.ndarray,
    run_x: np.ndarray,
    ref_tbl: history_compare.HistoryTable,
    run_tbl: history_compare.HistoryTable,
    series_specs: tuple[tuple[str, str, str], ...],
    *,
    ref_suffix: str = "MESA",
    run_suffix: str = "Flex",
    ref_alpha: float = 0.75,
) -> bool:
    """Overplot reference/candidate column groups using shared styling."""
    drawn = False
    ref_xmask = finite_mask(ref_x)
    run_xmask = finite_mask(run_x)
    for col, tag, c_hex in series_specs:
        if col in ref_tbl.columns:
            drawn |= _plot_series(
                ax,
                ref_x,
                ref_tbl.columns[col],
                label=f"{tag} {ref_suffix}",
                color=c_hex,
                ls="--",
                lw=1.1,
                alpha=ref_alpha,
                zorder=2,
                x_mask=ref_xmask,
            )
        if col in run_tbl.columns:
            drawn |= _plot_series(
                ax,
                run_x,
                run_tbl.columns[col],
                label=f"{tag} {run_suffix}",
                color=c_hex,
                lw=1.6,
                zorder=4,
                x_mask=run_xmask,
            )
    return drawn


def _plot_table_pair(
    ax: mpl_axes.Axes,
    ref_tbl: history_compare.HistoryTable,
    run_tbl: history_compare.HistoryTable,
    x_col: str,
    y_col: str,
    *,
    run_color: str,
    ref_color: str = style.OKABE_ITO["black"],
    ref_x: np.ndarray | None = None,
    run_x: np.ndarray | None = None,
) -> bool:
    """Plot a single variable from ``ref_tbl`` and ``run_tbl`` cleanly."""
    drawn = False
    rx = ref_x if ref_x is not None else ref_tbl.columns.get(x_col)
    fx = run_x if run_x is not None else run_tbl.columns.get(x_col)
    if rx is not None and y_col in ref_tbl.columns:
        drawn |= _plot_series(
            ax,
            rx,
            ref_tbl.columns[y_col],
            label="MESA 26.4.1",
            color=ref_color,
            ls="--",
            lw=1.3,
            zorder=2,
        )
    if fx is not None and y_col in run_tbl.columns:
        drawn |= _plot_series(
            ax,
            fx,
            run_tbl.columns[y_col],
            label="FlexMESA",
            color=run_color,
            lw=1.7,
            zorder=4,
        )
    return drawn


def _plot_residual_panel(
    ax: mpl_axes.Axes,
    ref_tbl: history_compare.HistoryTable,
    run_tbl: history_compare.HistoryTable,
    spec: case_manifest.CaseManifestEntry,
    comp: history_compare.HistoryComparison | None,
) -> None:
    """Plot tolerance-normalized track residuals and final tolerance status."""
    if comp is None or not spec.track_tolerances:
        ax.text(
            0.5,
            0.5,
            (
                "No comparison: histories could not be aligned"
                if comp is None
                else "No track tolerances configured"
            ),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8.5,
        )
        ax.set_title("Residuals vs. DECISIONS #20 Tolerance")
        return

    x_col = spec.independent_var
    if not (comp.overlap_max > comp.overlap_min and ref_tbl.num_rows >= 2):
        lines = ["No shared coordinate overlap for interpolated residuals."]
        for col, diff_v in comp.final_diffs.items():
            if col in spec.final_tolerances:
                tol = spec.final_tolerances[col]
                ratio = diff_v / tol if tol > 0 else float("nan")
                lines.append(f"{col}: {diff_v:+.2e} ({ratio:+.1f}x tol)")
        ax.text(
            0.05,
            0.90,
            "\n".join(lines[:5]),
            va="top",
            ha="left",
            transform=ax.transAxes,
            fontsize=7.5,
        )
        ax.set_title("Residuals (Normalized to Tol)")
        return

    if run_tbl.num_rows < 2:
        ax.text(
            0.5,
            0.5,
            f"Run stopped at {run_tbl.num_rows} row (insufficient track)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8.5,
        )
        ax.set_title("Residuals (Normalized to Tol)")
        return

    lo, hi = comp.overlap_min, comp.overlap_max
    grid = np.linspace(lo, hi, 300, dtype=np.float64)
    scale, xlabel = age_scale_and_label(x_col, max(abs(lo), abs(hi)))
    ref_x = ref_tbl.columns[x_col]
    run_x = run_tbl.columns[x_col]
    ref_order = np.argsort(ref_x, kind="mergesort")
    run_order = np.argsort(run_x, kind="mergesort")
    rx_s = ref_x[ref_order]
    fx_s = run_x[run_order]
    rx_xmask = finite_mask(rx_s)
    fx_xmask = finite_mask(fx_s)
    rx_coord = rx_s[rx_xmask]
    fx_coord = fx_s[fx_xmask]

    ax.axhspan(
        -1.0,
        1.0,
        color=style.OKABE_ITO["bluish_green"],
        alpha=0.14,
        zorder=0,
        label="Tol Band ($\\pm 1$)",
    )
    ax.axhline(0.0, color=style.OKABE_ITO["black"], ls="-", lw=0.7, zorder=1)
    for bnd in (1.0, -1.0):
        ax.axhline(
            bnd, color=style.OKABE_ITO["bluish_green"], ls=":", lw=0.9, zorder=1
        )

    max_ratio = 1.0
    plotted = False
    for idx, (col, tol) in enumerate(spec.track_tolerances.items()):
        if col not in ref_tbl.columns or col not in run_tbl.columns or tol <= 0:
            continue
        r_col = ref_tbl.columns[col][ref_order]
        f_col = run_tbl.columns[col][run_order]
        r_mask = rx_xmask & finite_mask(r_col)
        f_mask = fx_xmask & finite_mask(f_col)
        if np.sum(r_mask) < 2 or np.sum(f_mask) < 2:
            continue
        ref_i = np.interp(
            grid, rx_s[r_mask], r_col[r_mask], left=np.nan, right=np.nan
        )
        run_i = np.interp(
            grid, fx_s[f_mask], f_col[f_mask], left=np.nan, right=np.nan
        )
        r_valid = np.interp(
            grid,
            rx_coord,
            r_mask[rx_xmask].astype(np.float64),
            left=0.0,
            right=0.0,
        )
        f_valid = np.interp(
            grid,
            fx_coord,
            f_mask[fx_xmask].astype(np.float64),
            left=0.0,
            right=0.0,
        )
        valid = (r_valid >= 1.0) & (f_valid >= 1.0)
        norm_res = np.where(valid, (run_i - ref_i) / tol, np.nan)
        if not np.any(np.isfinite(norm_res)):
            continue
        peak = float(np.nanmax(np.abs(norm_res)))
        if math.isfinite(peak):
            max_ratio = max(max_ratio, peak)
        ax.plot(
            grid * scale,
            norm_res,
            color=_RESID_COLORS[idx % len(_RESID_COLORS)],
            lw=1.35,
            label=f"{col} ({tol:.0e})",
            zorder=3,
        )
        plotted = True

    if comp.first_diverging_x is not None and math.isfinite(
        comp.first_diverging_x
    ):
        step_txt = (
            f"1st div: step {comp.first_diverging_step}"
            if comp.first_diverging_step is not None
            else "1st div"
        )
        ax.axvline(
            comp.first_diverging_x * scale,
            color=style.OKABE_ITO["vermilion"],
            ls="--",
            lw=1.1,
            label=step_txt,
            zorder=4,
        )

    final_notes: list[str] = []
    for col, tol in spec.final_tolerances.items():
        if col in spec.track_tolerances:
            continue
        final_diff = comp.final_diffs.get(col)
        if final_diff is not None and math.isfinite(final_diff):
            final_notes.append(f"{col}={final_diff:+.2e} (tol {tol:.0e})")
    if plotted and max_ratio > 8.0:
        ax.set_yscale("symlog", linthresh=1.5)
        ax.set_ylim(top=max(12.0, max_ratio * 2.5))
    elif plotted:
        lo_lim = max(1.5, min(8.5, 1.25 * max_ratio))
        hi_lim = max(2.2, min(10.5, 1.55 * max_ratio))
        ax.set_ylim(-lo_lim, hi_lim)

    title = "Normalized Track Residuals"
    if final_notes:
        title += "\n" + "; ".join(final_notes[:2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("($\\mathrm{Flex} - \\mathrm{MESA}$) / Tol")
    ax.set_title(title, fontsize=9.0 if not final_notes else 8.2)
    if plotted:
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=6.5,
            ncol=2,
            borderaxespad=0.3,
        )


def format_provenance_title(
    spec: case_manifest.CaseManifestEntry,
    ref_tbl: history_compare.HistoryTable,
    run_tbl: history_compare.HistoryTable,
    comp: history_compare.HistoryComparison | None,
    *,
    termination_code: str,
    numerics_mode: str,
    controls_ignored: bool,
    max_steps: int | None,
    n_controls_ignored: int = 0,
) -> str:
    """Build multi-line provenance header identifying MESA ref and run.

    A run that ignored unsupported controls is labelled ``diagnostic`` and a
    step-capped run ``smoke``; neither is described as a pass even when its
    history lies inside the tolerances (``DECISIONS #26``). When no
    comparison was made the header says so instead of reporting a tolerance
    outcome.
    """
    ref_ver = (
        f"MESA {ref_tbl.mesa_version_number}"
        if ref_tbl.mesa_version_number
        else f"MESA {spec.reference_status}"
    )
    run_parts = [f"{run_tbl.num_rows} rows"]
    if numerics_mode:
        run_parts.append(f"mode={numerics_mode}")
    if controls_ignored:
        run_parts.append(
            f"diagnostic ({n_controls_ignored} controls ignored)"
            if n_controls_ignored > 0
            else "diagnostic (controls ignored)"
        )
    if max_steps is not None:
        run_parts.append(f"smoke: cap={max_steps} steps")
    if termination_code:
        run_parts.append(f"term={termination_code}")

    div_str = (
        f"1st div step={comp.first_diverging_step}"
        if comp is not None and comp.first_diverging_step is not None
        else "no div step"
    )
    if comp is None:
        tol_str = "not compared"
    elif comp.within_final_tolerances and comp.within_track_tolerances:
        tol_str = "within tol"
        if controls_ignored:
            tol_str += " (diagnostic, not a pass)"
        elif max_steps is not None:
            tol_str += " (smoke, not a pass)"
    else:
        tol_str = "outside tol"
    line1 = f"{spec.suite}/{spec.case} ({spec.reference_path or 'local ref'})"
    line2 = (
        f"Ref: {ref_ver} ({ref_tbl.num_rows} rows) | "
        f"FlexMESA: {', '.join(run_parts)} | {div_str}, {tol_str}"
    )
    return f"{line1}\n{line2}"


def _plot_star_comparison_panels(
    axes: np.ndarray,
    ref_tbl: history_compare.HistoryTable,
    run_tbl: history_compare.HistoryTable,
    spec: case_manifest.CaseManifestEntry,
    comp: history_compare.HistoryComparison | None,
) -> None:
    """Populate the 3x2 grid for ``star`` and ``astero`` comparisons."""
    col_ref = style.OKABE_ITO["black"]
    col_run = style.OKABE_ITO["blue"]

    # (0, 0) HR Diagram
    ax = axes[0, 0]
    _plot_table_pair(
        ax, ref_tbl, run_tbl, "log_Teff", "log_L", run_color=col_run
    )
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel("$\\log_{10}(T_{\\mathrm{eff}} / \\mathrm{K})$")
    ax.set_ylabel("$\\log_{10}(L / L_\\odot)$")
    ax.set_title("HR Diagram")
    ax.legend(loc="best", frameon=False, fontsize=7.5)

    # (0, 1) Central density vs central temperature
    ax = axes[0, 1]
    _plot_table_pair(
        ax,
        ref_tbl,
        run_tbl,
        "log_center_Rho",
        "log_center_T",
        run_color=col_run,
    )
    ax.set_xlabel("$\\log_{10}(\\rho_c / \\mathrm{g\\,cm^{-3}})$")
    ax.set_ylabel("$\\log_{10}(T_c / \\mathrm{K})$")
    ax.set_title("Core Trajectory")
    ax.legend(loc="best", frameon=False, fontsize=7.5)

    ref_x, run_x, x_label = scaled_coord_pair(ref_tbl, run_tbl, spec)

    # (1, 0) Nuclear luminosities
    ax = axes[1, 0]
    lum_specs = (
        ("log_L", "$L$", style.OKABE_ITO["blue"]),
        ("log_LH", "$L_{\\mathrm{H}}$", style.OKABE_ITO["vermilion"]),
        ("log_LHe", "$L_{\\mathrm{He}}$", style.OKABE_ITO["bluish_green"]),
    )
    if _plot_paired_series(ax, ref_x, run_x, ref_tbl, run_tbl, lum_specs):
        ax.legend(loc="best", frameon=False, fontsize=6.8, ncol=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("$\\log_{10}(L_i / L_\\odot)$")
    ax.set_title("Luminosities")

    # (1, 1) Core & convective masses
    ax = axes[1, 1]
    mass_specs = (
        ("mass_conv_core", "$M_{\\mathrm{cc}}$", style.OKABE_ITO["blue"]),
        ("he_core_mass", "$M_{\\mathrm{He}}$", style.OKABE_ITO["vermilion"]),
        ("co_core_mass", "$M_{\\mathrm{CO}}$", style.OKABE_ITO["bluish_green"]),
    )
    if _plot_paired_series(ax, ref_x, run_x, ref_tbl, run_tbl, mass_specs):
        ax.legend(loc="best", frameon=False, fontsize=6.8, ncol=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mass [$M_\\odot$]")
    ax.set_title("Core & Convective Masses")

    # (2, 0) Timestep, zone count & retries vs model_number
    ax = axes[2, 0]
    m_ref = ref_tbl.columns.get(
        "model_number", np.arange(1, ref_tbl.num_rows + 1, dtype=np.float64)
    )
    m_run = run_tbl.columns.get(
        "model_number", np.arange(1, run_tbl.num_rows + 1, dtype=np.float64)
    )
    for dt_col in ("log_dt", "lg_dt_yr"):
        has_left = False
        if dt_col in ref_tbl.columns:
            has_left |= _plot_series(
                ax,
                m_ref,
                ref_tbl.columns[dt_col],
                label="$\\log\\Delta t$ MESA",
                color=col_ref,
                ls="--",
                lw=1.1,
                zorder=2,
            )
        if dt_col in run_tbl.columns:
            has_left |= _plot_series(
                ax,
                m_run,
                run_tbl.columns[dt_col],
                label="$\\log\\Delta t$ Flex",
                color=col_run,
                lw=1.5,
                zorder=4,
            )
        if has_left:
            break
    ax.set_xlabel("Model Number")
    ax.set_ylabel("$\\log_{10}(\\Delta t / \\mathrm{yr})$")
    ax.set_title("Timestep, Zones & Retries")

    ax_r = ax.twinx()
    if "num_zones" in ref_tbl.columns:
        _plot_series(
            ax_r,
            m_ref,
            ref_tbl.columns["num_zones"],
            label="$N_z$ MESA",
            color=style.OKABE_ITO["grey"],
            ls=":",
            lw=1.1,
            zorder=1,
        )
    if "num_zones" in run_tbl.columns:
        _plot_series(
            ax_r,
            m_run,
            run_tbl.columns["num_zones"],
            label="$N_z$ Flex",
            color=style.OKABE_ITO["bluish_green"],
            lw=1.2,
            zorder=3,
        )
    if "num_retries" in run_tbl.columns:
        n_ret = run_tbl.columns["num_retries"]
        y_ret = (
            run_tbl.columns["num_zones"]
            if "num_zones" in run_tbl.columns
            else n_ret
        )
        ret_m = (
            finite_mask(m_run)
            & finite_mask(n_ret)
            & finite_mask(y_ret)
            & (n_ret > 0)
        )
        if np.any(ret_m):
            ax_r.scatter(
                m_run[ret_m],
                y_ret[ret_m],
                color=style.OKABE_ITO["vermilion"],
                marker="^",
                s=18,
                label="Retries>0",
                zorder=5,
            )
    ax_r.set_ylabel("Zones ($N_z$)")
    for sub_ax in (ax, ax_r):
        y0, y1 = sub_ax.get_ylim()
        sub_ax.set_ylim(y0, y0 + 1.18 * (y1 - y0))
    merge_twinx_legend(ax, ax_r, loc="upper left", fontsize=6.5, ncol=2)

    _plot_residual_panel(axes[2, 1], ref_tbl, run_tbl, spec, comp)


def _plot_binary_comparison_panels(
    axes: np.ndarray,
    ref_tbl: history_compare.HistoryTable,
    run_tbl: history_compare.HistoryTable,
    spec: case_manifest.CaseManifestEntry,
    comp: history_compare.HistoryComparison | None,
) -> None:
    """Populate the 3x2 grid for ``binary`` evolution comparisons."""
    col_run = style.OKABE_ITO["blue"]
    ref_x, run_x, x_label = scaled_coord_pair(ref_tbl, run_tbl, spec)

    single_panels = (
        (
            axes[0, 0],
            "period_days",
            "Orbital Period [$\\mathrm{d}$]",
            "Orbital Period",
            col_run,
        ),
        (
            axes[0, 1],
            "binary_separation",
            "Separation [$R_\\odot$]",
            "Binary Separation",
            col_run,
        ),
        (
            axes[1, 0],
            "lg_mtransfer_rate",
            (
                "$\\log_{10}(|\\dot{M}_{\\mathrm{RLOF}}| / "
                "M_\\odot\\,\\mathrm{yr}^{-1})$"
            ),
            "Mass-Transfer Rate",
            style.OKABE_ITO["vermilion"],
        ),
    )
    for ax, col_name, ylabel, title, f_col in single_panels:
        _plot_table_pair(
            ax,
            ref_tbl,
            run_tbl,
            "",
            col_name,
            run_color=f_col,
            ref_x=ref_x,
            run_x=run_x,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", frameon=False, fontsize=7.5)

    # (1, 1) Component Masses
    ax = axes[1, 1]
    mass_specs = (
        ("star_1_mass", "$M_1$", style.OKABE_ITO["blue"]),
        ("star_2_mass", "$M_2$", style.OKABE_ITO["bluish_green"]),
    )
    _plot_paired_series(
        ax, ref_x, run_x, ref_tbl, run_tbl, mass_specs, ref_alpha=0.8
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Component Mass [$M_\\odot$]")
    ax.set_title("Component Masses")
    ax.legend(loc="best", frameon=False, fontsize=7.0, ncol=2)

    # (2, 0) Roche-Lobe Relative Overflow
    ax = axes[2, 0]
    ax.axhline(0.0, color=style.OKABE_ITO["grey"], ls=":", lw=0.8)
    rl_specs = (
        ("rl_relative_overflow_1", "Star 1", style.OKABE_ITO["blue"]),
        ("rl_relative_overflow_2", "Star 2", style.OKABE_ITO["orange"]),
    )
    _plot_paired_series(
        ax, ref_x, run_x, ref_tbl, run_tbl, rl_specs, ref_alpha=0.8
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("$(R - R_{\\mathrm{RL}}) / R_{\\mathrm{RL}}$")
    ax.set_title("Roche-Lobe Overflow")
    ax.legend(loc="best", frameon=False, fontsize=7.0, ncol=2)

    _plot_residual_panel(axes[2, 1], ref_tbl, run_tbl, spec, comp)
