# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Upstream ``pgstar`` ``Dynamo``, ``dPg_dnu``, ``rti``, and CMD panels.

Ports ``Dyn_plot`` (`star/private/pgstar_dynamo.f90:57-315`),
``do_dPg_dnu_Plot`` (`star/private/pgstar_dpg_dnu.f90:55-84`),
``do_rti_Plot`` (`star/private/pgstar_rti.f90:55-481`), and
``do_Color_Magnitude_plot`` (`star/private/pgstar_color_magnitude.f90:589-923`)
strictly from saved MESA ``profileN.data`` and ``history.data`` files (`#173`).
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.axes as mpl_axes
import matplotlib.collections as mpl_collections
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np

from flexmesa import constants
from flexmesa.plotting import (
    _compare_panels,
    _plot_support,
    compare,
    pgstar_grid,
    pgstar_panels,
    style,
)
from flexmesa.testing import history_compare

_RTI_INDEX_RE = re.compile(r"^rti_type_(\d+)$")
_MIX_INDEX_RE = re.compile(r"^mix_type_(\d+)$")

_RTI_CORE_SPECS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("he_core_mass",), r"$M_{\mathrm{He}}$", style.OKABE_ITO["bluish_green"]),
    (
        ("c_core_mass", "co_core_mass"),
        r"$M_{\mathrm{C}}$",
        style.OKABE_ITO["orange"],
    ),
    (("o_core_mass",), r"$M_{\mathrm{O}}$", style.OKABE_ITO["sky_blue"]),
    (
        ("si_core_mass",),
        r"$M_{\mathrm{Si}}$",
        style.OKABE_ITO["reddish_purple"],
    ),
    (("fe_core_mass",), r"$M_{\mathrm{Fe}}$", style.OKABE_ITO["vermilion"]),
    (("shock_mass",), r"$M_{\mathrm{shock}}$", style.OKABE_ITO["black"]),
)


@dataclasses.dataclass(frozen=True)
class ColorMagnitudeSpec:
    """Parsed ``Color_magnitude<N>`` namelist specification."""

    xaxis1: str
    xaxis2: str
    xaxis_reversed: bool
    title: str
    panels: tuple[tuple[str, str, str, str, bool, bool], ...]


def _extract_log_profile_col(
    prof: compare.ProfileTable, log_col: str, lin_col: str
) -> tuple[str, bool] | None:
    """Return ``(col_name, is_linear)`` for a log/linear profile pair."""
    if log_col in prof.columns:
        return log_col, False
    if lin_col in prof.columns:
        return lin_col, True
    return None


def plot_dynamo_profile(
    profile_path: pathlib.Path | str,
    *,
    ref_profile_path: pathlib.Path | str | None = None,
    xaxis: str = "mass",
    ymin_left: float | None = None,
    ymax_left: float | None = None,
    ymin_right: float | None = None,
    ymax_right: float | None = None,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot radial ``Dynamo`` magnetic/rotation profiles (`pgstar_dynamo.f90`).

    Implements ``Dyn_plot`` (`star/private/pgstar_dynamo.f90:99-312`), drawing
    $\log_{10}(B_\phi/\mathrm{G})$ and $\log_{10}(B_r/\mathrm{G})$ on the left
    axis and $\log_{10}(\Omega/\mathrm{rad\,s^{-1}})$ plus
    $\log_{10}(j_{\mathrm{rot}} / 10^{20}\,\mathrm{cm^2\,s^{-1}})$ on the right
    axis, paired with the bottom mixing-region bar.

    Args:
        profile_path: Path to ``profileN.data``.
        ref_profile_path: Optional MESA ``26.4.1`` ``profileN.data`` overlay.
        xaxis: Radial profile coordinate (default ``"mass"``).
        ymin_left: Optional lower left-axis bound ($\log_{10}(B / \mathrm{G})$).
        ymax_left: Optional upper left-axis bound.
        ymin_right: Optional lower right-axis bound.
        ymax_right: Optional upper right-axis bound.
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)`` (primary left ``Axes``).

    Raises:
        KeyError: If no magnetic-field columns (``dynamo_log_B_phi`` /
            ``dynamo_B_phi`` or ``dynamo_log_B_r`` / ``dynamo_B_r``) exist in
            ``profileN.data``.
    """
    prof = compare.load_profile_file(profile_path)
    bphi_col = _extract_log_profile_col(
        prof, "dynamo_log_B_phi", "dynamo_B_phi"
    )
    br_col = _extract_log_profile_col(prof, "dynamo_log_B_r", "dynamo_B_r")
    if bphi_col is None and br_col is None:
        raise KeyError(
            "Dynamo panel requires magnetic profile columns "
            "'dynamo_log_B_phi'/'dynamo_B_phi' or "
            "'dynamo_log_B_r'/'dynamo_B_r' in profileN.data"
        )
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
    fig, ax = style.init_ax(ax, "single", 0.82)

    left_vals: list[np.ndarray] = []
    for spec_col, lbl, c_hex in (
        (bphi_col, r"$\log B_\phi$", style.OKABE_ITO["bluish_green"]),
        (br_col, r"$\log B_r$", style.OKABE_ITO["vermilion"]),
    ):
        if spec_col is None:
            continue
        col_name, is_lin = spec_col
        tf = pgstar_panels.safe_log10_pos if is_lin else None
        _plot_support.plot_profile_pair(
            ax,
            prof,
            x_run,
            ref_prof,
            x_ref,
            col_name,
            label=lbl,
            color=c_hex,
            lw=1.55,
            transform=tf,
        )
        yv = (
            tf(prof.columns[col_name])
            if tf is not None
            else prof.columns[col_name]
        )
        fin = yv[_compare_panels.finite_mask(yv) & (yv > -20.0)]
        if fin.size > 0:
            left_vals.append(fin)

    # star/private/pgstar_dynamo.f90:187-215
    lo_l = -2.0 if ymin_left is None else float(ymin_left)
    hi_l = 10.0 if ymax_left is None else float(ymax_left)
    if left_vals and ymin_left is None and ymax_left is None:
        cat_l = np.concatenate(left_vals)
        lo_l = min(lo_l, float(np.min(cat_l)))
        hi_l = max(hi_l, float(np.max(cat_l)))
    dy_l = max(hi_l - lo_l, 1.0)
    ax.set_ylim(lo_l - 0.1 * dy_l, hi_l + 0.1 * dy_l)
    ax.set_ylabel(r"Magnetic Field [$\log_{10}(B / \mathrm{G})$]")

    omega_col = _extract_log_profile_col(prof, "log_omega", "omega")
    jrot_col = _extract_log_profile_col(prof, "log_j_rot", "j_rot")
    if omega_col is not None or jrot_col is not None:
        ax_r = ax.twinx()
        right_vals: list[np.ndarray] = []
        if omega_col is not None:
            o_name, o_lin = omega_col
            tf_o = pgstar_panels.safe_log10_pos if o_lin else None
            _plot_support.plot_profile_pair(
                ax_r,
                prof,
                x_run,
                ref_prof,
                x_ref,
                o_name,
                label=r"$\log\Omega$",
                color=style.OKABE_ITO["blue"],
                ls="--",
                lw=1.4,
                transform=tf_o,
            )
            ov = (
                tf_o(prof.columns[o_name])
                if tf_o is not None
                else prof.columns[o_name]
            )
            fin_o = ov[_compare_panels.finite_mask(ov) & (ov > -30.0)]
            if fin_o.size > 0:
                right_vals.append(fin_o)
        if jrot_col is not None:
            j_name, j_lin = jrot_col

            def _shift_j(arr: np.ndarray, lin: bool = j_lin) -> np.ndarray:
                base = pgstar_panels.safe_log10_pos(arr) if lin else arr
                return base - 20.0

            _plot_support.plot_profile_pair(
                ax_r,
                prof,
                x_run,
                ref_prof,
                x_ref,
                j_name,
                label=r"$\log(j / 10^{20})$",
                color=style.OKABE_ITO["orange"],
                ls="-.",
                lw=1.4,
                transform=_shift_j,
            )
            jv = _shift_j(prof.columns[j_name])
            fin_j = jv[_compare_panels.finite_mask(jv) & (jv > -30.0)]
            if fin_j.size > 0:
                right_vals.append(fin_j)
        # star/private/pgstar_dynamo.f90:241-269
        lo_r = -10.0 if ymin_right is None else float(ymin_right)
        hi_r = 0.0 if ymax_right is None else float(ymax_right)
        if right_vals and ymin_right is None and ymax_right is None:
            cat_r = np.concatenate(right_vals)
            lo_r = min(lo_r, float(np.min(cat_r)))
            hi_r = max(hi_r, float(np.max(cat_r)))
        dy_r = max(hi_r - lo_r, 1.0)
        ax_r.set_ylim(lo_r - 0.1 * dy_r, hi_r + 0.1 * dy_r)
        ax_r.set_ylabel(
            r"$\log\Omega$ [$\mathrm{rad\,s^{-1}}$] / "
            r"$\log(j/10^{20}\,\mathrm{cm^2\,s^{-1}})$"
        )
        _compare_panels.merge_twinx_legend(
            ax, ax_r, loc="best", fontsize=6.5, ncol=2
        )
    else:
        ax.legend(loc="best", frameon=False, fontsize=6.8)

    if rev_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    _plot_support.draw_xaxis_mixing_bar(ax, x_run, prof)
    ax.set_xlabel(x_label)
    _plot_support.set_profile_panel_title(
        ax, "Dynamo", prof, ref_prof, standalone=standalone
    )
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def plot_dpg_dnu(
    history_path: pathlib.Path | str,
    *,
    ref_history_path: pathlib.Path | str | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot gravity-mode period spacing ``dPg_dnu`` (`pgstar_dpg_dnu.f90`).

    Implements ``do_dPg_dnu_Plot`` (`star/private/pgstar_dpg_dnu.f90:55-84`),
    plotting ``delta_Pg`` ($\Delta\Pi_1$ in seconds) against ``delta_nu``
    ($\Delta\nu$ in $\mu\mathrm{Hz}$) strictly from saved ``history.data``.

    Args:
        history_path: Path to ``history.data``.
        ref_history_path: Optional MESA ``26.4.1`` reference history overlay.
        xmin: Optional lower $\Delta\nu$ limit ($\mu\mathrm{Hz}$).
        xmax: Optional upper $\Delta\nu$ limit ($\mu\mathrm{Hz}$).
        ymin: Optional lower $\Delta\Pi_g$ limit ($\mathrm{s}$).
        ymax: Optional upper $\Delta\Pi_g$ limit ($\mathrm{s}$).
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.
    """
    fig, out_ax = pgstar_grid.plot_history_track(
        history_path,
        preset="dPg_dnu",
        ref_history_path=ref_history_path,
        ax=ax,
    )
    if xmin is not None or xmax is not None:
        cur_lo, cur_hi = out_ax.get_xlim()
        out_ax.set_xlim(
            cur_lo if xmin is None else float(xmin),
            cur_hi if xmax is None else float(xmax),
        )
    if ymin is not None or ymax is not None:
        cur_lo, cur_hi = out_ax.get_ylim()
        out_ax.set_ylim(
            cur_lo if ymin is None else float(ymin),
            cur_hi if ymax is None else float(ymax),
        )
    _compare_panels.save_figure(fig, save_path)
    return fig, out_ax


def _rti_xaxis_values(
    tbl: history_compare.HistoryTable,
    xaxis: Literal["model_number", "star_age"],
    *,
    age_units: Literal["yr", "Myr", "s"],
    time_from_present: bool,
    log_xaxis: bool,
) -> tuple[np.ndarray, str]:
    if xaxis not in tbl.columns:
        raise KeyError(f"rti panel requires {xaxis!r} column in history.data")
    xv = np.array(tbl.columns[xaxis], dtype=np.float64, copy=True)
    if xaxis == "model_number":
        return xv, "Model Number"
    unit_tag = "yr"
    if age_units == "s":
        xv = xv * constants.SECYER
        unit_tag = "s"
    elif age_units == "Myr":
        xv = xv * 1.0e-6
        unit_tag = "Myr"
    if time_from_present:
        t_now = float(xv[-1])
        xv = np.abs(t_now - xv) if log_xaxis else (xv - t_now)
    if log_xaxis:
        xv = np.log10(np.maximum(np.abs(xv), 1.0e-50))
        if time_from_present:
            lbl = (
                r"$\log_{10}((t_{\mathrm{final}} - t) / "
                rf"\mathrm{{{unit_tag}}})$"
            )
        else:
            lbl = rf"$\log_{{10}}(\mathrm{{Age}} / \mathrm{{{unit_tag}}})$"
        return xv, lbl
    lbl = (
        f"Age from Final Model [{unit_tag}]"
        if time_from_present
        else f"Stellar Age [{unit_tag}]"
    )
    return xv, lbl


def _collect_rti_segments_from_history(
    tbl: history_compare.HistoryTable,
    x_vals: np.ndarray,
    star_mass: np.ndarray,
    m_center: np.ndarray,
    *,
    rti_interval: int,
) -> tuple[list[np.ndarray], bool]:
    """Extract vertical RTI mass spans from ``rti_type_*`` or ``mix_type_*``."""
    rti_pairs: list[tuple[int, str, str]] = []
    for col in tbl.columns:
        m = _RTI_INDEX_RE.match(col)
        if m:
            idx = int(m.group(1))
            q_col = f"rti_qtop_{idx}"
            if q_col in tbl.columns:
                rti_pairs.append((idx, col, q_col))
    rti_pairs.sort(key=lambda item: item[0])

    mix_pairs: list[tuple[int, str, str]] = []
    if not rti_pairs:
        for col in tbl.columns:
            m = _MIX_INDEX_RE.match(col)
            if m:
                idx = int(m.group(1))
                q_col = f"mix_qtop_{idx}"
                if q_col in tbl.columns:
                    mix_pairs.append((idx, col, q_col))
        mix_pairs.sort(key=lambda item: item[0])

    if not rti_pairs and not mix_pairs:
        return [], False

    use_rti = bool(rti_pairs)
    pairs = rti_pairs if use_rti else mix_pairs
    step_inc = max(int(rti_interval), 1)
    segs: list[np.ndarray] = []

    for r_i in range(0, tbl.num_rows, step_inc):
        xv = float(x_vals[r_i])
        mc = float(m_center[r_i])
        xmass = float(star_mass[r_i]) - mc
        if not (np.isfinite(xv) and np.isfinite(xmass) and xmass > 0.0):
            continue
        qbot = 0.0
        for _k, t_col, q_col in pairs:
            t_val = round(float(tbl.columns[t_col][r_i]))
            if t_val < 0:
                break
            qtop = float(tbl.columns[q_col][r_i])
            is_active = (t_val > 0) if use_rti else (t_val == 6)
            if is_active and np.isfinite(qtop) and qtop > qbot:
                mbot = mc + xmass * qbot
                mtop = mc + xmass * qtop
                if mtop > 1.0e-50:
                    segs.append(
                        np.array([[xv, mbot], [xv, mtop]], dtype=np.float64)
                    )
            if np.isfinite(qtop):
                qbot = max(qbot, qtop)
    return segs, True


def _collect_rti_segments_from_profiles(
    profile_paths: Sequence[pathlib.Path | str],
    xaxis: Literal["model_number", "star_age"],
    *,
    age_units: Literal["yr", "Myr", "s"],
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """Extract RTI spans from profiles (`mixing_type == 6` or `alpha_RTI`)."""
    segs: list[np.ndarray] = []
    xs_all: list[float] = []
    m_max_all: list[float] = []
    for p_path in profile_paths:
        prof = compare.load_profile_file(p_path)
        if "mass" not in prof.columns:
            raise KeyError(f"rti profile {p_path} requires 'mass' column")
        if xaxis == "model_number":
            xv = float(prof.header.get("model_number", 0.0))
        else:
            age_yr = float(prof.header.get("star_age", 0.0))
            xv = (
                age_yr * constants.SECYER
                if age_units == "s"
                else (age_yr * 1.0e-6 if age_units == "Myr" else age_yr)
            )
        m_arr = prof.columns["mass"]
        has_mt = "mixing_type" in prof.columns
        has_arti = "alpha_RTI" in prof.columns
        if not (has_mt or has_arti):
            raise KeyError(
                f"rti profile {p_path} requires 'mixing_type' or 'alpha_RTI'"
            )
        active = np.zeros(m_arr.size, dtype=bool)
        if has_mt:
            active |= np.rint(prof.columns["mixing_type"]).astype(np.int64) == 6
        if has_arti:
            active |= prof.columns["alpha_RTI"] > 0.0
        order = np.argsort(m_arr, kind="mergesort")
        ms, acts = m_arr[order], active[order]
        xs_all.append(xv)
        m_max_all.append(float(ms[-1]))
        in_run = False
        run_bot = 0.0
        for idx in range(ms.size):
            if acts[idx] and not in_run:
                in_run = True
                run_bot = float(ms[idx])
            elif not acts[idx] and in_run:
                in_run = False
                run_top = float(ms[idx - 1])
                if run_top > run_bot:
                    segs.append(
                        np.array(
                            [[xv, run_bot], [xv, run_top]], dtype=np.float64
                        )
                    )
        if in_run and float(ms[-1]) > run_bot:
            segs.append(
                np.array([[xv, run_bot], [xv, float(ms[-1])]], dtype=np.float64)
            )
    return segs, xs_all, m_max_all


def plot_rti_history(
    history_path: pathlib.Path | str | None = None,
    *,
    profile_paths: Sequence[pathlib.Path | str] | None = None,
    ref_history_path: pathlib.Path | str | None = None,
    xaxis: Literal["model_number", "star_age"] = "model_number",
    age_units: Literal["yr", "Myr", "s"] = "yr",
    log_xaxis: bool = False,
    time_from_present: bool = False,
    xaxis_reversed: bool | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    rti_interval: int = 1,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot Rayleigh-Taylor Instability (``rti``) evolution (`pgstar_rti.f90`).

    Implements ``do_rti_Plot`` from ``star/private/pgstar_rti.f90:55-481``
    (`star/defaults/pgstar.defaults:1346-1439`), rendering vertical RTI regions
    from ``rti_type_*``/``rti_qtop_*`` (or ``mix_type_* == 6`` / profile
    ``mixing_type == 6`` / ``alpha_RTI``) alongside total, core, and shock mass
    boundaries.

    Args:
        history_path: Path to ``history.data`` containing ``star_mass`` and
            ``rti_type_*``/``rti_qtop_*`` (or ``mix_type_*``/``mix_qtop_*``).
        profile_paths: Optional sequence of ``profileN.data`` files supplying
            RTI zones when history region columns are absent.
        ref_history_path: Optional MESA ``26.4.1`` reference history overlay.
        xaxis: Horizontal coordinate (``"model_number"`` or ``"star_age"``).
        age_units: Units when ``xaxis == "star_age"`` (``"yr"``, ``"Myr"``,
            ``"s"``).
        log_xaxis: Apply $\log_{10}$ to ``star_age``.
        time_from_present: Plot time relative to the terminal model.
        xaxis_reversed: Optional axis-reversal override.
        ymin: Optional lower mass bound ($M_\odot$).
        ymax: Optional upper mass bound ($M_\odot$).
        rti_interval: Model stride between drawn vertical RTI segments.
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.
    """
    if history_path is None and not profile_paths:
        raise ValueError("provide history_path and/or profile_paths for rti")

    tbl: history_compare.HistoryTable | None = None
    ref_tbl: history_compare.HistoryTable | None = None
    ref_hdr: dict[str, Any] | None = None
    if history_path is not None:
        tbl, _ = _plot_support.load_history_with_header(history_path)
    if ref_history_path is not None:
        ref_tbl, ref_hdr = _plot_support.load_history_with_header(
            ref_history_path
        )

    segs: list[np.ndarray] = []
    has_rti_src = False
    x_run: np.ndarray | None = None
    x_lbl = "Model Number"
    star_mass: np.ndarray | None = None
    m_center: np.ndarray | None = None

    if tbl is not None:
        if "star_mass" not in tbl.columns:
            raise KeyError("rti panel requires 'star_mass' in history.data")
        x_run, x_lbl = _rti_xaxis_values(
            tbl,
            xaxis,
            age_units=age_units,
            time_from_present=time_from_present,
            log_xaxis=log_xaxis,
        )
        star_mass = tbl.columns["star_mass"]
        if "log_xmstar" in tbl.columns:
            m_center = np.maximum(
                star_mass
                - np.power(10.0, tbl.columns["log_xmstar"]) / constants.MSUN,
                0.0,
            )
        else:
            m_center = np.zeros(tbl.num_rows, dtype=np.float64)
        segs, has_rti_src = _collect_rti_segments_from_history(
            tbl, x_run, star_mass, m_center, rti_interval=rti_interval
        )

    prof_xs: list[float] = []
    prof_mmax: list[float] = []
    if profile_paths:
        p_segs, prof_xs, prof_mmax = _collect_rti_segments_from_profiles(
            profile_paths, xaxis, age_units=age_units
        )
        segs.extend(p_segs)
        has_rti_src = True

    if not has_rti_src:
        raise KeyError(
            "rti panel requires 'rti_type_*'/'rti_qtop_*' or "
            "'mix_type_*'/'mix_qtop_*' in history.data (or profile_paths)"
        )

    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.78)
    if segs:
        lc = mpl_collections.LineCollection(
            segs,
            colors=style.OKABE_ITO["blue"],
            linewidths=2.0,
            alpha=0.82,
            zorder=2,
            label="RTI",
        )
        ax.add_collection(lc)

    y_hi = (
        float(np.nanmax(star_mass))
        if star_mass is not None
        else max(prof_mmax, default=1.0)
    )
    y_lo = float(np.nanmin(m_center)) if m_center is not None else 0.0
    if tbl is not None and x_run is not None and star_mass is not None:
        _compare_panels._plot_series(
            ax,
            x_run,
            star_mass,
            label=r"$M_{\mathrm{total}}$",
            color=style.OKABE_ITO["grey"],
            lw=1.5,
            zorder=3,
        )
        x_ref = (
            _rti_xaxis_values(
                ref_tbl,
                xaxis,
                age_units=age_units,
                time_from_present=time_from_present,
                log_xaxis=log_xaxis,
            )[0]
            if ref_tbl is not None and xaxis in ref_tbl.columns
            else None
        )
        if (
            ref_tbl is not None
            and x_ref is not None
            and "star_mass" in ref_tbl.columns
        ):
            _compare_panels._plot_series(
                ax,
                x_ref,
                ref_tbl.columns["star_mass"],
                label=r"$M_{\mathrm{total}}$ MESA",
                color=style.OKABE_ITO["grey"],
                ls="--",
                lw=1.0,
                alpha=0.72,
                zorder=2,
            )
        for cands, lbl, c_hex in _RTI_CORE_SPECS:
            hit = next((c for c in cands if c in tbl.columns), None)
            if hit is None:
                continue
            yv = tbl.columns[hit]
            fin = _compare_panels.finite_mask(yv)
            if np.any(fin) and float(np.max(yv[fin])) > 1.0e-2 * y_hi:
                _compare_panels._plot_series(
                    ax, x_run, yv, label=lbl, color=c_hex, lw=1.35, zorder=4
                )
            if (
                ref_tbl is not None
                and x_ref is not None
                and hit in ref_tbl.columns
            ):
                rv = ref_tbl.columns[hit]
                rfin = _compare_panels.finite_mask(rv)
                if np.any(rfin) and float(np.max(rv[rfin])) > 1.0e-2 * y_hi:
                    _compare_panels._plot_series(
                        ax,
                        x_ref,
                        rv,
                        label="_nolegend_",
                        color=c_hex,
                        ls="--",
                        lw=0.95,
                        alpha=0.72,
                        zorder=3,
                    )
    elif prof_xs:
        ax.set_xlim(min(prof_xs) - 0.5, max(prof_xs) + 0.5)

    eff_ymin = y_lo if ymin is None else float(ymin)
    eff_ymax = y_hi * 1.04 if ymax is None else float(ymax)
    ax.set_ylim(eff_ymin, max(eff_ymax, eff_ymin + 0.1))

    do_rev = (
        (time_from_present and log_xaxis)
        if xaxis_reversed is None
        else xaxis_reversed
    )
    if do_rev and not ax.xaxis_inverted():
        ax.invert_xaxis()

    ax.set_xlabel(x_lbl)
    ax.set_ylabel(r"Enclosed Mass [$M / M_\odot$]")
    if tbl is not None:
        _plot_support.set_history_panel_title(
            ax,
            "Rayleigh-Taylor Instability",
            tbl,
            ref_tbl,
            ref_path=ref_history_path,
            ref_hdr=ref_hdr,
            standalone=standalone,
        )
    else:
        ax.set_title("Rayleigh-Taylor Instability")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", frameon=False, fontsize=6.6, ncol=2)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def format_cmd_label(col1: str, col2: str = "") -> str:
    """Format photometric axis label (`pgstar_color_magnitude.f90:887-920`)."""
    c1, c2 = col1.strip(), col2.strip()

    def _one(tok: str) -> str:
        if tok.startswith("bc_"):
            return tok[3:]
        if tok.startswith("abs_mag_"):
            return rf"$M_{{\mathrm{{{tok[8:]}}}}}$"
        return tok

    if not c2:
        return _one(c1)
    if c1.startswith("bc_") and c2.startswith("bc_"):
        return f"{c1[3:]} - {c2[3:]}"
    if c1.startswith("abs_mag_") and c2.startswith("abs_mag_"):
        return rf"$M_{{\mathrm{{{c1[8:]}}}}} - M_{{\mathrm{{{c2[8:]}}}}}$"
    return f"{_one(c1)} - {_one(c2)}"


def parse_inlist_cmd_spec(
    inlist_path: pathlib.Path | str,
    *,
    panel_index: int = 1,
) -> ColorMagnitudeSpec:
    """Parse ``Color_magnitude<N>_*`` controls from an ``inlist`` file."""
    if not 1 <= panel_index <= 9:
        raise ValueError(f"panel_index must be in 1..9, got {panel_index!r}")
    blocks = _plot_support.load_plotting_namelist_blocks(
        inlist_path, namelist_name="pgstar"
    )
    prefix = f"color_magnitude{panel_index}_"
    x1 = x2 = title = ""
    x_rev = False
    n_panels = 0
    y1: dict[int, str] = {}
    y2: dict[int, str] = {}
    oy1: dict[int, str] = {}
    oy2: dict[int, str] = {}
    y_rev: dict[int, bool] = {}
    oy_rev: dict[int, bool] = {}

    for blk in blocks:
        if blk.name.lower() != "pgstar":
            continue
        for asg in blk.assignments:
            k_low = asg.name.lower()
            if not k_low.startswith(prefix):
                continue
            tail = k_low[len(prefix) :]
            if asg.subscript is not None and ":" in asg.subscript:
                if tail == "yaxis1_name":
                    y1.clear()
                elif tail == "yaxis2_name":
                    y2.clear()
                elif tail == "other_yaxis1_name":
                    oy1.clear()
                elif tail == "other_yaxis2_name":
                    oy2.clear()
                continue
            sub_i = (
                int(asg.subscript)
                if asg.subscript is not None and asg.subscript.isdigit()
                else None
            )
            if tail == "xaxis1_name":
                x1 = str(asg.value).strip()
            elif tail == "xaxis2_name":
                x2 = str(asg.value).strip()
            elif tail == "xaxis_reversed":
                x_rev = bool(asg.value)
            elif tail == "title":
                title = str(asg.value).strip()
            elif tail == "num_panels":
                n_panels = int(asg.value)
            elif tail == "yaxis1_name" and sub_i is not None:
                y1[sub_i] = str(asg.value).strip()
            elif tail == "yaxis2_name" and sub_i is not None:
                y2[sub_i] = str(asg.value).strip()
            elif tail == "other_yaxis1_name" and sub_i is not None:
                oy1[sub_i] = str(asg.value).strip()
            elif tail == "other_yaxis2_name" and sub_i is not None:
                oy2[sub_i] = str(asg.value).strip()
            elif tail == "yaxis_reversed" and sub_i is not None:
                y_rev[sub_i] = bool(asg.value)
            elif tail == "other_yaxis_reversed" and sub_i is not None:
                oy_rev[sub_i] = bool(asg.value)

    n_slots = n_panels if n_panels > 0 else max(y1, default=0)
    rows: list[tuple[str, str, str, str, bool, bool]] = []
    for idx in range(1, n_slots + 1):
        a1 = y1.get(idx, "").strip()
        oa1 = oy1.get(idx, "").strip()
        if not a1 and not oa1:
            continue
        rows.append(
            (
                a1,
                y2.get(idx, "").strip(),
                oa1,
                oy2.get(idx, "").strip(),
                y_rev.get(idx, True),
                oy_rev.get(idx, True),
            )
        )
    if not x1 or not rows:
        raise ValueError(
            f"Color_magnitude{panel_index} specification incomplete in "
            f"{inlist_path}"
        )
    return ColorMagnitudeSpec(
        xaxis1=x1,
        xaxis2=x2,
        xaxis_reversed=x_rev,
        title=title,
        panels=tuple(rows),
    )


def _auto_detect_cmd_columns(
    tbl: history_compare.HistoryTable,
) -> tuple[str, str, list[tuple[str, str, str, str, bool, bool]]]:
    """Auto-detect a standard CMD color/magnitude pair from ``history.data``."""
    cols = set(tbl.columns)
    preferred_triples = (
        ("abs_mag_B", "abs_mag_V", "abs_mag_V"),
        ("abs_mag_Gbp", "abs_mag_Grp", "abs_mag_G"),
        ("abs_mag_g", "abs_mag_r", "abs_mag_g"),
        ("bc_B", "bc_V", "bc_V"),
    )
    for c1, c2, cy in preferred_triples:
        if c1 in cols and c2 in cols and cy in cols:
            return c1, c2, [(cy, "", "", "", True, True)]
    abs_mags = sorted(c for c in cols if c.startswith("abs_mag_"))
    if len(abs_mags) >= 2:
        return abs_mags[0], abs_mags[1], [(abs_mags[1], "", "", "", True, True)]
    bcs = sorted(c for c in cols if c.startswith("bc_"))
    if len(bcs) >= 2:
        return bcs[0], bcs[1], [(bcs[1], "", "", "", True, True)]
    raise KeyError(
        "Color_magnitude panel requires photometric columns (such as "
        "'abs_mag_*' or 'bc_*') in history.data or explicit xaxis1/panels"
    )


def _eval_cmd_series(
    tbl: history_compare.HistoryTable, col1: str, col2: str
) -> np.ndarray:
    if col1 not in tbl.columns:
        raise KeyError(
            f"Color_magnitude column {col1!r} not found in history.data"
        )
    arr = np.array(tbl.columns[col1], dtype=np.float64, copy=True)
    if col2:
        if col2 not in tbl.columns:
            raise KeyError(
                f"Color_magnitude column {col2!r} not found in history.data"
            )
        arr = arr - tbl.columns[col2]
    # star/private/pgstar_color_magnitude.f90:777-789
    return np.where(
        _compare_panels.finite_mask(arr), np.clip(arr, -100.0, 100.0), np.nan
    )


def plot_color_magnitude(
    history_path: pathlib.Path | str,
    xaxis1: str | None = None,
    xaxis2: str | None = None,
    panels: Sequence[
        str
        | tuple[str, str | None]
        | tuple[str, str | None, str | None, str | None]
    ]
    | None = None,
    *,
    ref_history_path: pathlib.Path | str | None = None,
    inlist_path: pathlib.Path | str | None = None,
    panel_index: int = 1,
    xaxis_reversed: bool | None = None,
    yaxis_reversed: bool = True,
    xmin: float | None = None,
    xmax: float | None = None,
    axes: Sequence[mpl_axes.Axes] | None = None,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, np.ndarray]:
    r"""Plot photometric ``Color_magnitude<N>`` diagrams.

    Implements ``do_Color_Magnitude_plot`` from
    ``star/private/pgstar_color_magnitude.f90:589-923``
    (`star/defaults/pgstar.defaults:5264-5380`), supporting arbitrary color
    differences ``xaxis1 - xaxis2`` and stacked single- or dual-axis magnitude
    or color panels read strictly from saved ``history.data`` columns.

    Args:
        history_path: Path to ``history.data``.
        xaxis1: Primary horizontal column (for example ``"abs_mag_B"`` or
            ``"bc_B"``).
        xaxis2: Optional second horizontal column subtracted from ``xaxis1``
            to form a color index (for example ``"abs_mag_V"``).
        panels: Sequence of panel specs: ``"y1"``, ``("y1", "y2")``, or
            ``("y1", "y2", "other_y1", "other_y2")``.
        ref_history_path: Optional MESA ``26.4.1`` reference history overlay.
        inlist_path: Optional ``&pgstar`` inlist defining
            ``Color_magnitude<N>_*``.
        panel_index: 1-based ``Color_magnitude<N>`` index (``1..9``).
        xaxis_reversed: Optional x-axis inversion override.
        yaxis_reversed: Invert magnitude y-axes by default (brighter upward).
        xmin: Optional lower horizontal bound.
        xmax: Optional upper horizontal bound.
        axes: Optional pre-allocated axes sequence matching ``len(panels)``.
        ax: Optional single target ``Axes`` when rendering a 1-panel CMD.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, axes_arr)`` where ``axes_arr`` is a 1D ``Axes`` array.
    """
    if isinstance(panels, (str, bytes)):
        raise TypeError(
            "panels must be a sequence of column names or tuples, not bare str"
        )
    tbl, _ = _plot_support.load_history_with_header(history_path)
    ref_tbl: history_compare.HistoryTable | None = None
    ref_hdr: dict[str, Any] | None = None
    if ref_history_path is not None:
        ref_tbl, ref_hdr = _plot_support.load_history_with_header(
            ref_history_path
        )

    eff_title = "Color-Magnitude Diagram"
    eff_xrev = False if xaxis_reversed is None else bool(xaxis_reversed)
    resolved_rows: list[tuple[str, str, str, str, bool, bool]] = []

    if xaxis1 is not None or panels is not None:
        if xaxis1 is None or not panels:
            raise ValueError(
                "pass both xaxis1 and non-empty panels, or omit both to use "
                "inlist_path or auto-detected abs_mag_*/bc_* columns"
            )
        eff_x1 = xaxis1.strip()
        eff_x2 = (xaxis2 or "").strip()
        for item in panels:
            if isinstance(item, str):
                resolved_rows.append(
                    (item.strip(), "", "", "", yaxis_reversed, yaxis_reversed)
                )
            elif len(item) == 2:
                resolved_rows.append(
                    (
                        item[0].strip(),
                        (item[1] or "").strip(),
                        "",
                        "",
                        yaxis_reversed,
                        yaxis_reversed,
                    )
                )
            else:
                resolved_rows.append(
                    (
                        item[0].strip(),
                        (item[1] or "").strip(),
                        (item[2] or "").strip(),
                        (item[3] or "").strip(),
                        yaxis_reversed,
                        yaxis_reversed,
                    )
                )
    elif inlist_path is not None:
        spec = parse_inlist_cmd_spec(inlist_path, panel_index=panel_index)
        eff_x1, eff_x2 = spec.xaxis1, spec.xaxis2
        eff_xrev = (
            spec.xaxis_reversed if xaxis_reversed is None else xaxis_reversed
        )
        eff_title = spec.title or eff_title
        resolved_rows = list(spec.panels)
    else:
        eff_x1, eff_x2, resolved_rows = _auto_detect_cmd_columns(tbl)

    x_run = _eval_cmd_series(tbl, eff_x1, eff_x2)
    x_ref: np.ndarray | None = None
    if (
        ref_tbl is not None
        and eff_x1 in ref_tbl.columns
        and (not eff_x2 or eff_x2 in ref_tbl.columns)
    ):
        x_ref = _eval_cmd_series(ref_tbl, eff_x1, eff_x2)

    # Validate all required columns before creating any Figure.
    y_run_list: list[tuple[np.ndarray | None, np.ndarray | None]] = []
    for y1, y2, oy1, oy2, _yr, _oyr in resolved_rows:
        yv = _eval_cmd_series(tbl, y1, y2) if y1 else None
        oyv = _eval_cmd_series(tbl, oy1, oy2) if oy1 else None
        y_run_list.append((yv, oyv))

    n_rows = len(resolved_rows)
    standalone = axes is None and ax is None
    if axes is not None:
        if len(axes) != n_rows:
            raise ValueError(f"expected {n_rows} target axes, got {len(axes)}")
        axes_arr = np.asarray(list(axes), dtype=object)
        fig_c = axes[0].get_figure()
        if not isinstance(fig_c, mpl_figure.Figure):
            raise TypeError("target axes must belong to a Figure")
        fig = fig_c
    elif ax is not None:
        if n_rows != 1:
            raise ValueError(
                f"ax=... supports 1 CMD row, got {n_rows}; pass axes=..."
            )
        fig_c = ax.get_figure()
        if not isinstance(fig_c, mpl_figure.Figure):
            raise TypeError("target ax must belong to a Figure")
        fig = fig_c
        axes_arr = np.asarray([ax], dtype=object)
    else:
        style.use_mesa_plot_style()
        fig, raw_ax = plt.subplots(
            n_rows,
            1,
            sharex=True,
            figsize=style.figsize("single", 0.56 * max(n_rows, 1) + 0.25),
            constrained_layout=True,
        )
        axes_arr = np.atleast_1d(raw_ax)

    for idx, ((y1, y2, oy1, oy2, y_rev, oy_rev), (yv, oyv)) in enumerate(
        zip(resolved_rows, y_run_list, strict=True)
    ):
        ax_i = axes_arr[idx]
        if yv is not None:
            y_lbl = format_cmd_label(y1, y2)
            if (
                ref_tbl is not None
                and x_ref is not None
                and y1 in ref_tbl.columns
                and (not y2 or y2 in ref_tbl.columns)
            ):
                yr_ref = _eval_cmd_series(ref_tbl, y1, y2)
                _compare_panels._plot_series(
                    ax_i,
                    x_ref,
                    yr_ref,
                    label=f"{y_lbl} MESA",
                    color=style.OKABE_ITO["black"],
                    ls="--",
                    lw=1.1,
                    alpha=0.75,
                )
            _compare_panels._plot_series(
                ax_i,
                x_run,
                yv,
                label=y_lbl,
                color=style.OKABE_ITO["blue"],
                lw=1.55,
            )
            valid_xy = _compare_panels.finite_mask(
                x_run
            ) & _compare_panels.finite_mask(yv)
            if np.any(valid_xy):
                ax_i.scatter(
                    [x_run[valid_xy][-1]],
                    [yv[valid_xy][-1]],
                    color=style.OKABE_ITO["vermilion"],
                    s=22,
                    zorder=4,
                    label="Current",
                )
            ax_i.set_ylabel(y_lbl)
            if y_rev and not ax_i.yaxis_inverted():
                ax_i.invert_yaxis()

        if oyv is not None:
            oy_lbl = format_cmd_label(oy1, oy2)
            ax_r = ax_i.twinx()
            _compare_panels._plot_series(
                ax_r,
                x_run,
                oyv,
                label=oy_lbl,
                color=style.OKABE_ITO["orange"],
                lw=1.35,
            )
            ax_r.set_ylabel(oy_lbl)
            if oy_rev and not ax_r.yaxis_inverted():
                ax_r.invert_yaxis()
            _compare_panels.merge_twinx_legend(
                ax_i, ax_r, loc="best", fontsize=6.5, ncol=2
            )
        else:
            ax_i.legend(loc="best", frameon=False, fontsize=6.8)

        if xmin is not None or xmax is not None:
            cur_lo, cur_hi = ax_i.get_xlim()
            ax_i.set_xlim(
                cur_lo if xmin is None else float(xmin),
                cur_hi if xmax is None else float(xmax),
            )
        if eff_xrev and not ax_i.xaxis_inverted():
            ax_i.invert_xaxis()
        if idx < n_rows - 1:
            ax_i.tick_params(labelbottom=False)

    axes_arr[-1].set_xlabel(format_cmd_label(eff_x1, eff_x2))
    if standalone and n_rows > 1:
        if ref_tbl is not None:
            prov = _plot_support.format_history_provenance(
                tbl, ref_tbl, ref_path=ref_history_path, ref_hdr=ref_hdr
            )
            fig.suptitle(f"{eff_title}\n{prov}", fontsize=8.6)
        else:
            fig.suptitle(eff_title, fontsize=9.0)
    else:
        _plot_support.set_history_panel_title(
            axes_arr[0],
            eff_title,
            tbl,
            ref_tbl,
            ref_path=ref_history_path,
            ref_hdr=ref_hdr,
            standalone=standalone,
        )
    _compare_panels.save_figure(fig, save_path)
    return fig, axes_arr
