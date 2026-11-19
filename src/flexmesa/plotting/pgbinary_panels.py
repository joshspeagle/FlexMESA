# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Binary ``pgbinary`` diagnostic and MESA ``26.4.1`` panels (`#173`).

Ports the static content of ``binary/private/pgbinary_orbit.f90:63-296``,
``binary/private/pgbinary_summary_history.f90:55-282``, and
``binary/private/pgbinary_star.f90:93-221`` strictly from saved
``binary_history.data`` and component ``LOGS1``/``LOGS2`` ``history.data``
files, with optional MESA ``26.4.1`` reference overplotting and provenance
subtitles.
"""

from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.axes as mpl_axes
import matplotlib.colors as mpl_colors
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np

from flexmesa.plotting import _compare_panels, _plot_support, compare, style
from flexmesa.testing import case_manifest, harness_runner, history_compare

_DEFAULT_SUMMARY_COLS: tuple[tuple[str, str], ...] = (
    ("period_days", r"$P_{\mathrm{orb}}$"),
    ("binary_separation", r"$a$"),
    ("lg_mtransfer_rate", r"$\log|\dot{M}_{\mathrm{RLOF}}|$"),
    ("eccentricity", r"$e$"),
    ("J_orb", r"$J_{\mathrm{orb}}$"),
)

_PALETTE: tuple[str, ...] = (
    style.OKABE_ITO["blue"],
    style.OKABE_ITO["vermilion"],
    style.OKABE_ITO["bluish_green"],
    style.OKABE_ITO["orange"],
    style.OKABE_ITO["reddish_purple"],
    style.OKABE_ITO["sky_blue"],
)


def _xl1_fit(q_ratio: float) -> float:
    """L1 distance from Star 1 (`binary/private/pgbinary_orbit.f90:250-261`)."""
    logq = math.log10(q_ratio)
    if q_ratio > 1.0:
        logq = -logq
    xl1 = (
        -1.72452947 / math.pi * math.atan(logq * 0.21625699)
        + 0.5
        + 0.01559149 * logq
        - 1.3924e-5 * logq * (logq + 1.5) * (logq + 4.0)
    )
    return 1.0 - xl1 if q_ratio > 1.0 else xl1


def _psi_fit(req: float, q_ratio: float) -> float:
    """Roche potential fit (`binary/private/pgbinary_orbit.f90:263-271`)."""
    return (
        -1.0 / req
        - q_ratio
        - 0.5533 * (1.0 + q_ratio) * req**2
        - 0.3642 * req**2 * (req**2 - 1.0)
        - 1.8693
        * req
        * (req - 0.1)
        * (req - 0.3)
        * (req - 0.7)
        * (req - 1.0414)
    )


def _roche_potential(radius: float, cos_phi: float, q_ratio: float) -> float:
    """Roche potential (`binary/private/pgbinary_orbit.f90:273-279`)."""
    dist2 = max(1.0 - 2.0 * radius * cos_phi + radius * radius, 1.0e-16)
    return (
        -1.0 / radius
        - q_ratio * (1.0 / math.sqrt(dist2) - radius * cos_phi)
        - 0.5 * (1.0 + q_ratio) * radius * radius
    )


def _solve_roche_ray(
    cos_phi: float,
    q_ratio: float,
    target_psi: float,
    r_max: float,
) -> float:
    """Solve ``roche(r, cos_phi) == target_psi`` on ``(1e-4, r_max]``."""
    r_lo = 1.0e-4
    r_hi = max(min(r_max, 1.0 - 1.0e-3), r_lo * 2.0)
    f_hi = _roche_potential(r_hi, cos_phi, q_ratio) - target_psi
    if f_hi <= 0.0:
        # Locate local potential maximum along the ray inside [r_lo, r_hi].
        probe = np.linspace(r_lo, r_hi, 32, dtype=np.float64)
        vals = np.array(
            [
                _roche_potential(float(rv), cos_phi, q_ratio) - target_psi
                for rv in probe
            ],
            dtype=np.float64,
        )
        pos_idx = np.flatnonzero(vals >= 0.0)
        if pos_idx.size == 0:
            return float(probe[int(np.argmax(vals))])
        r_hi = float(probe[int(pos_idx[0])])
        if int(pos_idx[0]) > 0:
            r_lo = float(probe[int(pos_idx[0]) - 1])

    for _ in range(48):
        r_mid = 0.5 * (r_lo + r_hi)
        if _roche_potential(r_mid, cos_phi, q_ratio) - target_psi < 0.0:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return 0.5 * (r_lo + r_hi)


def _equipotential_xy(
    req: float,
    q_ratio: float,
    *,
    a_star: float,
    ecc: float,
    second_star: bool,
    num_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed 2D contour (`binary/private/pgbinary_orbit.f90:130-206`)."""
    req_clamped = min(max(req, 2.0e-3), 0.95)
    xl1 = _xl1_fit(q_ratio)
    this_psi = _psi_fit(req_clamped, q_ratio)
    n_tot = 2 * num_points + 1
    xs = np.zeros(n_tot, dtype=np.float64)
    ys = np.zeros(n_tot, dtype=np.float64)

    for idx in range(num_points):
        phi = (idx + 0.5) * math.pi / num_points
        cos_p = math.cos(phi)
        r_ray = _solve_roche_ray(cos_p, q_ratio, this_psi, xl1)
        xs[idx] = r_ray * cos_p
        ys[idx] = r_ray * math.sin(phi)
        ys[2 * num_points - 1 - idx] = -ys[idx]

    icut = 0
    while icut < num_points - 1 and xs[icut] > xl1:
        icut += 1
    if icut > 0:
        xs[:icut] = xs[icut]
        ys[:icut] = ys[icut]

    shift = a_star * (1.0 - ecc)
    for idx in range(num_points):
        x_disp = -(xs[idx] - shift) if second_star else (xs[idx] - shift)
        xs[idx] = x_disp
        xs[2 * num_points - 1 - idx] = x_disp
    xs[-1] = xs[0]
    ys[-1] = ys[0]
    return xs, ys


def _select_row_index(
    tbl: history_compare.HistoryTable,
    model_number: int | None,
    row_index: int,
) -> int:
    """Resolve 0-based row index from ``model_number`` or ``row_index``."""
    if model_number is not None and "model_number" in tbl.columns:
        diffs = np.abs(tbl.columns["model_number"] - float(model_number))
        return int(np.argmin(diffs))
    idx = row_index if row_index >= 0 else tbl.num_rows + row_index
    return max(0, min(idx, tbl.num_rows - 1))


def _extract_orbit_state(
    tbl: history_compare.HistoryTable,
    idx: int,
) -> dict[str, float]:
    """Extract orbital and Roche state at ``idx`` from ``binary_history``."""
    if "star_1_mass" not in tbl.columns or "star_2_mass" not in tbl.columns:
        raise ValueError(
            "binary_history requires 'star_1_mass' and 'star_2_mass' columns"
        )
    m1 = float(tbl.columns["star_1_mass"][idx])
    m2 = float(tbl.columns["star_2_mass"][idx])
    ecc = (
        float(tbl.columns["eccentricity"][idx])
        if "eccentricity" in tbl.columns
        else 0.0
    )
    ecc = min(max(ecc if math.isfinite(ecc) else 0.0, 0.0), 0.99)
    sep = (
        float(tbl.columns["binary_separation"][idx])
        if "binary_separation" in tbl.columns
        else 1.0
    )
    sep = sep if math.isfinite(sep) and sep > 0.0 else 1.0
    rl1 = float(tbl.columns["rl_1"][idx]) if "rl_1" in tbl.columns else 0.0
    rl2 = float(tbl.columns["rl_2"][idx]) if "rl_2" in tbl.columns else 0.0

    def _star_r(star_i: int, rl_val: float) -> float:
        r_col = f"star_{star_i}_radius"
        ov_col = f"rl_relative_overflow_{star_i}"
        if r_col in tbl.columns and math.isfinite(
            float(tbl.columns[r_col][idx])
        ):
            return max(float(tbl.columns[r_col][idx]), 0.0)
        if ov_col in tbl.columns and rl_val > 0.0:
            ov = float(tbl.columns[ov_col][idx])
            if math.isfinite(ov) and abs(ov) < 1.0e50:
                return max(rl_val * (1.0 + ov), 0.0)
        return 0.0

    pm_idx = (
        round(float(tbl.columns["point_mass_index"][idx]))
        if "point_mass_index" in tbl.columns
        and math.isfinite(float(tbl.columns["point_mass_index"][idx]))
        else 0
    )
    mod_no = (
        float(tbl.columns["model_number"][idx])
        if "model_number" in tbl.columns
        else float(idx + 1)
    )
    return {
        "m1": m1,
        "m2": m2,
        "ecc": ecc,
        "sep": sep,
        "rl1": rl1,
        "rl2": rl2,
        "r1": _star_r(1, rl1),
        "r2": _star_r(2, rl2),
        "pm_idx": float(pm_idx),
        "model_number": mod_no,
    }


def _draw_orbit_geometry(
    ax: mpl_axes.Axes,
    state: dict[str, float],
    *,
    is_ref: bool,
    show_roche_lobes: bool,
    show_stars: bool,
) -> float:
    """Draw Keplerian orbits and Roche/stellar contours; return ``xmax``."""
    m1, m2, ecc, sep = state["m1"], state["m2"], state["ecc"], state["sep"]
    a1 = 1.0 / (1.0 + m1 / m2)
    a2 = 1.0 / (1.0 + m2 / m1)
    num_pts = 50
    idx_arr = np.arange(1, num_pts + 1, dtype=np.float64)
    thetas = (idx_arr - 0.5) * math.pi / num_pts
    r1_half = a1 * (1.0 - ecc * ecc) / (1.0 + ecc * np.cos(thetas))
    r2_half = (a2 / a1) * r1_half

    x1 = np.concatenate(
        [-r1_half * np.cos(thetas), (-r1_half * np.cos(thetas))[::-1]]
    )
    y1 = np.concatenate(
        [-r1_half * np.sin(thetas), (r1_half * np.sin(thetas))[::-1]]
    )
    x2 = np.concatenate(
        [r2_half * np.cos(thetas), (r2_half * np.cos(thetas))[::-1]]
    )
    y2 = np.concatenate(
        [r2_half * np.sin(thetas), (-r2_half * np.sin(thetas))[::-1]]
    )
    x1 = np.append(x1, x1[0])
    y1 = np.append(y1, y1[0])
    x2 = np.append(x2, x2[0])
    y2 = np.append(y2, y2[0])

    c1 = style.OKABE_ITO["black"] if is_ref else style.OKABE_ITO["orange"]
    c2 = style.OKABE_ITO["grey"] if is_ref else style.OKABE_ITO["blue"]
    ls_orb = "--" if is_ref else ":"
    lw_orb = 1.0 if is_ref else 1.25
    sfx = " (MESA)" if is_ref else ""

    ax.plot(x1, y1, color=c1, ls=ls_orb, lw=lw_orb, label=f"Star 1 orbit{sfx}")
    ax.plot(x2, y2, color=c2, ls=ls_orb, lw=lw_orb, label=f"Star 2 orbit{sfx}")
    xmax = max(float(np.max(np.abs(x1))), float(np.max(np.abs(x2))), 0.6)

    q12 = m2 / m1
    pm_idx = round(state["pm_idx"])
    if abs(math.log10(max(q12, 1.0e-12))) <= 2.0:
        for star_i, q_loc, a_loc, rl_v, r_v, c_hex in (
            (1, q12, a1, state["rl1"], state["r1"], c1),
            (2, 1.0 / q12, a2, state["rl2"], state["r2"], c2),
        ):
            second = star_i == 2
            x_focus = a_loc * (1.0 - ecc) if second else -a_loc * (1.0 - ecc)
            if show_roche_lobes and rl_v > 0.0:
                x_rl, y_rl = _equipotential_xy(
                    rl_v / sep,
                    q_loc,
                    a_star=a_loc,
                    ecc=ecc,
                    second_star=second,
                )
                ax.plot(
                    x_rl,
                    y_rl,
                    color=c_hex,
                    ls="--" if is_ref else "-",
                    lw=0.9 if is_ref else 1.1,
                    alpha=0.65 if is_ref else 0.85,
                )
                xmax = max(xmax, float(np.max(np.abs(x_rl))))
            if pm_idx == star_i or not show_stars or r_v <= 0.0:
                if not is_ref:
                    ax.scatter(
                        [x_focus],
                        [0.0],
                        color=c_hex,
                        marker="o" if pm_idx != star_i else "D",
                        s=28,
                        zorder=4,
                    )
            else:
                x_st, y_st = _equipotential_xy(
                    r_v / sep,
                    q_loc,
                    a_star=a_loc,
                    ecc=ecc,
                    second_star=second,
                )
                ax.plot(
                    x_st,
                    y_st,
                    color=c_hex,
                    ls="--" if is_ref else "-",
                    lw=1.1 if is_ref else 1.4,
                )
                if not is_ref:
                    ax.fill(
                        x_st,
                        y_st,
                        facecolor=mpl_colors.to_rgba(c_hex, 0.24),
                        edgecolor=c_hex,
                        hatch="///" if star_i == 1 else "\\\\\\",
                        lw=0.8,
                        zorder=3,
                    )
                xmax = max(xmax, float(np.max(np.abs(x_st))))
    return xmax


def plot_pgbinary_orbit(
    binary_history: pathlib.Path | str,
    *,
    model_number: int | None = None,
    row_index: int = -1,
    mesa_binary_history: pathlib.Path | str | None = None,
    show_roche_lobes: bool = True,
    show_stars: bool = True,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot 2D binary orbital paths and Roche geometry (`pgbinary_orbit.f90`).

    Implements ``orbit_panel`` from ``binary/private/pgbinary_orbit.f90:63-296``
    strictly from saved ``binary_history.data`` rows.

    Args:
        binary_history: Path to ``binary_history.data``.
        model_number: Optional ``model_number`` selecting the snapshot row.
        row_index: Row index used when ``model_number`` is ``None`` (default
            ``-1``, terminal model).
        mesa_binary_history: Optional MESA ``26.4.1`` ``binary_history.data``
            for dashed reference overlay.
        show_roche_lobes: Draw critical Roche-lobe equipotential contours.
        show_stars: Draw hatched stellar equipotential surfaces (or compact
            markers when ``point_mass_index`` marks a point mass).
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)`` of ``Figure`` and equal-aspect ``Axes``.
    """
    fig, ax = style.init_ax(ax, "single", 0.95)
    run_tbl, _ = _plot_support.load_history_with_header(binary_history)
    r_idx = _select_row_index(run_tbl, model_number, row_index)
    run_state = _extract_orbit_state(run_tbl, r_idx)

    xmax = 0.6
    if mesa_binary_history is not None:
        ref_tbl, _ = _plot_support.load_history_with_header(mesa_binary_history)
        target_mod = (
            model_number
            if model_number is not None
            else round(run_state["model_number"])
        )
        m_idx = _select_row_index(ref_tbl, target_mod, row_index)
        ref_state = _extract_orbit_state(ref_tbl, m_idx)
        xmax = max(
            xmax,
            _draw_orbit_geometry(
                ax,
                ref_state,
                is_ref=True,
                show_roche_lobes=show_roche_lobes,
                show_stars=show_stars,
            ),
        )

    xmax = max(
        xmax,
        _draw_orbit_geometry(
            ax,
            run_state,
            is_ref=False,
            show_roche_lobes=show_roche_lobes,
            show_stars=show_stars,
        ),
    )
    ax.scatter(
        [0.0],
        [0.0],
        color=style.OKABE_ITO["black"],
        marker="+",
        s=36,
        zorder=5,
        label="CoM",
    )
    lim = 1.15 * xmax
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        r"$x / a$ ($a = " + f"{run_state['sep']:.2f}" + r"\,R_\odot$)"
    )
    ax.set_ylabel(r"$y / a$")
    ax.set_title(
        f"Orbit & Roche Geometry (model {round(run_state['model_number'])})"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=6.8)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def plot_pgbinary_summary_history(
    binary_history: pathlib.Path | str,
    *,
    mesa_binary_history: pathlib.Path | str | None = None,
    columns: Sequence[str] | None = None,
    xaxis: Literal["model_number", "age"] = "model_number",
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot min-max normalized ``Summary_History`` binary traces (`pgbinary`).

    Implements ``do_summary_history_plot`` from
    ``binary/private/pgbinary_summary_history.f90:55-282``
    (`binary/defaults/pgbinary.defaults:309-335`).

    Args:
        binary_history: Path to ``binary_history.data``.
        mesa_binary_history: Optional MESA ``26.4.1`` ``binary_history.data``
            reference to overplot using identical normalization bounds.
        columns: Optional sequence of column names overriding upstream defaults.
        xaxis: Horizontal coordinate (``"model_number"`` or ``"age"``).
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)`` with normalized trajectories in ``[0, 1]``.
    """
    fig, ax = style.init_ax(ax, "single", 0.80)
    run_tbl, _ = _plot_support.load_history_with_header(binary_history)
    ref_tbl = (
        history_compare.load_history_file(mesa_binary_history)
        if mesa_binary_history is not None
        else None
    )

    x_run = run_tbl.columns.get(
        xaxis, np.arange(1, run_tbl.num_rows + 1, dtype=np.float64)
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

    col_specs = (
        tuple((c, c) for c in columns)
        if columns is not None
        else _DEFAULT_SUMMARY_COLS
    )
    for idx, (col, lbl) in enumerate(col_specs):
        if col not in run_tbl.columns:
            continue
        y_raw = run_tbl.columns[col]
        valid = _compare_panels.finite_mask(y_raw)
        if col == "lg_mtransfer_rate":
            valid = valid & (y_raw > -98.5)
        if int(np.sum(valid)) < 2:
            continue
        vmin = float(np.min(y_raw[valid]))
        vmax = float(np.max(y_raw[valid]))
        if vmax <= vmin:
            continue
        c_hex = _PALETTE[idx % len(_PALETTE)]
        y_scaled = np.where(valid, (y_raw - vmin) / (vmax - vmin), np.nan)
        _compare_panels._plot_series(
            ax, x_run * scale, y_scaled, label=lbl, color=c_hex, lw=1.55
        )
        if ref_tbl is not None and x_ref is not None and col in ref_tbl.columns:
            yr = ref_tbl.columns[col]
            vr = _compare_panels.finite_mask(yr)
            if col == "lg_mtransfer_rate":
                vr = vr & (yr > -98.5)
            yr_scaled = np.where(vr, (yr - vmin) / (vmax - vmin), np.nan)
            _compare_panels._plot_series(
                ax,
                x_ref,
                yr_scaled,
                label=f"{lbl} MESA",
                color=c_hex,
                ls="--",
                lw=1.1,
                alpha=0.75,
            )

    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("Model Number" if xaxis == "model_number" else x_lbl)
    ax.set_ylabel(r"Scaled $(y - y_{\min}) / (y_{\max} - y_{\min})$")
    ax.set_title("Binary Summary History")
    ax.legend(loc="best", frameon=False, fontsize=6.8, ncol=2)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def _discover_component_history(
    explicit: pathlib.Path | str | None,
    base_dir: pathlib.Path | None,
    star_idx: int,
) -> pathlib.Path | None:
    """Locate component ``LOGS<star_idx>/history.data`` when present."""
    if explicit is not None:
        p = pathlib.Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"missing Star {star_idx} history: {p}")
        return p
    if base_dir is None:
        return None
    cand = base_dir / f"LOGS{star_idx}" / "history.data"
    return cand if cand.is_file() else None


def _role_tag(bin_tbl: history_compare.HistoryTable, star_idx: int) -> str:
    """Classify Star ``star_idx`` role (`pgbinary_star.f90:167-178`)."""
    if "lg_mtransfer_rate" in bin_tbl.columns:
        mdot = bin_tbl.columns["lg_mtransfer_rate"]
        valid = _compare_panels.finite_mask(mdot) & (mdot > -90.0)
        if np.any(valid):
            if "donor_index" in bin_tbl.columns:
                d_last = round(float(bin_tbl.columns["donor_index"][valid][-1]))
                return "Donor" if d_last == star_idx else "Accretor"
            return "Active RLOF"
    return "Detached"


def _plot_donor_accretor_hr(
    ax: mpl_axes.Axes,
    run_tbl: history_compare.HistoryTable,
    ref_tbl: history_compare.HistoryTable | None,
    *,
    star1_hist: pathlib.Path | None,
    star2_hist: pathlib.Path | None,
    mesa_s1_hist: pathlib.Path | None,
    mesa_s2_hist: pathlib.Path | None,
) -> None:
    """Render Donor/Accretor HR tracks (`pgbinary_star.f90:167-221`)."""
    drawn = False
    colors_pair = (style.OKABE_ITO["blue"], style.OKABE_ITO["vermilion"])
    pm_idx = (
        round(float(run_tbl.columns["point_mass_index"][-1]))
        if "point_mass_index" in run_tbl.columns
        else 0
    )
    notes: list[str] = []

    for s_idx, s_path, m_path, c_hex in (
        (1, star1_hist, mesa_s1_hist, colors_pair[0]),
        (2, star2_hist, mesa_s2_hist, colors_pair[1]),
    ):
        role = _role_tag(run_tbl, s_idx)
        if pm_idx == s_idx:
            m_col = f"star_{s_idx}_mass"
            m_val = (
                float(run_tbl.columns[m_col][-1])
                if m_col in run_tbl.columns
                else float("nan")
            )
            notes.append(f"Star {s_idx}: point mass ({m_val:.2f} $M_\\odot$)")
            continue
        if m_path is not None:
            mt = history_compare.load_history_file(m_path)
            if "log_Teff" in mt.columns and "log_L" in mt.columns:
                drawn |= _compare_panels._plot_series(
                    ax,
                    mt.columns["log_Teff"],
                    mt.columns["log_L"],
                    label=f"Star {s_idx} MESA",
                    color=c_hex,
                    ls="--",
                    lw=1.1,
                    alpha=0.72,
                )
        if s_path is not None:
            st = history_compare.load_history_file(s_path)
            if "log_Teff" in st.columns and "log_L" in st.columns:
                drawn |= _compare_panels._plot_series(
                    ax,
                    st.columns["log_Teff"],
                    st.columns["log_L"],
                    label=f"Star {s_idx} ({role})",
                    color=c_hex,
                    lw=1.6,
                )
                continue

        t_col, l_col = f"log_Teff_{s_idx}", f"log_L_{s_idx}"
        if (
            ref_tbl is not None
            and t_col in ref_tbl.columns
            and l_col in ref_tbl.columns
        ):
            drawn |= _compare_panels._plot_series(
                ax,
                ref_tbl.columns[t_col],
                ref_tbl.columns[l_col],
                label=f"Star {s_idx} MESA",
                color=c_hex,
                ls="--",
                lw=1.1,
                alpha=0.72,
            )
        if t_col in run_tbl.columns and l_col in run_tbl.columns:
            drawn |= _compare_panels._plot_series(
                ax,
                run_tbl.columns[t_col],
                run_tbl.columns[l_col],
                label=f"Star {s_idx} ({role})",
                color=c_hex,
                lw=1.6,
            )

    if drawn:
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        if notes:
            ax.text(
                0.04,
                0.06,
                "\n".join(notes),
                transform=ax.transAxes,
                fontsize=7.2,
                va="bottom",
            )
        ax.legend(loc="best", frameon=False, fontsize=6.8)
    else:
        msg = (
            "\n".join(notes)
            if notes
            else "Component LOGS1/LOGS2 HR histories\nnot supplied for this run"
        )
        ax.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8.0,
        )
    ax.set_xlabel(r"$\log_{10}(T_{\mathrm{eff}} / \mathrm{K})$")
    ax.set_ylabel(r"$\log_{10}(L / L_\odot)$")
    ax.set_title("Donor & Accretor HR Tracks")


def plot_pgbinary_panels(
    binary_history: pathlib.Path | str | None = None,
    *,
    case: str | None = None,
    flexmesa_dir: pathlib.Path | str | None = None,
    mesa_binary_history: pathlib.Path | str | None = None,
    star1_history: pathlib.Path | str | None = None,
    star2_history: pathlib.Path | str | None = None,
    mesa_star1_history: pathlib.Path | str | None = None,
    mesa_star2_history: pathlib.Path | str | None = None,
    manifest_path: pathlib.Path | str | None = None,
    orbit_model_number: int | None = None,
    numerics_mode: str = "",
    termination_code: str = "",
    controls_ignored: bool = False,
    n_controls_ignored: int = 0,
    max_steps: int | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, np.ndarray]:
    r"""Plot the 6-panel ``pgbinary`` diagnostic and MESA ``26.4.1`` figure.

    Combines 2D ``Orbit`` Roche geometry, Donor + Accretor HR tracks, Roche-lobe
    filling factor $R / R_{\mathrm{RL}}$, ``lg_mtransfer_rate``, orbital period
    and separation, and component masses / residuals strictly from stored
    outputs.

    Args:
        binary_history: Direct path to ``binary_history.data``.
        case: Optional manifest ``binary`` case name for resolving stored
            MESA ``26.4.1`` reference history and tolerances.
        flexmesa_dir: Optional run directory containing ``binary_history.data``
            and optional ``LOGS1/``, ``LOGS2/`` component histories.
        mesa_binary_history: Optional reference MESA ``binary_history.data``.
        star1_history: Optional path to Star 1 ``history.data``.
        star2_history: Optional path to Star 2 ``history.data``.
        mesa_star1_history: Optional reference Star 1 ``history.data``.
        mesa_star2_history: Optional reference Star 2 ``history.data``.
        manifest_path: Optional override path to ``case_manifest.toml``.
        orbit_model_number: Optional ``model_number`` for the `(0, 0)` orbit
            snapshot (defaults to terminal model).
        numerics_mode: Optional arithmetic mode label for provenance banner.
        termination_code: Optional termination code for provenance banner.
        controls_ignored: Whether unsupported controls were bypassed.
        n_controls_ignored: Count of bypassed controls.
        max_steps: Optional step cap used during a smoke trace.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, axes)`` with ``axes`` of shape ``(3, 2)``.
    """
    spec: case_manifest.CaseManifestEntry | None = None
    if case is not None:
        spec = compare.resolve_case_entry(
            case, suite="binary", manifest_path=manifest_path
        )

    base_dir = pathlib.Path(flexmesa_dir) if flexmesa_dir is not None else None
    if binary_history is not None:
        run_path = pathlib.Path(binary_history)
        if base_dir is None:
            base_dir = run_path.parent
    elif base_dir is not None:
        located = harness_runner.locate_written_history(
            "binary", base_dir, True
        )
        if located is None:
            raise FileNotFoundError(
                f"no binary_history.data found under {base_dir}"
            )
        run_path = located
    else:
        raise FileNotFoundError(
            "provide binary_history or flexmesa_dir for plot_pgbinary_panels"
        )

    ref_path: pathlib.Path | None = None
    if mesa_binary_history is not None:
        ref_path = pathlib.Path(mesa_binary_history)
    elif spec is not None and spec.reference_path:
        m_root = (
            pathlib.Path(manifest_path).parent
            if manifest_path is not None
            else None
        )
        ref_path = harness_runner.resolve_reference_history_path(
            spec.reference_path, root_dir=m_root
        )

    run_tbl, run_hdr = _plot_support.load_history_with_header(run_path)
    ref_tbl: history_compare.HistoryTable | None = None
    ref_hdr: dict[str, Any] = {}
    if ref_path is not None:
        ref_tbl, ref_hdr = _plot_support.load_history_with_header(ref_path)

    style.use_mesa_plot_style()
    fig, axes = plt.subplots(
        3,
        2,
        figsize=style.figsize("double", 1.16),
        constrained_layout=True,
    )

    # (0, 0) 2D Orbit & Roche Geometry
    plot_pgbinary_orbit(
        run_path,
        model_number=orbit_model_number,
        mesa_binary_history=ref_path,
        ax=axes[0, 0],
    )

    # (0, 1) Donor + Accretor HR Tracks
    s1_p = _discover_component_history(star1_history, base_dir, 1)
    s2_p = _discover_component_history(star2_history, base_dir, 2)
    ref_base = ref_path.parent if ref_path is not None else None
    ms1_p = _discover_component_history(mesa_star1_history, ref_base, 1)
    ms2_p = _discover_component_history(mesa_star2_history, ref_base, 2)
    _plot_donor_accretor_hr(
        axes[0, 1],
        run_tbl,
        ref_tbl,
        star1_hist=s1_p,
        star2_hist=s2_p,
        mesa_s1_hist=ms1_p,
        mesa_s2_hist=ms2_p,
    )

    x_col = (
        spec.independent_var
        if spec is not None and spec.independent_var in run_tbl.columns
        else ("age" if "age" in run_tbl.columns else "model_number")
    )
    run_raw_x = run_tbl.columns.get(
        x_col, np.arange(1, run_tbl.num_rows + 1, dtype=np.float64)
    )
    ref_raw_x = (
        ref_tbl.columns.get(
            x_col, np.arange(1, ref_tbl.num_rows + 1, dtype=np.float64)
        )
        if ref_tbl is not None
        else None
    )
    max_x = _compare_panels.max_finite_abs(run_raw_x)
    if ref_raw_x is not None:
        max_x = max(max_x, _compare_panels.max_finite_abs(ref_raw_x))
    x_scale, x_label = _compare_panels.age_scale_and_label(x_col, max_x)
    run_x = run_raw_x * x_scale
    ref_x = ref_raw_x * x_scale if ref_raw_x is not None else None

    # (1, 0) Roche-Lobe Filling Factor R / R_RL = 1 + rl_relative_overflow
    ax = axes[1, 0]
    ax.axhline(
        1.0, color=style.OKABE_ITO["grey"], ls=":", lw=0.9, label="RLOF=1"
    )
    for s_idx, c_hex in (
        (1, style.OKABE_ITO["blue"]),
        (2, style.OKABE_ITO["orange"]),
    ):
        ov_col = f"rl_relative_overflow_{s_idx}"
        if (
            ref_tbl is not None
            and ref_x is not None
            and ov_col in ref_tbl.columns
        ):
            _compare_panels._plot_series(
                ax,
                ref_x,
                1.0 + ref_tbl.columns[ov_col],
                label=f"$R_{s_idx}/R_{{\\mathrm{{RL}},{s_idx}}}$ MESA",
                color=c_hex,
                ls="--",
                lw=1.1,
                alpha=0.75,
            )
        if ov_col in run_tbl.columns:
            _compare_panels._plot_series(
                ax,
                run_x,
                1.0 + run_tbl.columns[ov_col],
                label=f"$R_{s_idx}/R_{{\\mathrm{{RL}},{s_idx}}}$",
                color=c_hex,
                lw=1.55,
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Filling Factor [$R_i / R_{\mathrm{RL},i}$]")
    ax.set_title("Roche-Lobe Filling Factor")
    ax.legend(loc="best", frameon=False, fontsize=6.8, ncol=2)

    # (1, 1) Mass-Transfer Rate lg_mtransfer_rate
    ax = axes[1, 1]
    mdot_specs = (
        (
            "lg_mtransfer_rate",
            r"$\dot{M}_{\mathrm{RLOF}}$",
            style.OKABE_ITO["vermilion"],
        ),
        ("lg_mstar_dot_1", r"$\dot{M}_1$", style.OKABE_ITO["blue"]),
        ("lg_mstar_dot_2", r"$\dot{M}_2$", style.OKABE_ITO["bluish_green"]),
        ("lg_mdot_edd", r"$\dot{M}_{\mathrm{Edd}}$", style.OKABE_ITO["grey"]),
    )
    active_vals: list[np.ndarray] = []
    for col, tag, c_hex in mdot_specs:
        if ref_tbl is not None and ref_x is not None and col in ref_tbl.columns:
            yr = ref_tbl.columns[col]
            _compare_panels._plot_series(
                ax,
                ref_x,
                yr,
                label=f"{tag} MESA",
                color=c_hex,
                ls="--",
                lw=1.1,
                alpha=0.72,
            )
            act_r = yr[_compare_panels.finite_mask(yr) & (yr > -90.0)]
            if act_r.size > 0:
                active_vals.append(act_r)
        if col in run_tbl.columns:
            yf = run_tbl.columns[col]
            _compare_panels._plot_series(
                ax, run_x, yf, label=tag, color=c_hex, lw=1.55
            )
            act_f = yf[_compare_panels.finite_mask(yf) & (yf > -90.0)]
            if act_f.size > 0:
                active_vals.append(act_f)
    if active_vals:
        cat_act = np.concatenate(active_vals)
        y_lo = float(np.min(cat_act)) - 0.8
        y_hi = float(np.max(cat_act)) + 0.8
        ax.set_ylim(min(y_lo, y_hi - 1.5), max(y_hi, y_lo + 1.5))
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\log_{10}(|\dot{M}| / M_\odot\,\mathrm{yr}^{-1})$")
    ax.set_title("Mass-Transfer & Component Rates")
    ax.legend(loc="best", frameon=False, fontsize=6.6, ncol=2)

    # (2, 0) Orbital Period & Binary Separation
    ax = axes[2, 0]
    p_col = "period_days" if "period_days" in run_tbl.columns else "period_hr"
    p_lbl = (
        r"Period [$\mathrm{d}$]"
        if p_col == "period_days"
        else r"Period [$\mathrm{hr}$]"
    )
    if ref_tbl is not None and ref_x is not None and p_col in ref_tbl.columns:
        _compare_panels._plot_series(
            ax,
            ref_x,
            ref_tbl.columns[p_col],
            label="$P$ MESA",
            color=style.OKABE_ITO["blue"],
            ls="--",
            lw=1.1,
            alpha=0.75,
        )
    if p_col in run_tbl.columns:
        _compare_panels._plot_series(
            ax,
            run_x,
            run_tbl.columns[p_col],
            label="$P$",
            color=style.OKABE_ITO["blue"],
            lw=1.55,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(p_lbl)
    ax.set_title("Orbital Period & Separation")
    ax_sep = ax.twinx()
    if (
        ref_tbl is not None
        and ref_x is not None
        and "binary_separation" in ref_tbl.columns
    ):
        _compare_panels._plot_series(
            ax_sep,
            ref_x,
            ref_tbl.columns["binary_separation"],
            label="$a$ MESA",
            color=style.OKABE_ITO["bluish_green"],
            ls="--",
            lw=1.1,
            alpha=0.75,
        )
    if "binary_separation" in run_tbl.columns:
        _compare_panels._plot_series(
            ax_sep,
            run_x,
            run_tbl.columns["binary_separation"],
            label="$a$",
            color=style.OKABE_ITO["bluish_green"],
            lw=1.45,
        )
    ax_sep.set_ylabel(r"Separation [$R_\odot$]")
    _compare_panels.merge_twinx_legend(
        ax, ax_sep, loc="best", fontsize=6.6, ncol=2
    )

    # (2, 1) Residual panel when comparing a manifest case, else Masses + J_orb
    comp: history_compare.HistoryComparison | None = None
    if spec is not None and ref_tbl is not None:
        if (
            spec.independent_var in ref_tbl.columns
            and spec.independent_var in run_tbl.columns
        ):
            try:
                comp = history_compare.compare_histories(ref_tbl, run_tbl, spec)
            except (ValueError, KeyError):
                comp = None
        _compare_panels._plot_residual_panel(
            axes[2, 1], ref_tbl, run_tbl, spec, comp
        )
    else:
        ax_m = axes[2, 1]
        for col, tag, c_hex in (
            ("star_1_mass", "$M_1$", style.OKABE_ITO["blue"]),
            ("star_2_mass", "$M_2$", style.OKABE_ITO["bluish_green"]),
        ):
            if (
                ref_tbl is not None
                and ref_x is not None
                and col in ref_tbl.columns
            ):
                _compare_panels._plot_series(
                    ax_m,
                    ref_x,
                    ref_tbl.columns[col],
                    label=f"{tag} MESA",
                    color=c_hex,
                    ls="--",
                    lw=1.1,
                    alpha=0.75,
                )
            if col in run_tbl.columns:
                _compare_panels._plot_series(
                    ax_m,
                    run_x,
                    run_tbl.columns[col],
                    label=tag,
                    color=c_hex,
                    lw=1.55,
                )
        ax_m.set_xlabel(x_label)
        ax_m.set_ylabel(r"Component Mass [$M_\odot$]")
        ax_m.set_title("Component Masses")
        ax_m.legend(loc="best", frameon=False, fontsize=7.0, ncol=2)

    if spec is not None and ref_tbl is not None:
        suptitle = _compare_panels.format_provenance_title(
            spec,
            ref_tbl,
            run_tbl,
            comp,
            termination_code=termination_code,
            numerics_mode=numerics_mode,
            controls_ignored=controls_ignored,
            n_controls_ignored=n_controls_ignored,
            max_steps=max_steps,
        )
    elif ref_tbl is not None:
        ref_ver = ref_tbl.mesa_version_number or str(
            ref_hdr.get("version_number", "MESA ref")
        ).strip('"')
        ref_sdk = str(ref_hdr.get("MESA_SDK_version", "")).strip('"')
        sdk_note = f", SDK {ref_sdk}" if ref_sdk else ""
        suptitle = (
            f"pgbinary Comparison: {run_path.name}\n"
            f"Ref: MESA {ref_ver}{sdk_note} ({ref_tbl.num_rows} rows) | "
            f"Candidate: {run_tbl.num_rows} rows"
        )
    else:
        ver = str(run_hdr.get("version_number", "FlexMESA")).strip('"')
        suptitle = (
            f"pgbinary Summary: {run_path.name} "
            f"({ver}, {run_tbl.num_rows} rows)"
        )
    fig.suptitle(suptitle, fontsize=9.0)
    _compare_panels.save_figure(fig, save_path)
    return fig, axes
