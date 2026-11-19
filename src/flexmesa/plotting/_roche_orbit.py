# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Internal 2D Keplerian orbit and Roche equipotential solver (`#173`).

Mirrors ``orbit_panel``, ``xl1_fit``, ``Psi_fit``, and ``roche`` from
``binary/private/pgbinary_orbit.f90:63-296``.
"""

from __future__ import annotations

import math

import matplotlib.axes as mpl_axes
import matplotlib.colors as mpl_colors
import numpy as np

from flexmesa.plotting import style
from flexmesa.testing import history_compare


def xl1_fit(q_ratio: float) -> float:
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


def psi_fit(req: float, q_ratio: float) -> float:
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


def roche_potential(radius: float, cos_phi: float, q_ratio: float) -> float:
    """Roche potential (`binary/private/pgbinary_orbit.f90:273-279`)."""
    dist2 = max(1.0 - 2.0 * radius * cos_phi + radius * radius, 1.0e-16)
    return (
        -1.0 / radius
        - q_ratio * (1.0 / math.sqrt(dist2) - radius * cos_phi)
        - 0.5 * (1.0 + q_ratio) * radius * radius
    )


def solve_roche_ray(
    cos_phi: float,
    q_ratio: float,
    target_psi: float,
    r_max: float,
) -> float:
    """Solve ``roche(r, cos_phi) == target_psi`` on ``(1e-4, r_max]``."""
    r_lo = 1.0e-4
    r_hi = max(min(r_max, 1.0 - 1.0e-3), r_lo * 2.0)
    f_hi = roche_potential(r_hi, cos_phi, q_ratio) - target_psi
    if f_hi <= 0.0:
        probe = np.linspace(r_lo, r_hi, 32, dtype=np.float64)
        vals = np.array(
            [
                roche_potential(float(rv), cos_phi, q_ratio) - target_psi
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
        if roche_potential(r_mid, cos_phi, q_ratio) - target_psi < 0.0:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return 0.5 * (r_lo + r_hi)


def equipotential_xy(
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
    xl1 = xl1_fit(q_ratio)
    this_psi = psi_fit(req_clamped, q_ratio)
    n_tot = 2 * num_points + 1
    xs = np.zeros(n_tot, dtype=np.float64)
    ys = np.zeros(n_tot, dtype=np.float64)

    for idx in range(num_points):
        phi = (idx + 0.5) * math.pi / num_points
        cos_p = math.cos(phi)
        r_ray = solve_roche_ray(cos_p, q_ratio, this_psi, xl1)
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


def select_row_index(
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


#: ``binary_history`` columns the Orbit panel cannot be drawn without.
ORBIT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "eccentricity",
    "binary_separation",
    "rl_1",
    "rl_2",
)


class MissingOrbitColumnsError(ValueError):
    """Raised when ``binary_history`` lacks a column the Orbit panel needs."""


def extract_orbit_state(
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
    # Upstream's Orbit window reads ``b% eccentricity``, ``b% separation``
    # and ``b% rl`` from memory (``binary/private/pgbinary_orbit.f90:102``).
    # A saved history either carries those columns or the panel cannot be
    # drawn; no value is substituted for a missing or non-finite one.
    absent = [c for c in ORBIT_REQUIRED_COLUMNS if c not in tbl.columns]
    if absent:
        raise MissingOrbitColumnsError(
            "Orbit panel requires binary_history columns "
            + ", ".join(repr(c) for c in absent)
            + "; enable them in binary_history_columns.list"
        )
    ecc = float(tbl.columns["eccentricity"][idx])
    sep = float(tbl.columns["binary_separation"][idx])
    rl1 = float(tbl.columns["rl_1"][idx])
    rl2 = float(tbl.columns["rl_2"][idx])
    if not (math.isfinite(ecc) and 0.0 <= ecc < 1.0):
        raise ValueError(f"Orbit panel: eccentricity {ecc!r} outside [0, 1)")
    if not (math.isfinite(sep) and sep > 0.0):
        raise ValueError(f"Orbit panel: binary_separation {sep!r} not > 0")
    if not (math.isfinite(rl1) and math.isfinite(rl2)):
        raise ValueError("Orbit panel: non-finite rl_1 or rl_2")

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


def draw_orbit_geometry(
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
                x_rl, y_rl = equipotential_xy(
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
                x_st, y_st = equipotential_xy(
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
