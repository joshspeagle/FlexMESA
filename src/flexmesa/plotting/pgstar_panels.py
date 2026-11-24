# SPDX-License-Identifier: GPL-3.0-or-later
r"""Upstream ``pgstar`` ``TRho_Profile``, ``Power``, ``Abundance``, ``Mixing``.

Ports ``TRho_Profile`` (`star/private/pgstar_trho_profile.f90:56-445`),
``Power`` (`star/private/pgstar_power.f90:73-278`), ``Abundance``
(`star/private/pgstar_abundance.f90:74-403`), and ``Mixing``
(`star/private/pgstar_mixing_ds.f90:106-480`) strictly from saved
``profileN.data`` outputs and bundled MESA ``26.4.1`` ``plot_info/*.data``.
"""

from __future__ import annotations

import math
import pathlib
import re
from collections.abc import Sequence
from typing import Literal

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import numpy as np

from flexmesa.plotting import _compare_panels, _plot_support, compare, style

REACTION_CATEGORIES: tuple[str, ...] = (
    "pp",
    "cno",
    "tri_alpha",
    "burn_c",
    "burn_n",
    "burn_o",
    "burn_ne",
    "burn_na",
    "burn_mg",
    "burn_si",
    "burn_s",
    "burn_ar",
    "burn_ca",
    "burn_ti",
    "burn_cr",
    "burn_fe",
    "c12_c12",
    "c12_o16",
    "o16_o16",
    "pnhe4",
    "photo",
    "other",
)

_BURN_GUIDE_FILES: tuple[tuple[str, str], ...] = (
    ("hydrogen_burn.data", "H burn"),
    ("helium_burn.data", "He burn"),
    ("carbon_burn.data", "C burn"),
    ("oxygen_burn.data", "O burn"),
)

_ISO_NAME_RE = re.compile(r"^(?:neut|prot|[a-z]{1,2}\d{1,3})$")

SERIES_COLORS: tuple[str, ...] = (
    style.OKABE_ITO["blue"],
    style.OKABE_ITO["vermilion"],
    style.OKABE_ITO["bluish_green"],
    style.OKABE_ITO["orange"],
    style.OKABE_ITO["reddish_purple"],
    style.OKABE_ITO["sky_blue"],
    style.OKABE_ITO["black"],
    style.OKABE_ITO["grey"],
)


def safe_log10_pos(arr: np.ndarray) -> np.ndarray:
    """Compute ``log10(x)`` on positive finite values and ``NaN`` elsewhere."""
    valid = _compare_panels.finite_mask(arr) & (arr > 0.0)
    safe = np.where(valid, arr, 1.0)
    return np.where(valid, np.log10(safe), np.nan)


def plot_trho_profile(
    profile_path: pathlib.Path | str,
    *,
    ref_profile_path: pathlib.Path | str | None = None,
    xaxis: Literal["logRho", "column_depth", "logxm"] = "logRho",
    show_degeneracy_line: bool = True,
    show_pgas_prad_line: bool = True,
    show_burn_lines: bool = True,
    show_gamma1_4_3rd: bool = False,
    show_kap_lines: bool = False,
    show_mix_types: bool = True,
    full_trho_bounds: bool = False,
    mesa_dir: pathlib.Path | str | None = None,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot radial ``TRho_Profile`` with MESA ``26.4.1`` regime guide lines.

    Implements ``do_TRho_Profile_plot`` from
    ``star/private/pgstar_trho_profile.f90:56-445`` paired with bundled
    ``data/star_data/plot_info/*.data`` curves read via
    ``star/private/pgstar_support.f90:276-395``.

    Args:
        profile_path: Path to ``profileN.data``.
        ref_profile_path: Optional MESA ``26.4.1`` ``profileN.data`` overlay.
        xaxis: Horizontal coordinate (``"logRho"`` default,
            ``"column_depth"``, or ``"logxm"``,
            `pgstar_trho_profile.f90:82-95`).
        show_degeneracy_line: Draw $\epsilon_{\mathrm{F}}/k_{\mathrm{B}}T=4$
            from ``psi4.data`` (`pgstar_trho_profile.f90:174, 229-240`).
        show_pgas_prad_line: Draw $P_{\mathrm{rad}} \approx P_{\mathrm{gas}}$
            (`pgstar_trho_profile.f90:72, 175, 417-431`).
        show_burn_lines: Draw H/He/C/O ignition curves from
            ``*_burn.data`` (`pgstar_trho_profile.f90:176, 434-445`).
        show_gamma1_4_3rd: Draw $\Gamma_1 < 4/3$ locus from
            ``gamma_4_thirds.data`` (`pgstar_trho_profile.f90:173, 397-407`).
        show_kap_lines: Draw $\kappa_{\mathrm{rad}} = \kappa_{\mathrm{cond}}$
            and $e^-e^+$ curves from ``kap_rad_cond_eq.data`` and
            ``elect.data`` (`pgstar_trho_profile.f90:313-321`).
        show_mix_types: Highlight convective and non-convective mixing zones
            along the profile locus.
        full_trho_bounds: Use wide ``pgstar.defaults:326-329`` plane limits
            ``[-11.1, 10.2] x [2.6, 10.2]`` instead of framing the star.
        mesa_dir: Optional MESA ``26.4.1`` root directory.
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.
    """
    prof = compare.load_profile_file(profile_path)
    if "logT" not in prof.columns:
        raise KeyError("TRho_Profile requires 'logT' column in profileN.data")
    ref_prof = (
        compare.load_profile_file(ref_profile_path)
        if ref_profile_path is not None
        else None
    )

    if xaxis == "logRho":
        if "logRho" not in prof.columns:
            raise KeyError("xaxis='logRho' requires 'logRho' column")
        x_run = prof.columns["logRho"]
        x_label = r"$\log_{10}(\rho / \mathrm{g\,cm^{-3}})$"
        x_ref = ref_prof.columns.get("logRho") if ref_prof is not None else None
    else:
        x_run, x_label, _ = _plot_support.profile_xaxis(prof, xaxis)
        x_ref = (
            _plot_support.profile_xaxis(ref_prof, xaxis)[0]
            if ref_prof is not None
            else None
        )

    y_run = prof.columns["logT"]
    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.86)
    valid_xy = _compare_panels.finite_mask(x_run) & _compare_panels.finite_mask(
        y_run
    )
    if full_trho_bounds or not np.any(valid_xy):
        x_lo, x_hi, y_lo, y_hi = -11.1, 10.2, 2.6, 10.2
    else:
        x_lo = float(np.min(x_run[valid_xy])) - 1.1
        x_hi = float(np.max(x_run[valid_xy])) + 1.1
        y_lo = float(np.min(y_run[valid_xy])) - 0.55
        y_hi = float(np.max(y_run[valid_xy])) + 0.55
        if (
            ref_prof is not None
            and x_ref is not None
            and "logT" in ref_prof.columns
        ):
            vr = _compare_panels.finite_mask(
                x_ref
            ) & _compare_panels.finite_mask(ref_prof.columns["logT"])
            if np.any(vr):
                x_lo = min(x_lo, float(np.min(x_ref[vr])) - 1.1)
                x_hi = max(x_hi, float(np.max(x_ref[vr])) + 1.1)
                y_lo = min(
                    y_lo, float(np.min(ref_prof.columns["logT"][vr])) - 0.55
                )
                y_hi = max(
                    y_hi, float(np.max(ref_prof.columns["logT"][vr])) + 0.55
                )

    if xaxis == "logRho":
        if show_degeneracy_line:
            psi_rho, psi_t = _plot_support.load_plot_info_curve(
                "psi4.data", mesa_dir=mesa_dir
            )
            ax.plot(
                psi_rho,
                psi_t,
                color=style.OKABE_ITO["grey"],
                ls="--",
                lw=0.9,
                alpha=0.75,
                label=r"$\epsilon_{\mathrm{F}}/k_{\mathrm{B}}T \approx 4$",
                zorder=1,
            )
        if show_pgas_prad_line:
            # star/private/pgstar_trho_profile.f90:72, 417-431
            lgrho_ends = np.array([-8.0, 5.0], dtype=np.float64)
            lgt_ends = math.log10(3.2e7) + (lgrho_ends - math.log10(0.7)) / 3.0
            ax.plot(
                lgrho_ends,
                lgt_ends,
                color=style.OKABE_ITO["grey"],
                ls=":",
                lw=0.95,
                alpha=0.75,
                label=r"$P_{\mathrm{rad}} \approx P_{\mathrm{gas}}$",
                zorder=1,
            )
        if show_burn_lines:
            for fname, b_lbl in _BURN_GUIDE_FILES:
                b_rho, b_t = _plot_support.load_plot_info_curve(
                    fname, mesa_dir=mesa_dir
                )
                ax.plot(
                    b_rho,
                    b_t,
                    color=style.OKABE_ITO["grey"],
                    ls="-.",
                    lw=0.8,
                    alpha=0.60,
                    zorder=1,
                )
                in_box = (
                    (b_rho >= x_lo)
                    & (b_rho <= x_hi)
                    & (b_t >= y_lo)
                    & (b_t <= y_hi)
                )
                if np.any(in_box):
                    vis_i = int(np.flatnonzero(in_box)[0])
                    ax.text(
                        float(b_rho[vis_i]),
                        float(b_t[vis_i]) + 0.05,
                        b_lbl,
                        fontsize=6.8,
                        color=style.OKABE_ITO["grey"],
                        clip_on=True,
                    )
        if show_gamma1_4_3rd:
            g_rho, g_t = _plot_support.load_plot_info_curve(
                "gamma_4_thirds.data", mesa_dir=mesa_dir
            )
            ax.plot(
                g_rho,
                g_t,
                color=style.OKABE_ITO["orange"],
                ls="-",
                lw=1.2,
                alpha=0.85,
                label=r"$\Gamma_1 < 4/3$",
                zorder=1,
            )
        if show_kap_lines:
            kr_rho, kr_t = _plot_support.load_plot_info_curve(
                "kap_rad_cond_eq.data", mesa_dir=mesa_dir
            )
            el_rho, el_t = _plot_support.load_plot_info_curve(
                "elect.data", mesa_dir=mesa_dir
            )
            ax.plot(
                kr_rho,
                kr_t,
                color=style.OKABE_ITO["sky_blue"],
                ls=":",
                lw=0.95,
                label=r"$\kappa_{\mathrm{rad}}=\kappa_{\mathrm{cond}}$",
                zorder=1,
            )
            ax.plot(
                el_rho,
                el_t,
                color=style.OKABE_ITO["reddish_purple"],
                ls="--",
                lw=0.95,
                label=r"$e^-e^+$",
                zorder=1,
            )

    if (
        ref_prof is not None
        and x_ref is not None
        and "logT" in ref_prof.columns
    ):
        ref_lbl = _plot_support.format_ref_label(
            str(ref_prof.header.get("version_number", ""))
        )
        _compare_panels._plot_series(
            ax,
            x_ref,
            ref_prof.columns["logT"],
            label=ref_lbl,
            color=style.OKABE_ITO["black"],
            ls="--",
            lw=1.2,
            zorder=2,
        )
    _compare_panels._plot_series(
        ax,
        x_run,
        y_run,
        label="Profile",
        color=style.OKABE_ITO["blue"],
        lw=1.65,
        zorder=3,
    )
    if show_mix_types and "mixing_type" in prof.columns:
        mtypes = np.rint(prof.columns["mixing_type"]).astype(np.int64)
        for mcode, (mname, mhex, _mhatch) in _plot_support.MIXING_STYLE.items():
            mask = valid_xy & (mtypes == mcode)
            if np.any(mask):
                ax.scatter(
                    x_run[mask],
                    y_run[mask],
                    c=mhex,
                    s=11,
                    label=mname,
                    zorder=4,
                )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\log_{10}(T / \mathrm{K})$")
    _plot_support.set_profile_panel_title(
        ax,
        "TRho Profile",
        prof,
        ref_prof,
        standalone=standalone,
        always_subtitle=True,
    )
    ax.legend(loc="best", frameon=False, fontsize=6.6, ncol=2)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def plot_power_profile(
    profile_path: pathlib.Path | str,
    *,
    ref_profile_path: pathlib.Path | str | None = None,
    xaxis: str = "mass",
    ymin: float | None = None,
    ymax: float | None = None,
    max_categories: int = 10,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot nuclear reaction category ``Power`` profile (`pgstar_power.f90`).

    Implements ``do_power_panel`` from ``star/private/pgstar_power.f90:73-278``
    across all nuclear categories stored in ``profileN.data``, ranked by peak
    specific energy generation rate.

    Args:
        profile_path: Path to ``profileN.data``.
        ref_profile_path: Optional MESA ``26.4.1`` ``profileN.data`` overlay.
        xaxis: Radial coordinate name (default ``"mass"``).
        ymin: Optional lower $\log_{10}(\epsilon)$ bound overriding
            `pgstar_power.f90:161-170`.
        ymax: Optional upper $\log_{10}(\epsilon)$ bound.
        max_categories: Maximum number of active categories drawn in legend.
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

    ranked: list[tuple[float, str]] = []
    for cat in REACTION_CATEGORIES:
        if cat not in prof.columns:
            continue
        arr = prof.columns[cat]
        valid = _compare_panels.finite_mask(arr)
        ranked.append(
            (float(np.max(arr[valid])) if np.any(valid) else -1.0e99, cat)
        )
    if not ranked and "eps_nuc" in prof.columns:
        arr = prof.columns["eps_nuc"]
        valid = _compare_panels.finite_mask(arr)
        ranked.append(
            (float(np.max(arr[valid])) if np.any(valid) else -1.0e99, "eps_nuc")
        )
    if not ranked:
        raise ValueError(
            "Power panel requires reaction category columns (pp, cno, ...) "
            "or 'eps_nuc' in profileN.data"
        )
    ranked.sort(key=lambda item: item[0], reverse=True)

    # star/private/pgstar_power.f90:159-170
    top_log = math.log10(max(ranked[0][0], 1.0e-29))
    auto_ymin, auto_ymax = (
        (-4.1, 4.3) if top_log <= 4.0 else (-0.1, top_log * 1.1)
    )
    eff_ymin = auto_ymin if ymin is None else float(ymin)
    eff_ymax = auto_ymax if ymax is None else float(ymax)
    floor_lin = 10.0**eff_ymin

    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.82)
    drawn = 0
    for peak_val, cat in ranked:
        if drawn >= max_categories or peak_val < floor_lin:
            continue
        c_hex = SERIES_COLORS[drawn % len(SERIES_COLORS)]
        if _plot_support.plot_profile_pair(
            ax,
            prof,
            x_run,
            ref_prof,
            x_ref,
            cat,
            label=cat,
            color=c_hex,
            lw=1.55,
            transform=safe_log10_pos,
        ):
            drawn += 1

    if drawn == 0:
        _, top_cat = ranked[0]
        _compare_panels._plot_series(
            ax,
            x_run,
            safe_log10_pos(np.maximum(prof.columns[top_cat], 1.0e-29)),
            label=top_cat,
            color=SERIES_COLORS[0],
            lw=1.55,
        )

    ax.set_ylim(eff_ymin - 0.05, eff_ymax)
    if rev_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    _plot_support.draw_xaxis_mixing_bar(ax, x_run, prof)
    ax.set_xlabel(x_label)
    ax.set_ylabel(
        r"Specific Power [$\log_{10}(\epsilon / \mathrm{erg\,g^{-1}\,s^{-1}})$]"
    )
    _plot_support.set_profile_panel_title(
        ax, "Nuclear Reaction Power", prof, ref_prof, standalone=standalone
    )
    ax.legend(loc="best", frameon=False, fontsize=6.8, ncol=2)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


def _discover_isotope_columns(prof: compare.ProfileTable) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col, arr in prof.columns.items():
        if not _ISO_NAME_RE.match(col):
            continue
        valid = _compare_panels.finite_mask(arr)
        if not np.any(valid):
            continue
        vmin, vmax = float(np.min(arr[valid])), float(np.max(arr[valid]))
        if vmin < -1.0e-6 or vmax > 1.05:
            continue
        scored.append((vmax, col))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [name for _, name in scored]
    fallback: list[str] = [
        c
        for c in (
            "x_mass_fraction_H",
            "y_mass_fraction_He",
            "z_mass_fraction_metals",
        )
        if c in prof.columns
    ]
    return fallback


def plot_abundance_profile(
    profile_path: pathlib.Path | str,
    *,
    ref_profile_path: pathlib.Path | str | None = None,
    xaxis: str = "mass",
    isos: Sequence[str] | None = None,
    num_isos_to_show: int = 10,
    log_mass_frac_min: float = -5.1,
    log_mass_frac_max: float = 0.25,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot radial logarithmic ``Abundance`` profiles (`pgstar_abundance.f90`).

    Implements ``do_abundance_panel`` from
    ``star/private/pgstar_abundance.f90:74-403``.

    Args:
        profile_path: Path to ``profileN.data``.
        ref_profile_path: Optional MESA ``26.4.1`` ``profileN.data`` overlay.
        xaxis: Radial coordinate name (default ``"mass"``).
        isos: Optional explicit list of isotope column names to display.
        num_isos_to_show: Maximum number of ranked species plotted when
            ``isos`` is ``None`` (`pgstar_abundance.f90:277-296`).
        log_mass_frac_min: Lower $\log_{10} X_i$ (`pgstar_abundance.f90:155`).
        log_mass_frac_max: Upper $\log_{10} X_i$ (`pgstar_abundance.f90:163`).
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

    candidates = (
        list(isos) if isos is not None else _discover_isotope_columns(prof)
    )
    if not candidates:
        raise ValueError(
            "no isotope or H/He/Z mass-fraction columns in profile"
        )

    standalone = ax is None
    fig, ax = style.init_ax(ax, "single", 0.82)
    short_names = {
        "x_mass_fraction_H": "H",
        "y_mass_fraction_He": "He",
        "z_mass_fraction_metals": "Z",
    }
    cnt = 0
    for iso in candidates:
        if isos is None and cnt >= num_isos_to_show:
            break
        if iso not in prof.columns:
            continue
        log_xa = safe_log10_pos(prof.columns[iso])
        if not np.any(np.isfinite(log_xa)) or (
            float(np.nanmax(log_xa)) < log_mass_frac_min
        ):
            continue
        c_hex = SERIES_COLORS[cnt % len(SERIES_COLORS)]
        ls_run = ":" if cnt >= len(SERIES_COLORS) else "-"
        _plot_support.plot_profile_pair(
            ax,
            prof,
            x_run,
            ref_prof,
            x_ref,
            iso,
            label=short_names.get(iso, iso),
            color=c_hex,
            ls=ls_run,
            lw=1.5,
            transform=safe_log10_pos,
        )
        cnt += 1

    ax.set_ylim(log_mass_frac_min - 0.05, log_mass_frac_max)
    if rev_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    _plot_support.draw_xaxis_mixing_bar(ax, x_run, prof)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Log Mass Fraction [$\log_{10} X_i$]")
    _plot_support.set_profile_panel_title(
        ax, "Abundance Profile", prof, ref_prof, standalone=standalone
    )
    ax.legend(loc="best", frameon=False, fontsize=6.6, ncol=3)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax


_NON_ROT_MIXING_SPECS: tuple[tuple[str, str, str], ...] = (
    ("log_D_conv", r"$D_{\mathrm{conv}}$", style.OKABE_ITO["sky_blue"]),
    ("log_D_leftover", r"$D_{\mathrm{left}}$", style.OKABE_ITO["blue"]),
    ("log_D_ovr", r"$D_{\mathrm{ovr}}$", style.OKABE_ITO["orange"]),
    ("log_D_semi", r"$D_{\mathrm{semi}}$", style.OKABE_ITO["bluish_green"]),
    ("log_D_thrm", r"$D_{\mathrm{thrm}}$", style.OKABE_ITO["reddish_purple"]),
    (
        "log_D_rayleigh_taylor",
        r"$D_{\mathrm{RTI}}$",
        style.OKABE_ITO["vermilion"],
    ),
    ("log_D_minimum", r"$D_{\min}$", style.OKABE_ITO["grey"]),
    ("log_D_anon", r"$D_{\mathrm{anon}}$", style.OKABE_ITO["black"]),
)

# Rotational diffusion curves from star/private/pgstar_mixing_ds.f90:372-439
# paired with profile column names in star/private/profile_getval.f90:1585-1634.
_ROT_MIXING_SPECS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (
        ("am_log_D_ST", "log_D_ST", "D_ST"),
        r"$D_{\mathrm{ST}}$",
        style.OKABE_ITO["vermilion"],
        "-.",
    ),
    (
        ("am_log_D_visc", "log_D_visc", "D_visc"),
        r"$D_{\mathrm{visc}}$",
        style.OKABE_ITO["grey"],
        "--",
    ),
    (
        ("am_log_D_DSI", "log_D_DSI", "D_DSI"),
        r"$D_{\mathrm{DSI}}$",
        style.OKABE_ITO["orange"],
        "-.",
    ),
    (
        ("am_log_D_SH", "log_D_SH", "D_SH"),
        r"$D_{\mathrm{SH}}$",
        style.OKABE_ITO["reddish_purple"],
        "--",
    ),
    (
        ("am_log_D_SSI", "log_D_SSI", "D_SSI"),
        r"$D_{\mathrm{SSI}}$",
        style.OKABE_ITO["yellow"],
        "-.",
    ),
    (
        ("am_log_D_ES", "log_D_ES", "D_ES"),
        r"$D_{\mathrm{ES}}$",
        style.OKABE_ITO["blue"],
        "--",
    ),
    (
        ("am_log_D_GSF", "log_D_GSF", "D_GSF"),
        r"$D_{\mathrm{GSF}}$",
        style.OKABE_ITO["bluish_green"],
        "-.",
    ),
)

_ROT_TOTAL_CANDIDATES: tuple[str, ...] = (
    "log_D_mix_rotation",
    "D_mix_rotation",
    "log_D_omega",
    "D_omega",
)


def _mask_log_d(arr: np.ndarray) -> np.ndarray:
    """Mask non-finite or negligible ``log10(D) <= -20`` diffusivities."""
    valid = _compare_panels.finite_mask(arr) & (arr > -20.0)
    return np.where(valid, arr, np.nan)


def _linear_to_masked_log_d(arr: np.ndarray) -> np.ndarray:
    """Convert linear diffusivity to masked ``log10(D)`` with floor ``-20``."""
    return _mask_log_d(safe_log10_pos(arr))


def _find_diffusivity_col(
    prof: compare.ProfileTable, candidates: tuple[str, ...]
) -> tuple[str, bool] | None:
    """Return ``(col_name, is_linear)`` for the first present column."""
    for col in candidates:
        if col in prof.columns:
            return col, col.startswith("D_")
    return None


def plot_mixing_profile(
    profile_path: pathlib.Path | str,
    *,
    ref_profile_path: pathlib.Path | str | None = None,
    xaxis: str = "mass",
    show_rotation_details: bool = True,
    ymin: float | None = None,
    ymax: float | None = None,
    ax: mpl_axes.Axes | None = None,
    save_path: pathlib.Path | str | None = None,
) -> tuple[mpl_figure.Figure, mpl_axes.Axes]:
    r"""Plot radial ``Mixing`` diffusivity (`pgstar_mixing_ds.f90`).

    Implements ``MixDs_plot`` (`star/private/pgstar_mixing_ds.f90:106-480`),
    including non-rotational mixing regimes and rotational diffusion curves
    (``D_ST``, ``D_visc``, ``D_DSI``, ``D_SH``, ``D_SSI``, ``D_ES``,
    ``D_GSF``, and ``D_mix_rotation`` / ``log_D_omega`` per
    `star/private/pgstar_mixing_ds.f90:342-441`) whenever those profile
    columns are present.

    Args:
        profile_path: Path to ``profileN.data``.
        ref_profile_path: Optional MESA ``26.4.1`` ``profileN.data`` overlay.
        xaxis: Radial coordinate name (default ``"mass"``).
        show_rotation_details: Plot individual rotational diffusivity curves
            (``D_ST``, ``D_visc``, ``D_DSI``, ``D_SH``, ``D_SSI``, ``D_ES``,
            ``D_GSF``) when present; if ``False`` and no aggregate rotation
            column exists, plot their sum matching
            `star/private/pgstar_mixing_ds.f90:354-368`.
        ymin: Optional lower $\log_{10}(D / \mathrm{cm^2\,s^{-1}})$ bound.
        ymax: Optional upper $\log_{10}(D / \mathrm{cm^2\,s^{-1}})$ bound.
        ax: Optional target ``matplotlib.axes.Axes``.
        save_path: Optional output image path (`300 dpi`).

    Returns:
        Tuple ``(fig, ax)``.
    """
    prof = compare.load_profile_file(profile_path)
    base_col = (
        "log_D_mix"
        if "log_D_mix" in prof.columns
        else (
            "log_D_mix_non_rotation"
            if "log_D_mix_non_rotation" in prof.columns
            else None
        )
    )
    if base_col is None:
        raise KeyError("Mixing panel requires 'log_D_mix' in profileN.data")
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
    fig, ax = style.init_ax(ax, "single", 0.80)
    y_d = prof.columns[base_col]
    _plot_support.plot_profile_pair(
        ax,
        prof,
        x_run,
        ref_prof,
        x_ref,
        base_col,
        label=r"$\log D_{\mathrm{mix}}$",
        color=style.OKABE_ITO["grey"],
        lw=1.2,
    )
    if "mixing_type" in prof.columns:
        mtypes = np.rint(prof.columns["mixing_type"]).astype(np.int64)
        valid_d = (
            _compare_panels.finite_mask(x_run)
            & _compare_panels.finite_mask(y_d)
            & (y_d > -20.0)
        )
        for mcode, (mname, mhex, _mhatch) in _plot_support.MIXING_STYLE.items():
            mask = valid_d & (mtypes == mcode)
            if np.any(mask):
                ax.scatter(
                    x_run[mask],
                    y_d[mask],
                    c=mhex,
                    s=13,
                    label=mname,
                    zorder=4,
                )

    for comp_col, comp_lbl, comp_hex in _NON_ROT_MIXING_SPECS:
        if comp_col in prof.columns:
            _plot_support.plot_profile_pair(
                ax,
                prof,
                x_run,
                ref_prof,
                x_ref,
                comp_col,
                label=comp_lbl,
                color=comp_hex,
                ls=":",
                lw=1.35,
                transform=_mask_log_d,
            )

    # Rotational diffusion curves (star/private/pgstar_mixing_ds.f90:342-441)
    rot_total = _find_diffusivity_col(prof, _ROT_TOTAL_CANDIDATES)
    drew_rot_total = False
    if rot_total is not None:
        r_col, r_lin = rot_total
        drew_rot_total = _plot_support.plot_profile_pair(
            ax,
            prof,
            x_run,
            ref_prof,
            x_ref,
            r_col,
            label=r"$D_{\mathrm{rot}}$",
            color=style.OKABE_ITO["yellow"],
            ls="-",
            lw=1.55,
            transform=_linear_to_masked_log_d if r_lin else _mask_log_d,
        )

    if show_rotation_details:
        for cands, rot_lbl, rot_hex, rot_ls in _ROT_MIXING_SPECS:
            found = _find_diffusivity_col(prof, cands)
            if found is None:
                continue
            r_col, r_lin = found
            _plot_support.plot_profile_pair(
                ax,
                prof,
                x_run,
                ref_prof,
                x_ref,
                r_col,
                label=rot_lbl,
                color=rot_hex,
                ls=rot_ls,
                lw=1.35,
                transform=_linear_to_masked_log_d if r_lin else _mask_log_d,
            )
    elif not drew_rot_total:
        # Aggregate rotational mixing D_DSI + D_SH + D_SSI + D_ES + D_GSF + D_ST
        # matching star/private/pgstar_mixing_ds.f90:356-364 (excluding D_visc).
        rot_sum = np.zeros(prof.num_zones, dtype=np.float64)
        found_any = False
        for cands, rot_lbl, _rot_hex, _rot_ls in _ROT_MIXING_SPECS:
            if rot_lbl == r"$D_{\mathrm{visc}}$":
                continue
            found = _find_diffusivity_col(prof, cands)
            if found is None:
                continue
            r_col, r_lin = found
            raw_v = prof.columns[r_col]
            if r_lin:
                lin_v = np.where(
                    _compare_panels.finite_mask(raw_v) & (raw_v > 0.0),
                    raw_v,
                    0.0,
                )
            else:
                v_ok = _compare_panels.finite_mask(raw_v) & (raw_v > -20.0)
                lin_v = np.where(
                    v_ok, np.power(10.0, np.where(v_ok, raw_v, -30.0)), 0.0
                )
            rot_sum = rot_sum + lin_v
            found_any = True
        if found_any:
            _compare_panels._plot_series(
                ax,
                x_run,
                _linear_to_masked_log_d(rot_sum),
                label=r"$D_{\mathrm{rot}}$",
                color=style.OKABE_ITO["yellow"],
                ls="-",
                lw=1.55,
            )

    if ymin is not None or ymax is not None:
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(
            cur_lo if ymin is None else float(ymin),
            cur_hi if ymax is None else float(ymax),
        )
    if rev_x and not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\log_{10}(D_{\mathrm{mix}} / \mathrm{cm^2\,s^{-1}})$")
    _plot_support.set_profile_panel_title(
        ax, "Mixing Diffusivity", prof, ref_prof, standalone=standalone
    )
    ax.legend(loc="best", frameon=False, fontsize=6.4, ncol=2)
    _compare_panels.save_figure(fig, save_path)
    return fig, ax
